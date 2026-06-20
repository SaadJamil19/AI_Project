from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

try:
    import faiss  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise RuntimeError(
        "faiss-cpu is required for the Python sidecar. Install it with: "
        "python -m pip install faiss-cpu"
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise RuntimeError(
        "numpy is required for the Python sidecar. Install it with: "
        "python -m pip install numpy"
    ) from exc


LOGGER = logging.getLogger(__name__)
EMBEDDING_DIMENSIONS = 384


class DocumentRecord(BaseModel):
    """Searchable row loaded from SQLite source-of-truth storage."""

    model_config = ConfigDict(frozen=True)

    doc_rowid: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    source_type: str = Field(pattern=r"^(STATIC_DOCS|USER_TEMPLATE)$")
    binary_name: str = Field(min_length=1)
    subcommand_path: str = ""
    intent_description: str = Field(min_length=1)
    trust_state: str = Field(min_length=1)
    project_root_hash: str | None = None

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part
            for part in (self.binary_name, self.subcommand_path, self.intent_description)
            if part
        ).strip()


class VectorSearchHit(BaseModel):
    """A FAISS result mapped by explicit SQLite doc_rowid."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    doc_rowid: int
    source_type: str
    binary_name: str
    subcommand_path: str
    trust_state: str
    score: float
    distance_rank: int


class VectorSearchResult(BaseModel):
    """Search result plus internal matrix-scan telemetry."""

    model_config = ConfigDict(frozen=True)

    hits: list[VectorSearchHit]
    faiss_matrix_scan_duration_ms: float


class VectorCacheSnapshot(BaseModel):
    """Cheap status payload for daemon health checks and telemetry."""

    model_config = ConfigDict(frozen=True)

    db_path: str
    document_count: int
    index_size: int
    embedding_dimensions: int
    index_type: str


def open_read_only_database(db_path: Path) -> sqlite3.Connection:
    """Open SQLite source of truth in explicit read-only URI mode."""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {resolved}")

    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn
    except sqlite3.Error:
        LOGGER.exception("failed to open read-only SQLite database at %s", resolved)
        raise


def fetch_searchable_documents(conn: sqlite3.Connection) -> list[DocumentRecord]:
    """Load searchable documents while excluding disabled or revoked templates."""

    try:
        rows = conn.execute(
            """
            SELECT
                doc_rowid,
                doc_id,
                source_type,
                binary_name,
                subcommand_path,
                intent_description,
                trust_state,
                project_root_hash
            FROM unified_documents
            WHERE trust_state NOT IN ('DISABLED', 'REVOKED')
            ORDER BY doc_rowid ASC
            """
        ).fetchall()
    except sqlite3.Error:
        LOGGER.exception("failed to extract searchable rows from unified_documents")
        raise

    return [
        DocumentRecord(
            doc_rowid=int(row["doc_rowid"]),
            doc_id=str(row["doc_id"]),
            source_type=str(row["source_type"]),
            binary_name=str(row["binary_name"]),
            subcommand_path=str(row["subcommand_path"] or ""),
            intent_description=str(row["intent_description"]),
            trust_state=str(row["trust_state"]),
            project_root_hash=(
                str(row["project_root_hash"]) if row["project_root_hash"] is not None else None
            ),
        )
        for row in rows
    ]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return float32 vectors normalized by faiss.normalize_L2."""

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D vector matrix, got shape {matrix.shape}")
    if matrix.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"expected {EMBEDDING_DIMENSIONS} dimensions, got {matrix.shape[1]}"
        )
    if matrix.size == 0:
        return matrix.copy()

    normalized = np.ascontiguousarray(matrix.copy(), dtype="float32")
    faiss.normalize_L2(normalized)
    return normalized


def rowids_array(documents: Iterable[DocumentRecord]) -> np.ndarray:
    ids = np.asarray([doc.doc_rowid for doc in documents], dtype="int64")
    if ids.ndim != 1:
        raise ValueError("expected 1D FAISS id array")
    if ids.size and int(ids.min()) < 1:
        raise ValueError("SQLite doc_rowid ids must be positive integers")
    return ids


class VectorCache:
    """Thread-safe FAISS ID-mapped cache rebuilt from SQLite source of truth."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._index = self._new_index()
        self._documents_by_rowid: dict[int, DocumentRecord] = {}

    @staticmethod
    def _new_index() -> Any:
        return faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIMENSIONS))

    @property
    def index(self) -> Any:
        with self._lock:
            return self._index

    def snapshot(self) -> VectorCacheSnapshot:
        with self._lock:
            return VectorCacheSnapshot(
                db_path=str(self.db_path),
                document_count=len(self._documents_by_rowid),
                index_size=int(self._index.ntotal),
                embedding_dimensions=EMBEDDING_DIMENSIONS,
                index_type="IndexIDMap(IndexFlatIP)",
            )

    def documents(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._documents_by_rowid.values())

    def rebuild(self, embedded_vectors: np.ndarray, documents: Iterable[DocumentRecord]) -> None:
        docs = list(documents)
        if len(docs) != int(embedded_vectors.shape[0]):
            raise ValueError(
                f"document/vector count mismatch: {len(docs)} docs, "
                f"{embedded_vectors.shape[0]} vectors"
            )

        index = self._new_index()
        ids = rowids_array(docs)
        normalized = normalize_vectors(embedded_vectors)
        if docs:
            index.add_with_ids(normalized, ids)

        with self._lock:
            self._index = index
            self._documents_by_rowid = {doc.doc_rowid: doc for doc in docs}

        LOGGER.info("rebuilt FAISS IDMap cache with %d documents", len(docs))

    def search(self, normalized_query_vector: np.ndarray, limit: int) -> VectorSearchResult:
        """Search using an already L2-normalized query vector."""

        if limit <= 0:
            return VectorSearchResult(hits=[], faiss_matrix_scan_duration_ms=0.0)
        if normalized_query_vector.ndim != 2 or normalized_query_vector.shape[0] != 1:
            raise ValueError(
                "expected a single normalized query vector shaped "
                f"(1, {EMBEDDING_DIMENSIONS}), got {normalized_query_vector.shape}"
            )

        with self._lock:
            if self._index.ntotal == 0:
                return VectorSearchResult(hits=[], faiss_matrix_scan_duration_ms=0.0)
            search_limit = min(limit, int(self._index.ntotal))
            started = time.perf_counter()
            scores, rowids = self._index.search(normalized_query_vector, search_limit)
            scan_duration_ms = (time.perf_counter() - started) * 1000.0
            documents_by_rowid = dict(self._documents_by_rowid)

        hits: list[VectorSearchHit] = []
        for rank, (rowid, score) in enumerate(zip(rowids[0], scores[0]), start=1):
            if rowid < 0:
                continue
            doc = documents_by_rowid.get(int(rowid))
            if doc is None:
                LOGGER.error("FAISS returned unmapped SQLite doc_rowid=%s", rowid)
                continue
            hits.append(
                VectorSearchHit(
                    doc_id=doc.doc_id,
                    doc_rowid=doc.doc_rowid,
                    source_type=doc.source_type,
                    binary_name=doc.binary_name,
                    subcommand_path=doc.subcommand_path,
                    trust_state=doc.trust_state,
                    score=float(score),
                    distance_rank=rank,
                )
            )

        return VectorSearchResult(
            hits=hits,
            faiss_matrix_scan_duration_ms=scan_duration_ms,
        )
