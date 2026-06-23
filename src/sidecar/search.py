from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise RuntimeError(
        "numpy is required for the Python sidecar. Install it with: "
        "python -m pip install numpy"
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise RuntimeError(
        "sentence-transformers is required for the Python sidecar. Install it with: "
        "python -m pip install sentence-transformers"
    ) from exc

from .cache import (
    EMBEDDING_DIMENSIONS,
    VectorCache,
    fetch_searchable_documents,
    normalize_vectors,
    open_read_only_database,
)
from .gateway import FastPathGateway


LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
PROTOCOL_VERSION = "1.0.0"
RRF_K = 60
TOP_POOL_LIMIT = 5
FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+")


class SidecarRequest(BaseModel):
    """Request accepted over the UDS boundary from the trusted Rust client."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8192)
    limit: int = Field(default=5, ge=1, le=25)

    @field_validator("protocol_version")
    @classmethod
    def protocol_must_match(cls, value: str) -> str:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version {value!r}")
        return value


class RetrievalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fts5_lexical_score: float | None = None
    vector_cosine_distance: float | None = None
    vector_rank: int | None = None
    embedding_duration_ms: float


class RiskHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_path_arguments: bool = False


class SidecarTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faiss_matrix_scan_duration_ms: float = Field(ge=0.0)
    python_scheduling_delay_ms: float = Field(ge=0.0)
    lexical_lookup_duration_ms: float = Field(default=0.0, ge=0.0)
    hybrid_rrf_duration_ms: float = Field(default=0.0, ge=0.0)
    fast_path_evaluation_duration_ms: float = Field(default=0.0, ge=0.0)


class IntentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_template_id: str | None = None
    typed_intent: str
    retrieval_evidence: RetrievalEvidence
    risk_hints: RiskHints
    raw_untrusted_slots: dict[str, str] = Field(default_factory=dict)


class UntrustedProposal(BaseModel):
    """Strict packet returned to Rust. It is parseable, not trusted."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = PROTOCOL_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    source_provenance: str
    intent_proposal: IntentProposal
    telemetry: SidecarTelemetry


class CandidateRecord(BaseModel):
    """Fully hydrated retrieval candidate joined to its policy row."""

    model_config = ConfigDict(frozen=True)

    doc_rowid: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    binary_name: str = Field(min_length=1)
    subcommand_path: str = ""
    intent_description: str = Field(min_length=1)
    template_argv_json: str
    slot_schema_json: str
    policy_rule_id: str = Field(min_length=1)
    trust_state: str = Field(min_length=1)
    fast_path_allowed: bool
    required_confirmation_count: int = Field(ge=1)
    package_manager_risk_level: str = Field(min_length=1)
    privilege_risk_level: str = Field(min_length=1)
    destructive_recursive_level: str = Field(min_length=1)

    @property
    def typed_intent(self) -> str:
        parts = [self.binary_name, self.subcommand_path.replace(" ", "_")]
        return "_".join(part for part in parts if part).strip("_") or "unknown_intent"


class HybridHit(BaseModel):
    """Merged lexical/vector result used by the proposal compiler."""

    model_config = ConfigDict(frozen=True)

    candidate: CandidateRecord
    lexical_rank: int | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    vector_similarity: float | None = None
    rrf_score: float

    @property
    def vector_cosine_distance(self) -> float | None:
        if self.vector_similarity is None:
            return None
        return 1.0 - float(self.vector_similarity)


class QueryEncoder:
    """Hot resident Sentence-Transformers encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        warm_vector = self.encode("semantic cli agent warmup")
        if warm_vector.shape != (1, EMBEDDING_DIMENSIONS):
            raise RuntimeError(
                f"{model_name} produced shape {warm_vector.shape}; "
                f"expected (1, {EMBEDDING_DIMENSIONS})"
            )

    def encode(self, text: str | list[str]) -> np.ndarray:
        started = time.perf_counter()
        vectors = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"embedding dimension mismatch: got {matrix.shape[1]}, "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if duration_ms > 10.0:
            LOGGER.debug("embedding encode took %.3fms", duration_ms)
        return matrix


class SearchService:
    """Coordinates FTS5 lexical lookup, FAISS search, RRF, and fast-path routing."""

    def __init__(self, db_path: Path, encoder: QueryEncoder | None = None) -> None:
        self.db_path = db_path
        self.encoder = encoder or QueryEncoder()
        self.cache = VectorCache(db_path)
        self.gateway = FastPathGateway()
        self.rebuild_cache()

    def rebuild_cache(self) -> None:
        try:
            conn = open_read_only_database(self.db_path)
            try:
                documents = fetch_searchable_documents(conn)
            finally:
                conn.close()

            texts = [doc.searchable_text for doc in documents]
            if texts:
                vectors = self.encoder.encode(texts)
            else:
                vectors = np.empty((0, EMBEDDING_DIMENSIONS), dtype="float32")
            self.cache.rebuild(vectors, documents)
        except Exception:
            LOGGER.exception("failed to rebuild sidecar vector cache")
            raise

    def search(
        self,
        request: SidecarRequest,
        dequeued_at: float | None = None,
    ) -> UntrustedProposal:
        processing_started = time.perf_counter()
        scheduling_delay_ms = (
            max(0.0, (processing_started - dequeued_at) * 1000.0)
            if dequeued_at is not None
            else 0.0
        )

        conn = open_read_only_database(self.db_path)
        try:
            lexical_started = time.perf_counter()
            lexical_hits = self._lexical_lookup(conn, request.query, TOP_POOL_LIMIT)
            lexical_duration_ms = (time.perf_counter() - lexical_started) * 1000.0

            embedding_started = time.perf_counter()
            query_vector = self.encoder.encode(request.query)
            embedding_duration_ms = (time.perf_counter() - embedding_started) * 1000.0
            normalized_query = normalize_vectors(query_vector)
            vector_result = self.cache.search(normalized_query, TOP_POOL_LIMIT)

            vector_rowids = [hit.doc_rowid for hit in vector_result.hits]
            vector_candidates = self._fetch_candidates_by_rowid(conn, vector_rowids)

            rrf_started = time.perf_counter()
            merged_hits = self._merge_rankings(
                lexical_hits=lexical_hits,
                vector_hits=vector_result.hits,
                vector_candidates=vector_candidates,
                limit=min(request.limit, TOP_POOL_LIMIT),
            )
            rrf_duration_ms = (time.perf_counter() - rrf_started) * 1000.0

            telemetry = SidecarTelemetry(
                faiss_matrix_scan_duration_ms=vector_result.faiss_matrix_scan_duration_ms,
                python_scheduling_delay_ms=scheduling_delay_ms,
                lexical_lookup_duration_ms=lexical_duration_ms,
                hybrid_rrf_duration_ms=rrf_duration_ms,
                fast_path_evaluation_duration_ms=0.0,
            )

            fast_started = time.perf_counter()
            fast_decision = self.gateway.evaluate(
                query=request.query,
                request_id=request.request_id,
                hits=merged_hits,
                evidence_factory=lambda hit: self._evidence(
                    hit=hit,
                    embedding_duration_ms=embedding_duration_ms,
                ),
            )
            telemetry.fast_path_evaluation_duration_ms = (
                time.perf_counter() - fast_started
            ) * 1000.0
            if fast_decision.proposal is not None:
                fast_decision.proposal.telemetry = telemetry
                return fast_decision.proposal

            top_hit = merged_hits[0] if merged_hits else None
            return UntrustedProposal(
                request_id=request.request_id,
                source_provenance="LOCAL_HYBRID_RETRIEVAL",
                intent_proposal=IntentProposal(
                    candidate_template_id=top_hit.candidate.doc_id if top_hit else None,
                    typed_intent=(
                        top_hit.candidate.typed_intent if top_hit else "unknown_intent"
                    ),
                    retrieval_evidence=self._evidence(
                        hit=top_hit,
                        embedding_duration_ms=embedding_duration_ms,
                    ),
                    risk_hints=RiskHints(
                        contains_path_arguments=self._looks_like_path_request(request.query)
                    ),
                    raw_untrusted_slots={},
                ),
                telemetry=telemetry,
            )
        except Exception:
            LOGGER.exception(
                "failed to execute hybrid retrieval for request_id=%s", request.request_id
            )
            raise
        finally:
            conn.close()

    def _lexical_lookup(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
    ) -> list[HybridHit]:
        fts_query = self._compile_fts_query(query)
        if fts_query is None:
            return []

        try:
            rows = conn.execute(
                """
                SELECT
                    u.doc_rowid,
                    u.doc_id,
                    u.source_type,
                    u.binary_name,
                    u.subcommand_path,
                    u.intent_description,
                    u.template_argv_json,
                    u.slot_schema_json,
                    u.policy_rule_id,
                    u.trust_state,
                    p.fast_path_allowed,
                    p.required_confirmation_count,
                    p.package_manager_risk_level,
                    p.privilege_risk_level,
                    p.destructive_recursive_level,
                    bm25(docs_external_fts) AS lexical_bm25
                FROM docs_external_fts
                JOIN unified_documents AS u
                    ON u.doc_rowid = docs_external_fts.rowid
                JOIN policy_rules AS p
                    ON p.rule_id = u.policy_rule_id
                WHERE docs_external_fts MATCH ?
                  AND u.trust_state NOT IN ('DISABLED', 'REVOKED')
                ORDER BY lexical_bm25 ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.Error:
            LOGGER.exception("FTS5 lexical lookup failed for query=%r", query)
            raise

        hits: list[HybridHit] = []
        for rank, row in enumerate(rows, start=1):
            bm25_value = float(row["lexical_bm25"])
            lexical_score = 1.0 / (1.0 + abs(bm25_value))
            hits.append(
                HybridHit(
                    candidate=self._candidate_from_row(row),
                    lexical_rank=rank,
                    lexical_score=lexical_score,
                    rrf_score=0.0,
                )
            )
        return hits

    def _fetch_candidates_by_rowid(
        self,
        conn: sqlite3.Connection,
        rowids: list[int],
    ) -> dict[int, CandidateRecord]:
        if not rowids:
            return {}

        placeholders = ",".join("?" for _ in rowids)
        try:
            rows = conn.execute(
                f"""
                SELECT
                    u.doc_rowid,
                    u.doc_id,
                    u.source_type,
                    u.binary_name,
                    u.subcommand_path,
                    u.intent_description,
                    u.template_argv_json,
                    u.slot_schema_json,
                    u.policy_rule_id,
                    u.trust_state,
                    p.fast_path_allowed,
                    p.required_confirmation_count,
                    p.package_manager_risk_level,
                    p.privilege_risk_level,
                    p.destructive_recursive_level
                FROM unified_documents AS u
                JOIN policy_rules AS p
                    ON p.rule_id = u.policy_rule_id
                WHERE u.doc_rowid IN ({placeholders})
                  AND u.trust_state NOT IN ('DISABLED', 'REVOKED')
                """,
                tuple(rowids),
            ).fetchall()
        except sqlite3.Error:
            LOGGER.exception("failed to hydrate vector candidates")
            raise

        return {int(row["doc_rowid"]): self._candidate_from_row(row) for row in rows}

    @staticmethod
    def _merge_rankings(
        lexical_hits: list[HybridHit],
        vector_hits: list[object],
        vector_candidates: dict[int, CandidateRecord],
        limit: int,
    ) -> list[HybridHit]:
        by_rowid: dict[int, dict[str, object]] = {}

        for hit in lexical_hits:
            by_rowid[hit.candidate.doc_rowid] = {
                "candidate": hit.candidate,
                "lexical_rank": hit.lexical_rank,
                "lexical_score": hit.lexical_score,
                "vector_rank": None,
                "vector_similarity": None,
            }

        for vector_hit in vector_hits:
            rowid = int(getattr(vector_hit, "doc_rowid"))
            candidate = vector_candidates.get(rowid)
            if candidate is None:
                LOGGER.error("vector result rowid=%s failed relational hydration", rowid)
                continue
            entry = by_rowid.setdefault(
                rowid,
                {
                    "candidate": candidate,
                    "lexical_rank": None,
                    "lexical_score": None,
                    "vector_rank": None,
                    "vector_similarity": None,
                },
            )
            entry["vector_rank"] = int(getattr(vector_hit, "distance_rank"))
            entry["vector_similarity"] = float(getattr(vector_hit, "score"))

        merged: list[HybridHit] = []
        for entry in by_rowid.values():
            lexical_rank = entry["lexical_rank"]
            vector_rank = entry["vector_rank"]
            rrf_score = 0.0
            if isinstance(lexical_rank, int):
                rrf_score += 1.0 / (RRF_K + lexical_rank)
            if isinstance(vector_rank, int):
                rrf_score += 1.0 / (RRF_K + vector_rank)

            merged.append(
                HybridHit(
                    candidate=entry["candidate"],  # type: ignore[arg-type]
                    lexical_rank=lexical_rank if isinstance(lexical_rank, int) else None,
                    lexical_score=(
                        float(entry["lexical_score"])
                        if entry["lexical_score"] is not None
                        else None
                    ),
                    vector_rank=vector_rank if isinstance(vector_rank, int) else None,
                    vector_similarity=(
                        float(entry["vector_similarity"])
                        if entry["vector_similarity"] is not None
                        else None
                    ),
                    rrf_score=rrf_score,
                )
            )

        merged.sort(
            key=lambda hit: (
                hit.rrf_score,
                hit.vector_similarity if hit.vector_similarity is not None else -1.0,
                hit.lexical_score if hit.lexical_score is not None else -1.0,
            ),
            reverse=True,
        )
        return merged[:limit]

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
        return CandidateRecord(
            doc_rowid=int(row["doc_rowid"]),
            doc_id=str(row["doc_id"]),
            source_type=str(row["source_type"]),
            binary_name=str(row["binary_name"]),
            subcommand_path=str(row["subcommand_path"] or ""),
            intent_description=str(row["intent_description"]),
            template_argv_json=str(row["template_argv_json"]),
            slot_schema_json=str(row["slot_schema_json"]),
            policy_rule_id=str(row["policy_rule_id"]),
            trust_state=str(row["trust_state"]),
            fast_path_allowed=bool(int(row["fast_path_allowed"])),
            required_confirmation_count=int(row["required_confirmation_count"]),
            package_manager_risk_level=str(row["package_manager_risk_level"]),
            privilege_risk_level=str(row["privilege_risk_level"]),
            destructive_recursive_level=str(row["destructive_recursive_level"]),
        )

    @staticmethod
    def _compile_fts_query(query: str) -> str | None:
        tokens = []
        for token in FTS_TOKEN_RE.findall(query):
            cleaned = token.strip("./:+-")
            if not cleaned:
                continue
            escaped = cleaned.replace('"', '""')
            tokens.append(f'"{escaped}"')
        if not tokens:
            return None
        return " OR ".join(tokens[:12])

    @staticmethod
    def _evidence(
        hit: HybridHit | None,
        embedding_duration_ms: float,
    ) -> RetrievalEvidence:
        return RetrievalEvidence(
            fts5_lexical_score=hit.lexical_score if hit else None,
            vector_cosine_distance=hit.vector_cosine_distance if hit else None,
            vector_rank=hit.vector_rank if hit else None,
            embedding_duration_ms=embedding_duration_ms,
        )

    @staticmethod
    def _looks_like_path_request(query: str) -> bool:
        tokens = query.replace("\\", "/").split()
        return any("/" in token or "." in token or token.startswith("~") for token in tokens)

    def handle_json(self, payload: bytes, dequeued_at: float | None = None) -> bytes:
        try:
            request = SidecarRequest.model_validate_json(payload)
            proposal = self.search(request, dequeued_at=dequeued_at)
            return proposal.model_dump_json(exclude_none=False).encode("utf-8") + b"\n"
        except Exception:
            LOGGER.exception("failed to handle sidecar JSON payload")
            raise


def build_service(db_path: Path) -> SearchService:
    return SearchService(db_path=db_path)
