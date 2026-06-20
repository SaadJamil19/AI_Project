from __future__ import annotations

import logging
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
    VectorSearchHit,
    fetch_searchable_documents,
    normalize_vectors,
    open_read_only_database,
)


LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
PROTOCOL_VERSION = "1.0.0"


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

    fts5_lexical_rank: float | None = None
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
    source_provenance: str = "LOCAL_ML_SIDECAR"
    intent_proposal: IntentProposal
    telemetry: SidecarTelemetry


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
    """Composes SQLite extraction, embedding, FAISS rebuild, and proposal generation."""

    def __init__(self, db_path: Path, encoder: QueryEncoder | None = None) -> None:
        self.db_path = db_path
        self.encoder = encoder or QueryEncoder()
        self.cache = VectorCache(db_path)
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

        try:
            embedding_started = time.perf_counter()
            query_vector = self.encoder.encode(request.query)
            embedding_duration_ms = (time.perf_counter() - embedding_started) * 1000.0
            normalized_query = normalize_vectors(query_vector)
            vector_result = self.cache.search(normalized_query, request.limit)
        except Exception:
            LOGGER.exception("failed to execute vector search for request_id=%s", request.request_id)
            raise

        top_hit = vector_result.hits[0] if vector_result.hits else None
        proposal = IntentProposal(
            candidate_template_id=top_hit.doc_id if top_hit else None,
            typed_intent=self._intent_name(top_hit),
            retrieval_evidence=RetrievalEvidence(
                fts5_lexical_rank=None,
                vector_cosine_distance=self._cosine_distance(top_hit),
                vector_rank=top_hit.distance_rank if top_hit else None,
                embedding_duration_ms=embedding_duration_ms,
            ),
            risk_hints=RiskHints(
                contains_path_arguments=self._looks_like_path_request(request.query)
            ),
            raw_untrusted_slots={},
        )
        return UntrustedProposal(
            request_id=request.request_id,
            intent_proposal=proposal,
            telemetry=SidecarTelemetry(
                faiss_matrix_scan_duration_ms=vector_result.faiss_matrix_scan_duration_ms,
                python_scheduling_delay_ms=scheduling_delay_ms,
            ),
        )

    @staticmethod
    def _intent_name(hit: VectorSearchHit | None) -> str:
        if hit is None:
            return "unknown_intent"
        parts = [hit.binary_name, hit.subcommand_path.replace(" ", "_")]
        return "_".join(part for part in parts if part).strip("_") or "unknown_intent"

    @staticmethod
    def _cosine_distance(hit: VectorSearchHit | None) -> float | None:
        if hit is None:
            return None
        return 1.0 - float(hit.score)

    @staticmethod
    def _looks_like_path_request(query: str) -> bool:
        tokens = query.replace("\\", "/").split()
        return any("/" in token or "." in token for token in tokens)

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
