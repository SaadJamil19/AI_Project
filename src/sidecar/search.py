from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

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
from .llm_client import OllamaClient, OllamaProtocolError


LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
PROTOCOL_VERSION = "1.0.0"
RRF_K = 60
TOP_POOL_LIMIT = 5
FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+")
# SQLite FTS5's default unicode61 tokenizer does not remove stop words, so
# without this, a query sharing only a pronoun or article with some
# unrelated seeded description (e.g. "me" in "give me a curl command" vs.
# "me" in "show me the current ... status") registers as a real lexical
# match. That's enough to set lexical_rank, which short-circuits the
# topical-relevance floor below (any lexical match is trusted outright) -
# so an utterly unrelated query can still land a confident, wrong match.
FTS_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "from", "into", "as", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "doing", "have", "has", "had",
        "i", "me", "my", "you", "your", "it", "its", "this", "that",
        "these", "those", "he", "she", "they", "we", "us", "them",
        # Not classic linguistic stopwords, but generic request-framing
        # words that carry zero information about *which* command is
        # wanted. Without these, a learned template whose intent_description
        # is literally the user's own original phrasing (e.g. "give me a
        # curl command...") keeps lexically matching every future request
        # phrased the same generic way (e.g. "give me a command to commit
        # on github") purely on shared filler words - a real, observed false
        # match, not a hypothetical one.
        "give", "show", "get", "command", "want", "need", "please", "tell",
    }
)
# A hit ranked #1 in at least one of the two retrieval modalities scores at
# least 1/(RRF_K + 1) under reciprocal rank fusion. Below that, the best
# candidate wasn't anyone's top pick, so its slot extraction is treated as
# unreliable enough to hand off to the local LLM for a second pass.
HYBRID_CONFIDENCE_THRESHOLD = 1.0 / (RRF_K + 1)
# Absolute floor, used only when there is zero lexical overlap at all
# (see _is_topically_relevant) - a candidate that is the *closest
# available* match is not the same thing as a *relevant* match, since
# cosine similarity always returns a nearest neighbor even among entirely
# unrelated documents. 0.35 was too low: "give me a command to commit on
# github" (no shared words with any seeded template at all) scored 0.459
# against git_status and was wrongly accepted. Every genuine match observed
# so far ("undo my changes to main.rs", "make a new branch called
# feature-x", etc.) has real shared vocabulary and is accepted through the
# lexical-overlap branch instead, never reaching this floor - so raising
# it well above that observed false positive costs nothing they need.
# Still a heuristic tuned on a handful of real examples, not a derived
# constant; expect to revisit as the template corpus grows.
MIN_ABSOLUTE_VECTOR_SIMILARITY = 0.6


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


class CacheInvalidationRequest(BaseModel):
    """Active push notification from the trusted Rust client.

    Sent over the same UDS pipe used for search requests immediately after
    `ai-learn` commits a new template, instead of the sidecar passively
    polling a database flag for staleness.
    """

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = Field(min_length=1)
    command: Literal["invalidate_cache"]
    request_id: str = Field(min_length=1, max_length=128)

    @field_validator("protocol_version")
    @classmethod
    def protocol_must_match(cls, value: str) -> str:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version {value!r}")
        return value


class CacheInvalidationAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = PROTOCOL_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    source_provenance: str = "LOCAL_ML_SIDECAR"
    status: Literal["CACHE_REBUILT"] = "CACHE_REBUILT"
    document_count: int = Field(ge=0)
    rebuild_duration_ms: float = Field(ge=0.0)


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
    cache_freshness_check_duration_ms: float = Field(default=0.0, ge=0.0)
    llm_fallback_duration_ms: float = Field(default=0.0, ge=0.0)


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

    def __init__(
        self,
        db_path: Path,
        encoder: QueryEncoder | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.db_path = db_path
        self.encoder = encoder or QueryEncoder()
        self.cache = VectorCache(db_path)
        self.gateway = FastPathGateway()
        # Constructing OllamaClient only opens a urllib opener; it never
        # touches the network until generate_proposal() is actually called,
        # so the daemon still starts and serves fast-path/hybrid results
        # fine even if Ollama was never installed.
        self.llm_client = llm_client or OllamaClient()
        self._generation_lock = threading.Lock()
        self._known_generation: int | None = None
        self.rebuild_cache()

        conn = open_read_only_database(self.db_path)
        try:
            self._known_generation = self._read_generation(conn)
        finally:
            conn.close()

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

    @staticmethod
    def _read_generation(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'unified_documents_generation'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            LOGGER.error("unified_documents_generation value is not an integer: %r", row["value"])
            return 0

    def _ensure_cache_fresh(self, conn: sqlite3.Connection) -> None:
        """Lazy-loading safety net for a dropped active UDS invalidation signal.

        This is a single indexed primary-key lookup, so it stays cheap on the
        hot query path. It only pays the cost of a full `rebuild_cache()` when
        the trigger-maintained `unified_documents_generation` counter has
        actually moved past what this process last observed, which happens
        whenever the daemon missed (or raced) an explicit invalidation push.
        """

        current = self._read_generation(conn)
        if current == self._known_generation:
            return

        with self._generation_lock:
            if current == self._known_generation:
                return
            LOGGER.warning(
                "vector cache generation drifted (known=%s, current=%s); rebuilding",
                self._known_generation,
                current,
            )
            self.rebuild_cache()
            self._known_generation = self._read_generation(conn)

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
            freshness_started = time.perf_counter()
            self._ensure_cache_fresh(conn)
            freshness_duration_ms = (time.perf_counter() - freshness_started) * 1000.0

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
                cache_freshness_check_duration_ms=freshness_duration_ms,
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
            if top_hit is not None and not self._is_topically_relevant(top_hit):
                # Vector search always returns a nearest neighbor, even when
                # nothing in the index is actually related to the query -
                # there is no such thing as "no match" for cosine distance,
                # only "less close." Ranking #1 among irrelevant candidates
                # is not evidence of a real match. Without this floor, a
                # totally unrelated query (e.g. asking about curl) could
                # confidently resolve to whichever seeded template happens
                # to sit nearest in embedding space - and if that template
                # needs no slots, Rust has nothing left to catch the wrong
                # guess with. Falling through with top_hit = None routes
                # this into the exact same path as a genuine zero-hit miss.
                top_hit = None

            # Three independent reasons to consult the local LLM:
            #   1. zero hits, where retrieval has no candidate at all and the
            #      model may still emit a candidate_template_id that Rust will
            #      reload and verify before use.
            #   2. low confidence, where the candidate was not anyone's top
            #      pick, so even which template applies is shaky.
            #   3. the candidate IS confidently matched, but its schema
            #      declares slots, and this code path (unlike the fast-path
            #      gateway) has no deterministic way to fill them - without
            #      the LLM, raw_untrusted_slots stays empty and Rust will
            #      always reject the proposal as MissingRequiredSlot. A
            #      confident match to a slotted template is therefore not
            #      "done"; it has just identified what still needs slots.
            if top_hit is None:
                llm_proposal = self._attempt_zero_hit_llm_fallback(
                    request=request,
                    embedding_duration_ms=embedding_duration_ms,
                    telemetry=telemetry,
                )
                if llm_proposal is not None:
                    return llm_proposal
            elif (
                top_hit.rrf_score < HYBRID_CONFIDENCE_THRESHOLD
                or self._candidate_declares_slots(top_hit.candidate)
            ):
                local_slots = self._attempt_local_slot_extraction(
                    request=request,
                    top_hit=top_hit,
                    embedding_duration_ms=embedding_duration_ms,
                    telemetry=telemetry,
                )
                if local_slots is not None:
                    return local_slots

                llm_proposal = self._attempt_llm_fallback(
                    request=request,
                    top_hit=top_hit,
                    embedding_duration_ms=embedding_duration_ms,
                    telemetry=telemetry,
                )
                if llm_proposal is not None:
                    return llm_proposal

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
            if not cleaned or cleaned.lower() in FTS_STOPWORDS:
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

    @staticmethod
    def _is_topically_relevant(hit: HybridHit) -> bool:
        """True if a hit has real evidence of relevance, not just rank.

        A literal lexical match (any FTS5 hit at all) always counts: shared
        vocabulary is a much stronger signal than embedding-space proximity
        for short command-style queries. Absent that, the vector similarity
        itself must clear an absolute floor - being *closest* among the
        indexed documents is not evidence of being *close*.
        """

        if hit.lexical_rank is not None:
            return True
        return hit.vector_similarity is not None and hit.vector_similarity >= MIN_ABSOLUTE_VECTOR_SIMILARITY

    @staticmethod
    def _candidate_declares_slots(candidate: CandidateRecord) -> bool:
        try:
            schema = json.loads(candidate.slot_schema_json)
        except json.JSONDecodeError:
            LOGGER.error("slot_schema_json is malformed for doc_id=%s", candidate.doc_id)
            return False
        if isinstance(schema, dict):
            schema = schema.get("slots", [])
        return isinstance(schema, list) and len(schema) > 0

    def _attempt_local_slot_extraction(
        self,
        request: SidecarRequest,
        top_hit: HybridHit,
        embedding_duration_ms: float,
        telemetry: SidecarTelemetry,
    ) -> UntrustedProposal | None:
        """Extracts simple trusted-schema slots without invoking the LLM.

        This deliberately handles only narrow, deterministic cases. Rust still
        validates the produced slots against the trusted schema, path boundary,
        and policy rules before execution.
        """

        slots = self._extract_slots_from_query(top_hit.candidate, request.query)
        if slots is None:
            return None

        return UntrustedProposal(
            request_id=request.request_id,
            source_provenance="LOCAL_DETERMINISTIC_SLOTS",
            intent_proposal=IntentProposal(
                candidate_template_id=top_hit.candidate.doc_id,
                typed_intent=top_hit.candidate.typed_intent,
                retrieval_evidence=self._evidence(
                    hit=top_hit,
                    embedding_duration_ms=embedding_duration_ms,
                ),
                risk_hints=RiskHints(
                    contains_path_arguments=self._looks_like_path_request(request.query)
                ),
                raw_untrusted_slots=slots,
            ),
            telemetry=telemetry,
        )

    @staticmethod
    def _extract_slots_from_query(
        candidate: CandidateRecord,
        query: str,
    ) -> dict[str, str] | None:
        try:
            schema = json.loads(candidate.slot_schema_json)
        except json.JSONDecodeError:
            return None
        if isinstance(schema, dict):
            schema = schema.get("slots", [])
        if not isinstance(schema, list) or not schema:
            return None

        extracted: dict[str, str] = {}
        for rule in schema:
            if not isinstance(rule, dict):
                return None
            name = rule.get("name")
            formats = rule.get("allowed_formats", [])
            if not isinstance(name, str) or not isinstance(formats, list):
                return None

            if "relative_path" in formats:
                value = SearchService._extract_relative_path_token(query)
            elif "safe_token" in formats:
                value = SearchService._extract_named_safe_token(query)
            else:
                return None

            if value is None:
                return None
            extracted[name] = value

        return extracted

    @staticmethod
    def _extract_relative_path_token(query: str) -> str | None:
        for raw in reversed(query.split()):
            token = raw.strip(" \t\r\n'\"`.,;:()[]{}<>")
            if not token or token.startswith(("/", "\\")) or ".." in token:
                continue
            if "/" not in token and "." not in token:
                continue
            if all(ch.isascii() and (ch.isalnum() or ch in "._-/\\") for ch in token):
                return token
        return None

    @staticmethod
    def _extract_named_safe_token(query: str) -> str | None:
        patterns = (
            r"\bcalled\s+([A-Za-z0-9._:@-]+)",
            r"\bnamed\s+([A-Za-z0-9._:@-]+)",
            r"\bbranch\s+([A-Za-z0-9._:@-]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" \t\r\n'\"`.,;:()[]{}<>")
        return None

    def _attempt_llm_fallback(
        self,
        request: SidecarRequest,
        top_hit: HybridHit,
        embedding_duration_ms: float,
        telemetry: SidecarTelemetry,
    ) -> UntrustedProposal | None:
        """Asks the local Ollama model to extract slots for a weak hybrid hit.

        Only called when retrieval already named a specific candidate. The
        model is anchored to that one candidate's trusted schema; it is never
        allowed to override the candidate id chosen by retrieval, so
        `candidate_template_id` always comes from `top_hit`, not from the
        model's own (forbidden-to-trust-anyway) echo of it. Any failure here
        - Ollama not installed, not running, or returning malformed output -
        is logged once at warning level and treated as non-fatal: the caller
        falls back to the existing hybrid-retrieval guess instead of failing
        the whole request.
        """

        llm_started = time.perf_counter()
        try:
            llm_proposal = self.llm_client.generate_proposal(
                request_id=request.request_id,
                user_prompt=request.query,
                candidate_template_id=top_hit.candidate.doc_id,
                typed_intent=top_hit.candidate.typed_intent,
                slot_schema_json=top_hit.candidate.slot_schema_json,
            )
        except (OllamaProtocolError, ConnectionError):
            LOGGER.warning(
                "local LLM fallback unavailable for request_id=%s; using hybrid retrieval guess instead",
                request.request_id,
                exc_info=True,
            )
            return None
        finally:
            telemetry.llm_fallback_duration_ms = (time.perf_counter() - llm_started) * 1000.0

        return UntrustedProposal(
            request_id=request.request_id,
            source_provenance="LOCAL_LLM_GRAMMAR",
            intent_proposal=IntentProposal(
                candidate_template_id=top_hit.candidate.doc_id,
                typed_intent=top_hit.candidate.typed_intent,
                retrieval_evidence=self._evidence(
                    hit=top_hit,
                    embedding_duration_ms=embedding_duration_ms,
                ),
                risk_hints=RiskHints(
                    contains_path_arguments=(
                        self._looks_like_path_request(request.query)
                        or llm_proposal.intent_proposal.risk_hints.contains_path_arguments
                    )
                ),
                raw_untrusted_slots=llm_proposal.intent_proposal.raw_untrusted_slots,
            ),
            telemetry=telemetry,
        )

    def _attempt_zero_hit_llm_fallback(
        self,
        request: SidecarRequest,
        embedding_duration_ms: float,
        telemetry: SidecarTelemetry,
    ) -> UntrustedProposal | None:
        """Lets Ollama propose an intent when retrieval finds no template.

        This is intentionally weakly trusted. The sidecar may return a
        candidate_template_id, but Rust must still reload that id from SQLite
        and run slot, path, policy, and lifecycle validation before anything
        can execute. If Ollama is unavailable or emits a malformed envelope,
        the caller falls through to the normal no-template response.
        """

        llm_started = time.perf_counter()
        try:
            llm_proposal = self.llm_client.generate_proposal(
                request_id=request.request_id,
                user_prompt=request.query,
                candidate_template_id=None,
                typed_intent="unknown_intent",
                slot_schema_json="[]",
            )
        except (OllamaProtocolError, ConnectionError):
            LOGGER.warning(
                "zero-hit local LLM fallback unavailable for request_id=%s",
                request.request_id,
                exc_info=True,
            )
            return None
        finally:
            telemetry.llm_fallback_duration_ms = (time.perf_counter() - llm_started) * 1000.0

        candidate_template_id = llm_proposal.intent_proposal.candidate_template_id
        if not candidate_template_id:
            # Catches both an explicit null and a hallucinated empty string -
            # phi3:mini has been observed returning "" rather than null for
            # "no real candidate," and an empty string is not a real doc_id.
            # Letting it through would have Rust's fetch_trusted_template_by_doc_id
            # hard-error on a lookup that can never succeed, instead of the
            # clean OBSERVING outcome a genuine miss is supposed to produce.
            LOGGER.warning(
                "zero-hit LLM fallback returned no usable candidate_template_id for request_id=%s",
                request.request_id,
            )
            return None

        return UntrustedProposal(
            request_id=request.request_id,
            source_provenance="LOCAL_LLM_GRAMMAR",
            intent_proposal=IntentProposal(
                candidate_template_id=candidate_template_id,
                typed_intent=llm_proposal.intent_proposal.typed_intent,
                retrieval_evidence=RetrievalEvidence(
                    fts5_lexical_score=None,
                    vector_cosine_distance=None,
                    vector_rank=None,
                    embedding_duration_ms=embedding_duration_ms,
                ),
                risk_hints=RiskHints(
                    contains_path_arguments=(
                        self._looks_like_path_request(request.query)
                        or llm_proposal.intent_proposal.risk_hints.contains_path_arguments
                    )
                ),
                raw_untrusted_slots=llm_proposal.intent_proposal.raw_untrusted_slots,
            ),
            telemetry=telemetry,
        )

    def handle_json(self, payload: bytes, dequeued_at: float | None = None) -> bytes:
        try:
            request = SidecarRequest.model_validate_json(payload)
            proposal = self.search(request, dequeued_at=dequeued_at)
            return proposal.model_dump_json(exclude_none=False).encode("utf-8") + b"\n"
        except Exception:
            LOGGER.exception("failed to handle sidecar JSON payload")
            raise

    def handle_invalidate_json(self, payload: bytes) -> bytes:
        """Synchronously clear and rebuild the FAISS cache from SQLite.

        Runs on the worker thread handling the invalidation socket request,
        so the daemon has already rebuilt before it accepts the next query.
        """

        try:
            request = CacheInvalidationRequest.model_validate_json(payload)
            started = time.perf_counter()
            with self._generation_lock:
                self.rebuild_cache()
                conn = open_read_only_database(self.db_path)
                try:
                    self._known_generation = self._read_generation(conn)
                finally:
                    conn.close()
            rebuild_duration_ms = (time.perf_counter() - started) * 1000.0
            ack = CacheInvalidationAck(
                request_id=request.request_id,
                document_count=len(self.cache.documents()),
                rebuild_duration_ms=rebuild_duration_ms,
            )
            return ack.model_dump_json().encode("utf-8") + b"\n"
        except Exception:
            LOGGER.exception("failed to handle sidecar cache invalidation request")
            raise


def build_service(db_path: Path) -> SearchService:
    return SearchService(db_path=db_path)
