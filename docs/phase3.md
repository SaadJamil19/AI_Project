# Phase 3: Hybrid Retrieval and Fast-Path Routing

Phase 3 replaces vector-only lookup with a hybrid retrieval compiler. The
sidecar now combines SQLite FTS5 lexical matches with FAISS cosine search, then
uses Reciprocal Rank Fusion to choose the best candidate templates.

## Files

- `src/sidecar/search.py`: coordinates lexical FTS5 lookup, vector lookup,
  RRF merging, response schema validation, and telemetry.
- `src/sidecar/gateway.py`: evaluates deterministic fast-path eligibility
  before any LLM fallback is considered.

## Hybrid Retrieval

Every request runs through two retrieval paths:

1. FTS5 lexical search against `docs_external_fts`.
2. FAISS vector search against the memory-resident `IndexIDMap(IndexFlatIP)`.

Both paths are capped to the top five records. The FTS5 path always joins back
to `unified_documents` and applies:

```sql
WHERE trust_state NOT IN ('DISABLED', 'REVOKED')
```

That filter prevents deprecated templates from entering the ranking pool even if
their FTS rows still exist.

The merger uses Reciprocal Rank Fusion with `k = 60`. This prevents either FTS5
or FAISS from dominating solely because its raw score scale is larger. Lexical
BM25 is used for ranking and converted to a bounded evidence score for protocol
output.

## Fast-Path Gate

The fast-path gate only triggers when all checks pass:

1. The prompt starts with the exact approved binary token and subcommand path.
2. The candidate template has `STATIC_VERIFIED` or `PROMOTED_FASTPATH` trust.
3. The joined policy row has `fast_path_allowed = 1`, no elevated confirmation
   requirement, and the risky package-manager, privilege, and destructive
   surfaces are blocked by policy.

If a candidate passes, Python extracts simple slots from the template argv
placeholders and returns a `LOCAL_FAST_PATH` proposal. Those slots remain
untrusted. Rust must still reload the template by id, validate each slot against
trusted policy, and spawn only through argv-safe process APIs.

## Protocol Shape

All normal responses use the strict schema:

```json
{
  "protocol_version": "1.0.0",
  "request_id": "req_123",
  "source_provenance": "LOCAL_HYBRID_RETRIEVAL",
  "intent_proposal": {
    "candidate_template_id": "git-status-static",
    "typed_intent": "git_status",
    "retrieval_evidence": {
      "fts5_lexical_score": 0.98,
      "vector_cosine_distance": 0.04,
      "vector_rank": 1,
      "embedding_duration_ms": 7.6
    },
    "risk_hints": {
      "contains_path_arguments": false
    },
    "raw_untrusted_slots": {}
  },
  "telemetry": {
    "faiss_matrix_scan_duration_ms": 0.12,
    "python_scheduling_delay_ms": 0.03,
    "lexical_lookup_duration_ms": 0.20,
    "hybrid_rrf_duration_ms": 0.02,
    "fast_path_evaluation_duration_ms": 0.01
  }
}
```

Fast-path responses use the same object layout with
`source_provenance = "LOCAL_FAST_PATH"`.

## Verification

Syntax-check the sidecar:

```bash
python -m py_compile src/sidecar/search.py src/sidecar/gateway.py src/sidecar/cache.py src/sidecar/daemon.py
```

Run Rust storage tests:

```bash
cargo test
```
