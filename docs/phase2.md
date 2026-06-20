# Phase 2: Python Background Daemon and Vector Cache

Phase 2 adds the untrusted Python machine-learning sidecar. It is intentionally
decoupled from command execution. Rust remains responsible for policy, slot
binding, path validation, confirmation, and process spawning.

## Files

- `src/sidecar/daemon.py`: Unix Domain Socket server with peer UID checks and
  signal cleanup.
- `src/sidecar/cache.py`: read-only SQLite extraction and in-memory FAISS
  `IndexIDMap(IndexFlatIP)` cache.
- `src/sidecar/search.py`: hot Sentence-Transformers encoder and strict
  Pydantic request/response schemas.
- `requirements-sidecar.txt`: Python dependencies for the sidecar runtime.

## Runtime Model

The daemon listens on a Unix socket inside the secured runtime directory. On
Linux it verifies the peer UID with `SO_PEERCRED`, rejecting clients that do not
match the daemon UID.

The daemon reads SQLite using URI mode `mode=ro`. It extracts rows from
`unified_documents` while excluding `DISABLED` and `REVOKED` records. It builds
a memory-only FAISS cache from SQLite data at startup. SQLite remains the source
of truth; FAISS is a rebuildable cache.

FAISS is wrapped as `IndexIDMap(IndexFlatIP)`, not a raw flat index. The sidecar
adds vectors with `add_with_ids`, using SQLite `doc_rowid` values as the FAISS
IDs. This prevents sparse SQLite rows from drifting away from vector metadata
after document deletes or updates.

Both document vectors and query vectors are explicitly L2-normalized using
`faiss.normalize_L2` before they touch the inner-product index. With normalized
vectors, `IndexFlatIP` ranks by cosine similarity instead of raw vector scale.

## Protocol

Requests are newline-delimited JSON:

```json
{
  "protocol_version": "1.0.0",
  "request_id": "req_123",
  "query": "restore src/main.rs",
  "limit": 5
}
```

Responses are strict `untrusted_proposal_json` payloads. They are parseable, but
not trusted. Rust must still reload the trusted template and validate every slot.

Responses include a telemetry object:

```json
{
  "telemetry": {
    "faiss_matrix_scan_duration_ms": 0.14,
    "python_scheduling_delay_ms": 0.03
  }
}
```

`faiss_matrix_scan_duration_ms` measures only the FAISS lookup window.
`python_scheduling_delay_ms` measures the time between worker handler entry and
the start of request processing inside the search service.

## Setup

Install sidecar dependencies:

```bash
python3 -m pip install -r requirements-sidecar.txt
```

Run the sidecar:

```bash
python3 -m src.sidecar.daemon --runtime-dir "$SEMANTIC_CLI_AGENT_HOME"
```

If `SEMANTIC_CLI_AGENT_HOME` is not set, the daemon uses
`$XDG_DATA_HOME/semantic-cli-agent` or `$HOME/.local/share/semantic-cli-agent`.

## Security Boundary

The sidecar never executes commands and never mutates trusted policy. Its output
only contains a candidate template id, retrieval evidence, risk hints, and raw
untrusted slots.
