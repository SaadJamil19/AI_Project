# Phase 4: Grammar-Constrained LLM Parsing and Rust Slot Binding

Phase 4 adds the unsafe-language parsing path without changing the trust model.
Python can ask a local Ollama model to produce structured JSON, but Rust remains
the authority for template lookup, slot validation, and later command execution.

## Python Ollama Client

`src/sidecar/llm_client.py` wraps Ollama's local `/api/generate` endpoint. It
uses a persistent opener, requests non-streaming JSON, and sends `keep_alive =
-1` both at the top-level Ollama field and inside the options payload so the
local model stays resident.

The request uses Ollama's JSON schema `format` field to constrain output to the
`untrusted_proposal_json` envelope. The schema removes conversational output:
the model can only return the nested `intent_proposal` object with retrieval
evidence, risk hints, and `raw_untrusted_slots`.

The returned object is still untrusted. The Python client parses it with
Pydantic and adds local telemetry, but it does not validate path safety or
execute anything.

## Trusted Template Fetch

`cmd/cli-agent/src/storage.rs` now exposes
`fetch_trusted_template_by_doc_id`. Given a candidate template id from the
sidecar, Rust reloads the trusted database row and returns:

- `template_argv_json`
- `slot_schema_json`
- `policy_rule_id`
- `trust_state`

This prevents Python from defining or modifying the validation schema. The
schema that matters is always the one stored in SQLite.

The fetch routine deliberately returns rows in every lifecycle state, including
`DISABLED` and `REVOKED`. Filtering those rows out in Python or SQL is not
enough for a zero-trust design. Rust now checks `trust_state` inside the slot
binding path and returns `SlotValidationError::TemplateRevoked` before any slot
processing if the template has been disabled or revoked.

## Rust Slot Binding

`cmd/cli-agent/src/validate.rs` implements the zero-trust slot binder. It parses
trusted `slot_schema_json` and the daemon's raw slot map, then enforces:

- required slot presence
- rejection of unknown slots
- JSON string slot values for string formats
- native JSON integer primitives for slots that declare `kind: "integer"` or
  `allowed_formats: ["integer"]`
- per-slot byte limits
- integer lower and upper bounds through `min_value` and `max_value`
- allowed primitive formats such as `relative_path`, `filename`, `git_ref`,
  `flag`, `integer`, and `safe_token`

Errors use `thiserror` through `SlotValidationError`, so the caller can block
execution cleanly and report the exact reason.

## CLI Wrapper

`cli-agent validate-slots <request-id> <template-id> <slots-json>` reloads the
template schema from SQLite and validates the untrusted slot JSON against it.
This is a testable boundary for the later shell execution phase.

The wrapper opens a writable SQLite connection because validation failures must
be correlated with the active request. If Rust validation fails, it executes an
isolated transaction that moves the matching `request_records.execution_status`
to `SECURITY_BLOCKED` before returning the validation error to the shell layer.
That makes blocked execution auditable even when the failure happens before
command preview or process spawning.

## Verification

Run Rust tests:

```bash
cargo test
```

Syntax-check the new Python module:

```bash
python -m py_compile src/sidecar/llm_client.py
```
