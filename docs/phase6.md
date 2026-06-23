# Phase 6: Shell Integration, Signals, and Learning

Phase 6 closes the local trusted execution loop. Rust now owns direct process
spawning, confirmation prompts, signal-aware interruption checks, and explicit
user correction ingestion.

## Direct Execution

`cmd/cli-agent/src/execute.rs` executes only structured argv vectors. It uses
`std::process::Command` directly and never builds a shell command string.

Interactive execution maps streams directly to the parent shell:

- `stdin(Stdio::inherit())`
- `stdout(Stdio::inherit())`
- `stderr(Stdio::inherit())`

This keeps tools such as editors, pagers, and interactive CLIs usable while
avoiding `sh -c` injection risks.

## Signal Handling

`cmd/cli-agent/src/signals.rs` registers `SIGINT` and `SIGTERM` through
`signal-hook`. The CLI checks the shared shutdown flag before confirmation,
after confirmation, and after learning transactions so Ctrl+C can stop the flow
without mutating parent shell state.

## Interactive CLI

`cli-agent ai-run <command> [args...]` renders an ANSI preview frame with
sanitized argv tokens, asks `Execute? [y/N]:`, and only then calls the direct
execution engine.

`cli-agent execute-argv '<json-array>'` is a diagnostic path for executing a
pre-built argv vector without shell translation.

## Explicit Learning

`cli-agent ai-learn --request-id <id> "<corrected command>"` tokenizes the
corrected command with the `shell-words` crate (POSIX word-splitting, not
naive whitespace splitting), finds the matching policy rule, and audits the
compiled argv against that policy before anything is written. Only after the
argv passes the preventative policy audit does it open a writable SQLite
transaction, verify the historical request exists and is still inside its
request window, and insert an `UNVERIFIED` `USER_TEMPLATE` into
`unified_documents`.

The existing FTS5 triggers mirror the learned template immediately.

### Robust Argv Tokenization

`tokenize_command_line` in `cmd/cli-agent/src/storage.rs` delegates to
`shell_words::split`. Mismatched quotes or an unparseable corrected command
return a typed error before any database connection or transaction is opened;
`request_records.execution_status` is left untouched in that case because
nothing was ever attempted against the database.

### Preventative Policy Checks On Ingestion

Before the learned template is ever inserted, `learn_from_request` loads the
matched `PolicyRule` and calls `policy::audit_argv_policy`, an argv-only
variant of the trusted policy auditor with no live-environment checks (there
is no execution environment yet at learning time, only a candidate argv to
vet). If the corrected command contains a blocked flag, a privilege-escalation
token (`sudo`, `su`, `doas`, `--privileged`, ...), a blocked network argument,
or a destructive-recursive argument under a policy that blocks that risk
class, the historical `request_records.execution_status` is transitioned to
`SECURITY_BLOCKED` and the function returns an error without ever opening the
insert transaction.

### Cache Synchronization: A Dual-Control Pattern

Keeping the sidecar's in-memory FAISS cache in step with `unified_documents`
uses two complementary controls, not one. There is no longer any reference
anywhere in this codebase to a passive `schema_metadata.vector_cache_refresh_required`
flag that the sidecar had to poll on its own schedule; that approach left a
real desynchronization window (the sidecar could serve stale candidates for an
arbitrary stretch of time between writes and its next unrelated rebuild) and
has been removed outright.

**Control 1 — Active UDS push (low latency, best-effort).** After a learned
template commits successfully, the Rust client immediately connects to the
persistent Python sidecar's Unix Domain Socket (`<runtime_dir>/sidecar.sock`)
and sends a `command: "invalidate_cache"` envelope carrying the `request_id`.
`cmd/cli-agent/src/sidecar_signal.rs` implements the client side
(`notify_cache_invalidation`), gated behind `#[cfg(unix)]` since UDS does not
exist on non-Unix targets. It is intentionally best-effort: the learned
template is already committed to SQLite (the source of truth) by the time the
signal is sent, so a daemon that is offline, mid-restart, or slow to respond
only produces a warning on stderr, never a failed `ai-learn` invocation.

On the Python side, `src/sidecar/daemon.py`'s request handler inspects the
incoming JSON line for a `"command": "invalidate_cache"` field before routing
to `SearchService.handle_invalidate_json` (in `src/sidecar/search.py`) instead
of the normal search path. That handler synchronously calls `rebuild_cache()`
on the worker thread, so the daemon has already cleared and rebuilt its
in-memory FAISS cache from SQLite before it accepts the next query, and
returns a `CacheInvalidationAck` with the new document count and rebuild
duration.

**Control 2 — Lazy-loading SQLite state validation gate (resilient fallback).**
The active push has no delivery guarantee: the daemon might not be running yet,
the socket write can race a restart, or the process could simply be down. To
close that window, `cmd/cli-agent/src/storage.rs` now maintains a
trigger-backed `schema_metadata` row keyed `unified_documents_generation`
(seeded at `'0'` by `seed_schema_metadata`). Three triggers —
`trg_docs_generation_insert`, `trg_docs_generation_update`, and
`trg_docs_generation_delete` — increment that counter automatically on every
mutation to `unified_documents`, in the same style as the existing FTS5 sync
triggers, so it can never drift out of step with a forgotten call site.

`SearchService` (in `src/sidecar/search.py`) remembers the last generation
value it observed (`_known_generation`). Before running any incoming query,
`search()` calls `_ensure_cache_fresh()`, which performs one indexed
primary-key `SELECT` against `schema_metadata` — a single-row lookup, cheap
enough to run on every request — and compares it to `_known_generation`. A
match means the cache is already current and the request proceeds
immediately. A mismatch means a previously dropped (or merely delayed) socket
signal let the cache go stale, so the daemon synchronously rebuilds before
serving that request, double-checking the generation again under
`_generation_lock` so concurrent requests cannot trigger redundant rebuilds.
`handle_invalidate_json` updates the same `_known_generation` marker after an
explicit push-triggered rebuild, so the two controls never fight each other
into rebuilding twice for the same change. Telemetry exposes this as
`cache_freshness_check_duration_ms`, so a rebuild that happens to land on the
hot path is visible rather than silently absorbed into query latency.

Net effect: the active push keeps steady-state latency low by refreshing the
cache immediately in the common case, while the generation-counter gate
guarantees correctness even when that push never arrives.

## Shell Hook

`config/shell_hook.zsh` exports:

- `ai-run`
- `ai-learn`

Set `SEMANTIC_CLI_AGENT_BIN=/path/to/cli-agent` if `cli-agent` is not already on
`PATH`, then source the hook from `~/.zshrc`.

## Verification

Run:

```bash
CARGO_TARGET_DIR=/tmp/semantic-cli-agent-target-phase6 cargo test
```

The Phase 6 tests verify literal argv preservation, explicit learning through
FTS5 synchronization, `shell-words` rejecting mismatched quotes without
mutating request status, the preventative policy audit blocking a
privilege-risk corrected command and marking it `SECURITY_BLOCKED`, the
cache-invalidation signal round-tripping over a real Unix Domain Socket (and
failing fast when the socket is missing), `execute_interactive` mapping both
successful and non-zero child exit codes, and the `unified_documents_generation`
counter advancing exactly once per insert/update/delete (including as part of
the same `ai-learn` commit).

Syntax-check the updated sidecar modules:

```bash
python -m py_compile src/sidecar/search.py src/sidecar/daemon.py
```
