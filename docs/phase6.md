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

`cli-agent ai-learn --request-id <id> "<corrected command>"` opens a writable
SQLite transaction, verifies the historical request exists and is still inside
its request window, tokenizes the corrected command, finds the matching policy
rule, and inserts an `UNVERIFIED` `USER_TEMPLATE` into `unified_documents`.

The existing FTS5 triggers mirror the learned template immediately. The command
also updates `schema_metadata.vector_cache_refresh_required` so the sidecar can
observe that its FAISS cache should be rebuilt.

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

The Phase 6 tests verify literal argv preservation and explicit learning through
FTS5 synchronization.
