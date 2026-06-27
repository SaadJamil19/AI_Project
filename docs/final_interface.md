# The Natural Language Ingestor Client Loop

This layer connects plain language input (`ai-run "undo changes in src/main.rs"`) to the secure command pipeline built in Phases 1-6.

## Runtime Flow

1. `ai-run` receives arguments in `cmd/cli-agent/src/main.rs`.
   - If `argv[0]` is an executable or a hard allowlisted shell utility, Rust treats the input as a literal command.
   - Otherwise Rust joins the words into a `NATURAL_LANGUAGE_PROMPT`.

2. Rust records the transaction.
   - `insert_session_record` stores the terminal context.
   - `insert_request_record` creates a `request_records` row with `execution_status = 'PENDING'`.
   - `ipc::query_sidecar` sends `{ protocol_version, request_id, query, limit }` as one JSON line over the secured Unix Domain Socket.

3. Python sidecar searches for a command template.
   - `src/sidecar/search.py` runs FTS5 lexical lookup and FAISS vector lookup.
   - It merges both result sets with RRF.
   - If the fast-path gateway succeeds, it returns a deterministic proposal.
   - If retrieval is low-confidence, returns no template, or finds a slotted template that needs parameter extraction, it first tries deterministic local slot extraction for simple path/token slots, then calls `src/sidecar/llm_client.py` as fallback and wraps the grammar-constrained Ollama result in the standard `untrusted_proposal_json` envelope.

4. Rust distrusts the sidecar response.
   - Rust reloads the named template from SQLite using `candidate_template_id`.
   - Rust validates `raw_untrusted_slots` against the trusted `slot_schema_json`.
   - Rust compiles `template_argv_json` plus bound slots into a literal argv array.
   - Rust validates path slots against the workspace boundary.
   - Rust audits the final argv and inherited environment against `policy_rules`.

5. Rust previews and executes.
   - Preview text is sanitized before display.
   - The user must confirm `Execute? [y/N]:`.
   - Execution uses `std::process::Command` with inherited stdin/stdout/stderr, never `sh -c`.
   - Success marks the request `APPROVED`; security failures mark it `SECURITY_BLOCKED`.

## Seed Data

A fresh database has no command templates. Load the starter git templates after running `cli-agent init`:

```bash
sqlite3 "$SEMANTIC_CLI_AGENT_HOME/cli-agent.db" < config/seed_data.sql
```

The seed file inserts:

- `policy_git_core` for `git` with `/usr/bin/git` and `git` allowed.
- `seed_git_status`: `git status`.
- `seed_git_restore`: `git restore $target_file`, where `target_file` must be a `relative_path`.
- `seed_git_checkout_branch`: `git checkout -b $branch_name`, where `branch_name` must be a `safe_token`.
- `seed_git_stash_push`: `git stash push`.

The script is idempotent, so loading it twice does not duplicate rows.

## Verification

Run the Rust suite with an external target directory so the repo does not collect target artifacts:

```bash
CARGO_TARGET_DIR=/tmp/semantic-cli-agent-target-nl cargo test --workspace
```

Check the Python sidecar files for syntax errors:

```bash
python -m py_compile src/sidecar/search.py src/sidecar/daemon.py src/sidecar/cache.py src/sidecar/gateway.py src/sidecar/llm_client.py
```

The natural-language integration tests use a fake Unix socket, so they verify Rust's request lifecycle, untrusted proposal parsing, trusted template reload, slot binding, path validation, and policy audit without requiring a live Ollama process.
