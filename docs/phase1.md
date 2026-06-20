# Phase 1: Relational Storage Engine, WAL, and FTS5 Triggers

## What Was Built

Phase 1 creates the trusted local storage foundation for semantic-cli-agent.

The Rust crate under cmd/cli-agent now provides:

- A short-lived CLI binary entrypoint in cmd/cli-agent/src/main.rs.
- OS and session capture utilities in cmd/cli-agent/src/environment.rs.
- SQLite bootstrap, permission enforcement, PRAGMAs, schema creation, and FTS5 triggers in cmd/cli-agent/src/storage.rs.
- Integration tests in tests/integration_tests.rs.

The binary exposes three Phase 1 commands:

- cargo run -p cli-agent -- init
- cargo run -p cli-agent -- capture-session
- cargo run -p cli-agent -- record-session

init prepares the private runtime folder, creates the SQLite database, applies required PRAGMAs, and compiles the schema.

capture-session prints current process and terminal metadata as JSON.

record-session initializes the database if needed, captures the terminal session, and persists it into session_records.

## Low-Level System Boundary

The trusted Rust client is responsible for the first security boundary. It reads local process and terminal state before any Python or model component is involved.

Captured values include PID, parent PID, TTY descriptor where available, canonical current working directory, canonical Git repository root if available, local username, and hostname.

This keeps session state tied to the process that owns the terminal. The Python daemon remains an untrusted sidecar and does not get authority to mutate policy or execute commands.

## Storage Layout

The SQLite database contains these Phase 1 tables:

- policy_rules: trusted execution policy rows used later by the Rust policy engine.
- session_records: active terminal and session metadata.
- request_records: per-request transaction records linked to sessions with cascade delete.
- unified_documents: the single source of truth for static command documentation and learned user templates.
- schema_metadata: schema version marker.
- docs_external_fts: FTS5 virtual index backed by unified_documents.

unified_documents uses doc_rowid as an integer primary key. This is required because SQLite FTS5 external-content tables map to an integer rowid through content_rowid.

## Why SQLite WAL

Every connection enables foreign_keys, busy_timeout, journal_mode WAL, and synchronous NORMAL.

foreign_keys is required because SQLite does not enforce foreign keys unless each connection enables it.

WAL allows readers and a writer to coexist better than rollback journal mode. That matters for a terminal tool because the Python sidecar may read retrieval data while the Rust client inserts sessions, requests, or future learned templates.

busy_timeout is explicitly set to 5000ms on every connection. SQLite otherwise defaults to no wait window, so concurrent short-lived terminal processes can fail immediately with database is locked. The 5000ms timeout gives active writers a bounded period to clear before the CLI returns an error.

synchronous NORMAL is the practical WAL-mode performance setting for a desktop-local application.

## Why A Short-Lived Rust CLI

The Rust client starts fresh for each ai-run or ai-learn style invocation. That keeps trusted state simple:

- It reads current policies directly from storage.
- It avoids stale long-lived policy caches in Phase 1.
- It owns terminal IO and later command execution.
- It avoids Python interpreter startup on trusted execution paths.

The Python daemon can stay hot for ML work, but it does not become the trusted execution process.

## Why External-Content FTS5

docs_external_fts is an external-content FTS5 table. The text index does not become the source of truth; it mirrors unified_documents.

Three triggers keep the index synchronized:

- trg_docs_fts_insert
- trg_docs_fts_delete
- trg_docs_fts_update

When unified_documents changes, the FTS index receives the corresponding operation immediately. This avoids search drift where stale command documentation remains searchable after being changed or removed.

## Permission Model

On Unix targets, the storage layer enforces runtime directory mode 0700 and SQLite database file mode 0600.

When the runtime directory does not exist, the storage layer creates every missing parent directory one level at a time with mode 0700 at creation. This avoids a weaker create-then-tighten window where newly generated parent directories could briefly inherit broader process umask permissions.

This protects against other local OS users. It is not a same-user malware boundary; same-user compromise remains out of scope for v1.

## How To Verify Locally

From the project root, run cargo test.

The integration tests verify:

- Required SQLite PRAGMAs are active.
- Runtime directory and DB file permissions are private on Unix.
- Nested missing runtime directories are created with private permissions.
- FTS5 triggers reflect insert, update, and delete mutations immediately.
- Foreign keys reject documents that reference missing policy rules.
- Foreign keys reject request_records rows that reference missing sessions.

To initialize a local runtime DB manually, run cargo run -p cli-agent -- init.

To print captured session context, run cargo run -p cli-agent -- capture-session.

To persist a session record, run cargo run -p cli-agent -- record-session.

Set SEMANTIC_CLI_AGENT_HOME to choose a custom runtime directory. If unset, the CLI uses XDG_DATA_HOME/semantic-cli-agent or HOME/.local/share/semantic-cli-agent on Unix-like systems.
