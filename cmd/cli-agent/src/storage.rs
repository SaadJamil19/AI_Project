use crate::environment::SessionContext;
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;
use uuid::Uuid;

pub const SCHEMA_VERSION: &str = "phase1.2";
pub const SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedTemplate {
    pub doc_id: String,
    pub template_argv_json: String,
    pub slot_schema_json: String,
    pub policy_rule_id: String,
    pub trust_state: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedTemplate {
    pub doc_id: String,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservingRequest {
    pub request_id: String,
    pub raw_user_prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedCorrection {
    pub request_id: String,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequestLearningContext {
    request_id: String,
    raw_user_prompt: String,
    canonical_cwd: String,
    canonical_git_root: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub base_dir: PathBuf,
    pub db_path: PathBuf,
}

impl StorageConfig {
    pub fn discover() -> Result<Self> {
        let base_dir = match std::env::var_os("SEMANTIC_CLI_AGENT_HOME") {
            Some(path) => PathBuf::from(path),
            None => default_base_dir()?,
        };
        Ok(Self::from_base_dir(base_dir))
    }

    pub fn from_base_dir(base_dir: impl Into<PathBuf>) -> Self {
        let base_dir = base_dir.into();
        let db_path = base_dir.join("cli-agent.db");
        Self { base_dir, db_path }
    }

    pub fn prepare_filesystem(&self) -> Result<()> {
        ensure_private_dir(&self.base_dir)?;
        ensure_private_db_file(&self.db_path)?;
        Ok(())
    }

    pub fn open(&self) -> Result<Connection> {
        self.prepare_filesystem()?;
        let conn = Connection::open_with_flags(
            &self.db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("failed to open SQLite database at {}", self.db_path.display()))?;
        configure_connection(&conn)?;
        Ok(conn)
    }

    pub fn open_read_only(&self) -> Result<Connection> {
        let conn = Connection::open_with_flags(
            &self.db_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| {
            format!(
                "failed to open read-only SQLite database at {}",
                self.db_path.display()
            )
        })?;
        configure_read_only_connection(&conn)?;
        Ok(conn)
    }
}

pub fn initialize_database(conn: &Connection) -> Result<()> {
    configure_connection(conn)?;
    conn.execute_batch(SCHEMA_SQL)
        .context("failed to create semantic-cli-agent schema")?;
    seed_schema_metadata(conn)?;
    Ok(())
}

pub fn configure_connection(conn: &Connection) -> Result<()> {
    apply_common_pragmas(conn, true)
}

fn configure_read_only_connection(conn: &Connection) -> Result<()> {
    apply_common_pragmas(conn, false)
}

fn apply_common_pragmas(conn: &Connection, include_wal: bool) -> Result<()> {
    conn.busy_timeout(Duration::from_millis(SQLITE_BUSY_TIMEOUT_MS))
        .context("failed to install rusqlite busy timeout handler")?;

    conn.execute_batch("PRAGMA busy_timeout = 5000;")
        .context("failed to apply SQLite busy_timeout PRAGMA")?;

    if include_wal {
        conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            "#,
        )
        .context("failed to apply writable SQLite PRAGMAs")?;
    } else {
        conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = NORMAL;
            "#,
        )
        .context("failed to apply read-only SQLite PRAGMAs")?;
    }

    let foreign_keys: i64 = conn.query_row("PRAGMA foreign_keys;", [], |row| row.get(0))?;
    if foreign_keys != 1 {
        bail!("SQLite foreign key enforcement did not activate");
    }

    let busy_timeout: i64 = conn.query_row("PRAGMA busy_timeout;", [], |row| row.get(0))?;
    if busy_timeout != SQLITE_BUSY_TIMEOUT_MS as i64 {
        bail!(
            "SQLite busy_timeout is {}ms; expected {}ms",
            busy_timeout,
            SQLITE_BUSY_TIMEOUT_MS
        );
    }

    Ok(())
}

pub fn insert_session_record(conn: &Connection, context: &SessionContext) -> Result<String> {
    let session_id = Uuid::new_v4().to_string();
    conn.execute(
        r#"
        INSERT INTO session_records (
            session_id, user_id, hostname, tty_device, shell_pid, parent_pid,
            canonical_cwd, canonical_git_root, expires_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now', '+8 hours'))
        "#,
        params![
            session_id,
            context.user_id,
            context.hostname,
            context.tty_device,
            context.pid,
            context.ppid,
            context.canonical_cwd.to_string_lossy(),
            context
                .canonical_git_root
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
        ],
    )
    .context("failed to insert session record")?;
    Ok(session_id)
}

/// Reuses an existing, still-valid `session_records` row for this terminal
/// (matched by `tty_device` + `user_id` + `hostname`) instead of always
/// minting a fresh one, since `cli-agent` is a short-lived process that
/// re-runs from scratch on every invocation. Without this, "the current
/// terminal session" would never actually mean anything: two `ai-run`
/// invocations a few seconds apart would otherwise get two unrelated
/// `session_id`s, and nothing that correlates by session (like the
/// passive-observation window below) could ever find a match. Falls back
/// to `insert_session_record` when no matching, unexpired row exists.
pub fn find_or_create_session_record(conn: &Connection, context: &SessionContext) -> Result<String> {
    let existing: Option<String> = conn
        .query_row(
            r#"
            SELECT session_id
            FROM session_records
            WHERE tty_device = ?1
              AND user_id = ?2
              AND hostname = ?3
              AND expires_at > datetime('now')
            ORDER BY created_at DESC
            LIMIT 1
            "#,
            params![context.tty_device, context.user_id, context.hostname],
            |row| row.get(0),
        )
        .optional()
        .context("failed to look up an existing session record for this terminal")?;

    match existing {
        Some(session_id) => Ok(session_id),
        None => insert_session_record(conn, context),
    }
}

/// Opens the request-tracking row a natural-language prompt is judged
/// against for the rest of its lifecycle (slot validation, path validation,
/// policy audit, and eventual approval or `SECURITY_BLOCKED` outcome).
pub fn insert_request_record(
    conn: &Connection,
    session_id: &str,
    raw_user_prompt: &str,
) -> Result<String> {
    let request_id = Uuid::new_v4().to_string();
    conn.execute(
        r#"
        INSERT INTO request_records (
            request_id, session_id, raw_user_prompt, execution_status, expires_at
        )
        VALUES (?1, ?2, ?3, 'PENDING', datetime('now', '+15 minutes'))
        "#,
        params![request_id, session_id, raw_user_prompt],
    )
    .context("failed to insert request record")?;
    Ok(request_id)
}

/// Marks a request as `APPROVED` after its compiled argv has actually been
/// executed. Mirrors `mark_request_security_blocked`'s transaction shape so
/// every terminal state of a request goes through the same kind of guarded,
/// auditable transition.
pub fn mark_request_approved(conn: &mut Connection, request_id: &str) -> Result<()> {
    if request_id.trim().is_empty() {
        bail!("request_id must not be empty");
    }

    let tx = conn
        .transaction()
        .context("failed to start APPROVED request status transaction")?;
    let changed = tx
        .execute(
            r#"
            UPDATE request_records
            SET execution_status = 'APPROVED'
            WHERE request_id = ?1
              AND execution_status IN ('PENDING', 'REJECTED')
            "#,
            params![request_id],
        )
        .with_context(|| format!("failed to transition request_id={} to APPROVED", request_id))?;

    if changed != 1 {
        bail!(
            "request_id={} was not transitioned to APPROVED; request missing or already terminal",
            request_id
        );
    }

    tx.commit()
        .context("failed to commit APPROVED request status transaction")?;
    Ok(())
}

/// Marks a request `OBSERVING` instead of terminating it with a blind
/// failure when a natural-language prompt matched no template at all. This
/// is not a security failure — nothing unsafe was attempted, the system
/// simply doesn't know what was meant yet — so it does not use
/// `SECURITY_BLOCKED`. The expiry window is deliberately extended to 30
/// days (`PENDING`/fresh requests only get 15 minutes): the passive
/// observation loop this enables is explicitly meant to outlive a single
/// terminal session, and `learn_from_request` refuses to act on an expired
/// request_id.
pub fn mark_request_observing(conn: &mut Connection, request_id: &str) -> Result<()> {
    if request_id.trim().is_empty() {
        bail!("request_id must not be empty");
    }

    let tx = conn
        .transaction()
        .context("failed to start OBSERVING request status transaction")?;
    let changed = tx
        .execute(
            r#"
            UPDATE request_records
            SET execution_status = 'OBSERVING',
                expires_at = datetime('now', '+30 days')
            WHERE request_id = ?1
              AND execution_status = 'PENDING'
            "#,
            params![request_id],
        )
        .with_context(|| format!("failed to transition request_id={} to OBSERVING", request_id))?;

    if changed != 1 {
        bail!(
            "request_id={} was not transitioned to OBSERVING; request missing or already terminal",
            request_id
        );
    }

    tx.commit()
        .context("failed to commit OBSERVING request status transaction")?;
    Ok(())
}

/// Finds the most recent `OBSERVING` request for this terminal's session,
/// created within `window_minutes`. Used right after a literal `ai-run`
/// command actually executes, to decide whether that command is plausibly
/// "the manual correction" for a natural-language prompt that just missed.
pub fn find_recent_observing_request(
    conn: &Connection,
    session_id: &str,
    window_minutes: i64,
) -> Result<Option<ObservingRequest>> {
    if window_minutes <= 0 {
        bail!("window_minutes must be positive");
    }
    let window_modifier = format!("-{} minutes", window_minutes);

    conn.query_row(
        r#"
        SELECT request_id, raw_user_prompt
        FROM request_records
        WHERE session_id = ?1
          AND execution_status = 'OBSERVING'
          AND created_at > datetime('now', ?2)
        ORDER BY created_at DESC
        LIMIT 1
        "#,
        params![session_id, window_modifier],
        |row| {
            Ok(ObservingRequest {
                request_id: row.get(0)?,
                raw_user_prompt: row.get(1)?,
            })
        },
    )
    .optional()
    .context("failed to query for an active OBSERVING request")
}

/// Records `argv` as the candidate correction for `request_id`, but only
/// the first time: once a correction is recorded it is never overwritten,
/// so the *first* manual command after a miss is what gets remembered, not
/// whatever the user happened to type last before the window closed.
/// Returns whether this call actually recorded something new.
pub fn record_observed_correction(conn: &Connection, request_id: &str, argv: &[String]) -> Result<bool> {
    let argv_json = serde_json::to_string(argv).context("failed to encode observed correction argv JSON")?;
    let changed = conn
        .execute(
            r#"
            UPDATE request_records
            SET observed_correction_argv_json = ?1
            WHERE request_id = ?2
              AND execution_status = 'OBSERVING'
              AND observed_correction_argv_json IS NULL
            "#,
            params![argv_json, request_id],
        )
        .context("failed to record observed correction")?;
    Ok(changed == 1)
}

/// Finds a previously observed correction for the exact same prompt text,
/// regardless of session or how long ago it was observed — the point of
/// this lookup is specifically to answer "have I seen this exact prompt
/// before, with a known manual correction" across sessions, not just
/// within one terminal's short observation window.
pub fn find_observed_correction_for_prompt(
    conn: &Connection,
    raw_prompt: &str,
) -> Result<Option<ObservedCorrection>> {
    let row: Option<(String, String)> = conn
        .query_row(
            r#"
            SELECT request_id, observed_correction_argv_json
            FROM request_records
            WHERE raw_user_prompt = ?1
              AND execution_status = 'OBSERVING'
              AND observed_correction_argv_json IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
            "#,
            params![raw_prompt],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .context("failed to query for an observed correction")?;

    let Some((request_id, argv_json)) = row else {
        return Ok(None);
    };
    let argv: Vec<String> =
        serde_json::from_str(&argv_json).context("failed to decode observed correction argv JSON")?;
    Ok(Some(ObservedCorrection { request_id, argv }))
}

pub fn fetch_trusted_template_by_doc_id(
    conn: &Connection,
    candidate_template_id: &str,
) -> Result<TrustedTemplate> {
    if candidate_template_id.trim().is_empty() {
        bail!("candidate_template_id must not be empty");
    }

    conn.query_row(
        r#"
        SELECT
            doc_id,
            template_argv_json,
            slot_schema_json,
            policy_rule_id,
            trust_state
        FROM unified_documents
        WHERE doc_id = ?1
        LIMIT 1
        "#,
        params![candidate_template_id],
        |row| {
            Ok(TrustedTemplate {
                doc_id: row.get(0)?,
                template_argv_json: row.get(1)?,
                slot_schema_json: row.get(2)?,
                policy_rule_id: row.get(3)?,
                trust_state: row.get(4)?,
            })
        },
    )
    .with_context(|| {
        format!(
            "failed to fetch trusted template and slot schema for candidate_template_id={}",
            candidate_template_id
        )
    })
}

pub fn mark_request_security_blocked(conn: &mut Connection, request_id: &str) -> Result<()> {
    if request_id.trim().is_empty() {
        bail!("request_id must not be empty");
    }

    let tx = conn
        .transaction()
        .context("failed to start SECURITY_BLOCKED request status transaction")?;
    let changed = tx
        .execute(
            r#"
            UPDATE request_records
            SET execution_status = 'SECURITY_BLOCKED'
            WHERE request_id = ?1
              AND execution_status IN ('PENDING', 'APPROVED', 'REJECTED')
            "#,
            params![request_id],
        )
        .with_context(|| {
            format!(
                "failed to transition request_id={} to SECURITY_BLOCKED",
                request_id
            )
        })?;

    if changed != 1 {
        bail!(
            "request_id={} was not transitioned to SECURITY_BLOCKED; request missing or already terminal",
            request_id
        );
    }

    tx.commit()
        .context("failed to commit SECURITY_BLOCKED request status transaction")?;
    Ok(())
}

pub fn learn_from_request(
    conn: &mut Connection,
    request_id: &str,
    corrected_command: &str,
) -> Result<LearnedTemplate> {
    if request_id.trim().is_empty() {
        bail!("request_id must not be empty");
    }

    let argv = tokenize_command_line(corrected_command)
        .with_context(|| format!("failed to tokenize corrected command for request_id={}", request_id))?;
    if argv.is_empty() {
        bail!("corrected command must not be empty");
    }

    let binary_name = argv[0].clone();
    let subcommand_path = argv.get(1).cloned().unwrap_or_default();
    let policy_rule_id = find_policy_rule_id(conn, &binary_name, &subcommand_path)
        .with_context(|| format!("no policy rule matches learned command {:?}", argv))?;

    let policy = crate::policy::load_policy_rule(conn, &policy_rule_id).with_context(|| {
        format!(
            "failed to load policy rule {} while auditing learned command",
            policy_rule_id
        )
    })?;
    if let Err(policy_err) = crate::policy::audit_argv_policy(&policy, &argv) {
        mark_request_security_blocked(conn, request_id).with_context(|| {
            format!(
                "corrected command failed policy audit with {}; failed to mark request_id={} SECURITY_BLOCKED",
                policy_err, request_id
            )
        })?;
        bail!(
            "corrected command argv {:?} violates policy {}: {}",
            argv,
            policy_rule_id,
            policy_err
        );
    }

    let context = fetch_learning_context(conn, request_id)?;
    let doc_id = format!("learned_{}", Uuid::new_v4());
    let argv_json = serde_json::to_string(&argv).context("failed to encode learned argv JSON")?;
    let slot_schema_json = "[]";
    let scope_anchor = context
        .canonical_git_root
        .clone()
        .unwrap_or_else(|| context.canonical_cwd.clone());
    let project_root_hash = format!("learned:{}", scope_anchor);

    let tx = conn
        .transaction()
        .context("failed to start ai-learn template transaction")?;
    tx.execute(
        r#"
        INSERT INTO unified_documents (
            doc_id,
            source_type,
            binary_name,
            subcommand_path,
            intent_description,
            template_argv_json,
            slot_schema_json,
            policy_rule_id,
            project_root_hash,
            scope_mode,
            trust_state,
            created_from_request_id
        )
        VALUES (?1, 'USER_TEMPLATE', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'PROJECT', 'UNVERIFIED', ?9)
        "#,
        params![
            doc_id,
            binary_name,
            subcommand_path,
            context.raw_user_prompt,
            argv_json,
            slot_schema_json,
            policy_rule_id,
            project_root_hash,
            context.request_id,
        ],
    )
    .context("failed to insert learned template")?;
    tx.execute(
        "UPDATE request_records SET execution_status = 'APPROVED' WHERE request_id = ?1",
        params![request_id],
    )
    .context("failed to mark learned request as approved")?;
    tx.commit()
        .context("failed to commit ai-learn template transaction")?;

    Ok(LearnedTemplate { doc_id, argv })
}

pub fn tokenize_command_line(input: &str) -> Result<Vec<String>> {
    shell_words::split(input)
        .map_err(|err| anyhow!("failed to parse corrected command as POSIX shell words: {}", err))
}

fn fetch_learning_context(conn: &Connection, request_id: &str) -> Result<RequestLearningContext> {
    conn.query_row(
        r#"
        SELECT
            r.request_id,
            r.raw_user_prompt,
            s.canonical_cwd,
            s.canonical_git_root
        FROM request_records AS r
        JOIN session_records AS s ON s.session_id = r.session_id
        WHERE r.request_id = ?1
          AND r.expires_at > datetime('now')
        LIMIT 1
        "#,
        params![request_id],
        |row| {
            Ok(RequestLearningContext {
                request_id: row.get(0)?,
                raw_user_prompt: row.get(1)?,
                canonical_cwd: row.get(2)?,
                canonical_git_root: row.get(3)?,
            })
        },
    )
    .with_context(|| format!("request_id={} is missing or expired", request_id))
}

/// Looks up the policy row that should govern a learned command.
///
/// Tries an exact `(binary_name, subcommand_path)` match first, then falls
/// back to a binary-wide policy (`subcommand_path = ''`). The fallback
/// matters because `audit_argv` already treats an empty policy
/// `subcommand_path` as "no subcommand constraint" — one shared row can
/// cover many subcommands through its flag allowlist instead of needing a
/// dedicated row per subcommand. Without this fallback, learning anything
/// beyond the exact subcommand a binary-wide policy happened to be seeded
/// against would fail outright — and for a flag-only binary like `curl`,
/// where `argv[1]` is always a flag rather than a real subcommand, the
/// exact match can never succeed at all.
fn find_policy_rule_id(conn: &Connection, binary_name: &str, subcommand_path: &str) -> Result<String> {
    let exact: Option<String> = conn
        .query_row(
            r#"
            SELECT rule_id
            FROM policy_rules
            WHERE binary_name = ?1
              AND subcommand_path = ?2
            LIMIT 1
            "#,
            params![binary_name, subcommand_path],
            |row| row.get(0),
        )
        .optional()
        .with_context(|| {
            format!(
                "failed to query policy rule for binary={} subcommand={}",
                binary_name, subcommand_path
            )
        })?;
    if let Some(rule_id) = exact {
        return Ok(rule_id);
    }

    conn.query_row(
        r#"
        SELECT rule_id
        FROM policy_rules
        WHERE binary_name = ?1
          AND subcommand_path = ''
        LIMIT 1
        "#,
        params![binary_name],
        |row| row.get(0),
    )
    .with_context(|| {
        format!(
            "policy rule not found for binary={} subcommand={}",
            binary_name, subcommand_path
        )
    })
}

fn seed_schema_metadata(conn: &Connection) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO schema_metadata(key, value) VALUES('schema_version', ?1)",
        params![SCHEMA_VERSION],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO schema_metadata(key, value) VALUES('unified_documents_generation', '0')",
        [],
    )?;
    Ok(())
}

pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_rules (
    rule_id TEXT PRIMARY KEY NOT NULL,
    binary_name TEXT NOT NULL,
    subcommand_path TEXT DEFAULT '' NOT NULL,
    fast_path_allowed INTEGER NOT NULL CHECK (fast_path_allowed IN (0, 1)),
    required_confirmation_count INTEGER NOT NULL DEFAULT 1 CHECK (required_confirmation_count >= 1),
    executable_path_policy_json TEXT NOT NULL,
    env_variable_inheritance_json TEXT NOT NULL,
    positional_argument_rules_json TEXT NOT NULL,
    path_slot_policies_json TEXT NOT NULL,
    package_manager_risk_level TEXT NOT NULL CHECK (package_manager_risk_level IN ('ALLOW', 'BLOCK')),
    privilege_risk_level TEXT NOT NULL CHECK (privilege_risk_level IN ('ALLOW', 'BLOCK')),
    destructive_recursive_level TEXT NOT NULL CHECK (destructive_recursive_level IN ('ALLOW', 'BLOCK')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_policy_rules_match_key ON policy_rules(binary_name, subcommand_path);

CREATE TABLE IF NOT EXISTS session_records (
    session_id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL,
    hostname TEXT NOT NULL,
    tty_device TEXT NOT NULL,
    shell_pid INTEGER NOT NULL,
    parent_pid INTEGER NOT NULL,
    canonical_cwd TEXT NOT NULL,
    canonical_git_root TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS request_records (
    request_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    raw_user_prompt TEXT NOT NULL,
    execution_status TEXT NOT NULL CHECK (execution_status IN ('PENDING', 'APPROVED', 'REJECTED', 'SECURITY_BLOCKED', 'OBSERVING')),
    observed_correction_argv_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    FOREIGN KEY(session_id) REFERENCES session_records(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_request_session_fk ON request_records(session_id);
CREATE INDEX IF NOT EXISTS idx_request_observing_lookup ON request_records(session_id, execution_status, created_at);

CREATE TABLE IF NOT EXISTS unified_documents (
    doc_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('STATIC_DOCS', 'USER_TEMPLATE')),
    binary_name TEXT NOT NULL,
    subcommand_path TEXT DEFAULT '' NOT NULL,
    intent_description TEXT NOT NULL,
    template_argv_json TEXT NOT NULL,
    slot_schema_json TEXT NOT NULL,
    policy_rule_id TEXT NOT NULL,
    project_root_hash TEXT,
    scope_mode TEXT CHECK (scope_mode IN ('GLOBAL', 'PROJECT', 'GIT_ROOT')),
    trust_state TEXT NOT NULL CHECK (trust_state IN ('STATIC_VERIFIED','UNVERIFIED','USER_CONFIRMED','PROMOTED_FASTPATH','DISABLED','REVOKED')),
    created_from_request_id TEXT,
    success_count INTEGER DEFAULT 0 NOT NULL CHECK (success_count >= 0),
    failure_count INTEGER DEFAULT 0 NOT NULL CHECK (failure_count >= 0),
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    FOREIGN KEY(policy_rule_id) REFERENCES policy_rules(rule_id),
    FOREIGN KEY(created_from_request_id) REFERENCES request_records(request_id),
    CHECK (
        (source_type = 'STATIC_DOCS' AND trust_state = 'STATIC_VERIFIED' AND project_root_hash IS NULL AND scope_mode IS NULL AND created_from_request_id IS NULL)
        OR
        (source_type = 'USER_TEMPLATE' AND trust_state IN ('UNVERIFIED','USER_CONFIRMED','PROMOTED_FASTPATH','DISABLED','REVOKED') AND project_root_hash IS NOT NULL AND scope_mode IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_docs_routing ON unified_documents(source_type, trust_state, binary_name);
CREATE INDEX IF NOT EXISTS idx_docs_project_scope ON unified_documents(project_root_hash) WHERE project_root_hash IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS docs_external_fts USING fts5(
    binary_name,
    subcommand_path,
    intent_description,
    content='unified_documents',
    content_rowid='doc_rowid'
);

CREATE TRIGGER IF NOT EXISTS trg_docs_fts_insert AFTER INSERT ON unified_documents BEGIN
    INSERT INTO docs_external_fts(rowid, binary_name, subcommand_path, intent_description)
    VALUES (new.doc_rowid, new.binary_name, new.subcommand_path, new.intent_description);
END;

CREATE TRIGGER IF NOT EXISTS trg_docs_fts_delete AFTER DELETE ON unified_documents BEGIN
    INSERT INTO docs_external_fts(docs_external_fts, rowid, binary_name, subcommand_path, intent_description)
    VALUES ('delete', old.doc_rowid, old.binary_name, old.subcommand_path, old.intent_description);
END;

CREATE TRIGGER IF NOT EXISTS trg_docs_fts_update AFTER UPDATE ON unified_documents BEGIN
    INSERT INTO docs_external_fts(docs_external_fts, rowid, binary_name, subcommand_path, intent_description)
    VALUES ('delete', old.doc_rowid, old.binary_name, old.subcommand_path, old.intent_description);
    INSERT INTO docs_external_fts(rowid, binary_name, subcommand_path, intent_description)
    VALUES (new.doc_rowid, new.binary_name, new.subcommand_path, new.intent_description);
END;

CREATE TRIGGER IF NOT EXISTS trg_docs_generation_insert AFTER INSERT ON unified_documents BEGIN
    UPDATE schema_metadata
    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT), updated_at = CURRENT_TIMESTAMP
    WHERE key = 'unified_documents_generation';
END;

CREATE TRIGGER IF NOT EXISTS trg_docs_generation_update AFTER UPDATE ON unified_documents BEGIN
    UPDATE schema_metadata
    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT), updated_at = CURRENT_TIMESTAMP
    WHERE key = 'unified_documents_generation';
END;

CREATE TRIGGER IF NOT EXISTS trg_docs_generation_delete AFTER DELETE ON unified_documents BEGIN
    UPDATE schema_metadata
    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT), updated_at = CURRENT_TIMESTAMP
    WHERE key = 'unified_documents_generation';
END;
"#;

#[cfg(unix)]
fn ensure_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    if path.as_os_str().is_empty() {
        bail!("private directory path is empty");
    }

    let mut missing = Vec::new();
    let mut cursor = path;
    while !cursor.exists() {
        missing.push(cursor.to_path_buf());
        cursor = cursor
            .parent()
            .ok_or_else(|| anyhow!("failed to resolve parent path for {}", path.display()))?;
    }

    if !cursor.is_dir() {
        bail!("{} exists but is not a directory", cursor.display());
    }

    for dir in missing.iter().rev() {
        fs::DirBuilder::new()
            .mode(0o700)
            .create(dir)
            .with_context(|| format!("failed to create private directory {}", dir.display()))?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to enforce 0700 permissions on {}", dir.display()))?;
    }

    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to stat private directory {}", path.display()))?;
    if !metadata.is_dir() {
        bail!("{} exists but is not a directory", path.display());
    }

    let mode = metadata.permissions().mode() & 0o777;
    if mode != 0o700 {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to enforce 0700 permissions on {}", path.display()))?;
        let updated = fs::metadata(path)?.permissions().mode() & 0o777;
        if updated != 0o700 {
            bail!(
                "private directory {} has mode {:o}; expected 700",
                path.display(),
                updated
            );
        }
    }

    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create private directory {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_db_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if let Some(parent) = path.parent() {
        ensure_private_dir(parent)?;
    }

    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to create database file {}", path.display()))?;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("failed to set 0600 permissions on {}", path.display()))?;

    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to stat database file {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{} exists but is not a regular file", path.display());
    }

    let mode = metadata.permissions().mode() & 0o777;
    if mode != 0o600 {
        bail!(
            "database file {} has mode {:o}; expected 600",
            path.display(),
            mode
        );
    }

    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_db_file(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        ensure_private_dir(parent)?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to create database file {}", path.display()))?;
    Ok(())
}

fn default_base_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(path).join("semantic-cli-agent"));
    }

    if let Some(path) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(path)
            .join(".local")
            .join("share")
            .join("semantic-cli-agent"));
    }

    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return Ok(PathBuf::from(path).join("semantic-cli-agent"));
    }

    Err(anyhow!(
        "could not determine data directory; set SEMANTIC_CLI_AGENT_HOME"
    ))
}
