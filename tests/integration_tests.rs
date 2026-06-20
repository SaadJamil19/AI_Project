use anyhow::Result;
use cli_agent::storage::{initialize_database, StorageConfig, SQLITE_BUSY_TIMEOUT_MS};
use rusqlite::{params, Connection, Error as SqliteError};
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

fn test_base_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("semantic-cli-agent-{}-{}", name, Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("failed to create temp test dir");
    dir
}

fn open_test_db(name: &str) -> Result<(StorageConfig, Connection)> {
    let config = StorageConfig::from_base_dir(test_base_dir(name));
    let conn = config.open()?;
    initialize_database(&conn)?;
    seed_policy(&conn)?;
    Ok((config, conn))
}

fn seed_policy(conn: &Connection) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO policy_rules (
            rule_id,
            binary_name,
            subcommand_path,
            fast_path_allowed,
            required_confirmation_count,
            executable_path_policy_json,
            env_variable_inheritance_json,
            positional_argument_rules_json,
            path_slot_policies_json,
            package_manager_risk_level,
            privilege_risk_level,
            destructive_recursive_level
        )
        VALUES (?1, 'git', 'restore', 1, 1, '[]', '[]', '{}', '{}', 'BLOCK', 'BLOCK', 'BLOCK')
        "#,
        params!["policy_git_restore"],
    )?;
    Ok(())
}

fn insert_static_doc(conn: &Connection, doc_id: &str, description: &str) -> Result<i64> {
    conn.execute(
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
            trust_state
        )
        VALUES (?1, 'STATIC_DOCS', 'git', 'restore', ?2, '["git","restore","$target_file"]', '{}', 'policy_git_restore', 'STATIC_VERIFIED')
        "#,
        params![doc_id, description],
    )?;
    Ok(conn.last_insert_rowid())
}

fn fts_count(conn: &Connection, query: &str) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT count(*) FROM docs_external_fts WHERE docs_external_fts MATCH ?1",
        params![query],
        |row| row.get(0),
    )?)
}

#[test]
fn database_bootstrap_applies_required_pragmas() -> Result<()> {
    let (_config, conn) = open_test_db("pragmas")?;

    let foreign_keys: i64 = conn.query_row("PRAGMA foreign_keys;", [], |row| row.get(0))?;
    let journal_mode: String = conn.query_row("PRAGMA journal_mode;", [], |row| row.get(0))?;
    let synchronous: i64 = conn.query_row("PRAGMA synchronous;", [], |row| row.get(0))?;
    let busy_timeout: i64 = conn.query_row("PRAGMA busy_timeout;", [], |row| row.get(0))?;

    assert_eq!(foreign_keys, 1);
    assert_eq!(journal_mode.to_lowercase(), "wal");
    assert_eq!(synchronous, 1);
    assert_eq!(busy_timeout, SQLITE_BUSY_TIMEOUT_MS as i64);
    Ok(())
}

#[cfg(unix)]
#[test]
fn storage_filesystem_permissions_are_private() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let (config, _conn) = open_test_db("permissions")?;
    let dir_mode = fs::metadata(&config.base_dir)?.permissions().mode() & 0o777;
    let db_mode = fs::metadata(&config.db_path)?.permissions().mode() & 0o777;

    assert_eq!(dir_mode, 0o700);
    assert_eq!(db_mode, 0o600);
    Ok(())
}

#[cfg(unix)]
#[test]
fn nested_runtime_directories_are_created_private() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let root = test_base_dir("nested-private");
    let nested = root.join("missing").join("parent").join("runtime");
    let config = StorageConfig::from_base_dir(&nested);
    let conn = config.open()?;
    initialize_database(&conn)?;

    for dir in [root.join("missing"), root.join("missing").join("parent"), nested] {
        let mode = fs::metadata(&dir)?.permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "{} should be private", dir.display());
    }

    Ok(())
}

#[test]
fn fts_insert_update_delete_triggers_track_unified_documents() -> Result<()> {
    let (_config, conn) = open_test_db("fts")?;

    let rowid = insert_static_doc(&conn, "doc_git_restore", "restore a tracked file from git index")?;
    assert_eq!(fts_count(&conn, "restore")?, 1);

    conn.execute(
        "UPDATE unified_documents SET intent_description = 'commit staged work' WHERE doc_rowid = ?1",
        params![rowid],
    )?;
    assert_eq!(fts_count(&conn, "tracked")?, 0);
    assert_eq!(fts_count(&conn, "commit")?, 1);

    conn.execute("DELETE FROM unified_documents WHERE doc_rowid = ?1", params![rowid])?;
    assert_eq!(fts_count(&conn, "commit")?, 0);
    Ok(())
}

#[test]
fn foreign_keys_reject_documents_without_policy_rule() -> Result<()> {
    let config = StorageConfig::from_base_dir(test_base_dir("fk-doc"));
    let conn = config.open()?;
    initialize_database(&conn)?;

    let result = conn.execute(
        r#"
        INSERT INTO unified_documents (
            doc_id, source_type, binary_name, subcommand_path, intent_description,
            template_argv_json, slot_schema_json, policy_rule_id, trust_state
        )
        VALUES ('doc_missing_policy', 'STATIC_DOCS', 'git', 'restore', 'restore file', '[]', '{}', 'missing_policy', 'STATIC_VERIFIED')
        "#,
        [],
    );

    assert!(matches!(result, Err(SqliteError::SqliteFailure(_, _))));
    Ok(())
}

#[test]
fn foreign_keys_reject_requests_for_missing_session() -> Result<()> {
    let config = StorageConfig::from_base_dir(test_base_dir("fk-request"));
    let conn = config.open()?;
    initialize_database(&conn)?;

    let result = conn.execute(
        r#"
        INSERT INTO request_records (
            request_id,
            session_id,
            raw_user_prompt,
            execution_status,
            expires_at
        )
        VALUES ('req_missing_session', 'session_does_not_exist', 'undo last commit', 'PENDING', datetime('now', '+1 hour'))
        "#,
        [],
    );

    assert!(matches!(result, Err(SqliteError::SqliteFailure(_, _))));
    Ok(())
}
