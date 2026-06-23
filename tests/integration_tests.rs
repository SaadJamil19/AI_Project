use anyhow::Result;
use cli_agent::execute::execute_capture_stdout;
use cli_agent::path_validate::{
    sanitize_terminal_preview_token, validate_workspace_path, PathValidationError,
};
use cli_agent::policy::{audit_policy, audit_policy_for_request, load_policy_rule, PolicyError};
use cli_agent::storage::{
    fetch_trusted_template_by_doc_id, initialize_database, learn_from_request,
    mark_request_security_blocked, tokenize_command_line, StorageConfig, SQLITE_BUSY_TIMEOUT_MS,
};
use cli_agent::validate::{bind_and_validate_slots, bind_template_slots, SlotValidationError};
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
        VALUES (
            ?1,
            'git',
            'restore',
            1,
            1,
            '{"allowed_binaries":["git"]}',
            '{"allow":[],"block":["LD_PRELOAD"]}',
            '{"allowed_flags":["--","--source"],"blocked_flags":["--exec-path"]}',
            '{"allow_network_args":false}',
            'BLOCK',
            'BLOCK',
            'BLOCK'
        )
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
        VALUES (
            ?1,
            'STATIC_DOCS',
            'git',
            'restore',
            ?2,
            '["git","restore","$target_file"]',
            '[{"name":"target_file","kind":"string","required":true,"max_bytes":64,"allowed_formats":["relative_path"]}]',
            'policy_git_restore',
            'STATIC_VERIFIED'
        )
        "#,
        params![doc_id, description],
    )?;
    Ok(conn.last_insert_rowid())
}

fn insert_user_template_with_state(conn: &Connection, doc_id: &str, trust_state: &str) -> Result<i64> {
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
            project_root_hash,
            scope_mode,
            trust_state
        )
        VALUES (
            ?1,
            'USER_TEMPLATE',
            'git',
            'restore',
            'restore a file',
            '["git","restore","$target_file"]',
            '[{"name":"target_file","kind":"string","required":true,"max_bytes":64,"allowed_formats":["relative_path"]}]',
            'policy_git_restore',
            'project_hash',
            'PROJECT',
            ?2
        )
        "#,
        params![doc_id, trust_state],
    )?;
    Ok(conn.last_insert_rowid())
}

fn seed_session_and_request(conn: &Connection, request_id: &str) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO session_records (
            session_id,
            user_id,
            hostname,
            tty_device,
            shell_pid,
            parent_pid,
            canonical_cwd,
            expires_at
        )
        VALUES ('session_for_request', 'user', 'host', '/dev/pts/1', 10, 9, '/tmp', datetime('now', '+1 hour'))
        "#,
        [],
    )?;
    conn.execute(
        r#"
        INSERT INTO request_records (
            request_id,
            session_id,
            raw_user_prompt,
            execution_status,
            expires_at
        )
        VALUES (?1, 'session_for_request', 'git restore ../secret', 'PENDING', datetime('now', '+1 hour'))
        "#,
        params![request_id],
    )?;
    Ok(())
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


#[test]
fn trusted_template_fetch_returns_command_and_slot_schema() -> Result<()> {
    let (_config, conn) = open_test_db("trusted-template-fetch")?;
    insert_static_doc(&conn, "doc_git_restore_fetch", "restore a file")?;

    let template = fetch_trusted_template_by_doc_id(&conn, "doc_git_restore_fetch")?;

    assert_eq!(template.doc_id, "doc_git_restore_fetch");
    assert!(template.template_argv_json.contains("$target_file"));
    assert!(template.slot_schema_json.contains("target_file"));
    Ok(())
}

#[test]
fn rust_slot_binding_rejects_invalid_slot_arguments() -> Result<()> {
    let schema = r#"
    [
      {
        "name": "target_file",
        "kind": "string",
        "required": true,
        "max_bytes": 16,
        "allowed_formats": ["relative_path"]
      }
    ]
    "#;

    let path_escape = r#"{"target_file":"../secret"}"#;
    let result = bind_and_validate_slots(schema, path_escape);
    assert!(matches!(
        result,
        Err(SlotValidationError::FormatRejected { slot }) if slot == "target_file"
    ));

    let too_long = r#"{"target_file":"src/very_long_name.rs"}"#;
    let result = bind_and_validate_slots(schema, too_long);
    assert!(matches!(
        result,
        Err(SlotValidationError::SlotTooLong { slot, max_bytes: 16, .. }) if slot == "target_file"
    ));

    let valid = bind_and_validate_slots(schema, r#"{"target_file":"src/main.rs"}"#)?;
    assert_eq!(valid.get("target_file"), Some("src/main.rs"));
    Ok(())
}

#[test]
fn disabled_or_revoked_templates_fail_rust_lifecycle_validation() -> Result<()> {
    let (_config, conn) = open_test_db("template-lifecycle")?;
    insert_user_template_with_state(&conn, "doc_disabled_template", "DISABLED")?;
    insert_user_template_with_state(&conn, "doc_revoked_template", "REVOKED")?;

    for (doc_id, state) in [
        ("doc_disabled_template", "DISABLED"),
        ("doc_revoked_template", "REVOKED"),
    ] {
        let template = fetch_trusted_template_by_doc_id(&conn, doc_id)?;
        assert_eq!(template.trust_state, state);

        let result = bind_template_slots(
            &template.trust_state,
            &template.slot_schema_json,
            r#"{"target_file":"src/main.rs"}"#,
        );
        assert!(matches!(
            result,
            Err(SlotValidationError::TemplateRevoked(actual)) if actual == state
        ));
    }

    Ok(())
}

#[test]
fn integer_json_primitives_pass_slot_validation() -> Result<()> {
    let schema = r#"
    [
      {
        "name": "retry_count",
        "kind": "integer",
        "required": true,
        "min_value": 1,
        "max_value": 5,
        "allowed_formats": ["integer"]
      }
    ]
    "#;

    let bound = bind_and_validate_slots(schema, r#"{"retry_count":3}"#)?;
    assert_eq!(bound.get("retry_count"), Some("3"));

    let out_of_bounds = bind_and_validate_slots(schema, r#"{"retry_count":9}"#);
    assert!(matches!(
        out_of_bounds,
        Err(SlotValidationError::IntegerOutOfBounds { slot, value: 9 }) if slot == "retry_count"
    ));
    Ok(())
}

#[test]
fn validation_failure_can_mark_request_security_blocked() -> Result<()> {
    let (_config, mut conn) = open_test_db("request-security-blocked")?;
    seed_session_and_request(&conn, "req_blocked_by_validator")?;

    let schema = r#"
    [
      {
        "name": "target_file",
        "kind": "string",
        "required": true,
        "max_bytes": 16,
        "allowed_formats": ["relative_path"]
      }
    ]
    "#;
    let result = bind_and_validate_slots(schema, r#"{"target_file":"../secret"}"#);
    assert!(result.is_err());
    mark_request_security_blocked(&mut conn, "req_blocked_by_validator")?;

    let status: String = conn.query_row(
        "SELECT execution_status FROM request_records WHERE request_id = 'req_blocked_by_validator'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(status, "SECURITY_BLOCKED");
    Ok(())
}

#[test]
fn directory_traversal_crossing_workspace_is_blocked() -> Result<()> {
    let workspace = test_base_dir("path-boundary");
    fs::create_dir_all(workspace.join("src"))?;

    let traversal = validate_workspace_path(&workspace, "../outside.txt");
    assert!(matches!(
        traversal,
        Err(PathValidationError::ParentTraversal)
    ));

    let absolute_escape = validate_workspace_path(
        &workspace,
        workspace
            .parent()
            .expect("workspace temp dir has parent")
            .join("outside.txt")
            .to_string_lossy()
            .as_ref(),
    );
    assert!(matches!(
        absolute_escape,
        Err(PathValidationError::BoundaryEscape)
    ));

    let valid_missing = validate_workspace_path(&workspace, "src/generated/new_file.rs")?;
    assert!(valid_missing.resolved_path.starts_with(fs::canonicalize(&workspace)?));
    assert_eq!(valid_missing.missing_components.len(), 2);
    Ok(())
}

#[test]
fn terminal_preview_sanitizer_escapes_ansi_and_preserves_unicode() {
    let raw = "src/فائل.rs\u{1b}[2J\u{1b}]0;pwnd\u{7}";
    let sanitized = sanitize_terminal_preview_token(raw);

    assert!(sanitized.contains("src/فائل.rs"));
    assert!(!sanitized.contains('\u{1b}'));
    assert!(!sanitized.contains('\u{7}'));
    assert!(sanitized.contains("\\x1B[2J"));
    assert!(sanitized.contains("\\u{0007}"));
}

#[test]
fn terminal_preview_sanitizer_escapes_bidi_overrides() {
    let raw = "safe-name\u{202E}gpj.exe";
    let sanitized = sanitize_terminal_preview_token(raw);

    assert!(!sanitized.contains('\u{202E}'));
    assert!(sanitized.contains("\\u{202E}"));
    assert!(sanitized.starts_with("safe-name"));
}

#[test]
fn policy_audit_blocks_disallowed_flags_and_env() -> Result<()> {
    let (_config, conn) = open_test_db("policy-audit")?;
    let policy = load_policy_rule(&conn, "policy_git_restore")?;

    let ok_argv = vec![
        "git".to_owned(),
        "restore".to_owned(),
        "--source".to_owned(),
        "HEAD".to_owned(),
        "src/main.rs".to_owned(),
    ];
    let mut ok_env = std::collections::BTreeMap::new();
    ok_env.insert("PATH".to_owned(), "/usr/bin".to_owned());
    audit_policy(&policy, &ok_argv, &ok_env)?;

    let blocked_flag = vec![
        "git".to_owned(),
        "restore".to_owned(),
        "--exec-path".to_owned(),
    ];
    assert!(matches!(
        audit_policy(&policy, &blocked_flag, &ok_env),
        Err(PolicyError::FlagBlocked(flag)) if flag == "--exec-path"
    ));

    let mut bad_env = std::collections::BTreeMap::new();
    bad_env.insert("LD_PRELOAD".to_owned(), "/tmp/libhack.so".to_owned());
    assert!(matches!(
        audit_policy(&policy, &ok_argv, &bad_env),
        Err(PolicyError::EnvNotAllowlisted(key)) if key == "LD_PRELOAD"
    ));
    Ok(())
}

#[test]
fn policy_failure_marks_request_security_blocked() -> Result<()> {
    let (_config, mut conn) = open_test_db("policy-request-block")?;
    seed_session_and_request(&conn, "req_policy_blocked")?;
    let policy = load_policy_rule(&conn, "policy_git_restore")?;
    let argv = vec![
        "git".to_owned(),
        "restore".to_owned(),
        "--exec-path".to_owned(),
    ];
    let env = std::collections::BTreeMap::new();

    let result = audit_policy_for_request(&mut conn, "req_policy_blocked", &policy, &argv, &env);
    assert!(matches!(
        result,
        Err(PolicyError::FlagBlocked(flag)) if flag == "--exec-path"
    ));

    let status: String = conn.query_row(
        "SELECT execution_status FROM request_records WHERE request_id = 'req_policy_blocked'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(status, "SECURITY_BLOCKED");
    Ok(())
}

#[test]
fn command_tokenizer_preserves_quoted_structural_arguments() -> Result<()> {
    let argv = tokenize_command_line(r#"git restore "file with spaces.rs" '--literal flag'"#)?;
    assert_eq!(
        argv,
        vec![
            "git".to_owned(),
            "restore".to_owned(),
            "file with spaces.rs".to_owned(),
            "--literal flag".to_owned(),
        ]
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn process_execution_preserves_multi_word_arguments_as_literal_indices() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = test_base_dir("execute-argv");
    let script = dir.join("print-args.sh");
    fs::write(
        &script,
        "#!/bin/sh\nprintf '<%s>\\n' \"$1\"\nprintf '<%s>\\n' \"$2\"\nprintf '<%s>\\n' \"$3\"\n",
    )?;
    let mut perms = fs::metadata(&script)?.permissions();
    perms.set_mode(0o700);
    fs::set_permissions(&script, perms)?;

    let argv = vec![
        script.to_string_lossy().to_string(),
        "two words".to_owned(),
        "--flag=value with space".to_owned(),
        "\"quoted-token\"".to_owned(),
    ];
    let stdout = execute_capture_stdout(&argv)?;
    let rendered = String::from_utf8(stdout)?;
    assert_eq!(
        rendered,
        "<two words>\n<--flag=value with space>\n<\"quoted-token\">\n"
    );
    Ok(())
}

#[test]
fn ai_learn_inserts_template_and_updates_fts_index() -> Result<()> {
    let (_config, mut conn) = open_test_db("ai-learn")?;
    seed_session_and_request(&conn, "req_learn_restore")?;

    let learned = learn_from_request(
        &mut conn,
        "req_learn_restore",
        r#"git restore "file with spaces.rs""#,
    )?;
    assert_eq!(
        learned.argv,
        vec![
            "git".to_owned(),
            "restore".to_owned(),
            "file with spaces.rs".to_owned(),
        ]
    );

    let row: (String, String, String) = conn.query_row(
        "SELECT source_type, trust_state, template_argv_json FROM unified_documents WHERE doc_id = ?1",
        params![learned.doc_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert_eq!(row.0, "USER_TEMPLATE");
    assert_eq!(row.1, "UNVERIFIED");
    assert!(row.2.contains("file with spaces.rs"));
    assert_eq!(fts_count(&conn, "restore")?, 1);

    let refresh_required: String = conn.query_row(
        "SELECT value FROM schema_metadata WHERE key = 'vector_cache_refresh_required'",
        [],
        |row| row.get(0),
    )?;
    assert!(!refresh_required.is_empty());
    Ok(())
}
