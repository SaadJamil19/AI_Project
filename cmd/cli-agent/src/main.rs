use anyhow::{bail, Context, Result};
use cli_agent::environment::capture_session_context;
use cli_agent::execute::execute_interactive;
use cli_agent::path_validate::{sanitize_terminal_preview_token, validate_workspace_path};
use cli_agent::policy::{audit_policy_for_request, load_policy_rule, sanitize_preview_tokens};
use cli_agent::signals::ShutdownSignals;
use cli_agent::storage::{
    fetch_trusted_template_by_doc_id, initialize_database, insert_session_record, learn_from_request,
    mark_request_security_blocked, StorageConfig,
};
use cli_agent::validate::bind_template_slots;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() -> Result<()> {
    let command = std::env::args().nth(1).unwrap_or_else(|| "init".to_owned());

    match command.as_str() {
        "init" => init_database(),
        "capture-session" => print_session_context(),
        "record-session" => record_session(),
        "ai-run" => ai_run(),
        "execute-argv" => execute_argv_json(),
        "ai-learn" => ai_learn(),
        "validate-slots" => validate_slots(),
        "validate-path" => validate_path(),
        "audit-policy" => audit_policy_command(),
        "preview-token" => preview_token(),
        "preview-argv" => preview_argv(),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => bail!("unknown command '{}'; run cli-agent help", other),
    }
}

fn init_database() -> Result<()> {
    let config = StorageConfig::discover()?;
    let conn = config.open()?;
    initialize_database(&conn)?;
    println!("initialized database at {}", config.db_path.display());
    Ok(())
}

fn print_session_context() -> Result<()> {
    let context = capture_session_context()?;
    let json = serde_json::to_string_pretty(&context).context("failed to serialize session context")?;
    println!("{}", json);
    Ok(())
}

fn record_session() -> Result<()> {
    let config = StorageConfig::discover()?;
    let conn = config.open()?;
    initialize_database(&conn)?;
    let context = capture_session_context()?;
    let session_id = insert_session_record(&conn, &context)?;
    println!("{}", session_id);
    Ok(())
}

fn ai_run() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(2).collect();
    if argv.is_empty() {
        bail!("ai-run requires a command and arguments");
    }

    let signals = ShutdownSignals::install().context("failed to install signal handlers")?;
    render_preview(&argv)?;
    signals.check().context("interrupted before confirmation")?;
    if !confirm_execute(&signals)? {
        println!("aborted");
        return Ok(());
    }
    signals.check().context("interrupted before execution")?;
    let result = execute_interactive(&argv).context("child process execution failed")?;
    if !result.success {
        bail!("child exited with status {:?}", result.status_code);
    }
    Ok(())
}

fn execute_argv_json() -> Result<()> {
    let argv_json = std::env::args()
        .nth(2)
        .context("execute-argv requires argv JSON array")?;
    let argv: Vec<String> =
        serde_json::from_str(&argv_json).context("argv must be a JSON string array")?;
    let result = execute_interactive(&argv).context("child process execution failed")?;
    if !result.success {
        bail!("child exited with status {:?}", result.status_code);
    }
    Ok(())
}

fn ai_learn() -> Result<()> {
    let mut args = std::env::args().skip(2);
    let mut request_id = None;
    let mut corrected = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--request-id" {
            request_id = args.next();
        } else {
            corrected.push(arg);
            corrected.extend(args);
            break;
        }
    }

    let request_id = request_id.context("ai-learn requires --request-id <id>")?;
    let corrected_command = corrected.join(" ");
    if corrected_command.trim().is_empty() {
        bail!("ai-learn requires a corrected command string");
    }

    let signals = ShutdownSignals::install().context("failed to install signal handlers")?;
    signals.check().context("interrupted before learning")?;
    let config = StorageConfig::discover()?;
    let mut conn = config.open()?;
    initialize_database(&conn)?;
    let learned = learn_from_request(&mut conn, &request_id, &corrected_command)?;
    signals.check().context("interrupted after learning transaction")?;
    println!("{}", learned.doc_id);
    Ok(())
}

fn validate_slots() -> Result<()> {
    let request_id = std::env::args()
        .nth(2)
        .context("validate-slots requires request_id")?;
    let candidate_template_id = std::env::args()
        .nth(3)
        .context("validate-slots requires candidate_template_id")?;
    let raw_untrusted_slots_json = std::env::args()
        .nth(4)
        .context("validate-slots requires raw_untrusted_slots_json")?;

    let config = StorageConfig::discover()?;
    let mut conn = config.open()?;
    let template = fetch_trusted_template_by_doc_id(&conn, &candidate_template_id)?;
    let bound = match bind_template_slots(
        &template.trust_state,
        &template.slot_schema_json,
        &raw_untrusted_slots_json,
    ) {
        Ok(bound) => bound,
        Err(err) => {
            mark_request_security_blocked(&mut conn, &request_id).with_context(|| {
                format!(
                    "slot validation failed with {}; failed to mark request_id={} SECURITY_BLOCKED",
                    err, request_id
                )
            })?;
            return Err(err).context("untrusted slots failed trusted schema validation");
        }
    };
    let json = serde_json::to_string_pretty(&bound.into_inner())
        .context("failed to serialize bound slot map")?;
    println!("{}", json);
    Ok(())
}

fn validate_path() -> Result<()> {
    let workspace_root = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("validate-path requires workspace_root")?;
    let raw_untrusted_path = std::env::args()
        .nth(3)
        .context("validate-path requires raw_untrusted_path")?;

    let validated = validate_workspace_path(workspace_root, &raw_untrusted_path)
        .context("path validation failed")?;
    println!("{}", validated.resolved_path.display());
    Ok(())
}

fn audit_policy_command() -> Result<()> {
    let request_id = std::env::args()
        .nth(2)
        .context("audit-policy requires request_id")?;
    let rule_id = std::env::args()
        .nth(3)
        .context("audit-policy requires policy rule_id")?;
    let argv_json = std::env::args()
        .nth(4)
        .context("audit-policy requires argv JSON array")?;
    let env_json = std::env::args().nth(5).unwrap_or_else(|| "{}".to_owned());

    let argv: Vec<String> =
        serde_json::from_str(&argv_json).context("argv must be a JSON string array")?;
    let inherited_env: BTreeMap<String, String> =
        serde_json::from_str(&env_json).context("env must be a JSON string object")?;

    let config = StorageConfig::discover()?;
    let mut conn = config.open()?;
    let policy = load_policy_rule(&conn, &rule_id).context("failed to load policy rule")?;
    audit_policy_for_request(&mut conn, &request_id, &policy, &argv, &inherited_env)
        .context("policy audit failed")?;
    println!("policy-ok");
    Ok(())
}

fn preview_token() -> Result<()> {
    let raw = std::env::args()
        .nth(2)
        .context("preview-token requires a token")?;
    println!("{}", sanitize_terminal_preview_token(&raw));
    Ok(())
}

fn preview_argv() -> Result<()> {
    let argv_json = std::env::args()
        .nth(2)
        .context("preview-argv requires argv JSON array")?;
    let argv: Vec<String> =
        serde_json::from_str(&argv_json).context("argv must be a JSON string array")?;
    let sanitized = sanitize_preview_tokens(&argv);
    println!(
        "{}",
        serde_json::to_string_pretty(&sanitized).context("failed to serialize preview argv")?
    );
    Ok(())
}

fn render_preview(argv: &[String]) -> Result<()> {
    let sanitized = sanitize_preview_tokens(argv);
    println!("\x1b[1;36m┌─ semantic-cli-agent preview ─┐\x1b[0m");
    for (index, token) in sanitized.iter().enumerate() {
        println!("\x1b[36m│\x1b[0m argv[{index}] = \x1b[1m{token}\x1b[0m");
    }
    println!("\x1b[1;36m└──────────────────────────────┘\x1b[0m");
    Ok(())
}

fn confirm_execute(signals: &ShutdownSignals) -> Result<bool> {
    print!("Execute? [y/N]: ");
    io::stdout().flush().context("failed to flush confirmation prompt")?;
    signals.check().context("interrupted during confirmation")?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read confirmation")?;
    signals.check().context("interrupted during confirmation")?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "YES"))
}

fn print_help() {
    println!("semantic-cli-agent commands:");
    println!("  init                                      initialize storage, WAL, schema, and FTS5 triggers");
    println!("  capture-session                           print current terminal/session metadata as JSON");
    println!("  record-session                            initialize storage and persist current session record");
    println!("  ai-run <command> [args...]                 preview, confirm, and execute literal argv");
    println!("  execute-argv <argv-json>                   execute literal argv JSON without shell translation");
    println!("  ai-learn --request-id <id> <command>       learn an explicit corrected command");
    println!("  validate-slots <request-id> <template-id> <slots-json> validate untrusted daemon slots against trusted schema");
    println!("  validate-path <workspace-root> <path>      validate a path slot stays inside workspace root");
    println!("  audit-policy <request-id> <rule-id> <argv-json> [env] audit argv/env against trusted policy");
    println!("  preview-token <token>                      sanitize one token for terminal preview");
    println!("  preview-argv <argv-json>                   sanitize argv JSON for terminal preview");
}
