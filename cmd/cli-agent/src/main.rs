use anyhow::{bail, Context, Result};
use cli_agent::environment::capture_session_context;
use cli_agent::execute::execute_interactive;
use cli_agent::path_validate::{sanitize_terminal_preview_token, validate_workspace_path};
use cli_agent::pipeline::{
    observe_manual_command, resolve_natural_language_command, NaturalLanguageOutcome,
    OBSERVATION_WINDOW_MINUTES,
};
use cli_agent::policy::{audit_policy_for_request, load_policy_rule, sanitize_preview_tokens};
use cli_agent::sidecar_signal::notify_cache_invalidation;
use cli_agent::signals::ShutdownSignals;
use cli_agent::storage::{
    fetch_trusted_template_by_doc_id, initialize_database, insert_session_record, learn_from_request,
    mark_request_approved, mark_request_security_blocked, StorageConfig,
};
use cli_agent::validate::bind_template_slots;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Real shell-builtin-style commands that a user might literally type but
/// that never exist as files on `PATH`. Anything not on `PATH` and not in
/// this small allowlist is treated as a natural-language prompt instead of
/// a literal command to spawn.
const HARD_ALLOWLISTED_COMMANDS: &[&str] = &["cd", "pwd", "exit", "true", "false"];

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
        bail!("ai-run requires a command, or a natural-language description in quotes");
    }

    let signals = ShutdownSignals::install().context("failed to install signal handlers")?;

    if !looks_like_executable_argv(&argv) {
        return ai_run_natural_language(&argv, &signals);
    }

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
    observe_executed_literal_command(&argv);
    Ok(())
}

/// Best-effort hook into the passive-observation loop: lets it know a
/// literal command just ran successfully, in case it's the manual
/// correction for a recent natural-language miss in this terminal. Never
/// allowed to fail the command that already ran — any error here is only
/// ever printed to stderr, never propagated, since by this point the
/// user's actual command has already completed.
fn observe_executed_literal_command(argv: &[String]) {
    let outcome = (|| -> Result<Option<String>> {
        let config = StorageConfig::discover()?;
        let conn = config.open()?;
        initialize_database(&conn)?;
        let context = capture_session_context()?;
        observe_manual_command(&conn, &context, argv)
    })();

    match outcome {
        Ok(Some(prompt)) => eprintln!(
            "noted: if you ask ai-run \"{}\" again, I can offer to learn this command",
            prompt
        ),
        Ok(None) => {}
        Err(err) => eprintln!(
            "warning: passive-observation hook failed (command already ran fine): {}",
            err
        ),
    }
}

/// Decides whether `argv[0]` is a literal command to spawn directly, versus
/// a natural-language prompt that needs to go through the sidecar/template
/// pipeline. This is intentionally conservative: anything that isn't a real
/// executable on `PATH` (or in the small builtin allowlist) is treated as
/// natural language rather than guessed at.
fn looks_like_executable_argv(argv: &[String]) -> bool {
    let Some(first) = argv.first() else {
        return false;
    };

    if first.is_empty() || first.starts_with('-') {
        return false;
    }

    if HARD_ALLOWLISTED_COMMANDS.contains(&first.as_str()) {
        return true;
    }

    if first.contains('/') {
        return is_executable_file(Path::new(first));
    }

    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| is_executable_file(&dir.join(first)))
        })
        .unwrap_or(false)
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

fn ai_run_natural_language(argv: &[String], signals: &ShutdownSignals) -> Result<()> {
    let raw_prompt = argv.join(" ");
    println!("interpreting natural-language request: \"{}\"", raw_prompt);

    let config = StorageConfig::discover()?;
    let mut conn = config.open()?;
    initialize_database(&conn)?;
    let context = capture_session_context().context("failed to capture session context")?;
    signals.check().context("interrupted before sidecar lookup")?;

    let socket_path = config.base_dir.join("sidecar.sock");
    let outcome = resolve_natural_language_command(&mut conn, &context, &socket_path, &raw_prompt)
        .context("natural-language interpretation failed")?;

    match outcome {
        NaturalLanguageOutcome::Resolved(resolved) => {
            signals.check().context("interrupted before confirmation")?;
            render_preview(&resolved.argv)?;
            if !confirm_execute(signals)? {
                println!("aborted");
                return Ok(());
            }

            signals.check().context("interrupted before execution")?;
            let result =
                execute_interactive(&resolved.argv).context("child process execution failed")?;
            if result.success {
                mark_request_approved(&mut conn, &resolved.request_id)
                    .context("failed to mark request approved after execution")?;
            } else {
                bail!("child exited with status {:?}", result.status_code);
            }
            Ok(())
        }

        NaturalLanguageOutcome::Observing { request_id } => {
            println!("no matching command template yet for this prompt (request_id={request_id}).");
            println!(
                "if you run the right command manually with ai-run in the next {} minutes, \
                 I'll remember it and offer to learn it the next time you ask this.",
                OBSERVATION_WINDOW_MINUTES
            );
            Ok(())
        }

        NaturalLanguageOutcome::LearnFromObservedCorrection {
            request_id,
            suggested_argv,
        } => {
            signals.check().context("interrupted before learn confirmation")?;
            render_preview(&suggested_argv)?;
            let preview_line = shell_words::join(sanitize_preview_tokens(&suggested_argv));
            let prompt_text = format!(
                "I observed you manually run \"{}\" after this query last time. Learn it as a template? [y/N]: ",
                preview_line
            );
            if !confirm_prompt(signals, &prompt_text)? {
                println!("not learned; you can ask again later");
                return Ok(());
            }

            signals.check().context("interrupted before learning")?;
            let corrected_command = shell_words::join(&suggested_argv);
            let learned = learn_from_request(&mut conn, &request_id, &corrected_command)
                .context("failed to learn the observed correction")?;

            match notify_cache_invalidation(&socket_path, &request_id) {
                Ok(ack) => eprintln!(
                    "sidecar cache invalidated: {} documents rebuilt in {:.2}ms",
                    ack.document_count, ack.rebuild_duration_ms
                ),
                Err(err) => eprintln!(
                    "warning: learned template saved, but sidecar cache invalidation signal failed: {}",
                    err
                ),
            }

            println!("learned {}", learned.doc_id);
            Ok(())
        }
    }
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

    let socket_path = config.base_dir.join("sidecar.sock");
    match notify_cache_invalidation(&socket_path, &request_id) {
        Ok(ack) => eprintln!(
            "sidecar cache invalidated: {} documents rebuilt in {:.2}ms",
            ack.document_count, ack.rebuild_duration_ms
        ),
        Err(err) => eprintln!(
            "warning: learned template saved, but sidecar cache invalidation signal failed: {}",
            err
        ),
    }

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
    confirm_prompt(signals, "Execute? [y/N]: ")
}

fn confirm_prompt(signals: &ShutdownSignals, prompt: &str) -> Result<bool> {
    print!("{}", prompt);
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
    println!("  ai-run <natural language...>               resolve via the sidecar + template pipeline, then preview/confirm/execute");
    println!("  execute-argv <argv-json>                   execute literal argv JSON without shell translation");
    println!("  ai-learn --request-id <id> <command>       learn an explicit corrected command");
    println!("  validate-slots <request-id> <template-id> <slots-json> validate untrusted daemon slots against trusted schema");
    println!("  validate-path <workspace-root> <path>      validate a path slot stays inside workspace root");
    println!("  audit-policy <request-id> <rule-id> <argv-json> [env] audit argv/env against trusted policy");
    println!("  preview-token <token>                      sanitize one token for terminal preview");
    println!("  preview-argv <argv-json>                   sanitize argv JSON for terminal preview");
}
