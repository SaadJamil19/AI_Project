use anyhow::{bail, Context, Result};
use cli_agent::environment::capture_session_context;
use cli_agent::storage::{initialize_database, insert_session_record, StorageConfig};

fn main() -> Result<()> {
    let command = std::env::args().nth(1).unwrap_or_else(|| "init".to_owned());

    match command.as_str() {
        "init" => init_database(),
        "capture-session" => print_session_context(),
        "record-session" => record_session(),
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

fn print_help() {
    println!("semantic-cli-agent phase1 commands:");
    println!("  init             initialize storage, WAL, schema, and FTS5 triggers");
    println!("  capture-session  print current terminal/session metadata as JSON");
    println!("  record-session   initialize storage and persist current session record");
}
