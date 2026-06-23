use std::process::{Command, ExitStatus, Stdio};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecuteError {
    #[error("argv must contain at least one program")]
    EmptyArgv,
    #[error("failed to spawn child process {program}: {message}")]
    Spawn { program: String, message: String },
    #[error("failed while waiting for child process {program}: {message}")]
    Wait { program: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub status_code: Option<i32>,
    pub success: bool,
}

pub fn execute_interactive(argv: &[String]) -> Result<ExecutionResult, ExecuteError> {
    let program = argv.first().ok_or(ExecuteError::EmptyArgv)?;
    let mut child = Command::new(program)
        .args(&argv[1..])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| ExecuteError::Spawn {
            program: program.clone(),
            message: err.to_string(),
        })?;

    let status = child.wait().map_err(|err| ExecuteError::Wait {
        program: program.clone(),
        message: err.to_string(),
    })?;
    Ok(status_to_result(status))
}

pub fn execute_capture_stdout(argv: &[String]) -> Result<Vec<u8>, ExecuteError> {
    let program = argv.first().ok_or(ExecuteError::EmptyArgv)?;
    let output = Command::new(program)
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|err| ExecuteError::Spawn {
            program: program.clone(),
            message: err.to_string(),
        })?;
    Ok(output.stdout)
}

fn status_to_result(status: ExitStatus) -> ExecutionResult {
    ExecutionResult {
        status_code: status.code(),
        success: status.success(),
    }
}
