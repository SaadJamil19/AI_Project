use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct SessionContext {
    pub pid: i64,
    pub ppid: i64,
    pub tty_device: String,
    pub canonical_cwd: PathBuf,
    pub canonical_git_root: Option<PathBuf>,
    pub user_id: String,
    pub hostname: String,
}

pub fn capture_session_context() -> Result<SessionContext> {
    let cwd = std::env::current_dir().context("failed to read current working directory")?;
    let canonical_cwd = cwd
        .canonicalize()
        .with_context(|| format!("failed to canonicalize cwd {}", cwd.display()))?;

    Ok(SessionContext {
        pid: current_pid(),
        ppid: parent_pid(),
        tty_device: active_tty_device(),
        canonical_git_root: discover_git_root(&canonical_cwd)?,
        canonical_cwd,
        user_id: current_user_name(),
        hostname: host_name(),
    })
}

fn discover_git_root(cwd: &Path) -> Result<Option<PathBuf>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .arg("rev-parse")
        .arg("--show-toplevel")
        .output();

    let output = match output {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };

    if !output.status.success() {
        return Ok(None);
    }

    let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if root.is_empty() {
        return Ok(None);
    }

    let root_path = PathBuf::from(root);
    Ok(Some(root_path.canonicalize().with_context(|| {
        format!("git reported a root that could not be canonicalized: {}", root_path.display())
    })?))
}

#[cfg(unix)]
fn current_pid() -> i64 {
    i64::from(nix::unistd::getpid().as_raw())
}

#[cfg(not(unix))]
fn current_pid() -> i64 {
    i64::from(std::process::id())
}

#[cfg(unix)]
fn parent_pid() -> i64 {
    i64::from(nix::unistd::getppid().as_raw())
}

#[cfg(not(unix))]
fn parent_pid() -> i64 {
    0
}

#[cfg(unix)]
fn active_tty_device() -> String {
    std::fs::read_link("/proc/self/fd/0")
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
        .or_else(|| std::env::var("TTY").ok())
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(not(unix))]
fn active_tty_device() -> String {
    std::env::var("TTY")
        .or_else(|_| std::env::var("WT_SESSION"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

#[cfg(unix)]
fn current_user_name() -> String {
    use nix::unistd::{Uid, User};

    User::from_uid(Uid::current())
        .ok()
        .flatten()
        .map(|user| user.name)
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(not(unix))]
fn current_user_name() -> String {
    std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

#[cfg(unix)]
fn host_name() -> String {
    nix::unistd::gethostname()
        .ok()
        .and_then(|name| name.into_string().ok())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(not(unix))]
fn host_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned())
}
