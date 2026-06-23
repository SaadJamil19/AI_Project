use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "1.0.0";
const IO_TIMEOUT: Duration = Duration::from_millis(1_500);

#[derive(Debug, Error)]
pub enum SidecarSignalError {
    #[error("sidecar socket not found at {0}; the sidecar daemon may not be running")]
    SocketMissing(String),
    #[error("failed to connect to sidecar socket {path}: {message}")]
    Connect { path: String, message: String },
    #[error("failed to send cache invalidation request: {0}")]
    Send(String),
    #[error("failed to read cache invalidation acknowledgement: {0}")]
    Read(String),
    #[error("sidecar returned malformed acknowledgement JSON: {0}")]
    InvalidAckJson(String),
    #[error("sidecar rejected cache invalidation request: {0}")]
    Rejected(String),
}

#[derive(Debug, Clone, Serialize)]
struct InvalidationRequest<'a> {
    protocol_version: &'a str,
    command: &'a str,
    request_id: &'a str,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CacheInvalidationAck {
    pub protocol_version: String,
    pub request_id: String,
    pub status: String,
    pub document_count: i64,
    pub rebuild_duration_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct SidecarErrorEnvelope {
    error: SidecarErrorBody,
}

#[derive(Debug, Clone, Deserialize)]
struct SidecarErrorBody {
    code: String,
    message: String,
}

/// Actively pushes a cache invalidation command to the persistent Python
/// sidecar over its Unix Domain Socket so it rebuilds its in-memory FAISS
/// cache immediately, instead of relying on the sidecar to poll a database
/// flag on its own schedule.
#[cfg(unix)]
pub fn notify_cache_invalidation(
    socket_path: &Path,
    request_id: &str,
) -> Result<CacheInvalidationAck, SidecarSignalError> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    if !socket_path.exists() {
        return Err(SidecarSignalError::SocketMissing(
            socket_path.display().to_string(),
        ));
    }

    let stream = UnixStream::connect(socket_path).map_err(|err| SidecarSignalError::Connect {
        path: socket_path.display().to_string(),
        message: err.to_string(),
    })?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|err| SidecarSignalError::Connect {
            path: socket_path.display().to_string(),
            message: err.to_string(),
        })?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|err| SidecarSignalError::Connect {
            path: socket_path.display().to_string(),
            message: err.to_string(),
        })?;

    let request = InvalidationRequest {
        protocol_version: PROTOCOL_VERSION,
        command: "invalidate_cache",
        request_id,
    };
    let mut line =
        serde_json::to_vec(&request).map_err(|err| SidecarSignalError::Send(err.to_string()))?;
    line.push(b'\n');

    let mut writer = &stream;
    writer
        .write_all(&line)
        .map_err(|err| SidecarSignalError::Send(err.to_string()))?;
    writer
        .flush()
        .map_err(|err| SidecarSignalError::Send(err.to_string()))?;

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|err| SidecarSignalError::Read(err.to_string()))?;
    if response_line.trim().is_empty() {
        return Err(SidecarSignalError::Read(
            "empty response from sidecar".to_owned(),
        ));
    }

    if let Ok(envelope) = serde_json::from_str::<SidecarErrorEnvelope>(&response_line) {
        return Err(SidecarSignalError::Rejected(format!(
            "{}: {}",
            envelope.error.code, envelope.error.message
        )));
    }

    serde_json::from_str(&response_line)
        .map_err(|err| SidecarSignalError::InvalidAckJson(err.to_string()))
}

#[cfg(not(unix))]
pub fn notify_cache_invalidation(
    _socket_path: &Path,
    _request_id: &str,
) -> Result<CacheInvalidationAck, SidecarSignalError> {
    Err(SidecarSignalError::SocketMissing(
        "Unix Domain Sockets are not supported on this platform".to_owned(),
    ))
}
