use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "1.0.0";
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Generous read timeout: a query can fall through hybrid retrieval all the
/// way to a local LLM call, which is slow on CPU-only hardware.
const READ_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Error)]
pub enum IpcError {
    #[error("sidecar socket not found at {0}; the sidecar daemon may not be running")]
    SocketMissing(String),
    #[error("failed to connect to sidecar socket {path}: {message}")]
    Connect { path: String, message: String },
    #[error("failed to send search request: {0}")]
    Send(String),
    #[error("failed to read search response: {0}")]
    Read(String),
    #[error("sidecar returned malformed response JSON: {0}")]
    InvalidResponseJson(String),
    #[error("sidecar rejected the search request: {0}")]
    Rejected(String),
}

#[derive(Debug, Clone, Serialize)]
struct SidecarSearchRequest<'a> {
    protocol_version: &'a str,
    request_id: &'a str,
    query: &'a str,
    limit: u32,
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RetrievalEvidence {
    #[serde(default)]
    pub fts5_lexical_score: Option<f64>,
    #[serde(default)]
    pub vector_cosine_distance: Option<f64>,
    #[serde(default)]
    pub vector_rank: Option<i64>,
    #[serde(default)]
    pub embedding_duration_ms: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RiskHints {
    #[serde(default)]
    pub contains_path_arguments: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentProposal {
    #[serde(default)]
    pub candidate_template_id: Option<String>,
    pub typed_intent: String,
    #[serde(default)]
    pub retrieval_evidence: RetrievalEvidence,
    #[serde(default)]
    pub risk_hints: RiskHints,
    #[serde(default)]
    pub raw_untrusted_slots: BTreeMap<String, String>,
}

/// The untrusted proposal envelope read back from the Python sidecar. Every
/// field here is exactly as untrusted as the process that sent it: Rust must
/// reload the real template by `candidate_template_id` and re-validate
/// `raw_untrusted_slots` from scratch before any of this can influence a
/// real command. This struct exists only to get the bytes off the wire.
#[derive(Debug, Clone, Deserialize)]
pub struct UntrustedProposal {
    pub protocol_version: String,
    pub request_id: String,
    pub source_provenance: String,
    pub intent_proposal: IntentProposal,
}

/// Sends a natural-language query to the persistent Python sidecar over its
/// Unix Domain Socket and returns its untrusted proposal. Gated behind
/// `#[cfg(unix)]` since UDS does not exist on non-Unix targets.
#[cfg(unix)]
pub fn query_sidecar(
    socket_path: &Path,
    request_id: &str,
    query: &str,
    limit: u32,
) -> Result<UntrustedProposal, IpcError> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    if !socket_path.exists() {
        return Err(IpcError::SocketMissing(socket_path.display().to_string()));
    }

    let stream = UnixStream::connect(socket_path).map_err(|err| IpcError::Connect {
        path: socket_path.display().to_string(),
        message: err.to_string(),
    })?;
    stream
        .set_read_timeout(Some(READ_TIMEOUT))
        .map_err(|err| IpcError::Connect {
            path: socket_path.display().to_string(),
            message: err.to_string(),
        })?;
    stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .map_err(|err| IpcError::Connect {
            path: socket_path.display().to_string(),
            message: err.to_string(),
        })?;

    let request = SidecarSearchRequest {
        protocol_version: PROTOCOL_VERSION,
        request_id,
        query,
        limit,
    };
    let mut line =
        serde_json::to_vec(&request).map_err(|err| IpcError::Send(err.to_string()))?;
    line.push(b'\n');

    let mut writer = &stream;
    writer
        .write_all(&line)
        .map_err(|err| IpcError::Send(err.to_string()))?;
    writer
        .flush()
        .map_err(|err| IpcError::Send(err.to_string()))?;

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|err| IpcError::Read(err.to_string()))?;
    if response_line.trim().is_empty() {
        return Err(IpcError::Read("empty response from sidecar".to_owned()));
    }

    if let Ok(envelope) = serde_json::from_str::<SidecarErrorEnvelope>(&response_line) {
        return Err(IpcError::Rejected(format!(
            "{}: {}",
            envelope.error.code, envelope.error.message
        )));
    }

    serde_json::from_str(&response_line)
        .map_err(|err| IpcError::InvalidResponseJson(err.to_string()))
}

#[cfg(not(unix))]
pub fn query_sidecar(
    _socket_path: &Path,
    _request_id: &str,
    _query: &str,
    _limit: u32,
) -> Result<UntrustedProposal, IpcError> {
    Err(IpcError::SocketMissing(
        "Unix Domain Sockets are not supported on this platform".to_owned(),
    ))
}
