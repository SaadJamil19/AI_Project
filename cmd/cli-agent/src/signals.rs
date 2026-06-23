use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SignalError {
    #[error("failed to register signal handler: {0}")]
    Register(String),
    #[error("operation interrupted by terminal signal")]
    Interrupted,
}

#[derive(Clone, Debug)]
pub struct ShutdownSignals {
    interrupted: Arc<AtomicBool>,
}

impl ShutdownSignals {
    pub fn install() -> Result<Self, SignalError> {
        let interrupted = Arc::new(AtomicBool::new(false));
        flag::register(SIGINT, Arc::clone(&interrupted))
            .map_err(|err| SignalError::Register(err.to_string()))?;
        flag::register(SIGTERM, Arc::clone(&interrupted))
            .map_err(|err| SignalError::Register(err.to_string()))?;
        Ok(Self { interrupted })
    }

    pub fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<(), SignalError> {
        if self.interrupted() {
            Err(SignalError::Interrupted)
        } else {
            Ok(())
        }
    }
}
