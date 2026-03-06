use agent_client_protocol::Error as AcpError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ErrorKind {
    #[error("authentication required")]
    AuthRequired,
    #[error("requested model is unsupported: {requested}")]
    ModelUnsupported {
        requested: String,
        available: Vec<String>,
    },
    #[error("workspace trust required")]
    WorkspaceTrustRequired,
    #[error("cursor CLI binary not found")]
    CliMissing,
    #[error("cursor CLI invocation failed (exit code: {exit_code:?})")]
    CliInvocationFailed {
        exit_code: Option<i32>,
        stderr_excerpt: String,
    },
    #[error("cursor CLI protocol violation: {parser_phase}")]
    CliProtocolViolation {
        parser_phase: &'static str,
        raw_excerpt: String,
    },
    #[error("rate limited")]
    RateLimited,
    #[error("cancelled")]
    Cancelled,
    #[error("transient I/O error: {0}")]
    TransientIo(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl ErrorKind {
    pub fn to_acp_error(&self) -> AcpError {
        match self {
            Self::AuthRequired => AcpError::auth_required(),
            Self::ModelUnsupported {
                requested,
                available,
            } => {
                let mut error = AcpError::invalid_params();
                error.message = format!(
                    "Unsupported model `{requested}`. Available models: {}",
                    available.join(", ")
                );
                error.data = Some(serde_json::json!({
                    "reason": "model_unsupported",
                    "requested": requested,
                    "available": available,
                }));
                error
            }
            Self::WorkspaceTrustRequired => {
                let mut error = AcpError::invalid_request();
                error.message = "Workspace trust required by cursor-agent".to_string();
                error.data = Some(serde_json::json!({
                    "reason": "workspace_trust_required"
                }));
                error
            }
            Self::CliMissing => {
                let mut error = AcpError::internal_error();
                error.message = "cursor-agent CLI not installed or not found in PATH".to_string();
                error.data = Some(serde_json::json!({ "reason": "cli_missing" }));
                error
            }
            Self::CliInvocationFailed {
                exit_code,
                stderr_excerpt,
            } => {
                let mut error = AcpError::internal_error();
                error.message = "cursor-agent CLI invocation failed".to_string();
                error.data = Some(serde_json::json!({
                    "reason": "cli_invocation_failed",
                    "exit_code": exit_code,
                    "stderr_excerpt": stderr_excerpt,
                }));
                error
            }
            Self::CliProtocolViolation {
                parser_phase,
                raw_excerpt,
            } => {
                let mut error = AcpError::internal_error();
                error.message = "cursor-agent output did not match expected protocol".to_string();
                error.data = Some(serde_json::json!({
                    "reason": "cli_protocol_violation",
                    "parser_phase": parser_phase,
                    "raw_excerpt": raw_excerpt,
                }));
                error
            }
            Self::RateLimited => {
                let mut error = AcpError::internal_error();
                error.message = "rate limited".to_string();
                error
            }
            Self::Cancelled => {
                let mut error = AcpError::internal_error();
                error.message = "request cancelled".to_string();
                error.data = Some(serde_json::json!({ "reason": "cancelled" }));
                error
            }
            Self::TransientIo(detail) => {
                let mut error = AcpError::internal_error();
                error.message = "transient io error".to_string();
                error.data = Some(serde_json::json!({ "reason": "transient_io", "detail": detail }));
                error
            }
            Self::Internal(detail) => {
                let mut error = AcpError::internal_error();
                error.message = "internal error".to_string();
                error.data = Some(serde_json::json!({ "reason": "internal", "detail": detail }));
                error
            }
        }
    }
}
