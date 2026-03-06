use crate::cli::{CursorCliEvent, CursorCliRuntime, CursorPromptRequest, CursorPromptResult};
use crate::error::ErrorKind;
use crate::session::SessionState;
use agent_client_protocol::StopReason;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub struct PromptEngine {
    cli: Arc<CursorCliRuntime>,
}

#[derive(Debug, Clone)]
pub struct PromptRunOptions {
    pub prompt_text: String,
    pub mode: Option<String>,
    pub sandbox: Option<String>,
    pub allow_edits: bool,
    pub trust_workspace: bool,
}

impl PromptRunOptions {
    fn into_cli_request(self, session: &SessionState) -> CursorPromptRequest {
        CursorPromptRequest {
            prompt: self.prompt_text,
            mode: self.mode,
            sandbox: self.sandbox,
            model: Some(session.current_model.to_string()),
            resume: session.cursor_resume_id.clone(),
            cwd: Some(session.cwd.clone()),
            allow_edits: self.allow_edits,
            trust_workspace: self.trust_workspace,
        }
    }
}

impl PromptEngine {
    pub fn new(cli: Arc<CursorCliRuntime>) -> Self {
        Self { cli }
    }

    pub async fn run_prompt_non_stream(
        &self,
        session: &SessionState,
        options: PromptRunOptions,
    ) -> Result<(CursorPromptResult, StopReason), ErrorKind> {
        let result = self
            .cli
            .run_prompt_json(options.into_cli_request(session))
            .await?;

        Ok((result, StopReason::EndTurn))
    }

    pub async fn run_prompt_stream_events(
        &self,
        session: &SessionState,
        options: PromptRunOptions,
        event_tx: Option<UnboundedSender<CursorCliEvent>>,
    ) -> Result<(Vec<CursorCliEvent>, StopReason), ErrorKind> {
        let events = self
            .cli
            .run_prompt_stream(options.into_cli_request(session), event_tx)
            .await?;
        Ok((events, StopReason::EndTurn))
    }

    pub fn classify_failure_stop_reason(error: &ErrorKind) -> StopReason {
        match error {
            ErrorKind::Cancelled => StopReason::Cancelled,
            _ => StopReason::Refusal,
        }
    }
}
