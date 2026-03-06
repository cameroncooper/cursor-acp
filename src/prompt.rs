use crate::cli::{CursorCliEvent, CursorCliRuntime, CursorPromptResult};
use crate::error::ErrorKind;
use crate::session::SessionState;
use agent_client_protocol::StopReason;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub struct PromptEngine {
    cli: Arc<CursorCliRuntime>,
}

impl PromptEngine {
    pub fn new(cli: Arc<CursorCliRuntime>) -> Self {
        Self { cli }
    }

    pub async fn run_prompt_non_stream(
        &self,
        session: &SessionState,
        prompt_text: String,
        mode: Option<String>,
        sandbox: Option<String>,
        allow_edits: bool,
        trust_workspace: bool,
    ) -> Result<(CursorPromptResult, StopReason), ErrorKind> {
        let result = self
            .cli
            .run_prompt_json(
                prompt_text,
                mode,
                sandbox,
                Some(session.current_model.to_string()),
                session.cursor_resume_id.clone(),
                Some(session.cwd.clone()),
                allow_edits,
                trust_workspace,
            )
            .await?;

        Ok((result, StopReason::EndTurn))
    }

    pub async fn run_prompt_stream_events(
        &self,
        session: &SessionState,
        prompt_text: String,
        mode: Option<String>,
        sandbox: Option<String>,
        allow_edits: bool,
        trust_workspace: bool,
        event_tx: Option<UnboundedSender<CursorCliEvent>>,
    ) -> Result<(Vec<CursorCliEvent>, StopReason), ErrorKind> {
        let events = self
            .cli
            .run_prompt_stream(
                prompt_text,
                mode,
                sandbox,
                Some(session.current_model.to_string()),
                session.cursor_resume_id.clone(),
                Some(session.cwd.clone()),
                allow_edits,
                trust_workspace,
                event_tx,
            )
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
