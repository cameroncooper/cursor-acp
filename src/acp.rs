use crate::auth::{AUTH_METHOD_CURSOR_LOGIN, terminal_auth_method};
use crate::cli::{CursorCliConfig, CursorCliRuntime};
use crate::error::ErrorKind;
use crate::prompt::{PromptEngine, PromptRunOptions};
use crate::session::{SessionState, SessionStore};
use agent_client_protocol::{
    Agent, AgentCapabilities, AgentSideConnection, AuthenticateRequest, AuthenticateResponse,
    CancelNotification, Client, ClientCapabilities, ContentBlock, ContentChunk, CurrentModeUpdate,
    Error, Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
    ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, McpCapabilities,
    NewSessionRequest, NewSessionResponse, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus,
    PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion, SessionCapabilities,
    SessionConfigKind, SessionConfigOption, SessionConfigSelectOption, SessionConfigSelectOptions,
    SessionInfo, SessionInfoUpdate, SessionListCapabilities, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse, ToolCall,
    ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
    UsageUpdate, WriteTextFileRequest,
};
use chrono::{DateTime, Utc};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::warn;

pub struct CursorAcpAgent {
    session_store: Arc<Mutex<SessionStore>>,
    cli_runtime: Arc<CursorCliRuntime>,
    prompt_engine: Arc<PromptEngine>,
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    client_connection: Arc<Mutex<Option<Arc<AgentSideConnection>>>>,
    edit_enabled_sessions: Arc<Mutex<HashSet<agent_client_protocol::SessionId>>>,
    applied_managed_edit_calls: Arc<Mutex<HashSet<String>>>,
}

impl Default for CursorAcpAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl CursorAcpAgent {
    const LIST_SESSIONS_PAGE_SIZE: usize = 200;

    pub fn new() -> Self {
        Self::new_with_config(CursorCliConfig::default())
    }

    pub fn new_with_binary(binary: String) -> Self {
        Self::new_with_config(CursorCliConfig { binary })
    }

    pub fn new_with_config(config: CursorCliConfig) -> Self {
        let cli_runtime = Arc::new(CursorCliRuntime::new(config));
        Self {
            session_store: Arc::new(Mutex::new(SessionStore::new())),
            prompt_engine: Arc::new(PromptEngine::new(cli_runtime.clone())),
            cli_runtime,
            client_capabilities: Arc::new(Mutex::new(ClientCapabilities::default())),
            client_connection: Arc::new(Mutex::new(None)),
            edit_enabled_sessions: Arc::new(Mutex::new(HashSet::new())),
            applied_managed_edit_calls: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn attach_connection(&self, connection: Arc<AgentSideConnection>) {
        *self.client_connection.lock().unwrap() = Some(connection);
    }

    fn map_error(error: ErrorKind) -> Error {
        error.to_acp_error()
    }

    fn flatten_prompt(prompt: &[ContentBlock]) -> String {
        let mut text = String::new();
        for block in prompt {
            match block {
                ContentBlock::Text(content) => {
                    text.push_str(content.text.as_ref());
                    text.push('\n');
                }
                ContentBlock::Image(_)
                | ContentBlock::Audio(_)
                | ContentBlock::Resource(_)
                | ContentBlock::ResourceLink(_) => {}
                _ => {}
            }
        }
        text.trim().to_string()
    }

    async fn notify_session_update(
        &self,
        session_id: agent_client_protocol::SessionId,
        update: SessionUpdate,
    ) -> Result<(), Error> {
        let client = self.client_connection.lock().unwrap().clone();
        if let Some(client) = client {
            client
                .session_notification(SessionNotification::new(session_id, update))
                .await?;
        }
        Ok(())
    }

    fn event_to_session_update(event: &crate::cli::CursorCliEvent) -> Option<SessionUpdate> {
        match event {
            crate::cli::CursorCliEvent::AssistantDelta(text) => Some(
                SessionUpdate::AgentMessageChunk(ContentChunk::new(text.clone().into())),
            ),
            crate::cli::CursorCliEvent::ThinkingDelta(text) => Some(
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(text.clone().into())),
            ),
            crate::cli::CursorCliEvent::System { message, .. } => {
                if message.trim().is_empty() {
                    None
                } else {
                    Some(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                        message.clone().into(),
                    )))
                }
            }
            crate::cli::CursorCliEvent::ToolCallStart(tool_call) => {
                let locations: Vec<ToolCallLocation> = tool_call
                    .locations
                    .iter()
                    .map(ToolCallLocation::new)
                    .collect();
                Some(SessionUpdate::ToolCall(
                    ToolCall::new(tool_call.tool_call_id.clone(), tool_call.title.clone())
                        .kind(tool_call.kind)
                        .status(ToolCallStatus::InProgress)
                        .locations(locations)
                        .raw_input(tool_call.raw_input.clone()),
                ))
            }
            crate::cli::CursorCliEvent::ToolCallUpdate(tool_call) => {
                let tool_call = tool_call.as_ref();
                let locations: Vec<ToolCallLocation> = tool_call
                    .locations
                    .iter()
                    .map(ToolCallLocation::new)
                    .collect();
                let mut fields = ToolCallUpdateFields::new()
                    .status(tool_call.status)
                    .title(tool_call.title.clone())
                    .locations(locations)
                    .raw_input(tool_call.raw_input.clone())
                    .raw_output(tool_call.raw_output.clone());
                let message_implies_edit = tool_call
                    .message
                    .as_deref()
                    .map(|text| {
                        let lower = text.to_lowercase();
                        lower.contains("has been updated")
                            || lower.contains("created")
                            || lower.contains("deleted")
                            || lower.contains("written")
                    })
                    .unwrap_or(false);
                let mut content = Vec::new();
                let synthetic_diff = if tool_call.diff.is_none()
                    && tool_call.status == ToolCallStatus::Completed
                    && (tool_call.kind == agent_client_protocol::ToolKind::Edit
                        || message_implies_edit)
                {
                    tool_call.locations.first().and_then(|path| {
                        std::fs::read_to_string(path).ok().map(|new_text| {
                            agent_client_protocol::Diff::new(path.clone(), new_text)
                        })
                    })
                } else {
                    None
                };
                if let Some(diff) = tool_call.diff.as_ref().or(synthetic_diff.as_ref()) {
                    content.push(ToolCallContent::Diff(diff.clone()));
                }
                if let Some(message) = &tool_call.message {
                    let has_diff = !content.is_empty();
                    if !(has_diff && Self::is_generic_edit_message(message)) {
                        content.push(ToolCallContent::from(message.clone()));
                    }
                }
                if !content.is_empty() {
                    fields = fields.content(content);
                }
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    tool_call.tool_call_id.clone(),
                    fields,
                )))
            }
            crate::cli::CursorCliEvent::Result(result) => {
                if result.text.trim().is_empty() {
                    None
                } else {
                    Some(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        result.text.clone().into(),
                    )))
                }
            }
            crate::cli::CursorCliEvent::Other(_) => None,
        }
    }

    fn allow_edits_for_session(
        session: &SessionState,
        edit_enabled_sessions: &HashSet<agent_client_protocol::SessionId>,
    ) -> bool {
        let explicitly_enabled = edit_enabled_sessions.contains(&session.id);
        let mode_allows = session.current_mode.0.as_ref() == "agent";
        let config_force = session
            .config_options
            .iter()
            .find(|option| option.id.0.as_ref() == "cursor/edit_mode")
            .map(|option| match &option.kind {
                SessionConfigKind::Select(select) => select.current_value.0.as_ref() == "force",
                _ => false,
            });
        explicitly_enabled && (mode_allows || config_force.unwrap_or(false))
    }

    fn is_generic_edit_message(message: &str) -> bool {
        let lower = message.to_lowercase();
        (lower.contains("the file ")
            && (lower.contains(" has been updated.") || lower.contains(" was updated.")))
            || lower.contains("wrote contents to ")
    }

    fn workspace_trusted_from_meta(
        meta: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Option<bool> {
        meta.and_then(|meta| meta.get("workspace_trusted"))
            .and_then(serde_json::Value::as_bool)
    }

    fn cwd_from_meta(meta: Option<&serde_json::Map<String, serde_json::Value>>) -> Option<PathBuf> {
        let keys = [
            "cwd",
            "working_directory",
            "workspace_root",
            "workspaceRoot",
        ];
        for key in keys {
            let value = meta
                .and_then(|meta| meta.get(key))
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from);
            if let Some(path) = value
                && path.is_absolute()
            {
                return Some(path);
            }
        }
        None
    }

    fn normalize_path_for_compare(path: &Path) -> PathBuf {
        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                std::path::Component::CurDir => {}
                _ => normalized.push(component.as_os_str()),
            }
        }
        if normalized.as_os_str().is_empty() {
            path.to_path_buf()
        } else {
            normalized
        }
    }

    fn paths_match(filter: &Path, candidate: &Path) -> bool {
        if filter == candidate {
            return true;
        }
        let normalized_filter = Self::normalize_path_for_compare(filter);
        let normalized_candidate = Self::normalize_path_for_compare(candidate);
        if normalized_filter == normalized_candidate {
            return true;
        }
        match (
            std::fs::canonicalize(&normalized_filter),
            std::fs::canonicalize(&normalized_candidate),
        ) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    }

    fn parse_cursor_offset(cursor: Option<&str>) -> usize {
        let Some(raw) = cursor else {
            return 0;
        };
        let value = raw.strip_prefix("offset:").unwrap_or(raw);
        value.parse::<usize>().unwrap_or(0)
    }

    fn next_cursor_for(offset: usize, total: usize) -> Option<String> {
        if offset >= total {
            None
        } else {
            Some(format!("offset:{offset}"))
        }
    }

    fn cli_mode_for_session(session: &SessionState) -> Option<String> {
        match session.current_mode.0.as_ref() {
            "ask" => Some("ask".to_string()),
            "plan" => Some("plan".to_string()),
            _ => None,
        }
    }

    fn cli_sandbox_for_session(session: &SessionState) -> Option<String> {
        match session.current_mode.0.as_ref() {
            "ask" | "plan" => Some("enabled".to_string()),
            _ => None,
        }
    }

    fn format_session_title_from_text(text: &str) -> Option<String> {
        let normalized = text.replace(['\r', '\n'], " ");
        let trimmed = normalized.trim();
        if trimmed.is_empty() {
            return None;
        }
        const MAX_CHARS: usize = 72;
        if trimmed.chars().count() <= MAX_CHARS {
            return Some(trimmed.to_string());
        }
        let mut out = trimmed
            .chars()
            .take(MAX_CHARS.saturating_sub(3))
            .collect::<String>();
        out.push_str("...");
        Some(out)
    }

    fn managed_edits_enabled() -> bool {
        let raw = std::env::var("CURSOR_ACP_MANAGED_EDITS")
            .unwrap_or_else(|_| "1".to_string())
            .to_lowercase();
        !matches!(raw.as_str(), "0" | "false" | "off" | "no")
    }

    fn serialize_session_updates_for_history(
        updates: Vec<SessionUpdate>,
    ) -> Vec<serde_json::Value> {
        updates
            .into_iter()
            .filter_map(|update| serde_json::to_value(update).ok())
            .collect()
    }

    fn plan_from_todo_items(todo_items: &[crate::cli::CursorTodoItem]) -> Option<Plan> {
        let mut entries = Vec::new();
        for todo in todo_items {
            let status = match todo.status.as_str() {
                "TODO_STATUS_COMPLETED" => PlanEntryStatus::Completed,
                "TODO_STATUS_IN_PROGRESS" => PlanEntryStatus::InProgress,
                _ => PlanEntryStatus::Pending,
            };
            if !todo.content.trim().is_empty() {
                entries.push(PlanEntry::new(
                    todo.content.clone(),
                    PlanEntryPriority::Medium,
                    status,
                ));
            }
        }
        if entries.is_empty() {
            None
        } else {
            Some(Plan::new(entries))
        }
    }

    async fn apply_managed_edit_if_present(
        client: &AgentSideConnection,
        session_id: &agent_client_protocol::SessionId,
        event: &crate::cli::CursorCliEvent,
        applied_calls: &Arc<Mutex<HashSet<String>>>,
    ) -> Result<(), Error> {
        let crate::cli::CursorCliEvent::ToolCallUpdate(tool_call) = event else {
            return Ok(());
        };
        let tool_call = tool_call.as_ref();
        if tool_call.status != ToolCallStatus::Completed
            || tool_call.kind != agent_client_protocol::ToolKind::Edit
        {
            return Ok(());
        }
        let Some(path) = tool_call.locations.first() else {
            return Ok(());
        };
        let Some(content) = &tool_call.managed_write_text else {
            return Ok(());
        };
        let key = format!("{}::{}", session_id, tool_call.tool_call_id);
        {
            let mut applied = applied_calls.lock().unwrap();
            if applied.contains(&key) {
                return Ok(());
            }
            applied.insert(key);
        }
        client
            .write_text_file(WriteTextFileRequest::new(
                session_id.clone(),
                path.clone(),
                content.clone(),
            ))
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl Agent for CursorAcpAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        *self.client_capabilities.lock().unwrap() = request.client_capabilities;

        let mut capabilities = AgentCapabilities::new()
            .load_session(true)
            .prompt_capabilities(
                PromptCapabilities::new()
                    .image(true)
                    .audio(false)
                    .embedded_context(true),
            )
            .mcp_capabilities(McpCapabilities::new().http(false).sse(false));
        capabilities.session_capabilities =
            SessionCapabilities::new().list(SessionListCapabilities::new());

        let binary = std::env::var("CURSOR_AGENT_BIN")
            .or_else(|_| std::env::var("CURSOR_AGENT_PATH"))
            .unwrap_or_else(|_| "cursor-agent".to_string());

        Ok(InitializeResponse::new(ProtocolVersion::V1)
            .agent_capabilities(capabilities)
            .agent_info(
                Implementation::new("cursor-acp", env!("CARGO_PKG_VERSION")).title("Cursor ACP"),
            )
            .auth_methods(vec![terminal_auth_method(&binary)]))
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        if request.method_id.0.as_ref() != AUTH_METHOD_CURSOR_LOGIN {
            let mut error = Error::invalid_params();
            error.message = format!("Unsupported auth method: {}", request.method_id);
            return Err(error);
        }
        let auth = self
            .cli_runtime
            .check_auth()
            .await
            .map_err(Self::map_error)?;
        if !auth.authenticated {
            return Err(Error::auth_required().data(serde_json::json!({
                "reason": "auth_required",
                "hint": "Run cursor-agent login",
                "raw_output": auth.raw_output,
            })));
        }
        Ok(AuthenticateResponse::new())
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        let workspace_trusted =
            Self::workspace_trusted_from_meta(request.meta.as_ref()).unwrap_or(true);
        let cwd = request.cwd;
        let listed_models = self.cli_runtime.list_models().await.ok();
        let mut store = self.session_store.lock().unwrap();
        let session = store.create_session(cwd, workspace_trusted);
        self.edit_enabled_sessions
            .lock()
            .unwrap()
            .remove(&session.id);

        if let Some(model_ids) = listed_models {
            store.update_models(model_ids);
        }

        let modes = Some(store.mode_state(&session));
        let models = Some(store.model_state(&session));

        Ok(NewSessionResponse::new(session.id)
            .modes(modes)
            .models(models))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        let workspace_trusted = Self::workspace_trusted_from_meta(request.meta.as_ref());
        let listed_models = self.cli_runtime.list_models().await.ok();
        let (modes, models, replay_history_updates, replay_turns, session_id) = {
            let mut store = self.session_store.lock().unwrap();
            if let Some(model_ids) = listed_models {
                store.update_models(model_ids);
            }
            let mut session = store
                .get_session(&request.session_id)
                .cloned()
                .ok_or_else(|| Error::resource_not_found(None))?;
            if let Some(workspace_trusted) = workspace_trusted {
                store.set_workspace_trusted(&request.session_id, workspace_trusted);
                if let Some(updated) = store.get_session(&request.session_id) {
                    session = updated.clone();
                }
            }
            self.edit_enabled_sessions
                .lock()
                .unwrap()
                .remove(&session.id);
            (
                store.mode_state(&session),
                store.model_state(&session),
                session.history_updates.clone(),
                session.turns.clone(),
                session.id.clone(),
            )
        };

        if replay_history_updates.is_empty() {
            for turn in replay_turns {
                if !turn.user.trim().is_empty() {
                    self.notify_session_update(
                        session_id.clone(),
                        SessionUpdate::UserMessageChunk(ContentChunk::new(turn.user.into())),
                    )
                    .await?;
                }
                if !turn.assistant.trim().is_empty() {
                    self.notify_session_update(
                        session_id.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(turn.assistant.into())),
                    )
                    .await?;
                }
            }
        } else {
            for value in replay_history_updates {
                if let Ok(update) = serde_json::from_value::<SessionUpdate>(value) {
                    self.notify_session_update(session_id.clone(), update)
                        .await?;
                }
            }
        }

        Ok(LoadSessionResponse::new()
            .modes(Some(modes))
            .models(Some(models)))
    }

    async fn list_sessions(
        &self,
        request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        let store = self.session_store.lock().unwrap();
        let mut sessions = store.list_sessions();
        let filter_cwd = request
            .cwd
            .clone()
            .or_else(|| Self::cwd_from_meta(request.meta.as_ref()));
        if let Some(filter_cwd) = filter_cwd.as_ref() {
            sessions.retain(|session| Self::paths_match(filter_cwd, &session.cwd));
        }
        sessions.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.id.0.cmp(&left.id.0))
        });

        let total = sessions.len();
        let start = Self::parse_cursor_offset(request.cursor.as_deref()).min(total);
        let end = (start + Self::LIST_SESSIONS_PAGE_SIZE).min(total);
        let next_cursor = Self::next_cursor_for(end, total);

        let mut infos = Vec::new();
        for session in sessions.into_iter().skip(start).take(end - start) {
            let title = session.title.clone().or_else(|| {
                session
                    .cwd
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_string())
            });
            let updated_at = DateTime::<Utc>::from(session.updated_at).to_rfc3339();
            infos.push(
                SessionInfo::new(session.id.clone(), session.cwd.clone())
                    .title(title)
                    .updated_at(Some(updated_at)),
            );
        }
        Ok(ListSessionsResponse::new(infos).next_cursor(next_cursor))
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        let session = {
            let store = self.session_store.lock().unwrap();
            store
                .get_session(&request.session_id)
                .ok_or_else(|| Error::resource_not_found(None))?
                .clone()
        };
        let session_id = session.id.clone();

        let prompt_text = Self::flatten_prompt(&request.prompt);
        let mut persisted_updates = Vec::new();
        if !prompt_text.trim().is_empty() {
            persisted_updates.push(SessionUpdate::UserMessageChunk(ContentChunk::new(
                prompt_text.clone().into(),
            )));
        }
        if let Some(title) = Self::format_session_title_from_text(&prompt_text) {
            let should_notify = {
                let mut store = self.session_store.lock().unwrap();
                let existing_title = store
                    .get_session(&session_id)
                    .and_then(|value| value.title.clone());
                let has_title = existing_title
                    .as_deref()
                    .map(|value| !value.trim().is_empty())
                    .unwrap_or(false);
                if !has_title {
                    store.set_title(&session_id, Some(title.clone()));
                    true
                } else {
                    false
                }
            };
            if should_notify {
                let info_update =
                    SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(Some(title)));
                {
                    let serialized =
                        Self::serialize_session_updates_for_history(vec![info_update.clone()]);
                    let mut store = self.session_store.lock().unwrap();
                    store.append_history_updates(&session_id, serialized);
                }
                self.notify_session_update(session_id.clone(), info_update)
                    .await?;
            }
        }
        let trust_workspace = Self::workspace_trusted_from_meta(request.meta.as_ref())
            .unwrap_or(session.workspace_trusted);
        let edit_enabled_sessions = self.edit_enabled_sessions.lock().unwrap().clone();
        let allow_edits = Self::allow_edits_for_session(&session, &edit_enabled_sessions);
        let cli_mode = Self::cli_mode_for_session(&session);
        let cli_sandbox = Self::cli_sandbox_for_session(&session);
        let managed_edits = Self::managed_edits_enabled();
        let cli_allow_edits = allow_edits && !managed_edits;
        let prompt_options = PromptRunOptions {
            prompt_text: prompt_text.clone(),
            mode: cli_mode.clone(),
            sandbox: cli_sandbox.clone(),
            allow_edits: cli_allow_edits,
            trust_workspace,
        };
        let streaming_requested = request
            .meta
            .as_ref()
            .and_then(|meta| meta.get("stream"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let result = if streaming_requested {
            let (event_tx, mut event_rx) = mpsc::unbounded_channel();
            let client = self.client_connection.lock().unwrap().clone();
            let notify_session_id = session_id.clone();
            let applied_calls = self.applied_managed_edit_calls.clone();
            let managed_edits_enabled = managed_edits;
            let allow_managed_edits = allow_edits;
            let notify_task = tokio::task::spawn_local(async move {
                while let Some(event) = event_rx.recv().await {
                    if let Some(client) = &client {
                        if managed_edits_enabled
                            && allow_managed_edits
                            && let Err(error) = CursorAcpAgent::apply_managed_edit_if_present(
                                client,
                                &notify_session_id,
                                &event,
                                &applied_calls,
                            )
                            .await
                        {
                            warn!("managed edit write failed: {error}");
                        }
                        if let Some(update) = CursorAcpAgent::event_to_session_update(&event) {
                            client
                                .session_notification(SessionNotification::new(
                                    notify_session_id.clone(),
                                    update,
                                ))
                                .await?;
                        }
                        if let crate::cli::CursorCliEvent::ToolCallUpdate(tool_call) = &event
                            && let Some(todo_items) = tool_call.todo_items.as_ref()
                            && let Some(plan) = CursorAcpAgent::plan_from_todo_items(todo_items)
                        {
                            client
                                .session_notification(SessionNotification::new(
                                    notify_session_id.clone(),
                                    SessionUpdate::Plan(plan),
                                ))
                                .await?;
                        }
                        if let crate::cli::CursorCliEvent::Result(result) = &event
                            && let Some(usage) = result.usage
                        {
                            client
                                .session_notification(SessionNotification::new(
                                    notify_session_id.clone(),
                                    SessionUpdate::UsageUpdate(UsageUpdate::new(
                                        usage.used, usage.size,
                                    )),
                                ))
                                .await?;
                        }
                    }
                }
                Ok::<(), Error>(())
            });

            let streamed = self
                .prompt_engine
                .run_prompt_stream_events(&session, prompt_options.clone(), Some(event_tx))
                .await;
            let notify_result = notify_task.await.map_err(|error| {
                Error::internal_error().data(serde_json::json!({
                    "reason": "stream_notify_join_failed",
                    "detail": error.to_string(),
                }))
            })?;
            notify_result?;

            if let Ok((events, stop_reason)) = &streamed {
                let mut assistant_output = String::new();
                for event in events {
                    if let Some(update) = Self::event_to_session_update(event) {
                        persisted_updates.push(update);
                    }
                    if let crate::cli::CursorCliEvent::ToolCallUpdate(tool_call) = event
                        && let Some(todo_items) = tool_call.todo_items.as_ref()
                        && let Some(plan) = Self::plan_from_todo_items(todo_items)
                    {
                        persisted_updates.push(SessionUpdate::Plan(plan));
                    }
                    if let crate::cli::CursorCliEvent::AssistantDelta(text) = event {
                        assistant_output.push_str(text);
                    }
                    if let crate::cli::CursorCliEvent::Result(result) = event
                        && let Some(resume_id) = result
                            .meta
                            .get("session_id")
                            .and_then(serde_json::Value::as_str)
                    {
                        let mut store = self.session_store.lock().unwrap();
                        store.set_resume_id(&session_id, Some(resume_id.to_string()));
                        if !result.text.trim().is_empty() {
                            assistant_output = result.text.clone();
                        }
                        if let Some(usage) = result.usage {
                            persisted_updates.push(SessionUpdate::UsageUpdate(UsageUpdate::new(
                                usage.used, usage.size,
                            )));
                        }
                    }
                }
                if !prompt_text.trim().is_empty() || !assistant_output.trim().is_empty() {
                    let mut store = self.session_store.lock().unwrap();
                    store.append_turn(&session_id, prompt_text.clone(), assistant_output);
                }
                Ok((String::new(), *stop_reason))
            } else {
                streamed.map(|(_, stop_reason)| (String::new(), stop_reason))
            }
        } else {
            let (result, stop_reason) = self
                .prompt_engine
                .run_prompt_non_stream(&session, prompt_options)
                .await
                .map_err(Self::map_error)?;
            if let Some(usage) = result.usage {
                persisted_updates.push(SessionUpdate::UsageUpdate(UsageUpdate::new(
                    usage.used, usage.size,
                )));
                self.notify_session_update(
                    session_id.clone(),
                    SessionUpdate::UsageUpdate(UsageUpdate::new(usage.used, usage.size)),
                )
                .await?;
            }
            Ok((result.text, stop_reason))
        };

        match result {
            Ok((text, stop_reason)) => {
                if !text.trim().is_empty() {
                    persisted_updates.push(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        text.clone().into(),
                    )));
                }
                if !persisted_updates.is_empty() {
                    let serialized = Self::serialize_session_updates_for_history(persisted_updates);
                    let mut store = self.session_store.lock().unwrap();
                    store.append_history_updates(&session_id, serialized);
                }
                if !prompt_text.trim().is_empty() || !text.trim().is_empty() {
                    let mut store = self.session_store.lock().unwrap();
                    store.append_turn(&session_id, prompt_text.clone(), text.clone());
                }
                if !text.is_empty() {
                    self.notify_session_update(
                        session_id,
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(text.into())),
                    )
                    .await?;
                    Ok(PromptResponse::new(stop_reason))
                } else {
                    Ok(PromptResponse::new(stop_reason))
                }
            }
            Err(error) => {
                let stop_reason = PromptEngine::classify_failure_stop_reason(&error);
                warn!("prompt failed with stop_reason={stop_reason:?}: {error}");
                let failure_update = SessionUpdate::AgentMessageChunk(ContentChunk::new(
                    format!("Request failed: {error}").into(),
                ));
                persisted_updates.push(failure_update.clone());
                let serialized = Self::serialize_session_updates_for_history(persisted_updates);
                {
                    let mut store = self.session_store.lock().unwrap();
                    store.append_history_updates(&session.id, serialized);
                }
                self.notify_session_update(session.id.clone(), failure_update)
                    .await?;
                Ok(PromptResponse::new(stop_reason))
            }
        }
    }

    async fn cancel(&self, _args: CancelNotification) -> Result<(), Error> {
        Ok(())
    }

    async fn set_session_mode(
        &self,
        request: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        let mode_id = request.mode_id.clone();
        let ok = {
            let mut store = self.session_store.lock().unwrap();
            store.set_mode(&request.session_id, mode_id.clone())
        };
        if !ok {
            let mut error = Error::invalid_params();
            error.message = "Unknown mode or session".to_string();
            return Err(error);
        }

        {
            let mut gate = self.edit_enabled_sessions.lock().unwrap();
            if mode_id.0.as_ref() == "agent" {
                gate.insert(request.session_id.clone());
            } else {
                gate.remove(&request.session_id);
            }
        }

        self.notify_session_update(
            request.session_id.clone(),
            SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id.clone())),
        )
        .await?;
        {
            let serialized = Self::serialize_session_updates_for_history(vec![
                SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id)),
            ]);
            let mut store = self.session_store.lock().unwrap();
            store.append_history_updates(&request.session_id, serialized);
        }
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        request: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        let mut store = self.session_store.lock().unwrap();
        if let Err(available) = store.set_model(&request.session_id, request.model_id.clone()) {
            return Err(Self::map_error(ErrorKind::ModelUnsupported {
                requested: request.model_id.to_string(),
                available,
            }));
        }
        Ok(SetSessionModelResponse::default())
    }

    async fn set_session_config_option(
        &self,
        request: SetSessionConfigOptionRequest,
    ) -> Result<SetSessionConfigOptionResponse, Error> {
        let mut store = self.session_store.lock().unwrap();
        let session = store
            .get_session_mut(&request.session_id)
            .ok_or_else(|| Error::resource_not_found(None))?;

        if request.config_id.0.as_ref() != "cursor/edit_mode" {
            let mut error = Error::invalid_params();
            error.message = "Unknown config option".to_string();
            return Err(error);
        }
        if request.value.0.as_ref() != "read_only" && request.value.0.as_ref() != "force" {
            let mut error = Error::invalid_params();
            error.message = "Unknown config option value".to_string();
            return Err(error);
        }

        let option = SessionConfigOption::select(
            "cursor/edit_mode",
            "Edit mode",
            request.value.clone(),
            SessionConfigSelectOptions::Ungrouped(vec![
                SessionConfigSelectOption::new("read_only", "Read Only"),
                SessionConfigSelectOption::new("force", "Force"),
            ]),
        );
        session.config_options = vec![option.clone()];

        Ok(SetSessionConfigOptionResponse::new(vec![option]))
    }
}

#[cfg(test)]
mod tests {
    use super::CursorAcpAgent;
    use std::path::Path;

    #[test]
    fn generic_edit_message_detects_wrote_contents_variant() {
        assert!(CursorAcpAgent::is_generic_edit_message(
            "Wrote contents to /Users/cameron/Code/chia-wallet-sdk/jokes2.txt"
        ));
    }

    #[test]
    fn paths_match_ignores_trailing_slash_and_dot_segments() {
        assert!(CursorAcpAgent::paths_match(
            Path::new("/tmp/example/"),
            Path::new("/tmp/example")
        ));
        assert!(CursorAcpAgent::paths_match(
            Path::new("/tmp/example/./src"),
            Path::new("/tmp/example/src")
        ));
    }

    #[test]
    fn cursor_offset_parser_handles_prefixed_and_plain_offsets() {
        assert_eq!(CursorAcpAgent::parse_cursor_offset(None), 0);
        assert_eq!(CursorAcpAgent::parse_cursor_offset(Some("offset:12")), 12);
        assert_eq!(CursorAcpAgent::parse_cursor_offset(Some("7")), 7);
        assert_eq!(CursorAcpAgent::parse_cursor_offset(Some("nonsense")), 0);
    }
}
