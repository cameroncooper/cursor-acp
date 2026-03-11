use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::oneshot;

fn perm_log(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/cursor-acp-permissions.log")
    {
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        drop(writeln!(f, "[{ts}] {msg}"));
    }
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
}

/// Tracks state needed for interception (session ID, accumulated todos, models).
pub struct ProxyState {
    current_session_id: Option<String>,
    todos: Vec<TodoItem>,
    /// Last `createPlan` markdown per session ID (Zed session ID).
    plans: HashMap<String, PlanInfo>,
    /// The absolute cwd for each Zed session ID (best-effort).
    pub session_cwds: HashMap<String, String>,
    /// The last plan markdown file path emitted per session.
    plan_file_paths: HashMap<String, String>,
    /// Whether to emit plan markdown into a workspace file via `fs/write_text_file`.
    pub emit_plan_files: bool,
    /// Whether to also emit a chat message with a `file:///` link.
    pub emit_plan_file_messages: bool,
    /// Whether to try to link to Cursor's own plan markdown file (if it already wrote one).
    pub link_cursor_plan_files: bool,
    /// Accumulated assistant message text per session for markdown plan detection.
    plan_detection_buffers: HashMap<String, String>,
    /// Last derived markdown emitted from assistant-message detection per session.
    last_detected_plan_markdown: HashMap<String, String>,
    pub models: Vec<ModelInfo>,
    pub selected_model: Option<String>,
    /// The resolved `cursor-agent` binary path (or name) used to spawn the child.
    /// Used to provide a reliable login command to Zed via auth method metadata.
    pub agent_binary: Option<String>,
    pub stored_init_request: Option<String>,
    pub stored_auth_request: Option<String>,
    /// Response IDs to suppress (from replayed init/auth after child restart).
    suppress_response_ids: Vec<Value>,
    replay_id_counter: i64,
    /// The `id` of the original `initialize` request so we can patch the response.
    init_request_id: Option<Value>,
    /// Maps `session/new` request IDs to their `cwd` param for session tracking.
    pub pending_new_session_cwds: HashMap<String, String>,
    /// Stores modes/models from the most recent `session/new` response for use in
    /// `session/load` responses.
    pub last_session_modes: Option<Value>,
    pub last_session_models: Option<Value>,
    /// The session ID that Zed is using (from `session/prompt`), so we can
    /// remap child session IDs back after restarts.
    pub zed_session_id: Option<String>,
    /// The session ID the child process currently has active.
    /// Normally the same as `zed_session_id`, but differs after `session/load`
    /// (where we create a fresh child session while Zed uses the loaded ID).
    pub child_session_id: Option<String>,
    /// Sessions discovered from `session/new` responses but not yet persisted.
    /// Creation is deferred until the user actually sends a `session/prompt`.
    pub pending_sessions: HashMap<String, String>,
    /// Response IDs from Zed that should be silently dropped (for our synthesized
    /// `fs/read_text_file` and `fs/write_text_file` requests).
    suppress_zed_response_ids: Vec<Value>,
    /// Counter for generating unique negative IDs for internal proxy requests.
    internal_id_counter: i64,
    /// One-shot waiters keyed by the JSON-RPC request id string for internal
    /// `session/new` calls (used during `session/load` to ensure the child
    /// session exists before Zed starts prompting).
    pending_session_new_waiters: HashMap<String, oneshot::Sender<String>>,
    /// Pre-formatted conversation history to inject into the next `session/prompt`
    /// for a loaded session so the child agent gets context after a restart.
    pending_history_injection: HashMap<String, String>,
    /// The first user prompt per session, used as a fallback plan file name
    /// when no explicit plan name is available.
    session_first_prompt: HashMap<String, String>,
    /// Sessions for which we've already emitted the "Plan markdown saved to …"
    /// chat message. Prevents re-injecting the notification on every streaming chunk.
    plan_file_message_emitted: HashSet<String>,
    /// The workspace cwd for this proxy instance, learned from the first
    /// `session/new` request. Used to scope `session/list` to this workspace
    /// when the client doesn't provide a `cwd` filter.
    pub workspace_cwd: Option<String>,

    // --- Terminal synthesis for execute tool calls ---
    /// Maps tool_call_id → generated terminal UUID for execute tool calls
    /// that we've already patched with terminal_info.
    terminal_ids: HashMap<String, String>,
    /// Buffered first `tool_call` messages for execute tool calls that arrived
    /// without a command yet. Key is tool_call_id, value is the original
    /// JSON-RPC line. Once the command arrives, we discard the buffer and
    /// forward only the patched version.
    buffered_execute_tool_calls: HashMap<String, Value>,

    // --- PTY streaming ---
    /// Pending matches: command substring → (terminal_id, tool_call_id, session_id).
    /// Populated when we see a tool_call with kind=execute and a command.
    /// Consumed when a PTY stream spawn arrives with a matching command.
    pub pty_pending_matches: Vec<PtyStreamMatch>,
    /// Active streaming PTYs: child PID → terminal routing info.
    pub pty_stream_pids: HashMap<i32, PtyStreamMatch>,
    /// Terminal IDs that received streaming output (skip batch output on completion).
    pub pty_streamed_terminals: HashSet<String>,

    // --- Permission caching ---
    /// Tool kinds (e.g. "execute", "edit") for which the user has chosen
    /// "allow-always". When a new permission request arrives for a cached kind,
    /// the proxy auto-responds without prompting Zed.
    pub allowed_tool_kinds: HashSet<String>,
    /// Maps JSON-RPC request IDs for in-flight `session/request_permission`
    /// requests to (tool_kind, allow_always_option_id) so we can recognize
    /// allow-always responses when they come back from Zed.
    pending_permission_requests: HashMap<String, PendingPermission>,
}

#[derive(Clone, Debug)]
pub struct PtyStreamMatch {
    pub terminal_id: String,
    pub tool_call_id: String,
    pub session_id: String,
    pub command: String,
}

#[derive(Clone, Debug)]
struct PendingPermission {
    tool_kind: String,
    allow_always_option_id: Option<String>,
}

struct TodoItem {
    id: String,
    content: String,
    status: String,
}

#[derive(Clone, Debug)]
struct PlanInfo {
    markdown: String,
    name: Option<String>,
}

#[derive(Debug)]
pub enum AgentAction {
    Forward,
    Intercept {
        response_to_child: Option<String>,
        notifications_to_zed: Vec<String>,
    },
    /// Forward with modifications (e.g., inject models into session/new response).
    ForwardPatched(String),
    /// Forward the original line AND send additional notifications to Zed.
    ForwardWithExtra {
        line: String,
        extra_notifications: Vec<String>,
    },
    /// Forward the patched line AND spawn a local subprocess to stream terminal
    /// output to Zed in real-time (instead of waiting for cursor-agent's batch).
    #[allow(dead_code)]
    SpawnStreaming {
        line: String,
        command: String,
        cwd: Option<String>,
        terminal_id: String,
        tool_call_id: String,
        session_id: String,
    },
    /// Drop the message entirely (e.g., buffered execute tool call awaiting command).
    Drop,
}

pub enum ClientAction {
    /// Forward the message as-is.
    Forward,
    /// Forward a patched version of the message.
    ForwardPatched(String),
    /// Forward the message and record the user's prompt for session history/title.
    ForwardWithPrompt { line: String, prompt_text: String },
    /// Intercept the message: respond to Zed and optionally restart the child.
    Respond {
        response_to_zed: String,
        restart_with_model: Option<String>,
    },
    /// Drop the message entirely (don't forward to child).
    Drop,
}

impl ProxyState {
    pub fn new() -> Self {
        Self {
            current_session_id: None,
            todos: Vec::new(),
            plans: HashMap::new(),
            session_cwds: HashMap::new(),
            plan_file_paths: HashMap::new(),
            emit_plan_files: env_flag_or_default("CURSOR_ACP_WRITE_PLAN_FILE", true),
            emit_plan_file_messages: env_flag_or_default(
                "CURSOR_ACP_WRITE_PLAN_FILE_MESSAGE",
                true,
            ),
            link_cursor_plan_files: env_flag_or_default("CURSOR_ACP_LINK_CURSOR_PLAN_FILE", false),
            plan_detection_buffers: HashMap::new(),
            last_detected_plan_markdown: HashMap::new(),
            models: Vec::new(),
            selected_model: None,
            agent_binary: None,
            stored_init_request: None,
            stored_auth_request: None,
            suppress_response_ids: Vec::new(),
            replay_id_counter: -1000,
            init_request_id: None,
            pending_new_session_cwds: HashMap::new(),
            last_session_modes: None,
            last_session_models: None,
            zed_session_id: None,
            child_session_id: None,
            pending_sessions: HashMap::new(),
            suppress_zed_response_ids: Vec::new(),
            internal_id_counter: -20000,
            pending_session_new_waiters: HashMap::new(),
            pending_history_injection: HashMap::new(),
            session_first_prompt: HashMap::new(),
            plan_file_message_emitted: HashSet::new(),
            workspace_cwd: None,
            terminal_ids: HashMap::new(),
            buffered_execute_tool_calls: HashMap::new(),
            pty_pending_matches: Vec::new(),
            pty_stream_pids: HashMap::new(),
            pty_streamed_terminals: HashSet::new(),
            allowed_tool_kinds: HashSet::new(),
            pending_permission_requests: HashMap::new(),
        }
    }

    /// Register a response ID to suppress (the response will be silently dropped).
    pub fn suppress_response(&mut self, id: Value) {
        self.suppress_response_ids.push(id);
    }

    /// Generate a unique negative ID for internal proxy requests.
    pub fn next_internal_id(&mut self) -> Value {
        self.internal_id_counter -= 1;
        json!(self.internal_id_counter)
    }

    /// Register a waiter that will be fulfilled when the child responds to the
    /// internal `session/new` request with the same id.
    pub fn register_session_new_waiter(&mut self, id: &Value) -> oneshot::Receiver<String> {
        let (tx, rx) = oneshot::channel();
        self.pending_session_new_waiters.insert(id.to_string(), tx);
        rx
    }

    /// Stash a formatted history blob to inject into the next prompt for `session_id`.
    pub fn set_pending_history_injection(&mut self, session_id: &str, history: String) {
        if !history.trim().is_empty() {
            self.pending_history_injection
                .insert(session_id.to_string(), history);
        }
    }

    fn take_pending_history_injection(&mut self, session_id: &str) -> Option<String> {
        self.pending_history_injection.remove(session_id)
    }

    /// Register a Zed response ID to suppress (for our synthesized fs requests).
    fn suppress_zed_response(&mut self, id: Value) {
        self.suppress_zed_response_ids.push(id);
    }

    /// Prepare init/auth requests for replay after a child restart.
    /// Rewrites their IDs to unique values and registers them for suppression.
    pub fn prepare_replay_messages(&mut self) -> Vec<String> {
        let mut messages = Vec::new();

        for stored in [&self.stored_init_request, &self.stored_auth_request] {
            if let Some(text) = stored
                && let Ok(mut msg) = serde_json::from_str::<Value>(text)
            {
                self.replay_id_counter -= 1;
                let replay_id = json!(self.replay_id_counter);
                msg["id"] = replay_id.clone();
                self.suppress_response_ids.push(replay_id);
                messages.push(msg.to_string());
            }
        }

        messages
    }
}

fn env_flag_or_default(key: &str, default: bool) -> bool {
    parse_env_flag(std::env::var(key).ok().as_deref(), default)
}

fn parse_env_flag(raw: Option<&str>, default: bool) -> bool {
    match raw {
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        None => default,
    }
}

// ---------------------------------------------------------------------------
// Client → Agent direction
// ---------------------------------------------------------------------------

/// Inspect a JSON-RPC message from Zed and decide what to do with it.
pub fn process_client_message(msg: &Value, state: &mut ProxyState) -> ClientAction {
    // Broad logging: capture every Zed message for debugging permissions.
    if let Some(method) = msg.get("method").and_then(Value::as_str) {
        let id = msg.get("id").map(|v| v.to_string()).unwrap_or_default();
        perm_log(&format!(
            "ZED MSG method={method} id={id} has_params={}",
            msg.get("params").is_some()
        ));
    } else if msg.get("result").is_some() || msg.get("error").is_some() {
        let id = msg.get("id").map(|v| v.to_string()).unwrap_or_default();
        let has_option = msg.pointer("/result/outcome/optionId").is_some();
        perm_log(&format!(
            "ZED MSG response id={id} has_result={} has_error={} has_optionId={has_option}",
            msg.get("result").is_some(),
            msg.get("error").is_some()
        ));
    }

    // Check if this is a permission response and cache allow-always decisions.
    maybe_cache_permission_response(msg, state);

    // Suppress responses to our synthesized fs/ requests.
    if let Some(id) = msg.get("id")
        && (msg.get("result").is_some() || msg.get("error").is_some())
        && let Some(pos) = state.suppress_zed_response_ids.iter().position(|s| s == id)
    {
        state.suppress_zed_response_ids.swap_remove(pos);
        tracing::debug!(?id, "suppressed Zed response to internal fs request");
        return ClientAction::Drop;
    }

    let method = msg.get("method").and_then(Value::as_str);

    match method {
        Some("initialize") => {
            state.init_request_id = msg.get("id").cloned();
            state.stored_init_request = Some(strip_meta_from_init(msg));
            ClientAction::ForwardPatched(state.stored_init_request.clone().unwrap())
        }
        Some("authenticate") => {
            let line = msg.to_string();
            state.stored_auth_request = Some(line.clone());
            ClientAction::ForwardPatched(line)
        }
        Some("session/new") => {
            let cwd = msg
                .pointer("/params/cwd")
                .and_then(|v| v.as_str().map(String::from));

            if let (Some(id), Some(cwd)) = (msg.get("id").map(|v| v.to_string()), cwd) {
                state.pending_new_session_cwds.insert(id, cwd.to_string());
                if state.workspace_cwd.is_none() {
                    state.workspace_cwd = Some(cwd);
                }
            }
            ClientAction::Forward
        }
        Some("session/prompt") => {
            if let Some(session_id) = msg.pointer("/params/sessionId").and_then(Value::as_str) {
                state.zed_session_id = Some(session_id.to_string());
                state.plan_detection_buffers.remove(session_id);
                state.last_detected_plan_markdown.remove(session_id);
            }
            let mut patched = msg.clone();
            if let Some(session_id) = patched.pointer("/params/sessionId").and_then(Value::as_str)
                && let Some(history) = state.take_pending_history_injection(session_id)
                && let Some(arr) = patched
                    .pointer_mut("/params/prompt")
                    .and_then(Value::as_array_mut)
            {
                arr.insert(
                    0,
                    json!({
                        "type": "text",
                        "text": format!(
                            "Context from previous conversation (restored by host). Use as background; do not quote verbatim unless asked.\n\n{}",
                            history
                        )
                    }),
                );
            }

            // Detect plan mode and inject todos formatting instructions
            if let Some(prompt_text) = extract_prompt_text(&patched)
                && prompt_text.to_lowercase().contains("plan mode")
                && let Some(arr) = patched
                    .pointer_mut("/params/prompt")
                    .and_then(Value::as_array_mut)
            {
                arr.push(json!({
                    "type": "text",
                    "text": "<system_reminder>\nWhen calling CreatePlan, include todos as a JSON array in a fenced code block at the end of your plan markdown:\n\n```todos\n[{\"id\": \"task-1\", \"content\": \"First task\", \"status\": \"pending\"}, {\"id\": \"task-2\", \"content\": \"Second task\", \"status\": \"pending\"}]\n```\n\nThis ensures todos are preserved correctly. The `id` should be a short kebab-case identifier, `content` is the task description, and `status` should be \"pending\", \"in_progress\", or \"completed\".\n</system_reminder>"
                }));
            }

            let forwarded = remap_client_session_id(&patched, state);
            if let Some(prompt_text) = extract_prompt_text(msg) {
                if let Some(sid) = state.zed_session_id.as_deref() {
                    state
                        .session_first_prompt
                        .entry(sid.to_string())
                        .or_insert_with(|| prompt_text.clone());
                }
                ClientAction::ForwardWithPrompt {
                    line: forwarded,
                    prompt_text,
                }
            } else {
                ClientAction::ForwardPatched(forwarded)
            }
        }
        Some("session/set_model") => handle_set_model(msg, state),
        _ => {
            // For any other method with a sessionId, remap if needed.
            if msg.pointer("/params/sessionId").is_some() {
                ClientAction::ForwardPatched(remap_client_session_id(msg, state))
            } else {
                ClientAction::Forward
            }
        }
    }
}

/// If the child has a different session ID than Zed (after session/load),
/// patch the sessionId in the message before forwarding to the child.
fn remap_client_session_id(msg: &Value, state: &ProxyState) -> String {
    if let Some(child_sid) = &state.child_session_id {
        let zed_sid = msg
            .pointer("/params/sessionId")
            .and_then(Value::as_str)
            .unwrap_or("");
        if child_sid != zed_sid {
            let mut patched = msg.clone();
            patched["params"]["sessionId"] = json!(child_sid);
            tracing::debug!(
                zed_sid,
                child_sid = child_sid.as_str(),
                "remapped client sessionId to child"
            );
            return patched.to_string();
        }
    }
    msg.to_string()
}

fn strip_meta_from_init(msg: &Value) -> String {
    let mut patched = msg.clone();
    if let Some(caps) = patched.pointer_mut("/params/clientCapabilities")
        && let Some(obj) = caps.as_object_mut()
        && obj.remove("_meta").is_some()
    {
        tracing::debug!("stripped _meta from clientCapabilities");
    }
    patched.to_string()
}

fn handle_set_model(msg: &Value, state: &mut ProxyState) -> ClientAction {
    let id = &msg["id"];
    if let Some(session_id) = msg.pointer("/params/sessionId").and_then(Value::as_str) {
        // Zed provides the sessionId here; capture it so we can recreate/remap
        // the session after restarting the child with a different model.
        state.zed_session_id = Some(session_id.to_string());
        state.current_session_id = Some(session_id.to_string());
    }
    let model_id = msg
        .pointer("/params/modelId")
        .and_then(Value::as_str)
        .unwrap_or("auto")
        .to_string();

    tracing::info!(model = %model_id, "model selection changed");
    state.selected_model = Some(model_id.clone());
    // The child is about to restart; its current session (if any) becomes invalid.
    state.child_session_id = None;

    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {}
    });

    ClientAction::Respond {
        response_to_zed: response.to_string(),
        restart_with_model: Some(model_id),
    }
}

// ---------------------------------------------------------------------------
// Agent → Client direction
// ---------------------------------------------------------------------------

/// Inspect a JSON-RPC message from `agent acp` and decide what to do.
pub fn process_agent_message(msg: &Value, state: &mut ProxyState) -> AgentAction {
    // Broad logging: capture every agent message method for debugging permissions.
    if let Some(method) = msg.get("method").and_then(Value::as_str) {
        let id = msg.get("id").map(|v| v.to_string()).unwrap_or_default();
        perm_log(&format!(
            "AGENT MSG method={method} id={id} has_params={}",
            msg.get("params").is_some()
        ));
    } else if msg.get("result").is_some() || msg.get("error").is_some() {
        let id = msg.get("id").map(|v| v.to_string()).unwrap_or_default();
        perm_log(&format!(
            "AGENT MSG response id={id} has_result={} has_error={}",
            msg.get("result").is_some(),
            msg.get("error").is_some()
        ));
    }
    // Suppress responses to replayed init/auth after child restart.
    if let Some(id) = msg.get("id")
        && (msg.get("result").is_some() || msg.get("error").is_some())
        && let Some(pos) = state.suppress_response_ids.iter().position(|s| s == id)
    {
        state.suppress_response_ids.swap_remove(pos);
        tracing::debug!(?id, "suppressed replay response");
        return AgentAction::Intercept {
            response_to_child: None,
            notifications_to_zed: vec![],
        };
    }

    // Intercept initialize response to inject sessionCapabilities.list
    if let Some(init_id) = &state.init_request_id
        && msg.get("id") == Some(init_id)
        && msg.get("result").is_some()
    {
        state.init_request_id = None;
        return inject_session_list_capability_and_cursor_login_meta(msg, state);
    }

    if let Some(session_id) = extract_session_id(msg) {
        state.current_session_id = Some(session_id);
    }

    // Remap child session ID to Zed's session ID after child restarts.
    let remapped = remap_session_id(msg, state);
    let msg = remapped.as_ref().unwrap_or(msg);

    // Keep current_session_id in sync with zed_session_id so plan
    // notifications use the correct ID.
    if let Some(zed_sid) = &state.zed_session_id {
        state.current_session_id = Some(zed_sid.clone());
    }

    // Intercept session/new response to inject models and cache modes/models
    if is_session_new_response(msg) {
        state.last_session_modes = msg.pointer("/result/modes").cloned();
        state.last_session_models = msg.pointer("/result/models").cloned();

        if !state.models.is_empty() {
            return inject_models_into_session_response(msg, state);
        }
    }

    // Check for createPlan / updateTodos tool calls in session/update
    // notifications and convert them into plan entries for Zed's Plan UI.
    if let Some(extra) = maybe_extract_plan_from_tool_call(msg, state) {
        return AgentAction::ForwardWithExtra {
            line: msg.to_string(),
            extra_notifications: extra,
        };
    }

    // Fallback: detect checklist/numbered plans from assistant markdown text
    // even when no explicit plan tool call was emitted.
    if let Some(extra) = maybe_extract_plan_from_agent_message(msg, state) {
        return AgentAction::ForwardWithExtra {
            line: msg.to_string(),
            extra_notifications: extra,
        };
    }

    // Synthesize display-only terminals for execute tool calls so Zed renders
    // the rich terminal UI (command, output, exit code) instead of a bare card.
    if let Some(action) = maybe_synthesize_terminal_for_execute(msg, state) {
        return action;
    }

    // Auto-respond to permission requests for tool kinds the user already
    // "always allowed" in this session (cached in allowed_tool_kinds).
    if let Some(action) = maybe_auto_approve_permission(msg, state) {
        return action;
    }

    // Track permission requests being forwarded to Zed so we can detect
    // allow-always responses and cache the tool kind.
    track_pending_permission_request(msg, state);

    // Inject terminal content into session/request_permission so Zed renders
    // the terminal UI even for commands that need user approval.
    if let Some(action) = maybe_inject_terminal_into_request_permission(msg, state) {
        return action;
    }

    // Strip backtick wrapping from titles of tracked execute tool calls.
    // Cursor sends titles like `command` but Zed's terminal UI renders
    // them as markdown, so backticks cause unwanted inline-code styling.
    if let Some(action) = maybe_strip_execute_title_backticks(msg, state) {
        return action;
    }

    // Synthesize fs/ requests for edit tool calls so Zed's ActionLog tracks
    // the changed files (enabling the "Edits" panel with accept/reject).
    if let Some(extra) = maybe_synthesize_fs_for_edit(msg, state) {
        return AgentAction::ForwardWithExtra {
            line: msg.to_string(),
            extra_notifications: extra,
        };
    }

    let method = msg.get("method").and_then(Value::as_str);
    let has_id = msg.get("id").is_some();

    let action = match (method, has_id) {
        (Some("_cursor/update_todos"), true) => handle_update_todos(msg, state),
        (Some(m), true) if m.starts_with("_cursor/") => handle_unknown_cursor_request(msg),
        _ => AgentAction::Forward,
    };

    // If the message was remapped and the action is Forward, we need to send
    // the patched version instead.
    if remapped.is_some() && matches!(&action, AgentAction::Forward) {
        return AgentAction::ForwardPatched(msg.to_string());
    }
    action
}

fn inject_session_list_capability_and_cursor_login_meta(
    msg: &Value,
    state: &ProxyState,
) -> AgentAction {
    let mut patched = msg.clone();
    if let Some(caps) = patched.pointer_mut("/result/agentCapabilities")
        && let Some(obj) = caps.as_object_mut()
    {
        obj.insert("sessionCapabilities".to_string(), json!({ "list": {} }));
    }

    // Cursor's ACP server advertises an auth method (`cursor_login`) but may still require
    // the user to run `cursor-agent login` in a terminal. Zed supports an experimental
    // `terminal-auth` metadata field on auth methods to spawn a login command.
    if let Some(auth_methods) = patched.pointer_mut("/result/authMethods")
        && let Some(arr) = auth_methods.as_array_mut()
    {
        let command = state
            .agent_binary
            .clone()
            .unwrap_or_else(|| "cursor-agent".to_string());

        for method in arr.iter_mut() {
            let id = method.get("id").and_then(Value::as_str);
            let name = method.get("name").and_then(Value::as_str);
            let is_cursor_login =
                matches!(id, Some("cursor_login")) || matches!(name, Some("Cursor Login"));
            if !is_cursor_login {
                continue;
            }

            // Ensure `meta.terminal-auth` exists.
            if method.get("meta").is_none() {
                method["meta"] = json!({});
            }
            if method
                .pointer("/meta/terminal-auth")
                .and_then(|v| v.as_object())
                .is_some()
            {
                continue;
            }

            method["meta"]["terminal-auth"] = json!({
                "label": "cursor-agent login",
                "command": command,
                "args": ["login"],
                "env": {},
            });
        }
    }

    tracing::debug!("injected sessionCapabilities.list into initialize response");
    AgentAction::ForwardPatched(patched.to_string())
}

/// Track a new session from a `session/new` response without creating it in the
/// store yet. Session creation is deferred until the first `session/prompt`.
pub fn track_new_session(msg: &Value, state: &mut ProxyState) {
    if !is_session_new_response(msg) {
        return;
    }
    let session_id = msg.pointer("/result/sessionId").and_then(Value::as_str);
    let request_id = msg.get("id").map(|v| v.to_string());
    if let (Some(sid), Some(rid)) = (session_id, request_id) {
        // Cache modes/models from session/new responses even when the response
        // will later be suppressed (e.g. internal session/new during session/load).
        if let Some(modes) = msg.pointer("/result/modes") {
            state.last_session_modes = Some(modes.clone());
        }
        if let Some(models) = msg.pointer("/result/models") {
            state.last_session_models = Some(models.clone());
        }

        if let Some(tx) = state.pending_session_new_waiters.remove(&rid) {
            drop(tx.send(sid.to_string()));
        }

        // Always track the child's session ID.
        state.child_session_id = Some(sid.to_string());

        if let Some(cwd) = state.pending_new_session_cwds.remove(&rid) {
            // Zed-initiated session/new: both IDs match.
            state.zed_session_id = Some(sid.to_string());
            state.session_cwds.insert(sid.to_string(), cwd.clone());
            state.pending_sessions.insert(sid.to_string(), cwd);
        } else {
            // Load-triggered session/new: child got a new ID but Zed keeps the loaded ID.
            tracing::debug!(child_sid = sid, zed_sid = ?state.zed_session_id, "child session created for loaded session");
        }
    }
}

/// If the current Zed session was tracked but not yet persisted, remove it from
/// `pending_sessions` and return `(session_id, cwd)` so the caller can create it.
pub fn take_pending_session(state: &mut ProxyState) -> Option<(String, String)> {
    let sid = state.zed_session_id.as_ref()?.clone();
    let cwd = state.pending_sessions.remove(&sid)?;
    Some((sid, cwd))
}

/// Info extracted from a `session/update` notification for session tracking.
pub struct SessionUpdateInfo {
    pub session_id: String,
    pub update: Value,
    /// Explicit title from `session_info_update`.
    pub title: Option<String>,
    /// First user message text for auto-title when no explicit title exists.
    pub user_message: Option<String>,
}

/// Extract session update info from a `session/update` notification.
pub fn extract_session_update(msg: &Value) -> Option<SessionUpdateInfo> {
    if msg.get("method")?.as_str()? != "session/update" {
        return None;
    }
    let session_id = msg.pointer("/params/sessionId")?.as_str()?.to_string();
    let update = msg.pointer("/params/update")?.clone();
    let update_type = update.get("sessionUpdate").and_then(Value::as_str);

    let title = if update_type == Some("session_info_update") {
        update
            .get("title")
            .and_then(Value::as_str)
            .map(String::from)
    } else {
        None
    };

    let user_message = if update_type == Some("user_message_chunk") {
        update
            .pointer("/content/text")
            .and_then(Value::as_str)
            .map(String::from)
    } else {
        None
    };

    Some(SessionUpdateInfo {
        session_id,
        update,
        title,
        user_message,
    })
}

/// If the child uses a different session ID than what Zed expects, rewrite
/// the `sessionId` field so Zed routes the message to the correct session.
/// This covers all ACP methods: session/*, fs/*, terminal/*, _cursor/*, etc.
fn remap_session_id(msg: &Value, state: &ProxyState) -> Option<Value> {
    let child_sid = msg.pointer("/params/sessionId").and_then(Value::as_str)?;
    let zed_sid = state.zed_session_id.as_deref()?;
    if child_sid == zed_sid {
        return None;
    }
    let mut patched = msg.clone();
    patched["params"]["sessionId"] = json!(zed_sid);
    tracing::debug!(child_sid, zed_sid, "remapped session ID");
    Some(patched)
}

// ---------------------------------------------------------------------------
// Terminal synthesis for execute tool calls
// ---------------------------------------------------------------------------

/// Generate a unique terminal ID (simple hex counter, no uuid dependency).
fn generate_terminal_id(state: &mut ProxyState) -> String {
    state.internal_id_counter -= 1;
    format!("synth-term-{:x}", state.internal_id_counter.unsigned_abs())
}

/// Intercept execute tool calls to synthesize display-only terminals in Zed.
///
/// - On `tool_call` with `kind: "execute"` but no command yet: buffer the
///   message (return Drop) so Zed doesn't briefly show a bare card.
/// - On `tool_call` with `kind: "execute"` and `rawInput.command`: patch the
///   message with `_meta.terminal_info` and `content: [terminal]` so Zed
///   creates a display-only terminal and renders the rich terminal UI.
/// - On `tool_call_update` with `status: "completed"` and `rawOutput` for a
///   tracked execute tool call: inject `_meta.terminal_output` and
///   `_meta.terminal_exit` so Zed feeds the output into the terminal widget.
fn maybe_synthesize_terminal_for_execute(
    msg: &Value,
    state: &mut ProxyState,
) -> Option<AgentAction> {
    let update = msg.pointer("/params/update")?;
    let update_type = update.get("sessionUpdate")?.as_str()?;

    match update_type {
        "tool_call" => {
            let kind = update.get("kind").and_then(Value::as_str);
            if kind != Some("execute") {
                return None;
            }

            let tool_call_id = update.get("toolCallId")?.as_str()?.to_string();
            let command = update.pointer("/rawInput/command").and_then(Value::as_str);

            if command.is_none() || command == Some("") {
                // No command yet — buffer this message so Zed doesn't show a
                // bare "Terminal" card. We'll discard it when the real one arrives.
                state
                    .buffered_execute_tool_calls
                    .insert(tool_call_id, msg.clone());
                tracing::debug!("buffered execute tool_call (no command yet)");
                return Some(AgentAction::Drop);
            }

            let command = command.unwrap();

            // Discard any buffered version now that we have the real one.
            state.buffered_execute_tool_calls.remove(&tool_call_id);

            // Generate or reuse terminal ID for this tool call.
            let terminal_id = if let Some(existing) = state.terminal_ids.get(&tool_call_id) {
                existing.clone()
            } else {
                let id = generate_terminal_id(state);
                state.terminal_ids.insert(tool_call_id, id.clone());
                id
            };

            // Extract cwd from rawInput, falling back to workspace_cwd.
            let raw_cd = update.pointer("/rawInput/cd").and_then(Value::as_str);
            let raw_cwd = update.pointer("/rawInput/cwd").and_then(Value::as_str);
            let raw_wd = update
                .pointer("/rawInput/working_directory")
                .and_then(Value::as_str);
            let cwd = raw_cd
                .or(raw_cwd)
                .or(raw_wd)
                .map(String::from)
                .or_else(|| state.workspace_cwd.clone());

            // Build patched message with terminal_info in _meta and terminal
            // content so Zed creates a display-only terminal.
            let mut patched = msg.clone();
            let patch_update = patched.pointer_mut("/params/update")?;

            // Set _meta.terminal_info
            let meta = patch_update
                .as_object_mut()?
                .entry("_meta")
                .or_insert_with(|| json!({}));
            let meta_obj = meta.as_object_mut()?;
            let mut terminal_info = json!({ "terminal_id": terminal_id });
            if let Some(ref cwd) = cwd {
                terminal_info["cwd"] = json!(cwd);
            }
            meta_obj.insert("terminal_info".to_string(), terminal_info);

            // Set content to include the terminal reference.
            patch_update["content"] = json!([{
                "type": "terminal",
                "terminalId": terminal_id,
            }]);

            // Use the plain command as the title — Zed's Terminal::new wraps
            // it in a markdown code block, so we must not add backticks here.
            patch_update["title"] = json!(command);

            // Record for PTY stream matching so we can correlate the
            // streaming spawn notification with this tool call.
            let session_id = msg
                .pointer("/params/sessionId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let tc_id = update
                .get("toolCallId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            state.pty_pending_matches.push(PtyStreamMatch {
                terminal_id: terminal_id.clone(),
                tool_call_id: tc_id,
                session_id,
                command: command.to_string(),
            });

            tracing::debug!(
                terminal_id = terminal_id.as_str(),
                command,
                "synthesized terminal for execute tool call"
            );

            Some(AgentAction::ForwardPatched(patched.to_string()))
        }
        "tool_call_update" => {
            let tool_call_id = update.get("toolCallId")?.as_str()?;
            let terminal_id = state.terminal_ids.get(tool_call_id)?.clone();

            let raw_output = update.get("rawOutput")?;
            let status = update.get("status").and_then(Value::as_str);

            if !matches!(status, Some("completed") | Some("failed")) {
                return None;
            }

            let stdout = raw_output
                .get("stdout")
                .and_then(Value::as_str)
                .unwrap_or("");
            let stderr = raw_output
                .get("stderr")
                .and_then(Value::as_str)
                .unwrap_or("");
            let exit_code = raw_output.get("exitCode").and_then(Value::as_u64);

            // If we already streamed this terminal's output in real-time,
            // skip the batch output to avoid duplicates.
            let already_streamed = state.pty_streamed_terminals.remove(&terminal_id);

            let mut output = String::new();
            if !already_streamed {
                if !stdout.is_empty() {
                    output.push_str(stdout);
                }
                if !stderr.is_empty() {
                    if !output.is_empty() && !output.ends_with('\n') {
                        output.push('\n');
                    }
                    output.push_str(stderr);
                }
            }

            let mut patched = msg.clone();
            let patch_update = patched.pointer_mut("/params/update")?;

            let meta = patch_update
                .as_object_mut()?
                .entry("_meta")
                .or_insert_with(|| json!({}));
            let meta_obj = meta.as_object_mut()?;

            if !output.is_empty() {
                meta_obj.insert(
                    "terminal_output".to_string(),
                    json!({ "terminal_id": terminal_id, "data": output }),
                );
            }

            let mut exit_status = json!({ "terminal_id": terminal_id });
            if let Some(code) = exit_code {
                exit_status["exit_code"] = json!(code);
            }
            meta_obj.insert("terminal_exit".to_string(), exit_status);

            // Clean up tracking state.
            state.terminal_ids.remove(tool_call_id);

            tracing::debug!(
                terminal_id = terminal_id.as_str(),
                ?exit_code,
                output_len = output.len(),
                "injected terminal output/exit for execute tool call"
            );

            Some(AgentAction::ForwardPatched(patched.to_string()))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// PTY streaming helpers
// ---------------------------------------------------------------------------

/// Try to match a PTY spawn notification (from the pty-proxy addon) to a
/// pending execute tool call.  Returns the match if found.
pub fn match_pty_spawn(state: &mut ProxyState, pid: i32, cmd: &str) -> Option<PtyStreamMatch> {
    let pos = state
        .pty_pending_matches
        .iter()
        .position(|m| cmd.contains(&m.command) || m.command.contains(cmd));
    if let Some(idx) = pos {
        let matched = state.pty_pending_matches.remove(idx);
        state.pty_stream_pids.insert(pid, matched.clone());
        tracing::debug!(pid, terminal_id = %matched.terminal_id, "matched PTY stream to terminal");
        Some(matched)
    } else {
        // No match yet — store for later matching when tool_call arrives.
        tracing::debug!(pid, cmd, "PTY spawn received but no pending match");
        None
    }
}

/// Build a `session/update` notification that delivers streaming terminal output.
pub fn build_streaming_terminal_output(info: &PtyStreamMatch, data: &str) -> String {
    json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": info.session_id,
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": info.tool_call_id,
                "status": "in_progress",
                "_meta": {
                    "terminal_output": {
                        "terminal_id": info.terminal_id,
                        "data": data,
                    }
                }
            }
        }
    })
    .to_string()
}

/// Auto-respond to `session/request_permission` if the user previously chose
/// "allow-always" for this tool kind in the current session. Returns an
/// `Intercept` action that sends the approval response to the child and a
/// `session/update` notification to Zed (so Zed shows the tool as in-progress
/// rather than stuck waiting for permission).
fn maybe_auto_approve_permission(msg: &Value, state: &ProxyState) -> Option<AgentAction> {
    let method = msg.get("method").and_then(Value::as_str)?;
    if method != "session/request_permission" {
        return None;
    }
    let request_id = msg.get("id")?;

    let tool_kind = msg
        .pointer("/params/toolCall/kind")
        .and_then(Value::as_str)?;

    perm_log(&format!(
        "request_permission id={request_id} kind={tool_kind:?} cached_kinds={:?}",
        state.allowed_tool_kinds
    ));

    if !state.allowed_tool_kinds.contains(tool_kind) {
        perm_log("  -> NOT cached, forwarding to Zed");
        return None;
    }

    // Find the allow-always option ID from the options list.
    let options = msg.pointer("/params/options").and_then(Value::as_array)?;
    let option_id = options.iter().find_map(|opt| {
        let kind = opt.get("kind").and_then(Value::as_str)?;
        if kind == "allow_always" {
            opt.get("optionId")
                .and_then(Value::as_str)
                .map(String::from)
        } else {
            None
        }
    })?;

    perm_log(&format!(
        "  -> AUTO-APPROVE kind={tool_kind:?} option_id={option_id:?}"
    ));
    tracing::info!(
        tool_kind,
        option_id = option_id.as_str(),
        "auto-approving permission (cached allow-always)"
    );

    let response = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "optionId": option_id
            }
        }
    });

    Some(AgentAction::Intercept {
        response_to_child: Some(response.to_string()),
        notifications_to_zed: vec![],
    })
}

/// Record a pending permission request so we can detect the allow-always
/// response from Zed and cache the tool kind.
fn track_pending_permission_request(msg: &Value, state: &mut ProxyState) {
    let method = msg.get("method").and_then(Value::as_str);
    if method != Some("session/request_permission") {
        return;
    }

    let request_id = match msg.get("id") {
        Some(id) => id.to_string(),
        None => return,
    };

    let tool_kind = msg
        .pointer("/params/toolCall/kind")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let allow_always_option_id = msg
        .pointer("/params/options")
        .and_then(Value::as_array)
        .and_then(|opts| {
            opts.iter().find_map(|opt| {
                let kind = opt.get("kind").and_then(Value::as_str)?;
                if kind == "allow_always" {
                    opt.get("optionId")
                        .and_then(Value::as_str)
                        .map(String::from)
                } else {
                    None
                }
            })
        });

    state.pending_permission_requests.insert(
        request_id,
        PendingPermission {
            tool_kind,
            allow_always_option_id,
        },
    );
}

/// When Zed responds to a `session/request_permission` with the allow-always
/// option, cache the tool kind so future requests are auto-approved.
pub fn maybe_cache_permission_response(msg: &Value, state: &mut ProxyState) {
    // Only look at JSON-RPC responses (have `id` + `result`).
    if msg.get("method").is_some() || msg.get("result").is_none() {
        return;
    }
    let request_id = match msg.get("id") {
        Some(id) => id.to_string(),
        None => return,
    };

    let pending = match state.pending_permission_requests.remove(&request_id) {
        Some(p) => p,
        None => return,
    };

    perm_log(&format!(
        "permission_response id={request_id} tool_kind={:?} always_opt={:?} response={msg}",
        pending.tool_kind, pending.allow_always_option_id,
    ));

    let selected_option_id = msg
        .pointer("/result/outcome/optionId")
        .and_then(Value::as_str);

    if let (Some(selected), Some(always_id)) = (selected_option_id, &pending.allow_always_option_id)
        && selected == always_id
    {
        perm_log(&format!(
            "  -> CACHING allow-always for kind={:?}",
            pending.tool_kind
        ));
        tracing::info!(
            tool_kind = pending.tool_kind.as_str(),
            "caching allow-always for tool kind"
        );
        state.allowed_tool_kinds.insert(pending.tool_kind);
    }
}

/// When cursor-agent sends `session/request_permission` for an execute tool call,
/// the tool_call update inside it will overwrite our previously-injected Terminal
/// content. Re-inject it here so Zed still renders the terminal UI.
fn maybe_inject_terminal_into_request_permission(
    msg: &Value,
    state: &ProxyState,
) -> Option<AgentAction> {
    let method = msg.get("method").and_then(Value::as_str)?;
    if method != "session/request_permission" {
        return None;
    }

    let tool_call_id = msg
        .pointer("/params/toolCall/toolCallId")
        .and_then(Value::as_str)?;
    let terminal_id = state.terminal_ids.get(tool_call_id)?.clone();

    let mut patched = msg.clone();
    let tool_call = patched.pointer_mut("/params/toolCall")?;
    let tool_call_obj = tool_call.as_object_mut()?;

    tool_call_obj.insert(
        "content".to_string(),
        json!([{ "type": "terminal", "terminalId": terminal_id }]),
    );

    // Strip backtick wrapping from the title if present.
    if let Some(title) = tool_call_obj.get("title").and_then(Value::as_str)
        && let Some(stripped) = title.strip_prefix('`').and_then(|s| s.strip_suffix('`'))
    {
        tool_call_obj.insert("title".to_string(), json!(stripped));
    }

    let meta = tool_call_obj.entry("_meta").or_insert_with(|| json!({}));
    if let Some(meta_obj) = meta.as_object_mut() {
        meta_obj.insert(
            "terminal_info".to_string(),
            json!({ "terminal_id": terminal_id }),
        );
    }

    tracing::debug!(
        %terminal_id,
        tool_call_id,
        "injected terminal content into request_permission"
    );

    Some(AgentAction::ForwardPatched(patched.to_string()))
}

/// Strip surrounding backticks from the `title` field of session/update
/// messages for tracked execute tool calls. Cursor sends titles like
/// `` `echo hello` `` but Zed renders them as markdown, producing unwanted
/// inline-code formatting. This catches any update (tool_call or
/// tool_call_update) that our synthesize function didn't already patch.
fn maybe_strip_execute_title_backticks(msg: &Value, state: &ProxyState) -> Option<AgentAction> {
    let update = msg.pointer("/params/update")?;
    let tool_call_id = update.get("toolCallId").and_then(Value::as_str)?;

    if !state.terminal_ids.contains_key(tool_call_id) {
        return None;
    }

    let title = update.get("title").and_then(Value::as_str)?;
    let stripped = title.strip_prefix('`').and_then(|s| s.strip_suffix('`'))?;

    let mut patched = msg.clone();
    let patch_update = patched.pointer_mut("/params/update")?;
    patch_update["title"] = json!(stripped);

    Some(AgentAction::ForwardPatched(patched.to_string()))
}

/// When cursor-agent sends edit tool calls, synthesize `fs/read_text_file`
/// requests to Zed so its ActionLog tracks the changes (enabling the "Edits"
/// panel with accept/reject all).
///
/// - On `tool_call` with `kind: "edit"` and `status: "pending"`: send
///   `fs/read_text_file` so Zed opens the buffer with the OLD content.
///
/// We intentionally do NOT synthesize  We intentionally do NOT synthesize `fs/write_text_file` on
/// `tool_call_update` — Cursor already sends its own write request for
/// completed edits, and duplicating it causeson
/// `tool_call_update` — Cursor already sends its own write request for
/// completed edits, and duplicating it causes Zed to applyto apply the editedit twice,twice,
/// corruptingcorrupting the bufferbuffer.
fn maybe_synthesize_fs_for_edit(msg: &Value, state: &mut ProxyState) -> Option<Vec<String>> {
    let update = msg.pointer("/params/update")?;
    let update_type = update.get("sessionUpdate")?.as_str()?;
    // Use the sessionId on the message (already remapped to Zed's session id),
    // instead of state.zed_session_id, so this works during transitions and for
    // any non-primary sessions (e.g. subagents).
    let session_id = msg.pointer("/params/sessionId")?.as_str()?.to_string();

    match update_type {
        "tool_call" => {
            if update.get("kind").and_then(Value::as_str) != Some("edit") {
                return None;
            }
            if !matches!(
                update.get("status").and_then(Value::as_str),
                Some("pending")
            ) {
                return None;
            }

            let mut reqs = Vec::new();

            // Extract paths from locations or rawInput.path
            let mut paths = Vec::new();
            if let Some(locs) = update.get("locations").and_then(Value::as_array) {
                for loc in locs {
                    if let Some(p) = loc.get("path").and_then(Value::as_str) {
                        paths.push(p.to_string());
                    }
                }
            }
            if paths.is_empty()
                && let Some(p) = update.pointer("/rawInput/path").and_then(Value::as_str)
            {
                paths.push(p.to_string());
            }

            // Normalize to absolute paths so `fs/read_text_file` targets the correct file.
            let mut normalized = Vec::new();
            for p in paths {
                if let Some(abs) = normalize_path_for_session(&session_id, &p, state) {
                    if !normalized.contains(&abs) {
                        normalized.push(abs);
                    }
                } else {
                    tracing::debug!(
                        path = p.as_str(),
                        "skipped fs/read_text_file; could not normalize path"
                    );
                }
            }

            for path in &normalized {
                let id = state.next_internal_id();
                state.suppress_zed_response(id.clone());
                let req = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": "fs/read_text_file",
                    "params": {
                        "sessionId": session_id,
                        "path": path,
                    }
                });
                tracing::debug!(path, "synthesized fs/read_text_file for edit tracking");
                reqs.push(req.to_string());
            }

            if reqs.is_empty() { None } else { Some(reqs) }
        }
        "tool_call_update" => {
            // Don't synthesize fs/write_text_file here — Cursor already sends
            // its own fs/write_text_file for completed edits.  Synthesizing a
            // duplicate causes Zed to apply the write twice, corrupting the
            // buffer (token-level doubling).
            None
        }
        _ => None,
    }
}

fn normalize_path_for_session(session_id: &str, path: &str, state: &ProxyState) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let p = Path::new(path);
    if p.is_absolute() {
        return Some(path.to_string());
    }
    let cwd = state
        .session_cwds
        .get(session_id)
        .or_else(|| state.pending_sessions.get(session_id))?;
    if !Path::new(cwd).is_absolute() {
        return None;
    }
    Some(PathBuf::from(cwd).join(p).to_string_lossy().into_owned())
}

/// Extract the user's prompt text from a `session/prompt` request.
fn extract_prompt_text(msg: &Value) -> Option<String> {
    let prompt = msg.pointer("/params/prompt")?;
    if let Some(arr) = prompt.as_array() {
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|item| {
                if item.get("type").and_then(Value::as_str) == Some("text") {
                    item.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n"))
        }
    } else {
        None
    }
}

fn is_session_new_response(msg: &Value) -> bool {
    msg.get("result").is_some()
        && msg.pointer("/result/sessionId").is_some()
        && msg.pointer("/result/modes").is_some()
}

fn inject_models_into_session_response(msg: &Value, state: &ProxyState) -> AgentAction {
    let mut patched = msg.clone();

    let current = state.selected_model.as_deref().unwrap_or("auto");

    let available: Vec<Value> = state
        .models
        .iter()
        .map(|m| {
            json!({
                "modelId": m.id,
                "name": m.name,
            })
        })
        .collect();

    if let Some(result) = patched.get_mut("result") {
        result["models"] = json!({
            "currentModelId": current,
            "availableModels": available,
        });
    }

    tracing::debug!(
        count = state.models.len(),
        current,
        "injected models into session/new"
    );
    AgentAction::ForwardPatched(patched.to_string())
}

fn extract_session_id(msg: &Value) -> Option<String> {
    let method = msg.get("method")?.as_str()?;

    if method == "session/update" {
        return msg
            .pointer("/params/sessionId")
            .and_then(Value::as_str)
            .map(String::from);
    }

    None
}

/// Check if a `session/update` notification contains a `createPlan` or
/// `updateTodos` tool call. If so, parse the data into plan entries and
/// return plan notification(s) for Zed.
fn maybe_extract_plan_from_tool_call(msg: &Value, state: &mut ProxyState) -> Option<Vec<String>> {
    let update = msg.pointer("/params/update")?;
    if update.get("sessionUpdate")?.as_str()? != "tool_call" {
        return None;
    }
    let tool_name = update.pointer("/rawInput/_toolName")?.as_str()?;

    let session_id = state
        .zed_session_id
        .as_deref()
        .or(state.current_session_id.as_deref())?
        .to_string();

    let entries: Vec<Value> = match tool_name {
        "createPlan" => {
            let plan_text = update.pointer("/rawInput/plan")?.as_str()?;
            let plan_name = update
                .pointer("/rawInput/name")
                .and_then(Value::as_str)
                .map(|s| s.to_string());

            // Parse todos: first try code block, then fall back to checklist parsing.
            let parsed = parse_todos_from_code_block(plan_text)
                .unwrap_or_else(|| parse_plan_entries(plan_text));

            if !parsed.is_empty() && !plan_text.trim().is_empty() {
                // Strip the todos code block from saved markdown for cleaner files.
                let clean_markdown = strip_todos_code_block(plan_text);
                state.plans.insert(
                    session_id.clone(),
                    PlanInfo {
                        markdown: clean_markdown,
                        name: plan_name,
                    },
                );
            }
            parsed
        }
        // updateTodos tool_call updates are handled exclusively by
        // `handle_update_todos` (which intercepts the `_cursor/update_todos`
        // request). Processing them here too would clobber state built by that
        // handler with a partial rawInput snapshot.
        "updateTodos" => return None,
        _ => return None,
    };

    if entries.is_empty() {
        return None;
    }

    let mut plan_update = json!({
        "sessionUpdate": "plan",
        "entries": entries,
    });

    attach_plan_markdown_meta(&mut plan_update, &session_id, tool_name, state);

    // `plan_update` is moved into the notification below; keep a copy for
    // optional synthesized side-effects (e.g. writing plan markdown to a file).
    let plan_update_for_side_effects = plan_update.clone();

    let notification = json!({
      "jsonrpc": "2.0",
      "method": "session/update",
      "params": {
        "sessionId": session_id,
        "update": plan_update,
      }
    });

    tracing::debug!(
        count = entries.len(),
        tool_name,
        "emitted plan from tool call"
    );

    let mut notifications = vec![notification.to_string()];

    if should_emit_plan_file_side_effects(state)
        && let Some(mut extra) =
            maybe_emit_plan_markdown_file(&session_id, update, &plan_update_for_side_effects, state)
    {
        notifications.append(&mut extra);
    }

    Some(notifications)
}

fn maybe_extract_plan_from_agent_message(
    msg: &Value,
    state: &mut ProxyState,
) -> Option<Vec<String>> {
    if msg.get("method").and_then(Value::as_str) != Some("session/update") {
        return None;
    }

    let update = msg.pointer("/params/update")?;
    if update.get("sessionUpdate")?.as_str()? != "agent_message_chunk" {
        return None;
    }
    if update.pointer("/content/type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    let chunk = update.pointer("/content/text").and_then(Value::as_str)?;
    if chunk.trim().is_empty() {
        return None;
    }
    let session_id = msg.pointer("/params/sessionId")?.as_str()?.to_string();

    let accumulated = {
        let buffer = state
            .plan_detection_buffers
            .entry(session_id.clone())
            .or_default();
        if buffer.len() > 32_768 {
            // Keep memory bounded while preserving recent context for detection.
            let split_at = buffer.len().saturating_sub(16_384);
            buffer.drain(..split_at);
        }
        buffer.push_str(chunk);
        buffer.clone()
    };

    let entries = parse_plan_entries(&accumulated);
    if entries.len() < 2 || !looks_like_planning_message(&accumulated, entries.len()) {
        return None;
    }

    let derived_markdown = entries_to_markdown(&entries);
    if derived_markdown.trim().is_empty() {
        return None;
    }
    if state
        .last_detected_plan_markdown
        .get(&session_id)
        .is_some_and(|m| m == &derived_markdown)
    {
        return None;
    }
    state
        .last_detected_plan_markdown
        .insert(session_id.clone(), derived_markdown.clone());

    if !state.plans.contains_key(&session_id) {
        let name = extract_markdown_heading(&accumulated)
            .or_else(|| state.session_first_prompt.get(&session_id).cloned());
        state.plans.insert(
            session_id.clone(),
            PlanInfo {
                markdown: derived_markdown.clone(),
                name,
            },
        );
    } else if let Some(plan) = state.plans.get_mut(&session_id) {
        plan.markdown = derived_markdown.clone();
    }

    let mut plan_update = json!({
        "sessionUpdate": "plan",
        "entries": entries,
    });
    attach_plan_markdown_meta(&mut plan_update, &session_id, "agent_message_chunk", state);

    let notification = json!({
      "jsonrpc": "2.0",
      "method": "session/update",
      "params": {
        "sessionId": session_id,
        "update": plan_update,
      }
    });

    // Note: we intentionally do NOT emit plan files from agent message detection.
    // Plan files are only created from explicit CreatePlan tool calls with a `plan` field.
    Some(vec![notification.to_string()])
}

fn looks_like_planning_message(text: &str, entry_count: usize) -> bool {
    if entry_count < 2 {
        return false;
    }
    let lower = text.to_ascii_lowercase();
    let has_checkbox = lower.contains("- [ ]") || lower.contains("- [x]");
    if has_checkbox {
        return true;
    }

    let has_plan_keyword =
        lower.contains("to-do") || lower.contains("todo") || lower.contains("checklist");
    if has_plan_keyword {
        return true;
    }

    // Numbered 3+ step blocks are often plans even without explicit keywords.
    let numbered = text
        .lines()
        .filter(|line| strip_numbered_prefix(line.trim()).is_some())
        .count();
    numbered >= 3
}

fn should_emit_plan_file_side_effects(state: &ProxyState) -> bool {
    state.emit_plan_files || state.emit_plan_file_messages
}

/// Attach additional plan markdown metadata for clients that want richer plan UIs.
/// Uses ACP's `_meta` extension point so standard clients safely ignore it.
fn attach_plan_markdown_meta(
    plan_update: &mut Value,
    session_id: &str,
    source_tool: &str,
    state: &ProxyState,
) {
    // Always provide a derived markdown rendering of the current plan entries
    // (useful even when Cursor didn't provide a createPlan markdown blob).
    let derived = plan_update
        .get("entries")
        .and_then(Value::as_array)
        .map(|v| entries_to_markdown(v.as_slice()))
        .unwrap_or_default();

    let mut meta = json!({
        "cursor-acp": {
            "sourceTool": source_tool,
            "derivedTodosMarkdown": derived,
        }
    });

    if let Some(plan) = state.plans.get(session_id) {
        if !plan.markdown.trim().is_empty() {
            meta["cursor-acp"]["planMarkdown"] = json!(plan.markdown.as_str());
        }
        if let Some(name) = &plan.name {
            meta["cursor-acp"]["planName"] = json!(name);
        }
    }

    if meta.pointer("/cursor-acp/planName").is_none()
        && let Some(name) = state.session_first_prompt.get(session_id)
    {
        meta["cursor-acp"]["planName"] = json!(name);
    }

    plan_update["_meta"] = meta;
}

fn entries_to_markdown(entries: &[Value]) -> String {
    let mut out = String::new();
    for entry in entries {
        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if content.is_empty() {
            continue;
        }
        let status = entry
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("pending");

        let line = match status {
            "completed" => format!("- [x] {content}\n"),
            "in_progress" => format!("- [ ] (in progress) {content}\n"),
            _ => format!("- [ ] {content}\n"),
        };
        out.push_str(&line);
    }
    out
}

fn maybe_emit_plan_markdown_file(
    session_id: &str,
    tool_call_update: &Value,
    plan_update: &Value,
    state: &mut ProxyState,
) -> Option<Vec<String>> {
    // Pull markdown from ACP meta we already attached.
    let cursor_meta = plan_update.pointer("/_meta/cursor-acp")?;
    let markdown = cursor_meta
        .get("planMarkdown")
        .and_then(Value::as_str)
        .or_else(|| {
            cursor_meta
                .get("derivedTodosMarkdown")
                .and_then(Value::as_str)
        })?;
    if markdown.trim().is_empty() {
        return None;
    }

    // Prefer an explicit location path if present.
    let explicit_path = tool_call_update
        .pointer("/locations/0/path")
        .and_then(Value::as_str)
        .map(|s| s.to_string());

    let path = if let Some(ref p) = explicit_path
        && Path::new(p).is_absolute()
        && (p.ends_with(".md") || p.ends_with(".mdx") || p.ends_with(".markdown"))
    {
        p.clone()
    } else {
        let cwd = state
            .session_cwds
            .get(session_id)
            .cloned()
            .or_else(|| state.pending_sessions.get(session_id).cloned());
        let cwd = cwd?;
        if !Path::new(&cwd).is_absolute() {
            return None;
        }

        let plan_name = cursor_meta.get("planName").and_then(Value::as_str);
        // Try to discover Cursor's own plan file in `.cursor/plans/` first.
        if state.link_cursor_plan_files {
            if let Some(found) = discover_cursor_plan_file(&cwd, plan_name, markdown) {
                found
            } else {
                default_plan_file_path(&cwd, plan_name)
            }
        } else {
            default_plan_file_path(&cwd, plan_name)
        }
    };

    // If we discovered an existing Cursor plan file and we're only linking, skip writes.
    let should_write = state.emit_plan_files
        && !(state.link_cursor_plan_files
            && explicit_path.is_none()
            && cursor_plan_dir_path(&path));

    // De-dupe identical writes.
    if let Some(prev_path) = state.plan_file_paths.get(session_id)
        && prev_path == &path
    {
        // ok
    } else {
        state
            .plan_file_paths
            .insert(session_id.to_string(), path.clone());
    }

    let mut notifications = Vec::new();

    // Write file via ACP client fs/write_text_file
    if should_write {
        let id = state.next_internal_id();
        state.suppress_zed_response(id.clone());
        let write_req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "fs/write_text_file",
            "params": {
                "sessionId": session_id,
                "path": path,
                "content": markdown,
            }
        });
        notifications.push(write_req.to_string());
    }

    if state.emit_plan_file_messages && !state.plan_file_message_emitted.contains(session_id) {
        state
            .plan_file_message_emitted
            .insert(session_id.to_string());
        let abs_path = path.as_str();
        let file_url = format!(
            "file:///{}",
            encode_file_url_path(abs_path.trim_start_matches('/'))
        );
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": format!("Plan markdown saved to {file_url}\n") }
                }
            }
        });
        notifications.push(msg.to_string());
    }

    Some(notifications)
}

fn default_plan_file_path(cwd: &str, plan_name: Option<&str>) -> String {
    let file_name = plan_name
        .map(sanitize_filename_component)
        .filter(|s| !s.is_empty())
        .map(|name| format!("{name}.md"))
        .unwrap_or_else(|| "plan.md".to_string());
    PathBuf::from(cwd)
        .join(".plans")
        .join(file_name)
        .to_string_lossy()
        .into_owned()
}

fn cursor_plan_dir_path(abs_path: &str) -> bool {
    // Heuristic: treat `.cursor/plans/` files as "Cursor-owned".
    abs_path.contains("/.cursor/plans/") || abs_path.contains("\\.cursor\\plans\\")
}

fn discover_cursor_plan_file(cwd: &str, plan_name: Option<&str>, markdown: &str) -> Option<String> {
    let plans_dir = Path::new(cwd).join(".cursor").join("plans");
    let dir = std::fs::read_dir(&plans_dir).ok()?;

    let needles = plan_needles(plan_name, markdown);

    // Collect candidates with mtime so we can try newest first.
    let mut candidates: Vec<(SystemTime, PathBuf)> = Vec::new();
    for entry in dir.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "md" | "markdown" | "mdx") {
            continue;
        }
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((mtime, path));
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    for (_mtime, path) in candidates.into_iter().take(10) {
        let text = std::fs::read_to_string(&path).ok()?;
        if looks_like_same_plan(&text, &needles, markdown) {
            return Some(path.to_string_lossy().into_owned());
        }
    }
    None
}

fn plan_needles(plan_name: Option<&str>, markdown: &str) -> Vec<String> {
    let mut needles = Vec::new();
    if let Some(name) = plan_name
        && !name.trim().is_empty()
    {
        needles.push(name.trim().to_string());
    }
    for line in markdown.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        needles.push(line.to_string());
        if needles.len() >= 4 {
            break;
        }
    }
    needles
}

fn looks_like_same_plan(file_text: &str, needles: &[String], markdown: &str) -> bool {
    let f = file_text.trim();
    let m = markdown.trim();
    if !m.is_empty() && f.contains(m) {
        return true;
    }
    let mut hits = 0usize;
    for n in needles {
        if n.len() >= 4 && f.contains(n) {
            hits += 1;
        }
    }
    hits >= 2
}

/// Extract the first markdown heading (e.g. `## My Plan`) as a plan name.
fn extract_markdown_heading(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix('#') {
            let heading = rest.trim_start_matches('#').trim();
            if !heading.is_empty() {
                return Some(heading.to_string());
            }
        }
    }
    None
}

fn sanitize_filename_component(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
            out.push(c);
        } else if c.is_whitespace() {
            out.push('-');
        } else {
            out.push('_');
        }
        if out.len() >= 64 {
            break;
        }
    }
    out.trim_matches(['-', '_', '.']).to_string()
}

fn encode_file_url_path(path: &str) -> String {
    // Minimal percent-encoding for `file:///...` links that Zed parses via `url::Url::parse`.
    // Encode anything outside RFC3986 unreserved + '/'.
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        let c = b as char;
        let unreserved = c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '/');
        if unreserved {
            out.push(c);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

/// Parse todos from a fenced code block in the plan markdown.
/// Looks for: ```todos\n[...json...]\n```
fn parse_todos_from_code_block(plan_text: &str) -> Option<Vec<Value>> {
    let re = Regex::new(r"```todos\s*\n([\s\S]*?)\n```").ok()?;
    let caps = re.captures(plan_text)?;
    let json_str = caps.get(1)?.as_str();
    let arr: Vec<Value> = serde_json::from_str(json_str).ok()?;
    let todos: Vec<Value> = arr
        .iter()
        .filter_map(|item| {
            let content = item.get("content").and_then(Value::as_str)?;
            let status = normalize_todo_status(
                item.get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("pending"),
            );
            Some(json!({
                "content": content,
                "priority": "medium",
                "status": status,
            }))
        })
        .collect();
    if todos.is_empty() { None } else { Some(todos) }
}

/// Strip the todos code block from plan markdown for cleaner saved files.
fn strip_todos_code_block(plan_text: &str) -> String {
    let re = Regex::new(r"\n?```todos\s*\n[\s\S]*?\n```\n?").unwrap();
    re.replace(plan_text, "").trim().to_string()
}

/// Parse markdown list items into plan entries.
/// Supports: `- item`, `- [ ] item`, `- [x] item`, `1. item`, `2. item`, etc.
fn parse_plan_entries(plan_text: &str) -> Vec<Value> {
    plan_text
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("- [x] ")
                .or_else(|| trimmed.strip_prefix("- [X] "))
            {
                Some(json!({ "content": rest.trim(), "priority": "medium", "status": "completed" }))
            } else if let Some(rest) = trimmed.strip_prefix("- [ ] ") {
                Some(json!({ "content": rest.trim(), "priority": "medium", "status": "pending" }))
            } else if let Some(rest) = trimmed.strip_prefix("- ") {
                if rest.starts_with('#') || rest.is_empty() {
                    return None;
                }
                Some(json!({ "content": rest.trim(), "priority": "medium", "status": "pending" }))
            } else if let Some(rest) = strip_numbered_prefix(trimmed) {
                if rest.is_empty() {
                    return None;
                }
                Some(json!({ "content": rest.trim(), "priority": "medium", "status": "pending" }))
            } else {
                None
            }
        })
        .collect()
}

/// Strip a numbered list prefix like `1. `, `2. `, `10. ` etc.
fn strip_numbered_prefix(s: &str) -> Option<&str> {
    let dot_pos = s.find(". ")?;
    if dot_pos == 0 {
        return None;
    }
    if s[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
        Some(&s[dot_pos + 2..])
    } else {
        None
    }
}

fn handle_update_todos(msg: &Value, state: &mut ProxyState) -> AgentAction {
    let id = &msg["id"];
    let params = &msg["params"];

    let merge = params
        .get("merge")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !merge {
        state.todos.clear();
    }

    if let Some(items) = params.get("todos").and_then(Value::as_array) {
        for item in items {
            let todo_id = item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let content = item
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let status = normalize_todo_status(
                item.get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("pending"),
            );

            if let Some(existing) = state.todos.iter_mut().find(|t| t.id == todo_id) {
                existing.content = content;
                existing.status = status;
            } else {
                state.todos.push(TodoItem {
                    id: todo_id,
                    content,
                    status,
                });
            }
        }
    }

    // Purge cancelled/placeholder/empty items from state entirely.
    state.todos.retain(should_show_in_plan);

    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {}
    });

    let mut notifications = Vec::new();

    if let Some(session_id) = state
        .zed_session_id
        .as_deref()
        .or(state.current_session_id.as_deref())
    {
        let session_id = session_id.to_string();
        let entries: Vec<Value> = state
            .todos
            .iter()
            .map(|t| {
                json!({
                    "content": t.content,
                    "priority": "medium",
                    "status": t.status,
                })
            })
            .collect();

        if !entries.is_empty() {
            if !state.plans.contains_key(&session_id) {
                let name = state.session_first_prompt.get(&session_id).cloned();
                state.plans.insert(
                    session_id.clone(),
                    PlanInfo {
                        markdown: String::new(),
                        name,
                    },
                );
            }
            let mut plan_update = json!({
                "sessionUpdate": "plan",
                "entries": entries,
            });
            attach_plan_markdown_meta(&mut plan_update, &session_id, "_cursor/update_todos", state);
            let plan_update_for_side_effects = plan_update.clone();
            let plan_notification = json!({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": &session_id,
                    "update": plan_update
                }
            });
            notifications.push(plan_notification.to_string());

            if should_emit_plan_file_side_effects(state)
                && let Some(mut extra) = maybe_emit_plan_markdown_file(
                    &session_id,
                    params,
                    &plan_update_for_side_effects,
                    state,
                )
            {
                notifications.append(&mut extra);
            }
        }
    }

    AgentAction::Intercept {
        response_to_child: Some(response.to_string()),
        notifications_to_zed: notifications,
    }
}

fn handle_unknown_cursor_request(msg: &Value) -> AgentAction {
    let id = &msg["id"];
    let method = msg
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    tracing::debug!(method, "intercepting unknown _cursor/ request");

    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {}
    });

    AgentAction::Intercept {
        response_to_child: Some(response.to_string()),
        notifications_to_zed: vec![],
    }
}

fn normalize_todo_status(status: &str) -> String {
    match status {
        "completed" | "TODO_STATUS_COMPLETED" => "completed".to_string(),
        "in_progress" | "TODO_STATUS_IN_PROGRESS" => "in_progress".to_string(),
        "cancelled" | "TODO_STATUS_CANCELLED" => "cancelled".to_string(),
        _ => "pending".to_string(),
    }
}

fn should_show_in_plan(todo: &TodoItem) -> bool {
    if todo.status == "cancelled" {
        return false;
    }
    let c = todo.content.trim();
    !c.is_empty() && c != "(empty)" && c != "(placeholder)"
}

// ---------------------------------------------------------------------------
// Model list parsing
// ---------------------------------------------------------------------------

/// Parse the output of `cursor-agent --list-models` (or `cursor-agent models`).
/// Format: `id - Display Name  (current)` or `id - Display Name  (default)`.
pub fn parse_model_list(output: &str) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Available") || line.starts_with("Tip:") {
            continue;
        }
        if let Some((id, rest)) = line.split_once(" - ") {
            let name = rest
                .trim_end_matches("(current)")
                .trim_end_matches("(default)")
                .trim()
                .to_string();
            models.push(ModelInfo {
                id: id.trim().to_string(),
                name,
            });
        }
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a platform-appropriate absolute path for tests.
    /// On Unix `/foo` is absolute; on Windows we need `C:\foo`.
    fn test_abs(unix_path: &str) -> String {
        if cfg!(windows) {
            format!("C:{}", unix_path.replace('/', "\\"))
        } else {
            unix_path.to_string()
        }
    }

    #[test]
    fn forward_normal_notification() {
        let mut state = ProxyState::new();
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "hello" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward"),
        }
        assert_eq!(state.current_session_id.as_deref(), Some("sess-1"));
    }

    #[test]
    fn forward_response() {
        let mut state = ProxyState::new();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "foo": "bar" }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward"),
        }
    }

    #[test]
    fn intercept_update_todos_no_merge() {
        let mut state = ProxyState::new();
        state.current_session_id = Some("sess-1".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "_cursor/update_todos",
            "params": {
                "toolCallId": "tool-abc",
                "todos": [
                    { "id": "1", "content": "Research", "status": "pending" },
                    { "id": "2", "content": "Write summary", "status": "in_progress" }
                ],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                response_to_child,
                notifications_to_zed,
            } => {
                let resp: Value =
                    serde_json::from_str(response_to_child.as_ref().unwrap()).unwrap();
                assert_eq!(resp["id"], 5);

                assert_eq!(notifications_to_zed.len(), 1);
                let notif: Value = serde_json::from_str(&notifications_to_zed[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0]["content"], "Research");
                assert_eq!(entries[0]["status"], "pending");
                assert_eq!(entries[1]["content"], "Write summary");
                assert_eq!(entries[1]["status"], "in_progress");
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn intercept_update_todos_merge() {
        let mut state = ProxyState::new();
        state.current_session_id = Some("sess-1".to_string());
        state.todos.push(TodoItem {
            id: "1".to_string(),
            content: "Old task".to_string(),
            status: "pending".to_string(),
        });

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "_cursor/update_todos",
            "params": {
                "toolCallId": "tool-xyz",
                "todos": [
                    { "id": "1", "content": "Old task", "status": "completed" },
                    { "id": "2", "content": "New task", "status": "pending" }
                ],
                "merge": true
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                let notif: Value = serde_json::from_str(&notifications_to_zed[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0]["status"], "completed");
                assert_eq!(entries[1]["content"], "New task");
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn intercept_unknown_cursor_method() {
        let mut state = ProxyState::new();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "_cursor/ask_question",
            "params": { "question": "what?" }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                response_to_child,
                notifications_to_zed,
            } => {
                let resp: Value =
                    serde_json::from_str(response_to_child.as_ref().unwrap()).unwrap();
                assert_eq!(resp["id"], 10);
                assert!(notifications_to_zed.is_empty());
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn no_plan_emitted_without_session_id() {
        let mut state = ProxyState::new();

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "_cursor/update_todos",
            "params": {
                "todos": [{ "id": "1", "content": "Task", "status": "pending" }],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                assert!(notifications_to_zed.is_empty());
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn session_id_tracked_from_notifications() {
        let mut state = ProxyState::new();

        let update = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "abc-123",
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": { "type": "text", "text": "thinking" }
                }
            }
        });
        process_agent_message(&update, &mut state);
        assert_eq!(state.current_session_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn strip_meta_from_initialize() {
        let mut state = ProxyState::new();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": { "readTextFile": true, "writeTextFile": true },
                    "terminal": true,
                    "_meta": { "terminal_output": true }
                },
                "clientInfo": { "name": "zed", "version": "0.180.0" }
            }
        });

        match process_client_message(&msg, &mut state) {
            ClientAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                assert!(parsed.pointer("/params/clientCapabilities/_meta").is_none());
                assert_eq!(
                    parsed.pointer("/params/clientCapabilities/terminal"),
                    Some(&json!(true))
                );
            }
            _ => panic!("expected ForwardPatched"),
        }
        assert!(state.stored_init_request.is_some());
    }

    #[test]
    fn initialize_without_meta_still_stored() {
        let mut state = ProxyState::new();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": { "fs": { "readTextFile": true }, "terminal": true }
            }
        });

        match process_client_message(&msg, &mut state) {
            ClientAction::ForwardPatched(patched) => {
                assert!(state.stored_init_request.is_some());
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                assert_eq!(parsed["method"], "initialize");
            }
            _ => panic!("expected ForwardPatched"),
        }
    }

    #[test]
    fn set_model_intercept() {
        let mut state = ProxyState::new();
        state.child_session_id = Some("child-old".to_string());
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "session/set_model",
            "params": { "sessionId": "s1", "modelId": "opus-4.6-thinking" }
        });

        match process_client_message(&msg, &mut state) {
            ClientAction::Respond {
                response_to_zed,
                restart_with_model,
            } => {
                let resp: Value = serde_json::from_str(&response_to_zed).unwrap();
                assert_eq!(resp["id"], 5);
                assert_eq!(restart_with_model, Some("opus-4.6-thinking".to_string()));
                assert_eq!(state.selected_model.as_deref(), Some("opus-4.6-thinking"));
                assert_eq!(state.zed_session_id.as_deref(), Some("s1"));
                assert_eq!(state.current_session_id.as_deref(), Some("s1"));
                assert!(state.child_session_id.is_none());
            }
            _ => panic!("expected Respond"),
        }
    }

    #[test]
    fn inject_models_into_session_new_response() {
        let mut state = ProxyState::new();
        state.models = vec![
            ModelInfo {
                id: "auto".into(),
                name: "Auto".into(),
            },
            ModelInfo {
                id: "opus-4.6".into(),
                name: "Claude 4.6 Opus".into(),
            },
        ];
        state.selected_model = Some("opus-4.6".into());

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "sessionId": "sess-1",
                "modes": { "currentModeId": "agent", "availableModes": [] }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let models = &parsed["result"]["models"];
                assert_eq!(models["currentModelId"], "opus-4.6");
                let avail = models["availableModels"].as_array().unwrap();
                assert_eq!(avail.len(), 2);
                assert_eq!(avail[0]["modelId"], "auto");
                assert_eq!(avail[1]["name"], "Claude 4.6 Opus");
            }
            _ => panic!("expected ForwardPatched"),
        }
    }

    #[test]
    fn no_model_injection_when_empty() {
        let mut state = ProxyState::new();

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "sessionId": "sess-1",
                "modes": { "currentModeId": "agent", "availableModes": [] }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward when no models configured"),
        }
    }

    #[test]
    fn parse_model_list_output() {
        let output = r#"Available models

auto - Auto  (current)
opus-4.6-thinking - Claude 4.6 Opus (Thinking)  (default)
sonnet-4.5 - Claude 4.5 Sonnet
gpt-5.2 - GPT-5.2

Tip: use --model <id> to switch.
"#;
        let models = parse_model_list(output);
        assert_eq!(models.len(), 4);
        assert_eq!(models[0].id, "auto");
        assert_eq!(models[0].name, "Auto");
        assert_eq!(models[1].id, "opus-4.6-thinking");
        assert_eq!(models[1].name, "Claude 4.6 Opus (Thinking)");
        assert_eq!(models[2].id, "sonnet-4.5");
        assert_eq!(models[2].name, "Claude 4.5 Sonnet");
    }

    #[test]
    fn normalize_todo_status_variants() {
        assert_eq!(normalize_todo_status("pending"), "pending");
        assert_eq!(normalize_todo_status("TODO_STATUS_PENDING"), "pending");
        assert_eq!(normalize_todo_status("in_progress"), "in_progress");
        assert_eq!(
            normalize_todo_status("TODO_STATUS_IN_PROGRESS"),
            "in_progress"
        );
        assert_eq!(normalize_todo_status("completed"), "completed");
        assert_eq!(normalize_todo_status("TODO_STATUS_COMPLETED"), "completed");
        assert_eq!(normalize_todo_status("cancelled"), "cancelled");
        assert_eq!(normalize_todo_status("TODO_STATUS_CANCELLED"), "cancelled");
        assert_eq!(normalize_todo_status("unknown"), "pending");
    }

    #[test]
    fn parse_env_flag_defaults_and_overrides() {
        assert!(parse_env_flag(None, true));
        assert!(!parse_env_flag(None, false));

        assert!(!parse_env_flag(Some("0"), true));
        assert!(!parse_env_flag(Some("false"), true));
        assert!(!parse_env_flag(Some("off"), true));

        assert!(parse_env_flag(Some("1"), false));
        assert!(parse_env_flag(Some("true"), false));
        assert!(parse_env_flag(Some("on"), false));

        assert!(parse_env_flag(Some("unexpected"), true));
        assert!(!parse_env_flag(Some("unexpected"), false));
    }

    #[test]
    fn cancelled_and_empty_todos_filtered_from_plan() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("s1".to_string());

        // Use _cursor/update_todos (the request path) which is the canonical
        // handler. The tool_call session/update path for updateTodos is
        // intentionally skipped to avoid clobbering merged state.
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "_cursor/update_todos",
            "params": {
                "todos": [
                    { "id": "a", "content": "(empty)", "status": "TODO_STATUS_CANCELLED" },
                    { "id": "b", "content": "(empty)", "status": "TODO_STATUS_CANCELLED" },
                    { "id": "c", "content": "Real task", "status": "TODO_STATUS_PENDING" }
                ],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                assert_eq!(notifications_to_zed.len(), 1);
                let notif: Value = serde_json::from_str(&notifications_to_zed[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0]["content"], "Real task");
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn session_prompt_extracts_text() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "session/prompt",
            "params": {
                "sessionId": "zed-sess",
                "prompt": [
                    { "type": "text", "text": "What is Rust?" }
                ]
            }
        });

        match process_client_message(&msg, &mut state) {
            ClientAction::ForwardWithPrompt { prompt_text, .. } => {
                assert_eq!(prompt_text, "What is Rust?");
            }
            _ => panic!("expected ForwardWithPrompt"),
        }
    }

    #[test]
    fn remap_child_session_id() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-id".to_string());
        state.current_session_id = Some("zed-id".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "child-id",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "hello" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                assert_eq!(parsed["params"]["sessionId"], "zed-id");
            }
            _ => panic!("expected ForwardPatched with remapped session ID"),
        }
    }

    #[test]
    fn no_remap_when_ids_match() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("same-id".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "same-id",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "hello" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward (no remap needed)"),
        }
    }

    #[test]
    fn new_session_deferred_creation() {
        let mut state = ProxyState::new();
        state
            .pending_new_session_cwds
            .insert("3".to_string(), "/test".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "sessionId": "new-sess-id",
                "modes": { "currentModeId": "agent", "availableModes": [] }
            }
        });

        // track_new_session stores in pending_sessions, not created yet.
        track_new_session(&msg, &mut state);
        assert_eq!(state.zed_session_id.as_deref(), Some("new-sess-id"));
        assert_eq!(
            state
                .pending_sessions
                .get("new-sess-id")
                .map(String::as_str),
            Some("/test")
        );

        // take_pending_session removes from pending and returns it.
        let taken = take_pending_session(&mut state);
        assert!(taken.is_some());
        let (sid, cwd) = taken.unwrap();
        assert_eq!(sid, "new-sess-id");
        assert_eq!(cwd, "/test");

        // Second take returns None.
        assert!(take_pending_session(&mut state).is_none());
    }

    #[test]
    fn parse_plan_entries_from_markdown() {
        let plan = "# Create jokes.txt\n\nTwo-step task.\n\n- Create `jokes.txt`\n- Add a joke";
        let entries = parse_plan_entries(plan);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["content"], "Create `jokes.txt`");
        assert_eq!(entries[0]["status"], "pending");
        assert_eq!(entries[1]["content"], "Add a joke");
    }

    #[test]
    fn parse_plan_entries_with_checkboxes() {
        let plan = "- [x] Done task\n- [ ] Pending task\n- In progress";
        let entries = parse_plan_entries(plan);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["status"], "completed");
        assert_eq!(entries[1]["status"], "pending");
        assert_eq!(entries[2]["status"], "pending");
    }

    #[test]
    fn parse_plan_entries_numbered_list() {
        let plan = "# Jokes File Lifecycle\n\n1. Create an empty `jokes.txt` in the workspace root.\n2. Write a joke into `jokes.txt`.\n3. Delete `jokes.txt`.\n";
        let entries = parse_plan_entries(plan);
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0]["content"],
            "Create an empty `jokes.txt` in the workspace root."
        );
        assert_eq!(entries[1]["content"], "Write a joke into `jokes.txt`.");
        assert_eq!(entries[2]["content"], "Delete `jokes.txt`.");
    }

    #[test]
    fn parse_todos_from_code_block_json() {
        let plan = r#"# My Plan

Some description here.

- [ ] Fake checklist item (should be ignored)

```todos
[{"id": "task-1", "content": "First real task", "status": "pending"}, {"id": "task-2", "content": "Second task", "status": "in_progress"}]
```
"#;
        let entries = parse_todos_from_code_block(plan).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["content"], "First real task");
        assert_eq!(entries[0]["status"], "pending");
        assert_eq!(entries[1]["content"], "Second task");
        assert_eq!(entries[1]["status"], "in_progress");
    }

    #[test]
    fn strip_todos_code_block_cleans_markdown() {
        let plan = "# My Plan\n\nDescription.\n\n```todos\n[{\"id\": \"t1\", \"content\": \"Task\"}]\n```\n";
        let clean = strip_todos_code_block(plan);
        assert!(!clean.contains("```todos"));
        assert!(clean.contains("# My Plan"));
        assert!(clean.contains("Description."));
    }

    #[test]
    fn detect_plan_from_agent_message_checklist() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "## To-Do Plan\n\n- [ ] Create `joke.txt`\n- [ ] Add joke\n- [ ] Delete `joke.txt`"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 3);
                assert_eq!(entries[0]["content"], "Create `joke.txt`");
                assert_eq!(entries[1]["content"], "Add joke");
                assert_eq!(entries[2]["content"], "Delete `joke.txt`");
                let meta = &notif["params"]["update"]["_meta"]["cursor-acp"];
                assert_eq!(meta["sourceTool"], "agent_message_chunk");
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn detect_plan_from_agent_message_does_not_emit_plan_file() {
        // Agent message detection should emit a plan notification but NOT
        // plan file messages. Plan files are only created from explicit
        // CreatePlan tool calls.
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = true;
        state.emit_plan_file_messages = true;

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "## To-Do Plan\n\n- [ ] Create `joke.txt`\n- [ ] Add joke\n- [ ] Delete `joke.txt`"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                // Should only have the plan notification, no file-related messages.
                assert_eq!(extra_notifications.len(), 1);

                let plan_notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(
                    plan_notif
                        .pointer("/params/update/sessionUpdate")
                        .and_then(Value::as_str),
                    Some("plan")
                );

                let has_file_write = extra_notifications.iter().any(|s| {
                    serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                        v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                    })
                });
                assert!(!has_file_write, "agent messages should not emit plan files");
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn agent_message_plan_updates_only_emit_plan_notification() {
        // When agent messages are detected as plans, they should only emit
        // plan notifications, never file write requests or "saved" messages.
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = true;
        state.emit_plan_file_messages = true;

        let msg1 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "## Plan\n\n- [ ] Step one\n- [ ] Step two"
                    }
                }
            }
        });

        match process_agent_message(&msg1, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(
                    extra_notifications.len(),
                    1,
                    "should only emit plan notification"
                );
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(
                    notif
                        .pointer("/params/update/sessionUpdate")
                        .and_then(Value::as_str),
                    Some("plan")
                );
            }
            _ => panic!("expected ForwardWithExtra"),
        }

        // Second message with more steps should also only produce plan notification.
        let msg2 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "\n- [ ] Step three"
                    }
                }
            }
        });

        match process_agent_message(&msg2, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(
                    notif
                        .pointer("/params/update/sessionUpdate")
                        .and_then(Value::as_str),
                    Some("plan")
                );
            }
            _ => panic!("expected ForwardWithExtra on second call"),
        }
    }

    #[test]
    fn ignore_non_plan_bullet_lists_in_agent_message() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "Key points:\n- Keep output concise\n- Prefer deterministic tests"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward for non-plan bullets"),
        }
    }

    #[test]
    fn detect_streamed_agent_message_plan_once() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());

        let msg1 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "## To-Do Plan\n\n- [ ] Create file\n" }
                }
            }
        });
        let msg2 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "- [ ] Add joke\n- [ ] Delete file\n" }
                }
            }
        });
        let msg3 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "Done when all checks pass." }
                }
            }
        });

        match process_agent_message(&msg1, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected initial chunk to forward"),
        }
        match process_agent_message(&msg2, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 3);
            }
            _ => panic!("expected plan emission once enough chunks arrive"),
        }
        match process_agent_message(&msg3, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected deduped follow-up chunk to forward"),
        }
    }

    #[test]
    fn create_plan_tool_call_emits_plan() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "Create Plan: jokes",
                    "rawInput": {
                        "_toolName": "createPlan",
                        "name": "jokes",
                        "plan": "# Jokes\n\n- Create file\n- Add joke"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(notif["params"]["sessionId"], "zed-sess");
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0]["content"], "Create file");

                let meta = &notif["params"]["update"]["_meta"]["cursor-acp"];
                assert_eq!(meta["sourceTool"], "createPlan");
                assert_eq!(meta["planMarkdown"], "# Jokes\n\n- Create file\n- Add joke");
                assert_eq!(meta["planName"], "jokes");
                assert_eq!(
                    meta["derivedTodosMarkdown"],
                    "- [ ] Create file\n- [ ] Add joke\n"
                );
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn create_plan_can_emit_plan_file_write_request() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = true;
        state.emit_plan_file_messages = true;

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "Create Plan: jokes",
                    "rawInput": {
                        "_toolName": "createPlan",
                        "name": "jokes",
                        "plan": "# Jokes\n\n- Create file\n- Add joke"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                // We expect the synthesized plan update, a synthesized fs write request,
                // and a synthesized message with a file link. Some configurations may
                // emit an additional fs/read_text_file or similar internal request, so
                // assert a minimum and then locate the ones we care about.
                assert!(
                    extra_notifications.len() >= 3,
                    "expected at least plan + fs write + msg"
                );

                let plan_notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(
                    plan_notif.pointer("/params/update/sessionUpdate").unwrap(),
                    "plan"
                );

                let write_ix = extra_notifications
                    .iter()
                    .position(|s| {
                        serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                            v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                        })
                    })
                    .expect("missing fs/write_text_file request");
                let write_req: Value =
                    serde_json::from_str(&extra_notifications[write_ix]).unwrap();
                assert_eq!(write_req["method"], "fs/write_text_file");
                assert_eq!(write_req["params"]["sessionId"], "zed-sess");
                assert!(
                    write_req["params"]["path"]
                        .as_str()
                        .unwrap()
                        .ends_with(".md")
                );
                assert_eq!(
                    write_req["params"]["content"],
                    "# Jokes\n\n- Create file\n- Add joke"
                );

                let msg_ix = extra_notifications
                    .iter()
                    .position(|s| {
                        serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                            v.pointer("/params/update/sessionUpdate")
                                .and_then(Value::as_str)
                                == Some("agent_message_chunk")
                        })
                    })
                    .expect("missing agent_message_chunk notification");
                let msg_notif: Value = serde_json::from_str(&extra_notifications[msg_ix]).unwrap();
                assert_eq!(
                    msg_notif.pointer("/params/update/sessionUpdate").unwrap(),
                    "agent_message_chunk"
                );
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn create_plan_can_emit_plan_file_message_without_write() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = false;
        state.emit_plan_file_messages = true;

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "Create Plan: jokes",
                    "rawInput": {
                        "_toolName": "createPlan",
                        "name": "jokes",
                        "plan": "# Jokes\n\n- Create file\n- Add joke"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 2);
                let has_write = extra_notifications.iter().any(|s| {
                    serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                        v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                    })
                });
                assert!(!has_write, "message-only mode should not write file");
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn update_todos_tool_call_skipped_defers_to_request_handler() {
        // updateTodos tool_call updates (session/update with _toolName
        // "updateTodos") are intentionally ignored — all todo state is handled
        // by `handle_update_todos` via the `_cursor/update_todos` request.
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state.todos = vec![
            TodoItem {
                id: "task-a".to_string(),
                content: "First task".to_string(),
                status: "pending".to_string(),
            },
            TodoItem {
                id: "task-b".to_string(),
                content: "Second task".to_string(),
                status: "pending".to_string(),
            },
        ];

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-2",
                    "title": "Update TODOs",
                    "rawInput": {
                        "_toolName": "updateTodos",
                        "todos": [
                            { "id": "task-a", "content": "First task", "status": "completed" }
                        ],
                        "merge": true
                    }
                }
            }
        });

        assert!(
            matches!(
                process_agent_message(&msg, &mut state),
                AgentAction::Forward
            ),
            "updateTodos tool_call should pass through as Forward"
        );
        assert_eq!(
            state.todos.len(),
            2,
            "tool_call path must NOT mutate state.todos"
        );
        assert_eq!(
            state.todos[0].status, "pending",
            "task-a must remain pending (not modified by tool_call path)"
        );
    }

    #[test]
    fn update_todos_tool_call_without_todos_ignored() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-3",
                    "title": "Update TODOs",
                    "rawInput": {
                        "_toolName": "updateTodos"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward for updateTodos without todos"),
        }
    }

    #[test]
    fn update_todos_uses_zed_session_id() {
        let mut state = ProxyState::new();
        // Simulate: session/new gave us "zed-id", child restart changed internal ID
        state.zed_session_id = Some("zed-id".to_string());
        state.current_session_id = Some("child-id".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "_cursor/update_todos",
            "params": {
                "todos": [{ "id": "1", "content": "Task A", "status": "pending" }],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                assert_eq!(notifications_to_zed.len(), 1);
                let notif: Value = serde_json::from_str(&notifications_to_zed[0]).unwrap();
                // Plan notification should use zed_session_id, not child_id
                assert_eq!(notif["params"]["sessionId"], "zed-id");
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn update_todos_request_can_emit_plan_file_message_without_write() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-id".to_string());
        state.current_session_id = Some("zed-id".to_string());
        state
            .session_cwds
            .insert("zed-id".to_string(), test_abs("/tmp"));
        state.emit_plan_files = false;
        state.emit_plan_file_messages = true;

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "_cursor/update_todos",
            "params": {
                "todos": [{ "id": "1", "content": "Task A", "status": "pending" }],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                assert_eq!(notifications_to_zed.len(), 2);
                let has_write = notifications_to_zed.iter().any(|s| {
                    serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                        v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                    })
                });
                assert!(!has_write, "message-only mode should not write file");
            }
            _ => panic!("expected Intercept"),
        }
    }

    #[test]
    fn synthesize_read_for_edit_tool_call() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        let abs_path = test_abs("/tmp/foo.txt");

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": format!("Edit `{abs_path}`"),
                    "kind": "edit",
                    "status": "pending",
                    "rawInput": { "path": &abs_path },
                    "locations": [{ "path": &abs_path }]
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let req: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(req["method"], "fs/read_text_file");
                assert_eq!(req["params"]["sessionId"], "zed-sess");
                assert_eq!(req["params"]["path"], abs_path);
                assert!(req["id"].as_i64().unwrap() < -20000);
            }
            _ => panic!("expected ForwardWithExtra with read request"),
        }
        assert_eq!(state.suppress_zed_response_ids.len(), 1);
    }

    #[test]
    fn synthesize_read_normalizes_relative_paths_using_session_cwd() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        let cwd = test_abs("/work");
        state
            .session_cwds
            .insert("zed-sess".to_string(), cwd.clone());
        let expected = PathBuf::from(&cwd)
            .join("src/foo.md")
            .to_string_lossy()
            .into_owned();

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": "Edit file",
                    "kind": "edit",
                    "status": "pending",
                    "rawInput": { "path": "src/foo.md" },
                    "locations": [{ "path": "src/foo.md" }]
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                assert_eq!(extra_notifications.len(), 1);
                let req: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                assert_eq!(req["method"], "fs/read_text_file");
                assert_eq!(req["params"]["path"], expected);
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn no_synthesize_write_for_completed_edit() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        let abs_path = test_abs("/tmp/foo.txt");

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": "completed",
                    "content": [{
                        "type": "diff",
                        "path": &abs_path,
                        "oldText": "old content",
                        "newText": "new content"
                    }]
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward (no synthesized write for completed edits)"),
        }
        assert!(state.suppress_zed_response_ids.is_empty());
    }

    #[test]
    fn no_synthesize_fs_for_execute_tool_call() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-2",
                    "title": "Run command",
                    "kind": "execute",
                    "status": "pending",
                    "rawInput": { "command": "ls" }
                }
            }
        });

        // Execute tool calls are now intercepted by terminal synthesis
        // (not fs synthesis), so we expect ForwardPatched with terminal_info.
        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                assert!(parsed["params"]["update"]["_meta"]["terminal_info"].is_object());
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        assert!(state.suppress_zed_response_ids.is_empty());
    }

    #[test]
    fn suppress_zed_response_to_internal_request() {
        let mut state = ProxyState::new();
        let internal_id = json!(-20001);
        state.suppress_zed_response_ids.push(internal_id.clone());

        let response = json!({
            "jsonrpc": "2.0",
            "id": -20001,
            "result": { "content": "file contents" }
        });

        match process_client_message(&response, &mut state) {
            ClientAction::Drop => {}
            _ => panic!("expected Drop for suppressed response"),
        }
        assert!(state.suppress_zed_response_ids.is_empty());
    }

    #[test]
    fn extract_markdown_heading_basic() {
        assert_eq!(
            extract_markdown_heading("## My Plan\n\n- [ ] Step one"),
            Some("My Plan".to_string())
        );
        assert_eq!(
            extract_markdown_heading("# Fix the Bug\n- do stuff"),
            Some("Fix the Bug".to_string())
        );
        assert_eq!(
            extract_markdown_heading("### Deeply Nested\ntext"),
            Some("Deeply Nested".to_string())
        );
        assert_eq!(
            extract_markdown_heading("- [ ] No heading here\n- [ ] Just items"),
            None
        );
        assert_eq!(extract_markdown_heading(""), None);
    }

    #[test]
    fn agent_message_plan_extracts_heading_into_meta() {
        // Agent message detection should extract the markdown heading as planName
        // in the meta, but should NOT emit file writes.
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = true;
        state.emit_plan_file_messages = true;

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "## Fix Edit Bug\n\n- [ ] Find root cause\n- [ ] Apply fix\n- [ ] Test"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                // Should not have any file write requests.
                let has_write = extra_notifications.iter().any(|s| {
                    serde_json::from_str::<Value>(s).ok().is_some_and(|v| {
                        v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                    })
                });
                assert!(!has_write, "agent messages should not emit file writes");

                // The plan notification should have the heading in meta.
                let plan_notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                let plan_name = plan_notif
                    .pointer("/params/update/_meta/cursor-acp/planName")
                    .and_then(Value::as_str);
                assert_eq!(plan_name, Some("Fix Edit Bug"));
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn update_todos_plan_file_uses_prompt_fallback_name() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state.current_session_id = Some("zed-sess".to_string());
        state
            .session_cwds
            .insert("zed-sess".to_string(), test_abs("/tmp"));
        state.emit_plan_files = true;
        state.emit_plan_file_messages = true;
        state
            .session_first_prompt
            .insert("zed-sess".to_string(), "fix the doubling bug".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "_cursor/update_todos",
            "params": {
                "todos": [
                    { "id": "1", "content": "Find root cause", "status": "pending" },
                    { "id": "2", "content": "Apply fix", "status": "pending" }
                ],
                "merge": false
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Intercept {
                notifications_to_zed,
                ..
            } => {
                let write_notif = notifications_to_zed.iter().find_map(|s| {
                    serde_json::from_str::<Value>(s).ok().filter(|v| {
                        v.get("method").and_then(Value::as_str) == Some("fs/write_text_file")
                    })
                });
                let path = write_notif.expect("missing fs/write_text_file")["params"]["path"]
                    .as_str()
                    .unwrap()
                    .to_string();
                assert!(
                    path.contains("fix-the-doubling-bug"),
                    "expected prompt text in filename, got: {path}"
                );
            }
            _ => panic!("expected Intercept"),
        }
    }

    // -----------------------------------------------------------------------
    // Terminal synthesis for execute tool calls
    // -----------------------------------------------------------------------

    #[test]
    fn execute_tool_call_without_command_is_buffered() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": "Terminal",
                    "kind": "execute",
                    "rawInput": {}
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Drop => {}
            other => panic!("expected Drop, got {:?}", std::mem::discriminant(&other)),
        }
        assert!(state.buffered_execute_tool_calls.contains_key("tc-1"));
    }

    #[test]
    fn execute_tool_call_with_command_gets_terminal_info() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": "`echo hello`",
                    "kind": "execute",
                    "rawInput": { "command": "echo hello" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let update = &parsed["params"]["update"];

                // Should have _meta.terminal_info
                let terminal_info = &update["_meta"]["terminal_info"];
                assert!(terminal_info["terminal_id"].is_string());
                let terminal_id = terminal_info["terminal_id"].as_str().unwrap();

                // Should have content with terminal reference
                let content = update["content"].as_array().unwrap();
                assert_eq!(content.len(), 1);
                assert_eq!(content[0]["type"], "terminal");
                assert_eq!(content[0]["terminalId"], terminal_id);

                // Title should be the command in a code block
                assert!(update["title"].as_str().unwrap().contains("echo hello"));

                // Terminal ID should be tracked
                assert_eq!(
                    state.terminal_ids.get("tc-1").map(String::as_str),
                    Some(terminal_id)
                );
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn execute_tool_call_with_cwd_includes_cwd_in_terminal_info() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-2",
                    "title": "`ls`",
                    "kind": "execute",
                    "rawInput": { "command": "ls", "cd": "/tmp/myproject" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let terminal_info = &parsed["params"]["update"]["_meta"]["terminal_info"];
                assert_eq!(terminal_info["cwd"], "/tmp/myproject");
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn execute_completed_update_injects_terminal_output_and_exit() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state
            .terminal_ids
            .insert("tc-1".to_string(), "term-abc".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": "completed",
                    "rawOutput": {
                        "exitCode": 0,
                        "stdout": "hello world\n",
                        "stderr": ""
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let meta = &parsed["params"]["update"]["_meta"];

                // Should have terminal_output with stdout
                let term_out = &meta["terminal_output"];
                assert_eq!(term_out["terminal_id"], "term-abc");
                assert_eq!(term_out["data"], "hello world\n");

                // Should have terminal_exit with exit code
                let term_exit = &meta["terminal_exit"];
                assert_eq!(term_exit["terminal_id"], "term-abc");
                assert_eq!(term_exit["exit_code"], 0);

                // Terminal ID should be cleaned up
                assert!(!state.terminal_ids.contains_key("tc-1"));
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn execute_completed_update_combines_stdout_and_stderr() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state
            .terminal_ids
            .insert("tc-1".to_string(), "term-xyz".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": "completed",
                    "rawOutput": {
                        "exitCode": 1,
                        "stdout": "partial output",
                        "stderr": "error: something failed"
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let data = parsed["params"]["update"]["_meta"]["terminal_output"]["data"]
                    .as_str()
                    .unwrap();
                assert!(data.contains("partial output"));
                assert!(data.contains("error: something failed"));

                let exit_code =
                    parsed["params"]["update"]["_meta"]["terminal_exit"]["exit_code"].as_u64();
                assert_eq!(exit_code, Some(1));
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn execute_in_progress_update_not_intercepted_without_raw_output() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());
        state
            .terminal_ids
            .insert("tc-1".to_string(), "term-abc".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": "in_progress"
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            other => panic!("expected Forward, got {:?}", std::mem::discriminant(&other)),
        }
        // Terminal ID should still be tracked
        assert!(state.terminal_ids.contains_key("tc-1"));
    }

    #[test]
    fn non_execute_tool_call_not_intercepted() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-read",
                    "title": "Read file",
                    "kind": "read",
                    "rawInput": { "path": "/tmp/foo.txt" }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            other => panic!("expected Forward, got {:?}", std::mem::discriminant(&other)),
        }
        assert!(state.terminal_ids.is_empty());
    }

    #[test]
    fn execute_full_lifecycle() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

        // Step 1: tool_call with no command → buffered
        let msg1 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-lifecycle",
                    "title": "Terminal",
                    "kind": "execute",
                    "rawInput": {}
                }
            }
        });
        assert!(matches!(
            process_agent_message(&msg1, &mut state),
            AgentAction::Drop
        ));

        // Step 2: tool_call with command → patched with terminal_info
        let msg2 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-lifecycle",
                    "title": "`echo hello`",
                    "kind": "execute",
                    "rawInput": { "command": "echo hello" }
                }
            }
        });
        let terminal_id = match process_agent_message(&msg2, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                parsed["params"]["update"]["_meta"]["terminal_info"]["terminal_id"]
                    .as_str()
                    .unwrap()
                    .to_string()
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        };
        assert!(
            !state
                .buffered_execute_tool_calls
                .contains_key("tc-lifecycle")
        );

        // Step 3: tool_call_update in_progress → forwarded as-is
        let msg3 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-lifecycle",
                    "status": "in_progress"
                }
            }
        });
        assert!(matches!(
            process_agent_message(&msg3, &mut state),
            AgentAction::Forward
        ));

        // Step 4: tool_call_update completed → patched with terminal_output/exit
        let msg4 = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "zed-sess",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-lifecycle",
                    "status": "completed",
                    "rawOutput": {
                        "exitCode": 0,
                        "stdout": "hello\n",
                        "stderr": ""
                    }
                }
            }
        });
        match process_agent_message(&msg4, &mut state) {
            AgentAction::ForwardPatched(patched) => {
                let parsed: Value = serde_json::from_str(&patched).unwrap();
                let meta = &parsed["params"]["update"]["_meta"];
                assert_eq!(meta["terminal_output"]["terminal_id"], terminal_id);
                assert_eq!(meta["terminal_output"]["data"], "hello\n");
                assert_eq!(meta["terminal_exit"]["terminal_id"], terminal_id);
                assert_eq!(meta["terminal_exit"]["exit_code"], 0);
            }
            other => panic!(
                "expected ForwardPatched, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        // Terminal ID should be cleaned up after completion
        assert!(!state.terminal_ids.contains_key("tc-lifecycle"));
    }
}
