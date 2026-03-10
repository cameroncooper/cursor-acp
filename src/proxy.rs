use std::collections::HashMap;

use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
}

/// Tracks state needed for interception (session ID, accumulated todos, models).
pub struct ProxyState {
    current_session_id: Option<String>,
    todos: Vec<TodoItem>,
    pub models: Vec<ModelInfo>,
    pub selected_model: Option<String>,
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
}

struct TodoItem {
    id: String,
    content: String,
    status: String,
}

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
            models: Vec::new(),
            selected_model: None,
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

// ---------------------------------------------------------------------------
// Client → Agent direction
// ---------------------------------------------------------------------------

/// Inspect a JSON-RPC message from Zed and decide what to do with it.
pub fn process_client_message(msg: &Value, state: &mut ProxyState) -> ClientAction {
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
            if let (Some(id), Some(cwd)) = (
                msg.get("id").map(|v| v.to_string()),
                msg.pointer("/params/cwd").and_then(Value::as_str),
            ) {
                state.pending_new_session_cwds.insert(id, cwd.to_string());
            }
            ClientAction::Forward
        }
        Some("session/prompt") => {
            if let Some(session_id) = msg.pointer("/params/sessionId").and_then(Value::as_str) {
                state.zed_session_id = Some(session_id.to_string());
            }
            let forwarded = remap_client_session_id(msg, state);
            if let Some(prompt_text) = extract_prompt_text(msg) {
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
        return inject_session_list_capability(msg);
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

fn inject_session_list_capability(msg: &Value) -> AgentAction {
    let mut patched = msg.clone();
    if let Some(caps) = patched.pointer_mut("/result/agentCapabilities")
        && let Some(obj) = caps.as_object_mut()
    {
        obj.insert("sessionCapabilities".to_string(), json!({ "list": {} }));
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
        // Always track the child's session ID.
        state.child_session_id = Some(sid.to_string());

        if let Some(cwd) = state.pending_new_session_cwds.remove(&rid) {
            // Zed-initiated session/new: both IDs match.
            state.zed_session_id = Some(sid.to_string());
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

/// When cursor-agent sends edit tool calls, synthesize `fs/read_text_file` and
/// `fs/write_text_file` requests to Zed so its ActionLog tracks the changes
/// (enabling the "Edits" panel with accept/reject all).
///
/// - On `tool_call` with `kind: "edit"` and `status: "pending"`: send
///   `fs/read_text_file` so Zed opens the buffer with the OLD content.
/// - On `tool_call_update` with `status: "completed"` and diff content: send
///   `fs/write_text_file` so Zed applies the new content against the cached old
///   content, creating a tracked diff in the ActionLog.
fn maybe_synthesize_fs_for_edit(msg: &Value, state: &mut ProxyState) -> Option<Vec<String>> {
    let update = msg.pointer("/params/update")?;
    let update_type = update.get("sessionUpdate")?.as_str()?;
    let session_id = state.zed_session_id.clone()?;

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

            for path in &paths {
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
            if update.get("status").and_then(Value::as_str) != Some("completed") {
                return None;
            }
            let content = update.get("content")?.as_array()?;

            let mut reqs = Vec::new();
            for item in content {
                if item.get("type").and_then(Value::as_str) != Some("diff") {
                    continue;
                }
                let path = item.get("path").and_then(Value::as_str)?;
                let new_text = item.get("newText").and_then(Value::as_str)?;

                let id = state.next_internal_id();
                state.suppress_zed_response(id.clone());
                let req = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": "fs/write_text_file",
                    "params": {
                        "sessionId": session_id,
                        "path": path,
                        "content": new_text,
                    }
                });
                tracing::debug!(path, "synthesized fs/write_text_file for edit tracking");
                reqs.push(req.to_string());
            }

            if reqs.is_empty() { None } else { Some(reqs) }
        }
        _ => None,
    }
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

    let entries = match tool_name {
        "createPlan" => {
            let plan_text = update.pointer("/rawInput/plan")?.as_str()?;
            parse_plan_entries(plan_text)
        }
        "updateTodos" => {
            let todos = update.pointer("/rawInput/todos")?.as_array()?;
            if todos.is_empty() {
                return None;
            }
            let merge = update
                .pointer("/rawInput/merge")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !merge {
                state.todos.clear();
            }
            for item in todos {
                let id = item
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
                if let Some(existing) = state.todos.iter_mut().find(|t| t.id == id) {
                    existing.content = content;
                    existing.status = status;
                } else {
                    state.todos.push(TodoItem {
                        id,
                        content,
                        status,
                    });
                }
            }
            // Purge cancelled/placeholder/empty items from state entirely.
            state.todos.retain(should_show_in_plan);
            state
                .todos
                .iter()
                .map(|t| {
                    json!({
                        "content": t.content,
                        "priority": "medium",
                        "status": t.status,
                    })
                })
                .collect()
        }
        _ => return None,
    };

    if entries.is_empty() {
        return None;
    }

    let session_id = state
        .zed_session_id
        .as_deref()
        .or(state.current_session_id.as_deref())?;

    let notification = json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "plan",
                "entries": entries,
            }
        }
    });

    tracing::debug!(
        count = entries.len(),
        tool_name,
        "emitted plan from tool call"
    );
    Some(vec![notification.to_string()])
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

    if let Some(session_id) = &state.current_session_id {
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
            let plan_notification = json!({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "plan",
                        "entries": entries,
                    }
                }
            });
            notifications.push(plan_notification.to_string());
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
    fn cancelled_and_empty_todos_filtered_from_plan() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("s1".to_string());

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Update TODOs",
                    "rawInput": {
                        "_toolName": "updateTodos",
                        "todos": [
                            { "id": "a", "content": "(empty)", "status": "TODO_STATUS_CANCELLED" },
                            { "id": "b", "content": "(empty)", "status": "TODO_STATUS_CANCELLED" },
                            { "id": "c", "content": "Real task", "status": "TODO_STATUS_PENDING" }
                        ]
                    }
                }
            }
        });

        match process_agent_message(&msg, &mut state) {
            AgentAction::ForwardWithExtra {
                extra_notifications,
                ..
            } => {
                let notif: Value = serde_json::from_str(&extra_notifications[0]).unwrap();
                let entries = notif["params"]["update"]["entries"].as_array().unwrap();
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0]["content"], "Real task");
            }
            _ => panic!("expected ForwardWithExtra"),
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
            }
            _ => panic!("expected ForwardWithExtra"),
        }
    }

    #[test]
    fn update_todos_tool_call_emits_plan() {
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
                    "toolCallId": "tool-2",
                    "title": "Update TODOs: Create jokes.txt",
                    "rawInput": {
                        "_toolName": "updateTodos",
                        "todos": [
                            {
                                "id": "create-file",
                                "content": "Create jokes.txt in the workspace root",
                                "status": "TODO_STATUS_PENDING",
                                "createdAt": "1772824517287",
                                "updatedAt": "1772826101920",
                                "dependencies": []
                            },
                            {
                                "id": "add-joke",
                                "content": "Add a joke to jokes.txt",
                                "status": "TODO_STATUS_IN_PROGRESS",
                                "createdAt": "1772824517287",
                                "updatedAt": "1772826101920",
                                "dependencies": []
                            },
                            {
                                "id": "delete-file",
                                "content": "Delete jokes.txt",
                                "status": "TODO_STATUS_COMPLETED",
                                "createdAt": "1772824517287",
                                "updatedAt": "1772826101920",
                                "dependencies": []
                            }
                        ]
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
                assert_eq!(entries.len(), 3);
                assert_eq!(
                    entries[0]["content"],
                    "Create jokes.txt in the workspace root"
                );
                assert_eq!(entries[0]["status"], "pending");
                assert_eq!(entries[1]["content"], "Add a joke to jokes.txt");
                assert_eq!(entries[1]["status"], "in_progress");
                assert_eq!(entries[2]["content"], "Delete jokes.txt");
                assert_eq!(entries[2]["status"], "completed");
            }
            _ => panic!("expected ForwardWithExtra"),
        }
        // Internal state should be updated too
        assert_eq!(state.todos.len(), 3);
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
    fn synthesize_read_for_edit_tool_call() {
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
                    "title": "Edit `/tmp/foo.txt`",
                    "kind": "edit",
                    "status": "pending",
                    "rawInput": { "path": "/tmp/foo.txt" },
                    "locations": [{ "path": "/tmp/foo.txt" }]
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
                assert_eq!(req["params"]["path"], "/tmp/foo.txt");
                assert!(req["id"].as_i64().unwrap() < -20000);
            }
            _ => panic!("expected ForwardWithExtra with read request"),
        }
        assert_eq!(state.suppress_zed_response_ids.len(), 1);
    }

    #[test]
    fn synthesize_write_for_completed_edit() {
        let mut state = ProxyState::new();
        state.zed_session_id = Some("zed-sess".to_string());

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
                        "path": "/tmp/foo.txt",
                        "oldText": "old content",
                        "newText": "new content"
                    }]
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
                assert_eq!(req["method"], "fs/write_text_file");
                assert_eq!(req["params"]["sessionId"], "zed-sess");
                assert_eq!(req["params"]["path"], "/tmp/foo.txt");
                assert_eq!(req["params"]["content"], "new content");
            }
            _ => panic!("expected ForwardWithExtra with write request"),
        }
        assert_eq!(state.suppress_zed_response_ids.len(), 1);
    }

    #[test]
    fn no_synthesize_for_non_edit_tool_call() {
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

        match process_agent_message(&msg, &mut state) {
            AgentAction::Forward => {}
            _ => panic!("expected Forward for non-edit tool call"),
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
}
