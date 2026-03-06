use crate::error::ErrorKind;
use agent_client_protocol::{Diff, ToolCallStatus, ToolKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct CursorCliConfig {
    pub binary: String,
}

impl Default for CursorCliConfig {
    fn default() -> Self {
        Self {
            binary: std::env::var("CURSOR_AGENT_BIN")
                .or_else(|_| std::env::var("CURSOR_AGENT_PATH"))
                .unwrap_or_else(|_| "cursor-agent".to_string()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CursorCliRuntime {
    config: CursorCliConfig,
}

#[derive(Debug, Clone, Default)]
pub struct CursorAuthStatus {
    pub authenticated: bool,
    pub raw_output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPromptResult {
    pub text: String,
    pub meta: Value,
    pub usage: Option<CursorUsageUpdate>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CursorUsageUpdate {
    pub used: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct CursorPromptRequest {
    pub prompt: String,
    pub mode: Option<String>,
    pub sandbox: Option<String>,
    pub model: Option<String>,
    pub resume: Option<String>,
    pub cwd: Option<PathBuf>,
    pub allow_edits: bool,
    pub trust_workspace: bool,
}

#[derive(Debug, Clone)]
pub enum CursorCliEvent {
    AssistantDelta(String),
    ThinkingDelta(String),
    System { message: String, raw: Value },
    ToolCallStart(CursorCliToolCall),
    ToolCallUpdate(Box<CursorCliToolCallUpdate>),
    Result(CursorPromptResult),
    Other(Value),
}

#[derive(Debug, Clone)]
pub struct CursorCliToolCall {
    pub tool_call_id: String,
    pub title: String,
    pub kind: ToolKind,
    pub locations: Vec<String>,
    pub raw_input: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct CursorCliToolCallUpdate {
    pub tool_call_id: String,
    pub kind: ToolKind,
    pub status: ToolCallStatus,
    pub title: Option<String>,
    pub locations: Vec<String>,
    pub message: Option<String>,
    pub raw_input: Option<Value>,
    pub raw_output: Option<Value>,
    pub diff: Option<Diff>,
    pub managed_write_text: Option<String>,
    pub todo_items: Option<Vec<CursorTodoItem>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CursorTodoItem {
    pub content: String,
    pub status: String,
}

struct CommandOutput {
    stdout: String,
    stderr: String,
}

impl CursorCliRuntime {
    pub fn new(config: CursorCliConfig) -> Self {
        Self { config }
    }

    pub async fn check_auth(&self) -> Result<CursorAuthStatus, ErrorKind> {
        let output = self.run_command(vec!["status".to_string()], None).await?;
        let combined = format!("{}\n{}", output.stdout, output.stderr);
        let lower = combined.to_lowercase();
        let authenticated = lower.contains("logged in") || lower.contains("signed in");
        Ok(CursorAuthStatus {
            authenticated,
            raw_output: combined,
        })
    }

    pub async fn list_models(&self) -> Result<Vec<String>, ErrorKind> {
        let output = self.run_command(vec!["models".to_string()], None).await?;
        let mut models = Vec::new();
        for line in output.stdout.lines() {
            let Some(model_id) = parse_model_id_from_line(line) else {
                continue;
            };
            models.push(model_id);
        }
        if models.is_empty() {
            models.push("auto".to_string());
        }
        Ok(models)
    }

    pub async fn run_prompt_json(
        &self,
        request: CursorPromptRequest,
    ) -> Result<CursorPromptResult, ErrorKind> {
        let args = Self::build_prompt_args(&request, "json", false);
        let output = self.run_command(args, request.cwd).await?;
        let parsed: Value =
            serde_json::from_str(&output.stdout).map_err(|_| ErrorKind::CliProtocolViolation {
                parser_phase: "json_prompt",
                raw_excerpt: output.stdout.chars().take(600).collect(),
            })?;

        let text = parsed
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or(output.stdout.as_str())
            .to_string();

        let usage = extract_usage_update(&parsed);
        Ok(CursorPromptResult {
            text,
            meta: parsed,
            usage,
        })
    }

    pub async fn run_prompt_stream(
        &self,
        request: CursorPromptRequest,
        event_tx: Option<UnboundedSender<CursorCliEvent>>,
    ) -> Result<Vec<CursorCliEvent>, ErrorKind> {
        let args = Self::build_prompt_args(&request, "stream-json", true);
        let mut command = Command::new(&self.config.binary);
        command.args(args);
        if let Some(path) = request.cwd {
            command.current_dir(path);
        }
        command.stdin(Stdio::null());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = command.spawn().map_err(map_spawn_error)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ErrorKind::Internal("missing child stdout".to_string()))?;
        let mut reader = BufReader::new(stdout).lines();

        let mut events = Vec::new();
        let mut seen_tool_calls = HashSet::new();
        let mut last_assistant_chunk: Option<String> = None;
        let mut seen_large_assistant_chunks: HashSet<String> = HashSet::new();
        let mut assistant_streamed_text = String::new();
        while let Some(line) = reader
            .next_line()
            .await
            .map_err(|error| ErrorKind::TransientIo(error.to_string()))?
        {
            let parsed: Value =
                serde_json::from_str(&line).map_err(|_| ErrorKind::CliProtocolViolation {
                    parser_phase: "stream_json",
                    raw_excerpt: line.chars().take(400).collect(),
                })?;

            let event_type = parsed
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let normalized_event_type = normalize_event_type(event_type);
            let subtype = parsed
                .get("subtype")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let has_tool_payload = looks_like_tool_event(&parsed);

            let emit = |event: CursorCliEvent, events: &mut Vec<CursorCliEvent>| {
                if let Some(tx) = &event_tx {
                    let _ignored = tx.send(event.clone());
                }
                events.push(event);
            };

            match normalized_event_type.as_str() {
                "assistant" => {
                    for text in extract_assistant_deltas(&parsed) {
                        let normalized = normalize_text_for_dedupe(&text);
                        let streamed_normalized =
                            normalize_text_for_dedupe(&assistant_streamed_text);
                        let is_large_duplicate = normalized.chars().count() >= 24
                            && !normalized.is_empty()
                            && !seen_large_assistant_chunks.insert(normalized.clone());
                        let is_already_in_stream = !normalized.is_empty()
                            && normalized.chars().count() >= 24
                            && streamed_normalized.contains(&normalized);
                        if !text.trim().is_empty()
                            && last_assistant_chunk.as_deref() != Some(text.as_str())
                            && !is_large_duplicate
                            && !is_already_in_stream
                        {
                            assistant_streamed_text.push_str(&text);
                            last_assistant_chunk = Some(text.clone());
                            emit(CursorCliEvent::AssistantDelta(text), &mut events);
                        }
                    }
                    if has_tool_payload {
                        for event in
                            parse_tool_call_events(&parsed, "started", &mut seen_tool_calls)
                        {
                            emit(event, &mut events);
                        }
                    }
                }
                "tool_call" => {
                    for event in parse_tool_call_events(&parsed, subtype, &mut seen_tool_calls) {
                        emit(event, &mut events);
                    }
                }
                "thinking" => {
                    let text = parsed
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .unwrap_or_default()
                        .to_string();
                    if !text.is_empty() {
                        emit(CursorCliEvent::ThinkingDelta(text), &mut events);
                    }
                }
                "system" => {
                    let message = extract_system_message(&parsed, subtype);
                    if !message.trim().is_empty() && message != "System: init" && message != "init"
                    {
                        emit(
                            CursorCliEvent::System {
                                message,
                                raw: parsed,
                            },
                            &mut events,
                        );
                    }
                }
                "user" => {}
                "result" => {
                    let text = dedupe_result_text_for_stream(
                        &assistant_streamed_text,
                        extract_result_text(&parsed),
                    );
                    let should_stream_result = !text.trim().is_empty();
                    let usage = extract_usage_update(&parsed);
                    let result_event = CursorCliEvent::Result(CursorPromptResult {
                        text,
                        meta: parsed,
                        usage,
                    });
                    if should_stream_result && let Some(tx) = &event_tx {
                        let _ignored = tx.send(result_event.clone());
                    }
                    events.push(result_event);
                }
                _ => {
                    if has_tool_payload {
                        for event in parse_tool_call_events(
                            &parsed,
                            if subtype.is_empty() {
                                "started"
                            } else {
                                subtype
                            },
                            &mut seen_tool_calls,
                        ) {
                            emit(event, &mut events);
                        }
                    }
                }
            }
        }

        let status = child
            .wait()
            .await
            .map_err(|error| ErrorKind::TransientIo(error.to_string()))?;
        if !status.success() {
            return Err(ErrorKind::CliInvocationFailed {
                exit_code: status.code(),
                stderr_excerpt: "streaming prompt failed".to_string(),
            });
        }

        Ok(events)
    }

    fn build_prompt_args(
        request: &CursorPromptRequest,
        output_format: &str,
        stream_partial_output: bool,
    ) -> Vec<String> {
        let mut args = vec![
            "-p".to_string(),
            "--output-format".to_string(),
            output_format.to_string(),
        ];
        if stream_partial_output {
            args.push("--stream-partial-output".to_string());
        }
        if request.allow_edits {
            args.push("--force".to_string());
        }
        if request.trust_workspace {
            args.push("--trust".to_string());
        }
        if let Some(mode_id) = request.mode.as_ref() {
            args.push("--mode".to_string());
            args.push(mode_id.clone());
        }
        if let Some(sandbox_mode) = request.sandbox.as_ref() {
            args.push("--sandbox".to_string());
            args.push(sandbox_mode.clone());
        }
        if let Some(model_id) = request.model.as_ref() {
            args.push("--model".to_string());
            args.push(model_id.clone());
        }
        if let Some(resume_id) = request.resume.as_ref() {
            args.push("--resume".to_string());
            args.push(resume_id.clone());
        }
        args.push(request.prompt.clone());
        args
    }

    pub(crate) fn classify_cli_failure(exit_code: Option<i32>, stderr: &str) -> Option<ErrorKind> {
        let lower = stderr.to_lowercase();
        if lower.contains("cannot use this model") || lower.contains("available models") {
            return Some(parse_model_unsupported(stderr));
        }
        if lower.contains("not authenticated")
            || lower.contains("authentication")
            || lower.contains("please run: cursor-agent login")
            || lower.contains("login")
        {
            return Some(ErrorKind::AuthRequired);
        }
        if lower.contains("workspace trust required")
            || lower.contains("pass --trust")
            || lower.contains("do you trust the contents of this directory")
        {
            return Some(ErrorKind::WorkspaceTrustRequired);
        }
        if stderr.trim().is_empty() {
            return None;
        }
        Some(ErrorKind::CliInvocationFailed {
            exit_code,
            stderr_excerpt: stderr.chars().take(800).collect(),
        })
    }

    async fn run_command(
        &self,
        args: Vec<String>,
        cwd: Option<PathBuf>,
    ) -> Result<CommandOutput, ErrorKind> {
        let mut command = Command::new(&self.config.binary);
        command.args(args);
        if let Some(path) = cwd {
            command.current_dir(path);
        }
        command.stdin(Stdio::null());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let output = command.output().await.map_err(map_spawn_error)?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code();
        if !output.status.success() {
            if let Some(classified) = Self::classify_cli_failure(exit_code, &stderr) {
                return Err(classified);
            }
            return Err(ErrorKind::CliInvocationFailed {
                exit_code,
                stderr_excerpt: stderr.chars().take(800).collect(),
            });
        }

        Ok(CommandOutput { stdout, stderr })
    }
}

fn normalize_text_for_dedupe(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn dedupe_result_text_for_stream(streamed_text: &str, final_text: String) -> String {
    let streamed = normalize_text_for_dedupe(streamed_text);
    let normalized_final = normalize_text_for_dedupe(&final_text);
    let streamed_contains_final =
        !normalized_final.is_empty() && streamed.contains(&normalized_final);
    let final_contains_streamed = !streamed.is_empty() && normalized_final.contains(&streamed);
    let near_match = streamed_contains_final
        || normalized_final == streamed
        || (final_contains_streamed
            && normalized_final
                .chars()
                .count()
                .saturating_sub(streamed.chars().count())
                <= 8);
    if normalized_final.is_empty() || near_match {
        String::new()
    } else {
        final_text
    }
}

fn extract_result_text(parsed: &Value) -> String {
    if let Some(text) = parsed.get("result").and_then(Value::as_str) {
        return text.to_string();
    }
    collect_text_parts(parsed.get("result").unwrap_or(parsed))
}

fn extract_assistant_deltas(parsed: &Value) -> Vec<String> {
    let mut chunks = Vec::new();

    for candidate in [
        parsed.get("content"),
        parsed.get("delta"),
        parsed.pointer("/message/content"),
        parsed.pointer("/message/delta"),
    ]
    .into_iter()
    .flatten()
    {
        collect_text_chunks(candidate, &mut chunks);
    }

    if chunks.is_empty() {
        let fallback = collect_text_parts(parsed);
        if !fallback.is_empty() {
            chunks.push(fallback);
        }
    }

    chunks
}

fn extract_system_message(parsed: &Value, subtype: &str) -> String {
    let message = collect_text_parts(parsed.get("message").unwrap_or(parsed));
    if !message.is_empty() {
        return message;
    }
    if !subtype.is_empty() {
        return format!("System: {subtype}");
    }
    "System event".to_string()
}

fn parse_tool_call_events(
    parsed: &Value,
    subtype: &str,
    seen_tool_calls: &mut HashSet<String>,
) -> Vec<CursorCliEvent> {
    let descriptor = extract_tool_call_descriptor(parsed);
    let tool_call_id = descriptor.tool_call_id;
    let mut events = Vec::new();

    if !seen_tool_calls.contains(&tool_call_id) {
        seen_tool_calls.insert(tool_call_id.clone());
        events.push(CursorCliEvent::ToolCallStart(CursorCliToolCall {
            tool_call_id: tool_call_id.clone(),
            title: descriptor.title.clone(),
            kind: descriptor.kind,
            locations: descriptor.locations.clone(),
            raw_input: descriptor.raw_input.clone(),
        }));
    }

    events.push(CursorCliEvent::ToolCallUpdate(Box::new(
        CursorCliToolCallUpdate {
            tool_call_id,
            kind: descriptor.kind,
            status: map_tool_status(subtype),
            title: Some(descriptor.title),
            locations: descriptor.locations,
            message: descriptor.message,
            raw_input: descriptor.raw_input,
            raw_output: descriptor.raw_output,
            diff: descriptor.diff,
            managed_write_text: descriptor.managed_write_text,
            todo_items: descriptor.todo_items,
        },
    )));

    events
}

fn normalize_event_type(event_type: &str) -> String {
    event_type.to_lowercase().replace('-', "_")
}

fn looks_like_tool_event(parsed: &Value) -> bool {
    if parsed.get("tool_call").is_some() {
        return true;
    }
    parsed
        .as_object()
        .map(|obj| obj.keys().any(|key| key.ends_with("ToolCall")))
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
struct ToolCallDescriptor {
    tool_call_id: String,
    title: String,
    kind: ToolKind,
    locations: Vec<String>,
    message: Option<String>,
    raw_input: Option<Value>,
    raw_output: Option<Value>,
    diff: Option<Diff>,
    managed_write_text: Option<String>,
    todo_items: Option<Vec<CursorTodoItem>>,
}

fn extract_tool_call_descriptor(parsed: &Value) -> ToolCallDescriptor {
    let parsed_title = parsed
        .get("title")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("name").and_then(Value::as_str))
        .unwrap_or("tool");

    let mut tool_key = "tool_call".to_string();
    let mut tool_obj = parsed.get("tool_call").cloned().unwrap_or(Value::Null);

    if let Some(obj) = parsed.get("tool_call").and_then(Value::as_object) {
        // Shape A: {"tool_call":{"readToolCall":{"args":...,"result":...}}}
        if let Some((key, value)) = obj
            .iter()
            .find(|(key, value)| key.ends_with("ToolCall") && value.is_object())
        {
            tool_key = key.clone();
            tool_obj = value.clone();
        // Shape B: {"tool_call":{"name":"edit_file","args":...,"result":...}}
        } else if obj.contains_key("args") || obj.contains_key("result") || obj.contains_key("name")
        {
            if let Some(name) = obj.get("name").and_then(Value::as_str) {
                tool_key = name.to_string();
            }
            tool_obj = Value::Object(obj.clone());
        }
    }
    if tool_obj.is_null()
        && let Some(obj) = parsed.as_object()
        && let Some((key, value)) = obj.iter().find(|(key, value)| {
            key.ends_with("ToolCall") && (value.is_object() || value.is_array())
        })
    {
        tool_key = key.clone();
        tool_obj = value.clone();
    }

    let raw_input = tool_obj
        .get("args")
        .cloned()
        .or_else(|| parsed.get("args").cloned());
    let raw_output = tool_obj
        .get("result")
        .cloned()
        .or_else(|| parsed.get("result").cloned());
    let mut locations = raw_input
        .as_ref()
        .map(extract_locations)
        .unwrap_or_default();
    let raw_title = tool_obj
        .get("title")
        .and_then(Value::as_str)
        .or_else(|| tool_obj.get("name").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| prettify_tool_name(&tool_key, parsed_title));
    let kind = map_tool_kind(&tool_key);
    let mut title = format_tool_title(&raw_title, kind, &locations, raw_input.as_ref());

    let explicit_id = parsed
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("tool_call_id").and_then(Value::as_str))
        .or_else(|| tool_obj.get("tool_call_id").and_then(Value::as_str))
        .or_else(|| tool_obj.get("call_id").and_then(Value::as_str))
        .or_else(|| tool_obj.get("id").and_then(Value::as_str));
    let signature = format!(
        "{}:{}",
        tool_key,
        serde_json::to_string(raw_input.as_ref().unwrap_or(&Value::Null))
            .unwrap_or_else(|_| "null".to_string())
    );
    let tool_call_id = explicit_id
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| stable_tool_call_id(&signature));

    let message = shell_output_message(&tool_key, raw_input.as_ref(), raw_output.as_ref())
        .or_else(|| {
            raw_output
                .as_ref()
                .map(collect_text_parts)
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            parsed
                .get("message")
                .map(collect_text_parts)
                .filter(|value| !value.is_empty())
        });
    if locations.is_empty()
        && let Some(message) = &message
        && let Some(path) = extract_path_from_message(message)
    {
        locations.push(path);
        title = format_tool_title(&raw_title, kind, &locations, raw_input.as_ref());
    }
    let diff = build_diff_from_tool_payload(&tool_key, raw_input.as_ref(), raw_output.as_ref());
    let managed_write_text =
        managed_write_text_from_tool_payload(&tool_key, raw_input.as_ref(), raw_output.as_ref());
    let todo_items =
        todo_items_from_tool_payload(&tool_key, raw_input.as_ref(), raw_output.as_ref());

    ToolCallDescriptor {
        tool_call_id,
        title,
        kind,
        locations,
        message,
        raw_input,
        raw_output,
        diff,
        managed_write_text,
        todo_items,
    }
}

fn stable_tool_call_id(signature: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    signature.hash(&mut hasher);
    format!("cursor-tool-{:x}", hasher.finish())
}

fn prettify_tool_name(tool_key: &str, fallback: &str) -> String {
    if fallback != "tool" {
        return fallback.to_string();
    }
    let trimmed = tool_key
        .trim_end_matches("ToolCall")
        .trim_end_matches("_tool_call")
        .replace('_', " ");
    if trimmed.is_empty() {
        "tool".to_string()
    } else {
        trimmed
    }
}

fn format_tool_title(
    raw_title: &str,
    kind: ToolKind,
    locations: &[String],
    raw_input: Option<&Value>,
) -> String {
    if kind == ToolKind::Edit
        && let Some(path) = locations.first()
    {
        return format!("Edit {path}");
    }
    if kind == ToolKind::Execute
        && let Some(raw_input) = raw_input
        && let Some(command) = first_string(raw_input, &["command"])
    {
        return format!("Run {}", truncate_for_title(&command, 80));
    }
    raw_title.to_string()
}

fn map_tool_kind(tool_key: &str) -> ToolKind {
    let lower = tool_key.to_lowercase();
    if lower.contains("read") {
        ToolKind::Read
    } else if lower.contains("write") || lower.contains("edit") {
        ToolKind::Edit
    } else if lower.contains("delete") || lower.contains("remove") {
        ToolKind::Delete
    } else if lower.contains("rename") || lower.contains("move") {
        ToolKind::Move
    } else if lower.contains("search") || lower.contains("grep") || lower.contains("glob") {
        ToolKind::Search
    } else if lower.contains("run")
        || lower.contains("exec")
        || lower.contains("command")
        || lower.contains("shell")
        || lower.contains("terminal")
        || lower.contains("bash")
    {
        ToolKind::Execute
    } else if lower.contains("fetch") || lower.contains("http") || lower.contains("web") {
        ToolKind::Fetch
    } else if lower.contains("think") || lower.contains("plan") {
        ToolKind::Think
    } else if lower.contains("mode") {
        ToolKind::SwitchMode
    } else {
        ToolKind::Other
    }
}

fn map_tool_status(subtype: &str) -> ToolCallStatus {
    match subtype {
        "started" | "start" | "running" | "in_progress" => ToolCallStatus::InProgress,
        "completed" | "complete" | "done" | "success" | "succeeded" => ToolCallStatus::Completed,
        "failed" | "error" | "errored" => ToolCallStatus::Failed,
        _ => ToolCallStatus::Pending,
    }
}

fn extract_locations(args: &Value) -> Vec<String> {
    let mut locations = Vec::new();
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        locations.push(path.to_string());
    }
    if let Some(paths) = args.get("paths").and_then(Value::as_array) {
        for path in paths.iter().filter_map(Value::as_str) {
            locations.push(path.to_string());
        }
    }
    if let Some(path) = args.get("filePath").and_then(Value::as_str) {
        locations.push(path.to_string());
    }
    if let Some(path) = args.get("target_notebook").and_then(Value::as_str) {
        locations.push(path.to_string());
    }
    if let Some(path) = args
        .get("file")
        .and_then(|value| value.get("path"))
        .and_then(Value::as_str)
    {
        locations.push(path.to_string());
    }
    locations
}

fn build_diff_from_tool_payload(
    tool_key: &str,
    raw_input: Option<&Value>,
    raw_output: Option<&Value>,
) -> Option<Diff> {
    let args = raw_input?;
    let path = first_string(
        args,
        &[
            "path",
            "file_path",
            "filePath",
            "filepath",
            "target_file",
            "targetPath",
            "target_notebook",
            "filename",
        ],
    )?;

    let lower = tool_key.to_lowercase();
    if lower.contains("write") {
        let new_text = first_string(
            args,
            &["content", "text", "new_content", "newText", "streamContent"],
        )
        .or_else(|| extract_after_full_file_content(raw_output))?;
        return Some(Diff::new(path, new_text));
    }

    if lower.contains("edit") || lower.contains("replace") || lower.contains("patch") {
        let old_text = first_string(args, &["old_string", "oldString", "old_text", "oldText"])
            .or_else(|| extract_before_full_file_content(raw_output));
        let new_text = first_string(
            args,
            &[
                "new_string",
                "newString",
                "new_text",
                "newText",
                "streamContent",
                "content",
            ],
        )
        .or_else(|| extract_after_full_file_content(raw_output))?;
        return Some(Diff::new(path, new_text).old_text(old_text));
    }

    if lower.contains("delete") || lower.contains("remove") {
        return Some(Diff::new(path, "").old_text("[file deleted]"));
    }

    None
}

fn managed_write_text_from_tool_payload(
    tool_key: &str,
    raw_input: Option<&Value>,
    raw_output: Option<&Value>,
) -> Option<String> {
    let lower = tool_key.to_lowercase();
    if !(lower.contains("edit")
        || lower.contains("write")
        || lower.contains("delete")
        || lower.contains("remove"))
    {
        return None;
    }
    extract_after_full_file_content(raw_output).or_else(|| {
        raw_input.and_then(|args| {
            first_string(
                args,
                &["content", "text", "new_content", "newText", "streamContent"],
            )
        })
    })
}

fn todo_items_from_tool_payload(
    tool_key: &str,
    raw_input: Option<&Value>,
    raw_output: Option<&Value>,
) -> Option<Vec<CursorTodoItem>> {
    let lower = tool_key.to_lowercase();
    if !lower.contains("updatetodos") {
        return None;
    }
    let todos = raw_output
        .and_then(|value| value.get("success"))
        .and_then(|value| value.get("todos"))
        .and_then(Value::as_array)
        .or_else(|| {
            raw_input
                .and_then(|value| value.get("todos"))
                .and_then(Value::as_array)
        })?;
    let mut items = Vec::new();
    for todo in todos {
        let content = todo
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let status = todo
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if !content.is_empty() && !status.is_empty() {
            items.push(CursorTodoItem { content, status });
        }
    }
    if items.is_empty() { None } else { Some(items) }
}

fn shell_output_message(
    tool_key: &str,
    raw_input: Option<&Value>,
    raw_output: Option<&Value>,
) -> Option<String> {
    let lower = tool_key.to_lowercase();
    if !(lower.contains("shell") || lower.contains("command") || lower.contains("exec")) {
        return None;
    }
    let success = raw_output
        .and_then(|value| value.get("success"))
        .and_then(Value::as_object)?;
    let command = raw_input.and_then(|args| first_string(args, &["command"]));
    let exit_code = success.get("exitCode").and_then(Value::as_i64);
    let stdout = success
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stderr = success
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut parts = Vec::new();
    if let Some(command) = command
        && !command.trim().is_empty()
    {
        parts.push(format!(
            "Command: `{}`",
            truncate_for_title(command.trim(), 160)
        ));
    }
    if let Some(exit_code) = exit_code {
        parts.push(format!("Exit code: {exit_code}"));
    }
    if !stdout.trim().is_empty() {
        parts.push(format!(
            "stdout:\n```sh\n{}\n```",
            truncate_output(stdout, 3000)
        ));
    }
    if !stderr.trim().is_empty() {
        parts.push(format!(
            "stderr:\n```sh\n{}\n```",
            truncate_output(stderr, 3000)
        ));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

fn truncate_for_title(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

fn truncate_output(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text
        .chars()
        .take(max_chars.saturating_sub(19))
        .collect::<String>();
    out.push_str("\n... [output truncated]");
    out
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(text) = value.get(*key).and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn extract_before_full_file_content(raw_output: Option<&Value>) -> Option<String> {
    raw_output
        .and_then(|value| value.get("success"))
        .and_then(|value| value.get("beforeFullFileContent"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_after_full_file_content(raw_output: Option<&Value>) -> Option<String> {
    raw_output
        .and_then(|value| value.get("success"))
        .and_then(|value| value.get("afterFullFileContent"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_path_from_message(message: &str) -> Option<String> {
    let lower = message.to_lowercase();
    let markers = [
        "wrote contents to ",
        "the file ",
        "created file ",
        "updated file ",
        "deleted file ",
    ];
    for marker in markers {
        if let Some(idx) = lower.find(marker) {
            let start = idx + marker.len();
            let rest = message.get(start..)?.trim();
            if rest.is_empty() {
                continue;
            }
            let line = rest.lines().next().unwrap_or(rest).trim();
            let trimmed = line
                .trim_matches('`')
                .trim_end_matches('.')
                .trim_end_matches(':')
                .trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn extract_usage_update(parsed: &Value) -> Option<CursorUsageUpdate> {
    let usage = parsed.get("usage")?;
    let input = usage
        .get("inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = usage
        .get("cacheReadTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = usage
        .get("cacheWriteTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let used = input
        .saturating_add(output)
        .saturating_add(cache_read)
        .saturating_add(cache_write);
    if used == 0 {
        None
    } else {
        Some(CursorUsageUpdate { used, size: used })
    }
}

fn collect_text_parts(value: &Value) -> String {
    let mut chunks = Vec::new();
    collect_text_chunks(value, &mut chunks);
    chunks.join("")
}

fn collect_text_chunks(value: &Value, chunks: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if !text.is_empty() {
                chunks.push(text.clone());
            }
        }
        Value::Array(values) => {
            for item in values {
                collect_text_chunks(item, chunks);
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                chunks.push(text.to_string());
            }
            for key in ["content", "delta", "message", "result", "error", "success"] {
                if let Some(next) = map.get(key) {
                    collect_text_chunks(next, chunks);
                }
            }
        }
        _ => {}
    }
}

fn map_spawn_error(error: std::io::Error) -> ErrorKind {
    if error.kind() == std::io::ErrorKind::NotFound {
        return ErrorKind::CliMissing;
    }
    ErrorKind::TransientIo(error.to_string())
}

fn parse_model_unsupported(stderr: &str) -> ErrorKind {
    // Example: "Cannot use this model: opus-4.6-thinking. Available models: auto, gpt-5"
    let mut requested = "unknown".to_string();
    let mut available = Vec::new();

    if let Some(start) = stderr.find("Cannot use this model:") {
        let tail = &stderr[start + "Cannot use this model:".len()..];
        let candidate = if let Some(idx) = tail.find(". Available models:") {
            tail[..idx].trim()
        } else {
            tail.trim()
        };
        requested = candidate.to_string();
    }

    if let Some(start) = stderr.find("Available models:") {
        let tail = &stderr[start + "Available models:".len()..];
        available = tail
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect();
    }

    ErrorKind::ModelUnsupported {
        requested,
        available,
    }
}

fn parse_model_id_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower = trimmed.to_lowercase();
    if lower.contains("available models") {
        return None;
    }

    let mut candidate = trimmed
        .trim_start_matches("- ")
        .trim_start_matches("* ")
        .trim_start_matches("• ");

    if let Some((id, _)) = candidate.split_once(" - ") {
        candidate = id.trim();
    } else if let Some((id, _)) = candidate.split_once(':') {
        candidate = id.trim();
    } else {
        candidate = candidate.split_whitespace().next().unwrap_or_default();
    }

    if candidate.is_empty() {
        return None;
    }
    let looks_like_model = candidate
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.');
    if !looks_like_model {
        return None;
    }
    Some(candidate.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(path: &str) -> String {
        let mut full = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full.push(path);
        std::fs::read_to_string(full).expect("fixture should be readable")
    }

    fn replay_stream_json(input: &str) -> Vec<CursorCliEvent> {
        let mut events = Vec::new();
        let mut seen_tool_calls = HashSet::new();
        let mut last_assistant_chunk: Option<String> = None;
        let mut seen_large_assistant_chunks: HashSet<String> = HashSet::new();
        let mut assistant_streamed_text = String::new();

        for line in input.lines().filter(|line| !line.trim().is_empty()) {
            let parsed: Value = serde_json::from_str(line).expect("fixture line should parse");
            let event_type = parsed
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let normalized_event_type = normalize_event_type(event_type);
            let subtype = parsed
                .get("subtype")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let has_tool_payload = looks_like_tool_event(&parsed);
            match normalized_event_type.as_str() {
                "assistant" => {
                    for text in extract_assistant_deltas(&parsed) {
                        let normalized = normalize_text_for_dedupe(&text);
                        let streamed_normalized =
                            normalize_text_for_dedupe(&assistant_streamed_text);
                        let is_large_duplicate = normalized.chars().count() >= 24
                            && !normalized.is_empty()
                            && !seen_large_assistant_chunks.insert(normalized.clone());
                        let is_already_in_stream = !normalized.is_empty()
                            && normalized.chars().count() >= 24
                            && streamed_normalized.contains(&normalized);
                        if !text.trim().is_empty()
                            && last_assistant_chunk.as_deref() != Some(text.as_str())
                            && !is_large_duplicate
                            && !is_already_in_stream
                        {
                            assistant_streamed_text.push_str(&text);
                            last_assistant_chunk = Some(text.clone());
                            events.push(CursorCliEvent::AssistantDelta(text));
                        }
                    }
                    if has_tool_payload {
                        events.extend(parse_tool_call_events(
                            &parsed,
                            "started",
                            &mut seen_tool_calls,
                        ));
                    }
                }
                "tool_call" => {
                    events.extend(parse_tool_call_events(
                        &parsed,
                        subtype,
                        &mut seen_tool_calls,
                    ));
                }
                "thinking" => {
                    let text = parsed
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .unwrap_or_default()
                        .to_string();
                    if !text.is_empty() {
                        events.push(CursorCliEvent::ThinkingDelta(text));
                    }
                }
                "system" => {
                    let message = extract_system_message(&parsed, subtype);
                    if !message.trim().is_empty() && message != "System: init" && message != "init"
                    {
                        events.push(CursorCliEvent::System {
                            message,
                            raw: parsed,
                        });
                    }
                }
                "user" => {}
                "result" => {
                    let text = dedupe_result_text_for_stream(
                        &assistant_streamed_text,
                        extract_result_text(&parsed),
                    );
                    let usage = extract_usage_update(&parsed);
                    events.push(CursorCliEvent::Result(CursorPromptResult {
                        text,
                        meta: parsed,
                        usage,
                    }));
                }
                _ => {
                    if has_tool_payload {
                        events.extend(parse_tool_call_events(
                            &parsed,
                            if subtype.is_empty() {
                                "started"
                            } else {
                                subtype
                            },
                            &mut seen_tool_calls,
                        ));
                    }
                }
            }
        }

        events
    }

    #[test]
    fn classify_model_unsupported_from_stderr() {
        let stderr = fixture("tests/fixtures/stderr/unsupported_model.stderr");
        let classified = CursorCliRuntime::classify_cli_failure(Some(1), &stderr)
            .expect("classification should exist");
        match classified {
            ErrorKind::ModelUnsupported {
                requested,
                available,
            } => {
                assert_eq!(requested, "opus-4.6-thinking");
                assert_eq!(
                    available,
                    vec![
                        "auto".to_string(),
                        "gpt-5".to_string(),
                        "claude-3.7-sonnet".to_string(),
                    ]
                );
            }
            other => panic!("unexpected classification: {other:?}"),
        }
    }

    #[test]
    fn classify_auth_required_from_stderr() {
        let stderr = "User not authenticated. Please run: cursor-agent login";
        let classified = CursorCliRuntime::classify_cli_failure(Some(1), stderr)
            .expect("classification should exist");
        assert!(matches!(classified, ErrorKind::AuthRequired));
    }

    #[test]
    fn classify_workspace_trust_required_from_stderr() {
        let stderr = fixture("tests/fixtures/stderr/workspace_trust_required.stderr");
        let classified = CursorCliRuntime::classify_cli_failure(Some(1), &stderr)
            .expect("classification should exist");
        assert!(matches!(classified, ErrorKind::WorkspaceTrustRequired));
    }

    #[test]
    fn parse_model_id_from_varied_lines() {
        assert_eq!(
            parse_model_id_from_line("gpt-5 - GPT-5"),
            Some("gpt-5".to_string())
        );
        assert_eq!(
            parse_model_id_from_line("- claude-3.7-sonnet: Claude 3.7 Sonnet"),
            Some("claude-3.7-sonnet".to_string())
        );
        assert_eq!(
            parse_model_id_from_line("o3-mini"),
            Some("o3-mini".to_string())
        );
        assert_eq!(parse_model_id_from_line("Available models:"), None);
    }

    #[test]
    fn build_diff_from_edit_payload() {
        let args = serde_json::json!({
            "path": "src/main.rs",
            "old_string": "fn a() {}",
            "new_string": "fn a() { println!(\"x\"); }"
        });
        let diff = build_diff_from_tool_payload("editToolCall", Some(&args), None).expect("diff");
        assert_eq!(diff.path, std::path::PathBuf::from("src/main.rs"));
        assert_eq!(diff.old_text.as_deref(), Some("fn a() {}"));
        assert_eq!(diff.new_text, "fn a() { println!(\"x\"); }");
    }

    #[test]
    fn parse_flat_tool_call_shape_extracts_path_and_kind() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "call_1",
            "tool_call": {
                "name": "edit_file",
                "args": {
                    "path": "/tmp/example.txt",
                    "old_string": "a",
                    "new_string": "b"
                },
                "result": {
                    "success": "The file /tmp/example.txt has been updated."
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "completed", &mut seen);
        assert!(!events.is_empty());
        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(*update),
            _ => None,
        });
        let update = update.expect("expected ToolCallUpdate");
        assert_eq!(update.kind, ToolKind::Edit);
        assert!(update.locations.contains(&"/tmp/example.txt".to_string()));
        assert!(update.diff.is_some(), "expected diff for edit_file");
    }

    #[test]
    fn build_diff_from_cursor_stream_content_shape() {
        let args = serde_json::json!({
            "path": "/tmp/t.txt",
            "streamContent": "hello"
        });
        let result = serde_json::json!({
            "success": {
                "path": "/tmp/t.txt",
                "beforeFullFileContent": "",
                "afterFullFileContent": "hello"
            }
        });
        let diff = build_diff_from_tool_payload("editToolCall", Some(&args), Some(&result))
            .expect("expected diff");
        assert_eq!(diff.path, std::path::PathBuf::from("/tmp/t.txt"));
        assert_eq!(diff.old_text.as_deref(), Some(""));
        assert_eq!(diff.new_text, "hello");
    }

    #[test]
    fn parse_nested_tool_call_with_result_full_file_content_builds_diff() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "tool_1",
            "tool_call": {
                "editToolCall": {
                    "args": {
                        "path": "/tmp/demo.txt",
                        "streamContent": "after"
                    },
                    "result": {
                        "success": {
                            "beforeFullFileContent": "before",
                            "afterFullFileContent": "after"
                        }
                    }
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "completed", &mut seen);
        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(*update),
            _ => None,
        });
        let update = update.expect("expected ToolCallUpdate");
        let diff = update.diff.expect("expected diff");
        assert_eq!(diff.path, std::path::PathBuf::from("/tmp/demo.txt"));
        assert_eq!(diff.old_text.as_deref(), Some("before"));
        assert_eq!(diff.new_text, "after");
    }

    #[test]
    fn parse_top_level_tool_call_variant_extracts_diff() {
        let parsed = serde_json::json!({
            "type": "assistant",
            "subtype": "started",
            "writeToolCall": {
                "args": {
                    "path": "/tmp/new.txt",
                    "content": "new file"
                },
                "result": {
                    "success": {
                        "afterFullFileContent": "new file"
                    }
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "started", &mut seen);
        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(*update),
            _ => None,
        });
        let update = update.expect("expected ToolCallUpdate");
        assert_eq!(update.kind, ToolKind::Edit);
        assert!(update.locations.contains(&"/tmp/new.txt".to_string()));
        assert!(
            update.diff.is_some(),
            "expected diff for top-level writeToolCall"
        );
    }

    #[test]
    fn edit_tool_title_includes_path() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "started",
            "call_id": "call_2",
            "tool_call": {
                "editToolCall": {
                    "args": {
                        "path": "/tmp/path.txt",
                        "new_string": "hello"
                    }
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "started", &mut seen);
        let start = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallStart(call) => Some(call),
            _ => None,
        });
        let start = start.expect("expected ToolCallStart");
        assert_eq!(start.title, "Edit /tmp/path.txt");
    }

    #[test]
    fn tool_call_without_explicit_id_has_stable_id_and_no_duplicate_start() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "started",
            "tool_call": {
                "editToolCall": {
                    "args": {
                        "path": "/tmp/file.txt",
                        "old_string": "a",
                        "new_string": "b"
                    }
                }
            }
        });
        let mut seen = HashSet::new();
        let first_events = parse_tool_call_events(&parsed, "started", &mut seen);
        let second_events = parse_tool_call_events(&parsed, "completed", &mut seen);

        let first_start_count = first_events
            .iter()
            .filter(|event| matches!(event, CursorCliEvent::ToolCallStart(_)))
            .count();
        let second_start_count = second_events
            .iter()
            .filter(|event| matches!(event, CursorCliEvent::ToolCallStart(_)))
            .count();
        assert_eq!(first_start_count, 1, "first parse should include start");
        assert_eq!(
            second_start_count, 0,
            "duplicate parse should not include start"
        );

        let first_id = first_events.iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(update.tool_call_id.clone()),
            _ => None,
        });
        let second_id = second_events.iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(update.tool_call_id.clone()),
            _ => None,
        });
        assert_eq!(first_id, second_id, "fallback id should be stable");
    }

    #[test]
    fn build_diff_for_delete_payload() {
        let args = serde_json::json!({
            "path": "/tmp/deleted.txt"
        });
        let diff = build_diff_from_tool_payload("deleteToolCall", Some(&args), None)
            .expect("expected diff for delete");
        assert_eq!(diff.path, std::path::PathBuf::from("/tmp/deleted.txt"));
        assert_eq!(diff.old_text.as_deref(), Some("[file deleted]"));
        assert!(diff.new_text.is_empty());
    }

    #[test]
    fn replay_fixture_usage_long_extracts_usage_and_dedupes_result_text() {
        let events = replay_stream_json(&fixture("tests/fixtures/stream-json/usage_long.jsonl"));
        let result = events.into_iter().find_map(|event| match event {
            CursorCliEvent::Result(result) => Some(result),
            _ => None,
        });
        let result = result.expect("expected result event");
        assert!(
            result.text.is_empty(),
            "result text should be deduped against deltas"
        );
        assert_eq!(result.usage.map(|usage| usage.used), Some(1500));
    }

    #[test]
    fn replay_fixture_shell_tool_maps_execute_title_and_rich_output() {
        let events = replay_stream_json(&fixture("tests/fixtures/stream-json/terminal_only.jsonl"));
        let start = events.iter().find_map(|event| match event {
            CursorCliEvent::ToolCallStart(call) => Some(call),
            _ => None,
        });
        let start = start.expect("expected shell tool start");
        assert_eq!(start.kind, ToolKind::Execute);
        assert_eq!(start.title, "Run pwd && ls -1");

        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update)
                if update.status == ToolCallStatus::Completed =>
            {
                Some(*update)
            }
            _ => None,
        });
        let update = update.expect("expected completed tool update");
        let message = update.message.expect("expected formatted shell message");
        assert!(message.contains("stdout:"));
        assert!(message.contains("Exit code: 0"));
    }

    #[test]
    fn replay_fixture_multi_file_edit_covers_nested_shapes() {
        let events =
            replay_stream_json(&fixture("tests/fixtures/stream-json/multi_file_edit.jsonl"));
        let completed_updates: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                CursorCliEvent::ToolCallUpdate(update)
                    if update.status == ToolCallStatus::Completed =>
                {
                    Some(update.as_ref())
                }
                _ => None,
            })
            .collect();
        assert_eq!(completed_updates.len(), 2);
        assert!(completed_updates.iter().all(|update| update.diff.is_some()));
    }

    #[test]
    fn replay_fixture_plan_like_emits_thinking_and_suppresses_init_system() {
        let events = replay_stream_json(&fixture("tests/fixtures/stream-json/plan_like.jsonl"));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CursorCliEvent::ThinkingDelta(_)))
        );
        assert!(!events.iter().any(|event| {
            matches!(
                event,
                CursorCliEvent::System { message, .. } if message == "System: init" || message == "init"
            )
        }));
    }

    #[test]
    fn parse_path_from_wrote_contents_message_when_args_have_no_path() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {
                "name": "edit_file",
                "args": {},
                "result": {
                    "success": "Wrote contents to /Users/cameron/Code/chia-wallet-sdk/jokes.txt"
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "completed", &mut seen);
        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(*update),
            _ => None,
        });
        let update = update.expect("expected tool update");
        assert_eq!(
            update.locations.first().map(|value| value.as_str()),
            Some("/Users/cameron/Code/chia-wallet-sdk/jokes.txt")
        );
        assert_eq!(
            update.title.as_deref(),
            Some("Edit /Users/cameron/Code/chia-wallet-sdk/jokes.txt")
        );
    }

    #[test]
    fn parse_update_todos_tool_payload_extracts_items() {
        let parsed = serde_json::json!({
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {
                "updateTodosToolCall": {
                    "args": {
                        "todos": [
                            {"content": "First", "status": "TODO_STATUS_PENDING"},
                            {"content": "Second", "status": "TODO_STATUS_IN_PROGRESS"}
                        ]
                    },
                    "result": {
                        "success": {
                            "todos": [
                                {"content": "First", "status": "TODO_STATUS_COMPLETED"},
                                {"content": "Second", "status": "TODO_STATUS_IN_PROGRESS"}
                            ]
                        }
                    }
                }
            }
        });
        let mut seen = HashSet::new();
        let events = parse_tool_call_events(&parsed, "completed", &mut seen);
        let update = events.into_iter().find_map(|event| match event {
            CursorCliEvent::ToolCallUpdate(update) => Some(*update),
            _ => None,
        });
        let update = update.expect("expected tool update");
        let todos = update.todo_items.expect("expected todo items");
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].content, "First");
        assert_eq!(todos[0].status, "TODO_STATUS_COMPLETED");
    }

    #[test]
    fn dedupe_result_text_blanks_duplicate_final_result() {
        let deduped = dedupe_result_text_for_stream(
            "Created `jokes2.txt` with a short science joke in it.",
            "Created `jokes2.txt` with a short science joke in it.".to_string(),
        );
        assert!(deduped.is_empty());
    }

    #[test]
    fn replay_dedupes_non_consecutive_large_assistant_chunks() {
        let duplicate = "Added **\"Add a joke to the joke file\"** to the to-do list.".to_string();
        let lines = vec![
            serde_json::json!({
                "type":"assistant",
                "content":[{"type":"text","text":duplicate}]
            }),
            serde_json::json!({
                "type":"assistant",
                "content":[{"type":"text","text":"\n\n"}]
            }),
            serde_json::json!({
                "type":"assistant",
                "content":[{"type":"text","text":"Added **\"Add a joke to the joke file\"** to the to-do list."}]
            }),
            serde_json::json!({
                "type":"result",
                "result":"Added **\"Add a joke to the joke file\"** to the to-do list."
            }),
        ];
        let input = lines
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let events = replay_stream_json(&input);
        let assistant_chunks = events
            .iter()
            .filter(|event| matches!(event, CursorCliEvent::AssistantDelta(_)))
            .count();
        assert_eq!(
            assistant_chunks, 1,
            "duplicate large chunk should be suppressed"
        );
    }

    #[test]
    fn replay_suppresses_full_assistant_and_result_when_already_streamed() {
        let partial_1 = "Added **\"Add a joke to the joke file\"**";
        let partial_2 = " to the to-do list. Your list now has:\n1. Make a joke file\n2. Add a joke to the joke file";
        let full = format!("{partial_1}{partial_2}");
        let lines = vec![
            serde_json::json!({"type":"assistant","content":[{"type":"text","text":partial_1}]}),
            serde_json::json!({"type":"assistant","content":[{"type":"text","text":partial_2}]}),
            serde_json::json!({"type":"assistant","content":[{"type":"text","text":full.clone()}]}),
            serde_json::json!({"type":"result","result":full}),
        ];
        let input = lines
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let events = replay_stream_json(&input);
        let assistant_chunks = events
            .iter()
            .filter_map(|event| match event {
                CursorCliEvent::AssistantDelta(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            assistant_chunks.len(),
            2,
            "only partial chunks should remain"
        );
        let result_text = events.into_iter().find_map(|event| match event {
            CursorCliEvent::Result(result) => Some(result.text),
            _ => None,
        });
        assert_eq!(
            result_text.as_deref(),
            Some(""),
            "duplicate final result should be suppressed"
        );
    }
}
