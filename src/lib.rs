#![deny(clippy::print_stdout, clippy::print_stderr)]

use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc};
use tracing_subscriber::EnvFilter;

mod proxy;
mod sessions;

pub async fn run_main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let binary = resolve_agent_binary();
    tracing::info!(binary = %binary, "starting cursor-acp proxy");

    let models = fetch_models(&binary).await;
    tracing::info!(count = models.len(), "loaded models");

    let state = Arc::new(Mutex::new(proxy::ProxyState::new()));
    state.lock().await.models = models;

    let store = Arc::new(sessions::SessionStore::new().await);

    let (to_zed_tx, to_zed_rx) = mpsc::unbounded_channel::<String>();

    // Spawn initial child.
    let child_handle = Arc::new(Mutex::new(
        spawn_child(&binary, None).context("failed to spawn initial agent acp")?,
    ));

    let (to_child_tx, to_child_rx) = mpsc::unbounded_channel::<String>();
    let to_child_for_intercept = to_child_tx.clone();

    // Task: read Zed's stdin, process, forward to child.
    let stdin_state = Arc::clone(&state);
    let stdin_to_zed = to_zed_tx.clone();
    let stdin_child_handle = Arc::clone(&child_handle);
    let stdin_store = Arc::clone(&store);
    let binary_for_restart = binary.clone();
    let stdin_task = tokio::spawn(async move {
        let reader = BufReader::new(tokio::io::stdin());
        let mut lines = reader.lines();
        // When model changes, we need a new channel for the new child writer.
        let mut current_to_child = to_child_tx;

        while let Ok(Some(line)) = lines.next_line().await {
            let msg: Option<Value> = serde_json::from_str(&line).ok();

            // Handle session/list directly (not forwarded to child).
            if let Some(ref msg) = msg
                && msg.get("method").and_then(Value::as_str) == Some("session/list")
            {
                let cwd = msg.pointer("/params/cwd").and_then(Value::as_str);
                let request_id = msg.get("id").cloned().unwrap_or(Value::Null);
                let entries = stdin_store.list_sessions(cwd).await;
                tracing::info!(count = entries.len(), ?cwd, "listed sessions");
                let response = sessions::build_list_response(&request_id, &entries);
                drop(stdin_to_zed.send(response));
                continue;
            }

            // Handle session/load directly (replay stored history).
            if let Some(ref msg) = msg
                && msg.get("method").and_then(Value::as_str) == Some("session/load")
            {
                let request_id = msg.get("id").cloned().unwrap_or(Value::Null);
                let session_id = msg
                    .pointer("/params/sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("");

                let (modes, models) = {
                    let st = stdin_state.lock().await;
                    let modes = st.last_session_modes.clone().unwrap_or_else(|| {
                        serde_json::json!({
                            "currentModeId": "agent",
                            "availableModes": [
                                { "id": "agent", "name": "Agent" },
                                { "id": "ask", "name": "Ask" },
                                { "id": "plan", "name": "Plan" },
                            ]
                        })
                    });
                    let models = st.last_session_models.clone().unwrap_or_else(|| {
                        let current = st.selected_model.as_deref().unwrap_or("auto");
                        let available: Vec<serde_json::Value> = st
                            .models
                            .iter()
                            .map(|m| {
                                serde_json::json!({
                                    "modelId": m.id,
                                    "name": m.name,
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "currentModelId": current,
                            "availableModels": available,
                        })
                    });
                    (modes, models)
                };
                let response =
                    sessions::build_load_response(&request_id, Some(&modes), Some(&models));
                drop(stdin_to_zed.send(response));

                // Replay stored history as session/update notifications.
                let history = stdin_store.load_history(session_id).await;
                tracing::info!(
                    count = history.len(),
                    session_id,
                    "replaying session history"
                );
                for update in &history {
                    let notification = sessions::build_history_notification(session_id, update);
                    drop(stdin_to_zed.send(notification));
                }

                // Set the Zed session ID and create a fresh session in the child
                // so subsequent prompts have a valid context.
                let cwd = msg
                    .pointer("/params/cwd")
                    .and_then(Value::as_str)
                    .map(String::from);
                let stored_cwd = if cwd.is_none() {
                    stdin_store
                        .list_sessions(None)
                        .await
                        .iter()
                        .find(|s| s.id == session_id)
                        .map(|s| s.cwd.clone())
                } else {
                    None
                };
                let cwd = cwd.or(stored_cwd).unwrap_or_else(|| ".".to_string());

                let load_req_id = {
                    let mut st = stdin_state.lock().await;
                    st.zed_session_id = Some(session_id.to_string());
                    let req_id = serde_json::json!(-9999);
                    st.suppress_response(req_id.clone());
                    req_id
                };
                let child_new_request = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": load_req_id,
                    "method": "session/new",
                    "params": {
                        "cwd": cwd,
                        "mcpServers": []
                    }
                });
                tracing::debug!(session_id, %cwd, "sending session/new to child for loaded session");
                drop(current_to_child.send(child_new_request.to_string()));
                continue;
            }

            let action = if let Some(ref msg) = msg {
                let mut st = stdin_state.lock().await;
                proxy::process_client_message(msg, &mut st)
            } else {
                proxy::ClientAction::Forward
            };

            match action {
                proxy::ClientAction::Forward => {
                    drop(current_to_child.send(line));
                }
                proxy::ClientAction::ForwardPatched(patched) => {
                    drop(current_to_child.send(patched));
                }
                proxy::ClientAction::ForwardWithPrompt {
                    line: forward_line,
                    prompt_text,
                } => {
                    drop(current_to_child.send(forward_line));
                    // Materialize the session on first prompt (deferred creation).
                    let (pending, zed_sid) = {
                        let mut st = stdin_state.lock().await;
                        let pending = proxy::take_pending_session(&mut st);
                        let sid = st.zed_session_id.clone();
                        (pending, sid)
                    };
                    if let Some(sid) = zed_sid {
                        let s = Arc::clone(&stdin_store);
                        tokio::spawn(async move {
                            if let Some((id, cwd)) = pending {
                                tracing::debug!(session_id = %id, cwd = %cwd, "creating session on first prompt");
                                s.create_session(&id, &cwd).await;
                            }
                            let user_update = serde_json::json!({
                                "sessionUpdate": "user_message_chunk",
                                "content": { "type": "text", "text": prompt_text }
                            });
                            s.append_history(&sid, &user_update).await;
                            s.set_title_if_empty(&sid, &prompt_text).await;
                        });
                    }
                }
                proxy::ClientAction::Drop => {
                    // Silently discard (response to our synthesized fs request).
                }
                proxy::ClientAction::Respond {
                    response_to_zed,
                    restart_with_model,
                } => {
                    drop(stdin_to_zed.send(response_to_zed));

                    if let Some(model) = restart_with_model {
                        tracing::info!(model = %model, "restarting child with new model");

                        // Kill old child.
                        {
                            let mut handle = stdin_child_handle.lock().await;
                            drop(handle.stdin.take());
                            handle.kill().await.ok();
                        }

                        // Spawn new child.
                        match spawn_child(&binary_for_restart, Some(&model)) {
                            Ok(new_child) => {
                                *stdin_child_handle.lock().await = new_child;

                                // New channels for the new child's stdin.
                                let (new_tx, new_rx) = mpsc::unbounded_channel::<String>();
                                current_to_child = new_tx.clone();

                                // Spawn new child writer.
                                let new_stdin =
                                    stdin_child_handle.lock().await.stdin.take().unwrap();
                                tokio::spawn(write_to_sink(new_stdin, new_rx));

                                // Spawn new child reader.
                                let new_stdout =
                                    stdin_child_handle.lock().await.stdout.take().unwrap();
                                let reader_to_zed = stdin_to_zed.clone();
                                let reader_to_child = new_tx;
                                let reader_state = Arc::clone(&stdin_state);
                                let reader_store = Arc::clone(&stdin_store);
                                tokio::spawn(read_child_stdout(
                                    new_stdout,
                                    reader_to_zed,
                                    reader_to_child,
                                    reader_state,
                                    reader_store,
                                ));

                                // Replay init + auth with unique IDs (responses suppressed).
                                let replay_msgs = {
                                    let mut st = stdin_state.lock().await;
                                    st.prepare_replay_messages()
                                };
                                for msg in replay_msgs {
                                    drop(current_to_child.send(msg));
                                }
                            }
                            Err(e) => {
                                tracing::error!(err = %e, "failed to restart child");
                            }
                        }
                    }
                }
            }
        }
        tracing::debug!("stdin EOF");
    });

    // Take child's stdin/stdout for the initial child.
    {
        let mut handle = child_handle.lock().await;
        let child_stdin = handle.stdin.take().context("child stdin")?;
        let child_stdout = handle.stdout.take().context("child stdout")?;

        // Task: write to initial child's stdin.
        tokio::spawn(write_to_sink(child_stdin, to_child_rx));

        // Task: read from initial child's stdout.
        tokio::spawn(read_child_stdout(
            child_stdout,
            to_zed_tx,
            to_child_for_intercept,
            Arc::clone(&state),
            Arc::clone(&store),
        ));
    }

    // Task: write to Zed's stdout. Finishes when all to_zed_tx senders are dropped.
    let stdout_task = tokio::spawn(write_stdout(to_zed_rx));

    // Wait for stdin to close (Zed disconnected).
    drop(stdin_task.await);
    tracing::debug!("stdin task finished, draining child output");

    // Give the child time to finish processing and produce output,
    // then wait for stdout to drain. If the child exits on its own
    // (closing its stdout), the drain completes immediately.
    let drain = tokio::time::timeout(std::time::Duration::from_secs(5), stdout_task);
    if drain.await.is_err() {
        tracing::debug!("drain timeout, killing child");
    }

    let mut handle = child_handle.lock().await;
    handle.kill().await.ok();

    Ok(())
}

fn resolve_agent_binary() -> String {
    if let Ok(bin) = std::env::var("CURSOR_AGENT_BIN")
        .or_else(|_| std::env::var("CURSOR_AGENT_PATH"))
    {
        return bin;
    }

    if which_exists("cursor-agent") {
        return "cursor-agent".to_string();
    }

    if let Some(found) = probe_cursor_agent_paths() {
        tracing::info!(path = %found, "found cursor-agent via known install path");
        return found;
    }

    tracing::warn!(
        "cursor-agent not found on PATH; set CURSOR_AGENT_BIN to the full path of cursor-agent"
    );
    "cursor-agent".to_string()
}

fn which_exists(name: &str) -> bool {
    #[cfg(windows)]
    {
        std::process::Command::new("where")
            .arg(name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(not(windows))]
    {
        std::process::Command::new("which")
            .arg(name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
}

fn probe_cursor_agent_paths() -> Option<String> {
    let mut candidates = Vec::new();

    #[cfg(windows)]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            // Shim installed by `cursor --install-agent-cli`
            candidates.push(
                std::path::PathBuf::from(&local_app_data)
                    .join("cursor-agent")
                    .join("cursor-agent.ps1"),
            );
            for dir_name in ["cursor", "Cursor"] {
                let base = std::path::PathBuf::from(&local_app_data)
                    .join("Programs")
                    .join(dir_name);
                candidates.push(
                    base.join("resources")
                        .join("app")
                        .join("bin")
                        .join("cursor-agent.exe"),
                );
                candidates.push(base.join("cursor-agent.exe"));
            }
        }
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            let base = std::path::PathBuf::from(&program_files).join("Cursor");
            candidates.push(
                base.join("resources")
                    .join("app")
                    .join("bin")
                    .join("cursor-agent.exe"),
            );
        }
    }

    #[cfg(target_os = "macos")]
    {
        candidates.push(std::path::PathBuf::from(
            "/Applications/Cursor.app/Contents/Resources/app/bin/cursor-agent",
        ));
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(
                std::path::PathBuf::from(&home)
                    .join("Applications/Cursor.app/Contents/Resources/app/bin/cursor-agent"),
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        candidates.push(std::path::PathBuf::from("/usr/share/cursor/resources/app/bin/cursor-agent"));
        candidates.push(std::path::PathBuf::from("/opt/cursor/resources/app/bin/cursor-agent"));
        candidates.push(std::path::PathBuf::from("/opt/Cursor/resources/app/bin/cursor-agent"));
        if let Ok(home) = std::env::var("HOME") {
            let base = std::path::PathBuf::from(&home);
            candidates.push(base.join(".local/share/cursor/resources/app/bin/cursor-agent"));
            candidates.push(base.join(".local/bin/cursor-agent"));
        }
    }

    for path in &candidates {
        tracing::debug!(path = %path.display(), "probing for cursor-agent");
        if path.is_file() {
            return Some(path.to_string_lossy().into_owned());
        }
    }
    None
}

fn spawn_child(binary: &str, model: Option<&str>) -> Result<Child> {
    let mut cmd = build_command(binary);
    cmd.arg("acp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn `{binary} acp`"))?;
    Ok(child)
}

/// Build a `Command` that handles `.ps1` scripts on Windows by invoking
/// them through `powershell.exe`, since `.ps1` files are not directly
/// executable.
fn build_command(binary: &str) -> Command {
    #[cfg(windows)]
    if binary.ends_with(".ps1") {
        let mut cmd = Command::new("powershell.exe");
        cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", binary]);
        return cmd;
    }
    Command::new(binary)
}

async fn fetch_models(binary: &str) -> Vec<proxy::ModelInfo> {
    let output = build_command(binary)
        .arg("--list-models")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            proxy::parse_model_list(&text)
        }
        Ok(out) => {
            tracing::warn!(status = ?out.status, "failed to list models");
            Vec::new()
        }
        Err(e) => {
            tracing::warn!(err = %e, "failed to run --list-models");
            Vec::new()
        }
    }
}

async fn read_child_stdout(
    stdout: tokio::process::ChildStdout,
    to_zed: mpsc::UnboundedSender<String>,
    to_child: mpsc::UnboundedSender<String>,
    state: Arc<Mutex<proxy::ProxyState>>,
    store: Arc<sessions::SessionStore>,
) {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                drop(to_zed.send(line));
                continue;
            }
        };

        // Track new sessions (deferred — not persisted until first prompt).
        {
            let mut st = state.lock().await;
            proxy::track_new_session(&msg, &mut st);
        }

        if let Some(mut info) = proxy::extract_session_update(&msg) {
            // Use the Zed session ID for history storage so events are grouped
            // correctly even after child restarts.
            {
                let st = state.lock().await;
                if let Some(zed_sid) = &st.zed_session_id {
                    info.session_id = zed_sid.clone();
                }
            }
            let s = Arc::clone(&store);
            tokio::spawn(async move {
                s.append_history(&info.session_id, &info.update).await;
                if let Some(t) = info.title {
                    s.update_title(&info.session_id, &t).await;
                } else if let Some(text) = info.user_message {
                    s.set_title_if_empty(&info.session_id, &text).await;
                }
            });
        }

        let action = {
            let mut st = state.lock().await;
            proxy::process_agent_message(&msg, &mut st)
        };

        match action {
            proxy::AgentAction::Forward => {
                drop(to_zed.send(line));
            }
            proxy::AgentAction::ForwardPatched(patched) => {
                drop(to_zed.send(patched));
            }
            proxy::AgentAction::ForwardWithExtra {
                line: forwarded,
                extra_notifications,
            } => {
                drop(to_zed.send(forwarded));
                for notification in extra_notifications {
                    drop(to_zed.send(notification));
                }
            }
            proxy::AgentAction::Intercept {
                response_to_child,
                notifications_to_zed,
            } => {
                if let Some(resp) = response_to_child {
                    drop(to_child.send(resp));
                }
                for notification in notifications_to_zed {
                    drop(to_zed.send(notification));
                }
            }
        }
    }
    tracing::debug!("child stdout EOF");
}

async fn write_to_sink(
    mut sink: tokio::process::ChildStdin,
    mut rx: mpsc::UnboundedReceiver<String>,
) {
    while let Some(line) = rx.recv().await {
        if sink.write_all(line.as_bytes()).await.is_err() {
            break;
        }
        if sink.write_all(b"\n").await.is_err() {
            break;
        }
        if sink.flush().await.is_err() {
            break;
        }
    }
    tracing::debug!("child stdin writer done");
}

async fn write_stdout(mut rx: mpsc::UnboundedReceiver<String>) {
    let mut stdout = tokio::io::stdout();
    while let Some(line) = rx.recv().await {
        if stdout.write_all(line.as_bytes()).await.is_err() {
            break;
        }
        if stdout.write_all(b"\n").await.is_err() {
            break;
        }
        if stdout.flush().await.is_err() {
            break;
        }
    }
    tracing::debug!("stdout writer done");
}
