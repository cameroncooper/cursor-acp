use agent_client_protocol::{
    Agent, AgentSideConnection, Client, ClientSideConnection, ContentBlock, ContentChunk,
    CreateTerminalRequest, CreateTerminalResponse, ExtNotification, ExtRequest, ExtResponse,
    Implementation, InitializeRequest, KillTerminalCommandRequest, KillTerminalCommandResponse,
    ListSessionsRequest, LoadSessionRequest, NewSessionRequest, PlanEntryStatus, PromptRequest,
    ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse, ReleaseTerminalRequest,
    ReleaseTerminalResponse, RequestPermissionRequest, RequestPermissionResponse,
    SessionNotification, SessionUpdate, SetSessionModeRequest, TerminalOutputRequest,
    TerminalOutputResponse, WaitForTerminalExitRequest, WaitForTerminalExitResponse,
    WriteTextFileRequest, WriteTextFileResponse,
};
use cursor_acp::acp::CursorAcpAgent;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

fn managed_edits_env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

#[derive(Clone, Default)]
struct TestClient {
    notifications: Arc<Mutex<Vec<SessionNotification>>>,
    write_requests: Arc<Mutex<Vec<WriteTextFileRequest>>>,
}

#[async_trait::async_trait(?Send)]
impl Client for TestClient {
    async fn request_permission(
        &self,
        _arguments: RequestPermissionRequest,
    ) -> Result<RequestPermissionResponse, agent_client_protocol::Error> {
        Ok(RequestPermissionResponse::new(
            agent_client_protocol::RequestPermissionOutcome::Selected(
                agent_client_protocol::SelectedPermissionOutcome::new(
                    agent_client_protocol::PermissionOptionId::new("allow-once"),
                ),
            ),
        ))
    }

    async fn write_text_file(
        &self,
        arguments: WriteTextFileRequest,
    ) -> Result<WriteTextFileResponse, agent_client_protocol::Error> {
        self.write_requests.lock().unwrap().push(arguments);
        Ok(WriteTextFileResponse::default())
    }

    async fn read_text_file(
        &self,
        _arguments: ReadTextFileRequest,
    ) -> Result<ReadTextFileResponse, agent_client_protocol::Error> {
        Ok(ReadTextFileResponse::new(String::new()))
    }

    async fn session_notification(
        &self,
        args: SessionNotification,
    ) -> Result<(), agent_client_protocol::Error> {
        self.notifications.lock().unwrap().push(args);
        Ok(())
    }

    async fn create_terminal(
        &self,
        _args: CreateTerminalRequest,
    ) -> Result<CreateTerminalResponse, agent_client_protocol::Error> {
        unimplemented!()
    }

    async fn terminal_output(
        &self,
        _args: TerminalOutputRequest,
    ) -> Result<TerminalOutputResponse, agent_client_protocol::Error> {
        unimplemented!()
    }

    async fn kill_terminal_command(
        &self,
        _args: KillTerminalCommandRequest,
    ) -> Result<KillTerminalCommandResponse, agent_client_protocol::Error> {
        unimplemented!()
    }

    async fn release_terminal(
        &self,
        _args: ReleaseTerminalRequest,
    ) -> Result<ReleaseTerminalResponse, agent_client_protocol::Error> {
        unimplemented!()
    }

    async fn wait_for_terminal_exit(
        &self,
        _args: WaitForTerminalExitRequest,
    ) -> Result<WaitForTerminalExitResponse, agent_client_protocol::Error> {
        unimplemented!()
    }

    async fn ext_method(
        &self,
        _args: ExtRequest,
    ) -> Result<ExtResponse, agent_client_protocol::Error> {
        Err(agent_client_protocol::Error::method_not_found())
    }

    async fn ext_notification(
        &self,
        _args: ExtNotification,
    ) -> Result<(), agent_client_protocol::Error> {
        Ok(())
    }
}

fn create_connection_pair(
    client: &TestClient,
    agent: &Arc<CursorAcpAgent>,
) -> (ClientSideConnection, Arc<AgentSideConnection>) {
    let (client_to_agent_rx, client_to_agent_tx) = piper::pipe(1024);
    let (agent_to_client_rx, agent_to_client_tx) = piper::pipe(1024);

    let (client_conn, client_io_task) = ClientSideConnection::new(
        client.clone(),
        client_to_agent_tx,
        agent_to_client_rx,
        |fut| {
            tokio::task::spawn_local(fut);
        },
    );

    let (agent_conn, agent_io_task) = AgentSideConnection::new(
        agent.clone(),
        agent_to_client_tx,
        client_to_agent_rx,
        |fut| {
            tokio::task::spawn_local(fut);
        },
    );

    tokio::task::spawn_local(client_io_task);
    tokio::task::spawn_local(agent_io_task);

    let agent_conn = Arc::new(agent_conn);
    agent.attach_connection(agent_conn.clone());

    (client_conn, agent_conn)
}

fn write_probe_cursor_agent_script(
    dir: &Path,
    stem: &str,
    unix_body: &str,
    windows_body: &str,
) -> PathBuf {
    let extension = if cfg!(windows) { "cmd" } else { "sh" };
    let script = dir.join(format!("{stem}.{extension}"));
    let body = if cfg!(windows) {
        windows_body
    } else {
        unix_body
    };

    fs::write(&script, body).expect("failed to write fake cursor script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script, perms).unwrap();
    }
    script
}

fn write_fake_cursor_agent_script(dir: &Path) -> PathBuf {
    let unix_body = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  echo "Logged in"
  exit 0
fi

if [ "$1" = "models" ]; then
  echo "auto - Auto"
  echo "gpt-5 - GPT-5"
  exit 0
fi

fmt=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-format" ]; then
    fmt="$arg"
  fi
  prev="$arg"
done

if [ "$fmt" = "stream-json" ]; then
  echo '{"type":"system","subtype":"init","model":"gpt-5"}'
  echo '{"type":"tool_call","subtype":"started","tool_call":{"readToolCall":{"args":{"path":"src/main.rs"}}}}'
  echo '{"type":"assistant","message":{"content":[{"text":"stream piece"}]}}'
  echo '{"type":"tool_call","subtype":"completed","tool_call":{"readToolCall":{"args":{"path":"src/main.rs"},"result":{"success":{"totalLines":10}}}}}'
  echo '{"type":"system","subtype":"thinking","message":"analyzing"}'
  echo '{"type":"result","result":"final streamed result","session_id":"resume-123","usage":{"inputTokens":100,"outputTokens":30}}'
  exit 0
fi

echo '{"result":"non-stream result","usage":{"inputTokens":12,"outputTokens":7}}'
"#;
    let windows_body = r#"@echo off
if /I "%~1"=="status" (
  echo Logged in
  exit /b 0
)

if /I "%~1"=="models" (
  echo auto - Auto
  echo gpt-5 - GPT-5
  exit /b 0
)

set "fmt="
set "prev="
:parse_args
if "%~1"=="" goto done_args
if /I "%prev%"=="--output-format" set "fmt=%~1"
set "prev=%~1"
shift
goto parse_args
:done_args

if /I "%fmt%"=="stream-json" (
  echo {"type":"system","subtype":"init","model":"gpt-5"}
  echo {"type":"tool_call","subtype":"started","tool_call":{"readToolCall":{"args":{"path":"src/main.rs"}}}}
  echo {"type":"assistant","message":{"content":[{"text":"stream piece"}]}}
  echo {"type":"tool_call","subtype":"completed","tool_call":{"readToolCall":{"args":{"path":"src/main.rs"},"result":{"success":{"totalLines":10}}}}}
  echo {"type":"system","subtype":"thinking","message":"analyzing"}
  echo {"type":"result","result":"final streamed result","session_id":"resume-123","usage":{"inputTokens":100,"outputTokens":30}}
  exit /b 0
)

echo {"result":"non-stream result","usage":{"inputTokens":12,"outputTokens":7}}
"#;
    write_probe_cursor_agent_script(dir, "fake-cursor-agent", unix_body, windows_body)
}

fn write_force_probe_cursor_agent_script(dir: &Path) -> PathBuf {
    let unix_body = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  echo "Logged in"
  exit 0
fi

if [ "$1" = "models" ]; then
  echo "auto - Auto"
  echo "gpt-5 - GPT-5"
  exit 0
fi

forced=0
fmt=""
prev=""
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    forced=1
  fi
  if [ "$prev" = "--output-format" ]; then
    fmt="$arg"
  fi
  prev="$arg"
done

if [ "$fmt" = "stream-json" ]; then
  echo "{\"type\":\"assistant\",\"message\":{\"content\":[{\"text\":\"forced=$forced\"}]}}"
  echo "{\"type\":\"result\",\"result\":\"forced=$forced\"}"
  exit 0
fi

echo "{\"result\":\"forced=$forced\"}"
"#;
    let windows_body = r#"@echo off
if /I "%~1"=="status" (
  echo Logged in
  exit /b 0
)

if /I "%~1"=="models" (
  echo auto - Auto
  echo gpt-5 - GPT-5
  exit /b 0
)

set "forced=0"
set "fmt="
set "prev="
:parse_args
if "%~1"=="" goto done_args
if /I "%~1"=="--force" set "forced=1"
if /I "%prev%"=="--output-format" set "fmt=%~1"
set "prev=%~1"
shift
goto parse_args
:done_args

if /I "%fmt%"=="stream-json" (
  echo {"type":"assistant","message":{"content":[{"text":"forced=%forced%"}]}}
  echo {"type":"result","result":"forced=%forced%"}
  exit /b 0
)

echo {"result":"forced=%forced%"}
"#;
    write_probe_cursor_agent_script(
        dir,
        "fake-cursor-agent-force-probe",
        unix_body,
        windows_body,
    )
}

fn write_managed_edit_probe_cursor_agent_script(dir: &Path) -> PathBuf {
    let unix_body = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  echo "Logged in"
  exit 0
fi

if [ "$1" = "models" ]; then
  echo "auto - Auto"
  echo "gpt-5 - GPT-5"
  exit 0
fi

fmt=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-format" ]; then
    fmt="$arg"
  fi
  prev="$arg"
done

if [ "$fmt" = "stream-json" ]; then
  echo '{"type":"tool_call","subtype":"completed","tool_call":{"editToolCall":{"args":{"path":"/tmp/managed-edit.txt","new_string":"hello"},"result":{"success":{"afterFullFileContent":"hello"}}}}}'
  echo '{"type":"result","result":"done"}'
  exit 0
fi

echo '{"result":"done"}'
"#;
    let windows_body = r#"@echo off
if /I "%~1"=="status" (
  echo Logged in
  exit /b 0
)

if /I "%~1"=="models" (
  echo auto - Auto
  echo gpt-5 - GPT-5
  exit /b 0
)

set "fmt="
set "prev="
:parse_args
if "%~1"=="" goto done_args
if /I "%prev%"=="--output-format" set "fmt=%~1"
set "prev=%~1"
shift
goto parse_args
:done_args

if /I "%fmt%"=="stream-json" (
  echo {"type":"tool_call","subtype":"completed","tool_call":{"editToolCall":{"args":{"path":"C:\\tmp\\managed-edit.txt","new_string":"hello"},"result":{"success":{"afterFullFileContent":"hello"}}}}}
  echo {"type":"result","result":"done"}
  exit /b 0
)

echo {"result":"done"}
"#;
    write_probe_cursor_agent_script(
        dir,
        "fake-cursor-agent-managed-edit-probe",
        unix_body,
        windows_body,
    )
}

fn write_mode_probe_cursor_agent_script(dir: &Path) -> PathBuf {
    let unix_body = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  echo "Logged in"
  exit 0
fi

if [ "$1" = "models" ]; then
  echo "auto - Auto"
  echo "gpt-5 - GPT-5"
  exit 0
fi

fmt=""
mode=""
sandbox="none"
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-format" ]; then
    fmt="$arg"
  fi
  if [ "$prev" = "--mode" ]; then
    mode="$arg"
  fi
  if [ "$prev" = "--sandbox" ]; then
    sandbox="$arg"
  fi
  prev="$arg"
done

if [ "$fmt" = "stream-json" ]; then
  if [ -z "$mode" ]; then
    mode="none"
  fi
  echo "{\"type\":\"assistant\",\"message\":{\"content\":[{\"text\":\"mode=$mode sandbox=$sandbox\"}]}}"
  echo "{\"type\":\"result\",\"result\":\"mode=$mode sandbox=$sandbox\"}"
  exit 0
fi

echo '{"result":"mode=unknown sandbox=none"}'
"#;
    let windows_body = r#"@echo off
if /I "%~1"=="status" (
  echo Logged in
  exit /b 0
)

if /I "%~1"=="models" (
  echo auto - Auto
  echo gpt-5 - GPT-5
  exit /b 0
)

set "fmt="
set "mode="
set "sandbox=none"
set "prev="
:parse_args
if "%~1"=="" goto done_args
if /I "%prev%"=="--output-format" set "fmt=%~1"
if /I "%prev%"=="--mode" set "mode=%~1"
if /I "%prev%"=="--sandbox" set "sandbox=%~1"
set "prev=%~1"
shift
goto parse_args
:done_args

if "%mode%"=="" set "mode=none"

if /I "%fmt%"=="stream-json" (
  echo {"type":"assistant","message":{"content":[{"text":"mode=%mode% sandbox=%sandbox%"}]}}
  echo {"type":"result","result":"mode=%mode% sandbox=%sandbox%"}
  exit /b 0
)

echo {"result":"mode=unknown sandbox=none"}
"#;
    write_probe_cursor_agent_script(dir, "fake-cursor-agent-mode-probe", unix_body, windows_body)
}

fn write_todo_plan_probe_cursor_agent_script(dir: &Path) -> PathBuf {
    let unix_body = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  echo "Logged in"
  exit 0
fi

if [ "$1" = "models" ]; then
  echo "auto - Auto"
  echo "gpt-5 - GPT-5"
  exit 0
fi

fmt=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-format" ]; then
    fmt="$arg"
  fi
  prev="$arg"
done

if [ "$fmt" = "stream-json" ]; then
  echo '{"type":"tool_call","subtype":"completed","tool_call":{"updateTodosToolCall":{"args":{"todos":[{"id":"1","content":"Investigate issue","status":"TODO_STATUS_PENDING"},{"id":"2","content":"Implement fix","status":"TODO_STATUS_IN_PROGRESS"}],"merge":false},"result":{"success":{"todos":[{"id":"1","content":"Investigate issue","status":"TODO_STATUS_COMPLETED"},{"id":"2","content":"Implement fix","status":"TODO_STATUS_IN_PROGRESS"}]}}}}}'
  echo '{"type":"result","result":"todo updated"}'
  exit 0
fi

echo '{"result":"todo updated"}'
"#;
    let windows_body = r#"@echo off
if /I "%~1"=="status" (
  echo Logged in
  exit /b 0
)

if /I "%~1"=="models" (
  echo auto - Auto
  echo gpt-5 - GPT-5
  exit /b 0
)

set "fmt="
set "prev="
:parse_args
if "%~1"=="" goto done_args
if /I "%prev%"=="--output-format" set "fmt=%~1"
set "prev=%~1"
shift
goto parse_args
:done_args

if /I "%fmt%"=="stream-json" (
  echo {"type":"tool_call","subtype":"completed","tool_call":{"updateTodosToolCall":{"args":{"todos":[{"id":"1","content":"Investigate issue","status":"TODO_STATUS_PENDING"},{"id":"2","content":"Implement fix","status":"TODO_STATUS_IN_PROGRESS"}],"merge":false},"result":{"success":{"todos":[{"id":"1","content":"Investigate issue","status":"TODO_STATUS_COMPLETED"},{"id":"2","content":"Implement fix","status":"TODO_STATUS_IN_PROGRESS"}]}}}}}
  echo {"type":"result","result":"todo updated"}
  exit /b 0
)

echo {"result":"todo updated"}
"#;
    write_probe_cursor_agent_script(
        dir,
        "fake-cursor-agent-todo-plan-probe",
        unix_body,
        windows_body,
    )
}

#[tokio::test]
async fn initialize_and_session_round_trip() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_fake_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));

            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let init =
                client_conn
                    .initialize(InitializeRequest::new(ProtocolVersion::LATEST).client_info(
                        Implementation::new("test-client", "0.0.0").title("Test Client"),
                    ))
                    .await
                    .expect("initialize should succeed");
            assert_eq!(init.protocol_version, ProtocolVersion::V1);

            let new_session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed");
            assert_eq!(
                new_session
                    .models
                    .as_ref()
                    .expect("models should exist")
                    .available_models
                    .len(),
                2
            );
        })
        .await;
}

#[tokio::test]
async fn prompt_stream_emits_session_updates() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_fake_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));

            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let new_session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed");

            let _response = client_conn
                .prompt(
                    PromptRequest::new(new_session.session_id.clone(), vec!["hello".into()]).meta(
                        serde_json::Map::from_iter([(
                            "stream".to_string(),
                            serde_json::Value::Bool(true),
                        )]),
                    ),
                )
                .await
                .expect("prompt should succeed");

            for _ in 0..8 {
                tokio::task::yield_now().await;
            }

            let updates = client.notifications.lock().unwrap().clone();
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::AgentMessageChunk(_))),
                "expected at least one agent message chunk"
            );
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::AgentThoughtChunk(_))),
                "expected at least one thought chunk from streaming system events"
            );
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::ToolCall(_))),
                "expected a tool call event"
            );
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::ToolCallUpdate(_))),
                "expected a tool call update event"
            );
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::UsageUpdate(_))),
                "expected usage update"
            );
            assert!(
                updates
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::SessionInfoUpdate(_))),
                "expected session info update for title"
            );

            let has_final_text = updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("final streamed result")
                )
            });
            assert!(has_final_text, "expected final streamed result message");

            client.notifications.lock().unwrap().clear();

            client_conn
                .load_session(LoadSessionRequest::new(new_session.session_id, temp.path()))
                .await
                .expect("load_session should replay stored session history");

            for _ in 0..8 {
                tokio::task::yield_now().await;
            }

            let replayed = client.notifications.lock().unwrap();
            assert!(
                replayed
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::ToolCall(_))),
                "expected replayed tool call from stored history"
            );
            assert!(
                replayed
                    .iter()
                    .any(|n| matches!(n.update, SessionUpdate::ToolCallUpdate(_))),
                "expected replayed tool call update from stored history"
            );
        })
        .await;
}

#[tokio::test]
async fn load_unknown_session_id_returns_not_found() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_fake_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let result = client_conn
                .load_session(LoadSessionRequest::new("previous-session-id", temp.path()))
                .await;
            assert!(
                result.is_err(),
                "unknown session IDs should not auto-create"
            );
        })
        .await;
}

#[tokio::test]
async fn list_sessions_filters_by_cwd_and_orders_by_recency() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let cwd_a = temp.path().join("a");
            let cwd_b = temp.path().join("b");
            fs::create_dir_all(&cwd_a).unwrap();
            fs::create_dir_all(&cwd_b).unwrap();

            let script = write_fake_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let session_a = client_conn
                .new_session(NewSessionRequest::new(&cwd_a))
                .await
                .expect("new session A should succeed");
            let session_b = client_conn
                .new_session(NewSessionRequest::new(&cwd_b))
                .await
                .expect("new session B should succeed");

            client_conn
                .prompt(PromptRequest::new(
                    session_a.session_id.clone(),
                    vec!["first".into()],
                ))
                .await
                .expect("prompt A should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    session_b.session_id.clone(),
                    vec!["second".into()],
                ))
                .await
                .expect("prompt B should succeed");

            let all = client_conn
                .list_sessions(ListSessionsRequest::new())
                .await
                .expect("list sessions should succeed");
            assert!(
                all.sessions.len() >= 2,
                "expected at least two sessions in full list"
            );
            assert_eq!(all.sessions[0].session_id, session_b.session_id);

            let filtered = client_conn
                .list_sessions(ListSessionsRequest::new().cwd(cwd_a.clone()))
                .await
                .expect("filtered list sessions should succeed");
            assert_eq!(filtered.sessions.len(), 1);
            assert_eq!(filtered.sessions[0].session_id, session_a.session_id);

            let cwd_a_with_trailing_slash =
                std::path::PathBuf::from(format!("{}/", cwd_a.display()));
            let filtered_slash = client_conn
                .list_sessions(ListSessionsRequest::new().cwd(cwd_a_with_trailing_slash))
                .await
                .expect("trailing slash cwd filter should succeed");
            assert_eq!(filtered_slash.sessions.len(), 1);
            assert_eq!(filtered_slash.sessions[0].session_id, session_a.session_id);

            let paged = client_conn
                .list_sessions(ListSessionsRequest::new().cursor("offset:1"))
                .await
                .expect("cursor list sessions should succeed");
            assert!(
                !paged.sessions.is_empty(),
                "cursor paging should return remaining sessions"
            );
            assert!(
                paged
                    .sessions
                    .iter()
                    .all(|s| s.session_id != all.sessions[0].session_id),
                "offset cursor should skip first most-recent entry"
            );
        })
        .await;
}

#[tokio::test]
async fn ask_mode_disables_force_and_agent_mode_enables_it() {
    let _env_guard = managed_edits_env_lock().lock().await;
    // This test validates the --force behavior when managed edits are disabled.
    unsafe {
        std::env::set_var("CURSOR_ACP_MANAGED_EDITS", "0");
    }
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_force_probe_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed");
            let session_id = session.session_id.clone();

            client_conn
                .prompt(PromptRequest::new(session_id.clone(), vec!["hello".into()]))
                .await
                .expect("ask mode prompt should succeed");

            for _ in 0..6 {
                tokio::task::yield_now().await;
            }

            let ask_updates = client.notifications.lock().unwrap().clone();
            let ask_has_force_zero = ask_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("forced=0")
                )
            });
            assert!(ask_has_force_zero, "ask mode should not pass --force");

            client.notifications.lock().unwrap().clear();

            client_conn
                .set_session_mode(SetSessionModeRequest::new(
                    session.session_id.clone(),
                    "agent",
                ))
                .await
                .expect("set_session_mode should succeed");

            client_conn
                .prompt(PromptRequest::new(
                    session.session_id,
                    vec!["hello again".into()],
                ))
                .await
                .expect("agent mode prompt should succeed");

            for _ in 0..6 {
                tokio::task::yield_now().await;
            }

            let agent_updates = client.notifications.lock().unwrap();
            let agent_has_force_one = agent_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("forced=1")
                )
            });
            assert!(agent_has_force_one, "agent mode should pass --force");
        })
        .await;
    unsafe {
        std::env::remove_var("CURSOR_ACP_MANAGED_EDITS");
    }
}

#[tokio::test]
async fn ask_mode_blocks_managed_edit_writes() {
    let _env_guard = managed_edits_env_lock().lock().await;
    unsafe {
        std::env::set_var("CURSOR_ACP_MANAGED_EDITS", "1");
    }
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_managed_edit_probe_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed");
            let session_id = session.session_id.clone();

            client_conn
                .prompt(PromptRequest::new(session_id.clone(), vec!["hello".into()]))
                .await
                .expect("ask mode prompt should succeed");

            for _ in 0..8 {
                tokio::task::yield_now().await;
            }
            assert!(
                client.write_requests.lock().unwrap().is_empty(),
                "ask mode should not perform managed writes"
            );

            client_conn
                .set_session_mode(SetSessionModeRequest::new(session_id.clone(), "agent"))
                .await
                .expect("set_session_mode should succeed");

            client_conn
                .prompt(PromptRequest::new(session_id, vec!["hello again".into()]))
                .await
                .expect("agent mode prompt should succeed");

            for _ in 0..8 {
                tokio::task::yield_now().await;
            }
            assert!(
                !client.write_requests.lock().unwrap().is_empty(),
                "agent mode should perform managed writes"
            );
        })
        .await;
    unsafe {
        std::env::remove_var("CURSOR_ACP_MANAGED_EDITS");
    }
}

#[tokio::test]
async fn ask_and_plan_send_mode_flag_but_agent_omits_it() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_mode_probe_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed");
            let session_id = session.session_id.clone();

            client_conn
                .prompt(PromptRequest::new(session_id.clone(), vec!["hello".into()]))
                .await
                .expect("ask mode prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            let ask_updates = client.notifications.lock().unwrap().clone();
            let ask_has_mode = ask_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("mode=ask")
                )
            });
            assert!(ask_has_mode, "default ask mode should pass --mode ask");
            let ask_has_sandbox_enabled = ask_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("sandbox=enabled")
                )
            });
            assert!(
                ask_has_sandbox_enabled,
                "ask mode should pass --sandbox enabled"
            );

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(session_id.clone(), "plan"))
                .await
                .expect("set_session_mode plan should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    session_id.clone(),
                    vec!["plan please".into()],
                ))
                .await
                .expect("plan mode prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            let plan_updates = client.notifications.lock().unwrap().clone();
            let plan_has_mode = plan_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("mode=plan")
                )
            });
            assert!(plan_has_mode, "plan mode should pass --mode plan");
            let plan_has_sandbox_enabled = plan_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("sandbox=enabled")
                )
            });
            assert!(
                plan_has_sandbox_enabled,
                "plan mode should pass --sandbox enabled"
            );

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(session_id.clone(), "agent"))
                .await
                .expect("set_session_mode agent should succeed");
            client_conn
                .prompt(PromptRequest::new(session_id, vec!["do it".into()]))
                .await
                .expect("agent mode prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            let agent_updates = client.notifications.lock().unwrap().clone();
            let agent_has_no_mode = agent_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("mode=none")
                )
            });
            assert!(agent_has_no_mode, "agent mode should omit --mode");
            let agent_has_no_sandbox_flag = agent_updates.iter().any(|n| {
                matches!(
                    &n.update,
                    SessionUpdate::AgentMessageChunk(ContentChunk {
                        content: ContentBlock::Text(text),
                        ..
                    }) if text.text.contains("sandbox=none")
                )
            });
            assert!(
                agent_has_no_sandbox_flag,
                "agent mode should omit --sandbox override"
            );
        })
        .await;
}

#[tokio::test]
async fn mode_transition_matrix_behaves_as_expected() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_mode_probe_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let assert_last_mode = |client: &TestClient, expected: &str| {
                let updates = client.notifications.lock().unwrap().clone();
                let found = updates.iter().any(|n| {
                    matches!(
                        &n.update,
                        SessionUpdate::AgentMessageChunk(ContentChunk {
                            content: ContentBlock::Text(text),
                            ..
                        }) if text.text.contains(&format!("mode={expected}"))
                    )
                });
                assert!(
                    found,
                    "expected mode={expected} in agent message updates, got updates: {updates:?}"
                );
            };

            // Start in ask (default), then switch to plan and agent.
            let ask_session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("ask scenario new session should succeed")
                .session_id;
            client_conn
                .prompt(PromptRequest::new(
                    ask_session.clone(),
                    vec!["start ask".into()],
                ))
                .await
                .expect("ask prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "ask");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(ask_session.clone(), "plan"))
                .await
                .expect("switch ask->plan should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    ask_session.clone(),
                    vec!["ask to plan".into()],
                ))
                .await
                .expect("plan prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "plan");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(ask_session.clone(), "agent"))
                .await
                .expect("switch ask->agent should succeed");
            client_conn
                .prompt(PromptRequest::new(ask_session, vec!["ask to agent".into()]))
                .await
                .expect("agent prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "none");

            // Start in plan, then switch to ask and agent.
            client.notifications.lock().unwrap().clear();
            let plan_session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("plan scenario new session should succeed")
                .session_id;
            client_conn
                .set_session_mode(SetSessionModeRequest::new(plan_session.clone(), "plan"))
                .await
                .expect("set start plan should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    plan_session.clone(),
                    vec!["start plan".into()],
                ))
                .await
                .expect("plan start prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "plan");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(plan_session.clone(), "ask"))
                .await
                .expect("switch plan->ask should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    plan_session.clone(),
                    vec!["plan to ask".into()],
                ))
                .await
                .expect("ask prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "ask");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(plan_session.clone(), "agent"))
                .await
                .expect("switch plan->agent should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    plan_session,
                    vec!["plan to agent".into()],
                ))
                .await
                .expect("agent prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "none");

            // Start in agent, then switch to ask and agent.
            client.notifications.lock().unwrap().clear();
            let agent_session = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("agent scenario new session should succeed")
                .session_id;
            client_conn
                .set_session_mode(SetSessionModeRequest::new(agent_session.clone(), "agent"))
                .await
                .expect("set start agent should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    agent_session.clone(),
                    vec!["start agent".into()],
                ))
                .await
                .expect("agent start prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "none");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(agent_session.clone(), "ask"))
                .await
                .expect("switch agent->ask should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    agent_session.clone(),
                    vec!["agent to ask".into()],
                ))
                .await
                .expect("ask prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "ask");

            client.notifications.lock().unwrap().clear();
            client_conn
                .set_session_mode(SetSessionModeRequest::new(agent_session.clone(), "agent"))
                .await
                .expect("switch ask->agent again should succeed");
            client_conn
                .prompt(PromptRequest::new(
                    agent_session,
                    vec!["agent again".into()],
                ))
                .await
                .expect("agent prompt should succeed");
            for _ in 0..6 {
                tokio::task::yield_now().await;
            }
            assert_last_mode(&client, "none");
        })
        .await;
}

#[tokio::test]
async fn todo_tool_call_emits_plan_update() {
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async {
            let temp = tempfile::tempdir().unwrap();
            let script = write_todo_plan_probe_cursor_agent_script(temp.path());
            let client = TestClient::default();
            let agent = Arc::new(CursorAcpAgent::new_with_binary(
                script.to_string_lossy().to_string(),
            ));
            let (client_conn, _agent_conn) = create_connection_pair(&client, &agent);

            let session_id = client_conn
                .new_session(NewSessionRequest::new(temp.path()))
                .await
                .expect("new_session should succeed")
                .session_id;

            client_conn
                .prompt(PromptRequest::new(session_id, vec!["plan please".into()]))
                .await
                .expect("prompt should succeed");

            for _ in 0..8 {
                tokio::task::yield_now().await;
            }

            let updates = client.notifications.lock().unwrap().clone();
            let plan_update = updates.iter().find_map(|n| match &n.update {
                SessionUpdate::Plan(plan) => Some(plan.clone()),
                _ => None,
            });
            let plan = plan_update.expect("expected plan update from todo tool call");
            assert_eq!(plan.entries.len(), 2);
            assert_eq!(plan.entries[0].content, "Investigate issue");
            assert_eq!(plan.entries[0].status, PlanEntryStatus::Completed);
            assert_eq!(plan.entries[1].content, "Implement fix");
            assert_eq!(plan.entries[1].status, PlanEntryStatus::InProgress);
        })
        .await;
}
