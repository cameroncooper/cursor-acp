use std::io::Write as _;
use std::process::{Command, Stdio};

use serde_json::Value;

/// Builds a fake "agent acp" script that echoes back specific JSON-RPC
/// messages, including a `_cursor/update_todos` request.
fn write_fake_agent_script(dir: &std::path::Path) -> std::path::PathBuf {
    let script_path = dir.join("fake-agent");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let script = r#"#!/usr/bin/env bash
# Fake "agent acp" — reads JSON-RPC from stdin, responds with canned messages.
# Ignores the "acp" argument.

read_line() {
    IFS= read -r line
    echo "$line" >&2
}

send() {
    echo "$1"
}

# Wait for initialize request
read_line
send '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{"loadSession":true},"authMethods":[{"id":"test","name":"Test"}]}}'

# Wait for authenticate
read_line
send '{"jsonrpc":"2.0","id":2,"result":{}}'

# Wait for session/new
read_line
# Send session/new response
send '{"jsonrpc":"2.0","id":3,"result":{"sessionId":"test-sess-1","modes":{"currentModeId":"agent","availableModes":[{"id":"agent","name":"Agent"}]}}}'

# Wait for session/prompt
read_line

# Send a session/update notification (agent thought)
send '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test-sess-1","update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"thinking..."}}}}'

# Send a tool_call notification
send '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test-sess-1","update":{"sessionUpdate":"tool_call","toolCallId":"tool-1","title":"Update TODOs","kind":"other","status":"pending"}}}'

# Send _cursor/update_todos request (server→client)
send '{"jsonrpc":"2.0","id":0,"method":"_cursor/update_todos","params":{"toolCallId":"tool-1","todos":[{"id":"1","content":"Research topic","status":"pending"},{"id":"2","content":"Write summary","status":"in_progress"}],"merge":false}}'

# Wait for the response to _cursor/update_todos
read_line

# Send tool_call_update completed
send '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test-sess-1","update":{"sessionUpdate":"tool_call_update","toolCallId":"tool-1","status":"completed"}}}'

# Send agent message
send '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test-sess-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Here is your todo list."}}}}'

# Send prompt result
send '{"jsonrpc":"2.0","id":4,"result":{"stopReason":"end_turn"}}'
"#;
        std::fs::write(&script_path, script).unwrap();
        std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    script_path
}

/// Builds a fake script that sends a merge update_todos after the first.
fn write_fake_agent_merge_script(dir: &std::path::Path) -> std::path::PathBuf {
    let script_path = dir.join("fake-agent-merge");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let script = r#"#!/usr/bin/env bash
read_line() { IFS= read -r line; }
send() { echo "$1"; }

# initialize
read_line
send '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{},"authMethods":[]}}'
# authenticate
read_line
send '{"jsonrpc":"2.0","id":2,"result":{}}'
# session/new
read_line
send '{"jsonrpc":"2.0","id":3,"result":{"sessionId":"merge-sess","modes":{"currentModeId":"agent","availableModes":[]}}}'
# prompt
read_line
# session notification to set session ID
send '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"merge-sess","update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"ok"}}}}'

# First update_todos (no merge)
send '{"jsonrpc":"2.0","id":0,"method":"_cursor/update_todos","params":{"todos":[{"id":"1","content":"Step one","status":"pending"}],"merge":false}}'
read_line

# Second update_todos (merge=true, adds item 2 and updates item 1)
send '{"jsonrpc":"2.0","id":1,"method":"_cursor/update_todos","params":{"todos":[{"id":"1","content":"Step one","status":"completed"},{"id":"2","content":"Step two","status":"pending"}],"merge":true}}'
read_line

# Done
send '{"jsonrpc":"2.0","id":4,"result":{"stopReason":"end_turn"}}'
"#;
        std::fs::write(&script_path, script).unwrap();
        std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    script_path
}

#[cfg(unix)]
fn run_proxy_with_fake_agent(fake_script: &std::path::Path, input_lines: &[&str]) -> Vec<Value> {
    let binary = env!("CARGO_BIN_EXE_cursor-acp");

    let mut child = Command::new(binary)
        .env("CURSOR_AGENT_BIN", fake_script)
        .env("RUST_LOG", "warn")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start cursor-acp");

    let mut stdin = child.stdin.take().unwrap();
    for line in input_lines {
        writeln!(stdin, "{line}").unwrap();
        // Small delay to let the proxy process each message
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stdin);

    let output = child.wait_with_output().expect("wait failed");
    let stdout_str = String::from_utf8_lossy(&output.stdout);

    stdout_str
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect()
}

#[test]
#[cfg(unix)]
fn proxy_forwards_normal_messages_and_intercepts_todos() {
    let dir = tempfile::tempdir().unwrap();
    let script = write_fake_agent_script(dir.path());

    let input = vec![
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"authenticate","params":{"methodId":"test"}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"session/new","params":{"cwd":"/tmp"}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"session/prompt","params":{"sessionId":"test-sess-1","prompt":[{"type":"text","text":"hello"}]}}"#,
    ];

    let messages = run_proxy_with_fake_agent(&script, &input);

    let has_response: Vec<bool> = messages.iter().map(|m| m.get("result").is_some()).collect();

    // 1. Initialize response forwarded
    assert!(has_response[0], "initialize response");

    // 2. Authenticate response forwarded
    assert!(has_response[1], "authenticate response");

    // 3. session/new response forwarded
    let session_result = &messages[2]["result"];
    assert_eq!(session_result["sessionId"], "test-sess-1");

    // Find the Plan notification (synthesized from _cursor/update_todos)
    let plan_msgs: Vec<&Value> = messages
        .iter()
        .filter(|m| {
            m.get("method").and_then(Value::as_str) == Some("session/update")
                && m.pointer("/params/update/sessionUpdate")
                    .and_then(Value::as_str)
                    == Some("plan")
        })
        .collect();

    assert_eq!(plan_msgs.len(), 1, "exactly one Plan notification");
    let plan = &plan_msgs[0];
    assert_eq!(plan["params"]["sessionId"], "test-sess-1");

    let entries = plan["params"]["update"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["content"], "Research topic");
    assert_eq!(entries[0]["status"], "pending");
    assert_eq!(entries[0]["priority"], "medium");
    assert_eq!(entries[1]["content"], "Write summary");
    assert_eq!(entries[1]["status"], "in_progress");

    // The _cursor/update_todos should NOT appear in the output to Zed
    let cursor_methods: Vec<&Value> = messages
        .iter()
        .filter(|m| {
            m.get("method")
                .and_then(Value::as_str)
                .is_some_and(|s| s.starts_with("_cursor/"))
        })
        .collect();
    assert!(
        cursor_methods.is_empty(),
        "no _cursor/ methods forwarded to Zed"
    );

    // agent_thought_chunk should be forwarded
    let thought_msgs: Vec<&Value> = messages
        .iter()
        .filter(|m| {
            m.pointer("/params/update/sessionUpdate")
                .and_then(Value::as_str)
                == Some("agent_thought_chunk")
        })
        .collect();
    assert!(!thought_msgs.is_empty(), "thought chunks forwarded");

    // prompt result forwarded
    let prompt_result: Vec<&Value> = messages
        .iter()
        .filter(|m| m.get("id").and_then(Value::as_u64) == Some(4) && m.get("result").is_some())
        .collect();
    assert_eq!(prompt_result.len(), 1, "prompt result forwarded");
    assert_eq!(prompt_result[0]["result"]["stopReason"], "end_turn");
}

#[test]
#[cfg(unix)]
fn proxy_handles_todo_merge() {
    let dir = tempfile::tempdir().unwrap();
    let script = write_fake_agent_merge_script(dir.path());

    let input = vec![
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"authenticate","params":{"methodId":"test"}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"session/new","params":{"cwd":"/tmp"}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"session/prompt","params":{"sessionId":"merge-sess","prompt":[{"type":"text","text":"plan"}]}}"#,
    ];

    let messages = run_proxy_with_fake_agent(&script, &input);

    let plan_msgs: Vec<&Value> = messages
        .iter()
        .filter(|m| {
            m.pointer("/params/update/sessionUpdate")
                .and_then(Value::as_str)
                == Some("plan")
        })
        .collect();

    assert_eq!(
        plan_msgs.len(),
        2,
        "two Plan notifications (one per update_todos)"
    );

    // First plan: one entry
    let entries1 = plan_msgs[0]["params"]["update"]["entries"]
        .as_array()
        .unwrap();
    assert_eq!(entries1.len(), 1);
    assert_eq!(entries1[0]["content"], "Step one");
    assert_eq!(entries1[0]["status"], "pending");

    // Second plan: two entries (merged), step one now completed
    let entries2 = plan_msgs[1]["params"]["update"]["entries"]
        .as_array()
        .unwrap();
    assert_eq!(entries2.len(), 2);
    assert_eq!(entries2[0]["content"], "Step one");
    assert_eq!(entries2[0]["status"], "completed");
    assert_eq!(entries2[1]["content"], "Step two");
    assert_eq!(entries2[1]["status"], "pending");
}
