use agent_client_protocol::AuthMethod;
use serde_json::Map;

pub const AUTH_METHOD_CURSOR_LOGIN: &str = "cursor-agent-login";

pub fn terminal_auth_method(binary: &str) -> AuthMethod {
    let mut meta = Map::new();
    meta.insert(
        "terminal-auth".to_string(),
        serde_json::json!({
            "label": "Cursor CLI login",
            "command": binary,
            "args": ["login"],
            "env": {}
        }),
    );
    AuthMethod::new(AUTH_METHOD_CURSOR_LOGIN, "Login")
        .description("Login with your Cursor account")
        .meta(meta)
}
