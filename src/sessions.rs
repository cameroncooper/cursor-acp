use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SessionEntry {
    pub id: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct SessionIndex {
    sessions: Vec<SessionEntry>,
}

pub struct SessionStore {
    base_dir: PathBuf,
    index: Mutex<SessionIndex>,
}

impl SessionStore {
    pub async fn new() -> Self {
        let base_dir = session_store_dir();
        let index = read_index_from_disk(&base_dir).await;

        Self {
            base_dir,
            index: Mutex::new(index),
        }
    }

    #[cfg(test)]
    pub async fn with_dir(base_dir: PathBuf) -> Self {
        let index = read_index_from_disk(&base_dir).await;
        Self {
            base_dir,
            index: Mutex::new(index),
        }
    }

    pub async fn create_session(&self, id: &str, cwd: &str) {
        let now = current_time_ms();
        let entry = SessionEntry {
            id: id.to_string(),
            cwd: cwd.to_string(),
            title: None,
            created_at: now,
            updated_at: now,
        };
        {
            let mut index = self.index.lock().await;
            index.sessions.retain(|s| s.id != id);
            index.sessions.insert(0, entry);
        }
        self.flush_index().await;
    }

    pub async fn update_title(&self, session_id: &str, title: &str) {
        {
            let mut index = self.index.lock().await;
            if let Some(entry) = index.sessions.iter_mut().find(|s| s.id == session_id) {
                entry.title = Some(title.to_string());
                entry.updated_at = current_time_ms();
            } else {
                return;
            }
        }
        self.flush_index().await;
    }

    /// Set the title only if none exists yet (auto-title from first user message).
    pub async fn set_title_if_empty(&self, session_id: &str, title: &str) {
        let needs_update = {
            let index = self.index.lock().await;
            index
                .sessions
                .iter()
                .find(|s| s.id == session_id)
                .is_some_and(|s| s.title.is_none())
        };
        if needs_update {
            self.update_title(session_id, &format_session_title(title)).await;
        }
    }

    pub async fn append_history(&self, session_id: &str, update: &Value) {
        // Update the in-memory timestamp (no flush — index is flushed on
        // create/title changes which is sufficient for persistence).
        {
            let mut index = self.index.lock().await;
            if let Some(entry) = index.sessions.iter_mut().find(|s| s.id == session_id) {
                entry.updated_at = current_time_ms();
            }
        }

        let dir = self.base_dir.join("history");
        if let Err(e) = tokio::fs::create_dir_all(&dir).await {
            tracing::warn!(err = %e, "failed to create history directory");
            return;
        }
        let path = dir.join(format!("{session_id}.jsonl"));
        let line = format!("{update}\n");
        match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
        {
            Ok(mut f) => {
                if let Err(e) = f.write_all(line.as_bytes()).await {
                    tracing::warn!(err = %e, "failed to append history");
                }
            }
            Err(e) => tracing::warn!(err = %e, "failed to open history file"),
        }
    }

    pub async fn list_sessions(&self, cwd: Option<&str>) -> Vec<SessionEntry> {
        let index = self.index.lock().await;
        let mut sessions: Vec<SessionEntry> = index
            .sessions
            .iter()
            .filter(|s| {
                if let Some(filter) = cwd
                    && s.cwd != filter
                {
                    return false;
                }
                // Exclude ghost sessions: no title and never updated after creation.
                if s.title.is_none() && s.created_at == s.updated_at {
                    return false;
                }
                true
            })
            .cloned()
            .collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    pub async fn load_history(&self, session_id: &str) -> Vec<Value> {
        let path = self.base_dir.join("history").join(format!("{session_id}.jsonl"));
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => content
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn flush_index(&self) {
        if let Err(e) = tokio::fs::create_dir_all(&self.base_dir).await {
            tracing::warn!(err = %e, "failed to create session store directory");
            return;
        }
        let index = self.index.lock().await;
        match serde_json::to_string_pretty(&*index) {
            Ok(json) => {
                if let Err(e) = tokio::fs::write(self.base_dir.join("sessions.json"), json).await {
                    tracing::warn!(err = %e, "failed to write session index");
                }
            }
            Err(e) => tracing::warn!(err = %e, "failed to serialize session index"),
        }
    }
}

fn session_store_dir() -> PathBuf {
    if let Ok(path) = std::env::var("CURSOR_ACP_SESSIONS_FILE") {
        let p = PathBuf::from(path);
        return p.parent().map(|d| d.to_path_buf()).unwrap_or(p);
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cursor").join("cursor-acp")
}

async fn read_index_from_disk(base: &std::path::Path) -> SessionIndex {
    match tokio::fs::read_to_string(base.join("sessions.json")).await {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => SessionIndex::default(),
    }
}

fn format_session_title(text: &str) -> String {
    let trimmed = text.trim();
    let first_line = trimmed.lines().next().unwrap_or(trimmed);
    if first_line.len() <= 80 {
        first_line.to_string()
    } else {
        format!("{}…", &first_line[..77])
    }
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Convert unix milliseconds to an ISO 8601 / RFC 3339 timestamp string.
fn ms_to_rfc3339(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let nanos = ((ms % 1000) * 1_000_000) as u32;
    let dt = std::time::UNIX_EPOCH + std::time::Duration::new(secs as u64, nanos);
    let datetime: std::time::SystemTime = dt;
    // Format as "YYYY-MM-DDTHH:MM:SSZ"
    let dur = datetime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = dur.as_secs();
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = (total_secs / 3600) % 24;
    let days = total_secs / 86400;
    // Days since epoch to y/m/d (civil calendar from days)
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(days: u64) -> (i64, u64, u64) {
    // Algorithm from Howard Hinnant's chrono-compatible date library.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub fn build_list_response(request_id: &Value, sessions: &[SessionEntry]) -> String {
    let session_values: Vec<Value> = sessions
        .iter()
        .map(|s| {
            let mut entry = json!({
                "sessionId": s.id,
                "cwd": s.cwd,
            });
            if let Some(title) = &s.title {
                entry["title"] = json!(title);
            }
            entry["updatedAt"] = json!(ms_to_rfc3339(s.updated_at));
            entry
        })
        .collect();

    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "sessions": session_values,
        }
    })
    .to_string()
}

pub fn build_load_response(
    request_id: &Value,
    modes: Option<&Value>,
    models: Option<&Value>,
) -> String {
    let mut result = json!({});
    if let Some(m) = modes {
        result["modes"] = m.clone();
    }
    if let Some(m) = models {
        result["models"] = m.clone();
    }

    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    })
    .to_string()
}

pub fn build_history_notification(session_id: &str, update: &Value) -> String {
    json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_list_response_format() {
        let sessions = vec![
            SessionEntry {
                id: "abc-123".to_string(),
                cwd: "/some/path".to_string(),
                title: Some("My session".to_string()),
                created_at: 1700000000000,
                updated_at: 1700000000000,
            },
            SessionEntry {
                id: "def-456".to_string(),
                cwd: "/some/path".to_string(),
                title: None,
                created_at: 1699999000000,
                updated_at: 1699999000000,
            },
        ];

        let response = build_list_response(&json!(5), &sessions);
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["id"], 5);
        let list = parsed["result"]["sessions"].as_array().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0]["sessionId"], "abc-123");
        assert_eq!(list[0]["title"], "My session");
        assert_eq!(list[0]["cwd"], "/some/path");
        // updatedAt should be an RFC3339 string, not a number
        let updated = list[0]["updatedAt"].as_str().unwrap();
        assert!(updated.ends_with('Z'), "expected RFC3339 format, got {updated}");
        assert!(updated.contains('T'), "expected RFC3339 format, got {updated}");
        assert!(list[1].get("title").is_none());
    }

    #[test]
    fn ms_to_rfc3339_known_value() {
        // 1700000000000 ms = 2023-11-14T22:13:20Z
        assert_eq!(ms_to_rfc3339(1700000000000), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn build_load_response_format() {
        let modes = json!({ "currentModeId": "agent", "availableModes": [] });
        let models = json!({ "currentModelId": "auto", "availableModels": [] });

        let response = build_load_response(&json!(7), Some(&modes), Some(&models));
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["id"], 7);
        assert_eq!(parsed["result"]["modes"]["currentModeId"], "agent");
        assert_eq!(parsed["result"]["models"]["currentModelId"], "auto");
    }

    #[test]
    fn build_history_notification_format() {
        let update = json!({
            "sessionUpdate": "user_message_chunk",
            "content": { "type": "text", "text": "hello" }
        });

        let notif = build_history_notification("sess-1", &update);
        let parsed: Value = serde_json::from_str(&notif).unwrap();

        assert_eq!(parsed["method"], "session/update");
        assert_eq!(parsed["params"]["sessionId"], "sess-1");
        assert_eq!(parsed["params"]["update"]["sessionUpdate"], "user_message_chunk");
    }

    #[tokio::test]
    async fn session_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::with_dir(dir.path().to_path_buf()).await;

        store.create_session("s1", "/path/a").await;
        store.create_session("s2", "/path/a").await;
        store.create_session("s3", "/path/b").await;

        // Sessions without titles and no activity are filtered as ghosts.
        let list = store.list_sessions(Some("/path/a")).await;
        assert_eq!(list.len(), 0);

        // Give sessions titles so they appear in the list.
        store.update_title("s1", "First session").await;
        store.update_title("s2", "Second session").await;
        store.update_title("s3", "Third session").await;

        let list = store.list_sessions(Some("/path/a")).await;
        assert_eq!(list.len(), 2);
        let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"s1"));
        assert!(ids.contains(&"s2"));

        let s1 = list.iter().find(|s| s.id == "s1").unwrap();
        assert_eq!(s1.title.as_deref(), Some("First session"));

        let list_b = store.list_sessions(Some("/path/b")).await;
        assert_eq!(list_b.len(), 1);
        assert_eq!(list_b[0].id, "s3");

        // No filter returns all titled sessions
        let all = store.list_sessions(None).await;
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn format_title_short() {
        assert_eq!(format_session_title("Hello world"), "Hello world");
    }

    #[test]
    fn format_title_multiline() {
        assert_eq!(
            format_session_title("First line\nSecond line\nThird"),
            "First line"
        );
    }

    #[test]
    fn format_title_long_truncates() {
        let long = "a".repeat(100);
        let title = format_session_title(&long);
        assert!(title.len() <= 81); // 77 chars + "…" (3 bytes UTF-8)
        assert!(title.ends_with('…'));
    }

    #[tokio::test]
    async fn set_title_if_empty_only_sets_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::with_dir(dir.path().to_path_buf()).await;

        store.create_session("s1", "/path").await;
        store.set_title_if_empty("s1", "First message").await;
        store.set_title_if_empty("s1", "Second message").await;

        let list = store.list_sessions(Some("/path")).await;
        let s1 = list.iter().find(|s| s.id == "s1").unwrap();
        assert_eq!(s1.title.as_deref(), Some("First message"));
    }

    #[tokio::test]
    async fn history_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::with_dir(dir.path().to_path_buf()).await;

        let u1 = json!({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "hi"}});
        let u2 = json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hello"}});

        store.append_history("s1", &u1).await;
        store.append_history("s1", &u2).await;

        let history = store.load_history("s1").await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0]["content"]["text"], "hi");
        assert_eq!(history[1]["content"]["text"], "hello");

        let empty = store.load_history("nonexistent").await;
        assert!(empty.is_empty());
    }
}
