use agent_client_protocol::{
    ModelId, ModelInfo, SessionConfigOption, SessionId, SessionMode, SessionModeId,
    SessionModeState, SessionModelState,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SessionState {
    pub id: SessionId,
    pub cwd: PathBuf,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub title: Option<String>,
    pub current_mode: SessionModeId,
    pub current_model: ModelId,
    pub cursor_resume_id: Option<String>,
    pub workspace_trusted: bool,
    pub config_options: Vec<SessionConfigOption>,
    pub turns: Vec<SessionTurn>,
    pub history_updates: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct SessionTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Default)]
pub struct SessionStore {
    sessions: HashMap<SessionId, SessionState>,
    available_modes: Vec<SessionMode>,
    available_models: Vec<ModelInfo>,
    sessions_file: PathBuf,
}

impl SessionStore {
    fn build_session_state(
        &self,
        session_id: SessionId,
        cwd: PathBuf,
        workspace_trusted: bool,
    ) -> SessionState {
        let now = SystemTime::now();
        SessionState {
            id: session_id,
            cwd,
            created_at: now,
            updated_at: now,
            title: None,
            current_mode: SessionModeId::new("ask"),
            current_model: ModelId::new("auto"),
            cursor_resume_id: None,
            workspace_trusted,
            config_options: Vec::new(),
            turns: Vec::new(),
            history_updates: Vec::new(),
        }
    }

    pub fn new() -> Self {
        Self::new_with_path(default_sessions_file())
    }

    pub fn new_with_path(sessions_file: PathBuf) -> Self {
        let available_modes = vec![
            SessionMode::new("agent", "Agent")
                .description("Write and modify code with full tool access"),
            SessionMode::new("plan", "Plan")
                .description("Design and plan software changes without editing"),
            SessionMode::new("ask", "Ask").description("Read-only code understanding mode"),
        ];
        let available_models = vec![ModelInfo::new("auto", "Auto")];

        let mut store = Self {
            sessions: HashMap::new(),
            available_modes,
            available_models,
            sessions_file,
        };
        store.load_from_disk();
        store
    }

    fn load_from_disk(&mut self) {
        let Ok(contents) = fs::read_to_string(&self.sessions_file) else {
            return;
        };
        let Ok(persisted) = serde_json::from_str::<PersistedStore>(&contents) else {
            return;
        };
        let mut sessions = HashMap::new();
        for item in persisted.sessions {
            let session_id = SessionId::new(item.id.clone());
            let state = SessionState {
                id: session_id.clone(),
                cwd: item.cwd,
                created_at: from_unix_ms(item.created_at_unix_ms),
                updated_at: from_unix_ms(item.updated_at_unix_ms),
                title: item.title,
                current_mode: SessionModeId::new(item.current_mode),
                current_model: ModelId::new(item.current_model),
                cursor_resume_id: item.cursor_resume_id,
                workspace_trusted: item.workspace_trusted,
                config_options: Vec::new(),
                turns: item
                    .turns
                    .unwrap_or_default()
                    .into_iter()
                    .map(|turn| SessionTurn {
                        user: turn.user,
                        assistant: turn.assistant,
                    })
                    .collect(),
                history_updates: item.history_updates.unwrap_or_default(),
            };
            sessions.insert(session_id, state);
        }
        self.sessions = sessions;
    }

    fn persist_to_disk(&self) {
        if let Some(parent) = self.sessions_file.parent() {
            drop(fs::create_dir_all(parent));
        }
        let mut sessions = Vec::new();
        for session in self.sessions.values() {
            sessions.push(PersistedSession {
                id: session.id.to_string(),
                cwd: session.cwd.clone(),
                created_at_unix_ms: to_unix_ms(session.created_at),
                updated_at_unix_ms: to_unix_ms(session.updated_at),
                title: session.title.clone(),
                current_mode: session.current_mode.to_string(),
                current_model: session.current_model.to_string(),
                cursor_resume_id: session.cursor_resume_id.clone(),
                workspace_trusted: session.workspace_trusted,
                turns: Some(
                    session
                        .turns
                        .iter()
                        .map(|turn| PersistedSessionTurn {
                            user: turn.user.clone(),
                            assistant: turn.assistant.clone(),
                        })
                        .collect(),
                ),
                history_updates: Some(session.history_updates.clone()),
            });
        }
        let persisted = PersistedStore { sessions };
        let Ok(serialized) = serde_json::to_string_pretty(&persisted) else {
            return;
        };
        drop(fs::write(&self.sessions_file, serialized));
    }

    pub fn set_resume_id(&mut self, session_id: &SessionId, resume_id: Option<String>) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.cursor_resume_id = resume_id;
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
        }
    }

    pub fn append_turn(&mut self, session_id: &SessionId, user: String, assistant: String) {
        if user.trim().is_empty() && assistant.trim().is_empty() {
            return;
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.turns.push(SessionTurn { user, assistant });
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
        }
    }

    pub fn append_history_updates(&mut self, session_id: &SessionId, updates: Vec<Value>) {
        if updates.is_empty() {
            return;
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.history_updates.extend(updates);
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
        }
    }

    pub fn create_session(&mut self, cwd: PathBuf, workspace_trusted: bool) -> SessionState {
        let session_id = SessionId::new(Uuid::new_v4().to_string());
        let state = self.build_session_state(session_id.clone(), cwd, workspace_trusted);
        self.sessions.insert(session_id, state.clone());
        self.persist_to_disk();
        state
    }

    pub fn ensure_session(
        &mut self,
        session_id: SessionId,
        cwd: PathBuf,
        workspace_trusted: bool,
    ) -> SessionState {
        if let Some(existing) = self.sessions.get(&session_id) {
            return existing.clone();
        }
        let state = self.build_session_state(session_id.clone(), cwd, workspace_trusted);
        self.sessions.insert(session_id, state.clone());
        self.persist_to_disk();
        state
    }

    pub fn get_session(&self, session_id: &SessionId) -> Option<&SessionState> {
        self.sessions.get(session_id)
    }

    pub fn list_sessions(&self) -> Vec<SessionState> {
        self.sessions.values().cloned().collect()
    }

    pub fn get_session_mut(&mut self, session_id: &SessionId) -> Option<&mut SessionState> {
        self.sessions.get_mut(session_id)
    }

    pub fn set_model(
        &mut self,
        session_id: &SessionId,
        model_id: ModelId,
    ) -> Result<(), Vec<String>> {
        let available: Vec<String> = self
            .available_models
            .iter()
            .map(|model| model.model_id.to_string())
            .collect();
        if !available
            .iter()
            .any(|candidate| candidate == model_id.0.as_ref())
        {
            return Err(available);
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.current_model = model_id;
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
        }
        Ok(())
    }

    pub fn set_mode(&mut self, session_id: &SessionId, mode_id: SessionModeId) -> bool {
        let exists = self
            .available_modes
            .iter()
            .any(|mode| mode.id.0 == mode_id.0);
        if !exists {
            return false;
        }
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.current_mode = mode_id;
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
            return true;
        }
        false
    }

    pub fn set_workspace_trusted(&mut self, session_id: &SessionId, workspace_trusted: bool) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.workspace_trusted == workspace_trusted {
                return;
            }
            session.workspace_trusted = workspace_trusted;
            self.persist_to_disk();
        }
    }

    pub fn set_title(&mut self, session_id: &SessionId, title: Option<String>) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.title == title {
                return;
            }
            session.title = title;
            session.updated_at = SystemTime::now();
            self.persist_to_disk();
        }
    }

    pub fn update_models(&mut self, model_ids: Vec<String>) {
        let mut available = Vec::new();
        for model_id in model_ids {
            available.push(ModelInfo::new(model_id.clone(), model_id));
        }
        if !available
            .iter()
            .any(|model| model.model_id.0.as_ref() == "auto")
        {
            available.insert(0, ModelInfo::new("auto", "Auto"));
        }
        self.available_models = available;
    }

    pub fn model_state(&self, session: &SessionState) -> SessionModelState {
        SessionModelState::new(session.current_model.clone(), self.available_models.clone())
    }

    pub fn mode_state(&self, session: &SessionState) -> SessionModeState {
        SessionModeState::new(session.current_mode.clone(), self.available_modes.clone())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedStore {
    sessions: Vec<PersistedSession>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedSession {
    id: String,
    cwd: PathBuf,
    created_at_unix_ms: u128,
    updated_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    current_mode: String,
    current_model: String,
    cursor_resume_id: Option<String>,
    #[serde(default = "default_true")]
    workspace_trusted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turns: Option<Vec<PersistedSessionTurn>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    history_updates: Option<Vec<Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedSessionTurn {
    user: String,
    assistant: String,
}

fn to_unix_ms(value: SystemTime) -> u128 {
    value
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn from_unix_ms(value: u128) -> SystemTime {
    let millis = u64::try_from(value).unwrap_or(u64::MAX);
    SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(millis)
}

fn default_true() -> bool {
    true
}

fn default_sessions_file() -> PathBuf {
    if let Ok(path) = std::env::var("CURSOR_ACP_SESSIONS_FILE") {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cursor")
            .join("cursor-acp")
            .join("sessions.json");
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(profile)
            .join(".cursor")
            .join("cursor-acp")
            .join("sessions.json");
    }
    PathBuf::from(".cursor-acp-sessions.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn persists_and_reloads_sessions() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sessions_file = temp.path().join("sessions.json");

        let created_id = {
            let mut store = SessionStore::new_with_path(sessions_file.clone());
            let session = store.create_session(temp.path().to_path_buf(), true);
            store.set_resume_id(&session.id, Some("cursor-session-123".to_string()));
            session.id
        };

        let store = SessionStore::new_with_path(sessions_file);
        let restored = store
            .get_session(&created_id)
            .expect("session should be restored from disk");
        assert_eq!(
            restored.cursor_resume_id.as_deref(),
            Some("cursor-session-123")
        );
    }

    #[test]
    fn persists_and_reloads_session_title() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sessions_file = temp.path().join("sessions.json");

        let created_id = {
            let mut store = SessionStore::new_with_path(sessions_file.clone());
            let session = store.create_session(temp.path().to_path_buf(), true);
            store.set_title(&session.id, Some("Initial prompt title".to_string()));
            session.id
        };

        let store = SessionStore::new_with_path(sessions_file);
        let restored = store
            .get_session(&created_id)
            .expect("session should be restored from disk");
        assert_eq!(restored.title.as_deref(), Some("Initial prompt title"));
    }

    #[test]
    fn set_workspace_trusted_unchanged_keeps_updated_at() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sessions_file = temp.path().join("sessions.json");
        let mut store = SessionStore::new_with_path(sessions_file);
        let session = store.create_session(temp.path().to_path_buf(), true);
        let before = store
            .get_session(&session.id)
            .expect("session should exist")
            .updated_at;

        std::thread::sleep(Duration::from_millis(2));
        store.set_workspace_trusted(&session.id, true);

        let after = store
            .get_session(&session.id)
            .expect("session should exist")
            .updated_at;
        assert_eq!(before, after);
    }

    #[test]
    fn persists_and_reloads_history_updates() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sessions_file = temp.path().join("sessions.json");

        let created_id = {
            let mut store = SessionStore::new_with_path(sessions_file.clone());
            let session = store.create_session(temp.path().to_path_buf(), true);
            store.append_history_updates(
                &session.id,
                vec![
                    serde_json::json!({"kind":"user","text":"hello"}),
                    serde_json::json!({"kind":"tool_call","id":"tool_1"}),
                ],
            );
            session.id
        };

        let store = SessionStore::new_with_path(sessions_file);
        let restored = store
            .get_session(&created_id)
            .expect("session should be restored from disk");
        assert_eq!(restored.history_updates.len(), 2);
    }
}
