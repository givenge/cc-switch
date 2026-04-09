use crate::codex_config;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct CodexLiveConfig {
    pub auth: Value,
    pub config: String,
}

#[tauri::command]
pub async fn get_codex_live_config() -> Result<CodexLiveConfig, String> {
    let auth = codex_config::read_codex_auth_json().map_err(|e| e.to_string())?;
    let config = codex_config::read_codex_config_text().map_err(|e| e.to_string())?;

    Ok(CodexLiveConfig { auth, config })
}
