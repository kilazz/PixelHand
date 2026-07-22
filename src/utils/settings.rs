// src/utils/settings.rs

use crate::state::models::AppSettings;
use anyhow::Context;
use std::fs;
use std::path::{Path, PathBuf};

/// Resolves the absolute directory path of the portable data directory next to the executable.
/// Creates the directory if it does not already exist.
pub fn get_portable_app_data_dir() -> anyhow::Result<PathBuf> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().unwrap_or_else(|| Path::new(""));
    let portable_data_dir = exe_dir.join("PixelHand_Data");

    if !portable_data_dir.exists() {
        fs::create_dir_all(&portable_data_dir).with_context(|| {
            format!(
                "Failed to create portable data directory: {}",
                portable_data_dir.display()
            )
        })?;
    }

    Ok(portable_data_dir)
}

/// Serializes active UI state properties from the Slint global singletons and persists them to `Settings.json`.
pub fn save_settings(
    scan_config: &crate::app::ScanConfig,
    viewport_state: &crate::app::ViewportState,
) {
    let settings = AppSettings {
        paths: scan_config.get_paths(),
        extensions: scan_config.get_extensions(),
        ui: scan_config.get_ui(),
        visuals: scan_config.get_visuals(),
        prep: scan_config.get_prep(),
        qc: scan_config.get_qc(),
        ai: scan_config.get_ai(),
        tonemap: viewport_state.get_tonemap(),
        viewer: viewport_state.get_viewer(),
        preview: scan_config.get_preview(),

        // Global scanning settings
        similarity_threshold: scan_config.get_similarity_threshold(),
        batch_size: scan_config.get_batch_size(),
        search_method: scan_config.get_search_method(),
        execution_provider: scan_config.get_execution_provider(),
        search_precision: scan_config.get_search_precision(),
    };

    match get_portable_app_data_dir() {
        Ok(dir) => {
            let path = dir.join("Settings.json");
            match serde_json::to_string_pretty(&settings) {
                Ok(serialized) => {
                    if let Err(e) = fs::write(&path, serialized) {
                        tracing::error!(
                            "Failed to write Settings.json to {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize application settings: {}", e);
                }
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to resolve portable data directory to save settings: {}",
                e
            );
        }
    }
}

/// Loads and deserializes previously saved config values from `Settings.json` if available.
pub fn load_settings() -> Option<AppSettings> {
    let dir = match get_portable_app_data_dir() {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(
                "Failed to resolve portable data directory during settings load: {}",
                e
            );
            return None;
        }
    };

    let path = dir.join("Settings.json");
    if !path.exists() {
        return None;
    }

    match fs::read_to_string(&path) {
        Ok(content) => match serde_json::from_str::<AppSettings>(&content) {
            Ok(settings) => Some(settings),
            Err(e) => {
                tracing::error!(
                    "Corrupted Settings.json file format at {}: {}. Auto-healing file.",
                    path.display(),
                    e
                );
                None
            }
        },
        Err(e) => {
            tracing::error!(
                "Failed to read Settings.json from {}: {}",
                path.display(),
                e
            );
            None
        }
    }
}
