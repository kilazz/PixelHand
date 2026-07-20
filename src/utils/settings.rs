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

/// Serializes active UI state properties from the Slint global Store and persists them to `Settings.json`.
pub fn save_settings(store: &crate::app::Store) {
    let settings = AppSettings {
        paths: store.get_paths(),
        extensions: store.get_extensions(),
        ui: store.get_ui(),
        visuals: store.get_visuals(),
        prep: store.get_prep(),
        qc: store.get_qc(),
        ai: store.get_ai(),
        tonemap: store.get_tonemap(),
        viewer: store.get_viewer(),
        preview: store.get_preview(),

        // Global scanning settings
        similarity_threshold: store.get_similarity_threshold(),
        batch_size: store.get_batch_size(),
        search_method: store.get_search_method(),
        execution_provider: store.get_execution_provider(),
        search_precision: store.get_search_precision(),
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

/// Loads and deserializes previously saved config values from `Settings.json`.
/// Automatically auto-heals by overwriting the file with default settings if it is corrupted or outdated.
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
                tracing::warn!(
                    "Settings.json at {} is corrupted or outdated: {}. Auto-healing file with default configuration.",
                    path.display(),
                    e
                );

                // Self-healing: Overwrite the corrupted/outdated file with default settings immediately
                let default_settings = AppSettings::default();
                match serde_json::to_string_pretty(&default_settings) {
                    Ok(serialized) => {
                        if let Err(write_err) = fs::write(&path, serialized) {
                            tracing::error!("Failed to auto-heal Settings.json: {}", write_err);
                        } else {
                            tracing::info!(
                                "Settings.json successfully healed with default settings."
                            );
                        }
                    }
                    Err(serialize_err) => {
                        tracing::error!(
                            "Failed to serialize default settings during healing: {}",
                            serialize_err
                        );
                    }
                }
                Some(default_settings)
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
