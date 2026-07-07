// src/utils/settings.rs
use crate::app::AppWindow;
use crate::state::AppSettings;
use std::fs;
use std::path::{Path, PathBuf};

/// Locates or creates the persistent data storage directory next to the executable.
pub fn get_portable_app_data_dir() -> anyhow::Result<PathBuf> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().unwrap_or(Path::new(""));
    let portable_data_dir = exe_dir.join("PixelHand_Data");

    if !portable_data_dir.exists() {
        fs::create_dir_all(&portable_data_dir)?;
    }

    Ok(portable_data_dir)
}

/// Serializes and saves current window state settings to Settings.json
pub fn save_settings(ui: &AppWindow) {
    let settings = AppSettings {
        dir_a: ui.get_dir_a().to_string(),
        dir_b: ui.get_dir_b().to_string(),
        query_text: ui.get_query_text().to_string(),
        similarity_threshold: ui.get_similarity_threshold(),
        batch_size: ui.get_batch_size(),
        search_method: ui.get_search_method(),
        qc_mode: ui.get_qc_mode(),
        qc_npot: ui.get_qc_npot(),
        qc_mipmaps: ui.get_qc_mipmaps(),
        qc_block_align: ui.get_qc_block_align(),
        qc_bit_depth: ui.get_qc_bit_depth(),
        qc_solid_colors: ui.get_qc_solid_colors(),
        qc_normals: ui.get_qc_normals(),
        qc_normals_tags: ui.get_qc_normals_tags().to_string(),
        ext_png: ui.get_ext_png(),
        ext_tga: ui.get_ext_tga(),
        ext_dds: ui.get_ext_dds(),
        ext_bmp: ui.get_ext_bmp(),
        ext_exr: ui.get_ext_exr(),
        ext_hdr: ui.get_ext_hdr(),
        ext_tif: ui.get_ext_tif(),
        ext_webp: ui.get_ext_webp(),
        duplicates_panel_height: ui.get_duplicates_panel_height(),
    };

    if let Ok(dir) = get_portable_app_data_dir() {
        let path = dir.join("Settings.json");
        if let Ok(serialized) = serde_json::to_string_pretty(&settings) {
            let _ = fs::write(path, serialized);
        }
    }
}

/// Loads persistent settings structure from disk
pub fn load_settings() -> Option<AppSettings> {
    let path = get_portable_app_data_dir().ok()?.join("Settings.json");
    if path.exists() {
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str::<AppSettings>(&content).ok()
    } else {
        None
    }
}
