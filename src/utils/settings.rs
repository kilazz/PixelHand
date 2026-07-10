// src/utils/settings.rs

use crate::app::AppWindow;
use crate::state::AppSettings;
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

/// Serializes active UI state properties from the Slint component and persists them to `Settings.json`.
pub fn save_settings(ui: &AppWindow) {
    let settings = AppSettings {
        // Directories and Search Context
        dir_a: ui.get_dir_a().to_string(),
        dir_b: ui.get_dir_b().to_string(),
        query_text: ui.get_query_text().to_string(),
        similarity_threshold: ui.get_similarity_threshold(),
        batch_size: ui.get_batch_size(),
        search_method: ui.get_search_method(),
        execution_provider: ui.get_execution_provider(),

        // Quality Control Options
        qc_mode: ui.get_search_method() == 4,
        qc_npot: ui.get_qc_npot(),
        qc_mipmaps: ui.get_qc_mipmaps(),
        qc_block_align: ui.get_qc_block_align(),
        qc_bit_depth: ui.get_qc_bit_depth(),
        qc_solid_colors: ui.get_qc_solid_colors(),
        qc_normals: ui.get_qc_normals(),
        qc_normals_tags: ui.get_qc_normals_tags().to_string(),
        qc_match_by_stem: ui.get_qc_match_by_stem(),
        qc_hide_same_resolution: ui.get_qc_hide_same_resolution(),

        // Relative Quality Control parameters
        qc_check_bloat: ui.get_qc_check_bloat(),
        qc_check_alpha: ui.get_qc_check_alpha(),
        qc_check_colorspace: ui.get_qc_check_colorspace(),
        qc_check_compression: ui.get_qc_check_compression(),

        // Target Extensions
        ext_png: ui.get_ext_png(),
        ext_jpg: ui.get_ext_jpg(),
        ext_tga: ui.get_ext_tga(),
        ext_dds: ui.get_ext_dds(),
        ext_bmp: ui.get_ext_bmp(),
        ext_exr: ui.get_ext_exr(),
        ext_hdr: ui.get_ext_hdr(),
        ext_tif: ui.get_ext_tif(),
        ext_webp: ui.get_ext_webp(),
        ext_gif: ui.get_ext_gif(),
        ext_psd: ui.get_ext_psd(),
        ext_jxl: ui.get_ext_jxl(),
        ext_heic: ui.get_ext_heic(),
        ext_avif: ui.get_ext_avif(),

        // UI Drag States
        duplicates_panel_height: ui.get_duplicates_panel_height(),
        sidebar_width: ui.get_sidebar_width(),
        compare_sidebar_width: ui.get_compare_sidebar_width(),
        list_preview_size: ui.get_list_preview_size(),

        // Visual Reports Configurations
        save_visuals: ui.get_save_visuals(),
        visuals_columns: ui.get_visuals_columns(),
        visuals_max_count: ui.get_visuals_max_count(),
        visuals_font_size: ui.get_visuals_font_size(),
        visuals_scale: ui.get_visuals_scale(),

        // Image Pre-processing Logic Configurations
        prep_luminance: ui.get_prep_luminance(),
        prep_channels: ui.get_prep_channels(),
        prep_r: ui.get_prep_r(),
        prep_g: ui.get_prep_g(),
        prep_b: ui.get_prep_b(),
        prep_a: ui.get_prep_a(),
        prep_tags: ui.get_prep_tags().to_string(),
        prep_ignore_solid: ui.get_prep_ignore_solid(),

        // Exclude Filters, LanceDB Precision, and AI Model Configuration
        excluded_folders: ui.get_excluded_folders().to_string(),
        search_precision: ui.get_search_precision(),
        ai_model: ui.get_ai_model(),

        // Custom Model Local Options
        custom_model_path: ui.get_custom_model_path().to_string(),
        custom_model_arch: ui.get_custom_model_arch(),
        custom_model_dim: ui.get_custom_model_dim(),

        // HDR Tonemapping Options
        tonemap_enabled: ui.get_tonemap_enabled(),
        tonemap_operator: ui.get_tonemap_operator(),

        // Preview & Smart Filter settings
        enable_previews: ui.get_enable_previews(),
        preview_quality: ui.get_preview_quality(),
        filter_only_npot: ui.get_filter_only_npot(),
        filter_only_uncompressed: ui.get_filter_only_uncompressed(),
        filter_only_missing_mips: ui.get_filter_only_missing_mips(),
        filter_only_cubemaps: ui.get_filter_only_cubemaps(),
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
                    "Corrupted Settings.json file format at {}: {}",
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
