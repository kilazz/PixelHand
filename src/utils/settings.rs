// src/utils/settings.rs

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

/// Serializes active UI state properties from the Slint global Store and persists them to `Settings.json`.
pub fn save_settings(store: &crate::app::Store) {
    let settings = AppSettings {
        // Directories and Search Context
        dir_a: store.get_dir_a().to_string(),
        dir_b: store.get_dir_b().to_string(),
        query_text: store.get_query_text().to_string(),
        similarity_threshold: store.get_similarity_threshold(),
        batch_size: store.get_batch_size(),
        search_method: store.get_search_method(),
        execution_provider: store.get_execution_provider(),

        // Quality Control Options
        qc_mode: store.get_search_method() == 4,
        qc_npot: store.get_qc_npot(),
        qc_mipmaps: store.get_qc_mipmaps(),
        qc_block_align: store.get_qc_block_align(),
        qc_bit_depth: store.get_qc_bit_depth(),
        qc_solid_colors: store.get_qc_solid_colors(),
        qc_normals: store.get_qc_normals(),
        qc_normals_tags: store.get_qc_normals_tags().to_string(),
        qc_match_by_stem: store.get_qc_match_by_stem(),
        qc_hide_same_resolution: store.get_qc_hide_same_resolution(),

        // Relative Quality Control parameters
        qc_check_bloat: store.get_qc_check_bloat(),
        qc_check_alpha: store.get_qc_check_alpha(),
        qc_check_colorspace: store.get_qc_check_colorspace(),
        qc_check_compression: store.get_qc_check_compression(),

        // Target Extensions
        ext_png: store.get_ext_png(),
        ext_jpg: store.get_ext_jpg(),
        ext_tga: store.get_ext_tga(),
        ext_dds: store.get_ext_dds(),
        ext_bmp: store.get_ext_bmp(),
        ext_exr: store.get_ext_exr(),
        ext_hdr: store.get_ext_hdr(),
        ext_tif: store.get_ext_tif(),
        ext_webp: store.get_ext_webp(),
        ext_gif: store.get_ext_gif(),
        ext_psd: store.get_ext_psd(),
        ext_jxl: store.get_ext_jxl(),
        ext_heic: store.get_ext_heic(),
        ext_avif: store.get_ext_avif(),

        // UI Drag States
        duplicates_panel_height: store.get_duplicates_panel_height(),
        sidebar_width: store.get_sidebar_width(),
        compare_sidebar_width: store.get_compare_sidebar_width(),
        list_preview_size: store.get_list_preview_size(),

        // Visual Reports Configurations
        save_visuals: store.get_save_visuals(),
        visuals_columns: store.get_visuals_columns(),
        visuals_max_count: store.get_visuals_max_count(),
        visuals_font_size: store.get_visuals_font_size(),
        visuals_scale: store.get_visuals_scale(),

        // Image Pre-processing Logic Configurations
        prep_luminance: store.get_prep_luminance(),
        prep_channels: store.get_prep_channels(),
        prep_r: store.get_prep_r(),
        prep_g: store.get_prep_g(),
        prep_b: store.get_prep_b(),
        prep_a: store.get_prep_a(),
        prep_tags: store.get_prep_tags().to_string(),
        prep_ignore_solid: store.get_prep_ignore_solid(),

        // Exclude Filters, LanceDB Precision, and AI Model Configuration
        excluded_folders: store.get_excluded_folders().to_string(),
        search_precision: store.get_search_precision(),
        ai_model: store.get_ai_model(),

        // Custom Model Local Options
        custom_model_path: store.get_custom_model_path().to_string(),
        custom_model_arch: store.get_custom_model_arch(),
        custom_model_dim: store.get_custom_model_dim(),

        // HDR Tonemapping Options
        tonemap_enabled: store.get_tonemap_enabled(),
        tonemap_auto_exposure: store.get_tonemap_auto_exposure(),
        tonemap_operator: store.get_tonemap_operator(),

        // Preview & Smart Filter settings
        enable_previews: store.get_enable_previews(),
        preview_quality: store.get_preview_quality(),
        filter_only_npot: store.get_filter_only_npot(),
        filter_only_uncompressed: store.get_filter_only_uncompressed(),
        filter_only_missing_mips: store.get_filter_only_missing_mips(),
        filter_only_cubemaps: store.get_filter_only_cubemaps(),
        grid_cols: store.get_grid_cols(),
        grid_rows: store.get_grid_rows(),
        manual_brightness: store.get_manual_brightness(),
        manual_contrast: store.get_manual_contrast(),
        manual_gamma: store.get_manual_gamma(),
        aspect_ratio_modifier: store.get_aspect_ratio_modifier(),
        background_mode: store.get_background_mode(),
        flipbook_fps: store.get_flipbook_fps(),
        fit_to_window: store.get_fit_to_window(),
        play_speed: store.get_play_speed(),
        enable_frame_blending: store.get_enable_frame_blending(),
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
