// src/utils/settings.rs

use crate::state::models::{
    AiSettings, AppSettings, ExtensionSettings, PathSettings, PreprocessingSettings,
    PreviewSettings, QcSettings, TonemapSettings, UiLayoutSettings, ViewerSettings,
    VisualReportSettings,
};
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

// Declares the clean reverse-mapping macro to construct nested structures
macro_rules! build_sub_settings {
    ($struct_name:ident, { $($field:ident => $expr:expr),* $(,)? }) => {
        $struct_name {
            $(
                $field: $expr,
            )*
        }
    };
}

/// Serializes active UI state properties from the Slint global Store and persists them to `Settings.json`.
pub fn save_settings(store: &crate::app::Store) {
    let paths = store.get_paths();
    let ext = store.get_extensions();
    let ui = store.get_ui();
    let visuals = store.get_visuals();
    let prep = store.get_prep();
    let qc = store.get_qc();
    let ai = store.get_ai();
    let tonemap = store.get_tonemap();
    let viewer = store.get_viewer();
    let preview = store.get_preview();

    let settings = AppSettings {
        paths: build_sub_settings!(PathSettings, {
            dir_a => paths.dir_a.to_string(),
            dir_b => paths.dir_b.to_string(),
            query_text => paths.query_text.to_string(),
            excluded_folders => paths.excluded_folders.to_string(),
        }),
        extensions: build_sub_settings!(ExtensionSettings, {
            ext_png => ext.ext_png,
            ext_jpg => ext.ext_jpg,
            ext_tga => ext.ext_tga,
            ext_dds => ext.ext_dds,
            ext_bmp => ext.ext_bmp,
            ext_exr => ext.ext_exr,
            ext_hdr => ext.ext_hdr,
            ext_tif => ext.ext_tif,
            ext_webp => ext.ext_webp,
            ext_gif => ext.ext_gif,
            ext_psd => ext.ext_psd,
            ext_jxl => ext.ext_jxl,
            ext_heic => ext.ext_heic,
            ext_avif => ext.ext_avif,
        }),
        ui: build_sub_settings!(UiLayoutSettings, {
            duplicates_panel_height => ui.duplicates_panel_height,
            sidebar_width => ui.sidebar_width,
            compare_sidebar_width => ui.compare_sidebar_width,
            list_preview_size => ui.list_preview_size,
            grid_card_size => ui.grid_card_size,
        }),
        visuals: build_sub_settings!(VisualReportSettings, {
            save_visuals => visuals.save_visuals,
            visuals_columns => visuals.visuals_columns,
            visuals_max_count => visuals.visuals_max_count,
            visuals_font_size => visuals.visuals_font_size,
            visuals_scale => visuals.visuals_scale,
        }),
        prep: build_sub_settings!(PreprocessingSettings, {
            prep_luminance => prep.prep_luminance,
            prep_channels => prep.prep_channels,
            prep_r => prep.prep_r,
            prep_g => prep.prep_g,
            prep_b => prep.prep_b,
            prep_a => prep.prep_a,
            prep_tags => prep.prep_tags.to_string(),
            prep_ignore_solid => prep.prep_ignore_solid,
        }),
        qc: build_sub_settings!(QcSettings, {
            qc_mode => store.get_search_method() == 4,
            qc_npot => qc.qc_npot,
            qc_mipmaps => qc.qc_mipmaps,
            qc_block_align => qc.qc_block_align,
            qc_bit_depth => qc.qc_bit_depth,
            qc_solid_colors => qc.qc_solid_colors,
            qc_normals => qc.qc_normals,
            qc_normals_tags => qc.qc_normals_tags.to_string(),
            qc_match_by_stem => qc.qc_match_by_stem,
            qc_hide_same_resolution => qc.qc_hide_same_resolution,
            qc_check_bloat => qc.qc_check_bloat,
            qc_check_alpha => qc.qc_check_alpha,
            qc_check_colorspace => qc.qc_check_colorspace,
            qc_check_compression => qc.qc_check_compression,
        }),
        ai: build_sub_settings!(AiSettings, {
            ai_model => ai.ai_model,
            custom_model_path => ai.custom_model_path.to_string(),
            custom_model_arch => ai.custom_model_arch,
            custom_model_dim => ai.custom_model_dim,
        }),
        tonemap: build_sub_settings!(TonemapSettings, {
            tonemap_enabled => tonemap.tonemap_enabled,
            tonemap_auto_exposure => tonemap.tonemap_auto_exposure,
            tonemap_operator => tonemap.tonemap_operator,
        }),
        viewer: build_sub_settings!(ViewerSettings, {
            grid_cols => viewer.grid_cols,
            grid_rows => viewer.grid_rows,
            manual_brightness => viewer.manual_brightness,
            manual_contrast => viewer.manual_contrast,
            manual_gamma => viewer.manual_gamma,
            aspect_ratio_modifier => viewer.aspect_ratio_modifier,
            background_mode => viewer.background_mode,
            flipbook_fps => viewer.flipbook_fps,
            fit_to_window => viewer.fit_to_window,
            play_speed => viewer.play_speed,
            enable_frame_blending => viewer.enable_frame_blending,
        }),
        preview: build_sub_settings!(PreviewSettings, {
            enable_previews => preview.enable_previews,
            preview_quality => preview.preview_quality,
            filter_only_npot => preview.filter_only_npot,
            filter_only_uncompressed => preview.filter_only_uncompressed,
            filter_only_missing_mips => preview.filter_only_missing_mips,
            filter_only_cubemaps => preview.filter_only_cubemaps,
        }),

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
