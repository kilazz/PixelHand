// src/state/models.rs

use serde::{Deserialize, Serialize};

/// Persistent settings structure serialized and loaded from `Settings.json`.
/// Struct fields are ordered descending by memory size to minimize struct alignment padding.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSettings {
    // 24-byte String Allocations
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub qc_normals_tags: String,
    pub prep_tags: String,
    pub excluded_folders: String,
    pub custom_model_path: String,

    // 4-byte Integers & Floats
    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: i32,
    pub execution_provider: i32,
    pub duplicates_panel_height: f32,
    pub sidebar_width: f32,
    pub compare_sidebar_width: f32,
    pub list_preview_size: f32,
    pub visuals_columns: i32,
    pub visuals_max_count: i32,
    pub visuals_font_size: i32,
    pub visuals_scale: f32,
    pub search_precision: i32,
    pub ai_model: i32,
    pub custom_model_arch: i32,
    pub custom_model_dim: i32,
    pub tonemap_operator: i32,
    pub preview_quality: i32,

    // 1-byte Logical Flags
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_match_by_stem: bool,
    pub qc_hide_same_resolution: bool,
    pub qc_check_bloat: bool,
    pub qc_check_alpha: bool,
    pub qc_check_colorspace: bool,
    pub qc_check_compression: bool,
    pub ext_png: bool,
    pub ext_jpg: bool,
    pub ext_tga: bool,
    pub ext_dds: bool,
    pub ext_bmp: bool,
    pub ext_exr: bool,
    pub ext_hdr: bool,
    pub ext_tif: bool,
    pub ext_webp: bool,
    pub ext_gif: bool,
    pub ext_psd: bool,
    pub ext_jxl: bool,
    pub ext_heic: bool,
    pub ext_avif: bool,
    pub save_visuals: bool,
    pub prep_luminance: bool,
    pub prep_channels: bool,
    pub prep_r: bool,
    pub prep_g: bool,
    pub prep_b: bool,
    pub prep_a: bool,
    pub prep_ignore_solid: bool,
    pub tonemap_enabled: bool,
    pub tonemap_auto_exposure: bool,
    pub enable_previews: bool,
    pub filter_only_npot: bool,
    pub filter_only_uncompressed: bool,
    pub filter_only_missing_mips: bool,
    pub filter_only_cubemaps: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dir_a: String::new(),
            dir_b: String::new(),
            query_text: String::new(),
            qc_normals_tags: String::new(),
            prep_tags: String::new(),
            excluded_folders: ".git, .svn, cache, temp".to_string(),
            custom_model_path: String::new(),

            similarity_threshold: 90.0,
            batch_size: 128,
            search_method: 0,
            execution_provider: 0,
            duplicates_panel_height: 180.0,
            sidebar_width: 380.0,
            compare_sidebar_width: 440.0,
            list_preview_size: 40.0,
            visuals_columns: 6,
            visuals_max_count: 100,
            visuals_font_size: 14,
            visuals_scale: 1.5,
            search_precision: 1,
            ai_model: 0,
            custom_model_arch: 0,
            custom_model_dim: 512,
            tonemap_operator: 2,
            preview_quality: 1,

            qc_mode: false,
            qc_npot: true,
            qc_mipmaps: true,
            qc_block_align: true,
            qc_bit_depth: true,
            qc_solid_colors: true,
            qc_normals: true,
            qc_match_by_stem: true,
            qc_hide_same_resolution: false,
            qc_check_bloat: true,
            qc_check_alpha: true,
            qc_check_colorspace: true,
            qc_check_compression: true,
            ext_png: true,
            ext_jpg: true,
            ext_tga: true,
            ext_dds: true,
            ext_bmp: true,
            ext_exr: true,
            ext_hdr: true,
            ext_tif: true,
            ext_webp: true,
            ext_gif: true,
            ext_psd: true,
            ext_jxl: true,
            ext_heic: true,
            ext_avif: true,
            save_visuals: false,
            prep_luminance: false,
            prep_channels: false,
            prep_r: true,
            prep_g: true,
            prep_b: true,
            prep_a: true,
            prep_ignore_solid: true,
            tonemap_enabled: true,
            tonemap_auto_exposure: true,
            enable_previews: true,
            filter_only_npot: false,
            filter_only_uncompressed: false,
            filter_only_missing_mips: false,
            filter_only_cubemaps: false,
        }
    }
}

/// Represents the details of a single duplicate image file within a duplicate cluster.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DuplicateFileSummary {
    pub path: String,
    pub size: u64,
    pub width: usize,
    pub height: usize,
    pub format_str: String,
    pub compression_format: String,
    pub color_space: String,
    pub has_alpha: bool,
    pub bit_depth: u32,
    pub mipmap_count: u32,
    pub is_cubemap: bool,
    pub similarity: f32,
}

/// Represents an isolated duplicate cluster containing a list of matching duplicate files.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DuplicateGroupSummary {
    pub hash: String,
    pub files: Vec<DuplicateFileSummary>,
}

/// Represents a technical issue identified during Quality Control auditing.
#[derive(Serialize, Clone, Debug)]
pub struct QcIssueSummary {
    pub path: String,
    pub issue: String,
    pub details: String,
}

/// Represents a visual similarity match returned from the AI database search.
#[derive(Serialize, Clone, Debug)]
pub struct AiSearchResultSummary {
    pub path: String,
    pub similarity: f32,
}
