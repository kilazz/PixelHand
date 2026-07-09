// src/state/models.rs

use serde::{Deserialize, Serialize};

/// Persistent settings structure serialized and loaded from `Settings.json`.
/// This holds the core state of checkboxes, configuration parameters, and slider thresholds.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSettings {
    // Directories and Search Context
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: i32,
    pub execution_provider: i32,

    // Quality Control (QC) Options
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_normals_tags: String,
    pub qc_match_by_stem: bool,
    pub qc_hide_same_resolution: bool,

    // File Extensions
    pub ext_png: bool,
    pub ext_tga: bool,
    pub ext_dds: bool,
    pub ext_bmp: bool,
    pub ext_exr: bool,
    pub ext_hdr: bool,
    pub ext_tif: bool,
    pub ext_webp: bool,

    // UI Layout States
    pub duplicates_panel_height: f32,
    pub sidebar_width: f32,

    // Visual Reports (Contact Sheets)
    pub save_visuals: bool,
    pub visuals_columns: i32,
    pub visuals_max_count: i32,
    pub visuals_font_size: i32,
    pub visuals_scale: f32,

    // Image Pre-processing Options
    pub prep_luminance: bool,
    pub prep_channels: bool,
    pub prep_r: bool,
    pub prep_g: bool,
    pub prep_b: bool,
    pub prep_a: bool,
    pub prep_tags: String,
    pub prep_ignore_solid: bool,

    // Directory Traversing and Vector Matching Limits
    pub excluded_folders: String,
    pub search_precision: i32,

    // Registered active AI model state persistent property
    pub ai_model: i32, // 0: CLIP-B/32, 1: SigLIP-B, 2: DINOv2-B

    // HDR Tonemapping Configuration
    pub tonemap_enabled: bool,
    pub tonemap_operator: i32, // 0: ACES Filmic, 1: ICtCp, 2: Khronos PBR Neutral
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dir_a: String::new(),
            dir_b: String::new(),
            query_text: String::new(),
            similarity_threshold: 90.0,
            batch_size: 128,
            search_method: 0,
            execution_provider: 0,

            qc_mode: false,
            qc_npot: true,
            qc_mipmaps: true,
            qc_block_align: true,
            qc_bit_depth: true,
            qc_solid_colors: true,
            qc_normals: true,
            qc_normals_tags: String::new(),
            qc_match_by_stem: true,
            qc_hide_same_resolution: false,

            ext_png: true,
            ext_tga: true,
            ext_dds: true,
            ext_bmp: true,
            ext_exr: true,
            ext_hdr: true,
            ext_tif: true,
            ext_webp: true,

            duplicates_panel_height: 180.0,
            sidebar_width: 380.0,

            save_visuals: false,
            visuals_columns: 6,
            visuals_max_count: 100,
            visuals_font_size: 14,
            visuals_scale: 1.5,

            prep_luminance: false,
            prep_channels: false,
            prep_r: true,
            prep_g: true,
            prep_b: true,
            prep_a: true,
            prep_tags: String::new(),
            prep_ignore_solid: true,

            excluded_folders: ".git, .svn, cache, temp".to_string(),
            search_precision: 1, // Balanced (Default)

            // Default model index CLIP-B/32
            ai_model: 0,

            // Default Tonemapping Options
            tonemap_enabled: true,
            tonemap_operator: 0, // Default to ACES Filmic
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
