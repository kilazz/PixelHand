// src/state/models.rs
use serde::{Deserialize, Serialize};

/// Persistent settings structure (Saved to Settings.json)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSettings {
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: i32,
    pub execution_provider: i32,
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_normals_tags: String,
    pub ext_png: bool,
    pub ext_tga: bool,
    pub ext_dds: bool,
    pub ext_bmp: bool,
    pub ext_exr: bool,
    pub ext_hdr: bool,
    pub ext_tif: bool,
    pub ext_webp: bool,
    pub duplicates_panel_height: f32,
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
            ext_png: true,
            ext_tga: true,
            ext_dds: true,
            ext_bmp: true,
            ext_exr: true,
            ext_hdr: true,
            ext_tif: true,
            ext_webp: true,
            duplicates_panel_height: 180.0,
        }
    }
}

/// Represents a single duplicate image file
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

/// Represents a cluster of duplicate files
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DuplicateGroupSummary {
    pub hash: String,
    pub files: Vec<DuplicateFileSummary>,
}

/// Represents an issue found during Quality Control (QC)
#[derive(Serialize, Clone, Debug)]
pub struct QcIssueSummary {
    pub path: String,
    pub issue: String,
    pub details: String,
}

/// Represents a visual match from AI Semantic Search
#[derive(Serialize, Clone, Debug)]
pub struct AiSearchResultSummary {
    pub path: String,
    pub similarity: f32,
}
