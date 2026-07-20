// src/state/models.rs

use serde::{Deserialize, Serialize};

/// Strongly typed Enum for supported AI architectures.
/// Eliminates "magic numbers" across the codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiModelType {
    ClipVitB32 = 0,
    ClipVitL14 = 1,
    SiglipBase = 2,
    SiglipLarge = 3,
    DinoV2Base = 4,
    Siglip2Base = 5,
    Llm2ClipBase = 6,
    Custom = 7,
}

impl AiModelType {
    /// Maps a raw integer index from the Slint ComboBox to the strong Enum type.
    pub fn from_i32(val: i32) -> Self {
        match val {
            1 => Self::ClipVitL14,
            2 => Self::SiglipBase,
            3 => Self::SiglipLarge,
            4 => Self::DinoV2Base,
            5 => Self::Siglip2Base,
            6 => Self::Llm2ClipBase,
            7 => Self::Custom,
            _ => Self::ClipVitB32,
        }
    }

    /// Returns the standard caching folder name for the model weights.
    pub fn folder_name(&self) -> &'static str {
        match self {
            Self::ClipVitB32 => "clip_vit_b32",
            Self::ClipVitL14 => "clip_vit_l14",
            Self::SiglipBase => "siglip_base",
            Self::SiglipLarge => "siglip_large",
            Self::DinoV2Base => "dinov2_base",
            Self::Siglip2Base => "siglip2_base",
            Self::Llm2ClipBase => "llm2clip_base",
            Self::Custom => "custom",
        }
    }

    /// Returns the standard vector embedding dimensionality of the model architecture.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::ClipVitB32 => 512,
            Self::ClipVitL14 | Self::SiglipBase | Self::DinoV2Base | Self::Siglip2Base => 768,
            Self::SiglipLarge => 1024,
            Self::Llm2ClipBase => 1280,
            Self::Custom => 512,
        }
    }
}

/// Strongly typed Enum for the selected search methodology.
/// Eliminates raw integer indexing magic numbers like 0, 1, 2, 3, 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMethod {
    Exact = 0,
    Perceptual = 1,
    Ai = 2,
    Inventory = 3,
    Qc = 4,
}

impl SearchMethod {
    /// Maps a raw integer index from the Slint ComboBox to the strong Enum type.
    pub fn from_i32(val: i32) -> Self {
        match val {
            1 => Self::Perceptual,
            2 => Self::Ai,
            3 => Self::Inventory,
            4 => Self::Qc,
            _ => Self::Exact,
        }
    }
}

/// Strongly-typed sorting columns to replace magic strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Name,
    Size,
    Score,
    Path,
    Format,
    Dimensions,
    Mipmaps,
    Cubemap,
    Unknown,
}

impl From<&str> for SortColumn {
    fn from(s: &str) -> Self {
        match s {
            "name" => Self::Name,
            "size" => Self::Size,
            "score" => Self::Score,
            "path" => Self::Path,
            "format" => Self::Format,
            "dimensions" => Self::Dimensions,
            "mipmaps" => Self::Mipmaps,
            "cubemap" => Self::Cubemap,
            _ => Self::Unknown,
        }
    }
}

// ==========================================
// --- COMPOSITE APPSETTINGS ----------------
// ==========================================

/// Persistent settings structure serialized and loaded from `Settings.json`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSettings {
    pub paths: crate::app::PathSettings,
    pub extensions: crate::app::ExtensionSettings,
    pub ui: crate::app::UiLayoutSettings,
    pub visuals: crate::app::VisualReportSettings,
    pub prep: crate::app::PreprocessingSettings,
    pub qc: crate::app::QcSettings,
    pub ai: crate::app::AiSettings,
    pub tonemap: crate::app::TonemapSettings,
    pub viewer: crate::app::ViewerConfig,
    pub preview: crate::app::PreviewSettings,

    // Global scanning parameters
    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: i32,
    pub execution_provider: i32,
    pub search_precision: i32,
}

/// Explicitly define fallback configurations for the very first execution
impl Default for AppSettings {
    fn default() -> Self {
        Self {
            paths: crate::app::PathSettings {
                dir_a: "".into(),
                dir_b: "".into(),
                excluded_folders: ".git, .svn, cache, temp".into(),
                query_text: "".into(),
            },
            extensions: crate::app::ExtensionSettings {
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
            },
            ui: crate::app::UiLayoutSettings {
                duplicates_panel_height: 180.0,
                sidebar_width: 380.0,
                compare_sidebar_width: 440.0,
                list_preview_size: 40.0,
                grid_card_size: 135.0,
            },
            visuals: crate::app::VisualReportSettings {
                save_visuals: false,
                visuals_columns: 6,
                visuals_max_count: 100,
                visuals_font_size: 14,
                visuals_scale: 1.5,
            },
            prep: crate::app::PreprocessingSettings {
                prep_luminance: false,
                prep_channels: false,
                prep_r: true,
                prep_g: true,
                prep_b: true,
                prep_a: true,
                prep_tags: "".into(),
                prep_ignore_solid: true,
            },
            qc: crate::app::QcSettings {
                qc_mode: false,
                qc_npot: true,
                qc_mipmaps: true,
                qc_block_align: true,
                qc_bit_depth: true,
                qc_solid_colors: true,
                qc_normals: true,
                qc_normals_tags: "".into(),
                qc_match_by_stem: true,
                qc_hide_same_resolution: false,
                qc_check_bloat: true,
                qc_check_alpha: true,
                qc_check_colorspace: true,
                qc_check_compression: true,
            },
            ai: crate::app::AiSettings {
                ai_model: 0,
                custom_model_path: "".into(),
                custom_model_arch: 0,
                custom_model_dim: 512,
            },
            tonemap: crate::app::TonemapSettings {
                tonemap_enabled: true,
                tonemap_auto_exposure: true,
                tonemap_operator: 2,
            },
            viewer: crate::app::ViewerConfig {
                grid_cols: 1,
                grid_rows: 1,
                manual_brightness: 1.0,
                manual_contrast: 1.0,
                manual_gamma: 1.0,
                aspect_ratio_modifier: 1.0,
                background_mode: 0,
                flipbook_fps: 12.0,
                fit_to_window: true,
                play_speed: 1.0,
                enable_frame_blending: false,
            },
            preview: crate::app::PreviewSettings {
                enable_previews: true,
                preview_quality: 1,
                filter_only_npot: false,
                filter_only_uncompressed: false,
                filter_only_missing_mips: false,
                filter_only_cubemaps: false,
            },
            similarity_threshold: 90.0,
            batch_size: 128,
            search_method: 0,
            execution_provider: 0,
            search_precision: 1,
        }
    }
}

// ==========================================
// --- RESULT SUMMARIES ---------------------
// ==========================================

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
