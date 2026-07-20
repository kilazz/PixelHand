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
            Self::Custom => "custom", // Overridden dynamically by custom_model_path where used
        }
    }

    /// Returns the standard vector embedding dimensionality of the model architecture.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::ClipVitB32 => 512,
            Self::ClipVitL14 | Self::SiglipBase | Self::DinoV2Base | Self::Siglip2Base => 768,
            Self::SiglipLarge => 1024,
            Self::Llm2ClipBase => 1280,
            Self::Custom => 512, // Overridden dynamically by custom_model_dim where used
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

// ==========================================
// --- DECOMPOSED SETTINGS SUB-STRUCTS ------
// ==========================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PathSettings {
    pub dir_a: String,
    pub dir_b: String,
    pub excluded_folders: String,
    pub query_text: String,
}

impl Default for PathSettings {
    fn default() -> Self {
        Self {
            dir_a: String::new(),
            dir_b: String::new(),
            excluded_folders: ".git, .svn, cache, temp".to_string(),
            query_text: String::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExtensionSettings {
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
}

impl Default for ExtensionSettings {
    fn default() -> Self {
        Self {
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
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UiLayoutSettings {
    pub duplicates_panel_height: f32,
    pub sidebar_width: f32,
    pub compare_sidebar_width: f32,
    pub list_preview_size: f32,
}

impl Default for UiLayoutSettings {
    fn default() -> Self {
        Self {
            duplicates_panel_height: 180.0,
            sidebar_width: 380.0,
            compare_sidebar_width: 440.0,
            list_preview_size: 40.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisualReportSettings {
    pub save_visuals: bool,
    pub visuals_columns: i32,
    pub visuals_max_count: i32,
    pub visuals_font_size: i32,
    pub visuals_scale: f32,
}

impl Default for VisualReportSettings {
    fn default() -> Self {
        Self {
            save_visuals: false,
            visuals_columns: 6,
            visuals_max_count: 100,
            visuals_font_size: 14,
            visuals_scale: 1.5,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PreprocessingSettings {
    pub prep_luminance: bool,
    pub prep_channels: bool,
    pub prep_r: bool,
    pub prep_g: bool,
    pub prep_b: bool,
    pub prep_a: bool,
    pub prep_tags: String,
    pub prep_ignore_solid: bool,
}

impl Default for PreprocessingSettings {
    fn default() -> Self {
        Self {
            prep_luminance: false,
            prep_channels: false,
            prep_r: true,
            prep_g: true,
            prep_b: true,
            prep_a: true,
            prep_tags: String::new(),
            prep_ignore_solid: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QcSettings {
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
    pub qc_check_bloat: bool,
    pub qc_check_alpha: bool,
    pub qc_check_colorspace: bool,
    pub qc_check_compression: bool,
}

impl Default for QcSettings {
    fn default() -> Self {
        Self {
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
            qc_check_bloat: true,
            qc_check_alpha: true,
            qc_check_colorspace: true,
            qc_check_compression: true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AiSettings {
    pub ai_model: i32,
    pub custom_model_path: String,
    pub custom_model_arch: i32,
    pub custom_model_dim: i32,
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            ai_model: AiModelType::ClipVitB32 as i32,
            custom_model_path: String::new(),
            custom_model_arch: 0,
            custom_model_dim: 512,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TonemapSettings {
    pub tonemap_enabled: bool,
    pub tonemap_auto_exposure: bool,
    pub tonemap_operator: i32,
}

impl Default for TonemapSettings {
    fn default() -> Self {
        Self {
            tonemap_enabled: true,
            tonemap_auto_exposure: true,
            tonemap_operator: 2,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ViewerSettings {
    pub grid_cols: i32,
    pub grid_rows: i32,
    pub manual_brightness: f32,
    pub manual_contrast: f32,
    pub manual_gamma: f32,
    pub aspect_ratio_modifier: f32,
    pub background_mode: i32,
    pub flipbook_fps: f32,
    pub fit_to_window: bool,
    pub play_speed: f32,
    pub enable_frame_blending: bool,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        Self {
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
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PreviewSettings {
    pub enable_previews: bool,
    pub preview_quality: i32,
    pub filter_only_npot: bool,
    pub filter_only_uncompressed: bool,
    pub filter_only_missing_mips: bool,
    pub filter_only_cubemaps: bool,
}

impl Default for PreviewSettings {
    fn default() -> Self {
        Self {
            enable_previews: true,
            preview_quality: 1,
            filter_only_npot: false,
            filter_only_uncompressed: false,
            filter_only_missing_mips: false,
            filter_only_cubemaps: false,
        }
    }
}

// ==========================================
// --- COMPOSITE APPSETTINGS ----------------
// ==========================================

/// Persistent settings structure serialized and loaded from `Settings.json`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AppSettings {
    pub paths: PathSettings,
    pub extensions: ExtensionSettings,
    pub ui: UiLayoutSettings,
    pub visuals: VisualReportSettings,
    pub prep: PreprocessingSettings,
    pub qc: QcSettings,
    pub ai: AiSettings,
    pub tonemap: TonemapSettings,
    pub viewer: ViewerSettings,
    pub preview: PreviewSettings,

    // Global scanning parameters
    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: i32,
    pub execution_provider: i32,
    pub search_precision: i32,
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
