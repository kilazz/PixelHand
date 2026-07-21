// src/state/models.rs

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

// Import Slint-generated types and re-export them publicly to satisfy domain scanners
pub use crate::app::{
    AiModelType, AiSettings as SlintAiSettings, ExecutionProvider, ExtensionSettings, PathSettings,
    PreprocessingSettings, PreviewSettings, QcSettings, SearchMethod, TonemapSettings,
    UiLayoutSettings, ViewerConfig, VisualReportSettings,
};

// ==========================================
// --- SLINT ENUM EXTENSIONS ----------------
// ==========================================

impl AiModelType {
    /// Returns the standard subdirectory name for caching the model weights.
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

    /// Returns the standard vector embedding dimension of the model architecture.
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

// ==========================================
// --- SORTING TYPES ------------------------
// ==========================================

/// Strongly-typed sorting columns to replace magic strings across table headers.
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
// --- DISPATCHER SCAN PARAMETERS -----------
// ==========================================

#[derive(Clone, Debug)]
pub struct ScanPaths {
    pub dir_a: String,
    pub dir_b: String,
    pub excluded_folders: String,
    pub query_text: String,
}

#[derive(Clone, Debug)]
pub struct ScanQcRules {
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_normal_target: i32,
    pub qc_normals_tags: String,
    pub qc_match_by_stem: bool,
    pub qc_hide_same_resolution: bool,
    pub qc_check_bloat: bool,
    pub qc_check_alpha: bool,
    pub qc_check_colorspace: bool,
    pub qc_check_compression: bool,
}

#[derive(Clone, Debug)]
pub struct ScanVisualReports {
    pub save_visuals: bool,
    pub visuals_columns: usize,
    pub visuals_max_count: usize,
    pub visuals_font_size: usize,
    pub visuals_scale: f32,
}

#[derive(Clone, Debug)]
pub struct ScanPreprocessing {
    pub prep_luminance: bool,
    pub prep_channels: bool,
    pub prep_r: bool,
    pub prep_g: bool,
    pub prep_b: bool,
    pub prep_a: bool,
    pub prep_tags: String,
    pub prep_ignore_solid: bool,
}

#[derive(Clone, Debug)]
pub struct ScanAiSettings {
    pub search_precision: i32,
    pub ai_model: AiModelType,
    pub custom_model_path: String,
    pub custom_model_arch: i32,
    pub custom_model_dim: i32,
}

/// Consolidated configuration data structures mapping all UI options.
/// Passed down to background scanning threads to isolate execution from UI handles.
#[derive(Clone)]
pub struct ScanParams {
    pub paths: ScanPaths,
    pub qc: ScanQcRules,
    pub visuals: ScanVisualReports,
    pub prep: ScanPreprocessing,
    pub ai: ScanAiSettings,

    pub similarity: f32,
    pub batch_size: usize,
    pub search_method: SearchMethod,
    pub execution_provider: String,
    pub extensions: Vec<String>,
    pub cancel_token: Arc<AtomicBool>,

    #[allow(clippy::type_complexity)]
    pub on_progress: Option<Arc<dyn Fn(f32, usize, usize) + Send + Sync>>,
}

impl ScanParams {
    /// Compiles Slint UI properties from the global ScanConfig singleton into thread-safe ScanParams.
    pub fn from_store(scan_config: &crate::app::ScanConfig, cancel_token: Arc<AtomicBool>) -> Self {
        let paths = scan_config.get_paths();
        let ext = scan_config.get_extensions();
        let qc = scan_config.get_qc();
        let visuals = scan_config.get_visuals();
        let prep = scan_config.get_prep();
        let ai = scan_config.get_ai();

        let mut extensions = Vec::new();

        if ext.ext_png {
            extensions.push(".png".to_string());
        }
        if ext.ext_jpg {
            extensions.push(".jpg".to_string());
            extensions.push(".jpeg".to_string());
        }
        if ext.ext_tga {
            extensions.push(".tga".to_string());
        }
        if ext.ext_dds {
            extensions.push(".dds".to_string());
        }
        if ext.ext_bmp {
            extensions.push(".bmp".to_string());
        }
        if ext.ext_exr {
            extensions.push(".ext_exr".to_string());
            extensions.push(".exr".to_string());
        }
        if ext.ext_hdr {
            extensions.push(".hdr".to_string());
        }
        if ext.ext_tif {
            extensions.push(".tif".to_string());
            extensions.push(".tiff".to_string());
        }
        if ext.ext_webp {
            extensions.push(".webp".to_string());
        }
        if ext.ext_gif {
            extensions.push(".gif".to_string());
        }
        if ext.ext_psd {
            extensions.push(".psd".to_string());
        }
        if ext.ext_jxl {
            extensions.push(".jxl".to_string());
        }
        if ext.ext_heic {
            extensions.push(".heic".to_string());
            extensions.push(".heif".to_string());
        }
        if ext.ext_avif {
            extensions.push(".avif".to_string());
        }

        let execution_provider = match scan_config.get_execution_provider() {
            ExecutionProvider::DirectMl => "DirectML".to_string(),
            ExecutionProvider::Cuda => "CUDA".to_string(),
            ExecutionProvider::TensorRt => "TensorRT".to_string(),
            ExecutionProvider::CoreMl => "CoreML".to_string(),
            ExecutionProvider::Cpu => "CPU".to_string(),
        };

        Self {
            paths: ScanPaths {
                dir_a: paths.dir_a.to_string(),
                dir_b: paths.dir_b.to_string(),
                query_text: paths.query_text.to_string(),
                excluded_folders: paths.excluded_folders.to_string(),
            },
            qc: ScanQcRules {
                qc_mode: scan_config.get_search_method() == SearchMethod::Qc,
                qc_npot: qc.qc_npot,
                qc_mipmaps: qc.qc_mipmaps,
                qc_block_align: qc.qc_block_align,
                qc_bit_depth: qc.qc_bit_depth,
                qc_solid_colors: qc.qc_solid_colors,
                qc_normals: qc.qc_normals,
                qc_normal_target: qc.qc_normal_target,
                qc_normals_tags: qc.qc_normals_tags.to_string(),
                qc_match_by_stem: qc.qc_match_by_stem,
                qc_hide_same_resolution: qc.qc_hide_same_resolution,
                qc_check_bloat: qc.qc_check_bloat,
                qc_check_alpha: qc.qc_check_alpha,
                qc_check_colorspace: qc.qc_check_colorspace,
                qc_check_compression: qc.qc_check_compression,
            },
            visuals: ScanVisualReports {
                save_visuals: visuals.save_visuals,
                visuals_columns: visuals.visuals_columns as usize,
                visuals_max_count: visuals.visuals_max_count as usize,
                visuals_font_size: visuals.visuals_font_size as usize,
                visuals_scale: visuals.visuals_scale,
            },
            prep: ScanPreprocessing {
                prep_luminance: prep.prep_luminance,
                prep_channels: prep.prep_channels,
                prep_r: prep.prep_r,
                prep_g: prep.prep_g,
                prep_b: prep.prep_b,
                prep_a: prep.prep_a,
                prep_tags: prep.prep_tags.to_string(),
                prep_ignore_solid: prep.prep_ignore_solid,
            },
            ai: ScanAiSettings {
                search_precision: scan_config.get_search_precision(),
                ai_model: ai.ai_model,
                custom_model_path: ai.custom_model_path.to_string(),
                custom_model_arch: ai.custom_model_arch,
                custom_model_dim: ai.custom_model_dim,
            },
            similarity: scan_config.get_similarity_threshold(),
            batch_size: scan_config.get_batch_size() as usize,
            search_method: scan_config.get_search_method(),
            execution_provider,
            extensions,
            cancel_token,
            on_progress: None,
        }
    }
}

// ==========================================
// --- PERSISTENT APP SETTINGS --------------
// ==========================================

/// Persistent settings structure serialized to and loaded from `Settings.json`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSettings {
    pub paths: PathSettings,
    pub extensions: ExtensionSettings,
    pub ui: UiLayoutSettings,
    pub visuals: VisualReportSettings,
    pub prep: PreprocessingSettings,
    pub qc: QcSettings,
    pub ai: SlintAiSettings,
    pub tonemap: TonemapSettings,
    pub viewer: ViewerConfig,
    pub preview: PreviewSettings,

    pub similarity_threshold: f32,
    pub batch_size: i32,
    pub search_method: SearchMethod,
    pub execution_provider: ExecutionProvider,
    pub search_precision: i32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            paths: PathSettings {
                dir_a: "".into(),
                dir_b: "".into(),
                excluded_folders: ".git, .svn, cache, temp".into(),
                query_text: "".into(),
            },
            extensions: ExtensionSettings {
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
            ui: UiLayoutSettings {
                duplicates_panel_height: 180.0,
                sidebar_width: 380.0,
                compare_sidebar_width: 440.0,
                list_preview_size: 40.0,
                grid_card_size: 135.0,
            },
            visuals: VisualReportSettings {
                save_visuals: false,
                visuals_columns: 6,
                visuals_max_count: 100,
                visuals_font_size: 14,
                visuals_scale: 1.5,
            },
            prep: PreprocessingSettings {
                prep_luminance: false,
                prep_channels: false,
                prep_r: true,
                prep_g: true,
                prep_b: true,
                prep_a: true,
                prep_tags: "".into(),
                prep_ignore_solid: true,
            },
            qc: QcSettings {
                qc_mode: false,
                qc_npot: true,
                qc_mipmaps: true,
                qc_block_align: true,
                qc_bit_depth: true,
                qc_solid_colors: true,
                qc_normals: true,
                qc_normal_target: 0,
                qc_normals_tags: "".into(),
                qc_match_by_stem: true,
                qc_hide_same_resolution: false,
                qc_check_bloat: true,
                qc_check_alpha: true,
                qc_check_colorspace: true,
                qc_check_compression: true,
            },
            ai: SlintAiSettings {
                ai_model: AiModelType::ClipVitB32,
                custom_model_path: "".into(),
                custom_model_arch: 0,
                custom_model_dim: 512,
            },
            tonemap: TonemapSettings {
                tonemap_enabled: true,
                tonemap_auto_exposure: true,
                tonemap_operator: 2,
            },
            viewer: ViewerConfig {
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
            preview: PreviewSettings {
                enable_previews: true,
                preview_quality: 1,
                filter_only_npot: false,
                filter_only_uncompressed: false,
                filter_only_missing_mips: false,
                filter_only_cubemaps: false,
            },
            similarity_threshold: 90.0,
            batch_size: 128,
            search_method: SearchMethod::Exact,
            execution_provider: ExecutionProvider::Cpu,
            search_precision: 1,
        }
    }
}

// ==========================================
// --- DISPATCHER RESULT SUMMARIES ----------
// ==========================================

/// Represents a single duplicate image file within a duplicate cluster.
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
