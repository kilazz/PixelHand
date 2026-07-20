// src/scanners/mod.rs

pub mod ai;
pub mod exact;
pub mod perceptual;
pub mod qc;

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use xxhash_rust::xxh64::Xxh64;

use crate::core::perceptual::AnalysisType;
use crate::state::models::{AiModelType, SearchMethod};
use crate::state::{DuplicateGroupSummary, ResultsRowData};

pub static ENABLE_PREVIEWS: AtomicBool = AtomicBool::new(true);
pub static PREVIEW_QUALITY: AtomicI32 = AtomicI32::new(1); // 0: Fast, 1: Balanced, 2: High

// ==========================================
// --- DECOMPOSED SCANPARMS SUB-STRUCTS -----
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

// ==========================================
// --- COMPOSITE SCANPARAMS -----------------
// ==========================================

/// Data structure mapping all options gathered from the UI panel to orchestrate a scan.
#[derive(Clone)]
pub struct ScanParams {
    pub paths: ScanPaths,
    pub qc: ScanQcRules,
    pub visuals: ScanVisualReports,
    pub prep: ScanPreprocessing,
    pub ai: ScanAiSettings,

    // Core execution controls
    pub similarity: f32,
    pub batch_size: usize,
    pub search_method: SearchMethod,
    pub execution_provider: String,
    pub extensions: Vec<String>,
    pub cancel_token: Arc<std::sync::atomic::AtomicBool>,

    #[allow(clippy::type_complexity)]
    pub on_progress: Option<Arc<dyn Fn(f32, usize, usize) + Send + Sync>>,
}

impl ScanParams {
    /// Compiles Slint UI properties from the global Store singleton into thread-safe ScanParams.
    pub fn from_store(
        store: &crate::app::Store,
        cancel_token: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        // Fetch nested settings from Slint
        let paths = store.get_paths();
        let ext = store.get_extensions();
        let qc = store.get_qc();
        let visuals = store.get_visuals();
        let prep = store.get_prep();
        let ai = store.get_ai();

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

        let execution_provider = match store.get_execution_provider() {
            1 => "DirectML".to_string(),
            2 => "CUDA".to_string(),
            3 => "TensorRT".to_string(),
            4 => "CoreML".to_string(),
            _ => "CPU".to_string(),
        };

        Self {
            paths: ScanPaths {
                dir_a: paths.dir_a.to_string(),
                dir_b: paths.dir_b.to_string(),
                query_text: paths.query_text.to_string(),
                excluded_folders: paths.excluded_folders.to_string(),
            },
            qc: ScanQcRules {
                qc_mode: store.get_search_method() == 4,
                qc_npot: qc.qc_npot,
                qc_mipmaps: qc.qc_mipmaps,
                qc_block_align: qc.qc_block_align,
                qc_bit_depth: qc.qc_bit_depth,
                qc_solid_colors: qc.qc_solid_colors,
                qc_normals: qc.qc_normals,
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
                search_precision: store.get_search_precision(),
                ai_model: AiModelType::from_i32(ai.ai_model),
                custom_model_path: ai.custom_model_path.to_string(),
                custom_model_arch: ai.custom_model_arch,
                custom_model_dim: ai.custom_model_dim,
            },

            similarity: store.get_similarity_threshold(),
            batch_size: store.get_batch_size() as usize,
            search_method: SearchMethod::from_i32(store.get_search_method()),
            execution_provider,
            extensions,
            cancel_token,
            on_progress: None,
        }
    }
}

/// Represents a discrete analysis item mapped by the image channels splitter.
#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub path: PathBuf,
    pub analysis_type: AnalysisType,
}

/// Expands a flat list of file paths into distinct AnalysisItems based on pre-processing rules.
pub fn generate_analysis_items(paths: &[PathBuf], params: &ScanParams) -> Vec<AnalysisItem> {
    let mut items = Vec::new();

    let tags: Vec<String> = params
        .prep
        .prep_tags
        .split(',')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    for path in paths {
        let path_str = path.to_string_lossy().to_lowercase();

        let matches_tags = if tags.is_empty() {
            true
        } else {
            tags.iter().any(|tag| path_str.contains(tag))
        };

        if params.prep.prep_channels && matches_tags {
            if params.prep.prep_r {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::R,
                });
            }
            if params.prep.prep_g {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::G,
                });
            }
            if params.prep.prep_b {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::B,
                });
            }
            if params.prep.prep_a {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::A,
                });
            }
        } else if params.prep.prep_luminance {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Luminance,
            });
        } else {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Composite,
            });
        }
    }

    items
}

/// Central routing mechanism directing search configurations to proper execution engines.
pub async fn execute_scan(
    params: ScanParams,
) -> Result<(Vec<DuplicateGroupSummary>, Vec<ResultsRowData>)> {
    // Strongly-typed routing over the SearchMethod enum
    match params.search_method {
        SearchMethod::Qc => {
            if !params.paths.dir_b.trim().is_empty() {
                let ex_folders: Vec<String> = params
                    .paths
                    .excluded_folders
                    .split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect();

                let issues = qc::run_folder_compare(
                    params.paths.dir_a.clone(),
                    params.paths.dir_b.clone(),
                    params.extensions.clone(),
                    params.qc.qc_match_by_stem,
                    params.qc.qc_hide_same_resolution,
                    ex_folders,
                    params.qc.qc_check_bloat,
                    params.qc.qc_check_alpha,
                    params.qc.qc_check_colorspace,
                    params.qc.qc_check_compression,
                )
                .await?;
                let rows = map_qc_to_rows(&issues);
                Ok((Vec::new(), rows))
            } else {
                let issues = qc::run_qc_scan_internal(params).await?;
                let rows = map_qc_to_rows(&issues);
                Ok((Vec::new(), rows))
            }
        }
        SearchMethod::Inventory => {
            let mut rows = qc::run_asset_audit(params).await?;
            rows.sort_by(|a, b| a.path.cmp(&b.path));
            Ok((Vec::new(), rows))
        }
        SearchMethod::Ai => {
            if !params.paths.query_text.trim().is_empty() {
                let matches = ai::run_ai_search(params).await?;
                let rows = map_ai_search_to_rows(&matches);
                Ok((Vec::new(), rows))
            } else {
                let groups = ai::run_ai_duplicate_scan(params).await?;
                let rows = map_groups_to_rows(&groups);
                Ok((groups, rows))
            }
        }
        SearchMethod::Perceptual => {
            let groups = perceptual::run_perceptual_scan_internal(params).await?;
            let rows = map_groups_to_rows(&groups);
            Ok((groups, rows))
        }
        SearchMethod::Exact => {
            let groups = exact::run_exact_scan(params).await?;
            let rows = map_groups_to_rows(&groups);
            Ok((groups, rows))
        }
    }
}

// ---------------------------------------------------------
// PRIVATE RESULT-MAPPING DATA HELPERS
// ---------------------------------------------------------

static CACHE_DIR_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Loads thumbnail textures from disk cache, generating compressed fallback files during misses.
pub(crate) fn load_thumbnail_for_path(path_str: &str) -> Option<image::RgbaImage> {
    if !ENABLE_PREVIEWS.load(Ordering::Relaxed) {
        return None;
    }

    let path = PathBuf::from(path_str);
    if !path.is_file() {
        return None;
    }

    let cache_dir = crate::utils::settings::get_portable_app_data_dir()
        .ok()
        .map(|p| p.join(".cache").join("thumbnails"));

    let cache_path = if let Some(ref dir) = cache_dir {
        CACHE_DIR_INITIALIZED.get_or_init(|| {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::error!("Failed to initialize thumbnail cache directory: {}", e);
            }
        });

        if let Ok(metadata) = std::fs::metadata(&path) {
            let size = metadata.len();
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);

            let mut hasher = Xxh64::new(0);
            hasher.update(path.to_string_lossy().as_bytes());
            hasher.update(&size.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());

            Some(dir.join(format!("{:016x}.png", hasher.digest())))
        } else {
            None
        }
    } else {
        None
    };

    // Try reading the tiny pre-rendered PNG from the disk cache safely
    if let Some(ref cp) = cache_path
        && cp.is_file()
        && let Ok(img) = image::open(cp)
    {
        let rgba = img.to_rgba8();
        crate::utils::cache::store_in_thumbnail_memory_cache(path_str, rgba.clone());
        return Some(rgba);
    }

    let quality = PREVIEW_QUALITY.load(Ordering::Relaxed);
    let target_size = match quality {
        0 => 64,
        1 => 128,
        _ => 256,
    };
    let filter = match quality {
        0 => image::imageops::FilterType::Nearest,
        1 => image::imageops::FilterType::Triangle,
        _ => image::imageops::FilterType::Lanczos3,
    };

    // Fallback: Parse the actual texture and downscale
    if let Ok(mut img) =
        crate::format_loaders::open_image_with_dds_fallback(&path, Some(target_size), None)
    {
        if img.width() > target_size || img.height() > target_size {
            img = img.resize(target_size, target_size, filter);
        }
        let rgba = img.to_rgba8();

        crate::utils::cache::store_in_thumbnail_memory_cache(path_str, rgba.clone());

        if let Some(ref cp) = cache_path {
            let _ = rgba.save(cp);
        }
        return Some(rgba);
    }
    None
}

/// Converts core rust clustering models into the Slint-friendly row representation.
pub fn map_groups_to_rows(groups: &[DuplicateGroupSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    let is_pow2 = |n: usize| n != 0 && (n & (n - 1)) == 0;

    for (g_idx, group) in groups.iter().enumerate() {
        if group.files.is_empty() {
            continue;
        }

        // Add Group Cluster Header row
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: false,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: group.hash.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),

            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),

            size_str: crate::utils::helpers::format_size(
                group.files.first().map(|f| f.size).unwrap_or(0),
            ),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
            similarity: 100.0,

            size_bytes: 0,
            pixels_count: 0,

            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        });

        // Add Child file rows belonging to this cluster
        for (f_idx, file) in group.files.iter().enumerate() {
            let is_best = f_idx == 0;
            let thumbnail_data = load_thumbnail_for_path(&file.path);

            rows.push(ResultsRowData {
                is_header: false,
                is_qc: false,
                is_ai: false,
                group_index: g_idx as i32,
                hash_or_issue: String::new(),
                path: file.path.clone(),
                name: Path::new(&file.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                score_or_detail: if is_best {
                    "[Best]".to_string()
                } else {
                    format!("{:.1}%", file.similarity)
                },

                format_str: file.compression_format.clone(),
                dimensions_str: format!("{} x {}", file.width, file.height),
                mipmaps_str: file.mipmap_count.to_string(),
                cubemap_str: if file.is_cubemap {
                    "YES".to_string()
                } else {
                    "NO".to_string()
                },

                size_str: crate::utils::helpers::format_size(file.size),
                meta_str: {
                    let size_mb = file.size as f64 / 1024.0 / 1024.0;
                    let alpha_str = if file.has_alpha { "RGBA" } else { "RGB" };
                    format!(
                        "{}x{} • {:.2} MB • {} • {} • {}-bit • {} • Mips: {}",
                        file.width,
                        file.height,
                        size_mb,
                        file.format_str.to_uppercase(),
                        file.compression_format,
                        file.bit_depth,
                        alpha_str,
                        file.mipmap_count
                    )
                },
                is_best,
                is_checked: false,
                thumbnail_data,
                similarity: file.similarity,

                size_bytes: file.size,
                pixels_count: (file.width * file.height) as u64,

                is_npot: !is_pow2(file.width) || !is_pow2(file.height),
                is_uncompressed: file
                    .compression_format
                    .to_lowercase()
                    .contains("uncompressed"),
                is_missing_mips: file.mipmap_count <= 1,
                is_cubemap_bool: file.is_cubemap,
            });
        }
    }
    rows
}

/// Converts core technical Quality Control issues into Slint row models grouped by issue category.
pub fn map_qc_to_rows(issues: &[crate::state::QcIssueSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    if issues.is_empty() {
        return rows;
    }

    let mut grouped: std::collections::HashMap<String, Vec<&crate::state::QcIssueSummary>> =
        std::collections::HashMap::new();
    for issue in issues {
        grouped.entry(issue.issue.clone()).or_default().push(issue);
    }

    let mut sorted_issue_types: Vec<String> = grouped.keys().cloned().collect();
    sorted_issue_types.sort();

    for (g_idx, issue_type) in sorted_issue_types.iter().enumerate() {
        let group_issues = &grouped[issue_type];

        // Add Group Header Row for this specific issue type
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: true,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: issue_type.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),

            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),

            size_str: format!("{} files", group_issues.len()),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
            similarity: 0.0,

            size_bytes: 0,
            pixels_count: 0,

            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        });

        // Add Child Rows belonging to this issue category
        for issue in group_issues {
            let thumbnail_data = load_thumbnail_for_path(&issue.path);
            rows.push(ResultsRowData {
                is_header: false,
                is_qc: true,
                is_ai: false,
                group_index: g_idx as i32,
                hash_or_issue: issue.issue.clone(),
                path: issue.path.clone(),
                name: Path::new(&issue.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                score_or_detail: issue.issue.clone(),

                format_str: String::new(),
                dimensions_str: String::new(),
                mipmaps_str: String::new(),
                cubemap_str: String::new(),

                size_str: String::new(),
                meta_str: issue.details.clone(),
                is_best: false,
                is_checked: false,
                thumbnail_data,
                similarity: 0.0,

                size_bytes: 0,
                pixels_count: 0,

                is_npot: false,
                is_uncompressed: false,
                is_missing_mips: false,
                is_cubemap_bool: false,
            });
        }
    }

    rows
}

/// Converts semantic vector database matches into Slint row models.
pub fn map_ai_search_to_rows(
    matches: &[crate::state::AiSearchResultSummary],
) -> Vec<ResultsRowData> {
    matches
        .iter()
        .map(|res| ResultsRowData {
            is_header: false,
            is_qc: false,
            is_ai: true,
            group_index: -1,
            hash_or_issue: String::new(),
            path: res.path.clone(),
            name: Path::new(&res.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            score_or_detail: format!("{:.1}%", res.similarity),
            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),
            size_str: String::new(),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: load_thumbnail_for_path(&res.path),
            similarity: res.similarity,
            size_bytes: 0,
            pixels_count: 0,
            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        })
        .collect()
}

/// A unified pipeline for all scanners. Handles path validation, file discovery, filtering, and cancel token management.
pub fn run_scan_pipeline<F, R>(params: &ScanParams, process_files: F) -> Result<R>
where
    F: FnOnce(
        Vec<PathBuf>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        std::sync::Arc<dyn Fn(f32, usize, usize) + Send + Sync>,
    ) -> Result<R>,
{
    let path = PathBuf::from(&params.paths.dir_a);
    if !path.is_dir() {
        return Err(anyhow::anyhow!(
            "The specified path is not a valid directory"
        ));
    }

    let ex_folders: Vec<String> = params
        .paths
        .excluded_folders
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let (paths, warnings) =
        crate::utils::helpers::discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();

    // Fallback to a dummy callback if no progress reporter is attached
    let dummy_progress: Arc<dyn Fn(f32, usize, usize) + Send + Sync> = Arc::new(|_, _, _| {});
    let progress_cb = params.on_progress.clone().unwrap_or(dummy_progress);

    process_files(paths, cancel_token, progress_cb)
}
