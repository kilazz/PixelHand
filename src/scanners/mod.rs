// src/scanners/mod.rs

pub mod ai;
pub mod exact;
pub mod perceptual;
pub mod qc;

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::core::perceptual::AnalysisType;
use crate::state::{DuplicateGroupSummary, ResultsRowData};

/// All configuration parameters gathered from the UI panel to orchestrate a scan.
/// Derived `Clone` is required to allow app.rs to query the configuration parameters
/// after execute_scan consumes ownership of the original struct.
#[derive(Clone)]
pub struct ScanParams {
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub similarity: f32,
    pub batch_size: usize,
    pub search_method: i32, // 0: Exact (xxHash), 1: Perceptual (dHash), 2: AI Embeddings
    pub execution_provider: String,
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_normals_tags: String,
    pub extensions: Vec<String>,
    pub cancel_token: Arc<AtomicBool>,

    // Properties for Asynchronous Visual Reports (Contact Sheets)
    pub save_visuals: bool,
    pub visuals_columns: usize,
    pub visuals_max_count: usize,
    pub visuals_font_size: usize,
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

    // Exclude Filter and QC Matching controls
    pub excluded_folders: String,
    pub qc_match_by_stem: bool,
    pub qc_hide_same_resolution: bool,

    // IVF-PQ Index Tuning Parameters (Search Precision)
    pub search_precision: i32,
}

impl ScanParams {
    /// Compiles the current state of Slint UI properties into a safe, thread-safe ScanParams struct.
    pub fn from_ui(ui: &crate::app::AppWindow, cancel_token: Arc<AtomicBool>) -> Self {
        let mut extensions = Vec::new();
        if ui.get_ext_png() {
            extensions.push(".png".to_string());
            extensions.push(".jpg".to_string());
            extensions.push(".jpeg".to_string());
        }
        if ui.get_ext_tga() {
            extensions.push(".tga".to_string());
        }
        if ui.get_ext_dds() {
            extensions.push(".dds".to_string());
        }
        if ui.get_ext_bmp() {
            extensions.push(".bmp".to_string());
        }
        if ui.get_ext_exr() {
            extensions.push(".exr".to_string());
        }
        if ui.get_ext_hdr() {
            extensions.push(".hdr".to_string());
        }
        if ui.get_ext_tif() {
            extensions.push(".tif".to_string());
            extensions.push(".tiff".to_string());
        }
        if ui.get_ext_webp() {
            extensions.push(".webp".to_string());
        }

        // Decipher selected Execution Provider index
        let execution_provider = match ui.get_execution_provider() {
            1 => "DirectML".to_string(),
            2 => "CUDA".to_string(),
            3 => "TensorRT".to_string(),
            4 => "CoreML".to_string(),
            _ => "CPU".to_string(),
        };

        Self {
            dir_a: ui.get_dir_a().to_string(),
            dir_b: ui.get_dir_b().to_string(),
            query_text: ui.get_query_text().to_string(),
            similarity: ui.get_similarity_threshold(),
            batch_size: ui.get_batch_size() as usize,
            search_method: ui.get_search_method(),
            execution_provider,

            qc_mode: ui.get_qc_mode(),
            qc_npot: ui.get_qc_npot(),
            qc_mipmaps: ui.get_qc_mipmaps(),
            qc_block_align: ui.get_qc_block_align(),
            qc_bit_depth: ui.get_qc_bit_depth(),
            qc_solid_colors: ui.get_qc_solid_colors(),
            qc_normals: ui.get_qc_normals(),
            qc_normals_tags: ui.get_qc_normals_tags().to_string(),

            extensions,
            cancel_token,

            // Pull visuals settings from Slint UI properties
            save_visuals: ui.get_save_visuals(),
            visuals_columns: ui.get_visuals_columns() as usize,
            visuals_max_count: ui.get_visuals_max_count() as usize,
            visuals_font_size: ui.get_visuals_font_size() as usize,
            visuals_scale: ui.get_visuals_scale(),

            // Pull image pre-processing logic parameters
            prep_luminance: ui.get_prep_luminance(),
            prep_channels: ui.get_prep_channels(),
            prep_r: ui.get_prep_r(),
            prep_g: ui.get_prep_g(),
            prep_b: ui.get_prep_b(),
            prep_a: ui.get_prep_a(),
            prep_tags: ui.get_prep_tags().to_string(),
            prep_ignore_solid: ui.get_prep_ignore_solid(),

            // Exclude Filter and QC Matching controls
            excluded_folders: ui.get_excluded_folders().to_string(),
            qc_match_by_stem: ui.get_qc_match_by_stem(),
            qc_hide_same_resolution: ui.get_qc_hide_same_resolution(),

            // Precision parameters
            search_precision: ui.get_search_precision(),
        }
    }
}

/// Represents a discrete analysis task generated by the pre-processing module.
/// Can point to the same physical file multiple times if channel splitting is enabled.
#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub path: PathBuf,
    pub analysis_type: AnalysisType,
}

/// Expands a list of raw physical paths into a comprehensive list of logical `AnalysisItem`s
/// based on the configured pre-processing rules (Channels, Luminance, Tags).
pub fn generate_analysis_items(paths: &[PathBuf], params: &ScanParams) -> Vec<AnalysisItem> {
    let mut items = Vec::new();

    // Parse tag filter into lowercase tokens for substring matching
    let tags: Vec<String> = params
        .prep_tags
        .split(',')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    for path in paths {
        let path_str = path.to_string_lossy().to_lowercase();

        let matches_tags = if tags.is_empty() {
            true // No tags specified means filter applies to all files
        } else {
            tags.iter().any(|tag| path_str.contains(tag))
        };

        if params.prep_channels && matches_tags {
            // Explode single file into up to 4 separate items based on channel checks
            if params.prep_r {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::R,
                });
            }
            if params.prep_g {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::G,
                });
            }
            if params.prep_b {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::B,
                });
            }
            if params.prep_a {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::A,
                });
            }
        } else if params.prep_luminance {
            // Process as a unified grayscale layout
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Luminance,
            });
        } else {
            // Default fully-composited RGB/RGBA pass
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Composite,
            });
        }
    }

    items
}

/// Core routing mechanism designed to dispatch to the correct scanning strategy.
pub async fn execute_scan(
    params: ScanParams,
) -> Result<(Vec<DuplicateGroupSummary>, Vec<ResultsRowData>)> {
    if params.qc_mode {
        if !params.dir_b.trim().is_empty() {
            // Relative Folder A vs Folder B asset comparison
            let ex_folders: Vec<String> = params
                .excluded_folders
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();

            let issues = qc::run_folder_compare(
                params.dir_a.clone(),
                params.dir_b.clone(),
                params.extensions.clone(),
                params.qc_match_by_stem,
                params.qc_hide_same_resolution,
                ex_folders,
            )
            .await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        } else {
            // Absolute local folder technical quality control audit
            let issues = qc::run_qc_scan_internal(params).await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        }
    } else if params.search_method == 2 {
        // AI Vector Space Similarity Searches
        if !params.query_text.trim().is_empty() {
            // Semantic Text Search
            let matches = ai::run_ai_search(params).await?;
            let rows = map_ai_search_to_rows(&matches);
            Ok((Vec::new(), rows))
        } else {
            // AI Visual Duplicate Cluster Scan
            let groups = ai::run_ai_duplicate_scan(params).await?;
            let rows = map_groups_to_rows(&groups);
            Ok((groups, rows))
        }
    } else if params.search_method == 1 {
        // Linear Perceptual Similarity Engine (dHash + Hamming Distance)
        let groups = perceptual::run_perceptual_scan_internal(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    } else {
        // Binary Byte-Exact Scanning (xxHash64)
        let groups = exact::run_exact_scan(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    }
}

// ---------------------------------------------------------
// PRIVATE RESULT-MAPPING DATA HELPERS
// ---------------------------------------------------------

/// Instantiates a downscaled thumbnail from disk on a background task.
/// Utilizes smart DDS mipmap decimation at level 64px to minimize peak RAM usages.
fn load_thumbnail_for_path(path_str: &str) -> Option<image::RgbaImage> {
    let path = PathBuf::from(path_str);
    // --- OPTIMIZED: Request tiny 64px mipmap directly during thumbnail load step ---
    if let Ok(mut img) =
        crate::format_loaders::dds_loader::open_image_with_dds_fallback(&path, Some(64))
    {
        if img.width() > 64 || img.height() > 64 {
            img = img.thumbnail(64, 64);
        }
        return Some(img.to_rgba8());
    }
    None
}

/// Converts core rust clustering models into the Slint-friendly row representation.
fn map_groups_to_rows(groups: &[DuplicateGroupSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for (g_idx, group) in groups.iter().enumerate() {
        if group.files.is_empty() {
            continue;
        }

        // 1. Add Group Cluster Header row
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: false,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: group.hash.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),
            size_str: crate::utils::helpers::format_size(
                group.files.first().map(|f| f.size).unwrap_or(0),
            ),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
        });

        // 2. Add Child file rows belonging to this cluster
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
                size_str: crate::utils::helpers::format_size(file.size),
                meta_str: format!(
                    "{}x{} • {}",
                    file.width, file.height, file.compression_format
                ),
                is_best,
                is_checked: false,
                thumbnail_data,
            });
        }
    }
    rows
}

/// Converts core technical Quality Control issues into Slint row models.
fn map_qc_to_rows(issues: &[crate::state::QcIssueSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for issue in issues {
        let thumbnail_data = load_thumbnail_for_path(&issue.path);
        rows.push(ResultsRowData {
            is_header: false,
            is_qc: true,
            is_ai: false,
            group_index: -1,
            hash_or_issue: issue.issue.clone(),
            path: issue.path.clone(),
            name: Path::new(&issue.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            score_or_detail: issue.issue.clone(),
            size_str: String::new(),
            meta_str: issue.details.clone(),
            is_best: false,
            is_checked: false,
            thumbnail_data,
        });
    }
    rows
}

/// Converts semantic vector database matches into Slint row models.
fn map_ai_search_to_rows(matches: &[crate::state::AiSearchResultSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for res in matches {
        let thumbnail_data = load_thumbnail_for_path(&res.path);
        rows.push(ResultsRowData {
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
            size_str: String::new(),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data,
        });
    }
    rows
}
