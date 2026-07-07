// src/scanners/mod.rs
pub mod ai;
pub mod exact;
pub mod perceptual;
pub mod qc;

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::state::{DuplicateGroupSummary, ResultsRowData};

/// All configuration parameters compiled from the UI panel
pub struct ScanParams {
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub similarity: f32,
    pub batch_size: usize,
    pub search_method: i32, // 0: Exact, 1: Perceptual, 2: AI
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
    pub perceptual_channel: String,
    pub cancel_token: Arc<AtomicBool>,
}

impl ScanParams {
    /// Compiles current state of Slint checkboxes and input fields into ScanParams
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

        // Read the currently active channel filter on the board
        let perceptual_channel = if ui.get_active_r() {
            "R".to_string()
        } else if ui.get_active_g() {
            "G".to_string()
        } else if ui.get_active_b() {
            "B".to_string()
        } else if ui.get_active_a() {
            "A".to_string()
        } else {
            "Composite".to_string()
        };

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
            perceptual_channel,
            cancel_token,
        }
    }
}

/// Main orchestration routing logic
pub async fn execute_scan(
    params: ScanParams,
) -> Result<(Vec<DuplicateGroupSummary>, Vec<ResultsRowData>)> {
    if params.qc_mode {
        if !params.dir_b.trim().is_empty() {
            // Relative Folder B vs A comparison
            let issues =
                qc::run_folder_compare(params.dir_a, params.dir_b, params.extensions).await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        } else {
            // Absolute local folder technical QC auditing
            let issues = qc::run_qc_scan_internal(params).await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        }
    } else if params.search_method == 2 {
        // AI semantic/visual search
        if !params.query_text.trim().is_empty() {
            let matches = ai::run_ai_search(params).await?;
            let rows = map_ai_search_to_rows(&matches);
            Ok((Vec::new(), rows))
        } else {
            let groups = ai::run_ai_duplicate_scan(params).await?;
            let rows = map_groups_to_rows(&groups);
            Ok((groups, rows))
        }
    } else if params.search_method == 1 {
        // Simple perceptual scanner (dHash)
        let groups = perceptual::run_perceptual_scan_internal(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    } else {
        // Exact Byte-match (xxHash64)
        let groups = exact::run_exact_scan(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    }
}

// ---------------------------------------------------------
// PRIVATE RESULT-MAPPING DATA HELPERS
// ---------------------------------------------------------

fn load_thumbnail_for_path(path_str: &str) -> Option<image::RgbaImage> {
    let path = PathBuf::from(path_str);
    if let Ok(mut img) = crate::format_loaders::dds_loader::open_image_with_dds_fallback(&path) {
        if img.width() > 64 || img.height() > 64 {
            img = img.thumbnail(64, 64);
        }
        return Some(img.to_rgba8());
    }
    None
}

fn map_groups_to_rows(groups: &[DuplicateGroupSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for (g_idx, group) in groups.iter().enumerate() {
        if group.files.is_empty() {
            continue;
        }

        // Add Header row for duplicate cluster
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

        // Add file rows inside the duplicate cluster
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
