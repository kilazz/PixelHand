// src/scanners/qc.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::core::qc::{
    check_absolute, check_normal_map_integrity, check_relative, check_solid_texture,
    extract_qc_metadata,
};
use crate::state::QcIssueSummary;
use crate::utils::helpers::discover_files;

pub use crate::core::qc::calculate_diff_map;

/// Executes an absolute local directory Quality Control audit.
pub async fn run_qc_scan_internal(params: super::ScanParams) -> Result<Vec<QcIssueSummary>> {
    let path = PathBuf::from(&params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let ex_folders: Vec<String> = params
        .excluded_folders
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let (paths, warnings) = discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();
    let total = paths.len();
    let processed = std::sync::atomic::AtomicUsize::new(0);

    let issues: Vec<QcIssueSummary> = paths
        .par_iter()
        .flat_map(|p| {
            if cancel_token.load(Ordering::Relaxed) {
                return Vec::new();
            }

            let mut file_issues = Vec::new();
            let (w, h) = match imagesize::size(p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);

            let qc_meta =
                extract_qc_metadata(p).unwrap_or_else(|_| crate::core::qc::QcImageMetadata {
                    width: w as u32,
                    height: h as u32,
                    file_size: size,
                    format_str: ext.clone(),
                    compression_format: ext,
                    color_space: "sRGB".into(),
                    has_alpha: false,
                    bit_depth: 8,
                    mipmap_count: 1,
                    is_cubemap: false,
                });

            // 1. Run absolute single-file rules (NPOT, block alignments, mipmaps, bit depth)
            let abs_issues = check_absolute(
                &qc_meta,
                params.qc_npot,
                params.qc_mipmaps,
                params.qc_block_align,
                params.qc_bit_depth,
            );
            for issue in abs_issues {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details: String::new(),
                });
            }

            // 2. Perform optional solid flat color checking
            if params.qc_solid_colors
                && let Some((issue, details)) = check_solid_texture(p)
            {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details,
                });
            }

            // 3. Perform optional normal maps vector integrity audits
            if params.qc_normals {
                let path_str = p.to_string_lossy().to_lowercase();
                let should_check = if params.qc_normals_tags.is_empty() {
                    true
                } else {
                    params
                        .qc_normals_tags
                        .split(',')
                        .map(|t| t.trim().to_lowercase())
                        .filter(|t| !t.is_empty())
                        .any(|t| path_str.contains(&t))
                };

                let normal_format = if qc_meta.compression_format.to_uppercase().contains("BC5") {
                    crate::core::qc::NormalMapFormat::Bc5RxGy
                } else {
                    crate::core::qc::NormalMapFormat::TangentSpaceRgb
                };

                if should_check
                    && let Some((issue, details)) =
                        check_normal_map_integrity(p, 0.15, normal_format)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }
            }

            let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref cb) = params.on_progress {
                cb(current as f32 / total as f32, current, total);
            }

            file_issues
        })
        .collect();

    if params.cancel_token.load(Ordering::Relaxed) {
        return Err(anyhow!("Scan cancelled by user."));
    }

    Ok(issues)
}

/// Compares contents and specifications of Folder A against Folder B.
#[allow(clippy::too_many_arguments)]
pub async fn run_folder_compare(
    directory_a: String,
    directory_b: String,
    extensions: Vec<String>,
    match_by_stem: bool,
    hide_same_resolution: bool,
    excluded_folders: Vec<String>,
    check_bloat: bool,
    check_alpha: bool,
    check_colorspace: bool,
    check_compression: bool,
) -> Result<Vec<QcIssueSummary>> {
    let path_a = PathBuf::from(directory_a);
    let path_b = PathBuf::from(directory_b);
    if !path_a.is_dir() || !path_b.is_dir() {
        return Err(anyhow!("Both paths must be valid directories"));
    }

    let (files_a, warnings_a) = discover_files(&path_a, &extensions, &excluded_folders);
    for warn in warnings_a {
        crate::app::append_to_console_log(&warn);
    }
    let (files_b, warnings_b) = discover_files(&path_b, &extensions, &excluded_folders);
    for warn in warnings_b {
        crate::app::append_to_console_log(&warn);
    }

    let get_key = |p: &Path| -> String {
        if match_by_stem {
            p.file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        } else {
            p.file_name()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        }
    };

    let mut map_a = std::collections::HashMap::new();
    for p in &files_a {
        map_a.insert(get_key(p), p.clone());
    }

    let mut issues = Vec::new();

    for p_b in &files_b {
        let key = get_key(p_b);
        if let Some(p_a) = map_a.get(&key) {
            let size_a = fs::metadata(p_a).map(|m| m.len()).unwrap_or(0);
            let size_b = fs::metadata(p_b).map(|m| m.len()).unwrap_or(0);

            let (w_a, h_a) = match imagesize::size(p_a) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let (w_b, h_b) = match imagesize::size(p_b) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };

            let ext_a = p_a
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let ext_b = p_b
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();

            let meta_a =
                extract_qc_metadata(p_a).unwrap_or_else(|_| crate::core::qc::QcImageMetadata {
                    width: w_a as u32,
                    height: h_a as u32,
                    file_size: size_a,
                    format_str: ext_a.clone(),
                    compression_format: ext_a,
                    color_space: "sRGB".into(),
                    has_alpha: false,
                    bit_depth: 8,
                    mipmap_count: 1,
                    is_cubemap: false,
                });
            let meta_b =
                extract_qc_metadata(p_b).unwrap_or_else(|_| crate::core::qc::QcImageMetadata {
                    width: w_b as u32,
                    height: h_b as u32,
                    file_size: size_b,
                    format_str: ext_b.clone(),
                    compression_format: ext_b,
                    color_space: "sRGB".into(),
                    has_alpha: false,
                    bit_depth: 8,
                    mipmap_count: 1,
                    is_cubemap: false,
                });

            let rel_issues = check_relative(
                &meta_a,
                &meta_b,
                check_bloat,
                check_alpha,
                check_colorspace,
                check_compression,
            );

            // Bypass if resolutions match and no metadata issues exist when hide_same_resolution is true
            if hide_same_resolution
                && meta_a.width == meta_b.width
                && meta_a.height == meta_b.height
                && rel_issues.is_empty()
            {
                continue;
            }

            for issue in rel_issues {
                issues.push(QcIssueSummary {
                    path: p_b.to_string_lossy().to_string(),
                    issue,
                    details: format!(
                        "Compared to Folder A version ({})",
                        p_a.file_name().unwrap_or_default().to_string_lossy()
                    ),
                });
            }
        }
    }
    Ok(issues)
}

/// Executes a flat-list technical Asset Inventory scan across all targeted directories.
pub async fn run_asset_audit(
    params: super::ScanParams,
) -> Result<Vec<crate::state::ResultsRowData>> {
    let path = PathBuf::from(&params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let ex_folders: Vec<String> = params
        .excluded_folders
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let (paths, warnings) = discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();
    let total = paths.len();
    let processed = std::sync::atomic::AtomicUsize::new(0);

    let is_pow2 = |n: u32| n != 0 && (n & (n - 1)) == 0;

    let rows: Vec<crate::state::ResultsRowData> = paths
        .par_iter()
        .filter_map(|p| {
            if cancel_token.load(Ordering::Relaxed) {
                return None;
            }

            let (w, h) = match imagesize::size(p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);

            let qc_meta =
                extract_qc_metadata(p).unwrap_or_else(|_| crate::core::qc::QcImageMetadata {
                    width: w as u32,
                    height: h as u32,
                    file_size: size,
                    format_str: ext.clone(),
                    compression_format: ext.clone(),
                    color_space: "sRGB".into(),
                    has_alpha: false,
                    bit_depth: 8,
                    mipmap_count: 1,
                    is_cubemap: false,
                });

            let thumbnail_data = super::load_thumbnail_for_path(&p.to_string_lossy());

            let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref cb) = params.on_progress {
                cb(current as f32 / total as f32, current, total);
            }

            Some(crate::state::ResultsRowData {
                is_header: false,
                is_qc: false,
                is_ai: false,
                group_index: -1,
                hash_or_issue: String::new(),
                path: p.to_string_lossy().to_string(),
                name: p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                score_or_detail: String::new(),

                format_str: qc_meta.compression_format.clone(),
                dimensions_str: format!("{} x {}", qc_meta.width, qc_meta.height),
                mipmaps_str: qc_meta.mipmap_count.to_string(),
                cubemap_str: if qc_meta.is_cubemap {
                    "YES".to_string()
                } else {
                    "NO".to_string()
                },
                size_str: crate::utils::helpers::format_size(qc_meta.file_size),

                meta_str: String::new(),
                is_best: false,
                is_checked: false,
                thumbnail_data,
                similarity: 0.0,

                size_bytes: qc_meta.file_size,
                pixels_count: (qc_meta.width * qc_meta.height) as u64,

                is_npot: !is_pow2(qc_meta.width) || !is_pow2(qc_meta.height),
                is_uncompressed: qc_meta
                    .compression_format
                    .to_lowercase()
                    .contains("uncompressed"),
                is_missing_mips: qc_meta.mipmap_count <= 1,
                is_cubemap_bool: qc_meta.is_cubemap,
            })
        })
        .collect();

    if params.cancel_token.load(Ordering::Relaxed) {
        return Err(anyhow!("Scan cancelled by user."));
    }

    Ok(rows)
}
