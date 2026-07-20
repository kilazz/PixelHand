// src/scanners/qc.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::core::qc::{
    QcImageMetadata, check_absolute, check_empty_channels, check_normal_map_integrity,
    check_normal_map_orientation, check_relative, check_solid_texture,
};
use crate::state::QcIssueSummary;
use crate::utils::helpers::discover_files;

/// Executes an absolute local directory Quality Control audit.
pub async fn run_qc_scan_internal(params: super::ScanParams) -> Result<Vec<QcIssueSummary>> {
    // Delegate file discovery, exclusion folders parsing, and progress bootstrapping to the pipeline
    super::run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let total = paths.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        let issues: Vec<QcIssueSummary> = paths
            .par_iter()
            .flat_map(|p| {
                if cancel_token.load(Ordering::Relaxed) {
                    return Vec::new();
                }

                let mut file_issues = Vec::new();

                // DRY implementation: Safely extract metadata using the robust fallback factory
                let qc_meta = QcImageMetadata::extract_or_fallback(p);

                // Run absolute single-file rules (NPOT, block alignments, mipmaps, bit depth)
                let abs_issues = check_absolute(
                    &qc_meta,
                    params.qc.qc_npot,
                    params.qc.qc_mipmaps,
                    params.qc.qc_block_align,
                    params.qc.qc_bit_depth,
                );
                for issue in abs_issues {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details: String::new(),
                    });
                }

                // Perform optional solid flat color checking and empty channel packing checks
                if params.qc.qc_solid_colors {
                    if let Some((issue, details)) = check_solid_texture(p) {
                        file_issues.push(QcIssueSummary {
                            path: p.to_string_lossy().to_string(),
                            issue,
                            details,
                        });
                    }
                    if let Some((issue, details)) = check_empty_channels(p) {
                        file_issues.push(QcIssueSummary {
                            path: p.to_string_lossy().to_string(),
                            issue,
                            details,
                        });
                    }
                }

                // Perform optional normal maps vector integrity and DirectX vs OpenGL Y-axis audits
                if params.qc.qc_normals {
                    let path_str = p.to_string_lossy().to_lowercase();
                    let should_check = if params.qc.qc_normals_tags.is_empty() {
                        true
                    } else {
                        params
                            .qc
                            .qc_normals_tags
                            .split(',')
                            .map(|t| t.trim().to_lowercase())
                            .filter(|t| !t.is_empty())
                            .any(|t| path_str.contains(&t))
                    };

                    let normal_format = if qc_meta.compression_format.to_uppercase().contains("BC5")
                    {
                        crate::core::qc::NormalMapFormat::Bc5RxGy
                    } else {
                        crate::core::qc::NormalMapFormat::TangentSpaceRgb
                    };

                    if should_check {
                        if let Some((issue, details)) =
                            check_normal_map_integrity(p, 0.15, normal_format)
                        {
                            file_issues.push(QcIssueSummary {
                                path: p.to_string_lossy().to_string(),
                                issue,
                                details,
                            });
                        }
                        if let Some((issue, details)) = check_normal_map_orientation(p) {
                            file_issues.push(QcIssueSummary {
                                path: p.to_string_lossy().to_string(),
                                issue,
                                details,
                            });
                        }
                    }
                }

                let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_cb(current as f32 / total as f32, current, total);

                file_issues
            })
            .collect();

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        Ok(issues)
    })
}

/// Compares contents and specifications of Folder A against Folder B.
/// Note: Since this is a comparative scan between two distinct paths, it operates
/// outside of the single-directory ScanParams.dir_a wrapper and maintains custom walking.
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
            // DRY implementation: Safely extract metadata using the robust fallback factory
            let meta_a = QcImageMetadata::extract_or_fallback(p_a);
            let meta_b = QcImageMetadata::extract_or_fallback(p_b);

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
    // Delegate preparatory routines to the scan pipeline
    super::run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let total = paths.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        let is_pow2 = |n: u32| n != 0 && (n & (n - 1)) == 0;

        let rows: Vec<crate::state::ResultsRowData> = paths
            .par_iter()
            .filter_map(|p| {
                if cancel_token.load(Ordering::Relaxed) {
                    return None;
                }

                // DRY implementation: Safely extract metadata using the robust fallback factory
                let qc_meta = QcImageMetadata::extract_or_fallback(p);

                let thumbnail_data = super::load_thumbnail_for_path(&p.to_string_lossy());

                let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_cb(current as f32 / total as f32, current, total);

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

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        Ok(rows)
    })
}
