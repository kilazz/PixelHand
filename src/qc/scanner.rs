// src/qc/scanner.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::qc::rules::{
    QcImageMetadata, TargetNormalStyle, check_absolute, check_alpha_bleed,
    check_color_space_misconfiguration, check_empty_channels, check_normal_map_integrity,
    check_normal_map_orientation, check_relative, check_seamless_tiling, check_solid_texture,
    check_texel_density_oversize,
};
use crate::state::models::{DuplicateFileSummary, QcIssueSummary, ScanParams};
use crate::utils::helpers::{discover_files, run_scan_pipeline};

/// Executes a technical Quality Control audit over a targeted directory path.
pub async fn run_qc_scan_internal(params: ScanParams) -> Result<Vec<QcIssueSummary>> {
    run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let total = paths.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        let issues: Vec<QcIssueSummary> = paths
            .par_iter()
            .flat_map(|p| {
                if cancel_token.load(Ordering::Relaxed) {
                    return Vec::new();
                }

                let mut file_issues = Vec::new();
                let qc_meta = QcImageMetadata::extract_or_fallback(p);

                // Standalone rules (NPOT, block alignment, mipmaps, bit depth)
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

                // Solid color & empty channels check
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

                // Alpha Bleed (Edge Padding) check
                if params.qc.qc_alpha_bleed
                    && let Some((issue, details)) = check_alpha_bleed(p)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }

                // Color Space Misconfiguration check
                if params.qc.qc_check_colorspace {
                    let custom_tags = if !params.qc.qc_normals_tags.is_empty() {
                        &params.qc.qc_normals_tags
                    } else {
                        &params.prep.prep_tags
                    };

                    if let Some((issue, details)) =
                        check_color_space_misconfiguration(p, &qc_meta, custom_tags)
                    {
                        file_issues.push(QcIssueSummary {
                            path: p.to_string_lossy().to_string(),
                            issue,
                            details,
                        });
                    }
                }

                // Oversized texture check with user-defined limit
                if params.qc.qc_check_oversize
                    && let Some((issue, details)) =
                        check_texel_density_oversize(&qc_meta, params.qc.qc_max_resolution)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }

                // Seamless tiling check
                if params.qc.qc_seamless_tiling
                    && let Some((issue, details)) = check_seamless_tiling(p)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }

                // Normal map vector integrity & DirectX/OpenGL orientation check
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
                        crate::qc::rules::NormalMapFormat::Bc5RxGy
                    } else {
                        crate::qc::rules::NormalMapFormat::TangentSpaceRgb
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

                        let target_style = match params.qc.qc_normal_target {
                            1 => TargetNormalStyle::OpenGL,
                            2 => TargetNormalStyle::Any,
                            _ => TargetNormalStyle::DirectX,
                        };

                        if let Some((issue, details)) =
                            check_normal_map_orientation(p, target_style)
                        {
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

/// Compares properties and format specifications of Folder A against Folder B.
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

/// Compiles a flat asset inventory list of metadata specifications across targeted directories.
pub async fn run_asset_audit(params: ScanParams) -> Result<Vec<DuplicateFileSummary>> {
    run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let total = paths.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        let rows: Vec<DuplicateFileSummary> = paths
            .par_iter()
            .filter_map(|p| {
                if cancel_token.load(Ordering::Relaxed) {
                    return None;
                }

                let qc_meta = QcImageMetadata::extract_or_fallback(p);

                let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_cb(current as f32 / total as f32, current, total);

                Some(DuplicateFileSummary {
                    path: p.to_string_lossy().to_string(),
                    size: qc_meta.file_size,
                    width: qc_meta.width as usize,
                    height: qc_meta.height as usize,
                    format_str: qc_meta.format_str,
                    compression_format: qc_meta.compression_format,
                    color_space: qc_meta.color_space,
                    has_alpha: qc_meta.has_alpha,
                    bit_depth: qc_meta.bit_depth,
                    mipmap_count: qc_meta.mipmap_count,
                    is_cubemap: qc_meta.is_cubemap,
                    similarity: 0.0,
                })
            })
            .collect();

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        Ok(rows)
    })
}
