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

pub async fn run_qc_scan_internal(params: super::ScanParams) -> Result<Vec<QcIssueSummary>> {
    let path = PathBuf::from(params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let (paths, warnings) = discover_files(&path, &params.extensions);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();

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
                });

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

            if params.qc_solid_colors
                && let Some((issue, details)) = check_solid_texture(p)
            {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details,
                });
            }

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
                if should_check && let Some((issue, details)) = check_normal_map_integrity(p, 0.15)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }
            }
            file_issues
        })
        .collect();

    if params.cancel_token.load(Ordering::Relaxed) {
        return Err(anyhow!("Scan cancelled by user."));
    }

    Ok(issues)
}

pub async fn run_folder_compare(
    directory_a: String,
    directory_b: String,
    extensions: Vec<String>,
) -> Result<Vec<QcIssueSummary>> {
    let path_a = PathBuf::from(directory_a);
    let path_b = PathBuf::from(directory_b);
    if !path_a.is_dir() || !path_b.is_dir() {
        return Err(anyhow!("Both paths must be valid directories"));
    }

    let (files_a, warnings_a) = discover_files(&path_a, &extensions);
    for warn in warnings_a {
        crate::app::append_to_console_log(&warn);
    }
    let (files_b, warnings_b) = discover_files(&path_b, &extensions);
    for warn in warnings_b {
        crate::app::append_to_console_log(&warn);
    }

    let get_key = |p: &Path| -> String {
        p.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default()
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
                });

            let rel_issues = check_relative(&meta_a, &meta_b, true, true, true, true);
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
