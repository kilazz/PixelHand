// src/scanners/perceptual.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::core::perceptual::{calculate_hamming_distance, calculate_perceptual_hashes};
use crate::scanners::AnalysisItem;
use crate::state::{DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::helpers::discover_files;

pub async fn run_perceptual_scan_internal(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(&params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let (paths, warnings) = discover_files(&path, &params.extensions);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();

    // Dynamically generate specific channel or luminance tasks based on Pre-processing configurations
    let items = super::generate_analysis_items(&paths, &params);

    crate::app::append_to_console_log(&format!(
        "Perceptual Scan: Generated {} analysis items from {} files.",
        items.len(),
        paths.len()
    ));

    let hashes: Vec<(AnalysisItem, String)> = items
        .par_iter()
        .filter_map(|item| {
            if cancel_token.load(Ordering::Relaxed) {
                return None;
            }

            // Calculate perceptual hash for the dynamically assigned analysis type (Composite, Luma, R, G, B, A)
            let res = calculate_perceptual_hashes(
                &item.path,
                item.analysis_type,
                params.prep_ignore_solid,
            )?;

            Some((item.clone(), res.dhash))
        })
        .collect();

    let mut visited = vec![false; hashes.len()];
    let mut results = Vec::new();
    let mut group_id = 0;

    // Convert similarity percentage into allowed max hamming distance (max 64 bits)
    let max_dist = (((100.0 - params.similarity) / 100.0) * 64.0).round() as u32;

    for i in 0..hashes.len() {
        if params.cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        if visited[i] {
            continue;
        }
        let mut group_members = vec![hashes[i].clone()];

        for j in (i + 1)..hashes.len() {
            if visited[j] {
                continue;
            }

            // --- FIXED: Collapsed nested if blocks using stable let-chains style ---
            if let Some(dist) = calculate_hamming_distance(&hashes[i].1, &hashes[j].1)
                && dist <= max_dist
            {
                group_members.push(hashes[j].clone());
                visited[j] = true;
            }
        }

        if group_members.len() > 1 {
            group_id += 1;
            let mut file_summaries = Vec::new();

            // Track physical paths to avoid adding the exact same file twice to the same group
            // (For example, if its R channel mathematically matched its own G channel)
            let mut seen_paths = HashSet::new();

            for (item, _) in &group_members {
                let p_buf = &item.path;
                let path_str = p_buf.to_string_lossy().to_string();

                if seen_paths.contains(&path_str) {
                    continue;
                }
                seen_paths.insert(path_str.clone());

                let metadata = fs::metadata(p_buf)?;
                let size = metadata.len();
                let (width, height) = match imagesize::size(p_buf) {
                    Ok(dim) => (dim.width, dim.height),
                    Err(_) => (0, 0),
                };
                let qc_meta = crate::core::qc::extract_qc_metadata(p_buf).unwrap_or_else(|_| {
                    crate::core::qc::QcImageMetadata {
                        width: width as u32,
                        height: height as u32,
                        file_size: size,
                        format_str: p_buf
                            .extension()
                            .map(|e| e.to_string_lossy().to_string())
                            .unwrap_or_default(),
                        compression_format: "Unknown".into(),
                        color_space: "Unknown".into(),
                        has_alpha: false,
                        bit_depth: 8,
                        mipmap_count: 1,
                    }
                });

                file_summaries.push(DuplicateFileSummary {
                    path: path_str,
                    size,
                    width,
                    height,
                    format_str: qc_meta.format_str,
                    compression_format: qc_meta.compression_format,
                    color_space: qc_meta.color_space,
                    has_alpha: qc_meta.has_alpha,
                    bit_depth: qc_meta.bit_depth,
                    mipmap_count: qc_meta.mipmap_count,
                    similarity: 100.0,
                });
            }

            // A group might end up with only 1 unique physical file if the match was purely internal channels.
            // We only publish groups that map to multiple distinct files.
            if file_summaries.len() > 1 {
                file_summaries.sort_by(|a, b| {
                    let area_a = a.width * a.height;
                    let area_b = b.width * b.height;
                    if area_a != area_b {
                        area_b.cmp(&area_a)
                    } else {
                        b.size.cmp(&a.size)
                    }
                });

                let best_path = &file_summaries[0].path;
                let best_hash = group_members
                    .iter()
                    .find(|(item, _)| item.path.to_string_lossy() == *best_path)
                    .map(|(_, h)| h.as_str())
                    .unwrap_or(&group_members[0].1);

                for file in &mut file_summaries {
                    let cur_hash = group_members
                        .iter()
                        .find(|(item, _)| item.path.to_string_lossy() == file.path)
                        .map(|(_, h)| h.as_str())
                        .unwrap_or(&group_members[0].1);
                    let dist = calculate_hamming_distance(best_hash, cur_hash).unwrap_or(0);
                    file.similarity = (1.0 - (dist as f32 / 64.0)) * 100.0;
                }

                results.push(DuplicateGroupSummary {
                    hash: group_id.to_string(),
                    files: file_summaries,
                });
            }
        }
    }

    Ok(results)
}
