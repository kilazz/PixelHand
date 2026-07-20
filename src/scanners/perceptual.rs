// src/scanners/perceptual.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::core::perceptual::{calculate_hamming_distance, calculate_perceptual_hashes};
use crate::scanners::AnalysisItem;
use crate::state::{DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::clustering::UnionFind;
use crate::utils::helpers::discover_files;

/// Executes a perceptual duplicate image scan using difference hash (dHash) metrics.
pub async fn run_perceptual_scan_internal(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(&params.paths.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let ex_folders: Vec<String> = params
        .paths
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
    let items = super::generate_analysis_items(&paths, &params);

    crate::app::append_to_console_log(&format!(
        "Perceptual Scan: Generated {} analysis items from {} files.",
        items.len(),
        paths.len()
    ));

    let total = items.len();
    let processed = std::sync::atomic::AtomicUsize::new(0);

    // Compute perceptual dHash structures across all worker threads in parallel
    let hashes: Vec<(AnalysisItem, String)> = items
        .par_iter()
        .filter_map(|item| {
            if cancel_token.load(Ordering::Relaxed) {
                return None;
            }

            let res = calculate_perceptual_hashes(
                &item.path,
                item.analysis_type,
                params.prep.prep_ignore_solid,
            )?;

            let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref cb) = params.on_progress {
                cb(current as f32 / total as f32, current, total);
            }

            Some((item.clone(), res.dhash))
        })
        .collect();

    if cancel_token.load(Ordering::Relaxed) {
        return Err(anyhow!("Scan cancelled by user."));
    }

    // Convert similarity threshold into Hamming Distance bounds (64-bit space)
    let max_dist = (((100.0 - params.similarity) / 100.0) * 64.0).round() as u32;

    // Perform disjoint set union (Union-Find) clustering to group similar hashes
    let mut uf = UnionFind::new(hashes.len());
    for i in 0..hashes.len() {
        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }
        for j in (i + 1)..hashes.len() {
            if let Some(dist) = calculate_hamming_distance(&hashes[i].1, &hashes[j].1)
                && dist <= max_dist
            {
                uf.union(i, j);
            }
        }
    }

    let groups = uf.get_groups();
    let mut results = Vec::new();
    let mut group_counter = 0;

    for (_root_idx, member_indices) in groups {
        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        if member_indices.len() < 2 {
            continue;
        }

        let mut file_summaries = Vec::with_capacity(member_indices.len());
        let mut seen_paths = HashSet::with_capacity(member_indices.len());

        for idx in &member_indices {
            let (item, _) = &hashes[*idx];
            let path_str = item.path.to_string_lossy().to_string();

            if seen_paths.contains(&path_str) {
                continue;
            }
            seen_paths.insert(path_str.clone());

            // DRY implementation: Safely extract metadata using the robust fallback factory
            let qc_meta = crate::core::qc::QcImageMetadata::extract_or_fallback(&item.path);

            file_summaries.push(DuplicateFileSummary {
                path: path_str,
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
                similarity: 100.0,
            });
        }

        if file_summaries.len() > 1 {
            group_counter += 1;

            // Sort file summaries so that the highest resolution/largest size represents the "best" master copy
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
            let best_hash = member_indices
                .iter()
                .find(|&&idx| hashes[idx].0.path.to_string_lossy() == *best_path)
                .map(|&idx| &hashes[idx].1)
                .unwrap_or(&hashes[member_indices[0]].1);

            // Compute similarity against the chosen master copy hash
            for file in &mut file_summaries {
                let cur_hash = member_indices
                    .iter()
                    .find(|&&idx| hashes[idx].0.path.to_string_lossy() == file.path)
                    .map(|&idx| &hashes[idx].1)
                    .unwrap_or(&hashes[member_indices[0]].1);

                let dist = calculate_hamming_distance(best_hash, cur_hash).unwrap_or(0);
                file.similarity = (1.0 - (dist as f32 / 64.0)) * 100.0;
            }

            results.push(DuplicateGroupSummary {
                hash: group_counter.to_string(),
                files: file_summaries,
            });
        }
    }

    Ok(results)
}
