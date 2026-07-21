// src/perceptual/scanner.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::perceptual::hashing::{calculate_hamming_distance, calculate_perceptual_hash};
use crate::qc::rules::QcImageMetadata;
use crate::state::models::{DuplicateFileSummary, DuplicateGroupSummary, ScanParams};
use crate::utils::clustering::UnionFind;
use crate::utils::helpers::{AnalysisItem, generate_analysis_items, run_scan_pipeline};

/// Executes a perceptual duplicate image scan using difference hash (dHash) metrics.
pub async fn run_perceptual_scan_internal(
    params: ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let items: Vec<AnalysisItem> = generate_analysis_items(&paths, &params);

        crate::app::append_to_console_log(&format!(
            "Perceptual Scan: Generated {} analysis items from {} discovered files.",
            items.len(),
            paths.len()
        ));

        let total = items.len();
        let processed = AtomicUsize::new(0);

        // Compute perceptual dHash fingerprints stored directly as register u64 integers (Zero Heap Allocation)
        let hashes: Vec<(AnalysisItem, u64)> = items
            .par_iter()
            .filter_map(|item| {
                if cancel_token.load(Ordering::Relaxed) {
                    return None;
                }

                let hash = calculate_perceptual_hash(
                    &item.path,
                    item.analysis_type,
                    params.prep.prep_ignore_solid,
                )?;

                let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_cb(current as f32 / total as f32, current, total);

                Some((item.clone(), hash))
            })
            .collect();

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        let max_dist = (((100.0 - params.similarity) / 100.0) * 64.0).round() as u32;
        let total_hashes = hashes.len();

        // Lock-free Dynamic Work-Stealing Pairwise Comparison (Fannkuch Pattern)
        let chunk_size = 128usize;
        let next_chunk = AtomicUsize::new(0);

        let matching_pairs: Vec<(usize, usize)> = (0..rayon::current_num_threads())
            .into_par_iter()
            .flat_map(|_| {
                let mut local_pairs = Vec::new();
                loop {
                    if cancel_token.load(Ordering::Relaxed) {
                        break;
                    }
                    let start_i = next_chunk.fetch_add(chunk_size, Ordering::Relaxed);
                    if start_i >= total_hashes {
                        break;
                    }
                    let end_i = (start_i + chunk_size).min(total_hashes);

                    for i in start_i..end_i {
                        let h1 = hashes[i].1;
                        for (offset, (_, h2)) in hashes[i + 1..].iter().enumerate() {
                            let j = i + 1 + offset;
                            // Executed in 1 clock cycle via hardware POPCNT
                            if calculate_hamming_distance(h1, *h2) <= max_dist {
                                local_pairs.push((i, j));
                            }
                        }
                    }
                }
                local_pairs
            })
            .collect();

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        // Perform disjoint set union (Union-Find) clustering sequentially on matched pairs
        let mut uf = UnionFind::new(hashes.len());
        for (i, j) in matching_pairs {
            uf.union(i, j);
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

                let qc_meta = QcImageMetadata::extract_or_fallback(&item.path);

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
                    .map(|&idx| hashes[idx].1)
                    .unwrap_or(hashes[member_indices[0]].1);

                for file in &mut file_summaries {
                    let cur_hash = member_indices
                        .iter()
                        .find(|&&idx| hashes[idx].0.path.to_string_lossy() == file.path)
                        .map(|&idx| hashes[idx].1)
                        .unwrap_or(hashes[member_indices[0]].1);

                    let dist = calculate_hamming_distance(best_hash, cur_hash);
                    file.similarity = (1.0 - (dist as f32 / 64.0)) * 100.0;
                }

                results.push(DuplicateGroupSummary {
                    hash: group_counter.to_string(),
                    files: file_summaries,
                });
            }
        }

        Ok(results)
    })
}
