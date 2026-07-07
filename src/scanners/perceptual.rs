// src/scanners/perceptual.rs
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::core::perceptual::{
    AnalysisType, calculate_hamming_distance, calculate_perceptual_hashes,
};
use crate::state::{DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::helpers::discover_files;

pub async fn run_perceptual_scan_internal(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let (paths, warnings) = discover_files(&path, &params.extensions);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let ana_type = match params.perceptual_channel.as_str() {
        "R" => AnalysisType::R,
        "G" => AnalysisType::G,
        "B" => AnalysisType::B,
        "A" => AnalysisType::A,
        "Luminance" => AnalysisType::Luminance,
        _ => AnalysisType::Composite,
    };

    let cancel_token = params.cancel_token.clone();

    let hashes: Vec<(PathBuf, String)> = paths
        .par_iter()
        .filter_map(|p| {
            if cancel_token.load(Ordering::Relaxed) {
                return None;
            }

            let res = calculate_perceptual_hashes(p, ana_type, true)?;
            Some((p.clone(), res.dhash))
        })
        .collect();

    let mut visited = vec![false; hashes.len()];
    let mut results = Vec::new();
    let mut group_id = 0;

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
            for (p_buf, _) in &group_members {
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
                    path: p_buf.to_string_lossy().to_string(),
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
                .find(|(pb, _)| pb.to_string_lossy() == *best_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or(&group_members[0].1);

            for file in &mut file_summaries {
                let cur_hash = group_members
                    .iter()
                    .find(|(pb, _)| pb.to_string_lossy() == file.path)
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
    Ok(results)
}
