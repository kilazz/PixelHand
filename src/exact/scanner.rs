// src/exact/scanner.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::qc::rules::QcImageMetadata;
use crate::state::models::{DuplicateFileSummary, DuplicateGroupSummary, ScanParams};
use crate::utils::helpers::{calculate_xxhash, run_scan_pipeline};

#[derive(Debug, Clone)]
struct FileMetadata {
    path: PathBuf,
    size: u64,
    width: usize,
    height: usize,
    hash: String,
}

/// Executes a byte-exact duplicate scan across discovered files using high-speed xxHash64.
pub async fn run_exact_scan(params: ScanParams) -> Result<Vec<DuplicateGroupSummary>> {
    // Delegate file discovery, warnings capturing, and progress reporting to the pipeline helper
    run_scan_pipeline(&params, |paths, cancel_token, progress_cb| {
        let total = paths.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        // Process files in parallel, filtering out errors and zero-byte allocations
        let metadata_list: Vec<FileMetadata> = paths
            .par_iter()
            .filter_map(|p| {
                if cancel_token.load(Ordering::Relaxed) {
                    return None;
                }

                let metadata = fs::metadata(p).ok()?;
                let size = metadata.len();
                if size == 0 {
                    return None;
                }

                // Extract lightweight image size here to bypass heavier QC parsing on non-duplicates
                let (width, height) = imagesize::size(p)
                    .map(|dim| (dim.width, dim.height))
                    .unwrap_or((0, 0));

                let hash = calculate_xxhash(p).ok()?;

                let current = processed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_cb(current as f32 / total as f32, current, total);

                Some(FileMetadata {
                    path: p.clone(),
                    size,
                    width,
                    height,
                    hash,
                })
            })
            .collect();

        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        // Group files with identical hash values into buckets
        let mut groups: HashMap<String, Vec<FileMetadata>> = HashMap::new();
        for meta in metadata_list {
            groups.entry(meta.hash.clone()).or_default().push(meta);
        }

        // Retain only buckets containing actual duplicates (count > 1)
        let mut dups: Vec<(String, Vec<FileMetadata>)> =
            groups.into_iter().filter(|(_, f)| f.len() > 1).collect();

        // Sort groups by file size descending (largest duplicate clusters first)
        dups.sort_by(|a, b| {
            let size_a = a.1.first().map(|f| f.size).unwrap_or(0);
            let size_b = b.1.first().map(|f| f.size).unwrap_or(0);
            size_b.cmp(&size_a)
        });

        let mut results = Vec::with_capacity(dups.len());
        for (hash, files) in dups {
            let file_summaries = files
                .into_iter()
                .map(|f| {
                    // Centralized fallback handles errors gracefully inside extract_or_fallback
                    let qc_meta = QcImageMetadata::extract_or_fallback(&f.path);

                    DuplicateFileSummary {
                        path: f.path.to_string_lossy().to_string(),
                        size: f.size,
                        width: f.width,
                        height: f.height,
                        format_str: qc_meta.format_str,
                        compression_format: qc_meta.compression_format,
                        color_space: qc_meta.color_space,
                        has_alpha: qc_meta.has_alpha,
                        bit_depth: qc_meta.bit_depth,
                        mipmap_count: qc_meta.mipmap_count,
                        is_cubemap: qc_meta.is_cubemap,
                        similarity: 100.0,
                    }
                })
                .collect();

            results.push(DuplicateGroupSummary {
                hash,
                files: file_summaries,
            });
        }
        Ok(results)
    })
}
