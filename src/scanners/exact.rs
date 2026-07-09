// src/scanners/exact.rs

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::state::{DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::helpers::{calculate_xxhash, discover_files};

pub async fn run_exact_scan(params: super::ScanParams) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(params.dir_a.clone());
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    // Split raw tag string into dynamic token list
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
            let (width, height) = match imagesize::size(p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let hash = calculate_xxhash(p).ok()?;
            Some(FileMetadata {
                path: p.clone(),
                size,
                width,
                height,
                hash,
            })
        })
        .collect();

    if params.cancel_token.load(Ordering::Relaxed) {
        return Err(anyhow!("Scan cancelled by user."));
    }

    let mut groups: HashMap<String, Vec<FileMetadata>> = HashMap::new();
    for meta in metadata_list {
        groups.entry(meta.hash.clone()).or_default().push(meta);
    }

    let mut dups: Vec<(String, Vec<FileMetadata>)> =
        groups.into_iter().filter(|(_, f)| f.len() > 1).collect();
    dups.sort_by(|a, b| b.1[0].size.cmp(&a.1[0].size));

    let mut results = Vec::new();
    for (hash, files) in dups {
        let file_summaries = files
            .into_iter()
            .map(|f| {
                let qc_meta = crate::core::qc::extract_qc_metadata(&f.path).unwrap_or_else(|_| {
                    crate::core::qc::QcImageMetadata {
                        width: f.width as u32,
                        height: f.height as u32,
                        file_size: f.size,
                        format_str: f
                            .path
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
}

#[derive(Debug, Clone)]
struct FileMetadata {
    path: PathBuf,
    size: u64,
    width: usize,
    height: usize,
    hash: String,
}
