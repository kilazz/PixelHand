// src/utils/helpers.rs

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use xxhash_rust::xxh64::Xxh64;

use crate::perceptual::hashing::AnalysisType;
use crate::state::models::ScanParams;

// =============================================================================
// --- PREPROCESSING DATA MODELS -----------------------------------------------
// =============================================================================

/// Represents a discrete analysis unit mapped by the image channels splitter.
#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub path: PathBuf,
    pub analysis_type: AnalysisType,
}

// =============================================================================
// --- FILESYSTEM & HASHING HELPERS --------------------------------------------
// =============================================================================

/// Parses a comma-separated list of ignored folders into clean, trimmed String tokens.
pub fn parse_excluded_folders(excluded_str: &str) -> Vec<String> {
    excluded_str
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Recursively discovers targeted file paths using WalkDir filtering.
/// Caches lower-case trimmed exclusion tokens to prevent continuous heap allocations during recursion.
pub fn discover_files(
    root: &Path,
    extensions: &[String],
    excluded_folders: &[String],
) -> (Vec<PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut warnings = Vec::new();

    let excluded_set: HashSet<String> = excluded_folders
        .iter()
        .map(|f| f.trim().to_lowercase())
        .collect();

    // Traverse directory trees safely, skipping ignored folders immediately to optimize disk I/O
    let walker = walkdir::WalkDir::new(root).into_iter().filter_entry(|e| {
        if e.file_type().is_dir() {
            let name = e.file_name().to_string_lossy().to_lowercase();
            !excluded_set.contains(&name)
        } else {
            true
        }
    });

    for entry in walker {
        match entry {
            Ok(e) => {
                if e.file_type().is_file() {
                    let p = e.into_path();
                    if let Some(ext) = p.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        let lower_ext = if ext_str.starts_with('.') {
                            ext_str.clone()
                        } else {
                            format!(".{}", ext_str)
                        };
                        if extensions.contains(&lower_ext) || extensions.contains(&ext_str) {
                            files.push(p);
                        }
                    }
                }
            }
            Err(err) => {
                let path_str = err
                    .path()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown".into());
                warnings.push(format!(
                    "Access Warning: cannot read {} ({})",
                    path_str, err
                ));
            }
        }
    }
    (files, warnings)
}

/// Computes high-speed xxHash64 hash from file stream using an optimized buffer memory page.
pub fn calculate_xxhash(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Xxh64::new(0);

    // Configured to a 1 MB buffer chunk instead of 8 MB to protect CPU cache levels
    // and minimize system RAM footprint during parallel runs inside Rayon thread pools.
    let mut buffer = vec![0; 1024 * 1024];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.digest()))
}

// =============================================================================
// --- STRING FORMATTING HELPERS -----------------------------------------------
// =============================================================================

/// Converts bytes size into human-readable representation safely.
pub fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".to_string();
    }
    let k = 1024.0;
    let sizes = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    let mut i = (bytes as f64).log(k).floor() as usize;

    // Safeguard index to prevent out of bounds panics on extremely large inputs
    i = i.min(sizes.len() - 1);

    format!("{:.2} {}", bytes as f64 / k.powi(i as i32), sizes[i])
}

// =============================================================================
// --- PREPROCESSING GENERATORS ------------------------------------------------
// =============================================================================

/// Expands a flat list of file paths into distinct AnalysisItems based on pre-processing rules.
pub fn generate_analysis_items(paths: &[PathBuf], params: &ScanParams) -> Vec<AnalysisItem> {
    let mut items = Vec::new();

    let tags: Vec<String> = params
        .prep
        .prep_tags
        .split(',')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    for path in paths {
        let path_str = path.to_string_lossy().to_lowercase();

        let matches_tags = if tags.is_empty() {
            true
        } else {
            tags.iter().any(|tag| path_str.contains(tag))
        };

        if params.prep.prep_channels && matches_tags {
            if params.prep.prep_r {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::R,
                });
            }
            if params.prep.prep_g {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::G,
                });
            }
            if params.prep.prep_b {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::B,
                });
            }
            if params.prep.prep_a {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::A,
                });
            }
        } else if params.prep.prep_luminance {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Luminance,
            });
        } else {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Composite,
            });
        }
    }

    items
}

// =============================================================================
// --- COMMON SCANNERS PIPELINE ------------------------------------------------
// =============================================================================

/// A unified pipeline for all scanners. Handles path validation, file discovery, filtering, and cancel token management.
pub fn run_scan_pipeline<F, R>(params: &ScanParams, process_files: F) -> anyhow::Result<R>
where
    F: FnOnce(
        Vec<PathBuf>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        std::sync::Arc<dyn Fn(f32, usize, usize) + Send + Sync>,
    ) -> anyhow::Result<R>,
{
    let path = PathBuf::from(&params.paths.dir_a);
    if !path.is_dir() {
        return Err(anyhow::anyhow!(
            "The specified path is not a valid directory"
        ));
    }

    let ex_folders = parse_excluded_folders(&params.paths.excluded_folders);

    let (paths, warnings) = discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let cancel_token = params.cancel_token.clone();

    // Fallback to a dummy callback if no progress reporter is attached
    let dummy_progress: std::sync::Arc<dyn Fn(f32, usize, usize) + Send + Sync> =
        std::sync::Arc::new(|_, _, _| {});
    let progress_cb = params.on_progress.clone().unwrap_or(dummy_progress);

    process_files(paths, cancel_token, progress_cb)
}
