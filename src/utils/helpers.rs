// src/utils/helpers.rs

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use xxhash_rust::xxh64::Xxh64;

/// Discovers target files recursively based on given extension filters.
/// Excludes directories that match any tokens defined in the excluded_folders slice.
pub fn discover_files(
    root: &Path,
    extensions: &[String],
    excluded_folders: &[String],
) -> (Vec<PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut warnings = Vec::new();

    // Use walkdir's filter_entry to prevent scanning inside ignored directories at all, saving disk IO!
    let walker = walkdir::WalkDir::new(root).into_iter().filter_entry(|e| {
        if e.file_type().is_dir() {
            let name = e.file_name().to_string_lossy().to_lowercase();
            !excluded_folders
                .iter()
                .any(|ex| name == ex.trim().to_lowercase())
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

/// Formats bytes to human-readable sizes (e.g., "1.45 MB")
pub fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".to_string();
    }
    let k = 1024.0;
    let sizes = ["B", "KB", "MB", "GB", "TB"];
    let i = (bytes as f64).log(k).floor() as usize;
    format!("{:.2} {}", bytes as f64 / k.powi(i as i32), sizes[i])
}

/// Calculates a high-speed xxHash64 string representation for a file stream
pub fn calculate_xxhash(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Xxh64::new(0);
    let mut buffer = vec![0; 8 * 1024 * 1024]; // 8MB chunk buffer
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.digest()))
}

// MUTEX EXTENSION TRAIT FOR RESILIENT LOCKING
pub trait MutexExt<T> {
    /// Locks the mutex safely. If the mutex is poisoned due to a panic in another
    /// thread, it will recover the inner guard anyway instead of crashing.
    fn safe_lock(&self) -> std::sync::MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for std::sync::Mutex<T> {
    fn safe_lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|poisoned| {
            // Write a diagnostic warning to stderr (or console) but recover state
            eprintln!("[WARN] Mutex was poisoned! Recovering internal state safely.");
            poisoned.into_inner()
        })
    }
}
