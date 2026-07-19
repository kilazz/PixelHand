// src/utils/helpers.rs

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use xxhash_rust::xxh64::Xxh64;

/// Recursively discovers target files using WalkDir filtering.
/// Pre-caches trimmed excluded folder tokens into a HashSet to prevent redundant string allocations.
pub fn discover_files(
    root: &Path,
    extensions: &[String],
    excluded_folders: &[String],
) -> (Vec<PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut warnings = Vec::new();

    // Cache lower-case trimmed exclusion tokens to prevent continuous allocations inside recursive closure
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

/// Converts bytes size into human-readable representation safely avoiding boundaries panics.
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

/// Resilient Mutex locking wrapper trait.
pub trait MutexExt<T> {
    /// Locks the mutex safely. If another thread panicked and poisoned the lock,
    /// it recovers the inner guard state instead of crashing.
    fn safe_lock(&self) -> std::sync::MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for std::sync::Mutex<T> {
    fn safe_lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|poisoned| {
            eprintln!("[WARN] Recovering state from poisoned lock safely.");
            poisoned.into_inner()
        })
    }
}
