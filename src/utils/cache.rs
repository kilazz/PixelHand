// src/utils/cache.rs

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub struct DecodedCacheItem {
    pub mtime: std::time::SystemTime,
    pub last_accessed: std::time::Instant,
    pub image: image::RgbaImage,
}

// Public static OnceLock to allow app.rs to clear the cache upon tonemapping adjustments
pub static DECODED_CACHE: OnceLock<Mutex<HashMap<String, DecodedCacheItem>>> = OnceLock::new();

pub async fn get_channel_preview_image(path: &str, channel: &str) -> Option<image::RgbaImage> {
    let p = PathBuf::from(path);
    if !p.is_file() {
        return None;
    }

    let current_mtime = fs::metadata(&p)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let cache_mutex = DECODED_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache_mutex.lock().ok()?;

    if let Some(item) = cache.get_mut(path) {
        if item.mtime == current_mtime {
            item.last_accessed = std::time::Instant::now();
        } else {
            cache.remove(path);
        }
    }

    if !cache.contains_key(path) {
        // Load 1:1 original full-resolution pixel buffers for split-pane inspects
        let img = crate::format_loaders::dds_loader::open_image_with_dds_fallback(&p, None).ok()?;

        // Least-Recently-Used (LRU) eviction rule limiting cache to 4 massive images to avoid OOM
        if cache.len() >= 4 {
            let mut oldest_key = None;
            let mut oldest_time = std::time::Instant::now();
            for (k, item) in cache.iter() {
                if item.last_accessed < oldest_time {
                    oldest_time = item.last_accessed;
                    oldest_key = Some(k.clone());
                }
            }
            if let Some(k) = oldest_key {
                cache.remove(&k);
            }
        }

        cache.insert(
            path.to_string(),
            DecodedCacheItem {
                mtime: current_mtime,
                last_accessed: std::time::Instant::now(),
                image: img.into_rgba8(),
            },
        );
    }

    let cached_item = cache.get(path)?;
    let rgba = &cached_item.image;

    let out_img = if channel == "RGB" || channel == "Composite" {
        if crate::core::perceptual::is_vfx_transparent_texture(rgba) {
            image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        } else {
            image::DynamicImage::ImageRgba8(rgba.clone())
        }
    } else {
        let channel_idx = match channel {
            "R" => 0,
            "G" => 1,
            "B" => 2,
            _ => 3,
        };
        let mut out_rgb = image::RgbImage::new(rgba.width(), rgba.height());
        for (x, y, pixel) in rgba.enumerate_pixels() {
            let val = pixel[channel_idx];
            out_rgb.put_pixel(x, y, image::Rgb([val, val, val]));
        }
        image::DynamicImage::ImageRgb8(out_rgb)
    };

    Some(out_img.into_rgba8())
}

// --- SMART LANCEDB VECTOR CACHE GARBAGE COLLECTOR ---

/// Recursively calculates the total disk size of a directory in bytes.
fn get_dir_size(path: &Path) -> u64 {
    let mut total_size = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let metadata = entry.metadata();
            if let Ok(meta) = metadata {
                if meta.is_dir() {
                    total_size += get_dir_size(&entry.path());
                } else {
                    total_size += meta.len();
                }
            }
        }
    }
    total_size
}

/// Automatically runs a background cleaner over LanceDB caches.
/// Removes DB databases that haven't been accessed for 14 days,
/// or evicts the oldest caches if total `.lancedb_cache` folder size exceeds 1.0 GB.
pub fn run_vector_cache_garbage_collector() {
    let Ok(app_dir) = crate::utils::settings::get_portable_app_data_dir() else {
        return;
    };
    let lancedb_cache_dir = app_dir.join(".lancedb_cache");
    if !lancedb_cache_dir.is_dir() {
        return;
    }

    let Ok(entries) = fs::read_dir(&lancedb_cache_dir) else {
        return;
    };

    let mut cache_folders = Vec::new();
    let now = std::time::SystemTime::now();
    let max_age = std::time::Duration::from_secs(14 * 24 * 60 * 60); // 14 Days

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(meta) = path.metadata() else {
                continue;
            };
            let accessed_time = meta.accessed().unwrap_or(now);

            // COLLAPSED NESTED IF STATEMENTS WITH LET-CHAINS
            if let Ok(age) = now.duration_since(accessed_time)
                && age > max_age
            {
                let _ = fs::remove_dir_all(&path);
                continue;
            }

            let folder_size = get_dir_size(&path);
            cache_folders.push((path, accessed_time, folder_size));
        }
    }

    // Sort folders by accessed time (oldest first)
    cache_folders.sort_by_key(|f| f.1);

    // IDENTITY OP MULTIPLICATION REMOVED
    let max_cache_bytes: u64 = 1024_u64 * 1024 * 1024; // 1 GiB limit
    let mut current_total_size: u64 = cache_folders.iter().map(|f| f.2).sum();

    // Evict oldest caches until we are strictly under the 1.0 GB threshold
    for (path, _, size) in cache_folders {
        if current_total_size <= max_cache_bytes {
            break;
        }
        if fs::remove_dir_all(&path).is_ok() {
            current_total_size = current_total_size.saturating_sub(size);
            #[cfg(debug_assertions)]
            eprintln!("[GC] Evicted old LanceDB cache: {:?}", path);
        }
    }
}
