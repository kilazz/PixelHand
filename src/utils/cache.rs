// src/utils/cache.rs

use moka::sync::Cache;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[derive(Clone)]
pub struct DecodedCacheItem {
    pub mtime: std::time::SystemTime,
    pub image: Arc<image::RgbaImage>, // Wrapped in Arc to prevent copying heavy raw pixel allocations
}

/// Global dynamic memory cache to hold decrypted 1:1 original pixel buffers.
pub static DECODED_CACHE: OnceLock<Cache<String, DecodedCacheItem>> = OnceLock::new();

/// Decodes and returns the requested color channels (R, G, B, A, or Composite)
/// of the image at the specified path for a specific mipmap level, utilizing an automated weight-based memory cache.
pub async fn get_channel_preview_image(
    path: &str,
    channel: &str,
    mip_level: u32,
) -> Option<image::RgbaImage> {
    let p = PathBuf::from(path);
    if !p.is_file() {
        return None;
    }

    let current_mtime = fs::metadata(&p)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    // Generate unique cache key incorporating both path and mip level
    let cache_key = format!("{}_mip{}", path, mip_level);

    let cache = DECODED_CACHE.get_or_init(|| {
        Cache::builder()
            // Limit to approximately 2.0 GB of decompressed texture preview data in RAM
            .max_capacity(2 * 1024 * 1024 * 1024)
            // Weigh each item according to its actual decompressed pixel payload (Width * Height * 4 channels)
            .weigher(|_k, v: &DecodedCacheItem| {
                let bytes = v.image.width() as u64 * v.image.height() as u64 * 4;
                bytes.min(u32::MAX as u64) as u32
            })
            .build()
    });

    // Invalidate the cache entry if the underlying file has been modified on disk
    if let Some(item) = cache.get(&cache_key)
        && item.mtime != current_mtime
    {
        cache.invalidate(&cache_key);
    }

    // Atomically load and cache the image, preventing Cache Stampede.
    // If another thread is already loading this exact file, the current thread will wait for it to finish.
    let cached_item_res =
        cache.try_get_with(cache_key.clone(), || -> Result<DecodedCacheItem, String> {
            if let Ok(img) =
                crate::format_loaders::dds_loader::open_image_with_specific_mip(&p, mip_level)
            {
                Ok(DecodedCacheItem {
                    mtime: current_mtime,
                    image: Arc::new(img.into_rgba8()),
                })
            } else {
                Err("Failed to decode image".to_string())
            }
        });

    let cached_item = cached_item_res.ok()?;
    let rgba = &*cached_item.image;

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

/// Recursively calculates the total disk size of a directory in bytes.
fn get_dir_size(path: &Path) -> u64 {
    fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| {
            if let Ok(meta) = e.metadata() {
                if meta.is_dir() {
                    get_dir_size(&e.path())
                } else {
                    meta.len()
                }
            } else {
                0
            }
        })
        .sum()
}

/// Evaluates LanceDB workspace caches on startup. Wipes partitions untouched for
/// longer than 14 days, and evicts the oldest partitions if total disk size exceeds 1.0 GB.
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
    let max_age = std::time::Duration::from_secs(14 * 24 * 60 * 60);

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(meta) = path.metadata() else {
                continue;
            };
            let accessed_time = meta.accessed().unwrap_or(now);

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

    let max_cache_bytes: u64 = 1024 * 1024 * 1024; // 1 GiB capacity limit
    let mut current_total_size: u64 = cache_folders.iter().map(|f| f.2).sum();

    // Evict oldest caches until total size fits inside the limit
    for (path, _, size) in cache_folders {
        if current_total_size <= max_cache_bytes {
            break;
        }
        if fs::remove_dir_all(&path).is_ok() {
            current_total_size = current_total_size.saturating_sub(size);
        }
    }
}
