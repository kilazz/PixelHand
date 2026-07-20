// src/utils/cache.rs

use moka::sync::Cache;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use ustr::Ustr;

#[derive(Clone)]
pub struct DecodedCacheItem {
    pub mtime: std::time::SystemTime,
    pub image: Arc<image::RgbaImage>, // Wrapped in Arc to prevent copying heavy raw pixel allocations
}

/// Container holding composite and lazy-allocated channel-isolated preview buffers
#[derive(Clone)]
pub struct CachedThumbnail {
    pub composite: image::RgbaImage,
    pub r_channel: Arc<OnceLock<image::RgbaImage>>,
    pub g_channel: Arc<OnceLock<image::RgbaImage>>,
    pub b_channel: Arc<OnceLock<image::RgbaImage>>,
    pub a_channel: Arc<OnceLock<image::RgbaImage>>,
}

impl CachedThumbnail {
    pub fn new(composite: image::RgbaImage) -> Self {
        Self {
            composite,
            r_channel: Arc::new(OnceLock::new()),
            g_channel: Arc::new(OnceLock::new()),
            b_channel: Arc::new(OnceLock::new()),
            a_channel: Arc::new(OnceLock::new()),
        }
    }

    /// Retrieves or dynamically computes a color-isolated preview on a separate thread, caching the result
    pub fn get_channel(&self, channel: &str) -> image::RgbaImage {
        let lock = match channel {
            "R" => &self.r_channel,
            "G" => &self.g_channel,
            "B" => &self.b_channel,
            "A" => &self.a_channel,
            _ => return self.composite.clone(),
        };

        lock.get_or_init(|| {
            let idx = match channel {
                "R" => 0,
                "G" => 1,
                "B" => 2,
                _ => 3,
            };

            let width = self.composite.width();
            let height = self.composite.height();
            let mut out_rgba = image::RgbaImage::new(width, height);

            let composite_raw = self.composite.as_raw();
            let out_raw = out_rgba.as_mut();

            for (i, chunk) in composite_raw.chunks_exact(4).enumerate() {
                let val = chunk[idx];
                let dst_idx = i * 4;
                if idx == 3 {
                    out_raw[dst_idx] = val;
                    out_raw[dst_idx + 1] = val;
                    out_raw[dst_idx + 2] = val;
                    out_raw[dst_idx + 3] = val;
                } else {
                    out_raw[dst_idx] = val;
                    out_raw[dst_idx + 1] = val;
                    out_raw[dst_idx + 2] = val;
                    out_raw[dst_idx + 3] = 255;
                }
            }
            out_rgba
        })
        .clone()
    }
}

// ==========================================
// --- CONSOLIDATED MEMORY CACHE MANAGER ----
// ==========================================

pub struct CacheManager {
    pub decoded_images: Cache<String, DecodedCacheItem>,
    pub thumbnails: Cache<Ustr, CachedThumbnail>,
}

/// Retrieve the central thread-safe Memory Cache Manager singleton
pub fn get_cache_manager() -> &'static CacheManager {
    static INSTANCE: OnceLock<CacheManager> = OnceLock::new();
    INSTANCE.get_or_init(|| CacheManager {
        decoded_images: Cache::builder()
            // Limit to approximately 2.0 GB of decompressed texture preview data in RAM
            .max_capacity(2 * 1024 * 1024 * 1024)
            // Weigh each item according to its actual decompressed pixel payload
            .weigher(|_k, v: &DecodedCacheItem| {
                let bytes = v.image.width() as u64 * v.image.height() as u64 * 4;
                bytes.min(u32::MAX as u64) as u32
            })
            .build(),
        thumbnails: Cache::builder()
            // Limit to approximately 150 MB base capacity of decompressed texture preview data in RAM
            .max_capacity(150 * 1024 * 1024)
            // Weigh each item according to its composite pixel payload
            .weigher(|_k, v: &CachedThumbnail| {
                let bytes = v.composite.width() as u64 * v.composite.height() as u64 * 4;
                bytes.min(u32::MAX as u64) as u32
            })
            .build(),
    })
}

/// Normalizes path representations across Windows and Unix platforms to guarantee absolute cache hit consistency
pub fn normalize_path_key(path_str: &str) -> Ustr {
    let normalized: String = path_str
        .chars()
        .map(|c| {
            if c == '/' || c == '\\' {
                std::path::MAIN_SEPARATOR
            } else {
                c
            }
        })
        .collect();
    ustr::ustr(&normalized.to_lowercase())
}

/// Stores an image buffer in the centralized thumbnails memory cache
pub fn store_in_thumbnail_memory_cache(path: &str, img: image::RgbaImage) {
    let manager = get_cache_manager();
    let normalized_key = normalize_path_key(path);
    manager
        .thumbnails
        .insert(normalized_key, CachedThumbnail::new(img));
}

/// Decodes and returns the requested color channels of the image utilizing the unified cache manager
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

    let cache_key = format!("{}_mip{}", path, mip_level);
    let manager = get_cache_manager();

    if let Some(item) = manager.decoded_images.get(&cache_key)
        && item.mtime != current_mtime
    {
        manager.decoded_images.invalidate(&cache_key);
    }

    // Atomically load and cache the image, preventing Cache Stampede
    let cached_item_res = manager.decoded_images.try_get_with(
        cache_key.clone(),
        || -> Result<DecodedCacheItem, String> {
            let t_config = crate::app::get_active_tonemap_config();

            if let Ok(img) =
                crate::format_loaders::open_image_with_specific_mip(&p, mip_level, Some(t_config))
            {
                Ok(DecodedCacheItem {
                    mtime: current_mtime,
                    image: Arc::new(img.into_rgba8()),
                })
            } else {
                Err("Failed to decode image".to_string())
            }
        },
    );

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

        out_rgb
            .as_mut()
            .par_chunks_exact_mut(3)
            .zip(rgba.as_raw().par_chunks_exact(4))
            .for_each(|(out_pixel, in_pixel)| {
                let val = in_pixel[channel_idx];
                out_pixel[0] = val;
                out_pixel[1] = val;
                out_pixel[2] = val;
            });

        image::DynamicImage::ImageRgb8(out_rgb)
    };

    Some(out_img.into_rgba8())
}

/// Recursively calculates the total disk size of a directory in bytes
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

    cache_folders.sort_by_key(|f| f.1);

    let max_cache_bytes: u64 = 1024 * 1024 * 1024; // 1 GiB capacity limit
    let mut current_total_size: u64 = cache_folders.iter().map(|f| f.2).sum();

    for (path, _, size) in cache_folders {
        if current_total_size <= max_cache_bytes {
            break;
        }
        if fs::remove_dir_all(&path).is_ok() {
            current_total_size = current_total_size.saturating_sub(size);
        }
    }
}
