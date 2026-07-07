// src/utils/cache.rs
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

struct DecodedCacheItem {
    mtime: std::time::SystemTime,
    last_accessed: std::time::Instant,
    image: image::RgbaImage,
}

// Thread-safe global cache of decoded preview images.
static DECODED_CACHE: OnceLock<Mutex<HashMap<String, DecodedCacheItem>>> = OnceLock::new();

/// Extracts a chosen pixel channel in grayscale to display in the comparative viewport.
/// Uses a custom MRU (Most Recently Used) caching strategy at 100% original resolution.
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

    // Invalidate stale cache if file modified
    if let Some(item) = cache.get_mut(path) {
        if item.mtime == current_mtime {
            item.last_accessed = std::time::Instant::now();
        } else {
            cache.remove(path);
        }
    }

    if !cache.contains_key(path) {
        // Load image at 100% full original resolution (no downscaling!)
        let img = crate::format_loaders::dds_loader::open_image_with_dds_fallback(&p).ok()?;

        // Reduce cache limit to 4 items to strictly bound RAM usage when using full-res 4K/8K images
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
        // If it's a VFX transparent texture, ignore black backgrounds
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
