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

static DECODED_CACHE: OnceLock<Mutex<HashMap<String, DecodedCacheItem>>> = OnceLock::new();

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
        // --- FIXED: Load 1:1 original full-resolution pixel buffers for split-pane inspects ---
        let img = crate::format_loaders::dds_loader::open_image_with_dds_fallback(&p, None).ok()?;

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
