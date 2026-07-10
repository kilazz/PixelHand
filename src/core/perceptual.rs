// src/core/perceptual.rs

use image::{DynamicImage, RgbaImage};
use image_hasher::{HashAlg, HasherConfig};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AnalysisType {
    Composite,
    Luminance,
    R,
    G,
    B,
    A,
}

#[allow(dead_code)]
pub struct PerceptualHashes {
    pub dhash: String,
    pub phash: String,
}

/// Detects if an image has a fully transparent alpha channel but contains
/// valid RGB visual data (common in game engines/packed textures).
/// Processes sequentially to leverage CPU SIMD auto-vectorization on small sized blocks.
pub fn is_vfx_transparent_texture(rgba: &RgbaImage) -> bool {
    let pixels = rgba.as_raw();

    // Scan sequentially to avoid Rayon thread scheduling overhead on downscaled assets
    let has_opaque_alpha = pixels.chunks_exact(4).any(|pixel| pixel[3] > 0);
    if has_opaque_alpha {
        return false;
    }

    pixels
        .chunks_exact(4)
        .any(|pixel| pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0)
}

/// Shared color-channel splitting and luminance processing logic
/// for both Perceptual and AI scanning pipelines.
pub fn preprocess_image_channels(
    img: &DynamicImage,
    analysis_type: AnalysisType,
    ignore_solid_channels: bool,
) -> Option<DynamicImage> {
    let rgba_img = img.to_rgba8();

    let processed: DynamicImage = match analysis_type {
        AnalysisType::Luminance => DynamicImage::ImageLuma8(img.to_luma8()),
        AnalysisType::R | AnalysisType::G | AnalysisType::B | AnalysisType::A => {
            let idx = match analysis_type {
                AnalysisType::R => 0,
                AnalysisType::G => 1,
                AnalysisType::B => 2,
                _ => 3,
            };

            let width = rgba_img.width();
            let height = rgba_img.height();
            let mut raw_data = Vec::with_capacity((width * height) as usize);

            let mut min_val = 255u8;
            let mut max_val = 0u8;

            // Stream pixels linearly into flat memory avoiding coordinates multiplication overhead
            for pixel in rgba_img.pixels() {
                let v = pixel[idx];
                min_val = min_val.min(v);
                max_val = max_val.max(v);
                raw_data.push(v);
            }

            if ignore_solid_channels && (max_val - min_val < 5) {
                return None;
            }

            let ch_buf = image::GrayImage::from_raw(width, height, raw_data)?;
            DynamicImage::ImageLuma8(ch_buf)
        }
        AnalysisType::Composite => {
            if is_vfx_transparent_texture(&rgba_img) || !rgba_img.pixels().any(|p| p[3] < 255) {
                DynamicImage::ImageRgb8(img.to_rgb8())
            } else {
                let mut bg = RgbaImage::new(rgba_img.width(), rgba_img.height());
                for (x, y, pixel) in rgba_img.enumerate_pixels() {
                    let alpha = pixel[3] as f32 / 255.0;
                    let r = (pixel[0] as f32 * alpha) as u8;
                    let g = (pixel[1] as f32 * alpha) as u8;
                    let b = (pixel[2] as f32 * alpha) as u8;
                    bg.put_pixel(x, y, image::Rgba([r, g, b, 255]));
                }
                DynamicImage::ImageRgba8(bg)
            }
        }
    };
    Some(processed)
}

/// Computes both dHash and phash visual signatures for the target asset.
pub fn calculate_perceptual_hashes(
    path: &Path,
    analysis_type: AnalysisType,
    ignore_solid_channels: bool,
) -> Option<PerceptualHashes> {
    // Request a downscaled 128px mipmap during parsing to conserve CPU cycles and RAM
    let img =
        crate::format_loaders::dds_loader::open_image_with_dds_fallback(path, Some(128)).ok()?;
    let resized_img = img.resize(128, 128, image::imageops::FilterType::Nearest);

    let processed_image =
        preprocess_image_channels(&resized_img, analysis_type, ignore_solid_channels)?;

    let dhash_result = HasherConfig::new()
        .hash_alg(HashAlg::Gradient)
        .hash_size(8, 8)
        .to_hasher()
        .hash_image(&processed_image);
    let phash_result = HasherConfig::new()
        .hash_alg(HashAlg::Mean)
        .hash_size(8, 8)
        .to_hasher()
        .hash_image(&processed_image);

    Some(PerceptualHashes {
        dhash: dhash_result.to_base64(),
        phash: phash_result.to_base64(),
    })
}

/// Computes the bitwise Hamming distance between two base64-encoded visual hashes.
pub fn calculate_hamming_distance(hash1_base64: &str, hash2_base64: &str) -> Option<u32> {
    let h1 = image_hasher::ImageHash::<Vec<u8>>::from_base64(hash1_base64).ok()?;
    let h2 = image_hasher::ImageHash::<Vec<u8>>::from_base64(hash2_base64).ok()?;
    Some(h1.dist(&h2))
}
