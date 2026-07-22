// src/perceptual/hashing.rs

use image::{DynamicImage, RgbaImage};
use std::path::Path;

// ==========================================
// --- ANALYSIS TYPE ENUM -------------------
// ==========================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AnalysisType {
    Composite,
    Luminance,
    R,
    G,
    B,
    A,
}

// ==========================================
// --- IMAGE ANALYSIS MATHEMATICS ----------
// ==========================================

/// Detects if an image has a fully transparent alpha channel but contains
/// valid RGB visual data (common in game engines or packed textures).
pub fn is_vfx_transparent_texture(rgba: &RgbaImage) -> bool {
    let pixels = rgba.as_raw();

    let has_opaque_alpha = pixels.chunks_exact(4).any(|pixel| pixel[3] > 0);
    if has_opaque_alpha {
        return false;
    }

    pixels
        .chunks_exact(4)
        .any(|pixel| pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0)
}

/// Shared color-channel splitting and luminance processing logic.
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
            let raw_data = rgba_img.as_raw();

            let channel_bytes: Vec<u8> = raw_data.chunks_exact(4).map(|p| p[idx]).collect();

            if ignore_solid_channels {
                let mut min_val = 255u8;
                let mut max_val = 0u8;
                for &v in &channel_bytes {
                    min_val = min_val.min(v);
                    max_val = max_val.max(v);
                }
                if max_val - min_val < 5 {
                    return None;
                }
            }

            let ch_buf = image::GrayImage::from_raw(width, height, channel_bytes)?;
            DynamicImage::ImageLuma8(ch_buf)
        }
        AnalysisType::Composite => {
            if is_vfx_transparent_texture(&rgba_img) || !rgba_img.pixels().any(|p| p[3] < 255) {
                DynamicImage::ImageRgb8(img.to_rgb8())
            } else {
                let mut bg = RgbaImage::new(rgba_img.width(), rgba_img.height());
                let bg_slice: &mut [u8] = &mut bg;

                bg_slice
                    .chunks_exact_mut(4)
                    .zip(rgba_img.as_raw().chunks_exact(4))
                    .for_each(|(dst, src)| {
                        let alpha = src[3] as f32 / 255.0;
                        dst[0] = (src[0] as f32 * alpha) as u8;
                        dst[1] = (src[1] as f32 * alpha) as u8;
                        dst[2] = (src[2] as f32 * alpha) as u8;
                        dst[3] = 255;
                    });

                DynamicImage::ImageRgba8(bg)
            }
        }
    };
    Some(processed)
}

/// Computes the 64-bit dHash (Difference Hash) visual signature natively without external dependencies.
/// Resizes the image to 9x8 in Grayscale (9 columns yield 8 horizontal comparisons per row across 8 rows = 64 bits).
pub fn calculate_perceptual_hash(
    path: &Path,
    analysis_type: AnalysisType,
    ignore_solid_channels: bool,
) -> Option<u64> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(128), None).ok()?;

    // 1. Resize to 9x8 dimensions for horizontal pixel comparison
    let resized_img = img.resize_exact(9, 8, image::imageops::FilterType::Triangle);
    let processed_image =
        preprocess_image_channels(&resized_img, analysis_type, ignore_solid_channels)?;
    let luma = processed_image.to_luma8();

    let mut hash: u64 = 0;
    let mut bit_index = 0;

    // 2. Compare adjacent horizontal pixels to construct the 64-bit register signature
    for y in 0..8 {
        for x in 0..8 {
            let left = luma.get_pixel(x, y)[0];
            let right = luma.get_pixel(x + 1, y)[0];
            if left > right {
                hash |= 1u64 << bit_index;
            }
            bit_index += 1;
        }
    }

    Some(hash)
}

/// Hardware-accelerated Hamming distance calculated via hardware `POPCNT` instruction in 1 clock cycle.
#[inline(always)]
pub fn calculate_hamming_distance(h1: u64, h2: u64) -> u32 {
    (h1 ^ h2).count_ones()
}
