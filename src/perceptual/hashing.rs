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
// --- HARDWARE SIMD INTRINSICS (AVX2/SSE2) -
// ==========================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dhash(left: &[u8; 64], right: &[u8; 64]) -> u64 {
    use std::arch::x86_64::*;

    // Explicit unsafe block required by Rust 2024 safety lints
    unsafe {
        // Load 32-byte chunks into AVX2 registers
        let l0 = _mm256_loadu_si256(left.as_ptr() as *const __m256i);
        let r0 = _mm256_loadu_si256(right.as_ptr() as *const __m256i);
        // Compare greater-than (0xFF if left > right, else 0x00)
        let cmp0 = _mm256_cmpgt_epi8(l0, r0);
        // Extract sign bits directly into a 32-bit mask integer
        let mask0 = _mm256_movemask_epi8(cmp0) as u32 as u64;

        let l1 = _mm256_loadu_si256(left.as_ptr().add(32) as *const __m256i);
        let r1 = _mm256_loadu_si256(right.as_ptr().add(32) as *const __m256i);
        let cmp1 = _mm256_cmpgt_epi8(l1, r1);
        let mask1 = _mm256_movemask_epi8(cmp1) as u32 as u64;

        mask0 | (mask1 << 32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_dhash(left: &[u8; 64], right: &[u8; 64]) -> u64 {
    use std::arch::x86_64::*;
    let mut hash = 0u64;
    for i in 0..4 {
        unsafe {
            let l = _mm_loadu_si128(left.as_ptr().add(i * 16) as *const __m128i);
            let r = _mm_loadu_si128(right.as_ptr().add(i * 16) as *const __m128i);
            let cmp = _mm_cmpgt_epi8(l, r);
            let mask = (_mm_movemask_epi8(cmp) as u32) & 0xFFFF;
            hash |= (mask as u64) << (i * 16);
        }
    }
    hash
}

/// Fallback scalar implementation for non-x86 hardware
fn scalar_dhash(left: &[u8; 64], right: &[u8; 64]) -> u64 {
    let mut hash = 0u64;
    for i in 0..64 {
        if left[i] > right[i] {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Computes a 64-bit dHash using target feature runtime detection for optimal CPU vectorization.
pub fn compute_dhash_64(luma: &image::GrayImage) -> u64 {
    let raw = luma.as_raw();
    if raw.len() < 72 {
        return 0;
    }

    let mut left = [0u8; 64];
    let mut right = [0u8; 64];

    // Rearrange 9x8 matrix rows into contiguous 64-byte comparison buffers
    for y in 0..8 {
        let offset = y * 9;
        left[y * 8..y * 8 + 8].copy_from_slice(&raw[offset..offset + 8]);
        right[y * 8..y * 8 + 8].copy_from_slice(&raw[offset + 1..offset + 9]);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_dhash(&left, &right) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { sse2_dhash(&left, &right) };
        }
    }

    scalar_dhash(&left, &right)
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

            // Early Exit loop avoids creating heap allocation for solid channels
            if ignore_solid_channels {
                let mut min_val = 255u8;
                let mut max_val = 0u8;
                let mut is_solid = true;
                for chunk in raw_data.chunks_exact(4) {
                    let v = chunk[idx];
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                    if max_val - min_val >= 5 {
                        is_solid = false;
                        break; // Stop immediately as soon as variance is found!
                    }
                }
                if is_solid {
                    return None;
                }
            }

            let channel_bytes: Vec<u8> = raw_data.chunks_exact(4).map(|p| p[idx]).collect();
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

/// Computes the 64-bit dHash (Difference Hash) visual signature natively using hardware SIMD (AVX2/SSE2).
pub fn calculate_perceptual_hash(
    path: &Path,
    analysis_type: AnalysisType,
    ignore_solid_channels: bool,
) -> Option<u64> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(128), None).ok()?;

    // Resize to 9x8 dimensions for horizontal pixel comparison
    let resized_img = img.resize_exact(9, 8, image::imageops::FilterType::Triangle);
    let processed_image =
        preprocess_image_channels(&resized_img, analysis_type, ignore_solid_channels)?;
    let luma = processed_image.to_luma8();

    // Hardware SIMD accelerated 64-bit comparison
    Some(compute_dhash_64(&luma))
}

/// Hardware-accelerated Hamming distance calculated via hardware `POPCNT` instruction in 1 clock cycle.
#[inline(always)]
pub fn calculate_hamming_distance(h1: u64, h2: u64) -> u32 {
    (h1 ^ h2).count_ones()
}
