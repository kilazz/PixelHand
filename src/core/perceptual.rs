// src/core/perceptual.rs
use image::{DynamicImage, RgbaImage};
use image_hasher::{HashAlg, HasherConfig};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub fn is_vfx_transparent_texture(rgba: &RgbaImage) -> bool {
    let mut max_alpha = 0u8;
    let mut max_rgb = 0u8;
    for pixel in rgba.pixels() {
        if pixel[3] > max_alpha {
            max_alpha = pixel[3];
        }
        max_rgb = max_rgb.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }
    max_alpha == 0 && max_rgb > 0
}

pub fn calculate_perceptual_hashes(
    path: &Path,
    analysis_type: AnalysisType,
    ignore_solid_channels: bool,
) -> Option<PerceptualHashes> {
    let img = crate::format_loaders::dds_loader::open_image_with_dds_fallback(path).ok()?;
    let resized_img = img.resize(128, 128, image::imageops::FilterType::Nearest);
    let rgba_img = resized_img.to_rgba8();

    let processed_image: DynamicImage = match analysis_type {
        AnalysisType::Luminance => DynamicImage::ImageLuma8(resized_img.to_luma8()),
        AnalysisType::R | AnalysisType::G | AnalysisType::B | AnalysisType::A => {
            let idx = match analysis_type {
                AnalysisType::R => 0,
                AnalysisType::G => 1,
                AnalysisType::B => 2,
                _ => 3,
            };
            let mut ch_buf = image::GrayImage::new(rgba_img.width(), rgba_img.height());

            let mut min_val = 255u8;
            let mut max_val = 0u8;

            for (x, y, pixel) in rgba_img.enumerate_pixels() {
                let v = pixel[idx];
                min_val = min_val.min(v);
                max_val = max_val.max(v);
                ch_buf.put_pixel(x, y, image::Luma([v]));
            }
            if ignore_solid_channels && (max_val - min_val < 5) {
                return None;
            }
            DynamicImage::ImageLuma8(ch_buf)
        }
        AnalysisType::Composite => {
            if is_vfx_transparent_texture(&rgba_img) || !rgba_img.pixels().any(|p| p[3] < 255) {
                DynamicImage::ImageRgb8(resized_img.to_rgb8())
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

pub fn calculate_hamming_distance(hash1_base64: &str, hash2_base64: &str) -> Option<u32> {
    let h1 = image_hasher::ImageHash::<Vec<u8>>::from_base64(hash1_base64).ok()?;
    let h2 = image_hasher::ImageHash::<Vec<u8>>::from_base64(hash2_base64).ok()?;
    Some(h1.dist(&h2))
}
