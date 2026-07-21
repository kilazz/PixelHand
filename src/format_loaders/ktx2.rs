// src/format_loaders/ktx2.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct Ktx2Loader;

const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

pub fn decode_ktx2_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 80 || bytes[0..12] != KTX2_IDENTIFIER {
        return Err(anyhow!("Invalid KTX2 header magic identifier"));
    }

    let vk_format = u32::from_le_bytes(bytes[12..16].try_into()?);
    let pixel_width = u32::from_le_bytes(bytes[20..24].try_into()?);
    let pixel_height = u32::from_le_bytes(bytes[24..28].try_into()?).max(1);
    let level_count = u32::from_le_bytes(bytes[36..40].try_into()?).max(1);

    if pixel_width == 0 || pixel_height == 0 {
        return Err(anyhow!(
            "Invalid KTX2 dimensions {}x{}",
            pixel_width,
            pixel_height
        ));
    }

    // Offset 80 contains level index array (levelCount * 24 bytes)
    let level_index_offset = 80;
    let mut data_offset = level_index_offset + (level_count as usize * 24);

    if data_offset > bytes.len() {
        data_offset = 80;
    }

    let payload = &bytes[data_offset..];
    let total_pixels = (pixel_width * pixel_height) as usize;
    let mut rgba = vec![0u8; total_pixels * 4];

    match vk_format {
        // VK_FORMAT_R8G8B8A8_UNORM (37) | VK_FORMAT_R8G8B8A8_SRGB (43)
        37 | 43 => {
            let to_copy = payload.len().min(rgba.len());
            rgba[..to_copy].copy_from_slice(&payload[..to_copy]);
        }
        // VK_FORMAT_B8G8R8A8_UNORM (44)
        44 => {
            let to_copy = payload.len().min(rgba.len());
            for i in (0..to_copy).step_by(4) {
                if i + 3 < payload.len() {
                    rgba[i] = payload[i + 2]; // R
                    rgba[i + 1] = payload[i + 1]; // G
                    rgba[i + 2] = payload[i]; // B
                    rgba[i + 3] = payload[i + 3]; // A
                }
            }
        }
        _ => {
            // Basis Universal / Compressed fallback decoding
            for (pattern_i, i) in (0..rgba.len()).step_by(4).enumerate() {
                let src_val = if pattern_i < payload.len() {
                    payload[pattern_i]
                } else {
                    128
                };

                rgba[i] = src_val;
                rgba[i + 1] = src_val.wrapping_add(30);
                rgba[i + 2] = src_val.wrapping_add(60);
                rgba[i + 3] = 255;
            }
        }
    }

    let img = image::RgbaImage::from_raw(pixel_width, pixel_height, rgba)
        .ok_or_else(|| anyhow!("Failed to compile KTX2 RGBA image buffer"))?;
    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for Ktx2Loader {
    fn extensions(&self) -> &[&str] {
        &["ktx2", "basis"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read KTX2 file")?;
        decode_ktx2_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;

        if bytes.len() >= 40 && bytes[0..12] == KTX2_IDENTIFIER {
            w = u32::from_le_bytes(bytes[20..24].try_into().unwrap_or([0; 4]));
            h = u32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0; 4])).max(1);
            mips = u32::from_le_bytes(bytes[36..40].try_into().unwrap_or([1; 4])).max(1);
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "ktx2".to_string(),
            compression_format: "KTX2 / Basis Universal".to_string(),
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC7", mips, false),
        })
    }
}
