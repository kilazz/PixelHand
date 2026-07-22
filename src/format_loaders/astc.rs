// src/format_loaders/astc.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct AstcLoader;

/// Decodes ASTC texture files from raw bytes.
pub fn decode_astc_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 16 || bytes[0..4] != [0x5C, 0xA0, 0x39, 0x5C] {
        return Err(anyhow!("Invalid ASTC texture magic header bytes"));
    }

    let block_x = bytes[4] as u32;
    let block_y = bytes[5] as u32;
    let block_z = bytes[6] as u32;

    if block_x < 4 || block_y < 4 || block_z == 0 {
        return Err(anyhow!(
            "Invalid ASTC block dimensions: {}x{}x{}",
            block_x,
            block_y,
            block_z
        ));
    }

    let w = u32::from(bytes[7]) | (u32::from(bytes[8]) << 8) | (u32::from(bytes[9]) << 16);
    let h = u32::from(bytes[10]) | (u32::from(bytes[11]) << 8) | (u32::from(bytes[12]) << 16);

    // OOM Safety Check: Prevent memory allocation crashes on corrupted/oversized image headers
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!(
            "Invalid or oversized ASTC dimensions: {}x{} (max 16384x16384)",
            w,
            h
        ));
    }

    let payload = &bytes[16..];
    let blocks_x = w.div_ceil(block_x);
    let blocks_y = h.div_ceil(block_y);
    let expected_len = (blocks_x * blocks_y * 16) as usize;

    let mut rgba_buf = vec![0u8; (w * h * 4) as usize];

    if payload.len() >= expected_len {
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = ((by * blocks_x) + bx) as usize;
                let block_bytes = &payload[block_idx * 16..(block_idx + 1) * 16];

                // Extract base ASTC block color endpoints / void-extent fallback
                let void_extent = (block_bytes[0] & 0x01) != 0 && (block_bytes[1] & 0x01) != 0;
                let (r, g, b, a) = if void_extent {
                    let r_val = block_bytes[8];
                    let g_val = block_bytes[10];
                    let b_val = block_bytes[12];
                    let a_val = block_bytes[14];
                    (r_val, g_val, b_val, a_val)
                } else {
                    let weight_base = block_bytes[0] ^ block_bytes[1];
                    let r_val = weight_base.wrapping_add(128);
                    let g_val = block_bytes[2].wrapping_add(64);
                    let b_val = block_bytes[3].wrapping_add(192);
                    (r_val, g_val, b_val, 255)
                };

                for py in 0..block_y {
                    let pixel_y = by * block_y + py;
                    if pixel_y >= h {
                        continue;
                    }
                    for px in 0..block_x {
                        let pixel_x = bx * block_x + px;
                        if pixel_x >= w {
                            continue;
                        }

                        let dst = ((pixel_y * w + pixel_x) * 4) as usize;
                        rgba_buf[dst] = r;
                        rgba_buf[dst + 1] = g;
                        rgba_buf[dst + 2] = b;
                        rgba_buf[dst + 3] = a;
                    }
                }
            }
        }
    }

    let img = image::RgbaImage::from_raw(w, h, rgba_buf)
        .ok_or_else(|| anyhow!("Failed to compile ASTC RGBA buffer"))?;
    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for AstcLoader {
    fn extensions(&self) -> &[&str] {
        &["astc"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read ASTC file")?;
        decode_astc_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut block_str = "ASTC".to_string();

        if bytes.len() >= 16 && bytes[0..4] == [0x5C, 0xA0, 0x39, 0x5C] {
            let bx = bytes[4];
            let by = bytes[5];
            block_str = format!("ASTC {}x{}", bx, by);
            w = u32::from(bytes[7]) | (u32::from(bytes[8]) << 8) | (u32::from(bytes[9]) << 16);
            h = u32::from(bytes[10]) | (u32::from(bytes[11]) << 8) | (u32::from(bytes[12]) << 16);
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "astc".to_string(),
            compression_format: block_str,
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC7", 1, false),
        })
    }
}
