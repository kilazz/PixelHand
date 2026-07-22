// src/format_loaders/qoi.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct QoiLoader;

/// Decodes QOI raw bytes specification natively in zero-allocation pure Rust.
pub fn decode_qoi_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 14 {
        return Err(anyhow!("File too small to be a valid QOI image"));
    }
    if &bytes[0..4] != b"qoif" {
        return Err(anyhow!("Invalid QOI magic header bytes"));
    }

    let width = u32::from_be_bytes(bytes[4..8].try_into()?);
    let height = u32::from_be_bytes(bytes[8..12].try_into()?);

    // OOM Safety Check: Prevent memory allocation crashes on corrupted/oversized image headers
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(anyhow!(
            "Invalid or oversized QOI dimensions: {}x{} (max 16384x16384)",
            width,
            height
        ));
    }

    let _channels = bytes[12];
    let _colorspace = bytes[13];

    let total_pixels = (width as usize) * (height as usize);
    let mut rgba_buf = vec![0u8; total_pixels * 4];

    let mut index = [[0u8; 4]; 64];
    let mut px = [0u8, 0u8, 0u8, 255u8];

    let mut p = 14;
    let chunks_len = bytes.len().saturating_sub(8); // Exclude 8-byte end marker
    let mut run = 0;

    for out_idx in 0..total_pixels {
        if run > 0 {
            run -= 1;
        } else if p < chunks_len {
            let b1 = bytes[p];
            p += 1;

            if b1 == 0b11111111 {
                // QOI_OP_RGBA
                if p + 4 <= bytes.len() {
                    px[0] = bytes[p];
                    px[1] = bytes[p + 1];
                    px[2] = bytes[p + 2];
                    px[3] = bytes[p + 3];
                    p += 4;
                }
            } else if b1 == 0b11111110 {
                // QOI_OP_RGB
                if p + 3 <= bytes.len() {
                    px[0] = bytes[p];
                    px[1] = bytes[p + 1];
                    px[2] = bytes[p + 2];
                    p += 3;
                }
            } else if (b1 & 0b11000000) == 0b00000000 {
                // QOI_OP_INDEX
                let idx = (b1 & 0b00111111) as usize;
                px = index[idx];
            } else if (b1 & 0b11000000) == 0b01000000 {
                // QOI_OP_DIFF
                let dr = ((b1 >> 4) & 0x03).wrapping_sub(2);
                let dg = ((b1 >> 2) & 0x03).wrapping_sub(2);
                let db = (b1 & 0x03).wrapping_sub(2);
                px[0] = px[0].wrapping_add(dr);
                px[1] = px[1].wrapping_add(dg);
                px[2] = px[2].wrapping_add(db);
            } else if (b1 & 0b11000000) == 0b10000000 {
                // QOI_OP_LUMA
                if p < bytes.len() {
                    let b2 = bytes[p];
                    p += 1;
                    let vg = (b1 & 0b00111111).wrapping_sub(32);
                    let dr = vg.wrapping_add(((b2 >> 4) & 0x0f).wrapping_sub(8));
                    let db = vg.wrapping_add((b2 & 0x0f).wrapping_sub(8));
                    px[0] = px[0].wrapping_add(dr);
                    px[1] = px[1].wrapping_add(vg);
                    px[2] = px[2].wrapping_add(db);
                }
            } else if (b1 & 0b11000000) == 0b11000000 {
                // QOI_OP_RUN
                run = b1 & 0b00111111;
            }

            let index_pos = (px[0] as usize * 3
                + px[1] as usize * 5
                + px[2] as usize * 7
                + px[3] as usize * 11)
                % 64;
            index[index_pos] = px;
        }

        let dst = out_idx * 4;
        rgba_buf[dst..dst + 4].copy_from_slice(&px);
    }

    let img = image::RgbaImage::from_raw(width, height, rgba_buf)
        .ok_or_else(|| anyhow!("Failed to compile QOI RGBA pixel buffer"))?;
    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for QoiLoader {
    fn extensions(&self) -> &[&str] {
        &["qoi"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read QOI file")?;
        decode_qoi_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut has_alpha = false;

        if bytes.len() >= 14 && &bytes[0..4] == b"qoif" {
            w = u32::from_be_bytes(bytes[4..8].try_into().unwrap_or([0; 4]));
            h = u32::from_be_bytes(bytes[8..12].try_into().unwrap_or([0; 4]));
            let channels = bytes[12];
            has_alpha = channels == 4;
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "qoi".to_string(),
            compression_format: "QOI (Quite OK Image)".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "UNCOMPRESSED", 1, false),
        })
    }
}
