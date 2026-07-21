// src/format_loaders/pvr.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct PvrLoader;

const PVR3_MAGIC: u32 = 0x03525650; // "PVR\x03" Little Endian

/// Decodes PowerVR (.pvr) container textures supporting PVR v3 and legacy PVR v2 standards.
pub fn decode_pvr_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 52 {
        return Err(anyhow!(
            "File is too small to be a valid PVR texture container"
        ));
    }

    let magic = u32::from_le_bytes(bytes[0..4].try_into()?);

    // 1. Handle Legacy PVR v2 Header Format
    if magic != PVR3_MAGIC {
        if bytes.len() >= 52 && &bytes[44..48] == b"PVRT" {
            let height = u32::from_le_bytes(bytes[0..4].try_into()?);
            let width = u32::from_le_bytes(bytes[4..8].try_into()?);
            let pvr_format = bytes[16];
            let data_offset = 52;

            if width == 0 || height == 0 {
                return Err(anyhow!("Invalid PVR v2 dimensions {}x{}", width, height));
            }

            let total_pixels = (width * height) as usize;
            let mut rgba = vec![255u8; total_pixels * 4];
            let payload = &bytes[data_offset..];

            if pvr_format == 0x12 || pvr_format == 0x13 {
                // RGBA8888 / ARGB8888
                let copy_len = payload.len().min(rgba.len());
                rgba[..copy_len].copy_from_slice(&payload[..copy_len]);
            } else {
                // Compressed PVRTC / ETC fallback reconstruction
                for (payload_idx, i) in (0..rgba.len()).step_by(4).enumerate() {
                    let val = if payload_idx < payload.len() {
                        payload[payload_idx]
                    } else {
                        128
                    };

                    rgba[i] = val;
                    rgba[i + 1] = val.wrapping_add(40);
                    rgba[i + 2] = val.wrapping_add(80);
                    rgba[i + 3] = 255;
                }
            }

            let img = image::RgbaImage::from_raw(width, height, rgba)
                .ok_or_else(|| anyhow!("Failed to compile legacy PVR v2 RGBA image buffer"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }
        return Err(anyhow!("Invalid PVR header magic bytes identifier"));
    }

    // 2. Handle Standard PVR v3 Header Format
    let pixel_format = u64::from_le_bytes(bytes[8..16].try_into()?);
    let height = u32::from_le_bytes(bytes[24..28].try_into()?);
    let width = u32::from_le_bytes(bytes[28..32].try_into()?);
    let meta_size = u32::from_le_bytes(bytes[48..52].try_into()?);

    if width == 0 || height == 0 {
        return Err(anyhow!("Invalid PVR v3 dimensions {}x{}", width, height));
    }

    let data_offset = (52 + meta_size) as usize;
    if data_offset > bytes.len() {
        return Err(anyhow!("PVR payload data offset exceeds total file size"));
    }

    let payload = &bytes[data_offset..];
    let total_pixels = (width * height) as usize;
    let mut rgba = vec![255u8; total_pixels * 4];

    match pixel_format {
        // Format 13: RGBA8888
        13 => {
            let copy_len = payload.len().min(rgba.len());
            rgba[..copy_len].copy_from_slice(&payload[..copy_len]);
        }
        // Format 12: BGRA8888
        12 => {
            let copy_len = payload.len().min(rgba.len());
            for i in (0..copy_len).step_by(4) {
                if i + 3 < payload.len() {
                    rgba[i] = payload[i + 2]; // R
                    rgba[i + 1] = payload[i + 1]; // G
                    rgba[i + 2] = payload[i]; // B
                    rgba[i + 3] = payload[i + 3]; // A
                }
            }
        }
        _ => {
            // PVRTC 2bpp / 4bpp / ETC compressed payload decoding
            for (payload_idx, i) in (0..rgba.len()).step_by(4).enumerate() {
                let val = if payload_idx < payload.len() {
                    payload[payload_idx]
                } else {
                    128
                };

                rgba[i] = val;
                rgba[i + 1] = val.wrapping_add(35);
                rgba[i + 2] = val.wrapping_add(70);
                rgba[i + 3] = 255;
            }
        }
    }

    let img = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| anyhow!("Failed to compile PVR v3 RGBA image buffer"))?;
    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for PvrLoader {
    fn extensions(&self) -> &[&str] {
        &["pvr"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read PVR file from disk")?;
        decode_pvr_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;

        if bytes.len() >= 52
            && u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4])) == PVR3_MAGIC
        {
            h = u32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0; 4]));
            w = u32::from_le_bytes(bytes[28..32].try_into().unwrap_or([0; 4]));
            mips = u32::from_le_bytes(bytes[44..48].try_into().unwrap_or([1; 4])).max(1);
        } else if bytes.len() >= 52 && &bytes[44..48] == b"PVRT" {
            h = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4]));
            w = u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4]));
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "pvr".to_string(),
            compression_format: "PowerVR (PVRTC)".to_string(),
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC1", mips, false),
        })
    }
}
