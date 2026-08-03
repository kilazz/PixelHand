// src/format_loaders/atc.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::compression::gpu_block::{TextureBlockFormat, decode_gpu_block};
use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct AtcLoader;

fn parse_atc_header(bytes: &[u8]) -> Option<(usize, usize, bool, usize)> {
    if bytes.len() >= 16 && &bytes[0..4] == b"ATC " {
        let width = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let height = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let format_flag = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let is_rgba8 = format_flag != 0;
        if width > 0 && height > 0 && width <= 16384 && height <= 16384 {
            return Some((width, height, is_rgba8, 16));
        }
    }
    None
}

fn infer_atc_dimensions(payload_len: usize) -> Option<(usize, usize, bool)> {
    if payload_len == 0 {
        return None;
    }

    let side_rgba8 = (payload_len as f64).sqrt() as usize;
    if side_rgba8 > 0 && side_rgba8.is_multiple_of(4) && side_rgba8 * side_rgba8 == payload_len {
        return Some((side_rgba8, side_rgba8, true));
    }

    let total_pixels_rgb4 = payload_len * 2;
    let side_rgb4 = (total_pixels_rgb4 as f64).sqrt() as usize;
    if side_rgb4 > 0 && side_rgb4.is_multiple_of(4) && side_rgb4 * side_rgb4 == total_pixels_rgb4 {
        return Some((side_rgb4, side_rgb4, false));
    }

    let h_rgba8 = ((payload_len / 2) as f64).sqrt() as usize;
    if h_rgba8 > 0 && h_rgba8.is_multiple_of(4) && (h_rgba8 * 2) * h_rgba8 == payload_len {
        return Some((h_rgba8 * 2, h_rgba8, true));
    }

    let h_rgb4 = ((total_pixels_rgb4 / 2) as f64).sqrt() as usize;
    if h_rgb4 > 0 && h_rgb4.is_multiple_of(4) && (h_rgb4 * 2) * h_rgb4 == total_pixels_rgb4 {
        return Some((h_rgb4 * 2, h_rgb4, false));
    }

    None
}

pub fn decode_atc_bytes(bytes: &[u8], _path: &Path) -> Result<DynamicImage> {
    let (w, h, is_rgba8, offset) =
        if let Some((width, height, rgba, header_size)) = parse_atc_header(bytes) {
            (width, height, rgba, header_size)
        } else if let Some((width, height, rgba)) = infer_atc_dimensions(bytes.len()) {
            (width, height, rgba, 0)
        } else {
            return Err(anyhow!("Failed to resolve ATC texture dimensions"));
        };

    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!("Invalid or oversized ATC dimensions: {}x{}", w, h));
    }

    let payload = &bytes[offset..];
    let format = if is_rgba8 {
        TextureBlockFormat::AtcRgba8
    } else {
        TextureBlockFormat::AtcRgb4
    };

    decode_gpu_block(payload, w, h, format)
}

impl ImageFormatLoader for AtcLoader {
    fn extensions(&self) -> &[&str] {
        &["atc"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read ATC file from disk")?;
        decode_atc_bytes(&bytes, path)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let (w, h) = if let Some((width, height, _, _)) = parse_atc_header(&bytes) {
            (width as u32, height as u32)
        } else if let Some((width, height, _)) = infer_atc_dimensions(bytes.len()) {
            (width as u32, height as u32)
        } else {
            (0, 0)
        };

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "atc".to_string(),
            compression_format: "Qualcomm ATC".to_string(),
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC3", 1, false),
        })
    }
}
