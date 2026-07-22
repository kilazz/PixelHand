// src/format_loaders/atc.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct AtcLoader;

/// Decodes Adreno Texture Compression (ATC_RGB4 / ATC_RGBA8) files natively on CPU via texture2ddecoder.
pub fn decode_atc_bytes(bytes: &[u8], path: &Path) -> Result<DynamicImage> {
    let (w, h) = if let Ok(dim) = imagesize::size(path) {
        (dim.width, dim.height)
    } else {
        return Err(anyhow!("Failed to read ATC image dimensions"));
    };

    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!("Invalid or oversized ATC dimensions: {}x{}", w, h));
    }

    let mut rgba_u32 = vec![0u32; w * h];

    // Determine whether payload is ATC_RGB4 (8 bytes/block) or ATC_RGBA8 (16 bytes/block) based on buffer size
    let blocks_x = w.div_ceil(4);
    let blocks_y = h.div_ceil(4);
    let total_blocks = blocks_x * blocks_y;

    if bytes.len() >= total_blocks * 16 {
        texture2ddecoder::decode_atc_rgba8(bytes, w, h, &mut rgba_u32)
            .map_err(|e| anyhow!("ATC RGBA8 decoding failed: {:?}", e))?;
    } else {
        texture2ddecoder::decode_atc_rgb4(bytes, w, h, &mut rgba_u32)
            .map_err(|e| anyhow!("ATC RGB4 decoding failed: {:?}", e))?;
    }

    let mut raw_bytes: Vec<u8> = rgba_u32.into_iter().flat_map(|p| p.to_le_bytes()).collect();

    // Convert BGRA from texture2ddecoder into RGBA for Rust image crate
    for chunk in raw_bytes.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    let img = image::RgbaImage::from_raw(w as u32, h as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile ATC RGBA buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
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
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));

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
