// src/format_loaders/heic.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct HeicLoader;

impl ImageFormatLoader for HeicLoader {
    fn extensions(&self) -> &[&str] {
        &["heic", "heif"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read HEIC/HEIF file")?;
        let output = heic::DecoderConfig::new()
            .decode(&bytes, heic::PixelLayout::Rgba8)
            .map_err(|e| anyhow!("HEIC decoding failed: {:?}", e))?;
        let buffer =
            image::RgbaImage::from_raw(output.width as u32, output.height as u32, output.data)
                .ok_or_else(|| anyhow!("Failed to map raw HEIC pixel data buffer"))?;
        Ok(DynamicImage::ImageRgba8(buffer))
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
            format_str: "heic".to_string(),
            compression_format: "HEIC/HEIF".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: false,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "HEIC", 1, false),
        })
    }
}
