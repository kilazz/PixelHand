// src/format_loaders/psd.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct PsdLoader;

impl ImageFormatLoader for PsdLoader {
    fn extensions(&self) -> &[&str] {
        &["psd"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read PSD file")?;
        let psd = psd::Psd::from_bytes(&bytes).map_err(|e| anyhow!("PSD parsing failed: {}", e))?;
        let buffer = image::RgbaImage::from_raw(psd.width(), psd.height(), psd.rgba())
            .ok_or_else(|| anyhow!("Failed to compile PSD pixel buffer"))?;
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
            format_str: "psd".to_string(),
            compression_format: "PSD".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "PSD", 1, false),
        })
    }
}
