// src/format_loaders/svg.rs

use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct SvgLoader;

impl ImageFormatLoader for SvgLoader {
    fn extensions(&self) -> &[&str] {
        &["svg"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let _bytes = std::fs::read(path).context("Failed to read SVG file")?;
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((512, 512));

        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([140, 140, 160, 255]));
        Ok(DynamicImage::ImageRgba8(img))
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
            format_str: "svg".to_string(),
            compression_format: "SVG Vector".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "UNCOMPRESSED", 1, false),
        })
    }
}
