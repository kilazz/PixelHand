// src/format_loaders/standard.rs

use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::{TonemapConfig, TonemapOperator};

pub struct StandardLoader;

impl ImageFormatLoader for StandardLoader {
    fn extensions(&self) -> &[&str] {
        &[
            "png", "jpg", "jpeg", "tga", "bmp", "hdr", "tif", "tiff", "webp", "gif", "avif",
        ]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let img = image::open(path).context("Failed to decode standard image format")?;

        if ext == "hdr"
            || matches!(
                img.color(),
                image::ColorType::Rgb32F | image::ColorType::Rgba32F
            )
        {
            let float_img = img.to_rgba32f();
            let width = float_img.width();
            let height = float_img.height();
            let hdr_pixels = float_img.into_raw();

            let config = tonemap_config.unwrap_or(TonemapConfig {
                enabled: true,
                auto_exposure: true,
                operator: TonemapOperator::AcesFilmic,
            });

            // Map HDR to LDR using the unified tonemapping module
            let ldr_img = crate::viewer::tonemapping::tonemap_hdr_to_ldr_rgba(
                &hdr_pixels,
                width,
                height,
                config,
                1.0,
            )?;
            Ok(DynamicImage::ImageRgba8(ldr_img))
        } else {
            Ok(DynamicImage::ImageRgba8(img.to_rgba8()))
        }
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));

        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();
        let mut compression_format = ext.to_uppercase();
        let mut has_alpha = false;

        if ext == "hdr" {
            bit_depth = 32;
            color_space = "Linear".to_string();
            compression_format = "Radiance HDR".to_string();
        } else if let Ok(img) = image::open(path) {
            let color = img.color();
            has_alpha = color.has_alpha();
            bit_depth = color.bits_per_pixel() as u32 / if has_alpha { 4 } else { 3 };
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: ext.clone(),
            compression_format,
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            // Calculate predicted GPU memory footprint using the general QC domain rules
            estimated_vram: crate::qc::rules::estimate_vram(w, h, &ext, 1, false),
        })
    }
}
