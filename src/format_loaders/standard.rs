// src/format_loaders/standard.rs

use anyhow::{Context, Result};
use image::metadata::Orientation;
use image::{ColorType, DynamicImage, ImageDecoder, ImageReader};
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::{TonemapConfig, TonemapOperator};

pub struct StandardLoader;

impl ImageFormatLoader for StandardLoader {
    fn extensions(&self) -> &[&str] {
        &[
            "png", "jpg", "jpeg", "tga", "bmp", "hdr", "tif", "tiff", "webp", "gif", "avif", "ico",
        ]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();

        // Open reader and automatically guess real image format by Magic Bytes signature
        let mut reader = ImageReader::open(path)
            .context("Failed to open image file")?
            .with_guessed_format()
            .context("Failed to determine image magic bytes signature")?;

        // Disable default 512MB RAM allocation limit for large production textures
        reader.no_limits();

        let mut decoder = reader
            .into_decoder()
            .context("Failed to construct image format decoder")?;

        // Extract EXIF orientation metadata natively supported in image 0.25.10
        let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);

        let mut img = DynamicImage::from_decoder(decoder)
            .context("Failed to decode standard image format payload")?;

        // Apply EXIF orientation transforms (auto-rotate/flip)
        img.apply_orientation(orientation);

        // Handle HDR / 32-bit Float image formats via the tonemapping pipeline
        if ext == "hdr" || matches!(img.color(), ColorType::Rgb32F | ColorType::Rgba32F) {
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
            img = DynamicImage::ImageRgba8(ldr_img);
        }

        //  Downscale thumbnail if target_size is requested
        if let Some(target) = target_size
            && (img.width() > target || img.height() > target)
        {
            img = img.resize(target, target, image::imageops::FilterType::Triangle);
        }

        Ok(img)
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
        } else if let Ok(reader) = ImageReader::open(path).and_then(|r| r.with_guessed_format())
            && let Ok(decoder) = reader.into_decoder()
        {
            let color = decoder.color_type();
            has_alpha = color.has_alpha();
            let bpp = color.bits_per_pixel();
            let channels = color.channel_count() as u16;
            bit_depth = bpp.checked_div(channels).map_or(8, |d| d as u32);
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
            estimated_vram: crate::qc::rules::estimate_vram(w, h, &ext, 1, false),
        })
    }
}
