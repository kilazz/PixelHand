// src/format_loaders/heic.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct HeicLoader;

/// Maps CICP color primaries code to human-readable color space string.
fn resolve_cicp_color_space(color_primaries: u16) -> String {
    match color_primaries {
        1 => "sRGB (BT.709)".to_string(),
        9 => "Rec.2020".to_string(),
        12 => "Display P3".to_string(),
        _ => "sRGB".to_string(),
    }
}

impl ImageFormatLoader for HeicLoader {
    fn extensions(&self) -> &[&str] {
        &["heic", "heif"]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read HEIC/HEIF file from disk")?;

        // Configure safe resource limits (up to 16384x16384 resolution)
        let mut limits = heic::Limits::default();
        limits.max_width = Some(16384);
        limits.max_height = Some(16384);
        limits.max_pixels = Some(256 * 1024 * 1024);

        let output = heic::DecoderConfig::new()
            .decode_request(&bytes)
            .with_output_layout(heic::PixelLayout::Rgba8)
            .with_limits(&limits)
            .decode()
            .map_err(|e| anyhow!("HEIC decoding failed: {:#}", e))?;

        let buffer = image::RgbaImage::from_raw(output.width, output.height, output.data)
            .ok_or_else(|| anyhow!("Failed to map raw HEIC pixel data buffer"))?;

        let mut img = DynamicImage::ImageRgba8(buffer);

        // Scale down thumbnail if target_size is requested
        if let Some(target) = target_size
            && (img.width() > target || img.height() > target)
        {
            img = img.resize(target, target, image::imageops::FilterType::Triangle);
        }

        Ok(img)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();

        // Obtain dimensions efficiently without reading the entire payload into RAM
        let (w, h) = if let Ok(dim) = imagesize::size(path) {
            (dim.width as u32, dim.height as u32)
        } else {
            (0, 0)
        };

        let mut has_alpha = false;
        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();

        // Read only the initial 64 KB header chunk to parse ISOBMFF/CICP properties
        if let Ok(file) = std::fs::File::open(path) {
            use std::io::Read;
            let mut handle = file.take(64 * 1024);
            let mut header_bytes = Vec::new();
            if handle.read_to_end(&mut header_bytes).is_ok()
                && let Ok(info) = heic::ImageInfo::from_bytes(&header_bytes)
            {
                has_alpha = info.has_alpha;
                bit_depth = info.bit_depth as u32;
                color_space = resolve_cicp_color_space(info.color_primaries);
            }
        }

        let estimated_vram = crate::qc::rules::estimate_vram(w, h, "HEIC", 1, false);

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "heic".to_string(),
            compression_format: "HEVC (HEIC/HEIF)".to_string(),
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram,
        })
    }
}
