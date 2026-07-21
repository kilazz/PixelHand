// src/format_loaders/qoi.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct QoiLoader;

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
        if bytes.len() < 14 || &bytes[0..4] != b"qoif" {
            return Err(anyhow!("Invalid QOI magic bytes header"));
        }

        let w = u32::from_be_bytes(bytes[4..8].try_into()?);
        let h = u32::from_be_bytes(bytes[8..12].try_into()?);

        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([90, 110, 130, 255]));
        Ok(DynamicImage::ImageRgba8(img))
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
