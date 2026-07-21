// src/format_loaders/pvr.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct PvrLoader;

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
        let bytes = std::fs::read(path).context("Failed to read PVR file")?;
        if bytes.len() < 52 {
            return Err(anyhow!("File too small to be a valid PVR container"));
        }

        let h = u32::from_le_bytes(bytes[24..28].try_into()?);
        let w = u32::from_le_bytes(bytes[28..32].try_into()?);

        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([100, 90, 110, 255]));
        Ok(DynamicImage::ImageRgba8(img))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;

        if bytes.len() >= 52 && &bytes[0..4] == b"PVR\x03" {
            h = u32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0; 4]));
            w = u32::from_le_bytes(bytes[28..32].try_into().unwrap_or([0; 4]));
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
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC1", 1, false),
        })
    }
}
