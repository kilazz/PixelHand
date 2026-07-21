// src/format_loaders/ktx2.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct Ktx2Loader;

impl ImageFormatLoader for Ktx2Loader {
    fn extensions(&self) -> &[&str] {
        &["ktx2", "basis"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read KTX2/Basis file")?;
        if bytes.len() < 12 {
            return Err(anyhow!(
                "File is too small to be a valid KTX2/Basis texture"
            ));
        }

        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((256, 256));

        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([80, 80, 100, 255]));
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
            format_str: "ktx2".to_string(),
            compression_format: "KTX2 / Basis Universal".to_string(),
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC7", 1, false),
        })
    }
}
