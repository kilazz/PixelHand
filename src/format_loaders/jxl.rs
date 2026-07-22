// src/format_loaders/jxl.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct JxlLoader;

impl ImageFormatLoader for JxlLoader {
    fn extensions(&self) -> &[&str] {
        &["jxl"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read JXL file")?;
        let decoder = jxl_oxide::integration::JxlDecoder::new(std::io::Cursor::new(bytes))
            .map_err(|e| anyhow!("JXL decoder initialization failed: {:?}", e))?;
        let img = DynamicImage::from_decoder(decoder)
            .map_err(|e| anyhow!("JXL decoding pipeline failed: {:?}", e))?;
        Ok(img)
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
            format_str: "jxl".to_string(),
            compression_format: "JXL".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: false,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "JXL", 1, false),
        })
    }
}
