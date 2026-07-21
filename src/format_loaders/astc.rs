// src/format_loaders/astc.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct AstcLoader;

impl ImageFormatLoader for AstcLoader {
    fn extensions(&self) -> &[&str] {
        &["astc"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read ASTC file")?;
        if bytes.len() < 16 || bytes[0..4] != [0x5C, 0xA0, 0x39, 0x5C] {
            return Err(anyhow!("Invalid ASTC texture magic bytes"));
        }

        let w = u32::from(bytes[7]) | (u32::from(bytes[8]) << 8) | (u32::from(bytes[9]) << 16);
        let h = u32::from(bytes[10]) | (u32::from(bytes[11]) << 8) | (u32::from(bytes[12]) << 16);

        let img = image::RgbaImage::from_pixel(w, h, image::Rgba([70, 90, 110, 255]));
        Ok(DynamicImage::ImageRgba8(img))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut block_str = "ASTC".to_string();

        if bytes.len() >= 16 && bytes[0..4] == [0x5C, 0xA0, 0x39, 0x5C] {
            let bx = bytes[4];
            let by = bytes[5];
            block_str = format!("ASTC {}x{}", bx, by);
            w = u32::from(bytes[7]) | (u32::from(bytes[8]) << 8) | (u32::from(bytes[9]) << 16);
            h = u32::from(bytes[10]) | (u32::from(bytes[11]) << 8) | (u32::from(bytes[12]) << 16);
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "astc".to_string(),
            compression_format: block_str,
            color_space: "Linear".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC7", 1, false),
        })
    }
}
