// src/format_loaders/exr.rs

use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::{TonemapConfig, TonemapOperator};

pub struct ExrLoader;

impl ImageFormatLoader for ExrLoader {
    fn extensions(&self) -> &[&str] {
        &["exr", "ext_exr"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let (hdr_pixels, width, height) = crate::viewer::tonemapping::load_exr_rgba(path)
            .context("Failed to load EXR float pixels")?;

        let config = tonemap_config.unwrap_or(TonemapConfig {
            enabled: true,
            auto_exposure: true,
            operator: TonemapOperator::AcesFilmic,
        });

        let ldr_img = crate::viewer::tonemapping::tonemap_hdr_to_ldr_rgba(
            &hdr_pixels,
            width,
            height,
            config,
            1.0,
        )?;
        Ok(DynamicImage::ImageRgba8(ldr_img))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let mut w = 0;
        let mut h = 0;
        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();
        let mut compression_format = "EXR".to_string();
        let mut has_alpha = false;

        if let Ok(meta) = exr::prelude::MetaData::read_from_file(path, false)
            && let Some(header) = meta.headers.first()
        {
            w = header.shared_attributes.display_window.size.width() as u32;
            h = header.shared_attributes.display_window.size.height() as u32;
            bit_depth = 16;
            color_space = "Linear".to_string();
            compression_format = format!("{:?}", header.compression);
            has_alpha = header
                .channels
                .list
                .iter()
                .any(|c| c.name.to_string() == "A" || c.name.to_string() == "alpha");
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "exr".to_string(),
            compression_format,
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "EXR", 1, false),
        })
    }
}
