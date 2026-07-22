// src/format_loaders/crn.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::utils::image_processing::bgra_u32_to_rgba_bytes;
use crate::viewer::tonemapping::TonemapConfig;

pub struct CrnLoader;

pub fn decode_crn_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    let mut tex_info = texture2ddecoder::CrnTextureInfo::default();
    if !tex_info.crnd_get_texture_info(bytes, bytes.len() as u32) {
        return Err(anyhow!("Invalid Crunch file header or signature"));
    }

    let w = tex_info.width as usize;
    let h = tex_info.height as usize;

    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!(
            "Invalid or oversized Crunch dimensions: {}x{} (max 16384x16384)",
            w,
            h
        ));
    }

    let mut rgba_u32 = vec![0u32; w * h];

    // Try Unity Crunch first; if it fails, fallback to Standard Binomial Crunch
    if texture2ddecoder::decode_unity_crunch(bytes, w, h, &mut rgba_u32).is_err() {
        texture2ddecoder::decode_crunch(bytes, w, h, &mut rgba_u32).map_err(|e| {
            anyhow!(
                "Both Unity Crunch and Standard Crunch decoding failed: {:?}",
                e
            )
        })?;
    }

    let raw_bytes = bgra_u32_to_rgba_bytes(rgba_u32);

    let img = image::RgbaImage::from_raw(w as u32, h as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to build RGBA buffer from Crunch texture"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for CrnLoader {
    fn extensions(&self) -> &[&str] {
        &["crn"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read CRN file from disk")?;
        decode_crn_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;

        let mut tex_info = texture2ddecoder::CrnTextureInfo::default();
        if tex_info.crnd_get_texture_info(&bytes, bytes.len() as u32) {
            w = tex_info.width;
            h = tex_info.height;
            mips = tex_info.levels;
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "crn".to_string(),
            compression_format: "Crunch Compressed".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "DXT5", mips, false),
        })
    }
}
