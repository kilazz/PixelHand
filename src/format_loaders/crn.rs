// src/format_loaders/crn.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct CrnLoader;

pub fn decode_crn_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    let mut tex_info = texture2ddecoder::CrnTextureInfo::default();
    if !tex_info.crnd_get_texture_info(bytes, bytes.len() as u32) {
        return Err(anyhow!("Invalid Crunch file header"));
    }

    let w = tex_info.width as usize;
    let h = tex_info.height as usize;

    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!(
            "Invalid or oversized Crunch dimensions: {}x{}",
            w,
            h
        ));
    }

    let mut rgba_u32 = vec![0u32; w * h];

    // Распаковка Unity Crunch текстур
    texture2ddecoder::decode_unity_crunch(bytes, w, h, &mut rgba_u32)
        .map_err(|e| anyhow!("Unity Crunch decoding failed: {:?}", e))?;

    let mut raw_bytes: Vec<u8> = rgba_u32.into_iter().flat_map(|p| p.to_le_bytes()).collect();
    for chunk in raw_bytes.chunks_exact_mut(4) {
        chunk.swap(0, 2); // BGRA -> RGBA
    }

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
        let bytes = std::fs::read(path).context("Failed to read CRN file")?;
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
            compression_format: "Unity Crunch Compressed".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "DXT5", mips, false),
        })
    }
}
