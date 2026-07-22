// src/format_loaders/astc.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::utils::image_processing::bgra_u32_to_rgba_bytes;
use crate::viewer::tonemapping::TonemapConfig;

pub struct AstcLoader;

/// Decodes ASTC compressed textures natively on CPU via pure-Rust texture2ddecoder.
pub fn decode_astc_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 16 || bytes[0..4] != [0x5C, 0xA0, 0x39, 0x5C] {
        return Err(anyhow!("Invalid ASTC texture magic header bytes"));
    }

    let block_x = bytes[4] as usize;
    let block_y = bytes[5] as usize;
    let block_z = bytes[6] as usize;

    if block_x < 4 || block_y < 4 || block_z == 0 {
        return Err(anyhow!(
            "Invalid ASTC block dimensions: {}x{}x{}",
            block_x,
            block_y,
            block_z
        ));
    }

    let w =
        (u32::from(bytes[7]) | (u32::from(bytes[8]) << 8) | (u32::from(bytes[9]) << 16)) as usize;
    let h = (u32::from(bytes[10]) | (u32::from(bytes[11]) << 8) | (u32::from(bytes[12]) << 16))
        as usize;

    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(anyhow!(
            "Invalid or oversized ASTC dimensions: {}x{} (max 16384x16384)",
            w,
            h
        ));
    }

    let payload = &bytes[16..];
    let mut rgba_u32 = vec![0u32; w * h];

    texture2ddecoder::decode_astc(payload, w, h, block_x, block_y, &mut rgba_u32)
        .map_err(|e| anyhow!("ASTC decoding failed: {:?}", e))?;

    let raw_bytes = bgra_u32_to_rgba_bytes(rgba_u32);

    let img = image::RgbaImage::from_raw(w as u32, h as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile ASTC RGBA buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

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
        decode_astc_bytes(&bytes)
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
