// src/format_loaders/pvr.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct PvrLoader;

const PVR3_MAGIC: u32 = 0x03525650; // "PVR\x03" Little Endian

/// Decodes PowerVR (.pvr) container textures supporting PVR v3 and legacy PVR v2 standards.
pub fn decode_pvr_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if bytes.len() < 52 {
        return Err(anyhow!(
            "File is too small to be a valid PVR texture container"
        ));
    }

    let magic = u32::from_le_bytes(bytes[0..4].try_into()?);
    let width: usize;
    let height: usize;
    let payload: &[u8];
    let is_2bpp: bool;

    if magic == PVR3_MAGIC {
        let pixel_format = u64::from_le_bytes(bytes[8..16].try_into()?);
        height = u32::from_le_bytes(bytes[24..28].try_into()?) as usize;
        width = u32::from_le_bytes(bytes[28..32].try_into()?) as usize;
        let meta_size = u32::from_le_bytes(bytes[48..52].try_into()?) as usize;
        let data_offset = 52 + meta_size;

        if data_offset > bytes.len() || width == 0 || height == 0 || width > 16384 || height > 16384
        {
            return Err(anyhow!("Invalid or oversized PVR v3 dimensions"));
        }

        payload = &bytes[data_offset..];
        is_2bpp = pixel_format == 0 || pixel_format == 2;
    } else if bytes.len() >= 52 && &bytes[44..48] == b"PVRT" {
        height = u32::from_le_bytes(bytes[0..4].try_into()?) as usize;
        width = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
        let pvr_format = bytes[16];

        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(anyhow!("Invalid or oversized PVR v2 dimensions"));
        }

        payload = &bytes[52..];
        is_2bpp = pvr_format == 0x0c || pvr_format == 0x18;
    } else {
        return Err(anyhow!("Invalid PVR header magic bytes identifier"));
    }

    let mut rgba_u32 = vec![0u32; width * height];

    // True CPU PVRTC Decompression
    texture2ddecoder::decode_pvrtc(payload, width, height, &mut rgba_u32, is_2bpp)
        .map_err(|e| anyhow!("PVRTC decompression failed: {:?}", e))?;

    let mut raw_bytes: Vec<u8> = rgba_u32.into_iter().flat_map(|p| p.to_le_bytes()).collect();

    // Convert BGRA from texture2ddecoder into RGBA for Rust image crate
    for chunk in raw_bytes.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    let img = image::RgbaImage::from_raw(width as u32, height as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile PVR RGBA buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

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
        let bytes = std::fs::read(path).context("Failed to read PVR file from disk")?;
        decode_pvr_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;

        if bytes.len() >= 52
            && u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4])) == PVR3_MAGIC
        {
            h = u32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0; 4]));
            w = u32::from_le_bytes(bytes[28..32].try_into().unwrap_or([0; 4]));
            mips = u32::from_le_bytes(bytes[44..48].try_into().unwrap_or([1; 4])).max(1);
        } else if bytes.len() >= 52 && &bytes[44..48] == b"PVRT" {
            h = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4]));
            w = u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4]));
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
            mipmap_count: mips,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "BC1", mips, false),
        })
    }
}
