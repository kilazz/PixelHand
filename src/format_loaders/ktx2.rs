// src/format_loaders/ktx2.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct Ktx2Loader;

/// Decodes KTX2 container textures using `ktx2` reader and `texture2ddecoder` for payload decompression.
pub fn decode_ktx2_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    // Parse KTX2 file container header and level index tables securely via ktx2 crate
    let reader = ktx2::Reader::new(bytes)
        .map_err(|e| anyhow!("Invalid KTX2 header or corrupted file: {:?}", e))?;

    let header = reader.header();
    let width = header.pixel_width as usize;
    let height = header.pixel_height.max(1) as usize;

    // OOM Bounds Check: Prevent memory allocation crashes on corrupted/oversized image headers
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(anyhow!(
            "Invalid or oversized KTX2 dimensions: {}x{} (max 16384x16384)",
            width,
            height
        ));
    }

    // Retrieve raw level for the base mipmap level (Level 0)
    let first_level = reader
        .levels()
        .next()
        .ok_or_else(|| anyhow!("KTX2 container contains no mip levels"))?;

    // Extract raw byte slice from the Level struct
    let level_data: &[u8] = first_level.data;

    let format_raw = header.format.map(|f| f.value()).unwrap_or(0);
    let mut rgba_u32 = vec![0u32; width * height];

    match format_raw {
        // VK_FORMAT_R8G8B8A8_UNORM (37) | VK_FORMAT_R8G8B8A8_SRGB (43)
        37 | 43 => {
            let copy_len = level_data.len().min(width * height * 4);
            let img = image::RgbaImage::from_raw(
                width as u32,
                height as u32,
                level_data[..copy_len].to_vec(),
            )
            .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 UNORM data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }
        // VK_FORMAT_B8G8R8A8_UNORM (44) | VK_FORMAT_B8G8R8A8_SRGB (50)
        44 | 50 => {
            let copy_len = level_data.len().min(width * height * 4);
            let mut bgra_buf = level_data[..copy_len].to_vec();
            for chunk in bgra_buf.chunks_exact_mut(4) {
                chunk.swap(0, 2); // Convert BGRA -> RGBA
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, bgra_buf)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 BGRA data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }
        // EAC R11 (Single-channel heightmaps / precision masks)
        153 | 154 => {
            texture2ddecoder::decode_eacr(level_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R11 decoding failed: {:?}", e))?;
        }
        // EAC RG11 (Dual-channel normal maps Red+Green)
        155 | 156 => {
            texture2ddecoder::decode_eacrg(level_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG11 decoding failed: {:?}", e))?;
        }
        // ETC2 RGBA8 / RGB (147..=152)
        147..=152 => {
            texture2ddecoder::decode_etc2_rgba8(level_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 decoding failed: {:?}", e))?;
        }
        // ETC1 / Fallback block decoding
        _ => {
            texture2ddecoder::decode_etc1(level_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC1 decoding failed: {:?}", e))?;
        }
    }

    let mut raw_bytes: Vec<u8> = rgba_u32.into_iter().flat_map(|p| p.to_le_bytes()).collect();

    // Convert BGRA from texture2ddecoder into RGBA for Rust image crate
    for chunk in raw_bytes.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    let img = image::RgbaImage::from_raw(width as u32, height as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile KTX2 RGBA buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

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
        let bytes = std::fs::read(path).context("Failed to read KTX2 file from disk")?;
        decode_ktx2_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;
        let mut is_cubemap = false;
        let mut compression_format = "KTX2".to_string();
        let mut color_space = "Linear".to_string();

        if let Ok(reader) = ktx2::Reader::new(&bytes) {
            let header = reader.header();
            w = header.pixel_width;
            h = header.pixel_height.max(1);
            mips = header.level_count.max(1);
            // Precise Cubemap detection (face_count == 6 in Khronos Spec)
            is_cubemap = header.face_count == 6;

            // Extract precise format codec or supercompression scheme name
            if let Some(scheme) = header.supercompression_scheme {
                compression_format = format!("KTX2 ({:?})", scheme);
            } else if let Some(fmt) = header.format {
                compression_format = format!("KTX2 ({:?})", fmt);
            }

            // Extract exact Color Space from Transfer Function or Format enum
            if let Some(tf) = reader.transfer_function() {
                if tf == ktx2::TransferFunction::SRGB {
                    color_space = "sRGB".to_string();
                }
            } else if compression_format.contains("SRGB") {
                color_space = "sRGB".to_string();
            }
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        let estimated_vram =
            crate::qc::rules::estimate_vram(w, h, &compression_format, mips, is_cubemap);

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "ktx2".to_string(),
            compression_format,
            color_space,
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap,
            estimated_vram,
        })
    }
}
