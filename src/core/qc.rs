// src/core/qc.rs

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Represents the texture normal map encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormalMapFormat {
    TangentSpaceRgb, // Standard TS (X in Red, Y in Green, Z in Blue)
    Bc5RxGy,         // Two-channel (Z is reconstructed on the GPU: Z = sqrt(1 - X^2 - Y^2))
}

#[derive(Debug, Clone)]
pub struct QcImageMetadata {
    pub width: u32,
    pub height: u32,
    pub file_size: u64,
    pub format_str: String,
    pub compression_format: String,
    pub color_space: String,
    pub has_alpha: bool,
    pub bit_depth: u32,
    pub mipmap_count: u32,
    pub is_cubemap: bool,
}

/// Safely reads a little-endian u32 integer from a byte slice at the given offset with bounds checking.
fn safe_read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    bytes
        .get(offset..offset + 4)
        .and_then(|sub| sub.try_into().ok())
        .map(u32::from_le_bytes)
        .unwrap_or(0)
}

/// Extracts technical image specifications, format metrics, and metadata layout from target path.
pub fn extract_qc_metadata(path: &Path) -> Result<QcImageMetadata> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let (mut w, mut h) = match imagesize::size(path) {
        Ok(dim) => (dim.width as u32, dim.height as u32),
        Err(_) => (0, 0),
    };

    let format_str = ext.clone();
    let mut compression_format = ext.to_uppercase();
    let mut color_space = "sRGB".to_string();
    let mut has_alpha = false;
    let mut bit_depth = 8;
    let mut mipmap_count = 1;
    let mut is_cubemap = false;

    if ext == "dds" {
        let bytes = fs::read(path).context("Failed to read DDS bytes")?;
        if bytes.len() >= 128 {
            h = safe_read_u32_le(&bytes, 12);
            w = safe_read_u32_le(&bytes, 16);

            let mips = safe_read_u32_le(&bytes, 28);
            mipmap_count = if mips > 0 { mips } else { 1 };

            // dwCaps2 is located at offset 112. The flag 0x0000FE00 indicates a Cubemap.
            let dw_caps2 = safe_read_u32_le(&bytes, 112);
            is_cubemap = (dw_caps2 & 0x0000FE00) != 0;

            let pf_flags = safe_read_u32_le(&bytes, 80);
            if (pf_flags & 0x1) != 0 || (pf_flags & 0x2) != 0 {
                has_alpha = true;
            }

            let pf_fourcc = safe_read_u32_le(&bytes, 84);
            if pf_fourcc != 0 {
                if let Some(fourcc_slice) = bytes.get(84..88) {
                    compression_format = String::from_utf8_lossy(fourcc_slice).trim().to_string();
                }
                if compression_format == "DXT3" || compression_format == "DXT5" {
                    has_alpha = true;
                }

                if pf_fourcc == u32::from_le_bytes(*b"DX10") && bytes.len() >= 148 {
                    let dxgi_format = safe_read_u32_le(&bytes, 128);
                    compression_format = format!("DX10 (DXGI {})", dxgi_format);
                    if matches!(dxgi_format, 74 | 75 | 77 | 78 | 98 | 99) {
                        has_alpha = true;
                    }
                    if matches!(dxgi_format, 71 | 74 | 77 | 80 | 83 | 98) {
                        color_space = "Linear".to_string();
                    }
                }
            } else {
                compression_format = "Uncompressed".to_string();
                let rgb_bit_count = safe_read_u32_le(&bytes, 88);
                bit_depth = if rgb_bit_count / 4 == 0 {
                    8
                } else {
                    rgb_bit_count / 4
                };
            }
        }
    } else if ext == "exr" {
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
    } else if ext == "hdr" {
        bit_depth = 32;
        color_space = "Linear".to_string();
        compression_format = "Radiance HDR".to_string();
    } else {
        if let Ok(img) = image::open(path) {
            let color = img.color();
            has_alpha = color.has_alpha();
            bit_depth = color.bits_per_pixel() as u32 / if has_alpha { 4 } else { 3 };
        }
    }

    Ok(QcImageMetadata {
        width: w,
        height: h,
        file_size: size,
        format_str,
        compression_format,
        color_space,
        has_alpha,
        bit_depth,
        mipmap_count,
        is_cubemap,
    })
}

/// Evaluates absolute single-asset constraints including NPOT dimensions, block align bounds, and bit depth limits.
pub fn check_absolute(
    meta: &QcImageMetadata,
    check_npot: bool,
    check_block_align: bool,
    check_mipmaps: bool,
    check_bit_depth: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    let is_pow2 = |n: u32| n != 0 && (n & (n - 1)) == 0;

    if check_npot && (!is_pow2(meta.width) || !is_pow2(meta.height)) {
        issues.push("Non-Power-Of-Two (NPOT)".into());
    }
    if check_block_align && (!meta.width.is_multiple_of(4) || !meta.height.is_multiple_of(4)) {
        issues.push("Bad Block Alignment (Not divisible by 4)".into());
    }
    if check_mipmaps && std::cmp::min(meta.width, meta.height) >= 64 && meta.mipmap_count <= 1 {
        issues.push("Missing Mipmaps".into());
    }
    if check_bit_depth && meta.bit_depth > 8 {
        issues.push(format!("High Bit Depth ({}-bit)", meta.bit_depth));
    }

    issues
}

/// Evaluates relative constraints on file parameter changes between source (original) and destination (duplicate) assets.
pub fn check_relative(
    source: &QcImageMetadata,
    target: &QcImageMetadata,
    bloat: bool,
    alpha: bool,
    colorspace: bool,
    comp: bool,
) -> Vec<String> {
    let mut issues = Vec::new();

    if (target.width * target.height) < (source.width * source.height) {
        issues.push("Resolution Downgrade".into());
    }
    if bloat && target.file_size > (source.file_size as f64 * 1.5) as u64 {
        issues.push("Size Bloat (>1.5x File Size)".into());
    }
    if alpha && source.has_alpha && !target.has_alpha {
        issues.push("Lost Alpha Channel".into());
    }
    if colorspace && source.color_space != target.color_space {
        issues.push(format!(
            "Color Space Mismatch ({} -> {})",
            source.color_space, target.color_space
        ));
    }
    if comp && source.compression_format.to_uppercase() != target.compression_format.to_uppercase()
    {
        issues.push(format!(
            "Format Transition ({} -> {})",
            source.compression_format, target.compression_format
        ));
    }

    issues
}

/// Evaluates if the texture is a flat single-color block within a threshold tolerance.
pub fn check_solid_texture(path: &Path) -> Option<(String, String)> {
    let img =
        crate::format_loaders::dds_loader::open_image_with_dds_fallback(path, Some(128)).ok()?;
    let processed_img = if img.width() > 128 || img.height() > 128 {
        img.resize(128, 128, image::imageops::FilterType::Nearest)
    } else {
        img
    };

    let rgba_img = processed_img.to_rgba8();
    let mut pixels = rgba_img.pixels();
    let first = pixels.next()?;

    let tolerance = 2i16;
    for p in pixels {
        for i in 0..4 {
            if (p[i] as i16 - first[i] as i16).abs() > tolerance {
                return None;
            }
        }
    }

    Some((
        "Solid Color Texture".to_string(),
        format!(
            "Image is entirely a single color: R:{}, G:{}, B:{}, A:{}",
            first[0], first[1], first[2], first[3]
        ),
    ))
}

/// Computes normal map vector bounds, verifies normalize properties, and screens for invalid normal length directions.
pub fn check_normal_map_integrity(
    path: &Path,
    threshold: f32,
    format: NormalMapFormat,
) -> Option<(String, String)> {
    let img =
        crate::format_loaders::dds_loader::open_image_with_dds_fallback(path, Some(512)).ok()?;
    let processed_img = if img.width() > 512 || img.height() > 512 {
        img.resize(512, 512, image::imageops::FilterType::Nearest)
    } else {
        img
    };

    let rgb_img = processed_img.to_rgb8();
    let total_pixels = (rgb_img.width() * rgb_img.height()) as usize;
    if total_pixels == 0 {
        return None;
    }

    let mut bad_pixels = 0;

    match format {
        NormalMapFormat::TangentSpaceRgb => {
            let mut sum_z = 0.0f32;
            for pixel in rgb_img.pixels() {
                let z = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;
                sum_z += z;
            }
            let mean_z = sum_z / total_pixels as f32;

            if mean_z > 0.98 {
                // Flat normal maps (clipping check on XY unit boundary circle)
                for pixel in rgb_img.pixels() {
                    let x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                    let y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
                    let len_xy = (x * x + y * y).sqrt();
                    let diff_xy = (len_xy - 1.0).max(0.0);
                    if diff_xy > threshold {
                        bad_pixels += 1;
                    }
                }
                let bad_ratio = bad_pixels as f32 / total_pixels as f32;
                if bad_ratio > 0.10 {
                    return Some((
                        "Bad Normal Map (XY Clip)".to_string(),
                        format!(
                            "{:.0}% of pixels exceed unit circle bounds",
                            bad_ratio * 100.0
                        ),
                    ));
                }
            } else {
                // Standard normal maps (unit vector scale validations)
                for pixel in rgb_img.pixels() {
                    let x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                    let y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
                    let z = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;
                    let magnitude = (x * x + y * y + z * z).sqrt();
                    let diff = (magnitude - 1.0).abs();
                    if diff > threshold {
                        bad_pixels += 1;
                    }
                }
                let bad_ratio = bad_pixels as f32 / total_pixels as f32;
                if bad_ratio > 0.10 {
                    return Some((
                        "Bad Normal Map (Integrity)".to_string(),
                        format!(
                            "{:.0}% of pixels have invalid vector lengths",
                            bad_ratio * 100.0
                        ),
                    ));
                }
                if mean_z < 0.2 {
                    return Some((
                        "Inverted Z / Not Tangent Space".to_string(),
                        "Z-Axis represents flat or inverted normals".to_string(),
                    ));
                }
            }
        }
        NormalMapFormat::Bc5RxGy => {
            // Two-channel BC5/RG format: Blue channel is empty, Z must be reconstructed: Z = sqrt(max(0, 1 - X^2 - Y^2))
            for pixel in rgb_img.pixels() {
                let x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                let y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;

                // Verify if the 2D vector lies inside or reasonably close to the unit circle
                let sq_len = x * x + y * y;
                if sq_len > (1.0 + threshold) {
                    bad_pixels += 1;
                }
            }
            let bad_ratio = bad_pixels as f32 / total_pixels as f32;
            if bad_ratio > 0.10 {
                return Some((
                    "Bad BC5 Normal Map".to_string(),
                    format!(
                        "{:.0}% of pixels exceed reconstruction limits (X^2 + Y^2 > 1.0)",
                        bad_ratio * 100.0
                    ),
                ));
            }
        }
    }
    None
}
