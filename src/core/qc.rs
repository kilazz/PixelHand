// src/core/qc.rs

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

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
    let mut is_cubemap = false; // Initialize to false by default

    if ext == "dds" {
        let bytes = fs::read(path).context("Failed to read DDS bytes")?;
        if bytes.len() >= 128 {
            h = u32::from_le_bytes(bytes[12..16].try_into().unwrap_or_default());
            w = u32::from_le_bytes(bytes[16..20].try_into().unwrap_or_default());

            let dw_flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap_or_default());
            let mips = u32::from_le_bytes(bytes[28..32].try_into().unwrap_or_default());
            mipmap_count = if (dw_flags & 0x20000) != 0 && mips > 0 {
                mips
            } else {
                1
            };

            // --- CUBEMAP DETECTION ---
            // dwCaps2 is located at offset 112. The flag 0x0000FE00 indicates a Cubemap.
            let dw_caps2 = u32::from_le_bytes(bytes[112..116].try_into().unwrap_or_default());
            is_cubemap = (dw_caps2 & 0x0000FE00) != 0;

            let pf_flags = u32::from_le_bytes(bytes[80..84].try_into().unwrap_or_default());
            if (pf_flags & 0x1) != 0 || (pf_flags & 0x2) != 0 {
                has_alpha = true;
            }

            let pf_fourcc = u32::from_le_bytes(bytes[84..88].try_into().unwrap_or_default());
            if pf_fourcc != 0 {
                compression_format = String::from_utf8_lossy(&bytes[84..88]).trim().to_string();
                if compression_format == "DXT3" || compression_format == "DXT5" {
                    has_alpha = true;
                }

                if pf_fourcc == u32::from_le_bytes(*b"DX10") && bytes.len() >= 148 {
                    let dxgi_format =
                        u32::from_le_bytes(bytes[128..132].try_into().unwrap_or_default());
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
                let rgb_bit_count =
                    u32::from_le_bytes(bytes[88..92].try_into().unwrap_or_default());
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
        is_cubemap, // Set the flag in the returned struct
    })
}

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

pub fn check_solid_texture(path: &Path) -> Option<(String, String)> {
    // --- OPTIMIZED: Request low-res 128px mipmap during solid color check ---
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

pub fn check_normal_map_integrity(path: &Path, threshold: f32) -> Option<(String, String)> {
    // --- OPTIMIZED: Request low-res 512px mipmap directly during normal map check ---
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

    let mut sum_z = 0.0f32;
    for pixel in rgb_img.pixels() {
        let z = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;
        sum_z += z;
    }
    let mean_z = sum_z / total_pixels as f32;
    let mut bad_pixels = 0;

    if mean_z > 0.98 {
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
    None
}

/// Helper to run visual difference mapping at 100% original resolution and save output to disk
pub async fn calculate_diff_map(file1: &str, file2: &str) -> Result<String> {
    // --- OPTIMIZED: Diff map requires pixel-perfect 1:1 original resolution details (passed None) ---
    let img1 = crate::format_loaders::dds_loader::open_image_with_dds_fallback(file1, None)?;
    let img2 = crate::format_loaders::dds_loader::open_image_with_dds_fallback(file2, None)?;

    let diff =
        super::tonemapper::calculate_difference_map(&img1.to_rgba8(), &img2.to_rgba8(), true)?;
    let temp_dir = crate::utils::settings::get_portable_app_data_dir()?.join("temp");
    fs::create_dir_all(&temp_dir)?;
    let out_path = temp_dir.join("diff_output.png");
    diff.save(&out_path)?;

    Ok(out_path.to_string_lossy().to_string())
}
