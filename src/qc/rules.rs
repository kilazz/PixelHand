// src/qc/rules.rs

use std::fs;
use std::path::Path;
use thiserror::Error;

// ==========================================
// --- TECHNICAL ERROR TYPES ----------------
// ==========================================

#[derive(Error, Debug)]
pub enum QcError {
    #[error("Failed to read image structure: {0}")]
    MetadataExtractionFailed(String),
    #[error("File metadata access failed: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Image dimensions could not be read: {0}")]
    ImageSizeError(#[from] imagesize::ImageError),
}

// ==========================================
// --- DATA MODELS & CONSTANTS --------------
// ==========================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormalMapFormat {
    TangentSpaceRgb,
    Bc5RxGy,
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
    pub estimated_vram: u64,
}

impl QcImageMetadata {
    /// Safe metadata constructor that gracefully falls back to basic dimensions if full parsing fails.
    pub fn extract_or_fallback(path: &Path) -> Self {
        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let (width, height) = match imagesize::size(path) {
            Ok(dim) => (dim.width as u32, dim.height as u32),
            Err(_) => (0, 0),
        };

        extract_qc_metadata(path).unwrap_or_else(|_| {
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            QcImageMetadata {
                width,
                height,
                file_size: size,
                format_str: ext.clone(),
                compression_format: ext,
                color_space: "sRGB".to_string(),
                has_alpha: false,
                bit_depth: 8,
                mipmap_count: 1,
                is_cubemap: false,
                estimated_vram: 0,
            }
        })
    }
}

// ==========================================
// --- RULES EVALUATION LOGIC ---------------
// ==========================================

/// Estimates the expected VRAM footprint of a texture on GPU memory.
/// Considers standard block compressed ratios, row alignments, cubemaps, and mipmap chains.
pub fn estimate_vram(width: u32, height: u32, format: &str, mipmaps: u32, is_cubemap: bool) -> u64 {
    let format_upper = format.to_uppercase();
    let block_compressed =
        format_upper.contains("BC") || format_upper.contains("DXT") || format_upper.contains("ATI");

    let bytes_per_block: u64 = match format_upper.as_str() {
        cf if cf.contains("BC1")
            || cf.contains("DXT1")
            || cf.contains("BC4")
            || cf.contains("ATI1") =>
        {
            8
        }
        cf if cf.contains("BC2")
            || cf.contains("DXT3")
            || cf.contains("BC3")
            || cf.contains("DXT5")
            || cf.contains("BC5")
            || cf.contains("ATI2")
            || cf.contains("BC6")
            || cf.contains("BC7") =>
        {
            16
        }
        _ => 0,
    };

    let bytes_per_pixel: u64 = if !block_compressed {
        match format_upper.as_str() {
            cf if cf.contains("RGBA8")
                || cf.contains("BGRA8")
                || (cf.contains("UNCOMPRESSED") && cf.contains("ALPHA")) =>
            {
                4
            }
            cf if cf.contains("RGB8") || cf.contains("BGR8") || cf.contains("UNCOMPRESSED") => 3,
            cf if cf.contains("RGBA16") => 8,
            cf if cf.contains("RGBA32") => 16,
            _ => 4,
        }
    } else {
        0
    };

    let mut total_bytes = 0;
    let mips = mipmaps.max(1);
    let pitch_alignment: u64 = 256;

    for i in 0..mips {
        let w = (width >> i).max(1) as u64;
        let h = (height >> i).max(1) as u64;

        let mip_bytes = if block_compressed {
            let blocks_w = w.div_ceil(4);
            let blocks_h = h.div_ceil(4);
            let raw_row_pitch = blocks_w * bytes_per_block;
            let aligned_row_pitch = (raw_row_pitch + pitch_alignment - 1) & !(pitch_alignment - 1);
            aligned_row_pitch * blocks_h
        } else {
            let raw_row_pitch = w * bytes_per_pixel;
            let aligned_row_pitch = (raw_row_pitch + pitch_alignment - 1) & !(pitch_alignment - 1);
            aligned_row_pitch * h
        };
        total_bytes += mip_bytes;
    }

    if is_cubemap {
        total_bytes * 6
    } else {
        total_bytes
    }
}

/// Evaluates metadata using the polymorphic loaders registry.
pub fn extract_qc_metadata(path: &Path) -> Result<QcImageMetadata, QcError> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let registry = crate::format_loaders::get_registry();
    if let Some(loader) = registry.find_loader(&ext) {
        loader
            .extract_metadata(path)
            .map_err(|e| QcError::MetadataExtractionFailed(e.to_string()))
    } else {
        Err(QcError::MetadataExtractionFailed(format!(
            "No format loader mapped for extension: {}",
            ext
        )))
    }
}

/// Evaluates absolute, standalone texture specifications against game-engine standards.
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

/// Evaluates relative difference specifications between an original asset (A) and a target asset (B).
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

/// Downscales a texture to verify if it consists entirely of a single solid color.
pub fn check_solid_texture(path: &Path) -> Option<(String, String)> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(128), None).ok()?;
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

/// Examines color channel variation to detect flat, empty masks.
pub fn check_empty_channels(path: &Path) -> Option<(String, String)> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(128), None).ok()?;
    let has_alpha = img.color().has_alpha();
    let processed_img = if img.width() > 128 || img.height() > 128 {
        img.resize(128, 128, image::imageops::FilterType::Nearest)
    } else {
        img
    };
    let rgba_img = processed_img.to_rgba8();
    if rgba_img.pixels().len() == 0 {
        return None;
    }

    let mut min_val = [255u8; 4];
    let mut max_val = [0u8; 4];

    for p in rgba_img.pixels() {
        for i in 0..4 {
            min_val[i] = min_val[i].min(p[i]);
            max_val[i] = max_val[i].max(p[i]);
        }
    }

    let channel_names = [
        "Red (AO/Smoothness)",
        "Green (Roughness)",
        "Blue (Metalness)",
        "Alpha (Glossiness)",
    ];
    let threshold = 5;

    for i in 0..4 {
        if i == 3 && !has_alpha {
            continue;
        }
        if max_val[i] - min_val[i] < threshold {
            return Some((
                "Empty Channel (Flat Mask)".to_string(),
                format!(
                    "Channel {} is empty/solid color (Min: {}, Max: {}). Check your packing exports.",
                    channel_names[i], min_val[i], max_val[i]
                ),
            ));
        }
    }
    None
}

/// Evaluates relative gradients on the Green channel to predict OpenGL vs DirectX normal layout orientation.
pub fn check_normal_map_orientation(path: &Path) -> Option<(String, String)> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(256), None).ok()?;
    let rgb = img.to_rgb8();

    let mut top_lights = 0u64;
    let mut bottom_lights = 0u64;
    let w = rgb.width();
    let h = rgb.height();

    if w < 4 || h < 4 {
        return None;
    }

    let raw = rgb.as_raw();
    let stride = w as usize * 3;

    for y in 1..(h - 1) {
        let y_idx = y as usize * stride;
        let prev_y_idx = (y - 1) as usize * stride;
        let next_y_idx = (y + 1) as usize * stride;

        for x in 1..(w - 1) {
            let x_offset = x as usize * 3 + 1;
            let cur_g = raw[y_idx + x_offset] as f32;
            let prev_g = raw[prev_y_idx + x_offset] as f32;
            let next_g = raw[next_y_idx + x_offset] as f32;

            if (next_g - prev_g).abs() > 30.0 {
                if cur_g > 128.0 {
                    top_lights += 1;
                } else {
                    bottom_lights += 1;
                }
            }
        }
    }

    if top_lights > 0 && bottom_lights > 0 {
        let total = top_lights + bottom_lights;
        let ratio = top_lights as f32 / total as f32;
        if ratio > 0.65 {
            return Some((
                "Normal Map Y-Axis Orientation".to_string(),
                "OpenGL layout detected (+Y Green channel). Ensure this matches your project's specifications.".to_string()
            ));
        } else if ratio < 0.35 {
            return Some((
                "Normal Map Y-Axis Orientation".to_string(),
                "DirectX layout detected (-Y Green channel). Ensure this matches your project's specifications.".to_string()
            ));
        }
    }
    None
}

/// Analyzes tangent-space normal vectors, checking unit circle bounds and vector lengths to verify integrity.
pub fn check_normal_map_integrity(
    path: &Path,
    threshold: f32,
    format: NormalMapFormat,
) -> Option<(String, String)> {
    let img = crate::format_loaders::open_image_with_dds_fallback(path, Some(512), None).ok()?;
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
                            "{:.0}% of pixels exceed tangent unit circle bounds",
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
        }
        NormalMapFormat::Bc5RxGy => {
            for pixel in rgb_img.pixels() {
                let x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                let y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
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
