use exr::prelude::MetaData;
use std::cmp;
use std::convert::TryInto;
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
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes
        .get(offset..offset + 4)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}

#[inline]
fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

pub fn extract_qc_metadata(path: &Path) -> Result<QcImageMetadata, String> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // First get dimensions fast
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

    if ext == "dds" {
        // Read file bytes safely without panics
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() >= 128 {
            h = read_u32_le(&bytes, 12).unwrap_or(h);
            w = read_u32_le(&bytes, 16).unwrap_or(w);

            // Mipmap count is at bytes 28..32
            let dw_flags = read_u32_le(&bytes, 8).unwrap_or(0);
            let mips = read_u32_le(&bytes, 28).unwrap_or(0);

            // Check DDSD_MIPMAPCOUNT flag: 0x20000
            if (dw_flags & 0x20000) != 0 && mips > 0 {
                mipmap_count = mips;
            } else {
                mipmap_count = 1;
            }

            // Pixel format flags (bytes 80..84)
            let pf_flags = read_u32_le(&bytes, 80).unwrap_or(0);

            // Check DDPF_ALPHAPIXELS: 0x1, DDPF_ALPHA: 0x2
            if (pf_flags & 0x1) != 0 || (pf_flags & 0x2) != 0 {
                has_alpha = true;
            }

            let pf_fourcc = read_u32_le(&bytes, 84).unwrap_or(0);
            if pf_fourcc != 0 {
                if let Some(fourcc_bytes) = bytes.get(84..88) {
                    let fourcc_str = String::from_utf8_lossy(fourcc_bytes).to_string();
                    compression_format = fourcc_str.trim().to_string();
                    if compression_format == "DXT3" || compression_format == "DXT5" {
                        has_alpha = true;
                    }
                }

                if pf_fourcc == u32::from_le_bytes(*b"DX10") && bytes.len() >= 148 {
                    let dxgi_format = read_u32_le(&bytes, 128).unwrap_or(0);
                    compression_format = format!("DX10 (DXGI {})", dxgi_format);

                    // Standard DXGI formats with alpha (e.g., BC1 usually no, BC2/BC3/BC7 yes)
                    if matches!(dxgi_format, 74 | 75 | 77 | 78 | 98 | 99) {
                        has_alpha = true;
                    }
                    if matches!(dxgi_format, 72 | 75 | 78 | 99) {
                        color_space = "sRGB".to_string();
                    } else if matches!(dxgi_format, 71 | 74 | 77 | 80 | 83 | 98) {
                        color_space = "Linear".to_string();
                    }
                }
            } else {
                compression_format = "Uncompressed".to_string();

                // Check if RGB or RGBA uncompressed
                let rgb_bit_count = read_u32_le(&bytes, 88).unwrap_or(0);
                bit_depth = rgb_bit_count / 4;
                if bit_depth == 0 {
                    bit_depth = 8;
                }
            }
        }
    } else if ext == "exr" {
        // Parse metadata using exr crate
        if let Ok(meta) = MetaData::read_from_file(path, false) {
            if let Some(header) = meta.headers.first() {
                w = header.shared_attributes.display_window.size.width() as u32;
                h = header.shared_attributes.display_window.size.height() as u32;

                has_alpha = false;
                bit_depth = 16; // default for EXR is f16 half-float
                for channel in &header.channels.list {
                    let channel_name = channel.name.to_string();
                    if channel_name == "A" || channel_name == "alpha" {
                        has_alpha = true;
                    }

                    // check sample type
                    let sample_type_str = format!("{:?}", channel.sample_type);
                    if sample_type_str.contains("F32") || sample_type_str.contains("U32") {
                        bit_depth = 32;
                    }
                }

                // Color space: EXR is natively Linear (Scene-referred)
                color_space = "Linear".to_string();
                compression_format = format!("{:?}", header.compression);
            }
        } else {
            bit_depth = 16;
            color_space = "Linear".to_string();
            compression_format = "EXR (Unknown)".to_string();
        }
    } else if ext == "hdr" {
        bit_depth = 32;
        color_space = "Linear".to_string();
        compression_format = "Radiance HDR".to_string();
    } else {
        // Standard formats PNG, JPEG, TGA, WEBP, etc.
        if let Ok(img) = image::open(path) {
            let color = img.color();
            let has_alpha_val = matches!(
                color,
                image::ColorType::La8
                    | image::ColorType::Rgba8
                    | image::ColorType::La16
                    | image::ColorType::Rgba16
                    | image::ColorType::Rgba32F
            );
            has_alpha = has_alpha_val;

            bit_depth = match color {
                image::ColorType::L8
                | image::ColorType::La8
                | image::ColorType::Rgb8
                | image::ColorType::Rgba8 => 8,
                image::ColorType::L16
                | image::ColorType::La16
                | image::ColorType::Rgb16
                | image::ColorType::Rgba16 => 16,
                image::ColorType::Rgb32F | image::ColorType::Rgba32F => 32,
                _ => 8,
            };
        }

        // Linear normal maps naming conventions rules
        let path_lower = path.to_string_lossy().to_lowercase();
        if path_lower.contains("normal")
            || path_lower.contains("_n.")
            || path_lower.contains("_ddn.")
        {
            color_space = "Linear".to_string();
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
    })
}

pub fn check_normal_map_integrity(
    path: &std::path::Path,
    threshold: f32,
) -> Option<(String, String)> {
    let img = match crate::dds_loader::open_image_with_dds_fallback(path) {
        Ok(loaded_img) => loaded_img,
        Err(_) => return None,
    };

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

pub fn check_solid_texture(path: &std::path::Path) -> Option<(String, String)> {
    let img = match crate::dds_loader::open_image_with_dds_fallback(path) {
        Ok(loaded_img) => loaded_img,
        Err(_) => return None,
    };

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

pub fn check_absolute(
    meta: &QcImageMetadata,
    check_npot: bool,
    check_block_align: bool,
    check_mipmaps: bool,
    check_bit_depth: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    if check_npot && (!is_power_of_two(meta.width) || !is_power_of_two(meta.height)) {
        issues.push("Non-Power-Of-Two (NPOT)".to_string());
    }
    if check_block_align && (!meta.width.is_multiple_of(4) || !meta.height.is_multiple_of(4)) {
        issues.push("Bad Block Alignment (Not divisible by 4)".to_string());
    }
    if check_mipmaps && cmp::min(meta.width, meta.height) >= 64 && meta.mipmap_count <= 1 {
        issues.push("Missing Mipmaps".to_string());
    }
    if check_bit_depth && meta.bit_depth > 8 {
        issues.push(format!("High Bit Depth ({}-bit)", meta.bit_depth));
    }
    issues
}

pub fn check_relative(
    source: &QcImageMetadata,
    target: &QcImageMetadata,
    check_size_bloat: bool,
    check_alpha: bool,
    check_color_space: bool,
    check_compression: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    let area_source = source.width as u64 * source.height as u64;
    let area_target = target.width as u64 * target.height as u64;

    if area_target < area_source {
        issues.push("Resolution Downgrade".to_string());
    }
    if check_size_bloat && target.file_size > (source.file_size as f64 * 1.5) as u64 {
        issues.push("Size Bloat (>1.5x File Size)".to_string());
    }
    if check_alpha {
        if source.has_alpha && !target.has_alpha {
            issues.push("Lost Alpha Channel".to_string());
        } else if !source.has_alpha && target.has_alpha {
            issues.push("Added Empty Alpha Channel".to_string());
        }
    }
    if check_color_space && source.color_space != target.color_space {
        issues.push(format!(
            "Color Space Mismatch ({} -> {})",
            source.color_space, target.color_space
        ));
    }
    if check_compression {
        let fmt_a = source.compression_format.to_uppercase();
        let fmt_b = target.compression_format.to_uppercase();
        if fmt_a != fmt_b {
            issues.push(format!("Format Transition ({} -> {})", fmt_a, fmt_b));
        }
    }
    issues
}
