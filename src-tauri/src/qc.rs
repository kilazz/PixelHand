// src-tauri/src/qc.rs
use std::cmp;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct QcImageMetadata {
    pub path: PathBuf,
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
fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

pub fn check_normal_map_integrity(
    path: &std::path::Path,
    threshold: f32,
) -> Option<(String, String)> {
    let img = match image::open(path) {
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
    if check_block_align && (meta.width % 4 != 0 || meta.height % 4 != 0) {
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
