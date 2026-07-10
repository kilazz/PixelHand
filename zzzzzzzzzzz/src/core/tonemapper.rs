// src/core/tonemapper.rs
use anyhow::{Context, Result};
use exr::prelude::*;
use image::RgbaImage;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static TONEMAP_ENABLED: AtomicBool = AtomicBool::new(true);
// 0: AcesFilmic, 1: ICtCp Perceptual, 2: Khronos PBR Neutral
pub static TONEMAP_OPERATOR: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    None,
    AcesFilmic,
    ICtCpLumina,
    PbrNeutral,
}

// PQ Constants (ST.2084)
const PQ_N: f32 = 2610.0 / 4096.0 / 4.0;
const PQ_M: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
const PQ_MAX_NITS: f32 = 10000.0;

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn nit_to_pq(nits: f32) -> f32 {
    let y = nits / PQ_MAX_NITS;
    let y_pow = y.abs().powf(PQ_N);
    ((PQ_C1 + PQ_C2 * y_pow) / (1.0 + PQ_C3 * y_pow))
        .abs()
        .powf(PQ_M)
}

#[inline]
fn pq_to_nit(pq: f32) -> f32 {
    let y_pow = pq.abs().powf(1.0 / PQ_M);
    let num = (y_pow - PQ_C1).max(0.0);
    let den = PQ_C2 - (PQ_C3 * y_pow);
    (num / den).abs().powf(1.0 / PQ_N) * PQ_MAX_NITS
}

#[inline]
fn linear_to_st2084(linear_nits: [f32; 3]) -> [f32; 3] {
    [
        nit_to_pq(linear_nits[0]),
        nit_to_pq(linear_nits[1]),
        nit_to_pq(linear_nits[2]),
    ]
}

#[inline]
fn st2084_to_linear(pq: [f32; 3]) -> [f32; 3] {
    [pq_to_nit(pq[0]), pq_to_nit(pq[1]), pq_to_nit(pq[2])]
}

fn rgb_to_ictcp(col: [f32; 3]) -> [f32; 3] {
    let l = col[0] * 0.412109 + col[1] * 0.523926 + col[2] * 0.063965;
    let m = col[0] * 0.166748 + col[1] * 0.720459 + col[2] * 0.112793;
    let s = col[0] * 0.024170 + col[1] * 0.075439 + col[2] * 0.900391;

    let pq_lms = linear_to_st2084([l.max(0.0), m.max(0.0), s.max(0.0)]);

    let i = pq_lms[0] * 0.5000 + pq_lms[1] * 0.5000 + pq_lms[2] * 0.0000;
    let ct = pq_lms[0] * 1.6137 - pq_lms[1] * 3.3234 + pq_lms[2] * 1.7097;
    let cp = pq_lms[0] * 4.3780 - pq_lms[1] * 4.2455 - pq_lms[2] * 0.1325;

    [i, ct, cp]
}

fn ictcp_to_rgb(col: [f32; 3]) -> [f32; 3] {
    let l = col[0] * 1.0 + col[1] * 0.008605 + col[2] * 0.111036;
    let m = col[0] * 1.0 - col[1] * 0.008605 - col[2] * 0.111036;
    let s = col[0] * 1.0 + col[1] * 0.560049 - col[2] * 0.320637;

    let lms = st2084_to_linear([l, m, s]);

    let r = lms[0] * 3.43661 - lms[1] * 2.50645 + lms[2] * 0.069845;
    let g = -lms[0] * 0.79133 + lms[1] * 1.9836 - lms[2] * 0.192271;
    let b = -lms[0] * 0.02595 - lms[1] * 0.098914 + lms[2] * 1.12486;

    [r, g, b]
}

/// Khronos PBR Neutral Tonemapper
/// Developed specifically to perfectly preserve Base Color in 3D/PBR pipelines
/// without causing hue shifts at high intensities.
#[inline]
fn pbr_neutral_tonemapping(color_in: [f32; 3]) -> [f32; 3] {
    // Input colors must be strictly non-negative
    let mut color = [
        color_in[0].max(0.0),
        color_in[1].max(0.0),
        color_in[2].max(0.0),
    ];

    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;

    let x = color[0].min(color[1]).min(color[2]);
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };

    color[0] -= offset;
    color[1] -= offset;
    color[2] -= offset;

    let peak = color[0].max(color[1]).max(color[2]);
    if peak < start_compression {
        return color;
    }

    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);

    let ratio = new_peak / peak;
    color[0] *= ratio;
    color[1] *= ratio;
    color[2] *= ratio;

    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);

    let mix_factor = g;
    let target = new_peak;

    [
        color[0] * (1.0 - mix_factor) + target * mix_factor,
        color[1] * (1.0 - mix_factor) + target * mix_factor,
        color[2] * (1.0 - mix_factor) + target * mix_factor,
    ]
}

/// Hermite spline from the ITU-R BT.2390 (EETF) standard.
/// Smoothly compresses the luminance range (in PQ space) to prevent clipping.
#[inline]
fn eetf_bt2390(i: f32, max_in_pq: f32, max_out_pq: f32) -> f32 {
    // Knee Start point.
    // According to the standard, this is calculated as the intersection of display limits.
    let ks = (1.5 * max_out_pq - 0.5 * max_in_pq).max(0.0);
    let b = max_in_pq - ks;

    // If the range doesn't require compression
    if b <= 1e-5 {
        return i.min(max_out_pq);
    }

    if i < ks {
        // Linear section (shadows and midtones remain 1:1)
        i
    } else {
        // Smooth roll-off (Shoulder) using a Hermite polynomial
        let t = (i - ks) / b;
        let t2 = t * t;
        let t3 = t2 * t;

        let p0 = ks;
        let p1 = max_out_pq;
        let m0 = 1.0;
        let m1 = 0.0; // Flat top at the peak

        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
            + (t3 - 2.0 * t2 + t) * m0 * b
            + (-2.0 * t3 + 3.0 * t2) * p1
            + (t3 - t2) * m1 * b
    }
}

/// Hybrid operator: ICtCp + BT.2390 Spline + BT.2446 Method C Chroma Scaling.
/// This guarantees zero hue-shift while maintaining cinematic highlight roll-off.
fn tonemap_perceptual_bt2446c(color: [f32; 3], exposure: f32, scene_peak_nits: f32) -> [f32; 3] {
    // 1. Scale incoming Linear RGB (0.0-1.0+) to absolute nits.
    // Assuming 1.0 in SDR graphics = 100 nits.
    let nits = [
        (color[0] * exposure * 100.0).max(0.0),
        (color[1] * exposure * 100.0).max(0.0),
        (color[2] * exposure * 100.0).max(0.0),
    ];

    // 2. Convert Linear RGB to ICtCp (values are PQ encoded inside rgb_to_ictcp)
    let mut ictcp = rgb_to_ictcp(nits);
    let i_in = ictcp[0];

    // 3. Calculate limits in PQ space (ST.2084)
    let max_in_pq = nit_to_pq(scene_peak_nits);
    let max_out_pq = nit_to_pq(100.0); // Target SDR display is mapped to 100 nits

    // 4. Compress Intensity (Luminance) using the BT.2390 EETF curve
    let i_out = eetf_bt2390(i_in, max_in_pq, max_out_pq);
    ictcp[0] = i_out;

    // 5. BT.2446 Method C: Chroma scaling
    // Preserve hue by scaling chroma channels proportionally to luminance compression
    let compression_ratio = if i_in > 1e-5 { i_out / i_in } else { 1.0 };

    // Hunt Effect compensation:
    // When brightness is compressed, human eyes perceive colors as less saturated.
    // Slightly boost saturation so the SDR image doesn't look washed out.
    let saturation_boost = 1.15;

    // Path to White (Cinematic effect):
    // Physically bright light overexposes camera sensors to white.
    // Smoothly desaturate the most extremely bright pixels (top 10% of display luminance).
    let normalized_i = (i_out / max_out_pq).clamp(0.0, 1.0);
    let path_to_white = 1.0 - smoothstep(0.90, 1.0, normalized_i);

    let final_chroma_scale = compression_ratio * saturation_boost * path_to_white;

    ictcp[1] *= final_chroma_scale; // Ct
    ictcp[2] *= final_chroma_scale; // Cp

    // 6. Convert ICtCp back to Linear RGB (Nits)
    let rgb_sdr_nits = ictcp_to_rgb(ictcp);

    // 7. Normalize back to the graphics engine range (0.0 - 1.0)
    [
        (rgb_sdr_nits[0] / 100.0),
        (rgb_sdr_nits[1] / 100.0),
        (rgb_sdr_nits[2] / 100.0),
    ]
}

#[inline]
fn aces_tonemap_raw(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

pub fn load_exr_rgba(path: &Path) -> Result<(Vec<f32>, u32, u32)> {
    let image = read_first_rgba_layer_from_file(
        path,
        |size, _| vec![0.0f32; size.width() * size.height() * 4],
        |pixel_vector, position, (r, g, b, a): (f32, f32, f32, f32)| {
            let index = (position.y() * position.width() + position.x()) * 4;
            pixel_vector[index] = r;
            pixel_vector[index + 1] = g;
            pixel_vector[index + 2] = b;
            pixel_vector[index + 3] = a;
        },
    )
    .context("EXR read failed")?;

    let size = image.layer_data.size;
    Ok((
        image.layer_data.channel_data.pixels,
        size.width() as u32,
        size.height() as u32,
    ))
}

pub fn tonemap_hdr_to_ldr_rgba(
    hdr_pixels: &[f32],
    width: u32,
    height: u32,
    mut operator: TonemapOperator,
    exposure: f32,
) -> Result<RgbaImage> {
    let total_pixels = (width * height) as usize;
    if hdr_pixels.len() != total_pixels * 4 {
        return Err(anyhow::anyhow!("Buffer size mismatch"));
    }

    if !TONEMAP_ENABLED.load(Ordering::Relaxed) {
        operator = TonemapOperator::None;
    } else {
        let active_op = TONEMAP_OPERATOR.load(Ordering::Relaxed);
        if active_op == 2 {
            operator = TonemapOperator::PbrNeutral;
        } else if active_op == 1 {
            operator = TonemapOperator::ICtCpLumina;
        } else if active_op == 0 {
            operator = TonemapOperator::AcesFilmic;
        }
    }

    let mut ldr_pixels = vec![0u8; total_pixels * 4];

    ldr_pixels
        .par_chunks_exact_mut(4)
        .zip(hdr_pixels.par_chunks_exact(4))
        .for_each(|(ldr, hdr)| {
            let r_in = hdr[0] * exposure;
            let g_in = hdr[1] * exposure;
            let b_in = hdr[2] * exposure;

            let (r, g, b) = match operator {
                TonemapOperator::AcesFilmic => (
                    aces_tonemap_raw(r_in),
                    aces_tonemap_raw(g_in),
                    aces_tonemap_raw(b_in),
                ),
                TonemapOperator::ICtCpLumina => {
                    // Hybrid Perceptual Tonemapper mapping
                    // 4000 nits provides enough headroom for extreme VFX brightness
                    let mapped = tonemap_perceptual_bt2446c([r_in, g_in, b_in], 1.0, 4000.0);
                    (
                        mapped[0].clamp(0.0, 1.0),
                        mapped[1].clamp(0.0, 1.0),
                        mapped[2].clamp(0.0, 1.0),
                    )
                }
                TonemapOperator::PbrNeutral => {
                    // Khronos Group Standard
                    let mapped = pbr_neutral_tonemapping([r_in, g_in, b_in]);
                    (
                        mapped[0].clamp(0.0, 1.0),
                        mapped[1].clamp(0.0, 1.0),
                        mapped[2].clamp(0.0, 1.0),
                    )
                }
                TonemapOperator::None => (
                    r_in.clamp(0.0, 1.0),
                    g_in.clamp(0.0, 1.0),
                    b_in.clamp(0.0, 1.0),
                ),
            };

            ldr[0] = (r * 255.0) as u8;
            ldr[1] = (g * 255.0) as u8;
            ldr[2] = (b * 255.0) as u8;
            ldr[3] = (hdr[3].clamp(0.0, 1.0) * 255.0) as u8;
        });

    RgbaImage::from_raw(width, height, ldr_pixels).context("Failed to build RgbaImage")
}

pub fn calculate_difference_map(
    img1: &RgbaImage,
    img2: &RgbaImage,
    heatmap: bool,
) -> Result<RgbaImage> {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let resized_img2;
    let ref_img2 = if w1 != w2 || h1 != h2 {
        resized_img2 = image::imageops::resize(img2, w1, h1, image::imageops::FilterType::Triangle);
        &resized_img2
    } else {
        img2
    };

    let mut diff_img = RgbaImage::new(w1, h1);

    for (x, y, pixel) in diff_img.enumerate_pixels_mut() {
        let p1 = img1.get_pixel(x, y);
        let p2 = ref_img2.get_pixel(x, y);

        if heatmap {
            let diff_r = p1[0].abs_diff(p2[0]) as u16;
            let diff_g = p1[1].abs_diff(p2[1]) as u16;
            let diff_b = p1[2].abs_diff(p2[2]) as u16;
            let diff_a = p1[3].abs_diff(p2[3]) as u16;

            let intensity = ((diff_r + diff_g + diff_b + diff_a) / 4) as u8;

            // Simple blue-to-red heatmap mapping based on diff intensity
            let r = intensity;
            let g = 0;
            let b = 255 - intensity;

            *pixel = image::Rgba([r, g, b, 255]);
        } else {
            *pixel = image::Rgba([
                p1[0].abs_diff(p2[0]),
                p1[1].abs_diff(p2[1]),
                p1[2].abs_diff(p2[2]),
                255, // Always keep alpha fully opaque to clearly see RGB diffs
            ]);
        }
    }

    Ok(diff_img)
}
