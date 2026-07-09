// src/core/tonemapper.rs
use anyhow::{Context, Result};
use exr::prelude::*;
use image::RgbaImage;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static TONEMAP_ENABLED: AtomicBool = AtomicBool::new(true);
// 0: AcesFilmic, 1: ICtCpLumina
pub static TONEMAP_OPERATOR: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    None,
    AcesFilmic,
    ICtCpLumina,
}

// PQ Constants (ST.2084)
const PQ_N: f32 = 2610.0 / 4096.0 / 4.0;
const PQ_M: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
const PQ_MAX_NITS: f32 = 10000.0;

#[inline]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline]
fn lerp_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn max_vec3(a: [f32; 3], val: f32) -> [f32; 3] {
    [a[0].max(val), a[1].max(val), a[2].max(val)]
}

#[inline]
fn get_luma(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
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

fn linear_to_st2084(linear_nits: [f32; 3]) -> [f32; 3] {
    [
        nit_to_pq(linear_nits[0]),
        nit_to_pq(linear_nits[1]),
        nit_to_pq(linear_nits[2]),
    ]
}

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

fn compress_gamut(color: [f32; 3], threshold: f32, strength: f32) -> [f32; 3] {
    let luma = get_luma(color);
    let max_channel = color[0].max(color[1].max(color[2]));
    let saturation = max_channel - luma;
    let compression = ((saturation - threshold) * strength).clamp(0.0, 1.0);
    [
        lerp(color[0], luma, compression),
        lerp(color[1], luma, compression),
        lerp(color[2], luma, compression),
    ]
}

fn tonemap_lumina_op(c_color: [f32; 3], exposure: f32, scene_peak: f32) -> [f32; 3] {
    let clean_sample = compress_gamut(c_color, 0.8, 2.0);
    let clean_sample_nits = [
        clean_sample[0] * 100.0,
        clean_sample[1] * 100.0,
        clean_sample[2] * 100.0,
    ];

    let scene_peak_nits = scene_peak * 100.0 * exposure;

    let i_lin = get_luma(clean_sample_nits).max(1e-5);
    let i_pq = nit_to_pq(i_lin);

    let pq_source_max = nit_to_pq(scene_peak_nits.max(1.0));
    let pq_target_max = nit_to_pq(100.0);

    let shoulder_pivot = pq_target_max * 0.60;
    let toe_pivot = pq_target_max * 0.12;

    let t_toe = i_pq / toe_pivot.max(1e-5);
    let toe_val = toe_pivot * (t_toe + 0.15 * t_toe * (1.0 - t_toe));
    let mut mapped_i_pq = lerp(i_pq, toe_val, step(i_pq, toe_pivot));

    let mid_contrast = 0.06 * pq_target_max;
    let sin_input = (mapped_i_pq / pq_target_max.max(1e-5)) - 0.5;
    mapped_i_pq += mid_contrast * (sin_input * (1.0 - sin_input.abs()));

    let y_max = pq_target_max - shoulder_pivot;
    let x_max = (pq_source_max - shoulder_pivot).max(1e-5);
    let dx = (mapped_i_pq - shoulder_pivot).max(0.0);
    let shoulder_val = shoulder_pivot + y_max * (1.0 - (-(dx * 4.605) / x_max).exp());
    mapped_i_pq = lerp(mapped_i_pq, shoulder_val, step(shoulder_pivot, mapped_i_pq));
    mapped_i_pq = mapped_i_pq.clamp(0.0, 1.0);

    let chroma_t = ((mapped_i_pq / pq_target_max.max(1e-5) - 0.85) / 0.15).clamp(0.0, 1.0);
    let sat_scale = 1.0 - (chroma_t * chroma_t * (3.0 - 2.0 * chroma_t));
    let mut ictcp = rgb_to_ictcp(clean_sample_nits);
    ictcp[1] *= sat_scale;
    ictcp[2] *= sat_scale;
    ictcp[0] = mapped_i_pq;
    let pure_ictcp_rgb = max_vec3(ictcp_to_rgb(ictcp), 0.0);

    let mapped_i_lin_nits = pq_to_nit(mapped_i_pq);
    let mut gain = mapped_i_lin_nits / i_lin.max(1e-5);
    gain = lerp(1.0, gain, smoothstep(0.0, 0.02, i_lin / PQ_MAX_NITS));
    gain = gain.min(8.0);
    let rgb_mapped = [
        (clean_sample_nits[0] * gain).max(0.0),
        (clean_sample_nits[1] * gain).max(0.0),
        (clean_sample_nits[2] * gain).max(0.0),
    ];

    let normalized_pq = mapped_i_pq / pq_target_max.max(1e-5);
    let blend_factor = smoothstep(0.2, 0.8, normalized_pq);
    let output_nits = lerp_vec3(rgb_mapped, pure_ictcp_rgb, blend_factor * 0.75);

    [
        output_nits[0] / 100.0,
        output_nits[1] / 100.0,
        output_nits[2] / 100.0,
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
        if active_op == 1 {
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
                    let mapped = tonemap_lumina_op([r_in, g_in, b_in], 1.0, 10.0);
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

        let r_diff = (p1[0] as i16 - p2[0] as i16).unsigned_abs() as u8;
        let g_diff = (p1[1] as i16 - p2[1] as i16).unsigned_abs() as u8;
        let b_diff = (p1[2] as i16 - p2[2] as i16).unsigned_abs() as u8;
        let a_diff = (p1[3] as i16 - p2[3] as i16).unsigned_abs() as u8;

        if heatmap {
            let total_diff = (r_diff as u32 + g_diff as u32 + b_diff as u32 + a_diff as u32) / 4;
            let t = total_diff as f32 / 255.0;
            *pixel = image::Rgba([(t * 255.0) as u8, 0, ((1.0 - t) * 255.0) as u8, 255]);
        } else {
            *pixel = image::Rgba([r_diff, g_diff, b_diff, 255]);
        }
    }
    Ok(diff_img)
}
