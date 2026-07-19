// src/core/tonemapper.rs

use exr::prelude::*;
use image::RgbaImage;
use rayon::prelude::*;
use std::path::Path;
use thiserror::Error;

/// Strongly typed errors for tonemapping operations.
#[derive(Error, Debug)]
pub enum TonemapError {
    #[error("EXR file decoding failed: {0}")]
    ExrReadFailed(String),
    #[error("Buffer size mismatch: expected {expected} floats, but got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    #[error("Failed to build RgbaImage from low-dynamic-range pixel buffer")]
    ImageBuildFailed,
}

/// Supported tonemapping operators and visualization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    None,
    FalseColor,
    AcesFilmic,
    Aces2Fit,
    PbrNeutral,
    ICtCpBt2446c,
    ICtCpLumina, // Custom
}

/// Pure data structure representing the active tonemapping settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TonemapConfig {
    pub enabled: bool,
    pub auto_exposure: bool,
    pub operator: TonemapOperator,
}

// -----------------------------------------------------------------------------
// CONSTANTS: Perceptual Quantizer (PQ) ST 2084 / ITU-R BT.2100
// -----------------------------------------------------------------------------
// These constants are mathematically defined to map linear light (nits)
// to a non-linear perceptual curve that mimics human vision.
const PQ_N: f32 = 2610.0 / 4096.0 / 4.0;
const PQ_M: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
/// The absolute maximum luminance (in nits) that the PQ curve can represent.
const PQ_MAX_NITS: f32 = 10000.0;

// -----------------------------------------------------------------------------
// MATH UTILITIES
// -----------------------------------------------------------------------------

/// Performs smooth Hermite interpolation between 0 and 1.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Calculates Rec.709 relative luminance from a linear RGB color.
#[inline]
fn get_luma(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

/// Calculates the automatic exposure scaling factor based on the middle-gray standard (18% Gray).
/// Uses a high-performance parallel logarithmic reduction to compute the geometric mean luminance.
pub fn calculate_auto_exposure(hdr_pixels: &[f32]) -> f32 {
    let total_pixels = hdr_pixels.len() / 4;
    if total_pixels == 0 {
        return 1.0;
    }

    // Sum log-luminance in parallel across multiple CPU cores
    let sum_log_lum = hdr_pixels
        .par_chunks_exact(4)
        .map(|hdr| {
            let y = hdr[0] * 0.2126 + hdr[1] * 0.7152 + hdr[2] * 0.0722;
            // Guard against NaN poisoning by clamping y before applying log2
            (y.max(0.0) + 1e-5).log2()
        })
        .sum::<f32>();

    let avg_log_lum = sum_log_lum / total_pixels as f32;
    let geometric_mean = avg_log_lum.exp2();

    // Standard photographic key value representing middle grey (18%)
    let key_value = 0.18;

    if geometric_mean > 1e-5 {
        // Clamp the final factor to prevent extreme blowing out of dark textures or over-crushing bright ones
        (key_value / geometric_mean).clamp(0.01, 100.0)
    } else {
        1.0
    }
}

// -----------------------------------------------------------------------------
// COLOR SPACE & TRANSFER FUNCTIONS
// -----------------------------------------------------------------------------

/// Converts absolute linear luminance (nits) to PQ non-linear signal value (0.0 to 1.0).
#[inline]
fn nit_to_pq(nits: f32) -> f32 {
    let y = nits / PQ_MAX_NITS;
    let y_pow = y.abs().powf(PQ_N);
    ((PQ_C1 + PQ_C2 * y_pow) / (1.0 + PQ_C3 * y_pow))
        .abs()
        .powf(PQ_M)
}

/// Converts a PQ non-linear signal value (0.0 to 1.0) back to absolute linear luminance (nits).
#[inline]
fn pq_to_nit(pq: f32) -> f32 {
    let y_pow = pq.abs().powf(1.0 / PQ_M);
    let num = (y_pow - PQ_C1).max(0.0);
    let den = PQ_C2 - (PQ_C3 * y_pow);
    (num / den).abs().powf(1.0 / PQ_N) * PQ_MAX_NITS
}

/// Helper function to apply the ST 2084 PQ curve across an RGB array.
#[inline]
fn linear_to_st2084(linear_nits: [f32; 3]) -> [f32; 3] {
    [
        nit_to_pq(linear_nits[0]),
        nit_to_pq(linear_nits[1]),
        nit_to_pq(linear_nits[2]),
    ]
}

/// Helper function to linearize an RGB array encoded with the ST 2084 PQ curve.
#[inline]
fn st2084_to_linear(pq: [f32; 3]) -> [f32; 3] {
    [pq_to_nit(pq[0]), pq_to_nit(pq[1]), pq_to_nit(pq[2])]
}

/// Applies the sRGB Opto-Electronic Transfer Function (OETF) to a linear signal.
/// Necessary for correct display of linear image buffers on standard LDR monitors.
#[inline]
fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Matrix multiplication to transform colors from Rec.709 (sRGB gamut) to Rec.2020 (Wide Gamut).
/// Required prior to entering Dolby/ITU standardized HDR color spaces.
#[inline]
fn rec709_to_rec2020(col: [f32; 3]) -> [f32; 3] {
    [
        col[0] * 0.627404 + col[1] * 0.329282 + col[2] * 0.0433136,
        col[0] * 0.069097 + col[1] * 0.91954 + col[2] * 0.0113612,
        col[0] * 0.0163916 + col[1] * 0.0880132 + col[2] * 0.895595,
    ]
}

/// Matrix multiplication to transform colors from Rec.2020 back to Rec.709.
#[inline]
fn rec2020_to_rec709(col: [f32; 3]) -> [f32; 3] {
    [
        col[0] * 1.660491 + col[1] * -0.587641 + col[2] * -0.0728499,
        col[0] * -0.12455 + col[1] * 1.1329 + col[2] * -0.0083494,
        col[0] * -0.0181508 + col[1] * -0.100579 + col[2] * 1.11873,
    ]
}

/// Converts linear Rec.2020 RGB values to the ICtCp perceptual color space.
/// ICtCp separates Intensity (I) from Chroma (Ct, Cp) specifically for HDR manipulation.
fn rgb_to_ictcp(col: [f32; 3]) -> [f32; 3] {
    // Step 1: LMS transformation matrix (Rec. 2020 to LMS color space)
    let l = col[0] * 0.412109 + col[1] * 0.523926 + col[2] * 0.063965;
    let m = col[0] * 0.166748 + col[1] * 0.720459 + col[2] * 0.112793;
    let s = col[0] * 0.024170 + col[1] * 0.075439 + col[2] * 0.900391;

    // Step 2: Apply non-linear PQ transfer function to LMS cone responses
    let pq_lms = linear_to_st2084([l.max(0.0), m.max(0.0), s.max(0.0)]);

    // Step 3: LMS to ICtCp matrix conversion
    let i = pq_lms[0] * 0.5000 + pq_lms[1] * 0.5000 + pq_lms[2] * 0.0000;
    let ct = pq_lms[0] * 1.6137 - pq_lms[1] * 3.3234 + pq_lms[2] * 1.7097;
    let cp = pq_lms[0] * 4.3780 - pq_lms[1] * 4.2455 - pq_lms[2] * 0.1325;

    [i, ct, cp]
}

/// Converts ICtCp color space values back to linear Rec.2020 RGB.
fn ictcp_to_rgb(col: [f32; 3]) -> [f32; 3] {
    // Step 1: ICtCp to LMS matrix conversion
    let l = col[0] * 1.0 + col[1] * 0.008605 + col[2] * 0.111036;
    let m = col[0] * 1.0 - col[1] * 0.008605 - col[2] * 0.111036;
    let s = col[0] * 1.0 + col[1] * 0.560049 - col[2] * 0.320637;

    // Step 2: Linearize LMS using the PQ transfer function inverse
    let lms = st2084_to_linear([l, m, s]);

    // Step 3: LMS to Rec.2020 RGB conversion matrix
    let r = lms[0] * 3.43661 - lms[1] * 2.50645 + lms[2] * 0.069845;
    let g = -lms[0] * 0.79133 + lms[1] * 1.9836 - lms[2] * 0.192271;
    let b = -lms[0] * 0.02595 - lms[1] * 0.098914 + lms[2] * 1.12486;

    [r, g, b]
}

// -----------------------------------------------------------------------------
// TONEMAPPERS & VISUALIZERS
// -----------------------------------------------------------------------------

/// False Color / Exposure Heatmap visualizer.
/// Maps linear luminance into a thermal-style color gradient to easily spot clipping.
/// Blue = Shadows, Grey/Green = Midtones, Red/White = Highlights/Clipping.
#[inline]
fn tonemap_false_color(color: [f32; 3]) -> [f32; 3] {
    let luma = get_luma(color);
    // Logarithmic mapping for better visual distribution of light stops
    let log_luma = ((luma.max(1e-5).log2() + 10.0) * 0.071428).clamp(0.0, 1.0);

    let r = smoothstep(0.5, 0.8, log_luma) + if log_luma >= 0.9 { 1.0 } else { 0.0 };
    let g = smoothstep(0.2, 0.5, log_luma) - smoothstep(0.7, 0.9, log_luma);
    let b = smoothstep(0.0, 0.2, log_luma) - smoothstep(0.4, 0.5, log_luma);

    [r, g, b]
}

/// The core forward tone scale curve for ACES 2.0.
/// Calculates the S-curve compression for luminance.
#[inline]
fn aces2_tone_scale_fwd(y: f32) -> f32 {
    let m_2 = 1.0470;
    let s_2 = 0.9078;
    let g = 1.1500;
    let t_1 = 0.0400;

    let f = m_2 * (y.max(0.0) / (y + s_2)).powf(g);
    (f * f / (f + t_1)).max(0.0)
}

/// ACES 2.0 Fit Tonemapper.
/// Replaces legacy ACES to fix hue skewing (e.g., blue neon turning magenta).
/// Operates in the ICtCp perceptual space to isolate luminance scaling from color.
#[inline]
fn tonemap_aces2_fit(color: [f32; 3]) -> [f32; 3] {
    // Scale input linear Rec.709 from 0..1 to 0..100 nits domain to match PQ absolute scale
    let color_scaled = [color[0] * 100.0, color[1] * 100.0, color[2] * 100.0];
    let color_2020 = rec709_to_rec2020(color_scaled);
    let mut ictcp = rgb_to_ictcp(color_2020);

    // Map PQ Intensity to equivalent Linear domain (0 to 100 nits scale)
    let i_linear = pq_to_nit(ictcp[0]) / 100.0;
    let i_mapped = aces2_tone_scale_fwd(i_linear);

    // Dynamic Chroma Compression: Desaturates extreme highlights towards white.
    // Linearly interpolates chroma scale from 1.1 down to 0.0 based on mapped intensity.
    // Optimized: x^1.5 is mathematically x * sqrt(x). Sqrt is hardware-accelerated.
    let i_m = i_mapped.max(0.0);
    let t = i_m * i_m.sqrt();
    let chroma_scale = (1.1 - 1.1 * t).clamp(0.0, 1.0);

    // Map Linear back to PQ Intensity and apply chroma scaling
    ictcp[0] = nit_to_pq(i_mapped * 100.0);
    ictcp[1] *= chroma_scale;
    ictcp[2] *= chroma_scale;

    // Return to output gamut
    let rgb_2020 = ictcp_to_rgb(ictcp);
    let rgb_709 = rec2020_to_rec709(rgb_2020);

    // Scale output from 0..100 nits back to standard linear 0..1 range
    [rgb_709[0] / 100.0, rgb_709[1] / 100.0, rgb_709[2] / 100.0]
}

/// Khronos PBR Neutral Tonemapper.
/// Formulated strictly to preserve base texture/albedo color values for 3D artists.
/// Prevents hue shifts at high intensities without adding cinematic contrast.
#[inline]
fn pbr_neutral_tonemapping(color_in: [f32; 3]) -> [f32; 3] {
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

/// Hermite spline function defined in ITU-R BT.2390 for dynamic range compression (EETF).
#[inline]
fn eetf_bt2390(i: f32, max_in_pq: f32, max_out_pq: f32) -> f32 {
    let ks = (1.5 * max_out_pq - 0.5 * max_in_pq).max(0.0);
    let b = max_in_pq - ks;

    if b <= 1e-5 {
        return i.min(max_out_pq);
    }

    if i < ks {
        i
    } else {
        let t = (i - ks) / b;
        let t2 = t * t;
        let t3 = t2 * t;

        let p0 = ks;
        let p1 = max_out_pq;
        let m0 = 1.0;
        let m1 = 0.0;

        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
            + (t3 - 2.0 * t2 + t) * m0 * b
            + (-2.0 * t3 + 3.0 * t2) * p1
            + (t3 - t2) * m1 * b
    }
}

/// Perceptual tonemapping based on ITU-R BT.2446 Method C.
/// Uses dynamic EETF luminance compression while applying Hunt Effect compensation for chroma.
/// Best suited for broadcasting standards ensuring linear shadow preservation.
fn tonemap_perceptual_bt2446c(color: [f32; 3], exposure: f32, scene_peak_nits: f32) -> [f32; 3] {
    let nits = [
        (color[0] * exposure * 100.0).max(0.0),
        (color[1] * exposure * 100.0).max(0.0),
        (color[2] * exposure * 100.0).max(0.0),
    ];

    // BT.2446c is mathematically formulated around Rec.2020.
    // We must transform Rec.709 linear inputs into Rec.2020 prior to processing.
    let nits_2020 = rec709_to_rec2020(nits);

    let mut ictcp = rgb_to_ictcp(nits_2020);
    let i_in = ictcp[0];

    let max_in_pq = nit_to_pq(scene_peak_nits);
    let max_out_pq = nit_to_pq(100.0); // Map to standard SDR display peak (100 nits)

    // Apply EETF compression solely to the intensity channel
    let i_out = eetf_bt2390(i_in, max_in_pq, max_out_pq);
    ictcp[0] = i_out;

    let compression_ratio = if i_in > 1e-5 { i_out / i_in } else { 1.0 };

    // Saturating compensation mimicking Hunt Effect adjustments
    let saturation_boost = 1.15;

    // Smooth path-to-white translation for highlights above the 90% target scale
    let normalized_i = (i_out / max_out_pq).clamp(0.0, 1.0);
    let path_to_white = 1.0 - smoothstep(0.90, 1.0, normalized_i);

    let final_chroma_scale = compression_ratio * saturation_boost * path_to_white;

    ictcp[1] *= final_chroma_scale;
    ictcp[2] *= final_chroma_scale;

    let rgb_sdr_nits_2020 = ictcp_to_rgb(ictcp);

    // Transform back to Rec.709 linear
    let rgb_sdr_nits_709 = rec2020_to_rec709(rgb_sdr_nits_2020);

    [
        (rgb_sdr_nits_709[0] / 100.0),
        (rgb_sdr_nits_709[1] / 100.0),
        (rgb_sdr_nits_709[2] / 100.0),
    ]
}

/// Custom ICtCp Lumina Tonemapper (Dolby Vision & Khronos PBR Inspired).
/// Uses PQ Reshaping (S-Curve on the perceptual PQ signal) for deep, natural contrast.
/// Preserves 100% exact perceptual saturation (Khronos PBR style) up to the highlights.
/// Employs a late, smooth exponential roll-off to white to prevent neon clipping.
fn tonemap_ictcp_lumina(color: [f32; 3], exposure: f32, scene_peak_nits: f32) -> [f32; 3] {
    let nits = [
        (color[0] * exposure * 100.0).max(0.0),
        (color[1] * exposure * 100.0).max(0.0),
        (color[2] * exposure * 100.0).max(0.0),
    ];

    let nits_2020 = rec709_to_rec2020(nits);
    let mut ictcp = rgb_to_ictcp(nits_2020);
    let i_in = ictcp[0];

    let max_in_pq = nit_to_pq(scene_peak_nits);
    let max_out_pq = nit_to_pq(100.0); // SDR Peak

    // Base HDR Dynamic Range Compression
    let mut i_out = eetf_bt2390(i_in, max_in_pq, max_out_pq);

    // PQ RESHAPING (The "Dolby Vision" Punch)
    // Normalize intensity from 0.0 to 1.0 in perceptual space
    let norm_i = (i_out / max_out_pq).clamp(0.0, 1.0);

    // Soft S-curve (Smoothstep) applied directly to the PQ signal.
    // Deepens shadows and makes highlights pop without breaking midtones.
    let s_curve = norm_i * norm_i * (3.0 - 2.0 * norm_i);
    let punch_factor = 0.35; // Cinematic contrast strength (35%)
    let norm_contrasted = norm_i * (1.0 - punch_factor) + s_curve * punch_factor;

    i_out = norm_contrasted * max_out_pq;
    ictcp[0] = i_out;

    // PERCEPTUAL SATURATION (The "PBR Neutral" Color Purity)
    // In ICtCp, saturation = Chroma / Intensity.
    // To keep the color exactly as vibrant as it was before luminance compression,
    // we scale it by EXACTLY the same ratio!
    let mut chroma_scale = if i_in > 1e-5 { i_out / i_in } else { 1.0 };

    // Hunt Effect compensation (colors in SDR appear less vibrant than in HDR)
    chroma_scale *= 1.15;

    // LATE HIGHLIGHT ROLL-OFF (Beautiful light sources)
    // Instead of a linear color fade, we use an exponential curve (norm_contrasted^4).
    // This keeps colors rich up to 85-90% brightness, and gently rolls them off to white only at the very end.
    let path_to_white = (1.0 - norm_contrasted.powf(4.0)).max(0.0);

    chroma_scale *= path_to_white;

    ictcp[1] *= chroma_scale;
    ictcp[2] *= chroma_scale;

    let rgb_sdr_nits_2020 = ictcp_to_rgb(ictcp);
    let rgb_sdr_nits_709 = rec2020_to_rec709(rgb_sdr_nits_2020);

    [
        (rgb_sdr_nits_709[0] / 100.0).clamp(0.0, 1.0),
        (rgb_sdr_nits_709[1] / 100.0).clamp(0.0, 1.0),
        (rgb_sdr_nits_709[2] / 100.0).clamp(0.0, 1.0),
    ]
}

/// Legacy ACES Filmic curve (Krzysztof Narkowicz approximation).
#[inline]
fn aces_tonemap_raw(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

// -----------------------------------------------------------------------------
// EXR & IMAGE BUFFER PROCESSING
// -----------------------------------------------------------------------------

/// Decodes the first RGBA floating-point layer of an EXR file into a flat `f32` vector.
/// Uses Arc<AtomicUsize> to safely coordinate dimensions across 'static closures.
pub fn load_exr_rgba(path: &Path) -> std::result::Result<(Vec<f32>, u32, u32), TonemapError> {
    let image_width = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let image_width_clone1 = image_width.clone();
    let image_width_clone2 = image_width.clone();

    let image = read_first_rgba_layer_from_file(
        path,
        move |size, _| {
            image_width_clone1.store(size.width(), std::sync::atomic::Ordering::Relaxed);
            vec![0.0f32; size.width() * size.height() * 4]
        },
        move |pixel_vector, position, (r, g, b, a): (f32, f32, f32, f32)| {
            let width = image_width_clone2.load(std::sync::atomic::Ordering::Relaxed);
            let index = (position.y() * width + position.x()) * 4;
            pixel_vector[index] = r;
            pixel_vector[index + 1] = g;
            pixel_vector[index + 2] = b;
            pixel_vector[index + 3] = a;
        },
    )
    .map_err(|e| TonemapError::ExrReadFailed(e.to_string()))?;

    let size = image.layer_data.size;
    Ok((
        image.layer_data.channel_data.pixels, // Fixed to extract correct underlying Vec<f32> from SpecificChannels
        size.width() as u32,
        size.height() as u32,
    ))
}

/// Maps high-dynamic-range (HDR) float pixels into low-dynamic-range (LDR) u8 buffers.
/// Applies the globally selected Tonemap operator, exposure compensation, and parallel multi-threading.
pub fn tonemap_hdr_to_ldr_rgba(
    hdr_pixels: &[f32],
    width: u32,
    height: u32,
    config: TonemapConfig,
    exposure: f32,
) -> std::result::Result<RgbaImage, TonemapError> {
    let total_pixels = (width * height) as usize;
    if hdr_pixels.len() != total_pixels * 4 {
        return Err(TonemapError::BufferSizeMismatch {
            expected: total_pixels * 4,
            actual: hdr_pixels.len(),
        });
    }

    let active_operator = if !config.enabled {
        TonemapOperator::None
    } else {
        config.operator
    };

    // Photographic Auto Exposure calculation using geometric mean luminance
    let mut active_exposure = exposure;
    if config.auto_exposure {
        // Multiplies default exposure with the automatic factor acting as Exposure Compensation offset
        active_exposure *= calculate_auto_exposure(hdr_pixels);
    }

    // Calculate dynamic scene peak luminance via a parallel pre-pass.
    // Required specifically for the BT2446c EETF curve scaling.
    let mut scene_peak_nits = 1000.0;
    if active_operator == TonemapOperator::ICtCpBt2446c
        || active_operator == TonemapOperator::ICtCpLumina
    {
        let max_lum = hdr_pixels
            .par_chunks_exact(4)
            .map(|hdr| {
                // Approximate relative luminance
                let y = hdr[0] * 0.2126 + hdr[1] * 0.7152 + hdr[2] * 0.0722;
                let val = y * active_exposure;
                // Guard against NaN/Infinity poisoning
                if val.is_finite() { val } else { 0.0 }
            })
            // Safe manual maximum to avoid f32::max propagating NaNs silently
            .reduce(|| 0.0_f32, |a, b| if a > b { a } else { b });

        if max_lum > 0.0 {
            scene_peak_nits = (max_lum * 100.0).max(100.0); // Clamp minimum to 100 nits
        }
    }

    let mut ldr_pixels = vec![0u8; total_pixels * 4];

    // Main pixel mapping pipeline (executed in parallel chunks)
    ldr_pixels
        .par_chunks_exact_mut(4)
        .zip(hdr_pixels.par_chunks_exact(4))
        .for_each(|(ldr, hdr)| {
            let r_in = hdr[0] * active_exposure;
            let g_in = hdr[1] * active_exposure;
            let b_in = hdr[2] * active_exposure;

            // Apply selected Tonemap Operator
            let (mut r, mut g, mut b) = match active_operator {
                TonemapOperator::None => (r_in, g_in, b_in),
                TonemapOperator::FalseColor => {
                    let mapped = tonemap_false_color([r_in, g_in, b_in]);
                    (mapped[0], mapped[1], mapped[2])
                }
                TonemapOperator::AcesFilmic => (
                    aces_tonemap_raw(r_in),
                    aces_tonemap_raw(g_in),
                    aces_tonemap_raw(b_in),
                ),
                TonemapOperator::Aces2Fit => {
                    let mapped = tonemap_aces2_fit([r_in, g_in, b_in]);
                    (mapped[0], mapped[1], mapped[2])
                }
                TonemapOperator::PbrNeutral => {
                    let mapped = pbr_neutral_tonemapping([r_in, g_in, b_in]);
                    (mapped[0], mapped[1], mapped[2])
                }
                TonemapOperator::ICtCpBt2446c => {
                    let mapped =
                        tonemap_perceptual_bt2446c([r_in, g_in, b_in], 1.0, scene_peak_nits);
                    (mapped[0], mapped[1], mapped[2])
                }
                TonemapOperator::ICtCpLumina => {
                    let mapped = tonemap_ictcp_lumina([r_in, g_in, b_in], 1.0, scene_peak_nits);
                    (mapped[0], mapped[1], mapped[2])
                }
            };

            // Apply standard sRGB Gamma correction to all linear outputs EXCEPT FalseColor
            if active_operator != TonemapOperator::FalseColor {
                r = linear_to_srgb(r);
                g = linear_to_srgb(g);
                b = linear_to_srgb(b);
            }

            // Strictly clamp colors to valid 0..1 range prior to integer casting
            r = r.clamp(0.0, 1.0);
            g = g.clamp(0.0, 1.0);
            b = b.clamp(0.0, 1.0);

            ldr[0] = (r * 255.0) as u8;
            ldr[1] = (g * 255.0) as u8;
            ldr[2] = (b * 255.0) as u8;
            ldr[3] = (hdr[3].clamp(0.0, 1.0) * 255.0) as u8;
        });

    RgbaImage::from_raw(width, height, ldr_pixels).ok_or(TonemapError::ImageBuildFailed)
}

/// Computes pixel differences between two Rgba buffers in parallel.
/// Used for visual regression testing and sub-pixel artifact spotting.
pub fn calculate_difference_map(
    img1: &RgbaImage,
    img2: &RgbaImage,
    heatmap: bool,
) -> std::result::Result<RgbaImage, anyhow::Error> {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    // Auto-align resolution if dimensions mismatch
    let resized_img2;
    let ref_img2 = if w1 != w2 || h1 != h2 {
        resized_img2 = image::imageops::resize(img2, w1, h1, image::imageops::FilterType::Triangle);
        &resized_img2
    } else {
        img2
    };

    let mut diff_img = RgbaImage::new(w1, h1);

    // Map pixel computations across all available CPU threads in parallel using highly optimized raw slices.
    // Removes the `get_pixel` boundary checks and index calculations entirely.
    diff_img
        .as_mut()
        .par_chunks_exact_mut(4)
        .zip(img1.as_raw().par_chunks_exact(4))
        .zip(ref_img2.as_raw().par_chunks_exact(4))
        .for_each(|((pixel_out, p1), p2)| {
            if heatmap {
                let diff_r = p1[0].abs_diff(p2[0]) as u16;
                let diff_g = p1[1].abs_diff(p2[1]) as u16;
                let diff_b = p1[2].abs_diff(p2[2]) as u16;
                let diff_a = p1[3].abs_diff(p2[3]) as u16;

                let intensity = ((diff_r + diff_g + diff_b + diff_a) / 4) as u8;

                // Map absolute differences into a blue-to-red diagnostic heatmap
                pixel_out[0] = intensity;
                pixel_out[1] = 0;
                pixel_out[2] = 255 - intensity;
                pixel_out[3] = 255;
            } else {
                // Direct RGB delta subtraction
                pixel_out[0] = p1[0].abs_diff(p2[0]);
                pixel_out[1] = p1[1].abs_diff(p2[1]);
                pixel_out[2] = p1[2].abs_diff(p2[2]);
                pixel_out[3] = 255;
            }
        });

    Ok(diff_img)
}
