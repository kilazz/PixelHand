// src/core/tonemapper.rs
use anyhow::{Context, Result};
use exr::prelude::*;
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    AcesFilmic,
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
    operator: TonemapOperator,
    exposure: f32,
) -> Result<RgbaImage> {
    let total_pixels = (width * height) as usize;
    if hdr_pixels.len() != total_pixels * 4 {
        return Err(anyhow::anyhow!("Buffer size mismatch"));
    }

    let mut ldr_pixels = vec![0u8; total_pixels * 4];

    ldr_pixels
        .par_chunks_exact_mut(4)
        .zip(hdr_pixels.par_chunks_exact(4))
        .for_each(|(ldr, hdr)| {
            let mut r = hdr[0] * exposure;
            let mut g = hdr[1] * exposure;
            let mut b = hdr[2] * exposure;

            if operator == TonemapOperator::AcesFilmic {
                r = aces_tonemap_raw(r);
                g = aces_tonemap_raw(g);
                b = aces_tonemap_raw(b);
            }

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
            *pixel = Rgba([(t * 255.0) as u8, 0, ((1.0 - t) * 255.0) as u8, 255]);
        } else {
            *pixel = Rgba([r_diff, g_diff, b_diff, 255]);
        }
    }
    Ok(diff_img)
}
