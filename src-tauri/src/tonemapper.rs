// src-tauri/src/tonemapper.rs
use exr::prelude::*;
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::error::Error;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapOperator {
    AcesFilmic,
}

#[inline]
fn aces_tonemap_raw(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

pub fn load_exr_rgba(path: &Path) -> std::result::Result<(Vec<f32>, u32, u32), Box<dyn Error>> {
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
    )?;
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
) -> std::result::Result<RgbaImage, Box<dyn Error>> {
    let total_pixels = (width * height) as usize;
    if hdr_pixels.len() != total_pixels * 4 {
        return Err("Input float buffer size must equal width * height * 4 (RGBA)".into());
    }

    let mut ldr_pixels = vec![0u8; total_pixels * 4];

    ldr_pixels
        .par_chunks_exact_mut(4)
        .zip(hdr_pixels.par_chunks_exact(4))
        .for_each(|(ldr_pixel, hdr_pixel)| {
            let mut r = hdr_pixel[0] * exposure;
            let mut g = hdr_pixel[1] * exposure;
            let mut b = hdr_pixel[2] * exposure;
            let a = hdr_pixel[3];

            match operator {
                TonemapOperator::AcesFilmic => {
                    r = aces_tonemap_raw(r);
                    g = aces_tonemap_raw(g);
                    b = aces_tonemap_raw(b);
                }
            }

            ldr_pixel[0] = (r * 255.0) as u8;
            ldr_pixel[1] = (g * 255.0) as u8;
            ldr_pixel[2] = (b * 255.0) as u8;
            ldr_pixel[3] = (a.clamp(0.0, 1.0) * 255.0) as u8;
        });

    let image_buffer = RgbaImage::from_raw(width, height, ldr_pixels)
        .ok_or_else(|| Box::<dyn Error>::from("Failed to wrap LDR buffer into RgbaImage"))?;
    Ok(image_buffer)
}

pub fn calculate_difference_map(
    img1: &RgbaImage,
    img2: &RgbaImage,
    heatmap: bool,
) -> std::result::Result<RgbaImage, Box<dyn Error>> {
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

        let r_diff = (p1[0] as i16 - p2[0] as i16).abs() as u8;
        let g_diff = (p1[1] as i16 - p2[1] as i16).abs() as u8;
        let b_diff = (p1[2] as i16 - p2[2] as i16).abs() as u8;
        let a_diff = (p1[3] as i16 - p2[3] as i16).abs() as u8;

        if heatmap {
            let total_diff = (r_diff as u32 + g_diff as u32 + b_diff as u32 + a_diff as u32) / 4;
            let t = total_diff as f32 / 255.0;
            let r = (t * 255.0) as u8;
            let g = 0u8;
            let b = ((1.0 - t) * 255.0) as u8;
            *pixel = Rgba([r, g, b, 255]);
        } else {
            *pixel = Rgba([r_diff, g_diff, b_diff, 255]);
        }
    }
    Ok(diff_img)
}
