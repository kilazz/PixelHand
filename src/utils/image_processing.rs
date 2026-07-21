// src/utils/image_processing.rs

use slint::SharedPixelBuffer;

/// Generates a tileable dark or light checkerboard pattern directly in memory for transparent viewports.
pub fn generate_checkerboard(is_light: bool) -> slint::Image {
    let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(32, 32);
    let pixels = buffer.make_mut_slice();
    for y in 0..32 {
        for x in 0..32 {
            let is_first_square = (x / 8 + y / 8) % 2 == 0;
            pixels[y * 32 + x] = if is_light {
                if is_first_square {
                    slint::Rgba8Pixel {
                        r: 255,
                        g: 255,
                        b: 255,
                        a: 255,
                    } // White
                } else {
                    slint::Rgba8Pixel {
                        r: 200,
                        g: 200,
                        b: 200,
                        a: 255,
                    } // Light Gray
                }
            } else {
                if is_first_square {
                    slint::Rgba8Pixel {
                        r: 43,
                        g: 45,
                        b: 49,
                        a: 255,
                    }
                } else {
                    slint::Rgba8Pixel {
                        r: 60,
                        g: 63,
                        b: 65,
                        a: 255,
                    }
                }
            };
        }
    }
    slint::Image::from_rgba8(buffer)
}

/// Computes overlapping Grayscale/Luminance histograms of both original and duplicate textures,
/// rendering them on a dark background with semi-transparent overlapping areas.
pub fn generate_histogram_image(
    orig: &image::RgbaImage,
    dup: &image::RgbaImage,
) -> image::RgbaImage {
    let w = 256;
    let h = 100;
    let mut hist_orig = [0u32; 256];
    let mut hist_dup = [0u32; 256];

    for p in orig.pixels() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;
        let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as usize;
        if luma < 256 {
            hist_orig[luma] += 1;
        }
    }

    for p in dup.pixels() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;
        let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as usize;
        if luma < 256 {
            hist_dup[luma] += 1;
        }
    }

    let max_orig = *hist_orig.iter().max().unwrap_or(&1).max(&1) as f32;
    let max_dup = *hist_dup.iter().max().unwrap_or(&1).max(&1) as f32;

    let mut img = image::RgbaImage::from_pixel(w, h, image::Rgba([20, 20, 21, 255]));

    for x in 0..256 {
        let val_orig = (hist_orig[x] as f32 / max_orig * (h - 5) as f32).round() as u32;
        let val_dup = (hist_dup[x] as f32 / max_dup * (h - 5) as f32).round() as u32;

        for y in 0..h {
            let inv_y = h - 1 - y;
            let is_orig = y < val_orig;
            let is_dup = y < val_dup;

            if is_orig && is_dup {
                img.put_pixel(x as u32, inv_y, image::Rgba([140, 140, 180, 255]));
            } else if is_orig {
                img.put_pixel(x as u32, inv_y, image::Rgba([100, 149, 237, 255]));
            } else if is_dup {
                img.put_pixel(x as u32, inv_y, image::Rgba([240, 177, 50, 255]));
            }
        }
    }
    img
}

/// Slices a single frame from a tileable spritesheet grid.
pub fn slice_spritesheet_frame(
    img: &image::RgbaImage,
    cols: u32,
    rows: u32,
    frame: u32,
) -> image::RgbaImage {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let total_frames = cols * rows;
    let frame = frame % total_frames;

    let original_width = img.width();
    let original_height = img.height();
    let sprite_w = original_width / cols;
    let sprite_h = original_height / rows;

    if sprite_w == 0 || sprite_h == 0 {
        return img.clone();
    }

    let frame_x = (frame % cols) * sprite_w;
    let frame_y = (frame / cols) * sprite_h;

    image::imageops::crop_imm(img, frame_x, frame_y, sprite_w, sprite_h).to_image()
}

/// Applies manual aspect ratio resizing with bi-linear interpolation.
pub fn apply_aspect_ratio(img: &image::RgbaImage, ratio: f32) -> image::RgbaImage {
    if (ratio - 1.0).abs() < 1e-2 {
        return img.clone();
    }
    let new_w = (img.width() as f32 * ratio).max(1.0) as u32;
    let new_h = img.height();
    image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle)
}
