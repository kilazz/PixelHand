// src/core/visuals.rs

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use fast_image_resize as fr;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::path::{Path, PathBuf};

use crate::state::DuplicateGroupSummary;

// Base layout constants to be scaled proportionally by the scale factor
const BASE_THUMB_SIZE: f32 = 300.0;
const BASE_PADDING: f32 = 25.0;
const BASE_TEXT_AREA: f32 = 120.0;
const BASE_FOOTER_HEIGHT: f32 = 40.0;

const MAX_IMGS_PER_GROUP: usize = 200;
const IMGS_PER_FILE: usize = 50;

const FONT_DATA: &[u8] = include_bytes!("../../assets/font.ttf");

/// Encapsulates geometric and sizing logic for visual contact sheets layout
struct SheetLayout {
    thumb_size: u32,
    padding: u32,
    text_area: u32,
    footer_height: u32,
    scale_bold: PxScale,
    scale_normal: PxScale,
}

impl SheetLayout {
    fn new(scale_factor: f32, font_size: f32) -> Self {
        Self {
            thumb_size: (BASE_THUMB_SIZE * scale_factor) as u32,
            padding: (BASE_PADDING * scale_factor) as u32,
            text_area: (BASE_TEXT_AREA * scale_factor) as u32,
            footer_height: (BASE_FOOTER_HEIGHT * scale_factor) as u32,
            scale_bold: PxScale::from((font_size + 2.0) * scale_factor),
            scale_normal: PxScale::from(font_size * scale_factor),
        }
    }

    /// Computes the exact dimensions of the target canvas
    fn calculate_canvas_dimensions(&self, cols: u32, rows: u32) -> (u32, u32) {
        let width = cols * (self.thumb_size + self.padding) + self.padding;
        let height = rows * (self.thumb_size + self.text_area + self.padding)
            + self.padding
            + self.footer_height;
        (width, height)
    }
}

/// Generates visual comparison reports (Contact Sheets) as PNGs on disk safely without unwraps
pub async fn generate_visual_reports(
    groups: Vec<DuplicateGroupSummary>,
    max_columns: usize,
    max_groups: usize,
    font_size: usize,
    scale_factor: f32,
    out_dir: PathBuf,
) -> Result<()> {
    if groups.is_empty() || max_columns == 0 || max_groups == 0 {
        return Ok(());
    }

    std::fs::create_dir_all(&out_dir).context("Failed to create visual reports directory")?;

    tokio::task::spawn_blocking(move || {
        let font = FontRef::try_from_slice(FONT_DATA)
            .expect("Failed to construct Font from compiled bytes");
        let layout = SheetLayout::new(scale_factor, font_size as f32);

        // Initialize the fast_image_resize resizer once per thread execution
        let mut resizer = fr::Resizer::new();

        let color_bg = Rgba([45, 45, 45, 255]);
        let color_text_white = Rgba([220, 220, 220, 255]);
        let color_text_gray = Rgba([180, 180, 180, 255]);
        let color_best = Rgba([240, 177, 50, 255]);
        let color_err_bg = Rgba([60, 60, 60, 255]);

        let mut sorted_groups = groups.clone();
        sorted_groups.sort_by_key(|b| std::cmp::Reverse(b.files.len()));

        let groups_to_process = sorted_groups.iter().take(max_groups).enumerate();

        for (group_idx, group) in groups_to_process {
            if group.files.is_empty() {
                continue;
            }

            let mut files = group.files.clone();
            files.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            files.truncate(MAX_IMGS_PER_GROUP);

            let total_pages = files.chunks(IMGS_PER_FILE).len();

            for (page_idx, chunk) in files.chunks(IMGS_PER_FILE).enumerate() {
                let cols = std::cmp::min(max_columns, chunk.len()) as u32;
                let rows = (chunk.len() as u32).div_ceil(cols);

                let (canvas_width, canvas_height) = layout.calculate_canvas_dimensions(cols, rows);
                let mut canvas = RgbaImage::from_pixel(canvas_width, canvas_height, color_bg);

                for (i, file) in chunk.iter().enumerate() {
                    let col = (i as u32) % cols;
                    let row = (i as u32) / cols;

                    let x = layout.padding + col * (layout.thumb_size + layout.padding);
                    let y = layout.padding
                        + row * (layout.thumb_size + layout.text_area + layout.padding);

                    // Decode high-resolution textures with fast_image_resize fallback handling
                    let thumb = match crate::format_loaders::open_image_with_dds_fallback(
                        Path::new(&file.path),
                        Some(layout.thumb_size),
                        None,
                    ) {
                        Ok(img) => {
                            let mut rgba_img = img.to_rgba8();

                            // Safely attempt raw image mapping and scaling without unwraps
                            let processed_img = fr::images::Image::from_slice_u8(
                                rgba_img.width(),
                                rgba_img.height(),
                                rgba_img.as_mut(),
                                fr::PixelType::U8x4,
                            )
                            .map_err(|e| anyhow::anyhow!("Failed to map image slice: {:?}", e))
                            .and_then(|src_image| {
                                let mut dst_image = fr::images::Image::new(
                                    layout.thumb_size,
                                    layout.thumb_size,
                                    fr::PixelType::U8x4,
                                );

                                let mut options = fr::ResizeOptions::new();
                                options = options.resize_alg(fr::ResizeAlg::Interpolation(
                                    fr::FilterType::Lanczos3,
                                ));

                                resizer
                                    .resize(&src_image, &mut dst_image, Some(&options))
                                    .map_err(|e| anyhow::anyhow!("Resize failed: {:?}", e))?;

                                RgbaImage::from_raw(
                                    layout.thumb_size,
                                    layout.thumb_size,
                                    dst_image.into_vec(),
                                )
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Failed to build RgbaImage from raw resized buffer"
                                    )
                                })
                            });

                            match processed_img {
                                Ok(thumb_img) => thumb_img,
                                Err(err) => {
                                    tracing::error!(
                                        "Failed to resize thumbnail for '{}': {}",
                                        file.path,
                                        err
                                    );
                                    let mut err_img = RgbaImage::from_pixel(
                                        layout.thumb_size,
                                        layout.thumb_size,
                                        color_err_bg,
                                    );
                                    draw_text_mut(
                                        &mut err_img,
                                        color_text_white,
                                        10,
                                        10,
                                        layout.scale_bold,
                                        &font,
                                        "Error Resizing Image",
                                    );
                                    err_img
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Visual Report: Failed to load thumbnail for '{}': {}",
                                file.path,
                                e
                            );
                            let mut err_img = RgbaImage::from_pixel(
                                layout.thumb_size,
                                layout.thumb_size,
                                color_err_bg,
                            );
                            draw_text_mut(
                                &mut err_img,
                                color_text_white,
                                10,
                                10,
                                layout.scale_bold,
                                &font,
                                "Error Loading Image",
                            );
                            err_img
                        }
                    };

                    let offset_x = x + (layout.thumb_size.saturating_sub(thumb.width())) / 2;
                    let offset_y = y + (layout.thumb_size.saturating_sub(thumb.height())) / 2;
                    image::imageops::overlay(&mut canvas, &thumb, offset_x.into(), offset_y.into());

                    // Compute text coordinate offsets based on layout bounds
                    let text_y = y + layout.thumb_size + (8.0 * scale_factor) as u32;
                    let display_name = Path::new(&file.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    draw_text_mut(
                        &mut canvas,
                        color_text_white,
                        x as i32,
                        text_y as i32,
                        layout.scale_bold,
                        &font,
                        &display_name,
                    );

                    let dist_str = if file.similarity >= 99.9 {
                        "[BEST REPRESENTATION]".to_string()
                    } else {
                        format!("Similarity: {:.1}%", file.similarity)
                    };
                    let status_color = if file.similarity >= 99.9 {
                        color_best
                    } else {
                        color_text_white
                    };
                    let y_offset_1 = text_y + (20.0 * scale_factor) as u32;
                    draw_text_mut(
                        &mut canvas,
                        status_color,
                        x as i32,
                        y_offset_1 as i32,
                        layout.scale_normal,
                        &font,
                        &dist_str,
                    );

                    let short_path = elide_text(&file.path, 40);
                    let y_offset_2 = text_y + (40.0 * scale_factor) as u32;
                    draw_text_mut(
                        &mut canvas,
                        color_text_gray,
                        x as i32,
                        y_offset_2 as i32,
                        layout.scale_normal,
                        &font,
                        &short_path,
                    );

                    let meta_str = format!(
                        "{}x{} | {}",
                        file.width, file.height, file.compression_format
                    );
                    let y_offset_3 = text_y + (60.0 * scale_factor) as u32;
                    draw_text_mut(
                        &mut canvas,
                        color_text_gray,
                        x as i32,
                        y_offset_3 as i32,
                        layout.scale_normal,
                        &font,
                        &meta_str,
                    );
                }

                // Draw footer line separator
                let footer_y = canvas_height - layout.footer_height;
                let line_thickness = (1.5 * scale_factor).max(1.0) as u32;
                let line_rect = Rect::at(0, footer_y as i32).of_size(canvas_width, line_thickness);
                draw_filled_rect_mut(&mut canvas, line_rect, Rgba([80, 80, 80, 255]));

                let start_idx = page_idx * IMGS_PER_FILE + 1;
                let end_idx = start_idx + chunk.len() - 1;
                let footer_text = format!(
                    "Group {} | Images {} to {} of {}",
                    group_idx + 1,
                    start_idx,
                    end_idx,
                    files.len()
                );
                let footer_text_y = footer_y + (10.0 * scale_factor) as u32;
                draw_text_mut(
                    &mut canvas,
                    color_text_gray,
                    layout.padding as i32,
                    footer_text_y as i32,
                    layout.scale_normal,
                    &font,
                    &footer_text,
                );

                let filename = if total_pages > 1 {
                    format!("group_{:03}_part_{}.png", group_idx + 1, page_idx + 1)
                } else {
                    format!("group_{:03}.png", group_idx + 1)
                };

                let out_path = out_dir.join(&filename);
                if let Err(e) = canvas.save(&out_path) {
                    tracing::error!(
                        "Failed to save visualization report page to '{}': {}",
                        out_path.display(),
                        e
                    );
                } else {
                    tracing::info!("Saved visual report page: {}", filename);
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })
    .await??;

    Ok(())
}

/// Safely truncates a string with character-level bounds checking to prevent UTF-8 boundary panics.
fn elide_text(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let keep = max_chars / 2 - 2;
    let first_part: String = text.chars().take(keep).collect();
    let second_part: String = text.chars().skip(char_count - keep).collect();
    format!("{}...{}", first_part, second_part)
}
