// src/core/visuals.rs

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::{Rgba, RgbaImage, imageops::FilterType};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::path::{Path, PathBuf};

use crate::state::DuplicateGroupSummary;

// Nominal base dimensions to be scaled proportionally
const BASE_THUMB_SIZE: f32 = 300.0;
const BASE_PADDING: f32 = 25.0;
const BASE_TEXT_AREA: f32 = 120.0;
const BASE_FOOTER_HEIGHT: f32 = 40.0;

const MAX_IMGS_PER_GROUP: usize = 200;
const IMGS_PER_FILE: usize = 50;

const FONT_DATA: &[u8] = include_bytes!("../../assets/font.ttf");

/// Generates visual comparison reports (Contact Sheets) with custom scaling and font size
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
        let font = FontRef::try_from_slice(FONT_DATA).expect("Failed to construct Font from bytes");

        // Compute scaled geometric layout sizes
        let thumb_size = (BASE_THUMB_SIZE * scale_factor) as u32;
        let padding = (BASE_PADDING * scale_factor) as u32;
        let text_area = (BASE_TEXT_AREA * scale_factor) as u32;
        let footer_height = (BASE_FOOTER_HEIGHT * scale_factor) as u32;

        let scale_bold = PxScale::from((font_size as f32 + 2.0) * scale_factor);
        let scale_normal = PxScale::from(font_size as f32 * scale_factor);

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

                // Dynamically calculate canvas dimensions based on scaled elements
                let canvas_width = cols * (thumb_size + padding) + padding;
                let canvas_height =
                    rows * (thumb_size + text_area + padding) + padding + footer_height;

                let mut canvas = RgbaImage::from_pixel(canvas_width, canvas_height, color_bg);

                for (i, file) in chunk.iter().enumerate() {
                    let col = (i as u32) % cols;
                    let row = (i as u32) / cols;

                    let x = padding + col * (thumb_size + padding);
                    let y = padding + row * (thumb_size + text_area + padding);

                    // Decode and scale to the newly adjusted high-resolution thumbnail bounds
                    let thumb =
                        match crate::format_loaders::dds_loader::open_image_with_dds_fallback(
                            Path::new(&file.path),
                            Some(thumb_size),
                        ) {
                            Ok(img) => {
                                let resized =
                                    img.resize(thumb_size, thumb_size, FilterType::Lanczos3);
                                resized.to_rgba8()
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Visual Report: Failed to load {}: {}",
                                    file.path,
                                    e
                                );
                                let mut err_img =
                                    RgbaImage::from_pixel(thumb_size, thumb_size, color_err_bg);
                                draw_text_mut(
                                    &mut err_img,
                                    color_text_white,
                                    10,
                                    10,
                                    scale_bold,
                                    &font,
                                    "Error Loading",
                                );
                                err_img
                            }
                        };

                    let offset_x = x + (thumb_size.saturating_sub(thumb.width())) / 2;
                    let offset_y = y + (thumb_size.saturating_sub(thumb.height())) / 2;
                    image::imageops::overlay(&mut canvas, &thumb, offset_x.into(), offset_y.into());

                    // Compute scaled vertical offsets for metadata blocks
                    let text_y = y + thumb_size + (8.0 * scale_factor) as u32;
                    let display_name = Path::new(&file.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    draw_text_mut(
                        &mut canvas,
                        color_text_white,
                        x as i32,
                        text_y as i32,
                        scale_bold,
                        &font,
                        &display_name,
                    );

                    let dist_str = if file.similarity >= 99.9 {
                        "[BEST]".to_string()
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
                        scale_normal,
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
                        scale_normal,
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
                        scale_normal,
                        &font,
                        &meta_str,
                    );
                }

                // Dynamically scale the footer separator line thickness
                let footer_y = canvas_height - footer_height;
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
                    padding as i32,
                    footer_text_y as i32,
                    scale_normal,
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
                        "Failed to save visualization report to {}: {}",
                        out_path.display(),
                        e
                    );
                } else {
                    tracing::info!("Saved visual report: {}", filename);
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })
    .await??;

    Ok(())
}

fn elide_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let keep = max_chars / 2 - 2;
    format!("{}...{}", &text[0..keep], &text[text.len() - keep..])
}
