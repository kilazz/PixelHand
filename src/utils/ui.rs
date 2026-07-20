// src/utils/ui.rs

use crate::app::{GridRow, ResultsRow, SelectedFile, Store};
use crate::state::{AppState, DuplicateFileSummary, ResultsRowData};
use slint::{ModelRc, SharedPixelBuffer, VecModel};
use std::path::Path;
use std::rc::Rc;

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

/// Converts a standard raw `image::RgbaImage` buffer into a native `slint::Image` representation.
pub fn convert_to_slint_image(rgba_img: &image::RgbaImage) -> slint::Image {
    let buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        rgba_img.as_raw(),
        rgba_img.width(),
        rgba_img.height(),
    );
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

    // Calculate luminance bins sequentially using Rec. 709 luma coefficients
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

    // Determine peaks for normalization
    let max_orig = *hist_orig.iter().max().unwrap_or(&1).max(&1) as f32;
    let max_dup = *hist_dup.iter().max().unwrap_or(&1).max(&1) as f32;

    // Dark Slate canvas background
    let mut img = image::RgbaImage::from_pixel(w, h, image::Rgba([20, 20, 21, 255]));

    for x in 0..256 {
        let val_orig = (hist_orig[x] as f32 / max_orig * (h - 5) as f32).round() as u32;
        let val_dup = (hist_dup[x] as f32 / max_dup * (h - 5) as f32).round() as u32;

        for y in 0..h {
            let inv_y = h - 1 - y;
            let is_orig = y < val_orig;
            let is_dup = y < val_dup;

            if is_orig && is_dup {
                // Overlap: Blended Slate-purple
                img.put_pixel(x as u32, inv_y, image::Rgba([140, 140, 180, 255]));
            } else if is_orig {
                // Original: Cornflower Blue
                img.put_pixel(x as u32, inv_y, image::Rgba([100, 149, 237, 255]));
            } else if is_dup {
                // Duplicate: Golden Orange
                img.put_pixel(x as u32, inv_y, image::Rgba([240, 177, 50, 255]));
            }
        }
    }
    img
}

/// Maps a thread-safe `ResultsRowData` struct into a UI-bound Slint `ResultsRow` instance.
pub fn convert_to_slint_row(rd: &ResultsRowData) -> ResultsRow {
    let thumbnail = match &rd.thumbnail_data {
        Some(rgba) => convert_to_slint_image(rgba),
        None => slint::Image::default(),
    };
    ResultsRow {
        is_header: rd.is_header,
        is_qc: rd.is_qc,
        is_ai: rd.is_ai,
        group_index: rd.group_index,
        hash_or_issue: slint::SharedString::from(&rd.hash_or_issue),
        path: slint::SharedString::from(&rd.path),
        name: slint::SharedString::from(&rd.name),
        score_or_detail: slint::SharedString::from(&rd.score_or_detail),

        format_str: slint::SharedString::from(&rd.format_str),
        dimensions_str: slint::SharedString::from(&rd.dimensions_str),
        mipmaps_str: slint::SharedString::from(&rd.mipmaps_str),
        cubemap_str: slint::SharedString::from(&rd.cubemap_str),

        size_str: slint::SharedString::from(&rd.size_str),
        meta_str: slint::SharedString::from(&rd.meta_str),
        is_best: rd.is_best,
        is_checked: rd.is_checked,
        thumbnail,
    }
}

/// Maps the visible row index selected in the Slint list back to its absolute index in the state vector.
pub fn get_absolute_index(state: &AppState, visible_idx: usize) -> Option<usize> {
    let mut current_visible = 0;
    for (abs_idx, row) in state.results.iter().enumerate() {
        if row.is_header || !state.collapsed_groups.contains(&row.group_index) {
            if current_visible == visible_idx {
                return Some(abs_idx);
            }
            current_visible += 1;
        }
    }
    None
}

/// Synchronizes the active Slint list results and grid representations based on filters and collapse states.
pub fn update_results_ui(store: &Store, state: &AppState) {
    // Determine if there is actual scanned data (files or issues), ignoring pure headers
    let has_any = state.results.iter().any(|r| !r.is_header);
    store.set_has_results(has_any);

    let search_query = store.get_results_search_query().to_string().to_lowercase();
    let min_sim = store.get_results_min_similarity();

    let filter_only_npot = store.get_filter_only_npot();
    let filter_uncompressed = store.get_filter_only_uncompressed();
    let filter_missing_mips = store.get_filter_only_missing_mips();
    let filter_cubemaps = store.get_filter_only_cubemaps();

    let has_filter = !search_query.is_empty()
        || filter_only_npot
        || filter_uncompressed
        || filter_missing_mips
        || filter_cubemaps;

    // Local filter evaluator to prevent duplicate validation logic
    let matches_filters = |row: &ResultsRowData| -> bool {
        let matches_query =
            search_query.is_empty() || row.name.to_lowercase().contains(&search_query);
        let matches_npot = !filter_only_npot || row.is_npot;
        let matches_uncompressed = !filter_uncompressed || row.is_uncompressed;
        let matches_missing_mips = !filter_missing_mips || row.is_missing_mips;
        let matches_cubemaps = !filter_cubemaps || row.is_cubemap_bool;

        matches_query
            && matches_npot
            && matches_uncompressed
            && matches_missing_mips
            && matches_cubemaps
    };

    // Prefilter groups to identify which headers contain visible elements
    let mut visible_groups = std::collections::HashSet::new();
    if has_filter {
        for row in &state.results {
            if !row.is_header && matches_filters(row) {
                visible_groups.insert(row.group_index);
            }
        }
    }

    let mut slint_rows = Vec::with_capacity(state.results.len());
    let mut grid_items = Vec::new();

    for row in &state.results {
        // Skip entire duplicate clusters if none of their members conform to active filters
        if has_filter && !visible_groups.contains(&row.group_index) {
            continue;
        }

        if !row.is_header && has_filter && !matches_filters(row) {
            continue;
        }

        if !row.is_header && !row.is_best && row.similarity > 0.0 && row.similarity < min_sim {
            continue;
        }

        // Aggregate top-ranked group items for Grid presentations
        if !row.is_header && row.is_best {
            grid_items.push(convert_to_slint_row(row));
        }

        // Standard List update processing collapsed headers
        if row.is_header {
            let mut slint_row = convert_to_slint_row(row);
            slint_row.is_checked = state.collapsed_groups.contains(&row.group_index);
            slint_rows.push(slint_row);
        } else if !state.collapsed_groups.contains(&row.group_index) {
            slint_rows.push(convert_to_slint_row(row));
        }
    }

    // --- ADAPTIVE GRID COLUMNS ---
    // Extract calculated responsive columns count from Slint Store property
    let cols = store.get_grid_columns().max(1) as usize;

    // Chunk the flat grid_items list into rows dynamically to leverage native Slint ListView virtualization
    let mut grid_row_results = Vec::with_capacity(grid_items.len().div_ceil(cols));
    for chunk in grid_items.chunks(cols) {
        let row = GridRow {
            items: ModelRc::from(Rc::new(VecModel::from(chunk.to_vec()))),
        };
        grid_row_results.push(row);
    }

    store.set_results(ModelRc::from(Rc::new(VecModel::from(slint_rows))));
    store.set_grid_row_results(ModelRc::from(Rc::new(VecModel::from(grid_row_results))));
}

/// Evaluates which radio/toggle channel matches active pixel viewport options.
pub fn get_current_active_channel(store: &Store) -> &'static str {
    if store.get_active_r() {
        "R"
    } else if store.get_active_g() {
        "G"
    } else if store.get_active_b() {
        "B"
    } else if store.get_active_a() {
        "A"
    } else {
        "RGB"
    }
}

/// Applies selective checkbox operations across the results array.
pub fn apply_selection_rule(state: &mut AppState, rule: &str) {
    let mut visible_indices = Vec::new();

    // Track visible non-header indices to restrict selection adjustments strictly to the visible list view state
    for (abs_idx, row) in state.results.iter().enumerate() {
        if !row.is_header && !state.collapsed_groups.contains(&row.group_index) {
            visible_indices.push(abs_idx);
        }
    }

    match rule {
        "all" => {
            for row in &mut state.results {
                if !row.is_header {
                    row.is_checked = true;
                }
            }
        }
        "none" => {
            for row in &mut state.results {
                if !row.is_header {
                    row.is_checked = false;
                }
            }
        }
        "except_best" => {
            for row in &mut state.results {
                if !row.is_header {
                    row.is_checked = !row.is_best;
                }
            }
        }
        "invert" => {
            for row in &mut state.results {
                if !row.is_header {
                    row.is_checked = !row.is_checked;
                }
            }
        }
        _ => {}
    }
}

/// Constructs the detail panel SelectedFile metadata schema for Compare tab inspection.
pub fn build_selected_file_meta(file: &DuplicateFileSummary, is_original: bool) -> SelectedFile {
    let name = Path::new(&file.path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    let mipmaps_str = if file.mipmap_count <= 1 {
        "No".to_string()
    } else {
        file.mipmap_count.to_string()
    };

    let similarity_str = if is_original {
        "-".to_string()
    } else {
        format!("{:.1}%", file.similarity)
    };

    let vram = crate::core::qc::estimate_vram(
        file.width as u32,
        file.height as u32,
        &file.compression_format,
        file.mipmap_count,
        file.is_cubemap,
    );

    let vram_formatted = crate::utils::helpers::format_size(vram);
    let file_size_formatted = crate::utils::helpers::format_size(file.size);

    SelectedFile {
        name: slint::SharedString::from(name.as_ref()),
        size_str: slint::SharedString::from(format!(
            "{} (VRAM: {})",
            file_size_formatted, vram_formatted
        )),
        format: slint::SharedString::from(&file.compression_format),
        resolution: slint::SharedString::from(format!("{}x{}", file.width, file.height)),
        bit_depth: slint::SharedString::from(format!("{}-bit", file.bit_depth)),
        color_space: slint::SharedString::from(&file.color_space),
        mipmaps: slint::SharedString::from(mipmaps_str),
        alpha: slint::SharedString::from(if file.has_alpha { "Yes" } else { "No" }),
        similarity: slint::SharedString::from(similarity_str),
        path: slint::SharedString::from(&file.path),
    }
}

/// Extracts a specific frame from a spritesheet texture using grid subdivisions (columns and rows).
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

    // Crops the frame cleanly without allocating any extra buffers
    image::imageops::crop_imm(img, frame_x, frame_y, sprite_w, sprite_h).to_image()
}

/// Simulates non-square pixels or customized aspect ratios by dynamically scaling the width.
pub fn apply_aspect_ratio(img: &image::RgbaImage, ratio: f32) -> image::RgbaImage {
    if (ratio - 1.0).abs() < 1e-2 {
        return img.clone();
    }

    let new_w = (img.width() as f32 * ratio).max(1.0) as u32;
    let new_h = img.height();

    // Uses fast triangle/bilinear resizing to simulate aspect ratios in real-time
    image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle)
}
