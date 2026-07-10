// src/utils/ui.rs

use crate::app::{AppWindow, ResultsRow, SelectedFile};
use crate::state::{AppState, DuplicateFileSummary, ResultsRowData};
use slint::{ModelRc, SharedPixelBuffer, VecModel};
use std::path::Path;
use std::rc::Rc;

/// Generates a tile-able checkerboard pattern inside memory directly
pub fn generate_checkerboard() -> slint::Image {
    let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(32, 32);
    let pixels = buffer.make_mut_slice();
    for y in 0..32 {
        for x in 0..32 {
            let is_dark_square = (x / 8 + y / 8) % 2 == 0;
            pixels[y * 32 + x] = if is_dark_square {
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
            };
        }
    }
    slint::Image::from_rgba8(buffer)
}

/// Converts an image::RgbaImage into a native slint::Image directly in memory.
pub fn convert_to_slint_image(rgba_img: &image::RgbaImage) -> slint::Image {
    let buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        rgba_img.as_raw(),
        rgba_img.width(),
        rgba_img.height(),
    );
    slint::Image::from_rgba8(buffer)
}

/// Maps thread-safe ResultsRowData into UI-bound Slint ResultsRow on the main thread
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

        // --- SEPARATED COLUMNS FOR ASSET INVENTORY ---
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

/// Helper mapping to translate UI visible row indices back to absolute state indices
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

/// Syncs the Slint ListView based on the expanded/collapsed group masks with pre-allocated buffer sizes.
/// Dynamically filters and maps best items into a clean flat grid model to guarantee seamless horizontal grid alignments.
pub fn update_results_ui(ui: &AppWindow, state: &AppState) {
    let search_query = ui.get_results_search_query().to_string().to_lowercase();
    let min_sim = ui.get_results_min_similarity();

    // Determine which groups contain files matching the search query
    let mut visible_groups = std::collections::HashSet::new();
    if !search_query.is_empty() {
        for row in &state.results {
            if !row.is_header && row.name.to_lowercase().contains(&search_query) {
                visible_groups.insert(row.group_index);
            }
        }
    }

    let mut slint_rows = Vec::with_capacity(state.results.len());
    let mut grid_rows = Vec::new();

    for row in &state.results {
        // 1. Text Search Filter
        if !search_query.is_empty() && !visible_groups.contains(&row.group_index) {
            continue; // Hide entire group if no children match
        }
        if !row.is_header
            && !search_query.is_empty()
            && !row.name.to_lowercase().contains(&search_query)
        {
            continue; // Hide non-matching children
        }

        // 2. Similarity Post-Filter (Only to non-best duplicates)
        if !row.is_header && !row.is_best && row.similarity > 0.0 && row.similarity < min_sim {
            continue;
        }

        // 3. Collect best representations of groups for Grid view,
        // ensuring the array is strictly contiguous and flat for level left-to-right mapping
        if !row.is_header && row.is_best {
            grid_rows.push(convert_to_slint_row(row));
        }

        // 4. List Collection & Collapse Logic
        if row.is_header {
            let mut slint_row = convert_to_slint_row(row);

            // Pass the collapsed state via is_checked so Slint can render the ▶/▼ arrows natively
            // instead of hardcoding ugly string overwrites into meta_str.
            slint_row.is_checked = state.collapsed_groups.contains(&row.group_index);

            slint_rows.push(slint_row);
        } else if !state.collapsed_groups.contains(&row.group_index) {
            slint_rows.push(convert_to_slint_row(row));
        }
    }

    ui.set_results(ModelRc::from(Rc::new(VecModel::from(slint_rows))));
    ui.set_grid_results(ModelRc::from(Rc::new(VecModel::from(grid_rows))));
}

pub fn get_current_active_channel(ui: &AppWindow) -> &'static str {
    if ui.get_active_r() {
        "R"
    } else if ui.get_active_g() {
        "G"
    } else if ui.get_active_b() {
        "B"
    } else if ui.get_active_a() {
        "A"
    } else {
        "RGB"
    }
}

/// Applies selection checkboxes rules to the overall state
pub fn apply_selection_rule(state: &mut AppState, rule: &str) {
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

    SelectedFile {
        name: slint::SharedString::from(name.as_ref()),
        size_str: slint::SharedString::from(crate::utils::helpers::format_size(file.size)),
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
