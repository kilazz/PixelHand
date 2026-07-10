// src/utils/ui.rs

use crate::app::{AppWindow, GridRow, ResultsRow, SelectedFile};
use crate::state::{AppState, DuplicateFileSummary, ResultsRowData};
use slint::{ModelRc, SharedPixelBuffer, VecModel};
use std::path::Path;
use std::rc::Rc;

/// Generates a tileable dark checkerboard pattern directly in memory for transparent viewports.
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

/// Converts a standard raw `image::RgbaImage` buffer into a native `slint::Image` representation.
pub fn convert_to_slint_image(rgba_img: &image::RgbaImage) -> slint::Image {
    let buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        rgba_img.as_raw(),
        rgba_img.width(),
        rgba_img.height(),
    );
    slint::Image::from_rgba8(buffer)
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
pub fn update_results_ui(ui: &AppWindow, state: &AppState) {
    let search_query = ui.get_results_search_query().to_string().to_lowercase();
    let min_sim = ui.get_results_min_similarity();

    let filter_npot = ui.get_filter_only_npot();
    let filter_uncompressed = ui.get_filter_only_uncompressed();
    let filter_missing_mips = ui.get_filter_only_missing_mips();
    let filter_cubemaps = ui.get_filter_only_cubemaps();

    let has_filter = !search_query.is_empty()
        || filter_npot
        || filter_uncompressed
        || filter_missing_mips
        || filter_cubemaps;

    // Local filter evaluator to prevent duplicate validation logic
    let matches_filters = |row: &ResultsRowData| -> bool {
        let matches_query =
            search_query.is_empty() || row.name.to_lowercase().contains(&search_query);
        let matches_npot = !filter_npot || row.is_npot;
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

    // Chunk the flat grid_items list into rows of 4 columns to leverage native Slint ListView virtualization
    let mut grid_row_results = Vec::with_capacity(grid_items.len().div_ceil(4));
    for chunk in grid_items.chunks(4) {
        let row = GridRow {
            col1: chunk.first().cloned().unwrap_or_default(),
            col2: chunk.get(1).cloned().unwrap_or_default(),
            col3: chunk.get(2).cloned().unwrap_or_default(),
            col4: chunk.get(3).cloned().unwrap_or_default(),
            has_col1: !chunk.is_empty(),
            has_col2: chunk.get(1).is_some(),
            has_col3: chunk.get(2).is_some(),
            has_col4: chunk.get(3).is_some(),
        };
        grid_row_results.push(row);
    }

    ui.set_results(ModelRc::from(Rc::new(VecModel::from(slint_rows))));
    ui.set_grid_row_results(ModelRc::from(Rc::new(VecModel::from(grid_row_results))));
}

/// Evaluates which radio/toggle channel matches active pixel viewport options.
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

/// Applies selective checkbox operations across the results array.
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
