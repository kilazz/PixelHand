// src/utils/ui.rs
use crate::app::{AppWindow, ResultsRow, SelectedFile};
use crate::state::{AppState, DuplicateFileSummary, ResultsRowData};
use slint::{ModelRc, SharedPixelBuffer, VecModel};
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
        hash_or_issue: rd.hash_or_issue.clone().into(),
        path: rd.path.clone().into(),
        name: rd.name.clone().into(),
        score_or_detail: rd.score_or_detail.clone().into(),
        size_str: rd.size_str.clone().into(),
        meta_str: rd.meta_str.clone().into(),
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

/// Syncs the Slint ListView based on the expanded/collapsed group masks
pub fn update_results_ui(ui: &AppWindow, state: &AppState) {
    let mut slint_rows = Vec::new();
    for row in &state.results {
        if row.is_header {
            let mut slint_row = convert_to_slint_row(row);
            slint_row.meta_str = if state.collapsed_groups.contains(&row.group_index) {
                "▶ [Collapsed] Click Header to Expand".into()
            } else {
                "▼ [Expanded] Click Header to Collapse".into()
            };
            slint_rows.push(slint_row);
        } else if !state.collapsed_groups.contains(&row.group_index) {
            slint_rows.push(convert_to_slint_row(row));
        }
    }
    ui.set_results(ModelRc::from(Rc::new(VecModel::from(slint_rows))));
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
        "Composite"
    }
}

/// Applies global selection rules to the table items (All, None, Except Best, Invert)
pub fn apply_selection_rule(state: &mut AppState, rule: &str) {
    for row in &mut state.results {
        if !row.is_header {
            match rule {
                "all" => row.is_checked = true,
                "none" => row.is_checked = false,
                "except_best" => row.is_checked = !row.is_best,
                "invert" => row.is_checked = !row.is_checked,
                _ => {}
            }
        }
    }
}

/// Maps an internal core Summary File into a Slint UI display Struct
pub fn build_selected_file_meta(file: &DuplicateFileSummary, is_original: bool) -> SelectedFile {
    let name = std::path::Path::new(&file.path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    SelectedFile {
        name: name.into(),
        size_str: super::helpers::format_size(file.size).into(),
        format: file.compression_format.clone().into(),
        resolution: format!("{}x{}", file.width, file.height).into(),
        bit_depth: format!("{}-bit", file.bit_depth).into(),
        color_space: file.color_space.clone().into(),
        mipmaps: (if file.mipmap_count <= 1 {
            "No".to_string()
        } else {
            file.mipmap_count.to_string()
        })
        .into(),
        alpha: (if file.has_alpha { "Yes" } else { "No" }).into(),
        similarity: if is_original {
            "-".into()
        } else {
            format!("{:.1}%", file.similarity).into()
        },
        path: file.path.clone().into(),
    }
}
