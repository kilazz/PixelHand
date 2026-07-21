// src/utils/slint_conversions.rs

use crate::app::{GridRow, ResultsRow, ScanConfig, SelectedFile, ViewportState};
use crate::state::AppState;
use crate::state::models::{DuplicateFileSummary, QcIssueSummary, SearchMethod};
use crate::utils::cache::load_thumbnail_for_path;
use slint::{ModelRc, SharedPixelBuffer, VecModel};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

// ==========================================
// --- UTILITY SLINT CONVERSIONS ------------
// ==========================================

/// Returns the currently active channel string ("R", "G", "B", "A", or "RGB") from ViewportState.
pub fn get_current_active_channel(viewport_state: &ViewportState) -> &'static str {
    if viewport_state.get_active_r() {
        "R"
    } else if viewport_state.get_active_g() {
        "G"
    } else if viewport_state.get_active_b() {
        "B"
    } else if viewport_state.get_active_a() {
        "A"
    } else {
        "RGB"
    }
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

/// Maps a thread-safe domain summary directly into a UI-bound Slint `ResultsRow` instance.
pub fn convert_to_slint_row(
    file: &DuplicateFileSummary,
    is_best: bool,
    is_checked: bool,
    group_idx: i32,
) -> ResultsRow {
    let thumbnail = load_thumbnail_for_path(&file.path)
        .map(|img| convert_to_slint_image(&img))
        .unwrap_or_default();

    ResultsRow {
        is_header: false,
        is_qc: false,
        is_ai: false,
        group_index: group_idx,
        hash_or_issue: slint::SharedString::default(),
        path: slint::SharedString::from(&file.path),
        name: slint::SharedString::from(
            Path::new(&file.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        ),
        score_or_detail: slint::SharedString::from(if is_best {
            "[Best]".to_string()
        } else {
            format!("{:.1}%", file.similarity)
        }),
        format_str: slint::SharedString::from(&file.compression_format),
        dimensions_str: slint::SharedString::from(format!("{} x {}", file.width, file.height)),
        mipmaps_str: slint::SharedString::from(file.mipmap_count.to_string()),
        cubemap_str: slint::SharedString::from(if file.is_cubemap { "YES" } else { "NO" }),
        size_str: slint::SharedString::from(crate::utils::helpers::format_size(file.size)),
        meta_str: slint::SharedString::from(format!(
            "{}x{} • {} • {}-bit • Mips: {}",
            file.width, file.height, file.compression_format, file.bit_depth, file.mipmap_count
        )),
        is_best,
        is_checked,
        thumbnail,
    }
}

// ==========================================
// --- INTERFACE SYNCHRONIZATION ------------
// ==========================================

/// Synchronizes active Slint list results and grid representations based on filters and collapse states.
pub fn update_results_ui(scan_config: &ScanConfig, state: &mut AppState) {
    let search_method = scan_config.get_search_method();
    let is_empty = match search_method {
        SearchMethod::Qc => state.qc_issues.is_empty(),
        SearchMethod::Inventory => state.inventory_files.is_empty(),
        _ => state.groups.is_empty(),
    };
    scan_config.set_has_results(!is_empty);

    let search_query = scan_config
        .get_results_search_query()
        .to_string()
        .to_lowercase();
    let min_sim = scan_config.get_results_min_similarity();

    // Extract smart filters from the nested preview configuration inside ScanConfig
    let preview = scan_config.get_preview();
    let filter_only_npot = preview.filter_only_npot;
    let filter_uncompressed = preview.filter_only_uncompressed;
    let filter_missing_mips = preview.filter_only_missing_mips;
    let filter_cubemaps = preview.filter_only_cubemaps;

    let is_pow2 = |n: usize| n != 0 && (n & (n - 1)) == 0;

    let mut slint_rows = Vec::new();
    let mut grid_items = Vec::new();

    match search_method {
        SearchMethod::Qc => {
            // 1. Group and map QC issues on the fly
            let mut grouped_issues: HashMap<String, Vec<&QcIssueSummary>> = HashMap::new();
            for issue in &state.qc_issues {
                let filename = Path::new(&issue.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                if search_query.is_empty() || filename.contains(&search_query) {
                    grouped_issues
                        .entry(issue.issue.clone())
                        .or_default()
                        .push(issue);
                }
            }

            let mut sorted_types: Vec<String> = grouped_issues.keys().cloned().collect();
            sorted_types.sort();

            for (g_idx, issue_type) in sorted_types.iter().enumerate() {
                let group_issues = &grouped_issues[issue_type];

                slint_rows.push(ResultsRow {
                    is_header: true,
                    is_qc: true,
                    group_index: g_idx as i32,
                    hash_or_issue: issue_type.clone().into(),
                    size_str: format!("{} files", group_issues.len()).into(),
                    is_checked: state.collapsed_groups.contains(&(g_idx as i32)),
                    ..Default::default()
                });

                if state.collapsed_groups.contains(&(g_idx as i32)) {
                    continue;
                }

                for issue in group_issues {
                    let thumbnail = load_thumbnail_for_path(&issue.path)
                        .map(|img| convert_to_slint_image(&img))
                        .unwrap_or_default();
                    let is_checked = state.checked_paths.contains(&issue.path);

                    slint_rows.push(ResultsRow {
                        is_header: false,
                        is_qc: true,
                        group_index: g_idx as i32,
                        hash_or_issue: issue.issue.clone().into(),
                        path: issue.path.clone().into(),
                        name: Path::new(&issue.path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                            .into(),
                        score_or_detail: issue.issue.clone().into(),
                        meta_str: issue.details.clone().into(),
                        is_checked,
                        thumbnail,
                        ..Default::default()
                    });
                }
            }
        }
        SearchMethod::Inventory => {
            // 2. Map and filter flat Asset Inventory files on the fly
            for file in &state.inventory_files {
                let filename = Path::new(&file.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let matches_query =
                    search_query.is_empty() || filename.to_lowercase().contains(&search_query);
                let is_npot_bool = !is_pow2(file.width) || !is_pow2(file.height);
                let matches_npot = !filter_only_npot || is_npot_bool;
                let is_uncompressed_bool = file
                    .compression_format
                    .to_lowercase()
                    .contains("uncompressed");
                let matches_uncompressed = !filter_uncompressed || is_uncompressed_bool;
                let is_missing_mips_bool = file.mipmap_count <= 1;
                let matches_missing_mips = !filter_missing_mips || is_missing_mips_bool;
                let matches_cubemaps = !filter_cubemaps || file.is_cubemap;

                if matches_query
                    && matches_npot
                    && matches_uncompressed
                    && matches_missing_mips
                    && matches_cubemaps
                {
                    let thumbnail = load_thumbnail_for_path(&file.path)
                        .map(|img| convert_to_slint_image(&img))
                        .unwrap_or_default();
                    let is_checked = state.checked_paths.contains(&file.path);

                    slint_rows.push(ResultsRow {
                        is_header: false,
                        group_index: -1,
                        path: file.path.clone().into(),
                        name: filename.into(),
                        format_str: file.compression_format.clone().into(),
                        dimensions_str: format!("{} x {}", file.width, file.height).into(),
                        mipmaps_str: file.mipmap_count.to_string().into(),
                        cubemap_str: if file.is_cubemap { "YES" } else { "NO" }
                            .to_string()
                            .into(),
                        size_str: crate::utils::helpers::format_size(file.size).into(),
                        is_checked,
                        thumbnail,
                        ..Default::default()
                    });
                }
            }
        }
        _ => {
            // 3. Map, filter and group duplicate clusters on the fly
            for (g_idx, group) in state.groups.iter().enumerate() {
                let mut filtered_files = Vec::new();
                for (f_idx, file) in group.files.iter().enumerate() {
                    let is_best = f_idx == 0;

                    let matches_query =
                        search_query.is_empty() || file.path.to_lowercase().contains(&search_query);
                    let is_npot_bool = !is_pow2(file.width) || !is_pow2(file.height);
                    let matches_npot = !filter_only_npot || is_npot_bool;
                    let is_uncompressed_bool = file
                        .compression_format
                        .to_lowercase()
                        .contains("uncompressed");
                    let matches_uncompressed = !filter_uncompressed || is_uncompressed_bool;
                    let is_missing_mips_bool = file.mipmap_count <= 1;
                    let matches_missing_mips = !filter_missing_mips || is_missing_mips_bool;
                    let matches_cubemaps = !filter_cubemaps || file.is_cubemap;
                    let matches_similarity = is_best || file.similarity >= min_sim;

                    if matches_query
                        && matches_npot
                        && matches_uncompressed
                        && matches_missing_mips
                        && matches_cubemaps
                        && matches_similarity
                    {
                        filtered_files.push((file, is_best));
                    }
                }

                // Clusters require at least 2 files remaining after filters
                if filtered_files.len() < 2 {
                    continue;
                }

                slint_rows.push(ResultsRow {
                    is_header: true,
                    group_index: g_idx as i32,
                    hash_or_issue: group.hash.clone().into(),
                    size_str: crate::utils::helpers::format_size(
                        group.files.first().map(|f| f.size).unwrap_or(0),
                    )
                    .into(),
                    is_checked: state.collapsed_groups.contains(&(g_idx as i32)),
                    ..Default::default()
                });

                if state.collapsed_groups.contains(&(g_idx as i32)) {
                    continue;
                }

                for (file, is_best) in filtered_files {
                    let is_checked = state.checked_paths.contains(&file.path);
                    let row = convert_to_slint_row(file, is_best, is_checked, g_idx as i32);

                    if is_best {
                        grid_items.push(row.clone());
                    }
                    slint_rows.push(row);
                }
            }
        }
    }

    let cols = scan_config.get_grid_columns().max(1) as usize;
    let mut grid_row_results = Vec::with_capacity(grid_items.len().div_ceil(cols));
    for chunk in grid_items.chunks(cols) {
        let row = GridRow {
            items: ModelRc::from(Rc::new(VecModel::from(chunk.to_vec()))),
        };
        grid_row_results.push(row);
    }

    scan_config.set_results(ModelRc::from(Rc::new(VecModel::from(slint_rows))));
    scan_config.set_grid_row_results(ModelRc::from(Rc::new(VecModel::from(grid_row_results))));
}

/// Applies standard checkbox selection rules across checked_paths in the AppState.
pub fn apply_selection_rule(state: &mut AppState, rule: &str) {
    match rule {
        "all" => {
            state.checked_paths.clear();
            for issue in &state.qc_issues {
                state.checked_paths.insert(issue.path.clone());
            }
            for file in &state.inventory_files {
                state.checked_paths.insert(file.path.clone());
            }
            for group in &state.groups {
                for file in &group.files {
                    state.checked_paths.insert(file.path.clone());
                }
            }
        }
        "none" => {
            state.checked_paths.clear();
        }
        "except_best" => {
            state.checked_paths.clear();
            for group in &state.groups {
                if group.files.len() > 1 {
                    for file in group.files.iter().skip(1) {
                        state.checked_paths.insert(file.path.clone());
                    }
                }
            }
        }
        "invert" => {
            let mut all_paths = std::collections::HashSet::new();
            for issue in &state.qc_issues {
                all_paths.insert(issue.path.clone());
            }
            for file in &state.inventory_files {
                all_paths.insert(file.path.clone());
            }
            for group in &state.groups {
                for file in &group.files {
                    all_paths.insert(file.path.clone());
                }
            }

            let mut inverted = std::collections::HashSet::new();
            for path in all_paths {
                if !state.checked_paths.contains(&path) {
                    inverted.insert(path);
                }
            }
            state.checked_paths = inverted;
        }
        _ => {}
    }
}

/// Populates a `SelectedFile` DTO used inside the Comparison Specifications Matrix.
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

    let vram = crate::qc::rules::estimate_vram(
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
