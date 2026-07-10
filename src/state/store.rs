// src/state/store.rs

use super::models::DuplicateGroupSummary;
use std::collections::HashSet;

/// A thread-safe, heap-allocated representation of Slint's ResultsRow structure.
/// Enables background workers to populate and filter rows concurrently without holding
/// the main Slint UI thread lock. Fields are grouped descending by memory footprint to minimize alignment padding.
#[derive(Clone)]
pub struct ResultsRowData {
    // 24-byte Pointer allocations (on 64-bit architectures)
    pub path: String,
    pub name: String,
    pub hash_or_issue: String,
    pub score_or_detail: String,
    pub format_str: String,
    pub dimensions_str: String,
    pub mipmaps_str: String,
    pub cubemap_str: String,
    pub size_str: String,
    pub meta_str: String, // Maintained for detailed multi-line fallback rows inside list modes
    pub thumbnail_data: Option<image::RgbaImage>,

    // 8-byte Numeric values
    pub size_bytes: u64,
    pub pixels_count: u64,

    // 4-byte Numeric values
    pub group_index: i32,
    pub similarity: f32,

    // 1-byte Logical flags
    pub is_header: bool,
    pub is_qc: bool,
    pub is_ai: bool,
    pub is_best: bool,
    pub is_checked: bool,
    pub is_npot: bool,
    pub is_uncompressed: bool,
    pub is_missing_mips: bool,
    pub is_cubemap_bool: bool,
}

/// The global application state holding our persistent scan results and active view layouts.
#[derive(Default)]
pub struct AppState {
    pub results: Vec<ResultsRowData>,
    pub groups: Vec<DuplicateGroupSummary>,
    pub collapsed_groups: HashSet<i32>,
    pub sort_column: String,
    pub sort_ascending: bool,
}
