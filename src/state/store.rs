// src/state/store.rs

use super::models::DuplicateGroupSummary;
use std::collections::HashSet;

/// A thread-safe shadow representation of Slint's ResultsRow.
/// We use this so our background threads can build rows without needing Slint's UI thread lock.
#[derive(Clone)]
pub struct ResultsRowData {
    pub is_header: bool,
    pub is_qc: bool,
    pub is_ai: bool,
    pub group_index: i32,
    pub hash_or_issue: String,
    pub path: String,
    pub name: String,
    pub score_or_detail: String,

    // --- SEPARATED COLUMNS FOR ASSET INVENTORY ---
    pub format_str: String,
    pub dimensions_str: String,
    pub mipmaps_str: String,
    pub cubemap_str: String,

    pub size_str: String,
    pub meta_str: String, // Kept for backwards compatibility with legacy views
    pub is_best: bool,
    pub is_checked: bool,
    pub thumbnail_data: Option<image::RgbaImage>, // Thread-safe raw pixels
    pub similarity: f32,                          // Utilized for real-time post-scan filtering

    // --- HIDDEN NUMERIC FIELDS FOR SORTING ---
    pub size_bytes: u64,
    pub pixels_count: u64,
}

/// The global application state holding our scan results and UI view states.
#[derive(Default)]
pub struct AppState {
    pub results: Vec<ResultsRowData>,
    pub groups: Vec<DuplicateGroupSummary>,
    pub collapsed_groups: HashSet<i32>, // Holds indexes of collapsed groups

    // --- FIELDS FOR COLUMN SORTING ---
    pub sort_column: String,
    pub sort_ascending: bool,
}
