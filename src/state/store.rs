// src/state/store.rs

use super::models::DuplicateGroupSummary;
use std::collections::{HashMap, HashSet};
use ustr::Ustr;

/// A thread-safe, heap-allocated representation of Slint's ResultsRow structure.
/// Enables background workers to populate and filter rows concurrently without holding
/// the main Slint UI thread lock. Fields are grouped descending by memory footprint to minimize alignment padding.
#[derive(Clone, Default)]
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

    // Highly optimized indices to guarantee O(1) mutations and prevent lock contention
    pub path_to_idx: HashMap<Ustr, usize>,
    pub visible_to_abs: Vec<usize>,
}

impl AppState {
    /// Rebuilds the fast path-to-index lookup hashmap.
    /// This runs in O(N) but takes less than 1ms for 20k entries due to pre-hashed Ustr keys.
    pub fn rebuild_path_to_idx(&mut self) {
        self.path_to_idx.clear();
        for (idx, row) in self.results.iter().enumerate() {
            if !row.is_header {
                let key = crate::utils::fs::normalize_path_key(&row.path);
                self.path_to_idx.insert(key, idx);
            }
        }
    }
}
