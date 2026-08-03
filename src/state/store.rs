// src/state/store.rs

use slint::ComponentHandle;
use std::collections::HashSet;
use std::sync::Arc;

use crate::app::AppWindow;
use crate::state::models::{DuplicateFileSummary, DuplicateGroupSummary, QcIssueSummary};
use crate::utils::fs::normalize_path;

// ==========================================
// --- APP STATE DATA STRUCTURE -------------
// ==========================================

/// The global application state holding persistent scan results and active view layouts.
#[derive(Default)]
pub struct AppState {
    // Pure Domain Data (The absolute source of truth)
    pub groups: Vec<DuplicateGroupSummary>,
    pub qc_issues: Vec<QcIssueSummary>,
    pub inventory_files: Vec<DuplicateFileSummary>,

    // Lightweight UI States
    pub checked_paths: HashSet<String>,
    pub collapsed_groups: HashSet<i32>,
    pub sort_column: String,
    pub sort_ascending: bool,
}

impl AppState {
    /// Removes a specific file path across all state collections and clears its checked state.
    pub fn remove_path(&mut self, path: &str) {
        let normalized = normalize_path(path);
        self.checked_paths.remove(path);

        self.qc_issues
            .retain(|r| normalize_path(&r.path) != normalized);
        self.inventory_files
            .retain(|r| normalize_path(&r.path) != normalized);

        for group in &mut self.groups {
            group
                .files
                .retain(|r| normalize_path(&r.path) != normalized);
        }
        self.groups.retain(|g| g.files.len() >= 2);
    }

    /// Removes an entire duplicate group cluster by index and updates collapsed group indices.
    pub fn remove_group(&mut self, group_idx: i32) {
        let group_idx_us = group_idx as usize;
        if group_idx_us < self.groups.len() {
            let group = self.groups.remove(group_idx_us);
            for file in group.files {
                self.checked_paths.remove(&file.path);
            }

            let mut new_collapsed = HashSet::new();
            for &idx in &self.collapsed_groups {
                if idx < group_idx {
                    new_collapsed.insert(idx);
                } else if idx > group_idx {
                    new_collapsed.insert(idx - 1);
                }
            }
            self.collapsed_groups = new_collapsed;
        }
    }

    /// Removes all QC issues matching a specific issue category name.
    pub fn remove_qc_issue_type(&mut self, issue_type: &str) {
        let mut paths_to_uncheck = Vec::new();
        for issue in &self.qc_issues {
            if issue.issue == issue_type {
                paths_to_uncheck.push(issue.path.clone());
            }
        }
        for p in paths_to_uncheck {
            self.checked_paths.remove(&p);
        }

        self.qc_issues.retain(|r| r.issue != issue_type);
        self.collapsed_groups.clear();
    }
}

// ==========================================
// --- REACTIVE STATE STORE CONTAINER -------
// ==========================================

/// Thread-safe state store wrapper that automates Slint UI synchronization.
#[derive(Clone)]
pub struct AppStateStore {
    ui_weak: slint::Weak<AppWindow>,
    state: Arc<parking_lot::Mutex<AppState>>,
}

impl AppStateStore {
    pub fn new(ui_weak: slint::Weak<AppWindow>, state: Arc<parking_lot::Mutex<AppState>>) -> Self {
        Self { ui_weak, state }
    }

    /// Returns a thread-safe cloned reference to the inner Mutex.
    pub fn get_state_mutex(&self) -> Arc<parking_lot::Mutex<AppState>> {
        self.state.clone()
    }

    /// Provides read-only access to the state.
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&AppState) -> R,
    {
        let lock = self.state.lock();
        f(&lock)
    }

    /// Executes the mutation closure on the state lock, then automatically commits and flushes
    /// the resulting changes directly to Slint's results models inside the UI thread.
    pub fn update<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut AppState) -> R,
    {
        let mut lock = self.state.lock();
        let res = f(&mut lock);

        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<crate::app::ScanConfig>();
            crate::utils::slint_conversions::update_results_ui(&scan_config, &mut lock);
        }
        res
    }
}
