// src/state/store.rs

use slint::ComponentHandle;
use std::collections::HashSet;
use std::sync::Arc;

use crate::app::AppWindow;
use crate::state::models::{DuplicateFileSummary, DuplicateGroupSummary, QcIssueSummary};

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

// ==========================================
// --- REACTIVE STATE STORE CONTAINER -------
// ==========================================

/// Thread-safe state store wrapper that automates Slint UI synchronization.
/// Exposes mutability via an update closure which triggers update_results_ui on completion.
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

        // Automatically trigger UI update if the AppWindow is still alive
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<crate::app::ScanConfig>();
            crate::utils::slint_conversions::update_results_ui(&scan_config, &mut lock);
        }
        res
    }
}
