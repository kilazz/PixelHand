// src/handlers/mod.rs

pub mod files;
pub mod scan;
pub mod ui_state;

use crate::app::AppWindow;
use crate::state::AppState;
use std::sync::Arc;

/// Registers all interactive callbacks and event bindings for the Slint UI Window.
pub fn register_callbacks(
    app: &AppWindow,
    state: Arc<parking_lot::Mutex<AppState>>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    files::bind_directory_selection(app);
    scan::bind_scan_execution(app, state.clone(), cancel_token);
    files::bind_file_actions(app, state.clone());
    ui_state::bind_ui_state_and_settings(app, state);
    files::bind_drag_and_drop(app);
}
