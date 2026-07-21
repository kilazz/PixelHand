// src/handlers/mod.rs

pub mod actions;
pub mod scan;
pub mod viewer;

use slint::ComponentHandle;
use std::sync::Arc;

use crate::app::AppWindow;
use crate::state::AppState;
use crate::state::store::AppStateStore;
use crate::utils::notification::NotificationService;

pub struct AppController {
    scan_ctrl: Arc<scan::ScanController>,
    viewer_ctrl: Arc<viewer::ViewerController>,
    actions_ctrl: Arc<actions::ActionsController>,
}

impl AppController {
    pub fn new(
        ui: &AppWindow,
        state: Arc<parking_lot::Mutex<AppState>>,
        cancel_token: Arc<std::sync::atomic::AtomicBool>,
    ) -> Arc<Self> {
        let ui_weak = ui.as_weak();
        let notifier = Arc::new(NotificationService::new(ui_weak.clone()));

        // Initialize the reactive state store wrapper
        let store = AppStateStore::new(ui_weak.clone(), state);

        Arc::new(Self {
            scan_ctrl: scan::ScanController::new(
                ui_weak.clone(),
                store.clone(),
                cancel_token,
                notifier.clone(),
            ),
            viewer_ctrl: viewer::ViewerController::new(ui_weak.clone()),
            actions_ctrl: actions::ActionsController::new(ui_weak.clone(), store, notifier.clone()),
        })
    }

    /// Entry point to register all modular domain-specific UI callbacks.
    pub fn register_callbacks(&self) {
        let ui = self
            .scan_ctrl
            .ui_weak
            .upgrade()
            .expect("Failed to register callbacks: AppWindow is dead");

        self.scan_ctrl.register(&ui);
        self.viewer_ctrl.register(&ui);
        self.actions_ctrl.register(&ui);
    }
}
