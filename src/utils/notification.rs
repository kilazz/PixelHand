// src/utils/notification.rs

use slint::ComponentHandle;

/// Centralized service to handle logging, UI console updates, and status banner changes safely across threads.
#[derive(Clone)]
pub struct NotificationService {
    ui_weak: slint::Weak<crate::app::AppWindow>,
}

impl NotificationService {
    /// Creates a new `NotificationService` instance from a weak handle to the AppWindow.
    pub fn new(ui_weak: slint::Weak<crate::app::AppWindow>) -> Self {
        Self { ui_weak }
    }

    /// Logs an informational message, appends it to the UI console,
    /// and updates the status banner text in the UI thread.
    pub fn notify_info(&self, message: &str) {
        tracing::info!("{}", message);
        crate::app::append_to_console_log(message);

        let msg = message.to_string();
        let _ = self.ui_weak.upgrade_in_event_loop(move |ui| {
            let diag = ui.global::<crate::app::Diagnostics>();
            diag.set_status_text(msg.into());
        });
    }

    /// Centralizes error handling: logs the error with its context to the tracing output,
    /// appends it to the UI console, and updates the status banner to indicate an error occurred.
    /// Uses anyhow's alternate formatter `{:#}` to capture and print the entire error cause chain.
    pub fn notify_error(&self, error: &anyhow::Error, context: &str) {
        let message = format!("[ERROR] {}: {:#}", context, error);
        tracing::error!("{}", message);
        crate::app::append_to_console_log(&message);

        let status = format!("Error: {}", context);
        let _ = self.ui_weak.upgrade_in_event_loop(move |ui| {
            let diag = ui.global::<crate::app::Diagnostics>();
            diag.set_status_text(status.into());
        });
    }

    /// Logs a success message, appends it to the UI console,
    /// and updates the status banner to reflect successful completion.
    pub fn notify_success(&self, message: &str) {
        tracing::info!("[SUCCESS] {}", message);
        crate::app::append_to_console_log(message);

        let msg = message.to_string();
        let _ = self.ui_weak.upgrade_in_event_loop(move |ui| {
            let diag = ui.global::<crate::app::Diagnostics>();
            diag.set_status_text(msg.into());
        });
    }
}
