// src/utils/export.rs

use crate::state::AppState;
use crate::utils::helpers::MutexExt;
use std::sync::{Arc, Mutex};

/// Formats the active asset scan results and persists them to a CSV table via native save file dialogs.
pub fn export_results_to_csv(state: Arc<Mutex<AppState>>) {
    let lock = state.safe_lock();

    if lock.results.is_empty() {
        crate::app::append_to_console_log("No data available to export.");
        return;
    }

    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Asset Inventory to CSV")
        .add_filter("CSV File", &["csv"])
        .set_file_name("Asset_Inventory.csv")
        .save_file()
    {
        // Pre-allocate string capacity based on estimated row footprint to prevent memory re-allocations
        let mut csv_data = String::with_capacity(lock.results.len() * 150);
        csv_data.push_str("File Name,Format,Dimensions,MipMaps,Cubemap,Size,Full Path\n");

        for row in &lock.results {
            if row.is_header {
                continue;
            }

            // Escape quotes inside fields to comply with RFC 4180 CSV specifications
            let escaped_name = row.name.replace('"', "\"\"");
            let escaped_path = row.path.replace('"', "\"\"");

            // Directly format fields into the pre-allocated buffer avoiding intermediate heap allocations
            use std::fmt::Write;
            let _ = writeln!(
                &mut csv_data,
                "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
                escaped_name,
                row.format_str,
                row.dimensions_str,
                row.mipmaps_str,
                row.cubemap_str,
                row.size_str,
                escaped_path
            );
        }

        match std::fs::write(&path, csv_data) {
            Ok(_) => {
                crate::app::append_to_console_log(&format!(
                    "Successfully exported CSV report to: {:?}",
                    path
                ));
            }
            Err(e) => {
                crate::app::append_to_console_log(&format!(
                    "Failed to write CSV report to disk: {}",
                    e
                ));
            }
        }
    }
}

/// Persists the active UI diagnostic trace logs to a file on disk.
pub fn export_diagnostics_log(log_text: &str) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Console Log")
        .add_filter("Log Files", &["log", "txt"])
        .set_file_name("PixelHand_Diagnostics.log")
        .save_file()
    {
        match std::fs::write(&path, log_text) {
            Ok(_) => crate::app::append_to_console_log(&format!(
                "Console log diagnostics successfully exported to: {:?}",
                path
            )),
            Err(e) => crate::app::append_to_console_log(&format!(
                "Failed to write diagnostic logs to disk: {}",
                e
            )),
        }
    }
}
