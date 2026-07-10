// src/utils/export.rs

use crate::state::AppState;
use std::sync::{Arc, Mutex};

/// Formats the scan results (especially for Asset Audit) and saves them as a CSV file.
pub fn export_results_to_csv(state: Arc<Mutex<AppState>>) {
    let lock = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if lock.results.is_empty() {
        crate::app::append_to_console_log("No data to export.");
        return;
    }

    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Asset Inventory to CSV")
        .add_filter("CSV File", &["csv"])
        .set_file_name("Asset_Inventory.csv")
        .save_file()
    {
        let mut csv_data =
            String::from("File Name,Format,Dimensions,MipMaps,Cubemap,Size,Full Path\n");

        for row in &lock.results {
            if row.is_header {
                continue;
            }

            // Escape quotes inside fields to avoid CSV format breakage
            let line = format!(
                "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
                row.name.replace("\"", "\"\""),
                row.format_str,
                row.dimensions_str,
                row.mipmaps_str,
                row.cubemap_str,
                row.size_str,
                row.path.replace("\"", "\"\"")
            );
            csv_data.push_str(&line);
        }

        match std::fs::write(&path, csv_data) {
            Ok(_) => {
                crate::app::append_to_console_log(&format!(
                    "Successfully exported CSV to: {:?}",
                    path
                ));
            }
            Err(e) => {
                crate::app::append_to_console_log(&format!("Failed to export CSV: {}", e));
            }
        }
    }
}

/// Saves the console trace log output to a diagnostic file via a user file dialog.
pub fn export_diagnostics_log(log_text: &str) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Console Log")
        .add_filter("Log Files", &["log", "txt"])
        .set_file_name("PixelHand_Diagnostics.log")
        .save_file()
    {
        match std::fs::write(&path, log_text) {
            Ok(_) => crate::app::append_to_console_log(&format!(
                "Console log successfully exported to: {:?}",
                path
            )),
            Err(e) => {
                crate::app::append_to_console_log(&format!("Failed to export console log: {}", e))
            }
        }
    }
}
