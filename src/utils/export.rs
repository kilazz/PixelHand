// src/utils/export.rs

use crate::state::AppState;
use std::sync::Arc;

/// Formats active asset scan results and persists them to a CSV table via native save file dialogs.
pub fn export_results_to_csv(state: Arc<parking_lot::Mutex<AppState>>) {
    let lock = state.lock();

    let is_empty =
        lock.groups.is_empty() && lock.qc_issues.is_empty() && lock.inventory_files.is_empty();
    if is_empty {
        crate::app::append_to_console_log("No data available to export.");
        return;
    }

    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Asset Inventory to CSV")
        .add_filter("CSV File", &["csv"])
        .set_file_name("Asset_Inventory.csv")
        .save_file()
    {
        let mut csv_data = String::with_capacity(1024 * 10);
        csv_data.push_str("File Name,Format,Dimensions,MipMaps,Cubemap,Size,Full Path\n");

        use std::fmt::Write;

        // Export Duplicate Groups
        for group in &lock.groups {
            for file in &group.files {
                let name = std::path::Path::new(&file.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy();
                let _ = writeln!(
                    &mut csv_data,
                    "\"{}\",\"{}\",\"{}x{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
                    name.replace('"', "\"\""),
                    file.compression_format,
                    file.width,
                    file.height,
                    file.mipmap_count,
                    if file.is_cubemap { "YES" } else { "NO" },
                    crate::utils::helpers::format_size(file.size),
                    file.path.replace('"', "\"\"")
                );
            }
        }

        // Export QC Issues
        for issue in &lock.qc_issues {
            let name = std::path::Path::new(&issue.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let _ = writeln!(
                &mut csv_data,
                "\"{}\",\"QC Issue: {}\",\"-\",\"-\",\"-\",\"-\",\"{}\"",
                name.replace('"', "\"\""),
                issue.issue,
                issue.path.replace('"', "\"\"")
            );
        }

        // Export Inventory Files
        for file in &lock.inventory_files {
            let name = std::path::Path::new(&file.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let _ = writeln!(
                &mut csv_data,
                "\"{}\",\"{}\",\"{}x{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
                name.replace('"', "\"\""),
                file.compression_format,
                file.width,
                file.height,
                file.mipmap_count,
                if file.is_cubemap { "YES" } else { "NO" },
                crate::utils::helpers::format_size(file.size),
                file.path.replace('"', "\"\"")
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

/// Persists active UI diagnostic trace logs to a file on disk.
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
