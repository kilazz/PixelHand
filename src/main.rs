// src/main.rs

mod ai;
mod app;
mod cli;
mod exact;
mod format_loaders;
mod handlers;
mod perceptual;
mod qc;
mod reporting;
mod state;
mod utils;
mod viewer;

use anyhow::Result;
use std::env;

/// Registers a custom panic hook to catch and dump diagnostic crash reports
/// before standard thread termination and termination routines.
fn setup_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            *s
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.as_str()
        } else {
            "Unknown panic payload message"
        };

        let location = if let Some(loc) = panic_info.location() {
            format!("at {}:{}", loc.file(), loc.line())
        } else {
            "unknown location".to_string()
        };

        let backtrace = format!("{:?}", std::backtrace::Backtrace::capture());
        let report = format!(
            "==================================================\n\
             PIXELHAND CRASH DUMP REPORT\n\
             ==================================================\n\
             Panic message: {}\n\
             Location: {}\n\
             \n\
             Backtrace:\n\
             {}\n",
            message, location, backtrace
        );

        // Print report to standard error console
        eprintln!("{}", report);

        // Attempt to write the report to a physical log file in the portable data directory
        if let Ok(exe_path) = std::env::current_exe() {
            let exe_dir = exe_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new(""));
            let crash_dir = exe_dir.join("PixelHand_Data").join("CrashLogs");

            if std::fs::create_dir_all(&crash_dir).is_ok() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let file_path = crash_dir.join(format!("crash_report_{}.txt", now));
                let _ = std::fs::write(file_path, report);
            }
        }
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_panic_hook();

    // Safely initialize the global ONNX Runtime environment process-wide.
    let _ = ort::init().with_name("PixelHand").commit();

    let args: Vec<String> = env::args().collect();

    // Route between Terminal (CLI) Auditor and GUI Application
    let is_cli_mode = args
        .iter()
        .any(|arg| arg == "--cli" || arg == "-c" || arg == "--help" || arg == "-h");

    if is_cli_mode {
        // CLI logs stream natively to stdout via the standard tracing subscriber
        tracing_subscriber::fmt::init();
        cli::run(args).await?;
    } else {
        // GUI logs are captured and directed into the dedicated Console logging tab
        app::run_gui()?;
    }

    Ok(())
}
