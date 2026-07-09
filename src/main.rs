// src/main.rs

mod app;
mod cli;
mod core;
mod format_loaders;
mod scanners;
mod state;
mod utils;

use anyhow::Result;
use std::env;

/// Global interceptor to capture and dump diagnostic crash logs
/// into the application folder structure prior to system termination.
fn setup_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            *s
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.as_str()
        } else {
            "Unknown panic message"
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

        eprintln!("{}", report);

        // Resolve portable application data path and save crash log file
        if let Ok(exe_path) = std::env::current_exe() {
            let exe_dir = exe_path.parent().unwrap_or(std::path::Path::new(""));
            let portable_data_dir = exe_dir.join("PixelHand_Data");
            let crash_dir = portable_data_dir.join("CrashLogs");

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
    // Instantiate panic crash hook first to catch any early startup errors
    setup_panic_hook();

    // Prevent ONNX Runtime from printing massive C++ execution provider logs (CPU fallbacks).
    // We set this globally before initializing any inference engines.
    unsafe {
        env::set_var("ORT_LOGGING_LEVEL", "WARNING");
        env::set_var("ORT_LOG_LEVEL", "WARNING");
    }

    // Collect command line arguments
    let args: Vec<String> = env::args().collect();

    // Route the application: CLI mode vs GUI mode
    let is_cli_mode = args
        .iter()
        .any(|arg| arg == "--cli" || arg == "-c" || arg == "--help" || arg == "-h");

    if is_cli_mode {
        // Start logging subscriber ONLY for CLI mode diagnostics
        tracing_subscriber::fmt::init();

        // Delegate completely to the CLI module
        cli::run(args).await?;
    } else {
        // Removed standard tracing subscriber initialization here!
        // We delegate completely to app::run_gui(), which registers its own custom
        // UiLogWriter subscriber to pipe log statements into the GUI's Log Tab.
        app::run_gui()?;
    }

    Ok(())
}
