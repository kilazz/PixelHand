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

#[tokio::main]
async fn main() -> Result<()> {
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
        // Delegate completely to the CLI module
        cli::run(args).await?;
    } else {
        // Start logging subscriber for GUI mode diagnostics
        tracing_subscriber::fmt::init();

        // Delegate completely to the UI Controller module
        app::run_gui()?;
    }

    Ok(())
}
