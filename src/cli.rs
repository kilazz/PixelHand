// src/cli.rs

use crate::state::models::{
    AiModelType, ScanAiSettings, ScanParams, ScanPaths, ScanPreprocessing, ScanQcRules,
    ScanVisualReports, SearchMethod,
};
use anyhow::{Result, anyhow};

/// Main CLI entry point.
/// Parses inputs and executes audits entirely in the console.
pub async fn run(args: Vec<String>) -> Result<()> {
    println!("==================================================");
    println!("           PixelHand - CLI Auditor Mode           ");
    println!("==================================================");

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    // Process Byte-Exact Scan Routing
    if let Some(pos) = args.iter().position(|arg| arg == "--scan-exact") {
        if pos + 1 < args.len() {
            let dir = args[pos + 1].clone();
            run_exact_cli_scan(dir).await?;
        } else {
            return Err(anyhow!("Missing directory path for --scan-exact"));
        }
        return Ok(());
    }

    // Process Quality Control (QC) Routing
    if let Some(pos) = args.iter().position(|arg| arg == "--scan-qc") {
        if pos + 1 < args.len() {
            let dir = args[pos + 1].clone();
            run_qc_cli_scan(dir, &args).await?;
        } else {
            return Err(anyhow!("Missing directory path for --scan-qc"));
        }
        return Ok(());
    }

    println!("[ERROR] Unknown CLI arguments. Use -h or --help for instructions.");
    Ok(())
}

// ---------------------------------------------------------
// PRIVATE CLI ROUTINES
// ---------------------------------------------------------

/// Prints the CLI usage menu.
fn print_help() {
    println!("Usage:");
    println!("  pixelhand -c --scan-exact <directory_path>");
    println!("  pixelhand -c --scan-qc <directory_path> [options]");
    println!("\nQC Options:");
    println!("  --check-npot          Verify if dimensions are Non-Power-of-Two");
    println!("  --check-mipmaps       Verify if mipmaps are generated");
    println!("  --check-block         Verify if dimensions are 4px block aligned");
    println!("  --check-bit           Verify bit depths");
    println!("  --validate-normals    Validate typical normal maps format");
}

/// Helper constructor to centralize default scan options configuration.
fn create_default_cli_params(dir: String) -> ScanParams {
    ScanParams {
        paths: ScanPaths {
            dir_a: dir,
            dir_b: String::new(),
            query_text: String::new(),
            excluded_folders: ".git, .svn, cache, temp".to_string(),
        },
        qc: ScanQcRules {
            qc_mode: false,
            qc_npot: false,
            qc_mipmaps: false,
            qc_block_align: false,
            qc_bit_depth: false,
            qc_solid_colors: false,
            qc_normals: false,
            qc_normal_target: 0,
            qc_normals_tags: String::new(),
            qc_match_by_stem: true,
            qc_hide_same_resolution: false,
            qc_check_bloat: true,
            qc_check_alpha: true,
            qc_check_colorspace: true,
            qc_check_compression: true,
        },
        visuals: ScanVisualReports {
            save_visuals: false,
            visuals_columns: 6,
            visuals_max_count: 100,
            visuals_font_size: 14,
            visuals_scale: 1.0,
        },
        prep: ScanPreprocessing {
            prep_luminance: false,
            prep_channels: false,
            prep_r: true,
            prep_g: true,
            prep_b: true,
            prep_a: true,
            prep_tags: String::new(),
            prep_ignore_solid: true,
        },
        ai: ScanAiSettings {
            search_precision: 1,
            ai_model: AiModelType::ClipVitB32,
            custom_model_path: String::new(),
            custom_model_arch: 0,
            custom_model_dim: 512,
        },

        similarity: 100.0,
        batch_size: 128,
        search_method: SearchMethod::Exact,
        execution_provider: "CPU".to_string(),
        extensions: vec![
            ".png".to_string(),
            ".jpg".to_string(),
            ".jpeg".to_string(),
            ".tga".to_string(),
            ".dds".to_string(),
            ".exr".to_string(),
            ".hdr".to_string(),
            ".tif".to_string(),
            ".tiff".to_string(),
        ],
        cancel_token: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        on_progress: None,
    }
}

/// Executes a byte-exact xxHash scan and prints results.
async fn run_exact_cli_scan(dir: String) -> Result<()> {
    println!("[CLI] Running Byte-Exact Scan (xxHash64) on: {}\n", dir);
    let params = create_default_cli_params(dir);

    match crate::exact::scanner::run_exact_scan(params).await {
        Ok(results) => {
            println!(
                "[SUCCESS] Exact Scan Completed! Found {} duplicate groups:",
                results.len()
            );
            for (idx, group) in results.iter().enumerate() {
                println!("  Group #{} (Hash: {})", idx + 1, group.hash);
                for file in &group.files {
                    println!(
                        "    - {} (Size: {} bytes, Dim: {}x{})",
                        file.path, file.size, file.width, file.height
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("[ERROR] Exact Scan Failed: {}", e);
        }
    }
    Ok(())
}

/// Executes Quality Control validations on a target directory.
async fn run_qc_cli_scan(dir: String, args: &[String]) -> Result<()> {
    let check_npot = args.iter().any(|arg| arg == "--check-npot");
    let check_mipmaps = args.iter().any(|arg| arg == "--check-mipmaps");
    let check_block = args.iter().any(|arg| arg == "--check-block");
    let check_bit = args.iter().any(|arg| arg == "--check-bit");
    let validate_normals = args.iter().any(|arg| arg == "--validate-normals");

    println!("[CLI] Running Technical Quality Control Scan on: {}", dir);
    println!(
        "      Options: NPOT={}, Mipmaps={}, BlockAlign={}, BitDepth={}, Normals={}\n",
        check_npot, check_mipmaps, check_block, check_bit, validate_normals
    );

    let mut params = create_default_cli_params(dir);
    params.similarity = 90.0;
    params.search_method = SearchMethod::Qc;
    params.qc.qc_mode = true;
    params.qc.qc_npot = check_npot;
    params.qc.qc_mipmaps = check_mipmaps;
    params.qc.qc_block_align = check_block;
    params.qc.qc_bit_depth = check_bit;
    params.qc.qc_normals = validate_normals;

    match crate::qc::scanner::run_qc_scan_internal(params).await {
        Ok(results) => {
            println!(
                "[SUCCESS] QC Scan Completed! Found {} issues:",
                results.len()
            );
            for issue in results {
                println!("  - File: {}", issue.path);
                println!("    Issue: {}", issue.issue);
                println!("    Details: {}", issue.details);
            }
        }
        Err(e) => {
            eprintln!("[ERROR] QC Scan Failed: {}", e);
        }
    }
    Ok(())
}
