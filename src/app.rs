// src/app.rs

use anyhow::{Context, Result};
use slint::winit_030::WinitWindowAccessor;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

// Import state store settings
use crate::scanners;
use crate::state::{AppSettings, AppState};
use crate::utils;

// Generate Slint Rust code from the UI markup
slint::include_modules!();

// Global handle to push logs directly to the UI console from any thread
pub static APP_HANDLE: OnceLock<slint::Weak<AppWindow>> = OnceLock::new();

// Fast intermediate thread-safe queue to bypass immediate UI locked calls
static LOG_MESSAGES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

/// Custom tracing subscriber log handler wrapping native console pipes
struct UiLogWriter;

impl std::io::Write for UiLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(msg) = String::from_utf8(buf.to_vec()) {
            append_to_console_log(&msg);
        }

        // Write to physical log file on disk (thread-safe append)
        if let Some(file_mutex) = LOG_FILE.get()
            && let Ok(mut lock) = file_mutex.lock()
            && let Some(ref mut file) = *lock
        {
            let _ = file.write_all(buf);
            let _ = file.flush();
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(file_mutex) = LOG_FILE.get()
            && let Ok(mut lock) = file_mutex.lock()
            && let Some(ref mut file) = *lock
        {
            let _ = file.flush();
        }
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for UiLogWriter {
    type Writer = UiLogWriter;
    fn make_writer(&self) -> Self::Writer {
        UiLogWriter
    }
}

fn clean_sample_log(raw: String) -> String {
    raw.replace("\u{001b}[2m", "")
        .replace("\u{001b}[0m", "")
        .replace("\u{001b}[32m", "")
        .replace("\u{001b}[33m", "")
        .replace("\u{001b}[31m", "")
}

fn weak_upgrade_and_queue(weak: slint::Weak<AppWindow>, log_line: String) {
    let _ = weak.upgrade_in_event_loop(move |ui| {
        let current = ui.get_console_log().to_string();
        let next = if current.is_empty() {
            log_line
        } else {
            format!("{}\n{}", current, log_line)
        };
        ui.set_console_log(next.into());
    });
}

pub fn append_to_console_log(msg: &str) {
    let clean_msg = msg.trim_end().to_string();
    if clean_msg.is_empty() {
        return;
    }

    // Attempt direct main UI thread update if possible using collapsed conditionals
    if let Some(weak_handle) = APP_HANDLE.get()
        && let Some(_ui) = weak_handle.upgrade()
    {
        weak_upgrade_and_queue(weak_handle.clone(), clean_sample_log(clean_msg));
        return;
    }

    // Queue messages globally if UI is not fully instantiated yet
    let queue_mutex = LOG_MESSAGES.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut q) = queue_mutex.lock() {
        q.push(clean_msg);
    }
}

/// Main entry point for the GUI application
pub fn run_gui() -> Result<()> {
    // Initialize the physical log file inside the portable directory
    if let Ok(dir) = utils::settings::get_portable_app_data_dir() {
        let log_path = dir.join("PixelHand.log");
        if let Ok(file) = fs::OpenOptions::new()
            .create(true)
            .append(true) // Append logs so previous sessions are preserved (implicitly grants write permission)
            .open(log_path)
        {
            let _ = LOG_FILE.set(Mutex::new(Some(file)));
        }
    }

    // Init custom tracing to redirect ALL warnings and logs directly into our GUI and log file
    tracing_subscriber::fmt()
        .with_writer(UiLogWriter)
        .with_env_filter("info,ort=warn") // Show info from us, warnings from ONNX Runtime
        .with_ansi(false) // Disable ANSI color escapes to protect Slint layout parser and log files
        .init();

    let state = Arc::new(Mutex::new(AppState::default()));
    let cancel_token = Arc::new(std::sync::atomic::AtomicBool::new(false)); // Scan cancellation flag

    let app = AppWindow::new().context("Failed to initialize Slint UI Window")?;
    let _ = APP_HANDLE.set(app.as_weak());

    let loaded_settings = utils::settings::load_settings().unwrap_or_default();
    apply_settings_to_ui(&app, &loaded_settings);

    let checkerboard = utils::ui::generate_checkerboard();
    app.set_checkerboard_pattern(checkerboard);

    // Sync HDR Tonemapping global atomic states on application launch
    crate::core::tonemapper::TONEMAP_ENABLED
        .store(loaded_settings.tonemap_enabled, Ordering::Relaxed);
    crate::core::tonemapper::TONEMAP_OPERATOR
        .store(loaded_settings.tonemap_operator as usize, Ordering::Relaxed);

    // Flush any logs that accumulated before UI thread finished instantiating
    if let Some(queue_mutex) = LOG_MESSAGES.get()
        && let Ok(mut q) = queue_mutex.lock()
    {
        let app_weak_init = app.as_weak();
        let logs_to_flush = std::mem::take(&mut *q);
        if !logs_to_flush.is_empty() {
            let _ = app_weak_init.upgrade_in_event_loop(move |ui| {
                let mut current = ui.get_console_log().to_string();
                for line in logs_to_flush {
                    let cleaned = clean_sample_log(line);
                    if current.is_empty() {
                        current = cleaned;
                    } else {
                        current = format!("{}\n{}", current, cleaned);
                    }
                }
                ui.set_console_log(current.into());
            });
        }
    }

    // Start background log flushing loop (Updates UI at a steady 5Hz to prevent rendering stutter)
    let app_weak_log = app.as_weak();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            interval.tick().await;

            // Drain all pending logs atomically
            let pending: Vec<String> = {
                let queue = PENDING_LOGS.get_or_init(|| Mutex::new(Vec::new()));
                if let Ok(mut lock) = queue.lock() {
                    if lock.is_empty() {
                        continue;
                    }
                    std::mem::take(&mut *lock)
                } else {
                    Vec::new()
                }
            };

            if pending.is_empty() {
                continue;
            }

            let app_clone = app_weak_log.clone();
            let _ = app_clone.upgrade_in_event_loop(move |ui| {
                let current_log = ui.get_console_log().to_string();
                let mut lines: Vec<&str> = current_log.lines().collect();

                // Append the batch of new lines
                for p in &pending {
                    lines.push(p);
                }

                // Buffer Bounding: strictly keep only the last 200 lines to keep Slint TextEdit lightning-fast!
                if lines.len() > 200 {
                    let start = lines.len() - 200;
                    lines = lines[start..].to_vec();
                }

                let new_log = lines.join("\n");
                ui.set_console_log(new_log.into());
            });
        }
    });

    trigger_startup_model_download(app.as_weak());

    // ---------------------------------------------------------
    // BIND CALLBACKS
    // ---------------------------------------------------------

    let app_weak = app.as_weak();
    app.on_select_folder_a(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder A")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak.upgrade() {
                ui.set_dir_a(path_str.into());
                utils::settings::save_settings(&ui);
            }
        }
    });

    let app_weak = app.as_weak();
    app.on_select_folder_b(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder B")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak.upgrade() {
                ui.set_dir_b(path_str.into());
                utils::settings::save_settings(&ui);
            }
        }
    });

    let app_weak = app.as_weak();
    let state_clone = state.clone();
    let cancel_token_clone = cancel_token.clone();
    app.on_run_scan(move || {
        let app_copy = app_weak.clone();
        let state_copy = state_clone.clone();
        let ui = app_copy.unwrap();

        utils::settings::save_settings(&ui);

        // Reset cancellation
        cancel_token_clone.store(false, Ordering::Relaxed);
        let params = scanners::ScanParams::from_ui(&ui, cancel_token_clone.clone());

        ui.set_is_scanning(true);
        ui.set_status_text("Scanning assets...".into());
        ui.set_progress(0.0);
        tracing::info!("Starting scan on directory: {}", params.dir_a);

        let params_for_task = params.clone();
        let app_weak_download = app_weak.clone();

        tokio::spawn(async move {
            if params_for_task.search_method == 2
                && let Err(e) = crate::core::downloader::verify_and_download_models(
                    app_weak_download.clone(),
                    params_for_task.ai_model,
                )
                .await
            {
                let _ = app_weak_download.upgrade_in_event_loop(move |ui| {
                    ui.set_is_scanning(false);
                    ui.set_status_text(format!("AI Model download failed: {}", e).into());
                });
                return;
            }

            // Sync Tonemapping active choices inside background scanning thread
            // Bound to scoped block to ensure ref_ui is fully dropped prior to the execute_scan await boundary
            {
                if let Some(ref_ui) = app_weak_download.upgrade() {
                    crate::core::tonemapper::TONEMAP_ENABLED
                        .store(ref_ui.get_tonemap_enabled(), Ordering::Relaxed);
                    crate::core::tonemapper::TONEMAP_OPERATOR
                        .store(ref_ui.get_tonemap_operator() as usize, Ordering::Relaxed);
                }
            }

            let scan_result = scanners::execute_scan(params_for_task.clone()).await;

            // Generate visuals sheets if requested and scan completed successfully (Clippy Optimized Block)
            if params_for_task.save_visuals
                && let Ok((ref groups, _)) = scan_result
            {
                let _ = app_copy.upgrade_in_event_loop(|ui| {
                    ui.set_status_text("Generating contact sheet reports...".into());
                });

                if let Ok(app_dir) = crate::utils::settings::get_portable_app_data_dir() {
                    let out_dir = app_dir.join("duplicate_visuals");
                    if let Err(e) = crate::core::visuals::generate_visual_reports(
                        groups.clone(),
                        params_for_task.visuals_columns,
                        params_for_task.visuals_max_count,
                        params_for_task.visuals_font_size,
                        params_for_task.visuals_scale,
                        out_dir,
                    )
                    .await
                    {
                        tracing::error!("Failed to generate visual reports: {}", e);
                    }
                }
            }

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_is_scanning(false);
                ui.set_progress(1.0);
                match scan_result {
                    Ok((groups, rows)) => {
                        let mut state_lock = state_copy.lock().unwrap();
                        state_lock.collapsed_groups.clear();
                        state_lock.groups = groups;
                        state_lock.results = rows;

                        utils::ui::update_results_ui(&ui, &state_lock);

                        let msg = if params_for_task.save_visuals {
                            "Scan and visual reports finished successfully!"
                        } else {
                            "Scan finished successfully!"
                        };
                        ui.set_status_text(msg.into());
                        tracing::info!("Scan completed.");
                    }
                    Err(e) => {
                        ui.set_status_text(format!("Scan stopped: {}", e).into());
                        tracing::warn!("Scan interrupted: {}", e);
                    }
                }
            });
        });
    });

    // Cancel Button callback implementation
    let cancel_token_cancel = cancel_token.clone();
    app.on_cancel_scan(move || {
        cancel_token_cancel.store(true, Ordering::Relaxed);
        tracing::warn!("User requested scan cancellation. Waiting for threads to stop...");
    });

    // Open Original Image File callback implementation
    app.on_open_file_in_viewer(move |path| {
        let path_str = path.to_string();
        if !path_str.is_empty() {
            tracing::info!("Opening file in default viewer: {}", path_str);
            if let Err(e) = open::that(&path_str) {
                tracing::error!("Failed to open file: {}", e);
            }
        }
    });

    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_row_checkbox_toggled(move |idx| {
        let mut lock = state_clone.lock().unwrap();
        if let Some(abs_idx) = utils::ui::get_absolute_index(&lock, idx as usize)
            && let Some(row) = lock.results.get_mut(abs_idx)
        {
            row.is_checked = !row.is_checked;
        }
        if let Some(ui) = app_weak.upgrade() {
            utils::ui::update_results_ui(&ui, &lock);
        }
    });

    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_row_clicked(move |_idx, is_header, group_idx, path| {
        let app_copy = app_weak.clone();

        if is_header {
            let mut lock = state_clone.lock().unwrap();
            if lock.collapsed_groups.contains(&group_idx) {
                lock.collapsed_groups.remove(&group_idx);
            } else {
                lock.collapsed_groups.insert(group_idx);
            }
            if let Some(ui) = app_copy.upgrade() {
                utils::ui::update_results_ui(&ui, &lock);
            }
            return;
        }

        let path_str = path.to_string();
        let lock = state_clone.lock().unwrap();

        let group = match lock.groups.get(group_idx as usize) {
            Some(g) => g,
            None => return,
        };
        let original = match group.files.first() {
            Some(f) => f,
            None => return,
        };
        let duplicate = match group.files.iter().find(|f| f.path == path_str) {
            Some(f) => f,
            None => return,
        };

        let ui = app_copy.unwrap();
        ui.set_original_meta(utils::ui::build_selected_file_meta(original, true));
        ui.set_duplicate_meta(utils::ui::build_selected_file_meta(duplicate, false));

        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(utils::ui::convert_to_slint_row(row));
            }
        }
        ui.set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));
        trigger_viewport_update(
            app_weak.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    });

    let app_weak = app.as_weak();
    let state_copy = state.clone();
    app.on_trigger_action(move |action_type| {
        let app_copy = app_weak.clone();
        let state_copy_inner = state_copy.clone();
        let action = action_type.to_string();

        tokio::spawn(async move {
            let (checked_files, pairs) = {
                let lock = state_copy_inner.lock().unwrap();
                utils::fs::extract_selected_files(&lock)
            };

            if checked_files.is_empty() {
                return;
            }

            let _ = app_copy.upgrade_in_event_loop({
                let act = action.clone();
                move |ui| ui.set_status_text(format!("Processing selection: {}...", act).into())
            });

            let res = utils::fs::execute_file_action(&action, checked_files, pairs).await;

            let _ = app_copy.upgrade_in_event_loop(move |ui| match res {
                Ok(_) => {
                    ui.set_status_text(
                        format!("Successfully completed {} operation.", action).into(),
                    );
                    ui.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(Vec::new()))));
                    let mut lock = state_copy_inner.lock().unwrap();
                    lock.results.clear();
                    lock.groups.clear();
                    lock.collapsed_groups.clear();
                }
                Err(e) => {
                    ui.set_status_text(format!("Action failed: {}", e).into());
                    tracing::error!("Action failed: {}", e);
                }
            });
        });
    });

    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_trigger_selection_rule(move |rule| {
        let mut lock = state_clone.lock().unwrap();
        utils::ui::apply_selection_rule(&mut lock, rule.as_str());
        if let Some(ui) = app_weak.upgrade() {
            utils::ui::update_results_ui(&ui, &lock);
        }
    });

    let app_weak = app.as_weak();
    app.on_channel_toggled(move || {
        let app_copy = app_weak.clone();
        let ui = app_copy.unwrap();
        let orig_path = ui.get_original_meta().path.to_string();
        let dup_path = ui.get_duplicate_meta().path.to_string();

        if !orig_path.is_empty() && !dup_path.is_empty() {
            trigger_viewport_update(app_weak.clone(), orig_path, dup_path);
        }
    });

    // Handle generic saving request callback from Slint
    let app_weak_save = app.as_weak();
    app.on_save_settings(move || {
        if let Some(ui) = app_weak_save.upgrade() {
            utils::settings::save_settings(&ui);
        }
    });

    // Dynamic callback trigger linked to HDR Tonemapper sidebar panel selection updates
    let app_weak_tonemap = app.as_weak();
    app.on_tonemap_toggled(move || {
        let app_copy = app_weak_tonemap.clone();
        let ui = app_copy.unwrap();

        // Dynamically save modified settings to file
        utils::settings::save_settings(&ui);

        // Sync global atomic choices with background thread decoders
        crate::core::tonemapper::TONEMAP_ENABLED.store(ui.get_tonemap_enabled(), Ordering::Relaxed);
        crate::core::tonemapper::TONEMAP_OPERATOR
            .store(ui.get_tonemap_operator() as usize, Ordering::Relaxed);

        // Fully flush the decoded high-res preview cache so that EXR/DDS images instantly re-decode with the new tonemapper
        if let Some(cache_mutex) = utils::cache::DECODED_CACHE.get()
            && let Ok(mut cache) = cache_mutex.lock()
        {
            cache.clear();
        }

        let orig_path = ui.get_original_meta().path.to_string();
        let dup_path = ui.get_duplicate_meta().path.to_string();

        if !orig_path.is_empty() && !dup_path.is_empty() {
            trigger_viewport_update(app_weak_tonemap.clone(), orig_path, dup_path);
        }
    });

    let app_weak_dnd = app.as_weak();
    app.window().on_winit_window_event(move |_window, event| {
        if let slint::winit_030::winit::event::WindowEvent::DroppedFile(path_buf) = event {
            let path_str = path_buf.to_string_lossy().to_string();
            let app_copy = app_weak_dnd.clone();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_query_text(path_str.clone().into());
                ui.set_search_method(2);
                ui.set_status_text(format!("Reference image loaded: {}", path_str).into());
            });
        }
        slint::winit_030::EventResult::Propagate
    });

    app.run()
        .context("Slint event loop terminated with an error")?;
    Ok(())
}

fn apply_settings_to_ui(app: &AppWindow, settings: &AppSettings) {
    app.set_dir_a(settings.dir_a.clone().into());
    app.set_dir_b(settings.dir_b.clone().into());
    app.set_query_text(settings.query_text.clone().into());
    app.set_similarity_threshold(settings.similarity_threshold);
    app.set_batch_size(settings.batch_size);
    app.set_search_method(settings.search_method);
    app.set_execution_provider(settings.execution_provider);

    app.set_qc_mode(settings.qc_mode);
    app.set_qc_npot(settings.qc_npot);
    app.set_qc_mipmaps(settings.qc_mipmaps);
    app.set_qc_block_align(settings.qc_block_align);
    app.set_qc_bit_depth(settings.qc_bit_depth);
    app.set_qc_solid_colors(settings.qc_solid_colors);
    app.set_qc_normals(settings.qc_normals);
    app.set_qc_normals_tags(settings.qc_normals_tags.clone().into());

    app.set_ext_png(settings.ext_png);
    app.set_ext_tga(settings.ext_tga);
    app.set_ext_dds(settings.ext_dds);
    app.set_ext_bmp(settings.ext_bmp);
    app.set_ext_exr(settings.ext_exr);
    app.set_ext_hdr(settings.ext_hdr);
    app.set_ext_tif(settings.ext_tif);
    app.set_ext_webp(settings.ext_webp);

    app.set_duplicates_panel_height(settings.duplicates_panel_height);
    app.set_sidebar_width(settings.sidebar_width);

    // Apply Visual Reports configurations to Slint UI
    app.set_save_visuals(settings.save_visuals);
    app.set_visuals_columns(settings.visuals_columns);
    app.set_visuals_max_count(settings.visuals_max_count);
    app.set_visuals_font_size(settings.visuals_font_size);
    app.set_visuals_scale(settings.visuals_scale);

    // Apply Image Pre-processing configurations
    app.set_prep_luminance(settings.prep_luminance);
    app.set_prep_channels(settings.prep_channels);
    app.set_prep_r(settings.prep_r);
    app.set_prep_g(settings.prep_g);
    app.set_prep_b(settings.prep_b);
    app.set_prep_a(settings.prep_a);
    app.set_prep_tags(settings.prep_tags.clone().into());
    app.set_prep_ignore_solid(settings.prep_ignore_solid);

    // Apply Exclude Folders, QC match logic, AI Model index, and Search Precision level
    app.set_excluded_folders(settings.excluded_folders.clone().into());
    app.set_qc_match_by_stem(settings.qc_match_by_stem);
    app.set_qc_hide_same_resolution(settings.qc_hide_same_resolution);
    app.set_ai_model(settings.ai_model);
    app.set_search_precision(settings.search_precision);

    // Apply Tonemapping configurations to UI
    app.set_tonemap_enabled(settings.tonemap_enabled);
    app.set_tonemap_operator(settings.tonemap_operator);
}

fn trigger_startup_model_download(app_weak: slint::Weak<AppWindow>) {
    let app = app_weak.unwrap();
    let active_model = app.get_ai_model();

    tokio::spawn(async move {
        match crate::core::downloader::verify_and_download_models(app_weak.clone(), active_model)
            .await
        {
            Ok(_) => {
                let _ = app_weak.upgrade_in_event_loop(|ui| {
                    ui.set_status_text("AI models verified. System ready.".into());
                    ui.set_progress(1.0);
                });
            }
            Err(e) => {
                let _ = app_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_status_text(format!("Model verification failed: {}", e).into());
                    tracing::error!("Model download failed: {}", e);
                });
            }
        }
    });
}

fn trigger_viewport_update(app_weak: slint::Weak<AppWindow>, orig_path: String, dup_path: String) {
    let ui = app_weak.unwrap();
    let channel = utils::ui::get_current_active_channel(&ui).to_string();
    let compare_mode = ui.get_compare_mode();
    let app_weak_clone = app_weak.clone();

    // Dynamically align active Tonemapping state configs across threads
    crate::core::tonemapper::TONEMAP_ENABLED.store(ui.get_tonemap_enabled(), Ordering::Relaxed);
    crate::core::tonemapper::TONEMAP_OPERATOR
        .store(ui.get_tonemap_operator() as usize, Ordering::Relaxed);

    tokio::spawn(async move {
        if let Some(raw_orig) = utils::cache::get_channel_preview_image(&orig_path, &channel).await
        {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_original(utils::ui::convert_to_slint_image(&raw_orig));
            });
        }
        if let Some(raw_dup) = utils::cache::get_channel_preview_image(&dup_path, &channel).await {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_duplicate(utils::ui::convert_to_slint_image(&raw_dup));
            });
        }

        if compare_mode == 3
            && let Ok(diff_path) =
                crate::scanners::qc::calculate_diff_map(&orig_path, &dup_path).await
            && let Ok(diff_img) = image::open(&diff_path)
        {
            let raw_diff = diff_img.to_rgba8();
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_heatmap(utils::ui::convert_to_slint_image(&raw_diff));
            });
        }
    });
}

// Thread-safe fast intermediate queue implementation for incoming console log pipelines
static PENDING_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

// Thread-safe active log file handle mapping
pub static LOG_FILE: OnceLock<Mutex<Option<fs::File>>> = OnceLock::new();
