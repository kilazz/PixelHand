// src/app.rs
use anyhow::{Context, Result};
use slint::winit_030::WinitWindowAccessor;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::sync::{Arc, Mutex, OnceLock};

// Import state store settings (SelectedFile is locally brought into scope by include_modules)
use crate::scanners;
use crate::state::{AppSettings, AppState};
use crate::utils;

// Generate Slint Rust code from the UI markup
slint::include_modules!();

// Global handle to push logs directly to the UI console from any thread
pub static APP_HANDLE: OnceLock<slint::Weak<AppWindow>> = OnceLock::new();

/// Appends a message to the Slint UI log console safely from any thread.
pub fn append_to_console_log(msg: &str) {
    if let Some(app_weak) = APP_HANDLE.get() {
        let msg_clone = msg.to_string();
        let _ = app_weak.upgrade_in_event_loop(move |ui| {
            let current_log = ui.get_console_log().to_string();
            ui.set_console_log(format!("{}\n{}", current_log, msg_clone).into());
        });
    } else {
        println!("[WARNING] {}", msg);
    }
}

/// Main entry point for the GUI application
pub fn run_gui() -> Result<()> {
    // Thread-safe state holder for the list updates
    let state = Arc::new(Mutex::new(AppState::default()));

    // Instantiate Slint AppWindow
    let app = AppWindow::new().context("Failed to initialize Slint UI Window")?;
    let _ = APP_HANDLE.set(app.as_weak());

    // Load persistent settings
    let loaded_settings = utils::settings::load_settings().unwrap_or_default();
    apply_settings_to_ui(&app, &loaded_settings);

    // Generate checkerboard texture for transparent backgrounds
    let checkerboard = utils::ui::generate_checkerboard();
    app.set_checkerboard_pattern(checkerboard);

    // Auto-download AI models on startup
    trigger_startup_model_download(app.as_weak());

    // ---------------------------------------------------------
    // BIND CALLBACKS
    // ---------------------------------------------------------

    // 1. Browse Folder A
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

    // 2. Browse Folder B
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

    // 3. Scan Runner Callback
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_run_scan(move || {
        let app_copy = app_weak.clone();
        let state_copy = state_clone.clone();
        let ui = app_copy.unwrap();

        utils::settings::save_settings(&ui); // Save current settings locally

        // Extract parameters from UI
        let params = scanners::ScanParams::from_ui(&ui);

        ui.set_is_scanning(true);
        ui.set_status_text("Initializing graphical scan...".into());

        // Spawn background task to prevent freezing the UI
        tokio::spawn(async move {
            let scan_result = scanners::execute_scan(params).await;

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_is_scanning(false);
                match scan_result {
                    Ok((groups, rows)) => {
                        let mut state_lock = state_copy.lock().unwrap();
                        state_lock.collapsed_groups.clear();
                        state_lock.groups = groups;
                        state_lock.results = rows;

                        utils::ui::update_results_ui(&ui, &state_lock);
                        ui.set_status_text("Scan finished successfully!".into());
                    }
                    Err(e) => {
                        ui.set_status_text(format!("Scan failed: {}", e).into());
                    }
                }
            });
        });
    });

    // 4. File Table Checkbox Toggled Callback
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

    // 5. File Table Row Click Selection Callback (Expand/Collapse & Viewports)
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_row_clicked(move |_idx, is_header, group_idx, path| {
        let app_copy = app_weak.clone();

        if is_header {
            // Interactive collapse/expand toggle on header row click
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

        // Update Specifications Matrix
        ui.set_original_meta(utils::ui::build_selected_file_meta(original, true));
        ui.set_duplicate_meta(utils::ui::build_selected_file_meta(duplicate, false));

        // Map duplicates list to Slint right-hand compare miniature ListView
        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(utils::ui::convert_to_slint_row(row));
            }
        }
        ui.set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));

        // Load images into viewports
        trigger_viewport_update(
            app_weak.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    });

    // 6. Hardlink / Reflink / Trash Actions Handler
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
                Err(e) => ui.set_status_text(format!("Action failed: {}", e).into()),
            });
        });
    });

    // 7. Selection Rules Trigger Callback
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_trigger_selection_rule(move |rule| {
        let mut lock = state_clone.lock().unwrap();
        utils::ui::apply_selection_rule(&mut lock, rule.as_str());
        if let Some(ui) = app_weak.upgrade() {
            utils::ui::update_results_ui(&ui, &lock);
        }
    });

    // 8. Channel Toggled Callback
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

    // 9. OS System-level Drag & Drop handler (Native winit)
    let app_weak_dnd = app.as_weak();
    app.window().on_winit_window_event(move |_window, event| {
        if let slint::winit_030::winit::event::WindowEvent::DroppedFile(path_buf) = event {
            let path_str = path_buf.to_string_lossy().to_string();
            let app_copy = app_weak_dnd.clone();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_query_text(path_str.clone().into());
                ui.set_search_method(2); // Auto switch to AI Visual Search
                ui.set_status_text(format!("Reference image loaded: {}", path_str).into());
            });
        }
        slint::winit_030::EventResult::Propagate
    });

    // Launch main UI event loop
    app.run()
        .context("Slint event loop terminated with an error")?;

    Ok(())
}

// ---------------------------------------------------------
// PRIVATE HELPER FUNCTIONS
// ---------------------------------------------------------

/// Syncs the loaded `AppSettings` struct to the Slint UI Window properties
fn apply_settings_to_ui(app: &AppWindow, settings: &AppSettings) {
    app.set_dir_a(settings.dir_a.clone().into());
    app.set_dir_b(settings.dir_b.clone().into());
    app.set_query_text(settings.query_text.clone().into());
    app.set_similarity_threshold(settings.similarity_threshold);
    app.set_batch_size(settings.batch_size);
    app.set_search_method(settings.search_method);
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
}

/// Dispatches a background task to download required HuggingFace ONNX models
fn trigger_startup_model_download(app_weak: slint::Weak<AppWindow>) {
    tokio::spawn(async move {
        match crate::core::downloader::verify_and_download_models(app_weak.clone()).await {
            Ok(_) => {
                let _ = app_weak.upgrade_in_event_loop(|ui| {
                    ui.set_status_text("AI models verified. System ready.".into());
                    ui.set_progress(1.0);
                });
            }
            Err(e) => {
                let _ = app_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_status_text(format!("Model verification failed: {}", e).into());
                });
            }
        }
    });
}

/// Spawns a background task to load images, apply channel filters, and calculate diff maps
fn trigger_viewport_update(app_weak: slint::Weak<AppWindow>, orig_path: String, dup_path: String) {
    let ui = app_weak.unwrap();
    let channel = utils::ui::get_current_active_channel(&ui).to_string();
    let compare_mode = ui.get_compare_mode();
    let app_weak_clone = app_weak.clone();

    tokio::spawn(async move {
        // Load original
        if let Some(raw_orig) = utils::cache::get_channel_preview_image(&orig_path, &channel).await
        {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_original(utils::ui::convert_to_slint_image(&raw_orig));
            });
        }

        // Load duplicate
        if let Some(raw_dup) = utils::cache::get_channel_preview_image(&dup_path, &channel).await {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_duplicate(utils::ui::convert_to_slint_image(&raw_dup));
            });
        }

        // Calculate and load Heatmap Diff if needed
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
