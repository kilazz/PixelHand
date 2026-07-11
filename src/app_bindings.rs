// src/app_bindings.rs

use slint::ComponentHandle;
use slint::winit_030::WinitWindowAccessor;
use slint::{ModelRc, VecModel};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::app::{AppWindow, SelectedFile, Store};
use crate::scanners;
use crate::state::AppState;
use crate::utils;
use crate::utils::helpers::MutexExt;

/// Registers all interactive callbacks and event bindings for the Slint UI Window.
pub fn register_callbacks(
    app: &AppWindow,
    state: Arc<Mutex<AppState>>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    bind_directory_selection(app);
    bind_scan_execution(app, state.clone(), cancel_token);
    bind_file_actions(app, state.clone());
    bind_ui_state_and_settings(app, state);
    bind_drag_and_drop(app);
}

/// Binds UI handlers responsible for folder, reference image, and custom model selection.
fn bind_directory_selection(app: &AppWindow) {
    let app_weak_a = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_folder_a(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder A")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_a.upgrade() {
                let store = ui.global::<Store>();
                store.set_dir_a(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_b = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_folder_b(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder B")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_b.upgrade() {
                let store = ui.global::<Store>();
                store.set_dir_b(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_ref = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_reference_image(move || {
        if let Some(file) = rfd::FileDialog::new()
            .set_title("Select Reference Image")
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "tga", "dds", "exr", "hdr", "tif", "tiff", "webp", "gif",
                    "psd", "jxl", "heic", "heif", "avif",
                ],
            )
            .pick_file()
        {
            let path_str = file.to_string_lossy().to_string();
            if let Some(ui) = app_weak_ref.upgrade() {
                let store = ui.global::<Store>();
                store.set_query_text(path_str.into());
                store.set_search_method(2);
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_custom = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_custom_model(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Custom ONNX Model Directory")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_custom.upgrade() {
                let store = ui.global::<Store>();
                store.set_custom_model_path(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });
}

/// Binds scan execution pipeline, setup callbacks, and handles download step verification.
fn bind_scan_execution(
    app: &AppWindow,
    state: Arc<Mutex<AppState>>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    let app_weak_scan = app.as_weak();
    let state_clone = state.clone();
    let cancel_token_clone = cancel_token.clone();
    let store = app.global::<Store>();

    store.on_run_scan(move || {
        let app_copy = app_weak_scan.clone();
        let state_copy = state_clone.clone();

        let ui = match app_copy.upgrade() {
            Some(ui) => ui,
            None => return,
        };
        let store = ui.global::<Store>();

        utils::settings::save_settings(&store);
        cancel_token_clone.store(false, Ordering::Relaxed);

        let mut params = scanners::ScanParams::from_store(&store, cancel_token_clone.clone());

        let app_weak_progress = app_copy.clone();
        params.on_progress = Some(Arc::new(move |prog, current, total| {
            let _ = app_weak_progress.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                store.set_progress(prog);
                store.set_status_text(
                    format!("Processing assets ({} / {})...", current, total).into(),
                );
            });
        }));

        store.set_is_scanning(true);
        store.set_status_text("Scanning assets...".into());
        store.set_progress(0.0);
        tracing::info!("Starting scan on directory: {}", params.dir_a);

        let params_for_task = params.clone();
        let app_weak_download = app_copy.clone();

        tokio::spawn(async move {
            if params_for_task.search_method == 2
                && let Err(e) = crate::core::downloader::verify_and_download_models(
                    app_weak_download.clone(),
                    params_for_task.ai_model,
                )
                .await
            {
                let _ = app_weak_download.upgrade_in_event_loop(move |ui| {
                    let store = ui.global::<Store>();
                    store.set_is_scanning(false);
                    store.set_status_text(format!("AI Model download failed: {}", e).into());
                });
                return;
            }

            if let Some(ref_ui) = app_weak_download.upgrade() {
                let store = ref_ui.global::<Store>();
                crate::core::tonemapper::TONEMAP_ENABLED
                    .store(store.get_tonemap_enabled(), Ordering::Relaxed);
                crate::core::tonemapper::TONEMAP_OPERATOR
                    .store(store.get_tonemap_operator() as usize, Ordering::Relaxed);
            }

            let scan_result = scanners::execute_scan(params_for_task.clone()).await;

            if params_for_task.save_visuals
                && let Ok((ref groups, _)) = scan_result
            {
                let _ = app_copy.upgrade_in_event_loop(|ui| {
                    let store = ui.global::<Store>();
                    store.set_status_text("Generating contact sheet reports...".into());
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
                let store = ui.global::<Store>();
                store.set_is_scanning(false);
                store.set_progress(1.0);
                match scan_result {
                    Ok((groups, rows)) => {
                        let mut state_lock = state_copy.safe_lock();
                        state_lock.collapsed_groups.clear();
                        state_lock.groups = groups;
                        state_lock.results = rows;

                        utils::ui::update_results_ui(&store, &state_lock);

                        let msg = if params_for_task.save_visuals {
                            "Scan and visual reports finished successfully!"
                        } else {
                            "Scan finished successfully!"
                        };
                        store.set_status_text(msg.into());
                        tracing::info!("Scan completed.");
                    }
                    Err(e) => {
                        store.set_status_text(format!("Scan stopped: {}", e).into());
                        tracing::warn!("Scan interrupted: {}", e);
                    }
                }
            });
        });
    });

    let cancel_token_cancel = cancel_token.clone();
    let store = app.global::<Store>();
    store.on_cancel_scan(move || {
        cancel_token_cancel.store(true, Ordering::Relaxed);
        tracing::warn!("User requested scan cancellation. Waiting for threads to stop...");
    });
}

/// Binds UI handlers for files interactions, double clicks, checklist updates, and viewport previews.
fn bind_file_actions(app: &AppWindow, state: Arc<Mutex<AppState>>) {
    let app_weak_ctx = app.as_weak();
    let state_clone_ctx = state.clone();
    let store = app.global::<Store>();

    store.on_context_menu_action(move |action, path| {
        let path_str = path.to_string();
        if path_str.is_empty() {
            return;
        }

        match action.as_str() {
            "open" => {
                tracing::info!("Context Menu: Opening file {}", path_str);
                if let Err(e) = open::that(&path_str) {
                    tracing::error!("Failed to open file: {}", e);
                }
            }
            "explore" => {
                tracing::info!("Context Menu: Showing in explorer {}", path_str);
                if let Some(parent_dir) = std::path::Path::new(&path_str).parent()
                    && let Err(e) = open::that(parent_dir)
                {
                    tracing::error!("Failed to open directory: {}", e);
                }
            }
            "trash" => {
                let p = std::path::PathBuf::from(&path_str);
                if p.exists() {
                    match trash::delete(&p) {
                        Ok(_) => {
                            tracing::info!("Moved to trash via context menu: {}", path_str);
                            if let Some(ui) = app_weak_ctx.upgrade() {
                                let store = ui.global::<Store>();
                                store.set_status_text(
                                    format!(
                                        "Moved to trash: {}",
                                        p.file_name().unwrap_or_default().to_string_lossy()
                                    )
                                    .into(),
                                );
                                let mut lock = state_clone_ctx.safe_lock();
                                lock.results.retain(|r| r.path != path_str);
                                utils::ui::update_results_ui(&store, &lock);
                            }
                        }
                        Err(e) => tracing::error!("Failed to move to trash: {}", e),
                    }
                }
            }
            _ => {}
        }
    });

    let store = app.global::<Store>();
    store.on_open_file_in_viewer(move |path| {
        let path_str = path.to_string();
        if !path_str.is_empty() {
            tracing::info!("Opening file in default viewer: {}", path_str);
            if let Err(e) = open::that(&path_str) {
                tracing::error!("Failed to open file: {}", e);
            }
        }
    });

    let app_weak_checkbox = app.as_weak();
    let state_clone_cb = state.clone();
    let store = app.global::<Store>();
    store.on_row_checkbox_toggled(move |idx| {
        let mut lock = state_clone_cb.safe_lock();
        if let Some(abs_idx) = utils::ui::get_absolute_index(&lock, idx as usize)
            && let Some(row) = lock.results.get_mut(abs_idx)
        {
            row.is_checked = !row.is_checked;
        }
        if let Some(ui) = app_weak_checkbox.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_clicked = app.as_weak();
    let state_clone_click = state.clone();
    let store = app.global::<Store>();
    store.on_row_clicked(move |_idx, is_header, group_idx, path| {
        let app_copy = app_weak_clicked.clone();

        if is_header {
            let mut lock = state_clone_click.safe_lock();
            if lock.collapsed_groups.contains(&group_idx) {
                lock.collapsed_groups.remove(&group_idx);
            } else {
                lock.collapsed_groups.insert(group_idx);
            }
            if let Some(ui) = app_copy.upgrade() {
                let store = ui.global::<Store>();
                utils::ui::update_results_ui(&store, &lock);
            }
            return;
        }

        let path_str = path.to_string();
        let lock = state_clone_click.safe_lock();

        let group = lock.groups.get(group_idx as usize);

        let ui = match app_copy.upgrade() {
            Some(ui) => ui,
            None => return,
        };
        let store = ui.global::<Store>();

        let group = match group {
            None => {
                if let Ok(meta) =
                    crate::core::qc::extract_qc_metadata(std::path::Path::new(&path_str))
                {
                    let name = std::path::Path::new(&path_str)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();

                    let mipmaps_str = if meta.mipmap_count <= 1 {
                        "No".to_string()
                    } else {
                        meta.mipmap_count.to_string()
                    };

                    let file_meta = SelectedFile {
                        name: slint::SharedString::from(name.as_ref()),
                        size_str: slint::SharedString::from(format!(
                            "{} (VRAM: {})",
                            crate::utils::helpers::format_size(meta.file_size),
                            crate::utils::helpers::format_size(meta.estimated_vram)
                        )),
                        format: slint::SharedString::from(&meta.compression_format),
                        resolution: slint::SharedString::from(format!(
                            "{}x{}",
                            meta.width, meta.height
                        )),
                        bit_depth: slint::SharedString::from(format!("{}-bit", meta.bit_depth)),
                        color_space: slint::SharedString::from(&meta.color_space),
                        mipmaps: slint::SharedString::from(mipmaps_str),
                        alpha: slint::SharedString::from(if meta.has_alpha { "Yes" } else { "No" }),
                        similarity: slint::SharedString::from("-"),
                        path: slint::SharedString::from(&path_str),
                    };

                    store.set_original_meta(file_meta.clone());
                    store.set_duplicate_meta(file_meta);
                    store.set_max_available_mips(meta.mipmap_count as i32);
                    store.set_active_mip_level(0);
                }

                store.set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(
                    Vec::new(),
                ))));

                crate::app::trigger_viewport_update(
                    app_weak_clicked.clone(),
                    path_str.clone(),
                    path_str.clone(),
                );
                return;
            }
            Some(g) => g,
        };

        let original = match group.files.first() {
            Some(f) => f,
            None => return,
        };
        let duplicate = match group.files.iter().find(|f| f.path == path_str) {
            Some(f) => f,
            None => return,
        };

        store.set_original_meta(utils::ui::build_selected_file_meta(original, true));
        store.set_duplicate_meta(utils::ui::build_selected_file_meta(duplicate, false));
        store.set_max_available_mips(original.mipmap_count as i32);
        store.set_active_mip_level(0);

        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(utils::ui::convert_to_slint_row(row));
            }
        }
        store
            .set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));
        crate::app::trigger_viewport_update(
            app_weak_clicked.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    });

    let app_weak_action = app.as_weak();
    let state_copy_act = state.clone();
    let store = app.global::<Store>();
    store.on_trigger_action(move |action_type| {
        let app_copy = app_weak_action.clone();
        let state_copy_inner = state_copy_act.clone();
        let action = action_type.to_string();

        tokio::spawn(async move {
            let (checked_files, pairs) = {
                let lock = state_copy_inner.safe_lock();
                utils::fs::extract_selected_files(&lock)
            };

            if checked_files.is_empty() {
                return;
            }

            let _ = app_copy.upgrade_in_event_loop({
                let r#act = action.clone();
                move |ui| {
                    let store = ui.global::<Store>();
                    store.set_status_text(format!("Processing selection: {}...", r#act).into());
                }
            });

            let res = utils::fs::execute_file_action(&action, checked_files, pairs).await;

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                match res {
                    Ok(_) => {
                        store.set_status_text(
                            format!("Successfully completed {} operation.", action).into(),
                        );
                        store.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(
                            Vec::new(),
                        ))));
                        let mut lock = state_copy_inner.safe_lock();
                        lock.results.clear();
                        lock.groups.clear();
                        lock.collapsed_groups.clear();
                    }
                    Err(e) => {
                        store.set_status_text(format!("Action failed: {}", e).into());
                        tracing::error!("Action failed: {}", e);
                    }
                }
            });
        });
    });

    let app_weak_hover = app.as_weak();
    let state_hover = state;

    // Thread-safe async debouncing handle to capture rapid mouse hovering and eliminate event loop flooding
    let last_hover_task = Arc::new(Mutex::new(None::<tokio::task::JoinHandle<()>>));

    let store = app.global::<Store>();
    store.on_thumbnail_channel_hovered(move |path_str, channel| {
        let path_std = path_str.to_string();
        let channel_std = channel.to_string();
        let app_weak_clone = app_weak_hover.clone();
        let state_clone = state_hover.clone();

        let mut lock = last_hover_task.lock().unwrap();
        if let Some(handle) = lock.take() {
            handle.abort(); // Cancel any rapid intermediate hover tasks to protect the Slint event loop
        }

        // Spawn debounced hover processing task with an 80ms delay
        let handle = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;

            let normalized_path = scanners::normalize_path_key(&path_std);

            // Fetch the cached item directly from Moka sync cache (no explicit locking needed)
            let cached_img = {
                let cache = scanners::THUMBNAIL_MEMORY_CACHE.get();
                cache.and_then(|c| c.get(&normalized_path))
            };

            if let Some(cached_thumb) = cached_img {
                // Highly optimized: fetch pre-computed (or lazily initialized once) channel-isolated buffer
                let channel_img = cached_thumb.get_channel(&channel_std);

                // Sync the shared state representation under isolated scope
                {
                    let mut lock = state_clone.safe_lock();
                    for row in &mut lock.results {
                        if scanners::normalize_path_key(&row.path) == normalized_path {
                            row.thumbnail_data = Some(channel_img.clone());
                        }
                    }
                }

                // Update UI components dynamically inside the event loop
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    use slint::Model;
                    let store = ui.global::<Store>();
                    let slint_img = utils::ui::convert_to_slint_image(&channel_img);

                    // Update List Results
                    let results_model = store.get_results();
                    for i in 0..results_model.row_count() {
                        if let Some(mut r) = results_model.row_data(i)
                            && scanners::normalize_path_key(r.path.as_str()) == normalized_path
                        {
                            r.thumbnail = slint_img.clone();
                            results_model.set_row_data(i, r);
                        }
                    }

                    // Update Grid Results (mapping dynamically into virtualized GridRow columns)
                    let grid_model = store.get_grid_row_results();
                    for i in 0..grid_model.row_count() {
                        if let Some(mut r) = grid_model.row_data(i) {
                            if r.has_col1
                                && scanners::normalize_path_key(r.col1.path.as_str())
                                    == normalized_path
                            {
                                r.col1.thumbnail = slint_img.clone();
                                grid_model.set_row_data(i, r);
                            } else if r.has_col2
                                && scanners::normalize_path_key(r.col2.path.as_str())
                                    == normalized_path
                            {
                                r.col2.thumbnail = slint_img.clone();
                                grid_model.set_row_data(i, r);
                            } else if r.has_col3
                                && scanners::normalize_path_key(r.col3.path.as_str())
                                    == normalized_path
                            {
                                r.col3.thumbnail = slint_img.clone();
                                grid_model.set_row_data(i, r);
                            } else if r.has_col4
                                && scanners::normalize_path_key(r.col4.path.as_str())
                                    == normalized_path
                            {
                                r.col4.thumbnail = slint_img.clone();
                                grid_model.set_row_data(i, r);
                            }
                        }
                    }

                    // Update Selected Group Files (Compare Panel)
                    let group_model = store.get_selected_group_files();
                    for i in 0..group_model.row_count() {
                        if let Some(mut r) = group_model.row_data(i)
                            && scanners::normalize_path_key(r.path.as_str()) == normalized_path
                        {
                            r.thumbnail = slint_img.clone();
                            group_model.set_row_data(i, r);
                        }
                    }
                });
            }
        });
        *lock = Some(handle);
    });
}

/// Binds UI utility parameters, tonemapping options, sorting/filtering engines, and column resizers.
fn bind_ui_state_and_settings(app: &AppWindow, state: Arc<Mutex<AppState>>) {
    let app_weak_comp = app.as_weak();
    let store = app.global::<Store>();

    store.on_compare_mode_changed(move || {
        let app_copy = app_weak_comp.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_comp.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_tonemap = app.as_weak();
    let store = app.global::<Store>();
    store.on_tonemap_toggled(move || {
        let app_copy = app_weak_tonemap.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();

            utils::settings::save_settings(&store);

            crate::core::tonemapper::TONEMAP_ENABLED
                .store(store.get_tonemap_enabled(), Ordering::Relaxed);
            crate::core::tonemapper::TONEMAP_OPERATOR
                .store(store.get_tonemap_operator() as usize, Ordering::Relaxed);

            if let Some(cache) = utils::cache::DECODED_CACHE.get() {
                cache.invalidate_all();
            }

            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_tonemap.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_channel = app.as_weak();
    let store = app.global::<Store>();
    store.on_channel_toggled(move || {
        let app_copy = app_weak_channel.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_channel.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_mip = app.as_weak();
    let store = app.global::<Store>();
    store.on_mip_level_changed(move || {
        let app_copy = app_weak_mip.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_mip.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_rule = app.as_weak();
    let state_clone_rule = state.clone();
    let store = app.global::<Store>();
    store.on_trigger_selection_rule(move |rule| {
        let mut lock = state_clone_rule.safe_lock();
        utils::ui::apply_selection_rule(&mut lock, rule.as_str());
        if let Some(ui) = app_weak_rule.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_expand = app.as_weak();
    let state_clone_exp = state.clone();
    let store = app.global::<Store>();
    store.on_expand_all_groups(move || {
        let mut lock = state_clone_exp.safe_lock();
        lock.collapsed_groups.clear();
        if let Some(ui) = app_weak_expand.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_collapse = app.as_weak();
    let state_clone_col = state.clone();
    let store = app.global::<Store>();
    store.on_collapse_all_groups(move || {
        let mut lock = state_clone_col.safe_lock();
        lock.collapsed_groups.clear();

        let header_indices: Vec<i32> = lock
            .results
            .iter()
            .filter(|row| row.is_header)
            .map(|row| row.group_index)
            .collect();

        for group_index in header_indices {
            lock.collapsed_groups.insert(group_index);
        }

        if let Some(ui) = app_weak_collapse.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let store = app.global::<Store>();
    store.on_export_log(move |log_text| {
        utils::export::export_diagnostics_log(log_text.as_str());
    });

    let state_clone_csv = state.clone();
    let store = app.global::<Store>();
    store.on_export_csv(move || {
        utils::export::export_results_to_csv(state_clone_csv.clone());
    });

    let app_weak_col_sort = app.as_weak();
    let state_clone_col_sort = state.clone();
    let store = app.global::<Store>();
    store.on_sort_by_column(move |col| {
        let mut lock = state_clone_col_sort.safe_lock();
        let col_str = col.to_string();

        if lock.sort_column == col_str {
            lock.sort_ascending = !lock.sort_ascending;
        } else {
            lock.sort_column = col_str;
            lock.sort_ascending = true;
        }

        let asc = lock.sort_ascending;

        if !lock.groups.is_empty() {
            match lock.sort_column.as_str() {
                "name" => {
                    lock.groups.sort_by(|a, b| {
                        let n_a = a
                            .files
                            .first()
                            .map(|f| f.path.to_lowercase())
                            .unwrap_or_default();
                        let n_b = b
                            .files
                            .first()
                            .map(|f| f.path.to_lowercase())
                            .unwrap_or_default();
                        let res = n_a.cmp(&n_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                "size" => {
                    lock.groups.sort_by(|a, b| {
                        let s_a = a.files.first().map(|f| f.size).unwrap_or(0);
                        let s_b = b.files.first().map(|f| f.size).unwrap_or(0);
                        let res = s_a.cmp(&s_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                "score" => {
                    lock.groups.sort_by(|a, b| {
                        let sim_a = a.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                        let sim_b = b.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                        let res = sim_a
                            .partial_cmp(&sim_b)
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if asc { res } else { res.reverse() }
                    });
                }
                _ => {}
            }
            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
        } else if !lock.results.is_empty() {
            match lock.sort_column.as_str() {
                "name" => {
                    lock.results.sort_by(|a, b| {
                        let res = a.name.to_lowercase().cmp(&b.name.to_lowercase());
                        if asc { res } else { res.reverse() }
                    });
                }
                "format" => {
                    lock.results.sort_by(|a, b| {
                        let res = a
                            .format_str
                            .to_lowercase()
                            .cmp(&b.format_str.to_lowercase());
                        if asc { res } else { res.reverse() }
                    });
                }
                "dimensions" => {
                    lock.results.sort_by(|a, b| {
                        let res = a.pixels_count.cmp(&b.pixels_count);
                        if asc { res } else { res.reverse() }
                    });
                }
                "mipmaps" => {
                    lock.results.sort_by(|a, b| {
                        let m_a = a.mipmaps_str.parse::<u32>().unwrap_or(0);
                        let m_b = b.mipmaps_str.parse::<u32>().unwrap_or(0);
                        let res = m_a.cmp(&m_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                "cubemap" => {
                    lock.results.sort_by(|a, b| {
                        let res = a.cubemap_str.cmp(&b.cubemap_str);
                        if asc { res } else { res.reverse() }
                    });
                }
                "size" => {
                    lock.results.sort_by(|a, b| {
                        let res = a.size_bytes.cmp(&b.size_bytes);
                        if asc { res } else { res.reverse() }
                    });
                }
                "path" => {
                    lock.results.sort_by(|a, b| {
                        let res = a.path.to_lowercase().cmp(&b.path.to_lowercase());
                        if asc { res } else { res.reverse() }
                    });
                }
                _ => {}
            }
        }

        if let Some(ui) = app_weak_col_sort.upgrade() {
            let store = ui.global::<Store>();
            store.set_active_sort_column(lock.sort_column.clone().into());
            store.set_sort_ascending(lock.sort_ascending);
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let store = app.global::<Store>();
    store.on_clear_models(move || {
        if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
            let _ = std::fs::remove_dir_all(app_dir.join("models"));
            crate::app::append_to_console_log(
                "Downloaded AI model weights successfully cleared from disk.",
            );
        }
    });

    let store = app.global::<Store>();
    store.on_clear_cache(move || {
        // 1. Clear in-memory caches immediately using Moka API
        if let Some(cache) = crate::utils::cache::DECODED_CACHE.get() {
            cache.invalidate_all();
        }

        if let Some(cache) = crate::scanners::THUMBNAIL_MEMORY_CACHE.get() {
            cache.invalidate_all();
        }

        // 2. Safely delete database files in a background thread
        tokio::spawn(async move {
            if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
                let lancedb_dir = app_dir.join(".lancedb_cache");
                let cache_dir = app_dir.join(".cache");
                let mut success = true;

                if lancedb_dir.exists()
                    && let Err(e) = std::fs::remove_dir_all(&lancedb_dir)
                {
                    tracing::error!("Failed to clear LanceDB cache (possibly locked): {}", e);
                    success = false;
                }

                if cache_dir.exists()
                    && let Err(e) = std::fs::remove_dir_all(&cache_dir)
                {
                    tracing::error!("Failed to clear thumbnails cache: {}", e);
                    success = false;
                }

                if success {
                    crate::app::append_to_console_log(
                        "Scan database and thumbnail caches cleared successfully.",
                    );
                } else {
                    crate::app::append_to_console_log(
                        "Warning: Some cache files could not be cleared (files might be in use).",
                    );
                }
            }
        });
    });

    let app_weak_filter = app.as_weak();
    let state_clone_filt = state.clone();
    let store = app.global::<Store>();
    store.on_results_filter_changed(move || {
        if let Some(ui) = app_weak_filter.upgrade() {
            let lock = state_clone_filt.safe_lock();
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_sort = app.as_weak();
    let state_clone_sort = state.clone();
    let store = app.global::<Store>();
    store.on_results_sort_changed(move |sort_idx| {
        let mut lock = state_clone_sort.safe_lock();
        if !lock.groups.is_empty() {
            match sort_idx {
                0 => lock
                    .groups
                    .sort_by_key(|g| std::cmp::Reverse(g.files.len())),
                1 => lock.groups.sort_by_key(|g| {
                    std::cmp::Reverse(g.files.iter().map(|f| f.size).sum::<u64>())
                }),
                2 => lock.groups.sort_by(|a, b| {
                    let name_a = a.files.first().map(|f| f.path.as_str()).unwrap_or("");
                    let name_b = b.files.first().map(|f| f.path.as_str()).unwrap_or("");
                    name_a.cmp(name_b)
                }),
                _ => {}
            }
            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
        } else if !lock.results.is_empty() {
            match sort_idx {
                0 => lock.results.sort_by(|a, b| a.format_str.cmp(&b.format_str)),
                1 => lock
                    .results
                    .sort_by_key(|r| std::cmp::Reverse(r.size_bytes)),
                2 => lock.results.sort_by(|a, b| a.name.cmp(&b.name)),
                _ => {}
            }
        }
        if let Some(ui) = app_weak_sort.upgrade() {
            let lock = state_clone_sort.safe_lock();
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_auto = app.as_weak();
    let state_auto = state.clone();
    let store = app.global::<Store>();
    store.on_auto_size_columns(move || {
        if let Some(ui) = app_weak_auto.upgrade() {
            let lock = state_auto.safe_lock();
            if lock.results.is_empty() {
                return;
            }

            let mut max_file_len = 4;
            let mut max_score_len = 5;
            let mut max_path_len = 4;

            for row in &lock.results {
                if !row.is_header {
                    max_file_len = max_file_len.max(row.name.chars().count());
                    max_score_len = max_score_len.max(row.score_or_detail.chars().count());
                    max_path_len = max_path_len.max(row.path.chars().count());
                }
            }

            let file_w = (max_file_len as f32 * 7.2) + 68.0;
            let score_w = (max_score_len as f32 * 7.2) + 20.0;
            let path_w = (max_path_len as f32 * 6.5) + 20.0;

            let store = ui.global::<Store>();
            store.set_col_file_w(file_w.clamp(120.0, 600.0));
            store.set_col_score_w(score_w.clamp(60.0, 150.0));
            store.set_col_path_w(path_w.clamp(150.0, 800.0));

            tracing::info!(
                "Columns auto-resized. FILE: {:.0}px, SCORE: {:.0}px, PATH: {:.0}px",
                file_w,
                score_w,
                path_w
            );
        }
    });

    // Reset columns callback implementation
    let app_weak_reset = app.as_weak();
    let store = app.global::<Store>();
    store.on_reset_size_columns(move || {
        if let Some(ui) = app_weak_reset.upgrade() {
            let store = ui.global::<Store>();
            store.set_col_file_w(300.0);
            store.set_col_score_w(70.0);
            store.set_col_path_w(350.0);
            store.set_col_format_w(110.0);
            store.set_col_dimensions_w(110.0);
            store.set_col_mipmaps_w(75.0);
            store.set_col_cubemap_w(75.0);
            store.set_col_size_w(85.0);
            tracing::info!("Column sizes reset to defaults.");
        }
    });
}

/// Binds standard window OS events, such as native drag and drop files/directories drop targets.
fn bind_drag_and_drop(app: &AppWindow) {
    let app_weak_dnd = app.as_weak();
    app.window().on_winit_window_event(move |_window, event| {
        if let slint::winit_030::winit::event::WindowEvent::DroppedFile(path_buf) = event {
            let path_str = path_buf.to_string_lossy().to_string();
            let is_dir = path_buf.is_dir();
            let is_file = path_buf.is_file();
            let app_copy = app_weak_dnd.clone();

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                if is_dir {
                    store.set_dir_a(path_str.clone().into());
                    store.set_status_text(format!("Scan folder updated: {}", path_str).into());
                } else if is_file {
                    store.set_query_text(path_str.clone().into());
                    store.set_search_method(2);
                    store.set_status_text(format!("Reference image loaded: {}", path_str).into());
                }
                utils::settings::save_settings(&store);
            });
        }
        slint::winit_030::EventResult::Propagate
    });
}
