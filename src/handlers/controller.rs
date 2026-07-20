// src/handlers/controller.rs

use slint::ComponentHandle;
use slint::{Model, ModelRc, VecModel};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::app::{AppWindow, Diagnostics, ScanConfig, SelectedFile, ViewportState};
use crate::scanners::ScanParams;
use crate::state::AppState;
use crate::state::models::{SearchMethod, SortColumn};
use crate::utils;

pub struct AppController {
    ui_weak: slint::Weak<AppWindow>,
    state: Arc<parking_lot::Mutex<AppState>>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
}

impl AppController {
    pub fn new(
        ui: &AppWindow,
        state: Arc<parking_lot::Mutex<AppState>>,
        cancel_token: Arc<std::sync::atomic::AtomicBool>,
    ) -> Arc<Self> {
        Arc::new(Self {
            ui_weak: ui.as_weak(),
            state,
            cancel_token,
        })
    }

    /// Central callback registration dispatcher.
    pub fn register_callbacks(self: &Arc<Self>) {
        let ui = self
            .ui_weak
            .upgrade()
            .expect("Failed to bind callbacks: AppWindow is dead");

        self.register_scan_config_callbacks(&ui);
        self.register_viewport_state_callbacks(&ui);
        self.register_diagnostics_callbacks(&ui);
    }

    // ---------------------------------------------------------
    // --- DECOMPOSED REGISTRATION HELPERS ---------------------
    // ---------------------------------------------------------

    fn register_scan_config_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let scan_config = ui.global::<ScanConfig>();

        let self_clone = self.clone();
        scan_config.on_select_folder_a(move || {
            self_clone.select_folder_a();
        });

        let self_clone = self.clone();
        scan_config.on_select_folder_b(move || {
            self_clone.select_folder_b();
        });

        let self_clone = self.clone();
        scan_config.on_select_reference_image(move || {
            self_clone.select_reference_image();
        });

        let self_clone = self.clone();
        scan_config.on_select_custom_model(move || {
            self_clone.select_custom_model();
        });

        let self_clone = self.clone();
        scan_config.on_trigger_action(move |action| {
            self_clone.trigger_action(action.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_row_clicked(move |_, is_header, group_idx, path| {
            self_clone.handle_row_click(is_header, group_idx, path.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_row_checkbox_toggled(move |idx| {
            self_clone.handle_row_checkbox_toggled(idx as usize);
        });

        let self_clone = self.clone();
        scan_config.on_trigger_selection_rule(move |rule| {
            self_clone.trigger_selection_rule(rule.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_expand_all_groups(move || {
            self_clone.expand_all_groups();
        });

        let self_clone = self.clone();
        scan_config.on_collapse_all_groups(move || {
            self_clone.collapse_all_groups();
        });

        let self_clone = self.clone();
        scan_config.on_sort_by_column(move |col| {
            let col_str = col.to_string();
            let self_inner = self_clone.clone();
            let _ = slint::invoke_from_event_loop(move || {
                self_inner.sort_by_column(&col_str);
            });
        });

        let self_clone = self.clone();
        scan_config.on_results_filter_changed(move || {
            let self_inner = self_clone.clone();
            let _ = slint::invoke_from_event_loop(move || {
                self_inner.results_filter_changed();
            });
        });

        let self_clone = self.clone();
        scan_config.on_results_sort_changed(move |idx| {
            let self_inner = self_clone.clone();
            let _ = slint::invoke_from_event_loop(move || {
                self_inner.results_sort_changed(idx);
            });
        });

        let self_clone = self.clone();
        scan_config.on_clear_cache(move || {
            self_clone.clear_cache();
        });

        let self_clone = self.clone();
        scan_config.on_clear_models(move || {
            self_clone.clear_models();
        });

        let self_clone = self.clone();
        scan_config.on_auto_size_columns(move || {
            self_clone.auto_size_columns();
        });

        let self_clone = self.clone();
        scan_config.on_reset_size_columns(move || {
            self_clone.reset_size_columns();
        });

        let self_clone = self.clone();
        scan_config.on_context_menu_action(move |action, path| {
            self_clone.handle_context_menu_action(action.as_str(), path.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_thumbnail_channel_hovered(move |path, channel| {
            self_clone.thumbnail_channel_hovered(path.as_str(), channel.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_grid_columns_changed(move || {
            let self_inner = self_clone.clone();
            let _ = slint::invoke_from_event_loop(move || {
                self_inner.grid_columns_changed();
            });
        });

        let self_clone = self.clone();
        scan_config.on_save_settings(move || {
            self_clone.save_settings();
        });
    }

    fn register_viewport_state_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let viewport_state = ui.global::<ViewportState>();

        let self_clone = self.clone();
        viewport_state.on_open_file_in_viewer(move |path| {
            self_clone.open_file_in_viewer(path.as_str());
        });

        let self_clone = self.clone();
        viewport_state.on_channel_toggled(move || {
            self_clone.viewport_settings_changed();
        });

        let self_clone = self.clone();
        viewport_state.on_mip_level_changed(move || {
            self_clone.viewport_settings_changed();
        });

        let self_clone = self.clone();
        viewport_state.on_compare_mode_changed(move || {
            self_clone.viewport_settings_changed();
        });

        let self_clone = self.clone();
        viewport_state.on_active_frame_changed(move || {
            self_clone.viewport_settings_changed();
        });

        let self_clone = self.clone();
        viewport_state.on_tonemap_toggled(move || {
            self_clone.tonemap_toggled();
        });
    }

    fn register_diagnostics_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let diagnostics = ui.global::<Diagnostics>();

        let self_clone = self.clone();
        diagnostics.on_run_scan(move || {
            self_clone.run_scan();
        });

        let self_clone = self.clone();
        diagnostics.on_cancel_scan(move || {
            self_clone.cancel_scan();
        });

        let self_clone = self.clone();
        diagnostics.on_export_log(move |log| {
            self_clone.export_log(log.as_str());
        });

        let self_clone = self.clone();
        diagnostics.on_export_csv(move || {
            self_clone.export_csv();
        });
    }

    // ---------------------------------------------------------
    // --- PRIVATE SORTING HELPERS -----------------------------
    // ---------------------------------------------------------

    fn compare_duplicate_files(
        &self,
        col: SortColumn,
        a: &crate::state::DuplicateFileSummary,
        b: &crate::state::DuplicateFileSummary,
        asc: bool,
    ) -> std::cmp::Ordering {
        let order = match col {
            SortColumn::Name | SortColumn::Path => {
                a.path.to_lowercase().cmp(&b.path.to_lowercase())
            }
            SortColumn::Size => a.size.cmp(&b.size),
            SortColumn::Score => a
                .similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        };
        if asc { order } else { order.reverse() }
    }

    fn compare_duplicate_groups(
        &self,
        col: SortColumn,
        a: &crate::state::DuplicateGroupSummary,
        b: &crate::state::DuplicateGroupSummary,
        asc: bool,
    ) -> std::cmp::Ordering {
        let order = match col {
            SortColumn::Name | SortColumn::Path => {
                let p_a = a
                    .files
                    .first()
                    .map(|f| f.path.to_lowercase())
                    .unwrap_or_default();
                let p_b = b
                    .files
                    .first()
                    .map(|f| f.path.to_lowercase())
                    .unwrap_or_default();
                p_a.cmp(&p_b)
            }
            SortColumn::Size => {
                let s_a = a.files.first().map(|f| f.size).unwrap_or(0);
                let s_b = b.files.first().map(|f| f.size).unwrap_or(0);
                s_a.cmp(&s_b)
            }
            SortColumn::Score => {
                let sim_a = a.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                let sim_b = b.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            _ => std::cmp::Ordering::Equal,
        };
        if asc { order } else { order.reverse() }
    }

    fn compare_results_rows(
        &self,
        col: SortColumn,
        a: &crate::state::ResultsRowData,
        b: &crate::state::ResultsRowData,
        asc: bool,
    ) -> std::cmp::Ordering {
        let order = match col {
            SortColumn::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            SortColumn::Format => a
                .format_str
                .to_lowercase()
                .cmp(&b.format_str.to_lowercase()),
            SortColumn::Dimensions => a.pixels_count.cmp(&b.pixels_count),
            SortColumn::Mipmaps => {
                let m_a = a.mipmaps_str.parse::<u32>().unwrap_or(0);
                let m_b = b.mipmaps_str.parse::<u32>().unwrap_or(0);
                m_a.cmp(&m_b)
            }
            SortColumn::Cubemap => a.cubemap_str.cmp(&b.cubemap_str),
            SortColumn::Size => a.size_bytes.cmp(&b.size_bytes),
            SortColumn::Path => a.path.to_lowercase().cmp(&b.path.to_lowercase()),
            SortColumn::Score => a
                .similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        };
        if asc { order } else { order.reverse() }
    }

    // ---------------------------------------------------------
    // --- PRIVATE IMPLEMENTATION METHODS ----------------------
    // ---------------------------------------------------------

    fn select_folder_a(&self) {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder A")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let mut paths = scan_config.get_paths();
                paths.dir_a = path_str.into();
                scan_config.set_paths(paths);
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            }
        }
    }

    fn select_folder_b(&self) {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder B")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let mut paths = scan_config.get_paths();
                paths.dir_b = path_str.into();
                scan_config.set_paths(paths);
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            }
        }
    }

    fn select_reference_image(&self) {
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
            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let mut paths = scan_config.get_paths();
                paths.query_text = path_str.into();
                scan_config.set_paths(paths);
                scan_config.set_search_method(SearchMethod::Ai); // Configured directly with typed Slint Enum
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            }
        }
    }

    fn select_custom_model(&self) {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Custom ONNX Model Directory")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let mut ai = scan_config.get_ai();
                ai.custom_model_path = path_str.into();
                scan_config.set_ai(ai);
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            }
        }
    }

    fn cancel_scan(&self) {
        self.cancel_token.store(true, Ordering::Relaxed);
        tracing::warn!("User requested scan cancellation. Waiting for threads to stop...");
    }

    fn run_scan(&self) {
        let ui = match self.ui_weak.upgrade() {
            Some(ui) => ui,
            None => return,
        };

        let scan_config = ui.global::<ScanConfig>();
        let diagnostics = ui.global::<Diagnostics>();
        let viewport_state = ui.global::<ViewportState>();

        utils::settings::save_settings(&scan_config, &viewport_state);
        self.cancel_token.store(false, Ordering::Relaxed);

        // Fetch scan parameters cleanly from ScanConfig
        let mut params = ScanParams::from_store(&scan_config, self.cancel_token.clone());

        let ui_weak_progress = self.ui_weak.clone();
        params.on_progress = Some(Arc::new(move |prog, current, total| {
            let _ = ui_weak_progress.upgrade_in_event_loop(move |ui| {
                let diag = ui.global::<Diagnostics>();
                diag.set_progress(prog);
                diag.set_status_text(
                    format!("Processing assets ({} / {})...", current, total).into(),
                );
            });
        }));

        diagnostics.set_is_scanning(true);
        diagnostics.set_status_text("Scanning assets...".into());
        diagnostics.set_progress(0.0);
        tracing::info!("Starting scan on directory: {}", params.paths.dir_a);

        let params_for_task = params.clone();
        let ui_weak_download = self.ui_weak.clone();
        let state_clone = self.state.clone();
        let ui_weak_finish = self.ui_weak.clone();

        tokio::spawn(async move {
            // Verify models on Hugging Face if AI search method is selected
            if params_for_task.search_method == SearchMethod::Ai
                && let Err(e) = crate::core::downloader::verify_and_download_models(
                    ui_weak_download.clone(),
                    params_for_task.ai.ai_model,
                    params_for_task.cancel_token.clone(),
                )
                .await
            {
                let _ = ui_weak_download.upgrade_in_event_loop(move |ui| {
                    let diag = ui.global::<Diagnostics>();
                    diag.set_is_scanning(false);
                    diag.set_status_text(format!("AI Model download failed: {}", e).into());
                });
                return;
            }

            // Sync GUI tonemapping state to thread-safe app settings context prior to execution
            if let Some(ref_ui) = ui_weak_download.upgrade() {
                let vp = ref_ui.global::<ViewportState>();
                crate::app::update_viewer_settings(|s| {
                    let tonemap = vp.get_tonemap();
                    s.tonemap_enabled = tonemap.tonemap_enabled;
                    s.auto_exposure_enabled = tonemap.tonemap_auto_exposure;
                    s.tonemap_operator = tonemap.tonemap_operator as usize;
                });
            }

            let params_for_task_clone = params_for_task.clone();

            // Run scanner in tokio blocking pool
            let scan_result = crate::scanners::execute_scan(params_for_task_clone)
                .await
                .map_err(|e| anyhow::anyhow!("Background scanning failed: {}", e));

            // Generate optional contact sheets on duplicate clusters
            if params_for_task.visuals.save_visuals
                && let Ok((ref groups, _)) = scan_result
            {
                let _ = ui_weak_finish.upgrade_in_event_loop(|ui| {
                    let diag = ui.global::<Diagnostics>();
                    diag.set_status_text("Generating contact sheet reports...".into());
                });

                if let Ok(app_dir) = crate::utils::settings::get_portable_app_data_dir() {
                    let out_dir = app_dir.join("duplicate_visuals");
                    if let Err(e) = crate::core::visuals::generate_visual_reports(
                        groups.clone(),
                        params_for_task.visuals.visuals_columns,
                        params_for_task.visuals.visuals_max_count,
                        params_for_task.visuals.visuals_font_size,
                        params_for_task.visuals.visuals_scale,
                        out_dir,
                    )
                    .await
                    {
                        tracing::error!("Failed to generate visual reports: {}", e);
                    }
                }
            }

            let _ = ui_weak_finish.upgrade_in_event_loop(move |ui| {
                let diag = ui.global::<Diagnostics>();
                let scan_cfg = ui.global::<ScanConfig>();
                diag.set_is_scanning(false);
                diag.set_progress(1.0);
                match scan_result {
                    Ok((groups, rows)) => {
                        let mut state_lock = state_clone.lock();
                        state_lock.collapsed_groups.clear();
                        state_lock.groups = groups;
                        state_lock.results = rows;

                        // Unified UI layout update transaction
                        utils::ui::update_results_ui(&scan_cfg, &mut state_lock);

                        let msg = if params_for_task.visuals.save_visuals {
                            "Scan and visual reports finished successfully!"
                        } else {
                            "Scan finished successfully!"
                        };
                        diag.set_status_text(msg.into());
                        tracing::info!("Scan completed.");
                    }
                    Err(e) => {
                        diag.set_status_text(format!("Scan stopped: {}", e).into());
                        tracing::warn!("Scan interrupted: {}", e);
                    }
                }
            });
        });
    }

    fn trigger_action(&self, action: &str) {
        if self.ui_weak.upgrade().is_none() {
            return;
        }

        let self_weak = self.ui_weak.clone();
        let state_clone = self.state.clone();
        let action_owned = action.to_string();

        tokio::spawn(async move {
            let (checked_files, pairs) = {
                // Keep the Mutex lock brief to avoid starvation or thread contention during I/O
                let lock = state_clone.lock();
                utils::fs::extract_selected_files(&lock)
            };

            if checked_files.is_empty() {
                return;
            }

            let _ = self_weak.upgrade_in_event_loop({
                let act = action_owned.clone();
                move |ui| {
                    let diag = ui.global::<Diagnostics>();
                    diag.set_status_text(format!("Processing selection: {}...", act).into());
                }
            });

            let res = utils::fs::execute_file_action(&action_owned, checked_files, pairs).await;

            let _ = self_weak.upgrade_in_event_loop(move |ui| {
                let diag = ui.global::<Diagnostics>();
                let scan_cfg = ui.global::<ScanConfig>();
                match res {
                    Ok(_) => {
                        diag.set_status_text(
                            format!("Successfully completed {} operation.", action_owned).into(),
                        );
                        scan_cfg.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(
                            Vec::new(),
                        ))));
                        scan_cfg.set_has_results(false);

                        let mut lock = state_clone.lock();
                        lock.results.clear();
                        lock.groups.clear();
                        lock.collapsed_groups.clear();
                        lock.path_to_idx.clear();
                        lock.visible_to_abs.clear();
                    }
                    Err(e) => {
                        diag.set_status_text(format!("Action failed: {}", e).into());
                        tracing::error!("Action failed: {}", e);
                    }
                }
            });
        });
    }

    fn handle_row_click(&self, is_header: bool, group_idx: i32, path: &str) {
        let ui = match self.ui_weak.upgrade() {
            Some(ui) => ui,
            None => return,
        };

        let scan_config = ui.global::<ScanConfig>();
        let viewport_state = ui.global::<ViewportState>();

        if is_header {
            let mut lock = self.state.lock();
            if lock.collapsed_groups.contains(&group_idx) {
                lock.collapsed_groups.remove(&group_idx);
            } else {
                lock.collapsed_groups.insert(group_idx);
            }
            utils::ui::update_results_ui(&scan_config, &mut lock);
            return;
        }

        let path_str = path.to_string();
        let lock = self.state.lock();
        let group = lock.groups.get(group_idx as usize);

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

                    viewport_state.set_original_meta(file_meta.clone());
                    viewport_state.set_duplicate_meta(file_meta);
                    viewport_state.set_max_available_mips(meta.mipmap_count as i32);
                    viewport_state.set_active_mip_level(0);
                }

                scan_config.set_selected_group_files(ModelRc::from(std::rc::Rc::new(
                    VecModel::from(Vec::new()),
                )));
                crate::app::trigger_viewport_update(
                    self.ui_weak.clone(),
                    path_str.clone(),
                    path_str,
                );
                return;
            }
            Some(g) => g,
        };

        let original = match group.files.first() {
            Some(f) => f,
            None => return,
        };
        let normalized_target = utils::fs::normalize_path(&path_str);
        let duplicate = match group
            .files
            .iter()
            .find(|f| utils::fs::normalize_path(&f.path) == normalized_target)
        {
            Some(f) => f,
            None => return,
        };

        viewport_state.set_original_meta(utils::ui::build_selected_file_meta(original, true));
        viewport_state.set_duplicate_meta(utils::ui::build_selected_file_meta(duplicate, false));
        viewport_state.set_max_available_mips(original.mipmap_count as i32);
        viewport_state.set_active_mip_level(0);

        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(utils::ui::convert_to_slint_row(row));
            }
        }
        scan_config
            .set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));
        crate::app::trigger_viewport_update(
            self.ui_weak.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    }

    fn handle_row_checkbox_toggled(&self, idx: usize) {
        let mut lock = self.state.lock();
        if let Some(abs_idx) = utils::ui::get_absolute_index(&lock, idx)
            && let Some(row) = lock.results.get_mut(abs_idx)
        {
            row.is_checked = !row.is_checked;

            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let results_model = scan_config.get_results();

                if let Some(vec_model) = results_model
                    .as_any()
                    .downcast_ref::<VecModel<crate::app::ResultsRow>>()
                    && let Some(mut slint_row) = vec_model.row_data(idx)
                {
                    slint_row.is_checked = row.is_checked;
                    vec_model.set_row_data(idx, slint_row);
                }
            }
        }
    }

    fn trigger_selection_rule(&self, rule: &str) {
        let mut lock = self.state.lock();
        utils::ui::apply_selection_rule(&mut lock, rule);
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn expand_all_groups(&self) {
        let mut lock = self.state.lock();
        lock.collapsed_groups.clear();
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn collapse_all_groups(&self) {
        let mut lock = self.state.lock();
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

        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn sort_by_column(&self, col: &str) {
        let mut lock = self.state.lock();
        let col_type = SortColumn::from(col);

        if lock.sort_column == col {
            lock.sort_ascending = !lock.sort_ascending;
        } else {
            lock.sort_column = col.to_string();
            lock.sort_ascending = true;
        }

        let asc = lock.sort_ascending;

        if !lock.groups.is_empty() {
            for group in &mut lock.groups {
                if group.files.len() > 1 {
                    let (_, duplicates) = group.files.split_at_mut(1);
                    // Deduplicated: sorting of duplicate items inside groups
                    duplicates.sort_by(|a, b| self.compare_duplicate_files(col_type, a, b, asc));
                }
            }

            // Deduplicated: sorting of duplicate groups
            lock.groups
                .sort_by(|a, b| self.compare_duplicate_groups(col_type, a, b, asc));

            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
        } else if !lock.results.is_empty() {
            // Deduplicated: sorting of results rows in flat list mode
            lock.results
                .sort_by(|a, b| self.compare_results_rows(col_type, a, b, asc));
        }

        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            scan_config.set_active_sort_column(lock.sort_column.clone().into());
            scan_config.set_sort_ascending(lock.sort_ascending);
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn results_filter_changed(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let mut lock = self.state.lock();
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn results_sort_changed(&self, sort_idx: i32) {
        let mut lock = self.state.lock();
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
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn clear_cache(&self) {
        let manager = crate::utils::cache::get_cache_manager();
        manager.decoded_images.invalidate_all();
        manager.thumbnails.invalidate_all();

        let self_weak = self.ui_weak.clone();
        tokio::spawn(async move {
            if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
                let lancedb_dir = app_dir.join(".lancedb_cache");
                let cache_dir = app_dir.join(".cache");
                let mut success = true;

                if lancedb_dir.exists()
                    && let Err(e) = std::fs::remove_dir_all(&lancedb_dir)
                {
                    tracing::error!("Failed to clear LanceDB cache: {}", e);
                    success = false;
                }

                if cache_dir.exists()
                    && let Err(e) = std::fs::remove_dir_all(&cache_dir)
                {
                    tracing::error!("Failed to clear thumbnails cache: {}", e);
                    success = false;
                }

                let _ = self_weak.upgrade_in_event_loop(move |_ui| {
                    let log_str = if success {
                        "Scan database and thumbnail caches cleared successfully."
                    } else {
                        "Warning: Some cache files could not be cleared (files might be in use)."
                    };
                    crate::app::append_to_console_log(log_str);
                });
            }
        });
    }

    fn clear_models(&self) {
        if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
            let _ = std::fs::remove_dir_all(app_dir.join("models"));
            crate::app::append_to_console_log(
                "Downloaded AI model weights successfully cleared from disk.",
            );
        }
    }

    fn auto_size_columns(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let lock = self.state.lock();
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

            let scan_config = ui.global::<ScanConfig>();
            scan_config.set_col_file_w(file_w.clamp(120.0, 600.0));
            scan_config.set_col_score_w(score_w.clamp(60.0, 150.0));
            scan_config.set_col_path_w(path_w.clamp(150.0, 800.0));

            tracing::info!(
                "Columns auto-resized. FILE: {:.0}px, SCORE: {:.0}px, PATH: {:.0}px",
                file_w,
                score_w,
                path_w
            );
        }
    }

    fn reset_size_columns(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            scan_config.set_col_file_w(320.0);
            scan_config.set_col_score_w(80.0);
            scan_config.set_col_path_w(380.0);
            scan_config.set_col_format_w(110.0);
            scan_config.set_col_dimensions_w(110.0);
            scan_config.set_col_mipmaps_w(75.0);
            scan_config.set_col_cubemap_w(75.0);
            scan_config.set_col_size_w(85.0);
            tracing::info!("Column sizes reset to defaults.");
        }
    }

    fn handle_context_menu_action(&self, action: &str, path: &str) {
        let path_str = path.to_string();
        match action {
            "open" => {
                let _ = open::that(&path_str);
            }
            "explore" => {
                if let Some(parent) = std::path::Path::new(&path_str).parent() {
                    let _ = open::that(parent);
                }
            }
            "trash" => {
                let p = std::path::PathBuf::from(&path_str);
                if p.exists()
                    && trash::delete(&p).is_ok()
                    && let Some(ui) = self.ui_weak.upgrade()
                {
                    let diag = ui.global::<Diagnostics>();
                    let scan_cfg = ui.global::<ScanConfig>();
                    diag.set_status_text(
                        format!(
                            "Moved to trash: {}",
                            p.file_name().unwrap_or_default().to_string_lossy()
                        )
                        .into(),
                    );

                    let mut lock = self.state.lock();
                    let normalized = utils::fs::normalize_path(&path_str);
                    lock.results
                        .retain(|r| utils::fs::normalize_path(&r.path) != normalized);
                    utils::ui::update_results_ui(&scan_cfg, &mut lock);
                }
            }
            "trash_group" => {
                if let Ok(group_idx) = path_str.parse::<i32>() {
                    let mut lock = self.state.lock();
                    let mut paths_to_delete = Vec::new();

                    // If groups list is empty, we are likely in a flat/QC results list
                    if lock.groups.is_empty() {
                        for row in &lock.results {
                            if row.group_index == group_idx && !row.is_header {
                                paths_to_delete.push(std::path::PathBuf::from(&row.path));
                            }
                        }
                        paths_to_delete.retain(|p| p.exists());
                        if !paths_to_delete.is_empty()
                            && let Err(e) = trash::delete_all(&paths_to_delete)
                        {
                            tracing::error!("Failed to trash QC group items: {}", e);
                        }
                        lock.results.retain(|r| r.group_index != group_idx);
                    } else {
                        let group_idx_us = group_idx as usize;
                        if let Some(g) = lock.groups.get(group_idx_us) {
                            paths_to_delete = g
                                .files
                                .iter()
                                .map(|f| std::path::PathBuf::from(&f.path))
                                .filter(|p| p.exists())
                                .collect();
                            if !paths_to_delete.is_empty()
                                && let Err(e) = trash::delete_all(&paths_to_delete)
                            {
                                tracing::error!("Failed to trash duplicate cluster files: {}", e);
                            }
                        }
                        if group_idx_us < lock.groups.len() {
                            lock.groups.remove(group_idx_us);
                            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
                        }
                    }

                    if let Some(ui) = self.ui_weak.upgrade() {
                        let diag = ui.global::<Diagnostics>();
                        let scan_cfg = ui.global::<ScanConfig>();
                        diag.set_status_text(
                            format!(
                                "Moved {} files in the selected group to trash.",
                                paths_to_delete.len()
                            )
                            .into(),
                        );
                        utils::ui::update_results_ui(&scan_cfg, &mut lock);
                    }
                }
            }
            _ => {}
        }
    }

    fn thumbnail_channel_hovered(&self, path: &str, channel: &str) {
        let path_std = path.to_string();
        let channel_std = channel.to_string();
        let ui_weak = self.ui_weak.clone();
        let state_clone = self.state.clone();

        tokio::spawn(async move {
            let normalized_path_cache = utils::cache::normalize_path_key(&path_std);
            let normalized_path_fs = utils::fs::normalize_path_key(&path_std);

            let cached_img = {
                let manager = utils::cache::get_cache_manager();
                manager.thumbnails.get(&normalized_path_cache)
            };

            if let Some(cached_thumb) = cached_img {
                let channel_img = cached_thumb.get_channel(&channel_std);

                {
                    // Confining the Mutex lock strictly to the index-insertion logic to avoid long lockups
                    let mut lock = state_clone.lock();
                    if let Some(&idx) = lock.path_to_idx.get(&normalized_path_fs) {
                        lock.results[idx].thumbnail_data = Some(channel_img.clone());
                    }
                }

                let _ = ui_weak.upgrade_in_event_loop(move |ui| {
                    let slint_img = utils::ui::convert_to_slint_image(&channel_img);
                    let scan_config = ui.global::<ScanConfig>();

                    let results_model = scan_config.get_results();
                    for i in 0..results_model.row_count() {
                        if let Some(mut r) = results_model.row_data(i)
                            && utils::fs::normalize_path_key(r.path.as_str()) == normalized_path_fs
                        {
                            r.thumbnail = slint_img.clone();
                            results_model.set_row_data(i, r);
                        }
                    }
                });
            }
        });
    }

    fn grid_columns_changed(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let mut lock = self.state.lock();
            let scan_config = ui.global::<ScanConfig>();
            utils::ui::update_results_ui(&scan_config, &mut lock);
        }
    }

    fn save_settings(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            let viewport_state = ui.global::<ViewportState>();
            utils::settings::save_settings(&scan_config, &viewport_state);
        }
    }

    fn open_file_in_viewer(&self, path: &str) {
        let _ = open::that(path);
    }

    fn viewport_settings_changed(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let orig_path = viewport_state_changed_orig_path(&ui);
            let dup_path = viewport_state_changed_dup_path(&ui);

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(self.ui_weak.clone(), orig_path, dup_path);
            }
        }
    }

    fn tonemap_toggled(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            let viewport_state = ui.global::<ViewportState>();

            utils::settings::save_settings(&scan_config, &viewport_state);

            let manager = crate::utils::cache::get_cache_manager();
            manager.decoded_images.invalidate_all();

            self.viewport_settings_changed();
        }
    }

    fn export_log(&self, log_text: &str) {
        utils::export::export_diagnostics_log(log_text);
    }

    fn export_csv(&self) {
        utils::export::export_results_to_csv(self.state.clone());
    }
}

// Helper functions to fetch original/duplicate paths from ViewportState safely without thread leaks
fn viewport_state_changed_orig_path(ui: &AppWindow) -> String {
    let vp = ui.global::<ViewportState>();
    vp.get_original_meta().path.to_string()
}

fn viewport_state_changed_dup_path(ui: &AppWindow) -> String {
    let vp = ui.global::<ViewportState>();
    vp.get_duplicate_meta().path.to_string()
}
