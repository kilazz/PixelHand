// src/handlers/scan.rs

use slint::ComponentHandle;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::app::{AppWindow, Diagnostics, ScanConfig, ViewportState};
use crate::reporting::contact_sheets::generate_visual_reports;
use crate::state::models::{ScanParams, SearchMethod, SortColumn};
use crate::state::store::AppStateStore;
use crate::utils;
use crate::utils::notification::NotificationService;

#[inline]
fn compare_ord<T: Ord>(a: T, b: T, asc: bool) -> std::cmp::Ordering {
    if asc { a.cmp(&b) } else { b.cmp(&a) }
}

#[inline]
fn compare_partial<T: PartialOrd>(a: T, b: T, asc: bool) -> std::cmp::Ordering {
    let order = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
    if asc { order } else { order.reverse() }
}

pub struct ScanController {
    pub ui_weak: slint::Weak<AppWindow>,
    store: AppStateStore,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
    notifier: Arc<NotificationService>,
}

impl ScanController {
    pub fn new(
        ui_weak: slint::Weak<AppWindow>,
        store: AppStateStore,
        cancel_token: Arc<std::sync::atomic::AtomicBool>,
        notifier: Arc<NotificationService>,
    ) -> Arc<Self> {
        Arc::new(Self {
            ui_weak,
            store,
            cancel_token,
            notifier,
        })
    }

    pub fn register(self: &Arc<Self>, ui: &AppWindow) {
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
        scan_config.on_expand_all_groups(move || {
            self_clone.expand_all_groups();
        });

        let self_clone = self.clone();
        scan_config.on_collapse_all_groups(move || {
            self_clone.collapse_all_groups();
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
        scan_config.on_path_dropped(move |path_str| {
            self_clone.handle_path_dropped(path_str.as_str());
        });

        // Register main scan triggers in the Diagnostics interface
        let diagnostics = ui.global::<Diagnostics>();

        let self_clone = self.clone();
        diagnostics.on_run_scan(move || {
            self_clone.run_scan();
        });

        let self_clone = self.clone();
        diagnostics.on_cancel_scan(move || {
            self_clone.cancel_scan();
        });
    }

    fn handle_path_dropped(&self, path_str: &str) {
        let p = std::path::Path::new(path_str);
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            if p.is_dir() {
                let mut paths = scan_config.get_paths();
                if paths.dir_a.trim().is_empty() {
                    paths.dir_a = path_str.into();
                    self.notifier
                        .notify_info(&format!("Folder A set via Drag-and-Drop: {}", path_str));
                } else {
                    paths.dir_b = path_str.into();
                    self.notifier
                        .notify_info(&format!("Folder B set via Drag-and-Drop: {}", path_str));
                }
                scan_config.set_paths(paths);
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            } else if p.is_file() {
                let mut paths = scan_config.get_paths();
                paths.query_text = path_str.into();
                scan_config.set_paths(paths);
                scan_config.set_search_method(SearchMethod::Ai);
                self.notifier.notify_info(&format!(
                    "Reference Image set via Drag-and-Drop: {}",
                    path_str
                ));
                utils::settings::save_settings(&scan_config, &ui.global::<ViewportState>());
            }
        }
    }

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
                scan_config.set_search_method(SearchMethod::Ai);
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
        self.notifier
            .notify_info("User requested scan cancellation. Waiting for threads to stop...");
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
        self.notifier.notify_info(&format!(
            "Starting scan on directory: {}",
            params.paths.dir_a
        ));

        let params_for_task = params.clone();
        let params_for_task_clone = params_for_task.clone();
        let ui_weak_download = self.ui_weak.clone();
        let store_clone = self.store.clone();
        let ui_weak_finish = self.ui_weak.clone();
        let notifier_clone = self.notifier.clone();

        tokio::spawn(async move {
            // Verify models on Hugging Face if AI search method is active
            if params_for_task.search_method == SearchMethod::Ai
                && let Err(e) = crate::ai::downloader::verify_and_download_models(
                    ui_weak_download.clone(),
                    params_for_task.ai.ai_model,
                    params_for_task.cancel_token.clone(),
                )
                .await
            {
                let _ = ui_weak_download.upgrade_in_event_loop(move |ui| {
                    ui.global::<Diagnostics>().set_is_scanning(false);
                });
                notifier_clone.notify_error(&e, "AI model verification failed");
                return;
            }

            // Sync GUI tonemapping state to thread-safe app settings context
            if let Some(ref_ui) = ui_weak_download.upgrade() {
                let vp = ref_ui.global::<ViewportState>();
                crate::app::update_viewer_settings(|s| {
                    let tonemap = vp.get_tonemap();
                    s.tonemap_enabled = tonemap.tonemap_enabled;
                    s.auto_exposure_enabled = tonemap.tonemap_auto_exposure;
                    s.tonemap_operator = tonemap.tonemap_operator as usize;
                });
            }

            // Execute scan by routing directly to the domain scanner engines
            let scan_result = match params_for_task.search_method {
                SearchMethod::Qc => {
                    if !params_for_task.paths.dir_b.trim().is_empty() {
                        let ex_folders = utils::helpers::parse_excluded_folders(
                            &params_for_task.paths.excluded_folders,
                        );
                        match crate::qc::scanner::run_folder_compare(
                            params_for_task.paths.dir_a.clone(),
                            params_for_task.paths.dir_b.clone(),
                            params_for_task.extensions.clone(),
                            params_for_task.qc.qc_match_by_stem,
                            params_for_task.qc.qc_hide_same_resolution,
                            ex_folders,
                            params_for_task.qc.qc_check_bloat,
                            params_for_task.qc.qc_check_alpha,
                            params_for_task.qc.qc_check_colorspace,
                            params_for_task.qc.qc_check_compression,
                        )
                        .await
                        {
                            Ok(issues) => Ok((Vec::new(), issues, Vec::new())),
                            Err(e) => Err(anyhow::anyhow!("Folder comparison failed: {}", e)),
                        }
                    } else {
                        match crate::qc::scanner::run_qc_scan_internal(params_for_task_clone).await
                        {
                            Ok(issues) => Ok((Vec::new(), issues, Vec::new())),
                            Err(e) => Err(anyhow::anyhow!("QC scan failed: {}", e)),
                        }
                    }
                }
                SearchMethod::Inventory => {
                    match crate::qc::scanner::run_asset_audit(params_for_task_clone).await {
                        Ok(rows) => Ok((Vec::new(), Vec::new(), rows)),
                        Err(e) => Err(anyhow::anyhow!("Inventory audit failed: {}", e)),
                    }
                }
                SearchMethod::Ai => {
                    if !params_for_task.paths.query_text.trim().is_empty() {
                        match crate::ai::scanner::run_ai_search(params_for_task_clone).await {
                            Ok(matches) => {
                                // Map semantic AI results to inventory format for simpler direct layout reuse
                                let mapped: Vec<crate::state::models::DuplicateFileSummary> =
                                    matches
                                        .into_iter()
                                        .map(|res| crate::state::models::DuplicateFileSummary {
                                            path: res.path,
                                            size: 0,
                                            width: 0,
                                            height: 0,
                                            format_str: String::new(),
                                            compression_format: String::new(),
                                            color_space: String::new(),
                                            has_alpha: false,
                                            bit_depth: 8,
                                            mipmap_count: 1,
                                            is_cubemap: false,
                                            similarity: res.similarity,
                                        })
                                        .collect();
                                Ok((Vec::new(), Vec::new(), mapped))
                            }
                            Err(e) => Err(anyhow::anyhow!("AI Semantic Search failed: {}", e)),
                        }
                    } else {
                        match crate::ai::scanner::run_ai_duplicate_scan(params_for_task_clone).await
                        {
                            Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                            Err(e) => Err(anyhow::anyhow!("AI Duplicate Scan failed: {}", e)),
                        }
                    }
                }
                SearchMethod::Perceptual => {
                    match crate::perceptual::scanner::run_perceptual_scan_internal(
                        params_for_task_clone,
                    )
                    .await
                    {
                        Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                        Err(e) => Err(anyhow::anyhow!("Perceptual duplicate scan failed: {}", e)),
                    }
                }
                SearchMethod::Exact => {
                    match crate::exact::scanner::run_exact_scan(params_for_task_clone).await {
                        Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                        Err(e) => Err(anyhow::anyhow!("Byte-exact scan failed: {}", e)),
                    }
                }
            };

            // Generate optional contact sheets on duplicate clusters
            if params_for_task.visuals.save_visuals
                && let Ok((ref groups, _, _)) = scan_result
                && !groups.is_empty()
            {
                let _ = ui_weak_finish.upgrade_in_event_loop(|ui| {
                    let diag = ui.global::<Diagnostics>();
                    diag.set_status_text("Generating contact sheet reports...".into());
                });

                if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
                    let out_dir = app_dir.join("duplicate_visuals");
                    if let Err(e) = generate_visual_reports(
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
                diag.set_is_scanning(false);
                diag.set_progress(1.0);
                match scan_result {
                    Ok((groups, qc_issues, inventory_files)) => {
                        // Automatically updates Slint models using reactive store updates
                        store_clone.update(|state| {
                            state.collapsed_groups.clear();
                            state.checked_paths.clear();
                            state.groups = groups;
                            state.qc_issues = qc_issues;
                            state.inventory_files = inventory_files;
                        });

                        let msg = if params_for_task.visuals.save_visuals {
                            "Scan and visual reports finished successfully!"
                        } else {
                            "Scan finished successfully!"
                        };
                        notifier_clone.notify_success(msg);
                    }
                    Err(e) => {
                        notifier_clone.notify_error(&e, "Scanning process halted");
                    }
                }
            });
        });
    }

    fn expand_all_groups(&self) {
        self.store.update(|state| {
            state.collapsed_groups.clear();
        });
    }

    fn collapse_all_groups(&self) {
        self.store.update(|state| {
            state.collapsed_groups.clear();

            let header_indices: Vec<i32> = if !state.qc_issues.is_empty() {
                let mut grouped = std::collections::HashSet::new();
                for issue in &state.qc_issues {
                    grouped.insert(issue.issue.clone());
                }
                (0..grouped.len() as i32).collect()
            } else {
                (0..state.groups.len() as i32).collect()
            };

            for group_index in header_indices {
                state.collapsed_groups.insert(group_index);
            }
        });
    }

    fn clear_cache(&self) {
        let manager = crate::utils::cache::get_cache_manager();
        manager.decoded_images.invalidate_all();
        manager.thumbnails.invalidate_all();

        let self_weak = self.ui_weak.clone();
        let notifier_clone = self.notifier.clone();

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
                    if success {
                        notifier_clone.notify_success("Scan database and thumbnail caches cleared successfully.");
                    } else {
                        notifier_clone.notify_info("Warning: Some cache files could not be cleared (files might be in use).");
                    }
                });
            }
        });
    }

    fn clear_models(&self) {
        if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
            let _ = std::fs::remove_dir_all(app_dir.join("models"));
            self.notifier
                .notify_success("Downloaded AI model weights successfully cleared from disk.");
        }
    }

    fn auto_size_columns(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let results_empty = self.store.read(|state| {
                state.groups.is_empty()
                    && state.qc_issues.is_empty()
                    && state.inventory_files.is_empty()
            });
            if results_empty {
                return;
            }

            let (max_file_len, max_path_len) = self.store.read(|state| {
                let mut f_len = 4;
                let mut p_len = 4;
                for group in &state.groups {
                    for file in &group.files {
                        let name = std::path::Path::new(&file.path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy();
                        f_len = f_len.max(name.chars().count());
                        p_len = p_len.max(file.path.chars().count());
                    }
                }
                for issue in &state.qc_issues {
                    let name = std::path::Path::new(&issue.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    f_len = f_len.max(name.chars().count());
                    p_len = p_len.max(issue.path.chars().count());
                }
                for file in &state.inventory_files {
                    let name = std::path::Path::new(&file.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    f_len = f_len.max(name.chars().count());
                    p_len = p_len.max(file.path.chars().count());
                }
                (f_len, p_len)
            });

            let file_w = (max_file_len as f32 * 7.2) + 68.0;
            let score_w = 80.0f32;
            let path_w = (max_path_len as f32 * 6.5) + 20.0;

            let scan_config = ui.global::<ScanConfig>();
            scan_config.set_col_file_w(file_w.clamp(120.0, 600.0));
            scan_config.set_col_score_w(score_w);
            scan_config.set_col_path_w(path_w.clamp(150.0, 800.0));

            self.notifier.notify_info(&format!(
                "Columns auto-resized. FILE: {:.0}px, SCORE: {:.0}px, PATH: {:.0}px",
                file_w, score_w, path_w
            ));
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
            self.notifier.notify_info("Column sizes reset to defaults.");
        }
    }

    fn grid_columns_changed(&self) {
        self.store.update(|_| {});
    }

    fn save_settings(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            let viewport_state = ui.global::<ViewportState>();
            utils::settings::save_settings(&scan_config, &viewport_state);
        }
    }

    fn results_filter_changed(&self) {
        self.store.update(|_| {});
    }

    fn results_sort_changed(&self, sort_idx: i32) {
        self.store.update(|state| {
            if !state.groups.is_empty() {
                match sort_idx {
                    0 => state
                        .groups
                        .sort_by_key(|g| std::cmp::Reverse(g.files.len())),
                    1 => state.groups.sort_by_key(|g| {
                        std::cmp::Reverse(g.files.iter().map(|f| f.size).sum::<u64>())
                    }),
                    2 => state.groups.sort_by(|a, b| {
                        let name_a = a.files.first().map(|f| f.path.as_str()).unwrap_or("");
                        let name_b = b.files.first().map(|f| f.path.as_str()).unwrap_or("");
                        name_a.cmp(name_b)
                    }),
                    _ => {}
                }
            } else if !state.inventory_files.is_empty() {
                match sort_idx {
                    0 => state
                        .inventory_files
                        .sort_by(|a, b| a.format_str.cmp(&b.format_str)),
                    1 => state
                        .inventory_files
                        .sort_by_key(|r| std::cmp::Reverse(r.size)),
                    2 => state.inventory_files.sort_by(|a, b| a.path.cmp(&b.path)),
                    _ => {}
                }
            }
        });
    }

    fn sort_by_column(&self, col: &str) {
        let (sort_col, asc) = self.store.update(|state| {
            let col_type = SortColumn::from(col);

            if state.sort_column == col {
                state.sort_ascending = !state.sort_ascending;
            } else {
                state.sort_column = col.to_string();
                state.sort_ascending = true;
            }

            let asc = state.sort_ascending;

            if !state.groups.is_empty() {
                for group in &mut state.groups {
                    if group.files.len() > 1 {
                        let (_, duplicates) = group.files.split_at_mut(1);
                        duplicates
                            .sort_by(|a, b| self.compare_duplicate_files(col_type, a, b, asc));
                    }
                }

                state
                    .groups
                    .sort_by(|a, b| self.compare_duplicate_groups(col_type, a, b, asc));
            } else if !state.inventory_files.is_empty() {
                match col_type {
                    SortColumn::Name | SortColumn::Path => {
                        state.inventory_files.sort_by(|a, b| {
                            compare_ord(&a.path.to_lowercase(), &b.path.to_lowercase(), asc)
                        });
                    }
                    SortColumn::Size => {
                        state
                            .inventory_files
                            .sort_by(|a, b| compare_ord(a.size, b.size, asc));
                    }
                    SortColumn::Format => {
                        state.inventory_files.sort_by(|a, b| {
                            compare_ord(
                                &a.compression_format.to_lowercase(),
                                &b.compression_format.to_lowercase(),
                                asc,
                            )
                        });
                    }
                    SortColumn::Dimensions => {
                        state.inventory_files.sort_by(|a, b| {
                            compare_ord(a.width * a.height, b.width * b.height, asc)
                        });
                    }
                    SortColumn::Mipmaps => {
                        state
                            .inventory_files
                            .sort_by(|a, b| compare_ord(a.mipmap_count, b.mipmap_count, asc));
                    }
                    _ => {}
                }
            }
            (state.sort_column.clone(), asc)
        });

        if let Some(ui) = self.ui_weak.upgrade() {
            let scan_config = ui.global::<ScanConfig>();
            scan_config.set_active_sort_column(sort_col.into());
            scan_config.set_sort_ascending(asc);
        }
    }

    fn compare_duplicate_files(
        &self,
        col: SortColumn,
        a: &crate::state::models::DuplicateFileSummary,
        b: &crate::state::models::DuplicateFileSummary,
        asc: bool,
    ) -> std::cmp::Ordering {
        match col {
            SortColumn::Name | SortColumn::Path => {
                compare_ord(&a.path.to_lowercase(), &b.path.to_lowercase(), asc)
            }
            SortColumn::Size => compare_ord(a.size, b.size, asc),
            SortColumn::Score => compare_partial(a.similarity, b.similarity, asc),
            _ => std::cmp::Ordering::Equal,
        }
    }

    fn compare_duplicate_groups(
        &self,
        col: SortColumn,
        a: &crate::state::models::DuplicateGroupSummary,
        b: &crate::state::models::DuplicateGroupSummary,
        asc: bool,
    ) -> std::cmp::Ordering {
        match col {
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
                compare_ord(p_a, p_b, asc)
            }
            SortColumn::Size => {
                let s_a = a.files.first().map(|f| f.size).unwrap_or(0);
                let s_b = b.files.first().map(|f| f.size).unwrap_or(0);
                compare_ord(s_a, s_b, asc)
            }
            SortColumn::Score => {
                let sim_a = a.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                let sim_b = b.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                compare_partial(sim_a, sim_b, asc)
            }
            _ => std::cmp::Ordering::Equal,
        }
    }
}
