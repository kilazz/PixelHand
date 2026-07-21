// src/handlers/actions.rs

use slint::ComponentHandle;
use slint::Model;
use slint::ModelRc;
use slint::VecModel;
use std::sync::Arc;

use crate::app::{AppWindow, Diagnostics, ScanConfig, SelectedFile, ViewportState};
use crate::state::store::AppStateStore;
use crate::utils;
use crate::utils::notification::NotificationService;
use crate::utils::slint_conversions::{build_selected_file_meta, convert_to_slint_row};
use crate::viewer::viewport::trigger_viewport_update;

pub struct ActionsController {
    ui_weak: slint::Weak<AppWindow>,
    store: AppStateStore,
    notifier: Arc<NotificationService>,
}

impl ActionsController {
    pub fn new(
        ui_weak: slint::Weak<AppWindow>,
        store: AppStateStore,
        notifier: Arc<NotificationService>,
    ) -> Arc<Self> {
        Arc::new(Self {
            ui_weak,
            store,
            notifier,
        })
    }

    pub fn register(self: &Arc<Self>, ui: &AppWindow) {
        let scan_config = ui.global::<ScanConfig>();

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
        scan_config.on_context_menu_action(move |action, path| {
            self_clone.handle_context_menu_action(action.as_str(), path.as_str());
        });

        let self_clone = self.clone();
        scan_config.on_export_html(move || {
            self_clone.export_html();
        });

        // Register diagnostics / log exports
        let diagnostics = ui.global::<Diagnostics>();

        let self_clone = self.clone();
        diagnostics.on_export_log(move |log| {
            self_clone.export_log(log.as_str());
        });

        let self_clone = self.clone();
        diagnostics.on_export_csv(move || {
            self_clone.export_csv();
        });
    }

    fn trigger_action(&self, action: &str) {
        if self.ui_weak.upgrade().is_none() {
            return;
        }

        let self_weak = self.ui_weak.clone();
        let store_clone = self.store.clone();
        let action_owned = action.to_string();
        let notifier_clone = self.notifier.clone();

        tokio::spawn(async move {
            let (checked_files, pairs) = store_clone.read(utils::fs::extract_selected_files);

            if checked_files.is_empty() {
                return;
            }

            notifier_clone.notify_info(&format!("Processing selection: {}...", action_owned));

            let res = utils::fs::execute_file_action(&action_owned, checked_files, pairs).await;

            let _ = self_weak.upgrade_in_event_loop(move |ui| {
                let scan_cfg = ui.global::<ScanConfig>();
                match res {
                    Ok(_) => {
                        notifier_clone.notify_success(&format!(
                            "Successfully completed {} operation.",
                            action_owned
                        ));
                        scan_cfg.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(
                            Vec::new(),
                        ))));
                        scan_cfg.set_has_results(false);

                        // Clear and automatically commit/flush to UI
                        store_clone.update(|state| {
                            state.groups.clear();
                            state.qc_issues.clear();
                            state.inventory_files.clear();
                            state.checked_paths.clear();
                            state.collapsed_groups.clear();
                        });
                    }
                    Err(e) => {
                        notifier_clone
                            .notify_error(&e, &format!("Operation '{}' failed", action_owned));
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
            self.store.update(|state| {
                if state.collapsed_groups.contains(&group_idx) {
                    state.collapsed_groups.remove(&group_idx);
                } else {
                    state.collapsed_groups.insert(group_idx);
                }
            });
            return;
        }

        let path_str = path.to_string();

        let (group, qc_issues_empty, inventory_empty) = self.store.read(|state| {
            let group = state.groups.get(group_idx as usize).cloned();
            (
                group,
                state.qc_issues.is_empty(),
                state.inventory_files.is_empty(),
            )
        });

        let group = match group {
            None => {
                if qc_issues_empty && inventory_empty {
                    return;
                }
                let meta = crate::qc::rules::QcImageMetadata::extract_or_fallback(
                    std::path::Path::new(&path_str),
                );
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

                scan_config.set_selected_group_files(ModelRc::from(std::rc::Rc::new(
                    VecModel::from(Vec::new()),
                )));
                trigger_viewport_update(self.ui_weak.clone(), path_str.clone(), path_str);
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

        viewport_state.set_original_meta(build_selected_file_meta(original, true));
        viewport_state.set_duplicate_meta(build_selected_file_meta(duplicate, false));
        viewport_state.set_max_available_mips(original.mipmap_count as i32);
        viewport_state.set_active_mip_level(0);

        let group_files = self.store.read(|state| {
            let mut files = Vec::new();
            for file in &group.files {
                let is_checked = state.checked_paths.contains(&file.path);
                let is_best = file.path == original.path;
                files.push(convert_to_slint_row(file, is_best, is_checked, group_idx));
            }
            files
        });

        scan_config
            .set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));
        trigger_viewport_update(
            self.ui_weak.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    }

    fn handle_row_checkbox_toggled(&self, idx: usize) {
        let is_checked = self.store.update(|state| {
            if let Some(ui) = self.ui_weak.upgrade() {
                let scan_config = ui.global::<ScanConfig>();
                let results_model = scan_config.get_results();

                if let Some(row) = results_model.row_data(idx) {
                    let path = row.path.to_string();
                    if state.checked_paths.contains(&path) {
                        state.checked_paths.remove(&path);
                        Some(false)
                    } else {
                        state.checked_paths.insert(path);
                        Some(true)
                    }
                } else {
                    None
                }
            } else {
                None
            }
        });

        if let Some(checked) = is_checked
            && let Some(ui) = self.ui_weak.upgrade()
        {
            let scan_config = ui.global::<ScanConfig>();
            let results_model = scan_config.get_results();

            if let Some(vec_model) = results_model
                .as_any()
                .downcast_ref::<VecModel<crate::app::ResultsRow>>()
                && let Some(mut slint_row) = vec_model.row_data(idx)
            {
                slint_row.is_checked = checked;
                vec_model.set_row_data(idx, slint_row);
            }
        }
    }

    fn trigger_selection_rule(&self, rule: &str) {
        self.store.update(|state| {
            crate::utils::slint_conversions::apply_selection_rule(state, rule);
        });
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
                if p.exists() && trash::delete(&p).is_ok() {
                    if self.ui_weak.upgrade().is_some() {
                        self.notifier.notify_success(&format!(
                            "Moved to trash: {}",
                            p.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    }

                    self.store.update(|state| {
                        let normalized = utils::fs::normalize_path(&path_str);
                        state
                            .qc_issues
                            .retain(|r| utils::fs::normalize_path(&r.path) != normalized);
                        state
                            .inventory_files
                            .retain(|r| utils::fs::normalize_path(&r.path) != normalized);
                        for group in &mut state.groups {
                            group
                                .files
                                .retain(|r| utils::fs::normalize_path(&r.path) != normalized);
                        }
                        state.groups.retain(|g| g.files.len() >= 2);
                    });
                }
            }
            "trash_group" => {
                if let Ok(group_idx) = path_str.parse::<i32>() {
                    let (paths_to_delete, target_issue_type) = self.store.read(|state| {
                        let mut paths = Vec::new();
                        let mut issue_type = None;

                        if !state.groups.is_empty() {
                            let group_idx_us = group_idx as usize;
                            if let Some(g) = state.groups.get(group_idx_us) {
                                paths = g
                                    .files
                                    .iter()
                                    .map(|f| std::path::PathBuf::from(&f.path))
                                    .collect();
                            }
                        } else if !state.qc_issues.is_empty() {
                            let mut grouped: std::collections::HashMap<
                                String,
                                Vec<&crate::state::models::QcIssueSummary>,
                            > = std::collections::HashMap::new();
                            for issue in &state.qc_issues {
                                grouped.entry(issue.issue.clone()).or_default().push(issue);
                            }
                            let mut sorted_types: Vec<String> = grouped.keys().cloned().collect();
                            sorted_types.sort();

                            if let Some(t) = sorted_types.get(group_idx as usize) {
                                issue_type = Some(t.clone());
                                if let Some(issues) = grouped.get(t) {
                                    paths = issues
                                        .iter()
                                        .map(|i| std::path::PathBuf::from(&i.path))
                                        .collect();
                                }
                            }
                        }
                        (paths, issue_type)
                    });

                    let files_exists: Vec<std::path::PathBuf> =
                        paths_to_delete.into_iter().filter(|p| p.exists()).collect();
                    let count = files_exists.len();
                    if !files_exists.is_empty() {
                        let _ = trash::delete_all(&files_exists);
                    }

                    self.store.update(|state| {
                        if !state.groups.is_empty() {
                            let group_idx_us = group_idx as usize;
                            if group_idx_us < state.groups.len() {
                                state.groups.remove(group_idx_us);
                            }
                        } else if let Some(ref target_issue) = target_issue_type {
                            state.qc_issues.retain(|r| &r.issue != target_issue);
                        }
                    });

                    self.notifier.notify_success(&format!(
                        "Moved {} files in the selected group to trash.",
                        count
                    ));
                }
            }
            _ => {}
        }
    }

    fn export_log(&self, log_text: &str) {
        utils::export::export_diagnostics_log(log_text);
    }

    fn export_csv(&self) {
        utils::export::export_results_to_csv(self.store.get_state_mutex());
    }

    fn export_html(&self) {
        let (groups, qc_issues, inventory) = self.store.read(|state| {
            (
                state.groups.clone(),
                state.qc_issues.clone(),
                state.inventory_files.clone(),
            )
        });

        if groups.is_empty() && qc_issues.is_empty() && inventory.is_empty() {
            self.notifier.notify_info("No data available to export.");
            return;
        }

        if let Some(path) = rfd::FileDialog::new()
            .set_title("Export Interactive HTML/PDF Report")
            .add_filter("HTML Document / PDF Print Ready", &["html"])
            .set_file_name("PixelHand_Report.html")
            .save_file()
        {
            match crate::reporting::html_report::generate_html_report(
                &groups, &qc_issues, &inventory, &path,
            ) {
                Ok(_) => {
                    self.notifier.notify_success(&format!(
                        "Successfully exported HTML visual report to: {:?}",
                        path
                    ));
                }
                Err(e) => {
                    self.notifier
                        .notify_error(&e, "Failed to write HTML report to disk");
                }
            }
        }
    }
}
