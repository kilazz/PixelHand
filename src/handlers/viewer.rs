// src/handlers/viewer.rs

use slint::ComponentHandle;
use slint::Model;
use std::sync::Arc;

use crate::app::{AppWindow, ScanConfig, ViewportState};
use crate::utils;
use crate::utils::slint_conversions::convert_to_slint_image;
use crate::viewer::viewport::trigger_viewport_update;

pub struct ViewerController {
    pub ui_weak: slint::Weak<AppWindow>,
}

impl ViewerController {
    pub fn new(ui_weak: slint::Weak<AppWindow>) -> Arc<Self> {
        Arc::new(Self { ui_weak })
    }

    pub fn register(self: &Arc<Self>, ui: &AppWindow) {
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

        let scan_config = ui.global::<ScanConfig>();
        let self_clone = self.clone();
        scan_config.on_thumbnail_channel_hovered(move |path, channel| {
            self_clone.thumbnail_channel_hovered(path.as_str(), channel.as_str());
        });
    }

    fn open_file_in_viewer(&self, path: &str) {
        let _ = open::that(path);
    }

    fn viewport_settings_changed(&self) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let vp = ui.global::<ViewportState>();
            let orig_path = vp.get_original_meta().path.to_string();
            let dup_path = vp.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                trigger_viewport_update(self.ui_weak.clone(), orig_path, dup_path);
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

    fn thumbnail_channel_hovered(&self, path: &str, channel: &str) {
        let path_std = path.to_string();
        let channel_std = channel.to_string();
        let ui_weak = self.ui_weak.clone();

        tokio::spawn(async move {
            let normalized_path_cache = utils::cache::normalize_path_key(&path_std);
            let normalized_path_fs = utils::fs::normalize_path_key(&path_std);

            let cached_img = {
                let manager = utils::cache::get_cache_manager();
                manager.thumbnails.get(&normalized_path_cache)
            };

            if let Some(cached_thumb) = cached_img {
                let channel_img = cached_thumb.get_channel(&channel_std);

                let _ = ui_weak.upgrade_in_event_loop(move |ui| {
                    let slint_img = convert_to_slint_image(&channel_img);
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
}
