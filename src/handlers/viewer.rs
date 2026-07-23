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

        let self_clone = self.clone();
        viewport_state.on_inspect_pixel(move |x, y| {
            self_clone.inspect_pixel(x as u32, y as u32);
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

    fn inspect_pixel(&self, x: u32, y: u32) {
        let ui = match self.ui_weak.upgrade() {
            Some(ui) => ui,
            None => return,
        };

        let vp = ui.global::<ViewportState>();
        let orig_path = vp.get_original_meta().path.to_string();
        let dup_path = vp.get_duplicate_meta().path.to_string();
        let ui_weak = self.ui_weak.clone();

        tokio::spawn(async move {
            let orig_img =
                crate::utils::cache::get_channel_preview_image(&orig_path, "RGB", 0).await;
            let dup_img = crate::utils::cache::get_channel_preview_image(&dup_path, "RGB", 0).await;

            if let Some(o) = orig_img
                && x < o.width()
                && y < o.height()
            {
                let p = o.get_pixel(x, y);
                let r = p[0];
                let g = p[1];
                let b = p[2];
                let a = p[3];

                let hex = format!("#{:02X}{:02X}{:02X}{:02X}", r, g, b, a);

                // Calculate Tangent-Space Normal Vector N = (X, Y, Z)
                let mut nx = (r as f32 / 255.0) * 2.0 - 1.0;
                let mut ny = (g as f32 / 255.0) * 2.0 - 1.0;
                let mut nz = (b as f32 / 255.0) * 2.0 - 1.0;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                if len > 1e-4 {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }

                // Calculate scale ratio to map inspector sampling coordinates dynamically
                // when original and duplicate images have mismatched resolutions (e.g. 2048x2048 vs 1024x1024)
                let diff_str = if let Some(d) = dup_img {
                    let scale_x = d.width() as f32 / o.width() as f32;
                    let scale_y = d.height() as f32 / o.height() as f32;

                    let dx = ((x as f32 * scale_x).round() as u32).min(d.width().saturating_sub(1));
                    let dy =
                        ((y as f32 * scale_y).round() as u32).min(d.height().saturating_sub(1));

                    let dp = d.get_pixel(dx, dy);
                    let delta = (p[0].abs_diff(dp[0]) as f32
                        + p[1].abs_diff(dp[1]) as f32
                        + p[2].abs_diff(dp[2]) as f32)
                        / 3.0;
                    format!("{:.1}% (Δ {:.0})", (delta / 255.0) * 100.0, delta)
                } else {
                    "-".to_string()
                };

                let _ = ui_weak.upgrade_in_event_loop(move |ui| {
                    let vp = ui.global::<ViewportState>();
                    let mut insp = vp.get_inspection();
                    insp.valid = true;
                    insp.x = x as i32;
                    insp.y = y as i32;
                    insp.r_255 = r as i32;
                    insp.g_255 = g as i32;
                    insp.b_255 = b as i32;
                    insp.a_255 = a as i32;
                    insp.hex_code = hex.into();
                    insp.norm_x = (nx * 100.0).round() / 100.0;
                    insp.norm_y = (ny * 100.0).round() / 100.0;
                    insp.norm_z = (nz * 100.0).round() / 100.0;
                    insp.pixel_diff = diff_str.into();
                    vp.set_inspection(insp);
                });
            }
        });
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
