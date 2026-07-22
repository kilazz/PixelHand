// src/viewer/viewport.rs

use crate::app::{AppWindow, ViewportState};
use crate::utils::cache::get_channel_preview_image;
use crate::utils::image_processing::{
    apply_aspect_ratio, generate_histogram_image, slice_spritesheet_frame,
};
use crate::utils::slint_conversions::{convert_to_slint_image, get_current_active_channel};
use crate::viewer::tonemapping::{apply_manual_corrections, calculate_difference_map};
use slint::ComponentHandle;
use slint::Model;

/// Isolated parameter DTO extracted from UI thread state for safe background processing.
struct ViewportStateParams {
    channel: String,
    compare_mode: i32,
    mip_level: u32,
    brightness: f32,
    contrast: f32,
    gamma: f32,
    ratio: f32,
    grid_cols: u32,
    grid_rows: u32,
    active_frame: u32,
    enable_blending: bool,
    anim_source: i32,
}

/// Triggers an asynchronous, non-blocking viewport preview update.
/// Retrieves cached texture channel channels, slices spritesheet sub-regions or resolves sequence files,
/// applies manual color corrections and tonemapping configuration metrics, and dispatches outputs back to Slint's UI thread.
pub fn trigger_viewport_update(
    app_weak: slint::Weak<AppWindow>,
    orig_path: String,
    dup_path: String,
) {
    let ui = match app_weak.upgrade() {
        Some(ui) => ui,
        None => return,
    };
    let viewport_state = ui.global::<ViewportState>();
    let viewer = viewport_state.get_viewer();

    let anim_source = viewport_state.get_anim_source();
    let current_frame_idx = viewport_state.get_active_frame() as usize;

    // Extract thread-safe parameters to prevent capturing active Slint component reference handles
    let params = ViewportStateParams {
        channel: get_current_active_channel(&viewport_state).to_string(),
        compare_mode: viewport_state.get_compare_mode(),
        mip_level: viewport_state.get_active_mip_level() as u32,
        brightness: viewer.manual_brightness,
        contrast: viewer.manual_contrast,
        gamma: viewer.manual_gamma,
        ratio: viewer.aspect_ratio_modifier,
        grid_cols: viewer.grid_cols as u32,
        grid_rows: viewer.grid_rows as u32,
        active_frame: current_frame_idx as u32,
        enable_blending: viewer.enable_frame_blending,
        anim_source,
    };

    // Determine total frame count and active duplicate paths based on animation mode
    let scan_config = ui.global::<crate::app::ScanConfig>();
    let group_files = scan_config.get_selected_group_files();
    let total_group_files = group_files.row_count();

    let total_frames = if anim_source == 1 {
        total_group_files as u32
    } else {
        params.grid_cols * params.grid_rows
    };

    let next_frame = if total_frames > 1 {
        (params.active_frame + 1) % total_frames
    } else {
        params.active_frame
    };

    // Resolves current and next duplicate frame paths when playing a group files sequence
    let active_dup_path = if anim_source == 1 && total_group_files > 0 {
        let clamped_idx = current_frame_idx.min(total_group_files - 1);
        group_files
            .row_data(clamped_idx)
            .map(|r| r.path.to_string())
            .unwrap_or_else(|| dup_path.clone())
    } else {
        dup_path.clone()
    };

    let next_dup_path = if anim_source == 1 && total_group_files > 0 {
        let clamped_idx = (next_frame as usize).min(total_group_files - 1);
        group_files
            .row_data(clamped_idx)
            .map(|r| r.path.to_string())
            .unwrap_or_else(|| dup_path.clone())
    } else {
        dup_path.clone()
    };

    // Keep global thread-safe settings context updated before launching tasks
    crate::app::update_viewer_settings(|s| {
        let tonemap = viewport_state.get_tonemap();
        s.tonemap_enabled = tonemap.tonemap_enabled;
        s.auto_exposure_enabled = tonemap.tonemap_auto_exposure;
        s.tonemap_operator = tonemap.tonemap_operator as usize;
    });

    let app_weak_clone = app_weak.clone();

    tokio::spawn(async move {
        // Fetch requested channel preview buffers asynchronously (disk/memory-bound cache hits)
        let raw_orig =
            get_channel_preview_image(&orig_path, &params.channel, params.mip_level).await;
        let raw_dup =
            get_channel_preview_image(&active_dup_path, &params.channel, params.mip_level).await;

        // Fetch next frame buffers if frame blending animation is active
        let (raw_orig_next, raw_dup_next) = if params.enable_blending && total_frames > 1 {
            if params.anim_source == 1 {
                (
                    raw_orig.clone(),
                    get_channel_preview_image(&next_dup_path, &params.channel, params.mip_level)
                        .await,
                )
            } else {
                (raw_orig.clone(), raw_dup.clone())
            }
        } else {
            (None, None)
        };

        // Offload CPU-bound pixel math operations to a dedicated blocking worker thread pool
        let result = tokio::task::spawn_blocking(move || {
            let mut o = raw_orig;
            let mut d = raw_dup;
            let mut o_n = raw_orig_next;
            let mut d_n = raw_dup_next;

            // Perform sub-region frame cropping ONLY in Spritesheet Grid mode (anim_source == 0)
            if params.anim_source == 0 && (params.grid_cols > 1 || params.grid_rows > 1) {
                if let Some(ref img) = o {
                    o = Some(slice_spritesheet_frame(
                        img,
                        params.grid_cols,
                        params.grid_rows,
                        params.active_frame,
                    ));
                }
                if let Some(ref img) = d {
                    d = Some(slice_spritesheet_frame(
                        img,
                        params.grid_cols,
                        params.grid_rows,
                        params.active_frame,
                    ));
                }

                // Perform sub-region frame cropping (Next Frame)
                if params.enable_blending {
                    if let Some(ref img) = o_n {
                        o_n = Some(slice_spritesheet_frame(
                            img,
                            params.grid_cols,
                            params.grid_rows,
                            next_frame,
                        ));
                    }
                    if let Some(ref img) = d_n {
                        d_n = Some(slice_spritesheet_frame(
                            img,
                            params.grid_cols,
                            params.grid_rows,
                            next_frame,
                        ));
                    }
                }
            }

            // Apply Brightness, Contrast, and Gamma mathematical corrections via LUT (Current Frame)
            if let Some(ref mut img) = o {
                apply_manual_corrections(img, params.brightness, params.contrast, params.gamma);
            }
            if let Some(ref mut img) = d {
                apply_manual_corrections(img, params.brightness, params.contrast, params.gamma);
            }

            // Apply Brightness, Contrast, and Gamma mathematical corrections via LUT (Next Frame)
            if params.enable_blending {
                if let Some(ref mut img) = o_n {
                    apply_manual_corrections(img, params.brightness, params.contrast, params.gamma);
                }
                if let Some(ref mut img) = d_n {
                    apply_manual_corrections(img, params.brightness, params.contrast, params.gamma);
                }
            }

            // Apply aspect ratio resizing multipliers (Current Frame)
            if let Some(ref img) = o {
                o = Some(apply_aspect_ratio(img, params.ratio));
            }
            if let Some(ref img) = d {
                d = Some(apply_aspect_ratio(img, params.ratio));
            }

            // Apply aspect ratio resizing multipliers (Next Frame)
            if params.enable_blending {
                if let Some(ref img) = o_n {
                    o_n = Some(apply_aspect_ratio(img, params.ratio));
                }
                if let Some(ref img) = d_n {
                    d_n = Some(apply_aspect_ratio(img, params.ratio));
                }
            }

            // Extract exact dimensions before returning to the main thread
            let (sprite_w, sprite_h) = if let Some(ref img) = o {
                (img.width() as i32, img.height() as i32)
            } else {
                (0, 0)
            };

            // Compute difference heatmap or raw subtraction difference matrix
            let raw_diff = if params.compare_mode == 3
                && let (Some(orig), Some(dup)) = (&o, &d)
            {
                calculate_difference_map(orig, dup, true).ok()
            } else {
                None
            };

            // Build overlapping grayscale histograms
            let hist_img = if let (Some(orig), Some(dup)) = (&o, &d) {
                Some(generate_histogram_image(orig, dup))
            } else {
                None
            };

            (sprite_w, sprite_h, o, d, o_n, d_n, raw_diff, hist_img)
        })
        .await;

        // Safely update Slint singletons inside Slint's Event Loop on task completion
        match result {
            Ok((sprite_w, sprite_h, o, d, o_n, d_n, raw_diff, hist_img)) => {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let viewport_state = ui.global::<ViewportState>();
                    let diag = ui.global::<crate::app::Diagnostics>();

                    viewport_state.set_sprite_width(sprite_w);
                    viewport_state.set_sprite_height(sprite_h);
                    viewport_state.set_sprites_count(total_frames as i32);

                    if let Some(img) = o {
                        viewport_state.set_image_original(convert_to_slint_image(&img));
                    } else {
                        diag.set_status_text(
                            "Error: Failed to decode target image format. Check Log tab.".into(),
                        );
                    }

                    if let Some(img) = d {
                        viewport_state.set_image_duplicate(convert_to_slint_image(&img));
                    }

                    if params.enable_blending {
                        if let Some(img) = o_n {
                            viewport_state.set_image_original_next(convert_to_slint_image(&img));
                        }
                        if let Some(img) = d_n {
                            viewport_state.set_image_duplicate_next(convert_to_slint_image(&img));
                        }
                    }

                    if let Some(diff) = raw_diff {
                        viewport_state.set_image_heatmap(convert_to_slint_image(&diff));
                    }
                    if let Some(hist) = hist_img {
                        viewport_state.set_histogram_image(convert_to_slint_image(&hist));
                    }
                });
            }
            Err(e) => {
                tracing::error!(
                    "Blocking viewport processing task failed or panicked: {:?}",
                    e
                );
            }
        }
    });
}
