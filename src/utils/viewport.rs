// src/utils/viewport.rs

use crate::app::{AppWindow, ViewportState};
use crate::utils;
use slint::ComponentHandle;

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
}

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

    // Extract thread-safe viewport settings to avoid capturing Slint handles inside async tasks
    let params = ViewportStateParams {
        channel: utils::ui::get_current_active_channel(&viewport_state).to_string(),
        compare_mode: viewport_state.get_compare_mode(),
        mip_level: viewport_state.get_active_mip_level() as u32,
        brightness: viewer.manual_brightness,
        contrast: viewer.manual_contrast,
        gamma: viewer.manual_gamma,
        ratio: viewer.aspect_ratio_modifier,
        grid_cols: viewer.grid_cols as u32,
        grid_rows: viewer.grid_rows as u32,
        active_frame: viewport_state.get_active_frame() as u32,
        enable_blending: viewer.enable_frame_blending,
    };

    let total_frames = params.grid_cols * params.grid_rows;
    let next_frame = if total_frames > 1 {
        (params.active_frame + 1) % total_frames
    } else {
        params.active_frame
    };

    // Store state before asynchronous execution
    crate::app::update_viewer_settings(|s| {
        let tonemap = viewport_state.get_tonemap();
        s.tonemap_enabled = tonemap.tonemap_enabled;
        s.auto_exposure_enabled = tonemap.tonemap_auto_exposure;
        s.tonemap_operator = tonemap.tonemap_operator as usize;
    });

    let app_weak_clone = app_weak.clone();

    tokio::spawn(async move {
        // Fetch current frame previews asynchronously (disk/memory bound)
        let raw_orig =
            utils::cache::get_channel_preview_image(&orig_path, &params.channel, params.mip_level)
                .await;
        let raw_dup =
            utils::cache::get_channel_preview_image(&dup_path, &params.channel, params.mip_level)
                .await;

        // Fetch next frame previews for blending
        let (raw_orig_next, raw_dup_next) = if params.enable_blending && total_frames > 1 {
            (
                utils::cache::get_channel_preview_image(
                    &orig_path,
                    &params.channel,
                    params.mip_level,
                )
                .await,
                utils::cache::get_channel_preview_image(
                    &dup_path,
                    &params.channel,
                    params.mip_level,
                )
                .await,
            )
        } else {
            (None, None)
        };

        // Offload ALL CPU-bound image operations to a dedicated blocking worker thread.
        let result = tokio::task::spawn_blocking(move || {
            let mut o = raw_orig;
            let mut d = raw_dup;
            let mut o_n = raw_orig_next;
            let mut d_n = raw_dup_next;

            // Perform Spritesheet slicing for Frame N (Current)
            if params.grid_cols > 1 || params.grid_rows > 1 {
                if let Some(ref img) = o {
                    o = Some(utils::ui::slice_spritesheet_frame(
                        img,
                        params.grid_cols,
                        params.grid_rows,
                        params.active_frame,
                    ));
                }
                if let Some(ref img) = d {
                    d = Some(utils::ui::slice_spritesheet_frame(
                        img,
                        params.grid_cols,
                        params.grid_rows,
                        params.active_frame,
                    ));
                }

                // Perform Spritesheet slicing for Frame N+1 (Next)
                if params.enable_blending {
                    if let Some(ref img) = o_n {
                        o_n = Some(utils::ui::slice_spritesheet_frame(
                            img,
                            params.grid_cols,
                            params.grid_rows,
                            next_frame,
                        ));
                    }
                    if let Some(ref img) = d_n {
                        d_n = Some(utils::ui::slice_spritesheet_frame(
                            img,
                            params.grid_cols,
                            params.grid_rows,
                            next_frame,
                        ));
                    }
                }
            }

            // Apply manual Brightness, Contrast, and Gamma color corrections (Current Frame)
            if let Some(ref mut img) = o {
                crate::core::tonemapper::apply_manual_corrections(
                    img,
                    params.brightness,
                    params.contrast,
                    params.gamma,
                );
            }
            if let Some(ref mut img) = d {
                crate::core::tonemapper::apply_manual_corrections(
                    img,
                    params.brightness,
                    params.contrast,
                    params.gamma,
                );
            }

            // Apply manual Brightness, Contrast, and Gamma color corrections (Next Frame)
            if params.enable_blending {
                if let Some(ref mut img) = o_n {
                    crate::core::tonemapper::apply_manual_corrections(
                        img,
                        params.brightness,
                        params.contrast,
                        params.gamma,
                    );
                }
                if let Some(ref mut img) = d_n {
                    crate::core::tonemapper::apply_manual_corrections(
                        img,
                        params.brightness,
                        params.contrast,
                        params.gamma,
                    );
                }
            }

            // Apply aspect ratio scaling multipliers (Ratio Slider - Current Frame)
            if let Some(ref img) = o {
                o = Some(utils::ui::apply_aspect_ratio(img, params.ratio));
            }
            if let Some(ref img) = d {
                d = Some(utils::ui::apply_aspect_ratio(img, params.ratio));
            }

            // Apply aspect ratio scaling multipliers (Ratio Slider - Next Frame)
            if params.enable_blending {
                if let Some(ref img) = o_n {
                    o_n = Some(utils::ui::apply_aspect_ratio(img, params.ratio));
                }
                if let Some(ref img) = d_n {
                    d_n = Some(utils::ui::apply_aspect_ratio(img, params.ratio));
                }
            }

            // Pre-extract width and height before moving to event loop
            let (sprite_w, sprite_h) = if let Some(ref img) = o {
                (img.width() as i32, img.height() as i32)
            } else {
                (0, 0)
            };

            let raw_diff = if params.compare_mode == 3 {
                if let (Some(orig), Some(dup)) = (&o, &d) {
                    crate::core::tonemapper::calculate_difference_map(orig, dup, true).ok()
                } else {
                    None
                }
            } else {
                None
            };

            let hist_img = if let (Some(orig), Some(dup)) = (&o, &d) {
                Some(utils::ui::generate_histogram_image(orig, dup))
            } else {
                None
            };

            (sprite_w, sprite_h, o, d, o_n, d_n, raw_diff, hist_img)
        })
        .await;

        // Unpack results and dispatch UI updates safely, logging errors on failure
        match result {
            Ok((sprite_w, sprite_h, o, d, o_n, d_n, raw_diff, hist_img)) => {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let viewport_state = ui.global::<ViewportState>();

                    viewport_state.set_sprite_width(sprite_w);
                    viewport_state.set_sprite_height(sprite_h);
                    viewport_state.set_sprites_count(total_frames as i32);

                    if let Some(img) = o {
                        viewport_state.set_image_original(utils::ui::convert_to_slint_image(&img));
                    }
                    if let Some(img) = d {
                        viewport_state.set_image_duplicate(utils::ui::convert_to_slint_image(&img));
                    }

                    if params.enable_blending {
                        if let Some(img) = o_n {
                            viewport_state
                                .set_image_original_next(utils::ui::convert_to_slint_image(&img));
                        }
                        if let Some(img) = d_n {
                            viewport_state
                                .set_image_duplicate_next(utils::ui::convert_to_slint_image(&img));
                        }
                    }

                    if let Some(diff) = raw_diff {
                        viewport_state.set_image_heatmap(utils::ui::convert_to_slint_image(&diff));
                    }
                    if let Some(hist) = hist_img {
                        viewport_state
                            .set_histogram_image(utils::ui::convert_to_slint_image(&hist));
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
