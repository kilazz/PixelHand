// src/app.rs

use anyhow::{Context, Result};
use slint::ComponentHandle;
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use crate::state::{AppSettings, AppState};
use crate::utils;

// Generate Slint Rust code from the UI markup
slint::include_modules!();

// Global handle to push logs directly to the UI console from any thread
pub static APP_HANDLE: OnceLock<slint::Weak<AppWindow>> = OnceLock::new();

// Fast intermediate thread-safe queue to bypass immediate UI locked calls
static LOG_MESSAGES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

// Thread-safe settings context to bridge Slint states to background loops safely
#[derive(Debug, Clone, Copy)]
pub struct ViewerSettings {
    pub compare_mode: usize,
    pub flicker_interval: usize,
    pub tonemap_enabled: bool,
    pub auto_exposure_enabled: bool,
    pub tonemap_operator: usize,
    pub play_flipbook: bool,
    pub flipbook_fps: usize,
    pub brightness: usize, // 100 as 1.0
    pub contrast: usize,
    pub gamma: usize,
    pub ratio: usize,
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub active_frame: usize,
    pub play_speed: f32,
    pub enable_frame_blending: bool,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        Self {
            compare_mode: 0,
            flicker_interval: 333,
            tonemap_enabled: true,
            auto_exposure_enabled: true,
            tonemap_operator: 2,
            play_flipbook: false,
            flipbook_fps: 12,
            brightness: 100,
            contrast: 100,
            gamma: 100,
            ratio: 100,
            grid_cols: 1,
            grid_rows: 1,
            active_frame: 0,
            play_speed: 1.0,
            enable_frame_blending: false,
        }
    }
}

pub static VIEWER_SETTINGS: OnceLock<Mutex<ViewerSettings>> = OnceLock::new();

pub fn get_viewer_settings() -> ViewerSettings {
    let lock = VIEWER_SETTINGS.get_or_init(|| Mutex::new(ViewerSettings::default()));
    if let Ok(guard) = lock.lock() {
        *guard
    } else {
        ViewerSettings::default()
    }
}

pub fn update_viewer_settings<F>(f: F)
where
    F: FnOnce(&mut ViewerSettings),
{
    let lock = VIEWER_SETTINGS.get_or_init(|| Mutex::new(ViewerSettings::default()));
    if let Ok(mut guard) = lock.lock() {
        f(&mut guard);
    }
}

/// Custom tracing subscriber log handler wrapping native console pipes
struct UiLogWriter;

impl std::io::Write for UiLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(msg) = String::from_utf8(buf.to_vec()) {
            append_to_console_log(&msg);
        }

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

/// Strips ANSI terminal escape color sequences from tracing logs before pushing them to the UI.
fn clean_sample_log(raw: String) -> String {
    raw.replace("\u{001b}[2m", "")
        .replace("\u{001b}[0m", "")
        .replace("\u{001b}[32m", "")
        .replace("\u{001b}[33m", "")
        .replace("\u{001b}[31m", "")
}

/// Dispatches log updates to the batched global queues to protect the UI thread from bottlenecks.
pub fn append_to_console_log(msg: &str) {
    let clean_msg = msg.trim_end().to_string();
    if clean_msg.is_empty() {
        return;
    }

    let cleaned = clean_sample_log(clean_msg);

    // If the GUI window is loaded, stream logs through the throttled aggregator (200ms chunks)
    if APP_HANDLE.get().is_some() {
        let queue = PENDING_LOGS.get_or_init(|| Mutex::new(Vec::new()));
        if let Ok(mut lock) = queue.lock() {
            lock.push(cleaned);
        }
    } else {
        // Fallback to intermediate startup queue if UI event loop has not loaded yet
        let queue_mutex = LOG_MESSAGES.get_or_init(|| Mutex::new(Vec::new()));
        if let Ok(mut q) = queue_mutex.lock() {
            q.push(cleaned);
        }
    }
}

/// Dynamic state mapper producing a pure config instance for loaders on demand.
pub fn get_active_tonemap_config() -> crate::core::tonemapper::TonemapConfig {
    let s = get_viewer_settings();
    crate::core::tonemapper::TonemapConfig {
        enabled: s.tonemap_enabled,
        auto_exposure: s.auto_exposure_enabled,
        operator: match s.tonemap_operator {
            0 => crate::core::tonemapper::TonemapOperator::None,
            1 => crate::core::tonemapper::TonemapOperator::FalseColor,
            2 => crate::core::tonemapper::TonemapOperator::AcesFilmic,
            3 => crate::core::tonemapper::TonemapOperator::Aces2Fit,
            4 => crate::core::tonemapper::TonemapOperator::PbrNeutral,
            5 => crate::core::tonemapper::TonemapOperator::ICtCpBt2446c,
            6 => crate::core::tonemapper::TonemapOperator::ICtCpLumina,
            _ => crate::core::tonemapper::TonemapOperator::AcesFilmic,
        },
    }
}

/// Main entry point for the GUI application
pub fn run_gui() -> Result<()> {
    // Run database vector cache garbage collection synchronously
    // at early startup before any database or UI event loop initialization begins.
    // This prevents race conditions or locking conflicts when the scan database is initialized.
    utils::cache::run_vector_cache_garbage_collector();

    if let Ok(dir) = utils::settings::get_portable_app_data_dir() {
        let log_path = dir.join("PixelHand.log");
        if let Ok(file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            let _ = LOG_FILE.set(Mutex::new(Some(file)));
        }
    }

    // Mute DirectML hardware execution logs
    tracing_subscriber::fmt()
        .with_writer(UiLogWriter)
        .with_env_filter("info,ort=warn")
        .with_ansi(false)
        .init();

    let state = Arc::new(Mutex::new(AppState::default()));
    let cancel_token = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let app = AppWindow::new().context("Failed to initialize Slint UI Window")?;
    let _ = APP_HANDLE.set(app.as_weak());

    let store = app.global::<crate::app::Store>(); // Retrieve the global Store handle
    let loaded_settings = utils::settings::load_settings().unwrap_or_default();
    apply_settings_to_ui(&app, &loaded_settings);

    // Generate both dark and light checkerboard patterns
    let checkerboard_dark = utils::ui::generate_checkerboard(false);
    let checkerboard_light = utils::ui::generate_checkerboard(true);
    store.set_checkerboard_pattern(checkerboard_dark);
    store.set_checkerboard_pattern_light(checkerboard_light);

    // Initialize state context variables from AppSettings
    update_viewer_settings(|s| {
        s.tonemap_enabled = loaded_settings.tonemap_enabled;
        s.auto_exposure_enabled = loaded_settings.tonemap_auto_exposure;
        s.tonemap_operator = loaded_settings.tonemap_operator as usize;
        s.play_speed = loaded_settings.play_speed;
        s.enable_frame_blending = loaded_settings.enable_frame_blending;
    });

    // Flush initial startup logs queued before UI loop initialization
    if let Some(queue_mutex) = LOG_MESSAGES.get()
        && let Ok(mut q) = queue_mutex.lock()
    {
        let app_weak_init = app.as_weak();
        let logs_to_flush = std::mem::take(&mut *q);
        if !logs_to_flush.is_empty() {
            let _ = app_weak_init.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                let mut current = store.get_console_log().to_string();
                for line in logs_to_flush {
                    let cleaned = clean_sample_log(line);
                    if current.is_empty() {
                        current = cleaned;
                    } else {
                        current = format!("{}\n{}", current, cleaned);
                    }
                }
                store.set_console_log(current.into());
            });
        }
    }

    // Spawn async background tick to drain buffered log sequences every 200ms
    let app_weak_log = app.as_weak();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            interval.tick().await;

            let pending: Vec<String> = {
                if let Some(queue) = PENDING_LOGS.get() {
                    if let Ok(mut lock) = queue.lock() {
                        if lock.is_empty() {
                            continue;
                        }
                        std::mem::take(&mut *lock)
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };

            if pending.is_empty() {
                continue;
            }

            let app_clone = app_weak_log.clone();
            let _ = app_clone.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                let current_log = store.get_console_log().to_string();
                let mut lines: Vec<&str> = current_log.lines().collect();

                for p in &pending {
                    lines.push(p);
                }

                // Cap the viewport to maximum 200 rows of active console logs to avoid layout lag
                if lines.len() > 200 {
                    let start = lines.len() - 200;
                    lines = lines[start..].to_vec();
                }

                store.set_console_log(lines.join("\n").into());
            });
        }
    });

    // background task 1: Low-frequency (100ms) sync that safely polls Slint properties on the UI thread
    // and stores them into thread-safe context settings, preventing event loop flooding.
    let app_weak_sync = app.as_weak();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let success = app_weak_sync
                .upgrade_in_event_loop(|ui| {
                    let store = ui.global::<Store>();
                    update_viewer_settings(|s| {
                        s.compare_mode = store.get_compare_mode() as usize;
                        s.flicker_interval = store.get_flicker_interval_val() as usize;

                        // Sync manual adjustments to context
                        s.brightness = (store.get_manual_brightness() * 100.0) as usize;
                        s.contrast = (store.get_manual_contrast() * 100.0) as usize;
                        s.gamma = (store.get_manual_gamma() * 100.0) as usize;
                        s.ratio = (store.get_aspect_ratio_modifier() * 100.0) as usize;

                        s.grid_cols = store.get_grid_cols() as usize;
                        s.grid_rows = store.get_grid_rows() as usize;
                        s.active_frame = store.get_active_frame() as usize;
                        s.play_flipbook = store.get_play_flipbook();
                        s.flipbook_fps = store.get_flipbook_fps() as usize;
                        s.play_speed = store.get_play_speed();
                        s.enable_frame_blending = store.get_enable_frame_blending();
                    });
                })
                .is_ok();
            if !success {
                break;
            }
        }
    });

    // background task 2: High-frequency flicker precision loop executing strictly using
    // thread-safe context settings to prevent cross-thread Slint safety violations.
    let app_weak_flicker = app.as_weak();
    tokio::spawn(async move {
        let mut elapsed_ms: u64 = 0;
        let tick_rate_ms: u64 = 10; // Precision tick

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(tick_rate_ms)).await;

            let s = get_viewer_settings();
            if s.compare_mode == 4 {
                elapsed_ms += tick_rate_ms;
                let target_duration = s.flicker_interval as u64;

                if elapsed_ms >= target_duration.max(50) {
                    elapsed_ms = 0; // Reset timer

                    let _ = app_weak_flicker.upgrade_in_event_loop(|ui| {
                        let store = ui.global::<Store>();
                        // Double check comparison mode inside the safe context
                        if store.get_compare_mode() == 4 {
                            store.set_flicker_show_duplicate(!store.get_flicker_show_duplicate());
                        }
                    });
                }
            } else {
                elapsed_ms = 0;
            }
        }
    });

    // High-frequency Flipbook precision loop executing frame progressions
    let app_weak_flipbook = app.as_weak();
    tokio::spawn(async move {
        let mut continuous_time_ms: f64 = 0.0;
        let tick_rate_ms: u64 = 16; // ~60 FPS update cycle (16.67ms)

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(tick_rate_ms)).await;

            let s = get_viewer_settings();
            if s.play_flipbook {
                let cols = s.grid_cols.max(1);
                let rows = s.grid_rows.max(1);
                let total_frames = cols * rows;

                if total_frames > 1 {
                    // Frame duration at base FPS
                    let fps = s.flipbook_fps.max(1);
                    let frame_duration_ms = 1000.0 / fps as f64;

                    // Advance time with Speed Multiplier
                    continuous_time_ms += tick_rate_ms as f64 * s.play_speed as f64;

                    // Current animation position in frames
                    let frame_position = continuous_time_ms / frame_duration_ms;
                    let current_frame = (frame_position.floor() as usize) % total_frames;
                    let blend_factor = frame_position.fract() as f32;

                    let _ = app_weak_flipbook.upgrade_in_event_loop(move |ui| {
                        let store = ui.global::<Store>();
                        if store.get_play_flipbook() {
                            store.set_active_frame(current_frame as i32);
                            if s.enable_frame_blending {
                                store.set_blend_factor(blend_factor);
                            } else {
                                store.set_blend_factor(0.0);
                            }
                            store.invoke_active_frame_changed(); // Trigger viewport updates
                        }
                    });
                }
            } else {
                // Keep continuous time reset or in sync with current active frame
                // Read properties directly from the viewer settings context rather than a main thread closure
                let fps = s.flipbook_fps.max(1);
                let frame_duration_ms = 1000.0 / fps as f64;
                continuous_time_ms = s.active_frame as f64 * frame_duration_ms;
            }
        }
    });

    // Delegate callback hooks registration to handlers module
    crate::handlers::register_callbacks(&app, state, cancel_token);

    app.run()
        .context("Slint event loop terminated with an error")?;
    Ok(())
}

/// Maps serializable application settings structs into Slint component properties.
fn apply_settings_to_ui(app: &AppWindow, settings: &AppSettings) {
    app.set_dir_a(settings.dir_a.clone().into());
    app.set_dir_b(settings.dir_b.clone().into());
    app.set_query_text(settings.query_text.clone().into());
    app.set_similarity_threshold(settings.similarity_threshold);
    app.set_batch_size(settings.batch_size);

    let search_method = if settings.qc_mode {
        4
    } else {
        settings.search_method
    };
    app.set_search_method(search_method);
    app.set_qc_mode(settings.qc_mode);
    app.set_execution_provider(settings.execution_provider);

    app.set_qc_npot(settings.qc_npot);
    app.set_qc_mipmaps(settings.qc_mipmaps);
    app.set_qc_block_align(settings.qc_block_align);
    app.set_qc_bit_depth(settings.qc_bit_depth);
    app.set_qc_solid_colors(settings.qc_solid_colors);
    app.set_qc_normals(settings.qc_normals);
    app.set_qc_normals_tags(settings.qc_normals_tags.clone().into());

    app.set_qc_check_bloat(settings.qc_check_bloat);
    app.set_qc_check_alpha(settings.qc_check_alpha);
    app.set_qc_check_colorspace(settings.qc_check_colorspace);
    app.set_qc_check_compression(settings.qc_check_compression);

    app.set_ext_png(settings.ext_png);
    app.set_ext_jpg(settings.ext_jpg);
    app.set_ext_tga(settings.ext_tga);
    app.set_ext_dds(settings.ext_dds);
    app.set_ext_bmp(settings.ext_bmp);
    app.set_ext_exr(settings.ext_exr);
    app.set_ext_hdr(settings.ext_hdr);
    app.set_ext_tif(settings.ext_tif);
    app.set_ext_webp(settings.ext_webp);
    app.set_ext_gif(settings.ext_gif);
    app.set_ext_psd(settings.ext_psd);
    app.set_ext_jxl(settings.ext_jxl);
    app.set_ext_heic(settings.ext_heic);
    app.set_ext_avif(settings.ext_avif);

    app.set_duplicates_panel_height(settings.duplicates_panel_height);
    app.set_sidebar_width(settings.sidebar_width);
    app.set_compare_sidebar_width(settings.compare_sidebar_width);
    app.set_list_preview_size(settings.list_preview_size);

    app.set_save_visuals(settings.save_visuals);
    app.set_visuals_columns(settings.visuals_columns);
    app.set_visuals_max_count(settings.visuals_max_count);
    app.set_visuals_font_size(settings.visuals_font_size);
    app.set_visuals_scale(settings.visuals_scale);

    app.set_prep_luminance(settings.prep_luminance);
    app.set_prep_channels(settings.prep_channels);
    app.set_prep_r(settings.prep_r);
    app.set_prep_g(settings.prep_g);
    app.set_prep_b(settings.prep_b);
    app.set_prep_a(settings.prep_a);
    app.set_prep_tags(settings.prep_tags.clone().into());
    app.set_prep_ignore_solid(settings.prep_ignore_solid);

    app.set_excluded_folders(settings.excluded_folders.clone().into());
    app.set_qc_match_by_stem(settings.qc_match_by_stem);
    app.set_qc_hide_same_resolution(settings.qc_hide_same_resolution);
    app.set_ai_model(settings.ai_model);
    app.set_search_precision(settings.search_precision);

    app.set_custom_model_path(settings.custom_model_path.clone().into());
    app.set_custom_model_arch(settings.custom_model_arch);
    app.set_custom_model_dim(settings.custom_model_dim);

    app.set_tonemap_enabled(settings.tonemap_enabled);
    app.set_tonemap_auto_exposure(settings.tonemap_auto_exposure);
    app.set_tonemap_operator(settings.tonemap_operator);

    app.set_enable_previews(settings.enable_previews);
    app.set_preview_quality(settings.preview_quality);
    app.set_filter_only_npot(settings.filter_only_npot);
    app.set_filter_only_uncompressed(settings.filter_only_uncompressed);
    app.set_filter_only_missing_mips(settings.filter_only_missing_mips);
    app.set_filter_only_cubemaps(settings.filter_only_cubemaps);

    crate::scanners::ENABLE_PREVIEWS.store(settings.enable_previews, Ordering::Relaxed);
    crate::scanners::PREVIEW_QUALITY.store(settings.preview_quality, Ordering::Relaxed);

    // Apply the newly integrated GameTextureViewer settings mapped to AppWindow properties
    app.set_grid_cols(settings.grid_cols);
    app.set_grid_rows(settings.grid_rows);
    app.set_manual_brightness(settings.manual_brightness);
    app.set_manual_contrast(settings.manual_contrast);
    app.set_manual_gamma(settings.manual_gamma);
    app.set_aspect_ratio_modifier(settings.aspect_ratio_modifier);
    app.set_background_mode(settings.background_mode);
    app.set_flipbook_fps(settings.flipbook_fps);
    app.set_fit_to_window(settings.fit_to_window);

    // Apply the newly integrated Continuous Playback and Blending options
    app.set_play_speed(settings.play_speed);
    app.set_enable_frame_blending(settings.enable_frame_blending);
}

/// Helper function to push decoded graphics buffers to Slint preview targets.
pub fn trigger_viewport_update(
    app_weak: slint::Weak<AppWindow>,
    orig_path: String,
    dup_path: String,
) {
    let ui = match app_weak.upgrade() {
        Some(ui) => ui,
        None => return,
    };
    let store = ui.global::<crate::app::Store>(); // Obtained the global Store handle
    let channel = utils::ui::get_current_active_channel(&store).to_string();
    let compare_mode = store.get_compare_mode();
    let mip_level = store.get_active_mip_level() as u32;
    let app_weak_clone = app_weak.clone();

    // Capture the interactive parameters from the store
    let brightness = store.get_manual_brightness();
    let contrast = store.get_manual_contrast();
    let gamma = store.get_manual_gamma();
    let ratio = store.get_aspect_ratio_modifier();

    let grid_cols = store.get_grid_cols() as u32;
    let grid_rows = store.get_grid_rows() as u32;
    let active_frame = store.get_active_frame() as u32;

    // Decides indices of original and blending frames prior to dispatching tasks
    let total_frames = grid_cols * grid_rows;
    let next_frame = if total_frames > 1 {
        (active_frame + 1) % total_frames
    } else {
        active_frame
    };
    let enable_blending = store.get_enable_frame_blending();

    // Store state before asynchronous execution
    update_viewer_settings(|s| {
        s.tonemap_enabled = store.get_tonemap_enabled();
        s.auto_exposure_enabled = store.get_tonemap_auto_exposure();
        s.tonemap_operator = store.get_tonemap_operator() as usize;
    });

    tokio::spawn(async move {
        // Fetch current frame previews
        let mut raw_orig =
            utils::cache::get_channel_preview_image(&orig_path, &channel, mip_level).await;
        let mut raw_dup =
            utils::cache::get_channel_preview_image(&dup_path, &channel, mip_level).await;

        // Fetch next frame previews for blending
        let mut raw_orig_next = None;
        let mut raw_dup_next = None;

        if enable_blending && total_frames > 1 {
            raw_orig_next =
                utils::cache::get_channel_preview_image(&orig_path, &channel, mip_level).await;
            raw_dup_next =
                utils::cache::get_channel_preview_image(&dup_path, &channel, mip_level).await;
        }

        // Perform Spritesheet slicing for Frame N (Current)
        if grid_cols > 1 || grid_rows > 1 {
            if let Some(ref img) = raw_orig {
                raw_orig = Some(utils::ui::slice_spritesheet_frame(
                    img,
                    grid_cols,
                    grid_rows,
                    active_frame,
                ));
            }
            if let Some(ref img) = raw_dup {
                raw_dup = Some(utils::ui::slice_spritesheet_frame(
                    img,
                    grid_cols,
                    grid_rows,
                    active_frame,
                ));
            }

            // Perform Spritesheet slicing for Frame N+1 (Next)
            if enable_blending {
                if let Some(ref img) = raw_orig_next {
                    raw_orig_next = Some(utils::ui::slice_spritesheet_frame(
                        img, grid_cols, grid_rows, next_frame,
                    ));
                }
                if let Some(ref img) = raw_dup_next {
                    raw_dup_next = Some(utils::ui::slice_spritesheet_frame(
                        img, grid_cols, grid_rows, next_frame,
                    ));
                }
            }
        }

        // Apply manual Brightness, Contrast, and Gamma color corrections in parallel (Current Frame)
        if let Some(ref mut img) = raw_orig {
            crate::core::tonemapper::apply_manual_corrections(img, brightness, contrast, gamma);
        }
        if let Some(ref mut img) = raw_dup {
            crate::core::tonemapper::apply_manual_corrections(img, brightness, contrast, gamma);
        }

        // Apply manual Brightness, Contrast, and Gamma color corrections in parallel (Next Frame)
        if enable_blending {
            if let Some(ref mut img) = raw_orig_next {
                crate::core::tonemapper::apply_manual_corrections(img, brightness, contrast, gamma);
            }
            if let Some(ref mut img) = raw_dup_next {
                crate::core::tonemapper::apply_manual_corrections(img, brightness, contrast, gamma);
            }
        }

        // Apply aspect ratio scaling multipliers (Ratio Slider - Current Frame)
        if let Some(ref img) = raw_orig {
            raw_orig = Some(utils::ui::apply_aspect_ratio(img, ratio));
        }
        if let Some(ref img) = raw_dup {
            raw_dup = Some(utils::ui::apply_aspect_ratio(img, ratio));
        }

        // Apply aspect ratio scaling multipliers (Ratio Slider - Next Frame)
        if enable_blending {
            if let Some(ref img) = raw_orig_next {
                raw_orig_next = Some(utils::ui::apply_aspect_ratio(img, ratio));
            }
            if let Some(ref img) = raw_dup_next {
                raw_dup_next = Some(utils::ui::apply_aspect_ratio(img, ratio));
            }
        }

        // Pre-extract width and height before moving to event loop
        let (sprite_w, sprite_h) = if let Some(ref img) = raw_orig {
            (img.width() as i32, img.height() as i32)
        } else {
            (0, 0)
        };

        // Update the calculated metadata sizes (Sprite Width, Sprite Height, Sprites Count) based on the slice
        let app_weak_meta = app_weak_clone.clone();
        let _ = app_weak_meta.upgrade_in_event_loop(move |ui| {
            let store = ui.global::<crate::app::Store>();
            store.set_sprite_width(sprite_w);
            store.set_sprite_height(sprite_h);
            store.set_sprites_count((grid_cols * grid_rows) as i32);
        });

        let raw_diff = if compare_mode == 3 {
            if let (Some(orig), Some(dup)) = (&raw_orig, &raw_dup) {
                crate::core::tonemapper::calculate_difference_map(orig, dup, true).ok()
            } else {
                None
            }
        } else {
            None
        };

        let hist_img = if let (Some(orig), Some(dup)) = (&raw_orig, &raw_dup) {
            Some(utils::ui::generate_histogram_image(orig, dup))
        } else {
            None
        };

        // Dispatch Current Frames
        if let Some(img) = raw_orig {
            let app_weak_orig = app_weak_clone.clone();
            let _ = app_weak_orig.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                store.set_image_original(utils::ui::convert_to_slint_image(&img));
            });
        }
        if let Some(img) = raw_dup {
            let app_weak_dup = app_weak_clone.clone();
            let _ = app_weak_dup.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                store.set_image_duplicate(utils::ui::convert_to_slint_image(&img));
            });
        }

        // Dispatch Next Frames for GPU-based alphablending transitions
        if enable_blending {
            if let Some(img) = raw_orig_next {
                let app_weak_orig_next = app_weak_clone.clone();
                let _ = app_weak_orig_next.upgrade_in_event_loop(move |ui| {
                    let store = ui.global::<crate::app::Store>();
                    store.set_image_original_next(utils::ui::convert_to_slint_image(&img));
                });
            }
            if let Some(img) = raw_dup_next {
                let app_weak_dup_next = app_weak_clone.clone();
                let _ = app_weak_dup_next.upgrade_in_event_loop(move |ui| {
                    let store = ui.global::<crate::app::Store>();
                    store.set_image_duplicate_next(utils::ui::convert_to_slint_image(&img));
                });
            }
        }

        if let Some(diff) = raw_diff {
            let app_weak_diff = app_weak_clone.clone();
            let _ = app_weak_diff.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                store.set_image_heatmap(utils::ui::convert_to_slint_image(&diff));
            });
        }

        if let Some(hist) = hist_img {
            let app_weak_hist = app_weak_clone.clone();
            let _ = app_weak_hist.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<crate::app::Store>();
                store.set_histogram_image(utils::ui::convert_to_slint_image(&hist));
            });
        }
    });
}

// Thread-safe fast intermediate queue implementation for incoming console log pipelines
static PENDING_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

// Thread-safe active log file handle mapping
pub static LOG_FILE: OnceLock<Mutex<Option<fs::File>>> = OnceLock::new();
