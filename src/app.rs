// src/app.rs

use anyhow::{Context, Result};
use slint::ComponentHandle;
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};

use crate::handlers::controller::AppController;
use crate::state::{AppSettings, AppState};

// Generate Slint Rust code from the UI markup
slint::include_modules!();

// Re-export the extracted viewport pipeline for perfect backward compatibility
pub use crate::utils::viewport::trigger_viewport_update;

// ==========================================
// --- CENTRAL APPLICATION CONTEXT ----------
// ==========================================

/// Consolidated context containing all global dynamic state and logging buffers
pub struct AppContext {
    pub app_handle: OnceLock<slint::Weak<AppWindow>>,
    pub viewer_settings: parking_lot::Mutex<ViewerSettingsContext>,
    pub log_file: parking_lot::Mutex<Option<fs::File>>,
    pub pending_logs: parking_lot::Mutex<Vec<String>>,
    pub startup_logs: parking_lot::Mutex<Vec<String>>,
}

/// Retrieve the global thread-safe Application Context singleton
pub fn get_ctx() -> &'static AppContext {
    static INSTANCE: OnceLock<AppContext> = OnceLock::new();
    INSTANCE.get_or_init(|| AppContext {
        app_handle: OnceLock::new(),
        viewer_settings: parking_lot::Mutex::new(ViewerSettingsContext::default()),
        log_file: parking_lot::Mutex::new(None),
        pending_logs: parking_lot::Mutex::new(Vec::new()),
        startup_logs: parking_lot::Mutex::new(Vec::new()),
    })
}

// Thread-safe settings context to bridge Slint states to background loops safely
#[derive(Debug, Clone, Copy)]
pub struct ViewerSettingsContext {
    pub tonemap_enabled: bool,
    pub auto_exposure_enabled: bool,
    pub tonemap_operator: usize,
}

impl Default for ViewerSettingsContext {
    fn default() -> Self {
        Self {
            tonemap_enabled: true,
            auto_exposure_enabled: true,
            tonemap_operator: 2,
        }
    }
}

pub fn get_viewer_settings() -> ViewerSettingsContext {
    *get_ctx().viewer_settings.lock()
}

pub fn update_viewer_settings<F>(f: F)
where
    F: FnOnce(&mut ViewerSettingsContext),
{
    f(&mut get_ctx().viewer_settings.lock());
}

/// Custom tracing subscriber log handler wrapping native console pipes
struct UiLogWriter;

impl std::io::Write for UiLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(msg) = String::from_utf8(buf.to_vec()) {
            append_to_console_log(&msg);
        }

        let ctx = get_ctx();
        let mut lock = ctx.log_file.lock();
        if let Some(ref mut file) = *lock {
            let _ = file.write_all(buf);
            let _ = file.flush();
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let ctx = get_ctx();
        let mut lock = ctx.log_file.lock();
        if let Some(ref mut file) = *lock {
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
    let ctx = get_ctx();

    // If Slint UI window is loaded, queue into pending_logs, otherwise cache in startup_logs
    if ctx.app_handle.get().is_some() {
        ctx.pending_logs.lock().push(cleaned);
    } else {
        ctx.startup_logs.lock().push(cleaned);
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
    crate::utils::cache::run_vector_cache_garbage_collector();

    let ctx = get_ctx();

    if let Ok(dir) = crate::utils::settings::get_portable_app_data_dir() {
        let log_path = dir.join("PixelHand.log");
        if let Ok(file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            *ctx.log_file.lock() = Some(file);
        }
    }

    // Mute DirectML hardware execution logs
    tracing_subscriber::fmt()
        .with_writer(UiLogWriter)
        .with_env_filter("info,ort=warn")
        .with_ansi(false)
        .init();

    let state = Arc::new(parking_lot::Mutex::new(AppState::default()));
    let cancel_token = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let app = AppWindow::new().context("Failed to initialize Slint UI Window")?;
    let _ = ctx.app_handle.set(app.as_weak());

    let loaded_settings = crate::utils::settings::load_settings().unwrap_or_default();
    apply_settings_to_ui(&app, &loaded_settings);

    let checkerboard_dark = crate::utils::ui::generate_checkerboard(false);
    let checkerboard_light = crate::utils::ui::generate_checkerboard(true);

    // Assign patterns to the modular ViewportState singleton
    let viewport_state = app.global::<ViewportState>();
    viewport_state.set_checkerboard_pattern(checkerboard_dark);
    viewport_state.set_checkerboard_pattern_light(checkerboard_light);

    // Initialize state context variables from AppSettings
    update_viewer_settings(|s| {
        s.tonemap_enabled = loaded_settings.tonemap.tonemap_enabled;
        s.auto_exposure_enabled = loaded_settings.tonemap.tonemap_auto_exposure;
        s.tonemap_operator = loaded_settings.tonemap.tonemap_operator as usize;
    });

    crate::scanners::ENABLE_PREVIEWS
        .store(loaded_settings.preview.enable_previews, Ordering::Relaxed);
    crate::scanners::PREVIEW_QUALITY
        .store(loaded_settings.preview.preview_quality, Ordering::Relaxed);

    // Flush initial startup logs queued before UI loop initialization
    let mut startup_q = ctx.startup_logs.lock();
    let logs_to_flush = std::mem::take(&mut *startup_q);
    if !logs_to_flush.is_empty() {
        let app_weak_init = app.as_weak();
        let _ = app_weak_init.upgrade_in_event_loop(move |ui| {
            let diag = ui.global::<Diagnostics>();
            let mut current = diag.get_console_log().to_string();
            for line in logs_to_flush {
                let cleaned = clean_sample_log(line);
                if current.is_empty() {
                    current = cleaned;
                } else {
                    current = format!("{}\n{}", current, cleaned);
                }
            }
            diag.set_console_log(current.into());
        });
    }
    drop(startup_q); // Release the startup logs lock early

    // Spawn async background task to drain buffered log sequences every 200ms
    let app_weak_log = app.as_weak();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            interval.tick().await;

            let pending: Vec<String> = {
                let ctx = get_ctx();
                let mut lock = ctx.pending_logs.lock();
                if lock.is_empty() {
                    continue;
                }
                std::mem::take(&mut *lock)
            };

            if pending.is_empty() {
                continue;
            }

            let app_clone = app_weak_log.clone();
            let _ = app_clone.upgrade_in_event_loop(move |ui| {
                let diag = ui.global::<Diagnostics>();
                let current_log = diag.get_console_log().to_string();
                let mut lines: Vec<&str> = current_log.lines().collect();

                for p in &pending {
                    lines.push(p);
                }

                // Cap the viewport to maximum 200 rows of active console logs to avoid layout lag
                if lines.len() > 200 {
                    let start = lines.len() - 200;
                    lines = lines[start..].to_vec();
                }

                diag.set_console_log(lines.join("\n").into());
            });
        }
    });

    // 1. High-precision native Slint Timer for the Flicker Compare Mode
    // Executing entirely on the UI thread to remove cross-thread event loop queue overhead
    let flicker_timer = slint::Timer::default();
    let app_weak_flicker = app.as_weak();
    let mut flicker_elapsed_ms = 0u64;

    flicker_timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(10), // Ticker rate of 10ms
        move || {
            let ui = match app_weak_flicker.upgrade() {
                Some(ui) => ui,
                None => return,
            };
            let vp = ui.global::<ViewportState>();

            if vp.get_compare_mode() == 4 {
                flicker_elapsed_ms += 10;
                let target_duration = vp.get_flicker_interval_val() as u64;

                if flicker_elapsed_ms >= target_duration.max(50) {
                    flicker_elapsed_ms = 0; // Reset ticker accumulation
                    vp.set_flicker_show_duplicate(!vp.get_flicker_show_duplicate());
                }
            } else {
                flicker_elapsed_ms = 0;
            }
        },
    );

    // 2. High-precision native Slint Timer for the Flipbook Animation
    // Executing frame progressions at ~60 FPS synchronously on the UI thread
    let flipbook_timer = slint::Timer::default();
    let app_weak_flipbook = app.as_weak();
    let mut continuous_time_ms = 0.0f64;

    flipbook_timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16), // ~16.67ms ticker loop (60 FPS)
        move || {
            let ui = match app_weak_flipbook.upgrade() {
                Some(ui) => ui,
                None => return,
            };
            let vp = ui.global::<ViewportState>();
            let viewer = vp.get_viewer();

            if vp.get_play_flipbook() {
                let cols = viewer.grid_cols.max(1) as u32;
                let rows = viewer.grid_rows.max(1) as u32;
                let total_frames = cols * rows;

                if total_frames > 1 {
                    let fps = viewer.flipbook_fps.max(1.0) as f64;
                    let frame_duration_ms = 1000.0 / fps;

                    // Increment tracking timer with the speed multiplier
                    continuous_time_ms += 16.0 * viewer.play_speed as f64;

                    let frame_position = continuous_time_ms / frame_duration_ms;
                    let current_frame = (frame_position.floor() as u32) % total_frames;
                    let blend_factor = frame_position.fract() as f32;

                    vp.set_active_frame(current_frame as i32);
                    if viewer.enable_frame_blending {
                        vp.set_blend_factor(blend_factor);
                    } else {
                        vp.set_blend_factor(0.0);
                    }
                    vp.invoke_active_frame_changed(); // Trigger viewport redraw updates
                }
            } else {
                let fps = viewer.flipbook_fps.max(1.0) as f64;
                let frame_duration_ms = 1000.0 / fps;
                continuous_time_ms = vp.get_active_frame() as f64 * frame_duration_ms;
            }
        },
    );

    // Initialize our Controller pattern and wire all handlers
    let controller = AppController::new(&app, state, cancel_token);
    controller.register_callbacks();

    app.run()
        .context("Slint event loop terminated with an error")?;
    Ok(())
}

fn apply_settings_to_ui(app: &AppWindow, settings: &AppSettings) {
    let scan_config = app.global::<ScanConfig>();
    let viewport_state = app.global::<ViewportState>();

    // Apply grouped structures natively (zero conversions)
    scan_config.set_paths(settings.paths.clone());
    scan_config.set_extensions(settings.extensions.clone());
    scan_config.set_ui(settings.ui.clone());
    scan_config.set_visuals(settings.visuals.clone());
    scan_config.set_prep(settings.prep.clone());
    scan_config.set_qc(settings.qc.clone());
    scan_config.set_ai(settings.ai.clone());
    scan_config.set_preview(settings.preview.clone());

    viewport_state.set_tonemap(settings.tonemap.clone());
    viewport_state.set_viewer(settings.viewer.clone());

    // Apply global scalar settings
    scan_config.set_similarity_threshold(settings.similarity_threshold);
    scan_config.set_batch_size(settings.batch_size);
    scan_config.set_search_method(settings.search_method);
    scan_config.set_execution_provider(settings.execution_provider);
    scan_config.set_search_precision(settings.search_precision);

    crate::scanners::ENABLE_PREVIEWS.store(settings.preview.enable_previews, Ordering::Relaxed);
    crate::scanners::PREVIEW_QUALITY.store(settings.preview.preview_quality, Ordering::Relaxed);
}
