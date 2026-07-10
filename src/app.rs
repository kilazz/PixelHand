// src/app.rs

use anyhow::{Context, Result};
use slint::ComponentHandle;
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

// Import state store settings
use crate::state::{AppSettings, AppState};
use crate::utils;

// Generate Slint Rust code from the UI markup
slint::include_modules!();

// Global handle to push logs directly to the UI console from any thread
pub static APP_HANDLE: OnceLock<slint::Weak<AppWindow>> = OnceLock::new();

// Fast intermediate thread-safe queue to bypass immediate UI locked calls
static LOG_MESSAGES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

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

fn clean_sample_log(raw: String) -> String {
    raw.replace("\u{001b}[2m", "")
        .replace("\u{001b}[0m", "")
        .replace("\u{001b}[32m", "")
        .replace("\u{001b}[33m", "")
        .replace("\u{001b}[31m", "")
}

fn weak_upgrade_and_queue(weak: slint::Weak<AppWindow>, log_line: String) {
    let _ = weak.upgrade_in_event_loop(move |ui| {
        let current = ui.get_console_log().to_string();
        let next = if current.is_empty() {
            log_line
        } else {
            format!("{}\n{}", current, log_line)
        };
        ui.set_console_log(next.into());
    });
}

pub fn append_to_console_log(msg: &str) {
    let clean_msg = msg.trim_end().to_string();
    if clean_msg.is_empty() {
        return;
    }

    if let Some(weak_handle) = APP_HANDLE.get()
        && let Some(_ui) = weak_handle.upgrade()
    {
        weak_upgrade_and_queue(weak_handle.clone(), clean_sample_log(clean_msg));
        return;
    }

    let queue_mutex = LOG_MESSAGES.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut q) = queue_mutex.lock() {
        q.push(clean_msg);
    }
}

/// Main entry point for the GUI application
pub fn run_gui() -> Result<()> {
    // --- RUN EMBEDDED CACHE GARBAGE COLLECTOR ONCE ON STARTUP ---
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

    tracing_subscriber::fmt()
        .with_writer(UiLogWriter)
        .with_env_filter("info,ort=warn")
        .with_ansi(false)
        .init();

    let state = Arc::new(Mutex::new(AppState::default()));
    let cancel_token = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let app = AppWindow::new().context("Failed to initialize Slint UI Window")?;
    let _ = APP_HANDLE.set(app.as_weak());

    let loaded_settings = utils::settings::load_settings().unwrap_or_default();
    apply_settings_to_ui(&app, &loaded_settings);

    let checkerboard = utils::ui::generate_checkerboard();
    app.set_checkerboard_pattern(checkerboard);

    crate::core::tonemapper::TONEMAP_ENABLED
        .store(loaded_settings.tonemap_enabled, Ordering::Relaxed);
    crate::core::tonemapper::TONEMAP_OPERATOR
        .store(loaded_settings.tonemap_operator as usize, Ordering::Relaxed);

    if let Some(queue_mutex) = LOG_MESSAGES.get()
        && let Ok(mut q) = queue_mutex.lock()
    {
        let app_weak_init = app.as_weak();
        let logs_to_flush = std::mem::take(&mut *q);
        if !logs_to_flush.is_empty() {
            let _ = app_weak_init.upgrade_in_event_loop(move |ui| {
                let mut current = ui.get_console_log().to_string();
                for line in logs_to_flush {
                    let cleaned = clean_sample_log(line);
                    if current.is_empty() {
                        current = cleaned;
                    } else {
                        current = format!("{}\n{}", current, cleaned);
                    }
                }
                ui.set_console_log(current.into());
            });
        }
    }

    let app_weak_log = app.as_weak();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            interval.tick().await;

            let pending: Vec<String> = {
                let queue = PENDING_LOGS.get_or_init(|| Mutex::new(Vec::new()));
                if let Ok(mut lock) = queue.lock() {
                    if lock.is_empty() {
                        continue;
                    }
                    std::mem::take(&mut *lock)
                } else {
                    Vec::new()
                }
            };

            if pending.is_empty() {
                continue;
            }

            let app_clone = app_weak_log.clone();
            let _ = app_clone.upgrade_in_event_loop(move |ui| {
                let current_log = ui.get_console_log().to_string();
                let mut lines: Vec<&str> = current_log.lines().collect();

                for p in &pending {
                    lines.push(p);
                }

                if lines.len() > 200 {
                    let start = lines.len() - 200;
                    lines = lines[start..].to_vec();
                }

                let new_log = lines.join("\n");
                ui.set_console_log(new_log.into());
            });
        }
    });

    trigger_startup_model_download(app.as_weak());

    // --- DELEGATING ALL EVENT REGISTRATION TO SUB-MODULE ---
    crate::app_bindings::register_callbacks(&app, state, cancel_token);

    app.run()
        .context("Slint event loop terminated with an error")?;
    Ok(())
}

fn apply_settings_to_ui(app: &AppWindow, settings: &AppSettings) {
    app.set_dir_a(settings.dir_a.clone().into());
    app.set_dir_b(settings.dir_b.clone().into());
    app.set_query_text(settings.query_text.clone().into());
    app.set_similarity_threshold(settings.similarity_threshold);
    app.set_batch_size(settings.batch_size);

    // Dynamic state restoration for search method and QC mode
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
    app.set_tonemap_operator(settings.tonemap_operator);

    // --- APPLY NEW PREVIEW & SMART FILTER PROPERTIES ---
    app.set_enable_previews(settings.enable_previews);
    app.set_preview_quality(settings.preview_quality);
    app.set_filter_only_npot(settings.filter_only_npot);
    app.set_filter_only_uncompressed(settings.filter_only_uncompressed);
    app.set_filter_only_missing_mips(settings.filter_only_missing_mips);
    app.set_filter_only_cubemaps(settings.filter_only_cubemaps);

    // Sync global atomics dynamically on startup/config load
    crate::scanners::ENABLE_PREVIEWS.store(settings.enable_previews, Ordering::Relaxed);
    crate::scanners::PREVIEW_QUALITY.store(settings.preview_quality, Ordering::Relaxed);
}

fn trigger_startup_model_download(app_weak: slint::Weak<AppWindow>) {
    let app = app_weak.unwrap();
    let active_model = app.get_ai_model();

    tokio::spawn(async move {
        let app_weak_clone = app_weak.clone();
        match crate::core::downloader::verify_and_download_models(app_weak_clone, active_model)
            .await
        {
            Ok(_) => {
                let _ = app_weak.upgrade_in_event_loop(|ui| {
                    ui.set_status_text("AI models verified. System ready.".into());
                    ui.set_progress(1.0);
                });
            }
            Err(e) => {
                let err_msg = e.to_string();
                tracing::error!("Model download failed: {}", err_msg);
                let err_msg_clone = err_msg.clone();
                let _ = app_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_is_scanning(false);
                    ui.set_status_text(
                        format!("Model verification failed: {}", err_msg_clone).into(),
                    );
                });
            }
        }
    });
}

/// Helper function to push decoded graphics buffers to Slint preview targets.
pub fn trigger_viewport_update(
    app_weak: slint::Weak<AppWindow>,
    orig_path: String,
    dup_path: String,
) {
    let ui = app_weak.unwrap();
    let channel = utils::ui::get_current_active_channel(&ui).to_string();
    let compare_mode = ui.get_compare_mode();
    let app_weak_clone = app_weak.clone();

    crate::core::tonemapper::TONEMAP_ENABLED.store(ui.get_tonemap_enabled(), Ordering::Relaxed);
    crate::core::tonemapper::TONEMAP_OPERATOR
        .store(ui.get_tonemap_operator() as usize, Ordering::Relaxed);

    tokio::spawn(async move {
        if let Some(raw_orig) = utils::cache::get_channel_preview_image(&orig_path, &channel).await
        {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_original(utils::ui::convert_to_slint_image(&raw_orig));
            });
        }
        if let Some(raw_dup) = utils::cache::get_channel_preview_image(&dup_path, &channel).await {
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_duplicate(utils::ui::convert_to_slint_image(&raw_dup));
            });
        }

        if compare_mode == 3
            && let Ok(diff_path) =
                crate::scanners::qc::calculate_diff_map(&orig_path, &dup_path).await
            && let Ok(diff_img) = image::open(&diff_path)
        {
            let raw_diff = diff_img.to_rgba8();
            let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                ui.set_image_heatmap(utils::ui::convert_to_slint_image(&raw_diff));
            });
        }
    });
}

// Thread-safe fast intermediate queue implementation for incoming console log pipelines
static PENDING_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

// Thread-safe active log file handle mapping
pub static LOG_FILE: OnceLock<Mutex<Option<fs::File>>> = OnceLock::new();
