// src/handlers/scan.rs

use slint::ComponentHandle;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::app::{AppWindow, Store};
use crate::scanners;
use crate::state::AppState;
use crate::utils;
use crate::utils::helpers::MutexExt;

/// Binds the core background scanning orchestration loop and downloader verification steps.
pub fn bind_scan_execution(
    app: &AppWindow,
    state: Arc<Mutex<AppState>>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    let app_weak_scan = app.as_weak();
    let state_clone = state.clone();
    let cancel_token_clone = cancel_token.clone();
    let store = app.global::<Store>();

    store.on_run_scan(move || {
        let app_copy = app_weak_scan.clone();
        let state_clone_inner = state_clone.clone();

        let ui = match app_copy.upgrade() {
            Some(ui) => ui,
            None => return,
        };
        let store = ui.global::<Store>();

        utils::settings::save_settings(&store);
        cancel_token_clone.store(false, Ordering::Relaxed);

        let mut params = scanners::ScanParams::from_store(&store, cancel_token_clone.clone());

        let app_weak_progress = app_copy.clone();
        params.on_progress = Some(Arc::new(move |prog, current, total| {
            let _ = app_weak_progress.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                store.set_progress(prog);
                store.set_status_text(
                    format!("Processing assets ({} / {})...", current, total).into(),
                );
            });
        }));

        store.set_is_scanning(true);
        store.set_status_text("Scanning assets...".into());
        store.set_progress(0.0);
        tracing::info!("Starting scan on directory: {}", params.dir_a);

        let params_for_task = params.clone();
        let app_weak_download = app_copy.clone();

        tokio::spawn(async move {
            // Verify models on huggingface if AI search is selected
            if params_for_task.search_method == 2
                && let Err(e) = crate::core::downloader::verify_and_download_models(
                    app_weak_download.clone(),
                    params_for_task.ai_model,
                    params_for_task.cancel_token.clone(),
                )
                .await
            {
                let _ = app_weak_download.upgrade_in_event_loop(move |ui| {
                    let store = ui.global::<crate::app::Store>();
                    store.set_is_scanning(false);
                    store.set_status_text(format!("AI Model download failed: {}", e).into());
                });
                return;
            }

            // Sync GUI tonemapping state to thread-safe app static registers prior to execution
            if let Some(ref_ui) = app_weak_download.upgrade() {
                let store = ref_ui.global::<Store>();
                crate::app::TONEMAP_ENABLED.store(store.get_tonemap_enabled(), Ordering::Relaxed);
                crate::app::AUTO_EXPOSURE_ENABLED
                    .store(store.get_tonemap_auto_exposure(), Ordering::Relaxed);
                crate::app::TONEMAP_OPERATOR
                    .store(store.get_tonemap_operator() as usize, Ordering::Relaxed);
            }

            let params_for_task_clone = params_for_task.clone();

            // Clean asynchronous awaiting without thread pool deadlocking risk
            let scan_result = crate::scanners::execute_scan(params_for_task_clone)
                .await
                .map_err(|e| anyhow::anyhow!("Background scanning failed: {}", e));

            // Generate optional contact sheets on duplicate clusters
            if params_for_task.save_visuals
                && let Ok((ref groups, _)) = scan_result
            {
                let _ = app_copy.upgrade_in_event_loop(|ui| {
                    let store = ui.global::<Store>();
                    store.set_status_text("Generating contact sheet reports...".into());
                });

                if let Ok(app_dir) = crate::utils::settings::get_portable_app_data_dir() {
                    let out_dir = app_dir.join("duplicate_visuals");
                    if let Err(e) = crate::core::visuals::generate_visual_reports(
                        groups.clone(),
                        params_for_task.visuals_columns,
                        params_for_task.visuals_max_count,
                        params_for_task.visuals_font_size,
                        params_for_task.visuals_scale,
                        out_dir,
                    )
                    .await
                    {
                        tracing::error!("Failed to generate visual reports: {}", e);
                    }
                }
            }

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                store.set_is_scanning(false);
                store.set_progress(1.0);
                match scan_result {
                    Ok((groups, rows)) => {
                        let mut state_lock = state_clone_inner.safe_lock();
                        state_lock.collapsed_groups.clear();
                        state_lock.groups = groups;
                        state_lock.results = rows;

                        utils::ui::update_results_ui(&store, &state_lock);

                        let msg = if params_for_task.save_visuals {
                            "Scan and visual reports finished successfully!"
                        } else {
                            "Scan finished successfully!"
                        };
                        store.set_status_text(msg.into());
                        tracing::info!("Scan completed.");
                    }
                    Err(e) => {
                        store.set_status_text(format!("Scan stopped: {}", e).into());
                        tracing::warn!("Scan interrupted: {}", e);
                    }
                }
            });
        });
    });

    let cancel_token_cancel = cancel_token.clone();
    let store = app.global::<Store>();
    store.on_cancel_scan(move || {
        cancel_token_cancel.store(true, Ordering::Relaxed);
        tracing::warn!("User requested scan cancellation. Waiting for threads to stop...");
    });
}
