// src/handlers/ui_state.rs

use slint::ComponentHandle;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::app::{AppWindow, Store};
use crate::state::AppState;
use crate::utils;
use crate::utils::helpers::MutexExt;

/// Binds UI utility parameters, tonemapping options, sorting/filtering engines, and column resizers.
pub fn bind_ui_state_and_settings(app: &AppWindow, state: Arc<Mutex<AppState>>) {
    let app_weak_comp = app.as_weak();
    let store = app.global::<Store>();

    store.on_compare_mode_changed(move || {
        let app_copy = app_weak_comp.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_comp.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_tonemap = app.as_weak();
    let store = app.global::<Store>();
    store.on_tonemap_toggled(move || {
        let app_copy = app_weak_tonemap.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();

            utils::settings::save_settings(&store);

            crate::app::TONEMAP_ENABLED.store(store.get_tonemap_enabled(), Ordering::Relaxed);
            crate::app::AUTO_EXPOSURE_ENABLED
                .store(store.get_tonemap_auto_exposure(), Ordering::Relaxed);
            crate::app::TONEMAP_OPERATOR
                .store(store.get_tonemap_operator() as usize, Ordering::Relaxed);

            if let Some(cache) = utils::cache::DECODED_CACHE.get() {
                cache.invalidate_all();
            }

            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_tonemap.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_channel = app.as_weak();
    let store = app.global::<Store>();
    store.on_channel_toggled(move || {
        let app_copy = app_weak_channel.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_channel.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_mip = app.as_weak();
    let store = app.global::<Store>();
    store.on_mip_level_changed(move || {
        let app_copy = app_weak_mip.clone();
        if let Some(ui) = app_copy.upgrade() {
            let store = ui.global::<Store>();
            let orig_path = store.get_original_meta().path.to_string();
            let dup_path = store.get_duplicate_meta().path.to_string();

            if !orig_path.is_empty() && !dup_path.is_empty() {
                crate::app::trigger_viewport_update(app_weak_mip.clone(), orig_path, dup_path);
            }
        }
    });

    let app_weak_rule = app.as_weak();
    let state_clone_rule = state.clone();
    let store = app.global::<Store>();
    store.on_trigger_selection_rule(move |rule| {
        let mut lock = state_clone_rule.safe_lock();
        utils::ui::apply_selection_rule(&mut lock, rule.as_str());
        if let Some(ui) = app_weak_rule.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_expand = app.as_weak();
    let state_clone_exp = state.clone();
    let store = app.global::<Store>();
    store.on_expand_all_groups(move || {
        let mut lock = state_clone_exp.safe_lock();
        lock.collapsed_groups.clear();
        if let Some(ui) = app_weak_expand.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_collapse = app.as_weak();
    let state_clone_col = state.clone();
    let store = app.global::<Store>();
    store.on_collapse_all_groups(move || {
        let mut lock = state_clone_col.safe_lock();
        lock.collapsed_groups.clear();

        let header_indices: Vec<i32> = lock
            .results
            .iter()
            .filter(|row| row.is_header)
            .map(|row| row.group_index)
            .collect();

        for group_index in header_indices {
            lock.collapsed_groups.insert(group_index);
        }

        if let Some(ui) = app_weak_collapse.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let store = app.global::<Store>();
    store.on_export_log(move |log_text| {
        utils::export::export_diagnostics_log(log_text.as_str());
    });

    let state_clone_csv = state.clone();
    let store = app.global::<Store>();
    store.on_export_csv(move || {
        utils::export::export_results_to_csv(state_clone_csv.clone());
    });

    let app_weak_col_sort = app.as_weak();
    let state_clone_col_sort = state.clone();
    let store = app.global::<Store>();
    store.on_sort_by_column(move |col| {
        let mut lock = state_clone_col_sort.safe_lock();
        let col_str = col.to_string();

        if lock.sort_column == col_str {
            lock.sort_ascending = !lock.sort_ascending;
        } else {
            lock.sort_column = col_str.clone();
            lock.sort_ascending = true;
        }

        let asc = lock.sort_ascending;

        if !lock.groups.is_empty() {
            // 1. Sort duplicates INSIDE each group (keeping index 0 intact)
            for group in &mut lock.groups {
                if group.files.len() > 1 {
                    let (_, duplicates) = group.files.split_at_mut(1);
                    match col_str.as_str() {
                        "name" => duplicates.sort_by(|a, b| {
                            if asc {
                                a.path.to_lowercase().cmp(&b.path.to_lowercase())
                            } else {
                                b.path.to_lowercase().cmp(&a.path.to_lowercase())
                            }
                        }),
                        "size" => duplicates.sort_by(|a, b| {
                            if asc {
                                a.size.cmp(&b.size)
                            } else {
                                b.size.cmp(&a.size)
                            }
                        }),
                        "score" => duplicates.sort_by(|a, b| {
                            let res = a
                                .similarity
                                .partial_cmp(&b.similarity)
                                .unwrap_or(std::cmp::Ordering::Equal);
                            if asc { res } else { res.reverse() }
                        }),
                        "path" => duplicates.sort_by(|a, b| {
                            if asc {
                                a.path.to_lowercase().cmp(&b.path.to_lowercase())
                            } else {
                                b.path.to_lowercase().cmp(&a.path.to_lowercase())
                            }
                        }),
                        _ => {}
                    }
                }
            }

            // 2. Sort the groups themselves
            match col_str.as_str() {
                "name" => {
                    lock.groups.sort_by(|a, b| {
                        let n_a = a
                            .files
                            .first()
                            .map(|f| f.path.to_lowercase())
                            .unwrap_or_default();
                        let n_b = b
                            .files
                            .first()
                            .map(|f| f.path.to_lowercase())
                            .unwrap_or_default();
                        let res = n_a.cmp(&n_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                "size" => {
                    lock.groups.sort_by(|a, b| {
                        let s_a = a.files.first().map(|f| f.size).unwrap_or(0);
                        let s_b = b.files.first().map(|f| f.size).unwrap_or(0);
                        let res = s_a.cmp(&s_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                "score" => {
                    lock.groups.sort_by(|a, b| {
                        let sim_a = a.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                        let sim_b = b.files.iter().map(|f| f.similarity).fold(0.0, f32::max);
                        let res = sim_a
                            .partial_cmp(&sim_b)
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if asc { res } else { res.reverse() }
                    });
                }
                "path" => {
                    lock.groups.sort_by(|a, b| {
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
                        let res = p_a.cmp(&p_b);
                        if asc { res } else { res.reverse() }
                    });
                }
                _ => {}
            }
            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
        } else if !lock.results.is_empty() {
            // Flat list sorting
            match col_str.as_str() {
                "name" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.name.to_lowercase().cmp(&b.name.to_lowercase())
                    } else {
                        b.name.to_lowercase().cmp(&a.name.to_lowercase())
                    }
                }),
                "format" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.format_str
                            .to_lowercase()
                            .cmp(&b.format_str.to_lowercase())
                    } else {
                        b.format_str
                            .to_lowercase()
                            .cmp(&a.format_str.to_lowercase())
                    }
                }),
                "dimensions" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.pixels_count.cmp(&b.pixels_count)
                    } else {
                        b.pixels_count.cmp(&a.pixels_count)
                    }
                }),
                "mipmaps" => lock.results.sort_by(|a, b| {
                    let m_a = a.mipmaps_str.parse::<u32>().unwrap_or(0);
                    let m_b = b.mipmaps_str.parse::<u32>().unwrap_or(0);
                    if asc { m_a.cmp(&m_b) } else { m_b.cmp(&m_a) }
                }),
                "cubemap" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.cubemap_str.cmp(&b.cubemap_str)
                    } else {
                        b.cubemap_str.cmp(&a.cubemap_str)
                    }
                }),
                "size" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.size_bytes.cmp(&b.size_bytes)
                    } else {
                        b.size_bytes.cmp(&a.size_bytes)
                    }
                }),
                "path" => lock.results.sort_by(|a, b| {
                    if asc {
                        a.path.to_lowercase().cmp(&b.path.to_lowercase())
                    } else {
                        b.path.to_lowercase().cmp(&a.path.to_lowercase())
                    }
                }),
                "score" => lock.results.sort_by(|a, b| {
                    let res = a
                        .similarity
                        .partial_cmp(&b.similarity)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if asc { res } else { res.reverse() }
                }),
                _ => {}
            }
        }

        if let Some(ui) = app_weak_col_sort.upgrade() {
            let store = ui.global::<Store>();
            store.set_active_sort_column(lock.sort_column.clone().into());
            store.set_sort_ascending(lock.sort_ascending);
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let store = app.global::<Store>();
    store.on_clear_models(move || {
        if let Ok(app_dir) = utils::settings::get_portable_app_data_dir() {
            let _ = std::fs::remove_dir_all(app_dir.join("models"));
            crate::app::append_to_console_log(
                "Downloaded AI model weights successfully cleared from disk.",
            );
        }
    });

    let store = app.global::<Store>();
    store.on_clear_cache(move || {
        // 1. Clear in-memory caches immediately using Moka API
        if let Some(cache) = crate::utils::cache::DECODED_CACHE.get() {
            cache.invalidate_all();
        }

        if let Some(cache) = crate::scanners::THUMBNAIL_MEMORY_CACHE.get() {
            cache.invalidate_all();
        }

        // 2. Safely delete database files in a background thread
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

                if success {
                    crate::app::append_to_console_log(
                        "Scan database and thumbnail caches cleared successfully.",
                    );
                } else {
                    crate::app::append_to_console_log(
                        "Warning: Some cache files could not be cleared (files might be in use).",
                    );
                }
            }
        });
    });

    let app_weak_filter = app.as_weak();
    let state_clone_filt = state.clone();
    let store = app.global::<Store>();
    store.on_results_filter_changed(move || {
        if let Some(ui) = app_weak_filter.upgrade() {
            let lock = state_clone_filt.safe_lock();
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_sort = app.as_weak();
    let state_clone_sort = state.clone();
    let store = app.global::<Store>();
    store.on_results_sort_changed(move |sort_idx| {
        let mut lock = state_clone_sort.safe_lock();
        if !lock.groups.is_empty() {
            match sort_idx {
                0 => lock
                    .groups
                    .sort_by_key(|g| std::cmp::Reverse(g.files.len())),
                1 => lock.groups.sort_by_key(|g| {
                    std::cmp::Reverse(g.files.iter().map(|f| f.size).sum::<u64>())
                }),
                2 => lock.groups.sort_by(|a, b| {
                    let name_a = a.files.first().map(|f| f.path.as_str()).unwrap_or("");
                    let name_b = b.files.first().map(|f| f.path.as_str()).unwrap_or("");
                    name_a.cmp(name_b)
                }),
                _ => {}
            }
            lock.results = crate::scanners::map_groups_to_rows(&lock.groups);
        } else if !lock.results.is_empty() {
            match sort_idx {
                0 => lock.results.sort_by(|a, b| a.format_str.cmp(&b.format_str)),
                1 => lock
                    .results
                    .sort_by_key(|r| std::cmp::Reverse(r.size_bytes)),
                2 => lock.results.sort_by(|a, b| a.name.cmp(&b.name)),
                _ => {}
            }
        }
        if let Some(ui) = app_weak_sort.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_auto = app.as_weak();
    let state_auto = state.clone();
    let store = app.global::<Store>();
    store.on_auto_size_columns(move || {
        if let Some(ui) = app_weak_auto.upgrade() {
            let lock = state_auto.safe_lock();
            if lock.results.is_empty() {
                return;
            }

            let mut max_file_len = 4;
            let mut max_score_len = 5;
            let mut max_path_len = 4;

            for row in &lock.results {
                if !row.is_header {
                    max_file_len = max_file_len.max(row.name.chars().count());
                    max_score_len = max_score_len.max(row.score_or_detail.chars().count());
                    max_path_len = max_path_len.max(row.path.chars().count());
                }
            }

            let file_w = (max_file_len as f32 * 7.2) + 68.0;
            let score_w = (max_score_len as f32 * 7.2) + 20.0;
            let path_w = (max_path_len as f32 * 6.5) + 20.0;

            let store = ui.global::<Store>();
            store.set_col_file_w(file_w.clamp(120.0, 600.0));
            store.set_col_score_w(score_w.clamp(60.0, 150.0));
            store.set_col_path_w(path_w.clamp(150.0, 800.0));

            tracing::info!(
                "Columns auto-resized. FILE: {:.0}px, SCORE: {:.0}px, PATH: {:.0}px",
                file_w,
                score_w,
                path_w
            );
        }
    });

    let app_weak_reset = app.as_weak();
    let store = app.global::<Store>();
    store.on_reset_size_columns(move || {
        if let Some(ui) = app_weak_reset.upgrade() {
            let store = ui.global::<Store>();
            store.set_col_file_w(320.0);
            store.set_col_score_w(80.0);
            store.set_col_path_w(380.0);
            store.set_col_format_w(110.0);
            store.set_col_dimensions_w(110.0);
            store.set_col_mipmaps_w(75.0);
            store.set_col_cubemap_w(75.0);
            store.set_col_size_w(85.0);
            tracing::info!("Column sizes reset to defaults.");
        }
    });

    // Subscribes to layout changes to rebuild grid UI chunkings gracefully dynamically
    let app_weak_grid = app.as_weak();
    let state_grid = state.clone();
    let store = app.global::<Store>();
    store.on_grid_columns_changed(move || {
        if let Some(ui) = app_weak_grid.upgrade() {
            let lock = state_grid.safe_lock();
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });
}
