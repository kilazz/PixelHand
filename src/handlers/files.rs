// src/handlers/files.rs

use slint::ComponentHandle;
use slint::winit_030::WinitWindowAccessor;
use slint::{ModelRc, VecModel};
use std::sync::{Arc, Mutex};

use crate::app::{AppWindow, SelectedFile, Store};
use crate::scanners;
use crate::state::AppState;
use crate::utils;
use crate::utils::helpers::MutexExt;

/// Binds UI handlers responsible for folder, reference image, and custom model selection.
pub fn bind_directory_selection(app: &AppWindow) {
    let app_weak_a = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_folder_a(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder A")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_a.upgrade() {
                let store = ui.global::<Store>();
                store.set_dir_a(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_b = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_folder_b(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder B")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_b.upgrade() {
                let store = ui.global::<Store>();
                store.set_dir_b(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_ref = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_reference_image(move || {
        if let Some(file) = rfd::FileDialog::new()
            .set_title("Select Reference Image")
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "tga", "dds", "exr", "hdr", "tif", "tiff", "webp", "gif",
                    "psd", "jxl", "heic", "heif", "avif",
                ],
            )
            .pick_file()
        {
            let path_str = file.to_string_lossy().to_string();
            if let Some(ui) = app_weak_ref.upgrade() {
                let store = ui.global::<Store>();
                store.set_query_text(path_str.into());
                store.set_search_method(2);
                utils::settings::save_settings(&store);
            }
        }
    });

    let app_weak_custom = app.as_weak();
    let store = app.global::<Store>();

    store.on_select_custom_model(move || {
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Custom ONNX Model Directory")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            if let Some(ui) = app_weak_custom.upgrade() {
                let store = ui.global::<Store>();
                store.set_custom_model_path(path_str.into());
                utils::settings::save_settings(&store);
            }
        }
    });
}

/// Binds UI handlers for files interactions, double clicks, checklist updates, and viewport previews.
pub fn bind_file_actions(app: &AppWindow, state: Arc<Mutex<AppState>>) {
    let app_weak_ctx = app.as_weak();
    let state_clone_ctx = state.clone();
    let store = app.global::<Store>();

    store.on_context_menu_action(move |action, path| {
        let path_str = path.to_string();
        if path_str.is_empty() {
            return;
        }

        match action.as_str() {
            "open" => {
                tracing::info!("Context Menu: Opening file {}", path_str);
                if let Err(e) = open::that(&path_str) {
                    tracing::error!("Failed to open file: {}", e);
                }
            }
            "explore" => {
                tracing::info!("Context Menu: Showing in explorer {}", path_str);
                if let Some(parent_dir) = std::path::Path::new(&path_str).parent()
                    && let Err(e) = open::that(parent_dir)
                {
                    tracing::error!("Failed to open directory: {}", e);
                }
            }
            "trash" => {
                let p = std::path::PathBuf::from(&path_str);
                if p.exists() {
                    match trash::delete(&p) {
                        Ok(_) => {
                            tracing::info!("Moved to trash via context menu: {}", path_str);
                            if let Some(ui) = app_weak_ctx.upgrade() {
                                let store = ui.global::<Store>();
                                store.set_status_text(
                                    format!(
                                        "Moved to trash: {}",
                                        p.file_name().unwrap_or_default().to_string_lossy()
                                    )
                                    .into(),
                                );
                                let mut lock = state_clone_ctx.safe_lock();
                                lock.results.retain(|r| r.path != path_str);
                                utils::ui::update_results_ui(&store, &lock);
                            }
                        }
                        Err(e) => tracing::error!("Failed to move to trash: {}", e),
                    }
                }
            }
            _ => {}
        }
    });

    let store = app.global::<Store>();
    store.on_open_file_in_viewer(move |path| {
        let path_str = path.to_string();
        if !path_str.is_empty() {
            tracing::info!("Opening file in default viewer: {}", path_str);
            if let Err(e) = open::that(&path_str) {
                tracing::error!("Failed to open file: {}", e);
            }
        }
    });

    let app_weak_checkbox = app.as_weak();
    let state_clone_cb = state.clone();
    let store = app.global::<Store>();
    store.on_row_checkbox_toggled(move |idx| {
        let mut lock = state_clone_cb.safe_lock();
        if let Some(abs_idx) = utils::ui::get_absolute_index(&lock, idx as usize)
            && let Some(row) = lock.results.get_mut(abs_idx)
        {
            row.is_checked = !row.is_checked;
        }
        if let Some(ui) = app_weak_checkbox.upgrade() {
            let store = ui.global::<Store>();
            utils::ui::update_results_ui(&store, &lock);
        }
    });

    let app_weak_clicked = app.as_weak();
    let state_clone_click = state.clone();
    let store = app.global::<Store>();
    store.on_row_clicked(move |_idx, is_header, group_idx, path| {
        let app_copy = app_weak_clicked.clone();

        if is_header {
            let mut lock = state_clone_click.safe_lock();
            if lock.collapsed_groups.contains(&group_idx) {
                lock.collapsed_groups.remove(&group_idx);
            } else {
                lock.collapsed_groups.insert(group_idx);
            }
            if let Some(ui) = app_copy.upgrade() {
                let store = ui.global::<Store>();
                utils::ui::update_results_ui(&store, &lock);
            }
            return;
        }

        let path_str = path.to_string();
        let lock = state_clone_click.safe_lock();

        let group = lock.groups.get(group_idx as usize);

        let ui = match app_copy.upgrade() {
            Some(ui) => ui,
            None => return,
        };
        let store = ui.global::<Store>();

        let group = match group {
            None => {
                if let Ok(meta) =
                    crate::core::qc::extract_qc_metadata(std::path::Path::new(&path_str))
                {
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

                    store.set_original_meta(file_meta.clone());
                    store.set_duplicate_meta(file_meta);
                    store.set_max_available_mips(meta.mipmap_count as i32);
                    store.set_active_mip_level(0);
                }

                store.set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(
                    Vec::new(),
                ))));

                crate::app::trigger_viewport_update(
                    app_weak_clicked.clone(),
                    path_str.clone(),
                    path_str.clone(),
                );
                return;
            }
            Some(g) => g,
        };

        let original = match group.files.first() {
            Some(f) => f,
            None => return,
        };
        let duplicate = match group.files.iter().find(|f| f.path == path_str) {
            Some(f) => f,
            None => return,
        };

        store.set_original_meta(utils::ui::build_selected_file_meta(original, true));
        store.set_duplicate_meta(utils::ui::build_selected_file_meta(duplicate, false));
        store.set_max_available_mips(original.mipmap_count as i32);
        store.set_active_mip_level(0);

        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(utils::ui::convert_to_slint_row(row));
            }
        }
        store
            .set_selected_group_files(ModelRc::from(std::rc::Rc::new(VecModel::from(group_files))));
        crate::app::trigger_viewport_update(
            app_weak_clicked.clone(),
            original.path.clone(),
            duplicate.path.clone(),
        );
    });

    let app_weak_action = app.as_weak();
    let state_copy_act = state.clone();
    let store = app.global::<Store>();
    store.on_trigger_action(move |action_type| {
        let app_copy = app_weak_action.clone();
        let state_copy_inner = state_copy_act.clone();
        let action = action_type.to_string();

        tokio::spawn(async move {
            let (checked_files, pairs) = {
                let lock = state_copy_inner.safe_lock();
                utils::fs::extract_selected_files(&lock)
            };

            if checked_files.is_empty() {
                return;
            }

            let _ = app_copy.upgrade_in_event_loop({
                let r#act = action.clone();
                move |ui| {
                    let store = ui.global::<Store>();
                    store.set_status_text(format!("Processing selection: {}...", r#act).into());
                }
            });

            let res = utils::fs::execute_file_action(&action, checked_files, pairs).await;

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                match res {
                    Ok(_) => {
                        store.set_status_text(
                            format!("Successfully completed {} operation.", action).into(),
                        );
                        store.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(
                            Vec::new(),
                        ))));
                        store.set_has_results(false); // Reset persistence flag on clear
                        let mut lock = state_copy_inner.safe_lock();
                        lock.results.clear();
                        lock.groups.clear();
                        lock.collapsed_groups.clear();
                    }
                    Err(e) => {
                        store.set_status_text(format!("Action failed: {}", e).into());
                        tracing::error!("Action failed: {}", e);
                    }
                }
            });
        });
    });

    let app_weak_hover = app.as_weak();
    let state_hover = state.clone();

    // Thread-safe async debouncing handle to capture rapid mouse hovering and eliminate event loop flooding
    let last_hover_task = Arc::new(Mutex::new(None::<tokio::task::JoinHandle<()>>));

    let store = app.global::<Store>();
    store.on_thumbnail_channel_hovered(move |path_str, channel| {
        let path_std = path_str.to_string();
        let channel_std = channel.to_string();
        let app_weak_clone = app_weak_hover.clone();
        let state_clone = state_hover.clone();

        let mut lock = last_hover_task.lock().unwrap();
        if let Some(handle) = lock.take() {
            handle.abort(); // Cancel any rapid intermediate hover tasks to protect the Slint event loop
        }

        // Spawn debounced hover processing task with an 80ms delay
        let handle = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;

            let normalized_path = scanners::normalize_path_key(&path_std);

            // Fetch the cached item directly from Moka sync cache (no explicit locking needed)
            let cached_img = {
                let cache = scanners::THUMBNAIL_MEMORY_CACHE.get();
                cache.and_then(|c| c.get(&normalized_path))
            };

            if let Some(cached_thumb) = cached_img {
                // Highly optimized: fetch pre-computed (or lazily initialized once) channel-isolated buffer
                let channel_img = cached_thumb.get_channel(&channel_std);

                // Sync the shared state representation under isolated scope
                {
                    let mut lock = state_clone.safe_lock();
                    for row in &mut lock.results {
                        if scanners::normalize_path_key(&row.path) == normalized_path {
                            row.thumbnail_data = Some(channel_img.clone());
                        }
                    }
                }

                // Update UI components dynamically inside the event loop
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    use slint::Model;
                    let store = ui.global::<Store>();
                    let slint_img = utils::ui::convert_to_slint_image(&channel_img);

                    // Update List Results
                    let results_model = store.get_results();
                    for i in 0..results_model.row_count() {
                        if let Some(mut r) = results_model.row_data(i)
                            && scanners::normalize_path_key(r.path.as_str()) == normalized_path
                        {
                            r.thumbnail = slint_img.clone();
                            results_model.set_row_data(i, r);
                        }
                    }

                    // Update Grid Results (mapping dynamically into virtualized GridRow columns)
                    let grid_model = store.get_grid_row_results();
                    for i in 0..grid_model.row_count() {
                        if let Some(mut r) = grid_model.row_data(i) {
                            let mut changed = false;
                            let mut items: Vec<_> = r.items.iter().collect();

                            for item in items.iter_mut() {
                                if scanners::normalize_path_key(item.path.as_str())
                                    == normalized_path
                                {
                                    item.thumbnail = slint_img.clone();
                                    changed = true;
                                }
                            }

                            if changed {
                                r.items = ModelRc::from(std::rc::Rc::new(VecModel::from(items)));
                                grid_model.set_row_data(i, r);
                            }
                        }
                    }

                    // Update Selected Group Files (Compare Panel)
                    let group_model = store.get_selected_group_files();
                    for i in 0..group_model.row_count() {
                        if let Some(mut r) = group_model.row_data(i)
                            && scanners::normalize_path_key(r.path.as_str()) == normalized_path
                        {
                            r.thumbnail = slint_img.clone();
                            group_model.set_row_data(i, r);
                        }
                    }
                });
            }
        });
        *lock = Some(handle);
    });
}

/// Binds standard window OS events, such as native drag and drop files/directories drop targets.
pub fn bind_drag_and_drop(app: &AppWindow) {
    let app_weak_dnd = app.as_weak();
    app.window().on_winit_window_event(move |_window, event| {
        if let slint::winit_030::winit::event::WindowEvent::DroppedFile(path_buf) = event {
            let path_str = path_buf.to_string_lossy().to_string();
            let is_dir = path_buf.is_dir();
            let is_file = path_buf.is_file();
            let app_copy = app_weak_dnd.clone();

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                let store = ui.global::<Store>();
                if is_dir {
                    store.set_dir_a(path_str.clone().into());
                    store.set_status_text(format!("Scan folder updated: {}", path_str).into());
                } else if is_file {
                    store.set_query_text(path_str.clone().into());
                    store.set_search_method(2);
                    store.set_status_text(format!("Reference image loaded: {}", path_str).into());
                }
                utils::settings::save_settings(&store);
            });
        }
        slint::winit_030::EventResult::Propagate
    });
}
