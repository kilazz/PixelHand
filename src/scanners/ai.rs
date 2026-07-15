// src/scanners/ai.rs

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Condvar, Mutex, OnceLock};
use xxhash_rust::xxh64::Xxh64;

use crate::core::database::{DatabaseService, DbRecord};
use crate::core::inference::{InferenceEngine, PreprocessingConfig};
use crate::scanners::AnalysisItem;
use crate::state::{AiSearchResultSummary, DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::helpers::discover_files;

struct DecoupledSemaphore {
    count: Mutex<usize>,
    condvar: Condvar,
}

pub struct DecoupledSemaphoreGuard<'a> {
    sem: &'a DecoupledSemaphore,
}

impl DecoupledSemaphore {
    pub fn new(limit: usize) -> Self {
        Self {
            count: Mutex::new(limit),
            condvar: Condvar::new(),
        }
    }

    pub fn acquire(&self) -> DecoupledSemaphoreGuard<'_> {
        let mut lock = self.count.lock().unwrap();
        while *lock == 0 {
            lock = self.condvar.wait(lock).unwrap();
        }
        *lock -= 1;
        DecoupledSemaphoreGuard { sem: self }
    }
}

impl<'a> Drop for DecoupledSemaphoreGuard<'a> {
    fn drop(&mut self) {
        let mut lock = self.sem.count.lock().unwrap();
        *lock += 1;
        self.sem.condvar.notify_one();
    }
}

static DECODE_SEMAPHORE: OnceLock<DecoupledSemaphore> = OnceLock::new();

fn get_decode_semaphore() -> &'static DecoupledSemaphore {
    DECODE_SEMAPHORE.get_or_init(|| DecoupledSemaphore::new(2))
}

fn map_precision_presets(precision_idx: i32) -> (usize, usize) {
    match precision_idx {
        0 => (8, 1),
        1 => (20, 3),
        2 => (80, 8),
        3 => (256, 20),
        _ => (20, 3),
    }
}

async fn scan_and_index_directory(
    params: &super::ScanParams,
) -> Result<(DatabaseService, InferenceEngine, Vec<DbRecord>)> {
    let path = PathBuf::from(&params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let ex_folders: Vec<String> = params
        .excluded_folders
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let (paths, warnings) = discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let (folder_name, dim) = match params.ai_model {
        1 => ("clip_vit_l14".to_string(), 768),
        2 => ("siglip_base".to_string(), 768),
        3 => ("siglip_large".to_string(), 1024),
        4 => ("dinov2_base".to_string(), 768),
        5 => ("siglip2_base".to_string(), 768),
        6 => ("llm2clip_base".to_string(), 1280),
        7 => (
            params.custom_model_path.clone(),
            params.custom_model_dim as usize,
        ),
        _ => ("clip_vit_b32".to_string(), 512),
    };

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let folder_hash = {
        let mut hasher = Xxh64::new(0);
        hasher.update(params.dir_a.as_bytes());
        hasher.digest()
    };

    let db_dir = app_dir
        .join(".lancedb_cache")
        .join(format!("{}_{:016x}", folder_name, folder_hash));

    let model_dir = if params.ai_model == 7 {
        PathBuf::from(&folder_name)
    } else {
        app_dir.join("models").join(&folder_name)
    };

    let mut db = DatabaseService::new();

    if let Err(e) = db.initialize(&db_dir, "images", dim).await {
        crate::app::append_to_console_log(&format!(
            "[Error] LanceDB partition initialization failed: {}. Rebuilding clean state...",
            e
        ));
        let _ = std::fs::remove_dir_all(&db_dir);
        db.initialize(&db_dir, "images", dim)
            .await
            .context("Re-initialization after cache wipe failed")?;
        crate::app::append_to_console_log("[Cache] Local database rebuilt successfully.");
    }

    let mut cache_map = std::collections::HashMap::new();
    if let Ok(db_entries) = db.load_cache_metadata().await {
        for (path, channel, size, mtime, vec) in db_entries {
            cache_map.insert((path, channel), (size, mtime, vec));
        }
    }

    let engine = InferenceEngine::new(&model_dir, dim, 4, &params.execution_provider)?;
    let items = super::generate_analysis_items(&paths, params);

    let mut final_records = Vec::with_capacity(items.len());
    let mut items_to_encode = Vec::new();
    let mut paths_to_prune = std::collections::HashSet::new();

    for item in &items {
        let p_str = item.path.to_string_lossy().to_string();
        let chan_str = match item.analysis_type {
            crate::core::perceptual::AnalysisType::R => "R",
            crate::core::perceptual::AnalysisType::G => "G",
            crate::core::perceptual::AnalysisType::B => "B",
            crate::core::perceptual::AnalysisType::A => "A",
            crate::core::perceptual::AnalysisType::Luminance => "Luminance",
            _ => "Composite",
        };

        let file_meta = fs::metadata(&item.path);
        let size = file_meta.as_ref().map(|m| m.len()).unwrap_or(0);
        let mtime = file_meta
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let key = (p_str.clone(), chan_str.to_string());

        if let Some((cached_size, cached_mtime, cached_vec)) = cache_map.get(&key) {
            if *cached_size == size && *cached_mtime == mtime {
                final_records.push(DbRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector: cached_vec.clone(),
                    path: p_str,
                    channel: chan_str.to_string(),
                    file_size: size,
                    mtime,
                });
                continue;
            } else {
                paths_to_prune.insert(p_str.clone());
            }
        }

        items_to_encode.push(item.clone());
    }

    if !items_to_encode.is_empty() {
        crate::app::append_to_console_log(&format!(
            "[Cache] Encoding {} new/modified files via {}...",
            items_to_encode.len(),
            folder_name
        ));

        let chunks: Vec<&[AnalysisItem]> = items_to_encode.chunks(params.batch_size).collect();
        let cancel_token = params.cancel_token.clone();
        let total_items = items_to_encode.len();
        let processed_items = std::sync::atomic::AtomicUsize::new(0);
        let mut newly_encoded_records = Vec::new();

        for chunk in chunks {
            if cancel_token.load(Ordering::Relaxed) {
                return Err(anyhow!("Scan cancelled by user."));
            }

            let loaded_images: Vec<(AnalysisItem, image::DynamicImage)> = chunk
                .par_iter()
                .filter_map(|item| {
                    if cancel_token.load(Ordering::Relaxed) {
                        return None;
                    }

                    let p = &item.path;
                    let file_size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                    let (width, height) = match imagesize::size(p) {
                        Ok(dim) => (dim.width, dim.height),
                        Err(_) => (0, 0),
                    };

                    let _guard = if file_size > 10 * 1024 * 1024 || width > 2048 || height > 2048 {
                        Some(get_decode_semaphore().acquire())
                    } else {
                        None
                    };

                    // Select closest mip level dynamically based on AI model architecture
                    let target_size = match params.ai_model {
                        2 | 3 | 5 => 384,
                        _ => 256,
                    };

                    let img = crate::format_loaders::dds_loader::open_image_with_dds_fallback(
                        p,
                        Some(target_size),
                    )
                    .ok()?;

                    let processed = crate::core::perceptual::preprocess_image_channels(
                        &img,
                        item.analysis_type,
                        params.prep_ignore_solid,
                    )?;

                    let mut final_img = processed;
                    if final_img.width() > 512 || final_img.height() > 512 {
                        final_img =
                            final_img.resize_exact(512, 512, image::imageops::FilterType::Triangle);
                    }

                    let current = processed_items.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(ref cb) = params.on_progress {
                        cb(current as f32 / total_items as f32, current, total_items);
                    }

                    Some((item.clone(), final_img))
                })
                .collect();

            let (chunk_items, imgs): (Vec<AnalysisItem>, Vec<image::DynamicImage>) =
                loaded_images.into_iter().unzip();
            let mut chunk_records = Vec::new();

            if !imgs.is_empty()
                && let Ok(vectors) = engine.encode_images_batch(
                    &imgs,
                    &PreprocessingConfig::for_model(params.ai_model, params.custom_model_arch),
                )
            {
                for (item, vector) in chunk_items.into_iter().zip(vectors) {
                    let chan_str = match item.analysis_type {
                        crate::core::perceptual::AnalysisType::R => "R",
                        crate::core::perceptual::AnalysisType::G => "G",
                        crate::core::perceptual::AnalysisType::B => "B",
                        crate::core::perceptual::AnalysisType::A => "A",
                        crate::core::perceptual::AnalysisType::Luminance => "Luminance",
                        _ => "Composite",
                    };

                    let file_meta = fs::metadata(&item.path);
                    let size = file_meta.as_ref().map(|m| m.len()).unwrap_or(0);
                    let mtime = file_meta
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);

                    chunk_records.push(DbRecord {
                        id: uuid::Uuid::new_v4().to_string(),
                        vector,
                        path: item.path.to_string_lossy().to_string(),
                        channel: chan_str.to_string(),
                        file_size: size,
                        mtime,
                    });
                }
            }
            newly_encoded_records.extend(chunk_records);
        }

        if !paths_to_prune.is_empty() {
            let mut prune_filter = String::new();
            for (idx, old_p) in paths_to_prune.iter().enumerate() {
                if idx > 0 {
                    prune_filter.push_str(" OR ");
                }
                prune_filter.push_str(&format!("path = '{}'", old_p.replace("'", "''")));
            }
            let _ = db.delete(&prune_filter).await;
        }

        db.add_batch(&newly_encoded_records).await?;
        final_records.extend(newly_encoded_records);
    } else {
        crate::app::append_to_console_log(
            "[Cache] 100% Cache Hit! Local database is fully synchronized.",
        );
    }

    db.create_vector_index().await?;

    Ok((db, engine, final_records))
}

pub async fn run_ai_duplicate_scan(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let (db, _engine, records) = scan_and_index_directory(&params).await?;

    let dist_threshold = 1.0 - (params.similarity / 100.0);
    let mut uf = crate::utils::clustering::UnionFind::new(records.len());

    let (nprobes, refine_factor) = map_precision_presets(params.search_precision);

    {
        use futures::StreamExt;
        let query_items: Vec<(usize, Vec<f32>)> = records
            .iter()
            .enumerate()
            .map(|(idx, r)| (idx, r.vector.clone()))
            .collect();

        let mut query_stream = futures::stream::iter(query_items)
            .map(|(idx, query_vec)| {
                let db_clone = db.clone();
                async move {
                    let matches = db_clone
                        .search_similarity(&query_vec, 10, Some(nprobes), Some(refine_factor))
                        .await?;
                    Ok::<_, anyhow::Error>((idx, matches))
                }
            })
            .buffer_unordered(16);

        while let Some(res) = query_stream.next().await {
            if params.cancel_token.load(Ordering::Relaxed) {
                return Err(anyhow!("Scan cancelled by user."));
            }

            let (src_idx, matches) = res?;
            for m in matches {
                if let Some(target_idx) = records
                    .iter()
                    .position(|r| r.path == m.path && r.channel == m.channel)
                    && target_idx != src_idx
                    && m.distance <= dist_threshold
                {
                    uf.union(src_idx, target_idx);
                }
            }
        }
    }

    let groups = uf.get_groups();
    let mut results = Vec::new();

    for (_root_idx, member_indices) in groups {
        if params.cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Scan cancelled by user."));
        }

        if member_indices.len() < 2 {
            continue;
        }

        let mut file_summaries = Vec::new();
        for idx in member_indices {
            let record = &records[idx];
            let p = PathBuf::from(&record.path);
            let metadata = fs::metadata(&p)?;
            let size = metadata.len();
            let (width, height) = match imagesize::size(&p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let qc_meta = crate::core::qc::extract_qc_metadata(&p).unwrap_or_else(|_| {
                crate::core::qc::QcImageMetadata {
                    width: width as u32,
                    height: height as u32,
                    file_size: size,
                    format_str: p
                        .extension()
                        .map(|e| e.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    compression_format: "Unknown".into(),
                    color_space: "Unknown".into(),
                    has_alpha: false,
                    bit_depth: 8,
                    mipmap_count: 1,
                    is_cubemap: false,
                    estimated_vram: 0,
                }
            });
            file_summaries.push(DuplicateFileSummary {
                path: record.path.clone(),
                size,
                width,
                height,
                format_str: qc_meta.format_str,
                compression_format: qc_meta.compression_format,
                color_space: qc_meta.color_space,
                has_alpha: qc_meta.has_alpha,
                bit_depth: qc_meta.bit_depth,
                mipmap_count: qc_meta.mipmap_count,
                is_cubemap: qc_meta.is_cubemap,
                similarity: 100.0,
            });
        }

        file_summaries.sort_by(|a, b| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            if area_a != area_b {
                area_b.cmp(&area_a)
            } else {
                b.size.cmp(&a.size)
            }
        });

        let best_path = file_summaries[0].path.clone();
        let best_vector = records
            .iter()
            .find(|r| r.path == best_path)
            .map(|r| r.vector.clone())
            .unwrap_or_default();

        for file in &mut file_summaries {
            if file.path == best_path {
                file.similarity = 100.0;
                continue;
            }
            let current_vec = records
                .iter()
                .find(|r| r.path == file.path)
                .map(|r| r.vector.clone())
                .unwrap_or_default();
            if !best_vector.is_empty() && !current_vec.is_empty() {
                let dot_product: f32 = best_vector
                    .iter()
                    .zip(current_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                file.similarity = (dot_product * 100.0).clamp(0.0, 100.0);
            }
        }

        let mut hasher = Xxh64::new(0);
        hasher.update(best_path.as_bytes());
        let hash = format!("{:016x}", hasher.digest());

        results.push(DuplicateGroupSummary {
            hash,
            files: file_summaries,
        });
    }
    Ok(results)
}

pub async fn run_ai_search(params: super::ScanParams) -> Result<Vec<AiSearchResultSummary>> {
    let (db, engine, _records) = scan_and_index_directory(&params).await?;
    let (nprobes, refine_factor) = map_precision_presets(params.search_precision);

    let query_vector = engine.encode_text(&params.query_text)?;
    let search_results = db
        .search_similarity(&query_vector, 20, Some(nprobes), Some(refine_factor))
        .await?;

    let results = search_results
        .into_iter()
        .map(|r| {
            let similarity = (1.0 - r.distance) * 100.0;
            AiSearchResultSummary {
                path: r.path,
                similarity,
            }
        })
        .collect();

    Ok(results)
}
