// src/ai/scanner.rs

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use xxhash_rust::xxh64::Xxh64;

use crate::ai::database::{DatabaseService, DbRecord};
use crate::ai::inference::{InferenceEngine, PreprocessingConfig};
use crate::perceptual::hashing::AnalysisType;
use crate::qc::rules::QcImageMetadata;
use crate::state::models::{
    AiModelType, AiSearchResultSummary, DuplicateFileSummary, DuplicateGroupSummary, ScanParams,
};
use crate::utils::clustering::UnionFind;
use crate::utils::helpers::{AnalysisItem, discover_files, generate_analysis_items};

// ==========================================
// --- DECOUPLED SEMAPHORE FOR PARALEL IO ---
// ==========================================

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

// ==========================================
// --- PRECISION PRESETS MAPPING ------------
// ==========================================

fn map_precision_presets(precision_idx: i32) -> (usize, usize) {
    match precision_idx {
        0 => (8, 1),
        1 => (20, 3),
        2 => (80, 8),
        3 => (256, 20),
        _ => (20, 3),
    }
}

// ==========================================
// --- DIRECTORY INDEXING ENGINE ------------
// ==========================================

async fn scan_and_index_directory(
    params: &ScanParams,
) -> Result<(DatabaseService, Arc<InferenceEngine>, Vec<DbRecord>)> {
    let path = PathBuf::from(&params.paths.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let ex_folders: Vec<String> = params
        .paths
        .excluded_folders
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let (paths, warnings) = discover_files(&path, &params.extensions, &ex_folders);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    // Resolve caching directory names and parameters based on model type
    let folder_name = if params.ai.ai_model == AiModelType::Custom {
        params.ai.custom_model_path.clone()
    } else {
        params.ai.ai_model.folder_name().to_string()
    };

    let dim = if params.ai.ai_model == AiModelType::Custom {
        params.ai.custom_model_dim as usize
    } else {
        params.ai.ai_model.dimensions()
    };

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let folder_hash = {
        let mut hasher = Xxh64::new(0);
        hasher.update(params.paths.dir_a.as_bytes());
        hasher.digest()
    };

    let db_dir = app_dir
        .join(".lancedb_cache")
        .join(format!("{}_{:016x}", folder_name, folder_hash));

    let model_dir = if params.ai.ai_model == AiModelType::Custom {
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

    let engine = Arc::new(InferenceEngine::new(
        &model_dir,
        dim,
        4,
        &params.execution_provider,
    )?);

    // Provide logging feedback on the active backend compilation provider
    crate::app::append_to_console_log(&format!(
        "[AI] ONNX Session compiled. Active hardware acceleration provider: {}",
        engine.actual_provider
    ));

    let items = generate_analysis_items(&paths, params);

    let mut final_records = Vec::with_capacity(items.len());
    let mut items_to_encode = Vec::new();
    let mut paths_to_prune = std::collections::HashSet::new();

    for item in &items {
        let p_str = item.path.to_string_lossy().to_string();
        let chan_str = match item.analysis_type {
            AnalysisType::R => "R",
            AnalysisType::G => "G",
            AnalysisType::B => "B",
            AnalysisType::A => "A",
            AnalysisType::Luminance => "Luminance",
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
                let mut hasher = Xxh64::new(0);
                hasher.update(p_str.as_bytes());
                hasher.update(chan_str.as_bytes());
                let deterministic_id = format!("{:016x}", hasher.digest());

                final_records.push(DbRecord {
                    id: deterministic_id,
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
        let processed_items = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut newly_encoded_records = Vec::new();

        for chunk in chunks {
            if cancel_token.load(Ordering::Relaxed) {
                return Err(anyhow!("Scan cancelled by user."));
            }

            let chunk_vec = chunk.to_vec();
            let cancel_token_clone = cancel_token.clone();
            let engine_clone = engine.clone();
            let params_ai_model = params.ai.ai_model;
            let params_ai_custom_arch = params.ai.custom_model_arch;
            let params_prep_ignore_solid = params.prep.prep_ignore_solid;
            let on_progress_clone = params.on_progress.clone();
            let processed_items_clone = processed_items.clone();

            // Offload CPU-heavy file decoding and ONNX batch encoding to blocking task pools
            let chunk_records_res = tokio::task::spawn_blocking(move || {
                let loaded_images: Vec<(AnalysisItem, image::DynamicImage)> = chunk_vec
                    .par_iter()
                    .filter_map(|item| {
                        if cancel_token_clone.load(Ordering::Relaxed) {
                            return None;
                        }

                        let p = &item.path;
                        let file_size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                        let (width, height) = match imagesize::size(p) {
                            Ok(dim) => (dim.width, dim.height),
                            Err(_) => (0, 0),
                        };

                        let _guard =
                            if file_size > 10 * 1024 * 1024 || width > 2048 || height > 2048 {
                                Some(get_decode_semaphore().acquire())
                            } else {
                                None
                            };

                        // Select closest target resolution bounds based on model requirements
                        let target_size = match params_ai_model {
                            AiModelType::SiglipBase
                            | AiModelType::SiglipLarge
                            | AiModelType::Siglip2Base => 384,
                            _ => 256,
                        };

                        let img = crate::format_loaders::open_image_with_dds_fallback(
                            p,
                            Some(target_size),
                            None,
                        )
                        .ok()?;

                        let processed = crate::perceptual::hashing::preprocess_image_channels(
                            &img,
                            item.analysis_type,
                            params_prep_ignore_solid,
                        )?;

                        let mut final_img = processed;
                        if final_img.width() > 512 || final_img.height() > 512 {
                            final_img = final_img.resize_exact(
                                512,
                                512,
                                image::imageops::FilterType::Triangle,
                            );
                        }

                        let current = processed_items_clone.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(ref cb) = on_progress_clone {
                            cb(current as f32 / total_items as f32, current, total_items);
                        }

                        Some((item.clone(), final_img))
                    })
                    .collect();

                let (chunk_items, imgs): (Vec<AnalysisItem>, Vec<image::DynamicImage>) =
                    loaded_images.into_iter().unzip();
                let mut chunk_records = Vec::new();

                if !imgs.is_empty()
                    && let Ok(vectors) = engine_clone.encode_images_batch(
                        &imgs,
                        &PreprocessingConfig::for_model(params_ai_model, params_ai_custom_arch),
                    )
                {
                    for (item, vector) in chunk_items.into_iter().zip(vectors) {
                        let chan_str = match item.analysis_type {
                            AnalysisType::R => "R",
                            AnalysisType::G => "G",
                            AnalysisType::B => "B",
                            AnalysisType::A => "A",
                            AnalysisType::Luminance => "Luminance",
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

                        let path_str = item.path.to_string_lossy().to_string();
                        let mut hasher = Xxh64::new(0);
                        hasher.update(path_str.as_bytes());
                        hasher.update(chan_str.as_bytes());
                        let deterministic_id = format!("{:016x}", hasher.digest());

                        chunk_records.push(DbRecord {
                            id: deterministic_id,
                            vector,
                            path: path_str,
                            channel: chan_str.to_string(),
                            file_size: size,
                            mtime,
                        });
                    }
                }
                Ok::<_, anyhow::Error>(chunk_records)
            })
            .await?;

            let chunk_records = chunk_records_res?;
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

// ==========================================
// --- DUPLICATE SCANNERS IMPLEMENTATIONS ---
// ==========================================

/// Scans the folder using semantic vector similarity threshold margins to cluster duplicate groups.
pub async fn run_ai_duplicate_scan(params: ScanParams) -> Result<Vec<DuplicateGroupSummary>> {
    let (db, _engine, records) = scan_and_index_directory(&params).await?;

    let dist_threshold = 1.0 - (params.similarity / 100.0);
    let mut uf = UnionFind::new(records.len());

    let (nprobes, refine_factor) = map_precision_presets(params.ai.search_precision);

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

            let qc_meta = QcImageMetadata::extract_or_fallback(&p);

            file_summaries.push(DuplicateFileSummary {
                path: record.path.clone(),
                size: qc_meta.file_size,
                width: qc_meta.width as usize,
                height: qc_meta.height as usize,
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

// ==========================================
// --- SEMANTIC SEARCH EXECUTION ------------
// ==========================================

/// Runs a semantic query search (text queries or image reference queries) using the vector database.
pub async fn run_ai_search(params: ScanParams) -> Result<Vec<AiSearchResultSummary>> {
    let (db, engine, _records) = scan_and_index_directory(&params).await?;
    let (nprobes, refine_factor) = map_precision_presets(params.ai.search_precision);

    let query_path = std::path::Path::new(&params.paths.query_text);

    // Route query extraction: if the string represents a valid image file, run through Vision Encoder
    let query_vector = if query_path.is_file() {
        tracing::info!("Query target detected as a file path. Encoding reference image...");

        let img = crate::format_loaders::open_image_with_dds_fallback(query_path, Some(256), None)
            .map_err(|e| anyhow!("Failed to load reference image for query: {}", e))?;

        let config =
            PreprocessingConfig::for_model(params.ai.ai_model, params.ai.custom_model_arch);

        let mut vectors = engine.encode_images_batch(&[img], &config)?;
        vectors
            .pop()
            .ok_or_else(|| anyhow!("Failed to generate visual embedding for query reference"))?
    } else {
        tracing::info!("Query target is a text string. Encoding via Text Tokenizer...");
        engine.encode_text(&params.paths.query_text)?
    };

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
