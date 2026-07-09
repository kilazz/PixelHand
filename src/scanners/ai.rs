// src/scanners/ai.rs

use anyhow::{Result, anyhow};
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

// Synchronous thread-safe Semaphore to cap concurrent heavy image decoding pipelines
struct SyncSemaphore {
    count: Mutex<usize>,
    condvar: Condvar,
}

pub struct SyncSemaphoreGuard<'a> {
    sem: &'a SyncSemaphore,
}

impl SyncSemaphore {
    pub fn new(limit: usize) -> Self {
        Self {
            count: Mutex::new(limit),
            condvar: Condvar::new(),
        }
    }

    pub fn acquire(&self) -> SyncSemaphoreGuard<'_> {
        let mut lock = self.count.lock().unwrap();
        while *lock == 0 {
            lock = self.condvar.wait(lock).unwrap();
        }
        *lock -= 1;
        SyncSemaphoreGuard { sem: self }
    }
}

impl<'a> Drop for SyncSemaphoreGuard<'a> {
    fn drop(&mut self) {
        let mut lock = self.sem.count.lock().unwrap();
        *lock += 1;
        self.sem.condvar.notify_one();
    }
}

static DECODE_SEMAPHORE: OnceLock<SyncSemaphore> = OnceLock::new();

/// Limit concurrent decoding threads for heavy assets to prevent peak RAM spikes (MAX_CONCURRENT_IMAGE_LOADS = 2)
fn get_decode_semaphore() -> &'static SyncSemaphore {
    DECODE_SEMAPHORE.get_or_init(|| SyncSemaphore::new(2))
}

/// Maps Slint Search Precision ComboBox indices to IVF-PQ probes limits and refinement multipliers.
fn map_precision_presets(precision_idx: i32) -> (usize, usize) {
    match precision_idx {
        0 => (8, 1),    // Fast (minimal cluster probing)
        1 => (20, 3),   // Balanced (Default)
        2 => (80, 8),   // Accurate
        3 => (256, 20), // Exhaustive
        _ => (20, 3),
    }
}

pub async fn run_ai_duplicate_scan(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(params.dir_a.clone());
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

    // Dynamic ИИ Model and Database Dimension mapping
    let (folder_name, dim) = match params.ai_model {
        1 => ("siglip_base", 768),
        2 => ("dinov2_base", 768),
        _ => ("clip_vit_b32", 512),
    };

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir.join("models").join(folder_name);

    let mut db = DatabaseService::new();
    db.initialize(&db_dir, "images", dim).await?;

    let engine = InferenceEngine::new(&model_dir, dim, 4, &params.execution_provider)?;

    // Explode input files into logical channel splitting/luminance comparison tasks
    let items = super::generate_analysis_items(&paths, &params);
    let chunks: Vec<&[AnalysisItem]> = items.chunks(params.batch_size).collect();

    let mut records = Vec::new();
    let cancel_token = params.cancel_token.clone();

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

                // Acquire semaphore lock if decoding a heavy texture to prevent RAM spikes
                let _guard = if file_size > 10 * 1024 * 1024 || width > 2048 || height > 2048 {
                    Some(get_decode_semaphore().acquire())
                } else {
                    None
                };

                // Request decimated 512px mipmap level from DDS loader if available
                let img =
                    crate::format_loaders::dds_loader::open_image_with_dds_fallback(p, Some(512))
                        .ok()?;

                // Extract target color-channels or Luminance map
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
                Some((item.clone(), final_img))
            })
            .collect();

        let (chunk_items, imgs): (Vec<AnalysisItem>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();
        let mut chunk_records = Vec::new();

        if !imgs.is_empty()
            && let Ok(vectors) = engine.encode_images_batch(&imgs, &PreprocessingConfig::default())
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

                chunk_records.push(DbRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    path: item.path.to_string_lossy().to_string(),
                    channel: chan_str.to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

    db.add_batch(&records).await?;
    db.create_vector_index().await?;

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
                // Match path AND tagged channel explicitly to avoid crossing channel spaces
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

        // Decouple mutable and immutable borrows cleanly by cloning path strings
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
    let path = PathBuf::from(params.dir_a.clone());
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

    // Dynamic ИИ Model and Database Dimension mapping
    let (folder_name, dim) = match params.ai_model {
        1 => ("siglip_base", 768),
        2 => ("dinov2_base", 768),
        _ => ("clip_vit_b32", 512),
    };

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir.join("models").join(folder_name);

    let mut db = DatabaseService::new();
    db.initialize(&db_dir, "images", dim).await?;

    let engine = InferenceEngine::new(&model_dir, dim, 4, &params.execution_provider)?;

    let items = super::generate_analysis_items(&paths, &params);
    let chunks: Vec<&[AnalysisItem]> = items.chunks(params.batch_size).collect();

    let mut records = Vec::new();
    let cancel_token = params.cancel_token.clone();

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

                let img =
                    crate::format_loaders::dds_loader::open_image_with_dds_fallback(p, Some(512))
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
                Some((item.clone(), final_img))
            })
            .collect();

        let (chunk_items, imgs): (Vec<AnalysisItem>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();
        let mut chunk_records = Vec::new();

        if !imgs.is_empty()
            && let Ok(vectors) = engine.encode_images_batch(&imgs, &PreprocessingConfig::default())
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
                chunk_records.push(DbRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    path: item.path.to_string_lossy().to_string(),
                    channel: chan_str.to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

    db.add_batch(&records).await?;
    db.create_vector_index().await?;

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
