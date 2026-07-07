// src/scanners/ai.rs
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use xxhash_rust::xxh64::Xxh64;

use crate::core::database::{DatabaseService, DbRecord};
use crate::core::inference::{InferenceEngine, PreprocessingConfig};
use crate::state::{AiSearchResultSummary, DuplicateFileSummary, DuplicateGroupSummary};
use crate::utils::helpers::discover_files;

pub async fn run_ai_duplicate_scan(
    params: super::ScanParams,
) -> Result<Vec<DuplicateGroupSummary>> {
    let path = PathBuf::from(params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let (paths, warnings) = discover_files(&path, &params.extensions);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = DatabaseService::new();
    db.initialize(&db_dir, "images", 512).await?;
    let engine = InferenceEngine::new(&model_dir, 512, 4, "CPU")?;

    let mut records = Vec::new();
    let chunks: Vec<&[PathBuf]> = paths.chunks(params.batch_size).collect();

    for chunk in chunks {
        // Parallel load & immediate scale-down to protect RAM allocations
        let loaded_images: Vec<(PathBuf, image::DynamicImage)> = chunk
            .par_iter()
            .filter_map(|p| {
                let mut img =
                    crate::format_loaders::dds_loader::open_image_with_dds_fallback(p).ok()?;

                // Prevent OOM spikes by downscaling heavily before collecting
                if img.width() > 512 || img.height() > 512 {
                    img = img.resize_exact(512, 512, image::imageops::FilterType::Triangle);
                }
                Some((p.clone(), img))
            })
            .collect();

        let (chunk_paths, imgs): (Vec<PathBuf>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();
        let mut chunk_records = Vec::new();

        if !imgs.is_empty()
            && let Ok(vectors) = engine.encode_images_batch(&imgs, &PreprocessingConfig::default())
        {
            for (path, vector) in chunk_paths.into_iter().zip(vectors) {
                chunk_records.push(DbRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    path: path.to_string_lossy().to_string(),
                    channel: "Composite".to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

    db.add_batch(&records).await?;
    db.create_vector_index().await?;

    let dist_threshold = 1.0 - (params.similarity / 100.0);
    let mut uf = crate::utils::clustering::UnionFind::new(records.len());

    // Vector Similarity Search utilizing LanceDB
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
                    let matches = db_clone.search_similarity(&query_vec, 10).await?;
                    Ok::<_, anyhow::Error>((idx, matches))
                }
            })
            .buffer_unordered(16);

        while let Some(res) = query_stream.next().await {
            let (src_idx, matches) = res?;
            for m in matches {
                if let Some(target_idx) = records.iter().position(|r| r.path == m.path)
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

    for (root_idx, member_indices) in groups {
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

        let best_path = &file_summaries[0].path;
        let best_vec = records
            .iter()
            .find(|r| r.path == *best_path)
            .map(|r| &r.vector);

        if let Some(bv) = best_vec {
            for file in &mut file_summaries {
                if let Some(r) = records.iter().find(|r| r.path == file.path) {
                    let dot: f32 = r.vector.iter().zip(bv.iter()).map(|(a, b)| a * b).sum();
                    file.similarity = dot.clamp(0.0, 1.0) * 100.0;
                }
            }
        }

        let mut hasher = Xxh64::new(0);
        hasher.update(records[root_idx].path.as_bytes());
        let hash = format!("{:x}", hasher.digest());
        results.push(DuplicateGroupSummary {
            hash,
            files: file_summaries,
        });
    }
    Ok(results)
}

pub async fn run_ai_search(params: super::ScanParams) -> Result<Vec<AiSearchResultSummary>> {
    let path = PathBuf::from(params.dir_a);
    if !path.is_dir() {
        return Err(anyhow!("The specified path is not a valid directory"));
    }

    let (paths, warnings) = discover_files(&path, &params.extensions);
    for warn in warnings {
        crate::app::append_to_console_log(&warn);
    }

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = DatabaseService::new();
    db.initialize(&db_dir, "images", 512).await?;
    let engine = InferenceEngine::new(&model_dir, 512, 4, "CPU")?;

    let mut records = Vec::new();
    let chunks: Vec<&[PathBuf]> = paths.chunks(params.batch_size).collect();

    for chunk in chunks {
        let loaded_images: Vec<(PathBuf, image::DynamicImage)> = chunk
            .par_iter()
            .filter_map(|p| {
                let mut img =
                    crate::format_loaders::dds_loader::open_image_with_dds_fallback(p).ok()?;
                if img.width() > 512 || img.height() > 512 {
                    img = img.resize_exact(512, 512, image::imageops::FilterType::Triangle);
                }
                Some((p.clone(), img))
            })
            .collect();

        let (chunk_paths, imgs): (Vec<PathBuf>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();
        let mut chunk_records = Vec::new();

        if !imgs.is_empty()
            && let Ok(vectors) = engine.encode_images_batch(&imgs, &PreprocessingConfig::default())
        {
            for (path, vector) in chunk_paths.into_iter().zip(vectors) {
                chunk_records.push(DbRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    path: path.to_string_lossy().to_string(),
                    channel: "Composite".to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

    db.add_batch(&records).await?;
    db.create_vector_index().await?;

    let query_vec = engine.encode_text(&params.query_text)?;
    let search_results = db.search_similarity(&query_vec, 20).await?;

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
