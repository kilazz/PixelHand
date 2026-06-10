// src-tauri/src/main.rs
mod database;
mod dds_loader;
mod downloader;
mod inference;
mod perceptual;
mod qc;
mod tonemapper;

use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use xxhash_rust::xxh64::Xxh64;

/// Disjoint Set Union (Union-Find) structure for clustering similar image vectors.
struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }

    fn find(&mut self, s: &str) -> String {
        if !self.parent.contains_key(s) {
            self.parent.insert(s.to_string(), s.to_string());
            return s.to_string();
        }
        let mut path = Vec::new();
        let mut root = s.to_string();
        while let Some(parent) = self.parent.get(&root) {
            if *parent == root {
                break;
            }
            path.push(root.clone());
            root = parent.clone();
        }
        for node in path {
            self.parent.insert(node, root.clone());
        }
        root
    }

    fn union(&mut self, s1: &str, s2: &str) {
        let root1 = self.find(s1);
        let root2 = self.find(s2);
        if root1 != root2 {
            self.parent.insert(root2, root1);
        }
    }

    fn get_groups(mut self) -> HashMap<String, Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        let keys: Vec<String> = self.parent.keys().cloned().collect();
        for key in keys {
            let root = self.find(&key);
            groups.entry(root).or_default().push(key);
        }
        groups
    }
}

/// Locates or creates the persistent data storage directory next to the executable.
fn get_portable_app_data_dir() -> Result<PathBuf, String> {
    let exe_path =
        std::env::current_exe().map_err(|e| format!("Failed to get executable path: {}", e))?;
    let exe_dir = exe_path
        .parent()
        .ok_or("Failed to determine executable directory")?;
    let portable_data_dir = exe_dir.join("PixelHand_Data");
    fs::create_dir_all(&portable_data_dir)
        .map_err(|e| format!("Failed to create portable data dir: {}", e))?;
    Ok(portable_data_dir)
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct DuplicateFileSummary {
    pub path: String,
    pub size: u64,
    pub width: usize,
    pub height: usize,
    pub format_str: String,
    pub compression_format: String,
    pub color_space: String,
    pub has_alpha: bool,
    pub bit_depth: u32,
    pub mipmap_count: u32,
    pub similarity: f32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct DuplicateGroupSummary {
    pub hash: String,
    pub files: Vec<DuplicateFileSummary>,
}

#[derive(serde::Serialize, Clone)]
pub struct QcIssueSummary {
    pub path: String,
    pub issue: String,
    pub details: String,
}

#[derive(serde::Serialize, Clone)]
pub struct PerceptualGroupSummary {
    pub group_id: usize,
    pub files: Vec<String>,
}

#[derive(serde::Serialize, Clone)]
pub struct AiSearchResultSummary {
    pub path: String,
    pub similarity: f32,
}

/// Cache item for storing decoded images in memory to prevent repetitive disk I/O on hover.
struct DecodedCacheItem {
    mtime: std::time::SystemTime,
    image: image::RgbaImage,
}

// Thread-safe global cache of decoded preview images.
static DECODED_CACHE: OnceLock<Mutex<HashMap<String, DecodedCacheItem>>> = OnceLock::new();

/// Command: Downloads necessary AI model files with real-time UI progress updates.
#[tauri::command]
async fn download_models(app: tauri::AppHandle) -> Result<(), String> {
    let model_dir = get_portable_app_data_dir()?
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");
    fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;
    let files = [
        (
            "tokenizer.json",
            "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
        ),
        (
            "text.onnx",
            "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
        ),
        (
            "visual.onnx",
            "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
        ),
    ];
    for (name, url) in files {
        let dest = model_dir.join(name);
        if !dest.exists() {
            downloader::download_file_with_progress(&app, url, &dest, name)
                .await
                .map_err(|e| format!("Failed to download {}: {}", name, e))?;
        }
    }
    Ok(())
}

/// Command: Scans a directory to locate byte-exact duplicates using xxHash64.
#[tauri::command]
async fn run_exact_scan(directory: String) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path);
    let metadata_list: Vec<FileMetadata> = paths
        .par_iter()
        .filter_map(|p| {
            let metadata = fs::metadata(p).ok()?;
            let size = metadata.len();
            if size == 0 {
                return None;
            }
            let (width, height) = match imagesize::size(p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let hash = calculate_xxhash(p).ok()?;
            Some(FileMetadata {
                path: p.clone(),
                size,
                width,
                height,
                hash,
            })
        })
        .collect();

    let mut groups: HashMap<String, Vec<FileMetadata>> = HashMap::new();
    for meta in metadata_list {
        groups.entry(meta.hash.clone()).or_default().push(meta);
    }
    let mut dups: Vec<(String, Vec<FileMetadata>)> =
        groups.into_iter().filter(|(_, f)| f.len() > 1).collect();
    dups.sort_by(|a, b| b.1[0].size.cmp(&a.1[0].size));

    let mut results = Vec::new();
    for (hash, files) in dups {
        let file_summaries = files
            .into_iter()
            .map(|f| {
                let qc_meta =
                    qc::extract_qc_metadata(&f.path).unwrap_or_else(|_| qc::QcImageMetadata {
                        path: f.path.clone(),
                        width: f.width as u32,
                        height: f.height as u32,
                        file_size: f.size,
                        format_str: f
                            .path
                            .extension()
                            .map(|e| e.to_string_lossy().to_string())
                            .unwrap_or_default(),
                        compression_format: "Unknown".to_string(),
                        color_space: "Unknown".to_string(),
                        has_alpha: false,
                        bit_depth: 8,
                        mipmap_count: 1,
                    });
                DuplicateFileSummary {
                    path: f.path.to_string_lossy().to_string(),
                    size: f.size,
                    width: f.width,
                    height: f.height,
                    format_str: qc_meta.format_str,
                    compression_format: qc_meta.compression_format,
                    color_space: qc_meta.color_space,
                    has_alpha: qc_meta.has_alpha,
                    bit_depth: qc_meta.bit_depth,
                    mipmap_count: qc_meta.mipmap_count,
                    similarity: 100.0,
                }
            })
            .collect();
        results.push(DuplicateGroupSummary {
            hash,
            files: file_summaries,
        });
    }
    Ok(results)
}

/// Command: Conducts an absolute technical quality control audit on target assets.
#[tauri::command]
async fn run_qc_scan(
    directory: String,
    check_npot: bool,
    check_mipmaps: bool,
    check_block_align: bool,
    check_bit_depth: bool,
    validate_normals: bool,
    normals_tags: String,
) -> Result<Vec<QcIssueSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path);

    let issues: Vec<QcIssueSummary> = paths
        .par_iter()
        .flat_map(|p| {
            let mut file_issues = Vec::new();
            let (w, h) = match imagesize::size(p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);

            let qc_meta = qc::extract_qc_metadata(p).unwrap_or_else(|_| qc::QcImageMetadata {
                path: p.clone(),
                width: w as u32,
                height: h as u32,
                file_size: size,
                format_str: ext.clone(),
                compression_format: ext,
                color_space: "sRGB".to_string(),
                has_alpha: false,
                bit_depth: 8,
                mipmap_count: 1,
            });

            let abs_issues = qc::check_absolute(
                &qc_meta,
                check_npot,
                check_block_align,
                check_mipmaps,
                check_bit_depth,
            );
            for issue in abs_issues {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details: String::new(),
                });
            }

            if validate_normals {
                let path_str = p.to_string_lossy().to_lowercase();
                let should_check = if normals_tags.is_empty() {
                    true
                } else {
                    normals_tags
                        .split(',')
                        .map(|t| t.trim().to_lowercase())
                        .filter(|t| !t.is_empty())
                        .any(|t| path_str.contains(&t))
                };
                if should_check {
                    if let Some((issue, details)) = qc::check_normal_map_integrity(p, 0.15) {
                        file_issues.push(QcIssueSummary {
                            path: p.to_string_lossy().to_string(),
                            issue,
                            details,
                        });
                    }
                }
            }
            file_issues
        })
        .collect();
    Ok(issues)
}

/// Command: Compares assets between two folders to discover relative QC modifications.
#[tauri::command]
async fn run_folder_compare(
    directory_a: String,
    directory_b: String,
    check_size_bloat: bool,
    check_alpha: bool,
    check_color_space: bool,
    check_compression: bool,
    match_by_stem: bool,
) -> Result<Vec<QcIssueSummary>, String> {
    let path_a = PathBuf::from(directory_a);
    let path_b = PathBuf::from(directory_b);
    if !path_a.is_dir() || !path_b.is_dir() {
        return Err("Both target paths must be valid directories".into());
    }

    let files_a = discover_files(&path_a);
    let files_b = discover_files(&path_b);

    let get_key = |p: &Path, by_stem: bool| -> String {
        if by_stem {
            p.file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        } else {
            p.file_name()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        }
    };

    let mut map_a = HashMap::new();
    for p in &files_a {
        map_a.insert(get_key(p, match_by_stem), p.clone());
    }

    let mut issues = Vec::new();

    for p_b in &files_b {
        let key = get_key(p_b, match_by_stem);
        if let Some(p_a) = map_a.get(&key) {
            let size_a = fs::metadata(p_a).map(|m| m.len()).unwrap_or(0);
            let size_b = fs::metadata(p_b).map(|m| m.len()).unwrap_or(0);

            let (w_a, h_a) = match imagesize::size(p_a) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let (w_b, h_b) = match imagesize::size(p_b) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };

            let ext_a = p_a
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            let ext_b = p_b
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();

            let meta_a = qc::extract_qc_metadata(p_a).unwrap_or_else(|_| qc::QcImageMetadata {
                path: p_a.clone(),
                width: w_a as u32,
                height: h_a as u32,
                file_size: size_a,
                format_str: ext_a.clone(),
                compression_format: ext_a,
                color_space: "sRGB".to_string(),
                has_alpha: false,
                bit_depth: 8,
                mipmap_count: 1,
            });

            let meta_b = qc::extract_qc_metadata(p_b).unwrap_or_else(|_| qc::QcImageMetadata {
                path: p_b.clone(),
                width: w_b as u32,
                height: h_b as u32,
                file_size: size_b,
                format_str: ext_b.clone(),
                compression_format: ext_b,
                color_space: "sRGB".to_string(),
                has_alpha: false,
                bit_depth: 8,
                mipmap_count: 1,
            });

            let rel_issues = qc::check_relative(
                &meta_a,
                &meta_b,
                check_size_bloat,
                check_alpha,
                check_color_space,
                check_compression,
            );

            for issue in rel_issues {
                issues.push(QcIssueSummary {
                    path: p_b.to_string_lossy().to_string(),
                    issue,
                    details: format!(
                        "Folder B vs A ({})",
                        p_a.file_name().unwrap_or_default().to_string_lossy()
                    ),
                });
            }
        }
    }

    Ok(issues)
}

/// Command: Computes advanced visual hashes (supporting custom channels and pHash).
#[tauri::command]
async fn run_perceptual_scan(
    directory: String,
    threshold: u32,
    analysis_type: String,
    ignore_solid_channels: bool,
    use_phash: bool,
) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path);

    let ana_type = match analysis_type.as_str() {
        "Luminance" => perceptual::AnalysisType::Luminance,
        "R" => perceptual::AnalysisType::R,
        "G" => perceptual::AnalysisType::G,
        "B" => perceptual::AnalysisType::B,
        "A" => perceptual::AnalysisType::A,
        _ => perceptual::AnalysisType::Composite,
    };

    let hashes: Vec<(PathBuf, String)> = paths
        .par_iter()
        .filter_map(|p| {
            let res = perceptual::calculate_perceptual_hashes(p, ana_type, ignore_solid_channels)?;
            let hash_str = if use_phash { res.phash } else { res.dhash };
            Some((p.clone(), hash_str))
        })
        .collect();

    let mut visited = vec![false; hashes.len()];
    let mut results = Vec::new();
    let mut group_id = 0;

    // Convert similarity percentage S% (e.g. 90) to Hamming threshold
    let max_dist = (((100.0 - threshold as f32) / 100.0) * 64.0).round() as u32;

    for i in 0..hashes.len() {
        if visited[i] {
            continue;
        }
        let mut group_members = vec![hashes[i].clone()];
        for j in (i + 1)..hashes.len() {
            if visited[j] {
                continue;
            }
            if let Some(dist) = perceptual::calculate_hamming_distance(&hashes[i].1, &hashes[j].1) {
                if dist <= max_dist {
                    group_members.push(hashes[j].clone());
                    visited[j] = true;
                }
            }
        }
        if group_members.len() > 1 {
            group_id += 1;
            let mut file_summaries = Vec::new();
            for (p_buf, hash_str) in &group_members {
                let metadata = fs::metadata(p_buf).map_err(|e| e.to_string())?;
                let size = metadata.len();
                let (width, height) = match imagesize::size(p_buf) {
                    Ok(dim) => (dim.width, dim.height),
                    Err(_) => (0, 0),
                };
                let qc_meta =
                    qc::extract_qc_metadata(p_buf).unwrap_or_else(|_| qc::QcImageMetadata {
                        path: p_buf.clone(),
                        width: width as u32,
                        height: height as u32,
                        file_size: size,
                        format_str: p_buf
                            .extension()
                            .map(|e| e.to_string_lossy().to_string())
                            .unwrap_or_default(),
                        compression_format: "Unknown".to_string(),
                        color_space: "Unknown".to_string(),
                        has_alpha: false,
                        bit_depth: 8,
                        mipmap_count: 1,
                    });

                file_summaries.push(DuplicateFileSummary {
                    path: p_buf.to_string_lossy().to_string(),
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

            // Recompute similarity against the new "best" representative at file_summaries[0]
            let best_path = &file_summaries[0].path;
            let best_hash = group_members
                .iter()
                .find(|(pb, _)| pb.to_string_lossy().to_string() == *best_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or(&group_members[0].1);

            for file in &mut file_summaries {
                let cur_hash = group_members
                    .iter()
                    .find(|(pb, _)| pb.to_string_lossy().to_string() == file.path)
                    .map(|(_, h)| h.as_str())
                    .unwrap_or(&group_members[0].1);
                let dist = perceptual::calculate_hamming_distance(best_hash, cur_hash).unwrap_or(0);
                file.similarity = (1.0 - (dist as f32 / 64.0)) * 100.0;
            }

            results.push(DuplicateGroupSummary {
                hash: group_id.to_string(),
                files: file_summaries,
            });
        }
    }
    Ok(results)
}

/// Command: Discovers similar duplicates using ONNX embeddings.
#[tauri::command]
async fn run_ai_duplicate_scan(
    directory: String,
    threshold: f32,
) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path);
    let app_dir = get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = database::DatabaseService::new();
    db.initialize(&db_dir, "images", 512)
        .await
        .map_err(|e| e.to_string())?;
    let engine = inference::InferenceEngine::new(&model_dir, 512, 4).map_err(|e| e.to_string())?;

    let records: Vec<database::DbRecord> = paths
        .par_iter()
        .filter_map(|p| {
            let img = dds_loader::open_image_with_dds_fallback(p).ok()?;
            let vector = engine
                .encode_image(&img, &inference::PreprocessingConfig::default())
                .ok()?;
            let id = uuid::Uuid::new_v4().to_string();
            Some(database::DbRecord {
                id,
                vector,
                path: p.to_string_lossy().to_string(),
                channel: "Composite".to_string(),
            })
        })
        .collect();

    db.add_batch(&records).await.map_err(|e| e.to_string())?;
    db.create_vector_index().await.map_err(|e| e.to_string())?;

    let dist_threshold = 1.0 - (threshold / 100.0);
    let mut uf = UnionFind::new();

    for r in &records {
        let matches = db
            .search_similarity(&r.vector, 10)
            .await
            .map_err(|e| e.to_string())?;
        for m in matches {
            if m.path != r.path && m.distance <= dist_threshold {
                uf.union(&r.path, &m.path);
            }
        }
    }

    let groups = uf.get_groups();
    let mut results = Vec::new();

    for (root, members) in groups {
        if members.len() < 2 {
            continue;
        }
        let mut file_summaries = Vec::new();
        for member_path in members {
            let p = PathBuf::from(&member_path);
            let metadata = fs::metadata(&p).map_err(|e| e.to_string())?;
            let size = metadata.len();
            let (width, height) = match imagesize::size(&p) {
                Ok(dim) => (dim.width, dim.height),
                Err(_) => (0, 0),
            };
            let qc_meta = qc::extract_qc_metadata(&p).unwrap_or_else(|_| qc::QcImageMetadata {
                path: p.clone(),
                width: width as u32,
                height: height as u32,
                file_size: size,
                format_str: p
                    .extension()
                    .map(|e| e.to_string_lossy().to_string())
                    .unwrap_or_default(),
                compression_format: "Unknown".to_string(),
                color_space: "Unknown".to_string(),
                has_alpha: false,
                bit_depth: 8,
                mipmap_count: 1,
            });
            file_summaries.push(DuplicateFileSummary {
                path: member_path,
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
                    let sim_pct = dot.clamp(0.0, 1.0) * 100.0;
                    file.similarity = sim_pct;
                } else {
                    file.similarity = 100.0;
                }
            }
        }

        let mut hasher = Xxh64::new(0);
        hasher.update(root.as_bytes());
        let hash = format!("{:x}", hasher.digest());
        results.push(DuplicateGroupSummary {
            hash,
            files: file_summaries,
        });
    }
    Ok(results)
}

/// Command: Queries semantic image contents using a descriptive text query.
#[tauri::command]
async fn run_ai_search(
    directory: String,
    query: String,
) -> Result<Vec<AiSearchResultSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path);
    let app_dir = get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = database::DatabaseService::new();
    db.initialize(&db_dir, "images", 512)
        .await
        .map_err(|e| e.to_string())?;
    let engine = inference::InferenceEngine::new(&model_dir, 512, 4).map_err(|e| e.to_string())?;

    let records: Vec<database::DbRecord> = paths
        .par_iter()
        .filter_map(|p| {
            let img = dds_loader::open_image_with_dds_fallback(p).ok()?;
            let vector = engine
                .encode_image(&img, &inference::PreprocessingConfig::default())
                .ok()?;
            let id = uuid::Uuid::new_v4().to_string();
            Some(database::DbRecord {
                id,
                vector,
                path: p.to_string_lossy().to_string(),
                channel: "Composite".to_string(),
            })
        })
        .collect();

    db.add_batch(&records).await.map_err(|e| e.to_string())?;
    db.create_vector_index().await.map_err(|e| e.to_string())?;

    let query_vec = engine.encode_text(&query).map_err(|e| e.to_string())?;
    let search_results = db
        .search_similarity(&query_vec, 20)
        .await
        .map_err(|e| e.to_string())?;
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

/// Command: Queries semantic contents using a reference image.
#[tauri::command]
async fn run_image_search(
    directory: String,
    reference_image: String,
) -> Result<Vec<AiSearchResultSummary>, String> {
    let path = PathBuf::from(directory);
    let ref_path = PathBuf::from(reference_image);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    if !ref_path.is_file() {
        return Err("The reference image path is invalid".into());
    }
    let paths = discover_files(&path);
    let app_dir = get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = database::DatabaseService::new();
    db.initialize(&db_dir, "images", 512)
        .await
        .map_err(|e| e.to_string())?;
    let engine = inference::InferenceEngine::new(&model_dir, 512, 4).map_err(|e| e.to_string())?;

    let records: Vec<database::DbRecord> = paths
        .par_iter()
        .filter_map(|p| {
            let img = dds_loader::open_image_with_dds_fallback(p).ok()?;
            let vector = engine
                .encode_image(&img, &inference::PreprocessingConfig::default())
                .ok()?;
            let id = uuid::Uuid::new_v4().to_string();
            Some(database::DbRecord {
                id,
                vector,
                path: p.to_string_lossy().to_string(),
                channel: "Composite".to_string(),
            })
        })
        .collect();

    db.add_batch(&records).await.map_err(|e| e.to_string())?;
    db.create_vector_index().await.map_err(|e| e.to_string())?;

    let ref_img = dds_loader::open_image_with_dds_fallback(&ref_path).map_err(|e| e.to_string())?;
    let query_vec = engine
        .encode_image(&ref_img, &inference::PreprocessingConfig::default())
        .map_err(|e| e.to_string())?;
    let search_results = db
        .search_similarity(&query_vec, 20)
        .await
        .map_err(|e| e.to_string())?;
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

/// Command: Renders an HDR file preview in a temporary directory.
#[tauri::command]
async fn test_tonemap(exr_path: String) -> Result<String, String> {
    let path = PathBuf::from(exr_path);
    if !path.is_file() {
        return Err("Target path is not a valid file".into());
    }
    let (hdr, w, h) = tonemapper::load_exr_rgba(&path).map_err(|e| e.to_string())?;
    let ldr = tonemapper::tonemap_hdr_to_ldr_rgba(
        &hdr,
        w,
        h,
        tonemapper::TonemapOperator::AcesFilmic,
        1.5,
    )
    .map_err(|e| e.to_string())?;
    let temp_dir = get_portable_app_data_dir()?.join("temp");
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let out_path = temp_dir.join("tonemapped_preview.png");
    ldr.save(&out_path).map_err(|e| e.to_string())?;
    Ok(out_path.to_string_lossy().to_string())
}

/// Command: Generates a comparison difference map image.
///
/// Optimizes execution speed on extremely high-resolution assets by dynamically
/// downscaling targets down to 1024px on the fly.
#[tauri::command]
async fn calculate_diff(file1: String, file2: String) -> Result<String, String> {
    let p1 = PathBuf::from(file1);
    let p2 = PathBuf::from(file2);
    if !p1.is_file() || !p2.is_file() {
        return Err("One or both targets are not valid files".into());
    }

    let mut img1 = dds_loader::open_image_with_dds_fallback(&p1).map_err(|e| e.to_string())?;
    let mut img2 = dds_loader::open_image_with_dds_fallback(&p2).map_err(|e| e.to_string())?;

    if img1.width() > 1024 || img1.height() > 1024 {
        img1 = img1.thumbnail(1024, 1024);
    }
    if img2.width() > 1024 || img2.height() > 1024 {
        img2 = img2.thumbnail(1024, 1024);
    }

    let rgba1 = img1.to_rgba8();
    let rgba2 = img2.to_rgba8();

    let diff =
        tonemapper::calculate_difference_map(&rgba1, &rgba2, true).map_err(|e| e.to_string())?;
    let temp_dir = get_portable_app_data_dir()?.join("temp");
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let out_path = temp_dir.join("diff_output.png");
    diff.save(&out_path).map_err(|e| e.to_string())?;
    Ok(out_path.to_string_lossy().to_string())
}

/// Command: Moves chosen paths safely to the system Trash/Recycle Bin.
#[tauri::command]
async fn delete_files(paths: Vec<String>) -> Result<(), String> {
    let mut files_to_delete = Vec::new();
    for path_str in paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            files_to_delete.push(path);
        }
    }
    if !files_to_delete.is_empty() {
        trash::delete_all(&files_to_delete)
            .map_err(|e| format!("Failed to move files to Trash: {}", e))?;
    }
    Ok(())
}

/// Command: Converts chosen duplicate paths into hardlinks to the source file.
#[tauri::command]
async fn create_hardlinks(pairs: Vec<(String, String)>) -> Result<(), String> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);
        if !source.exists() {
            return Err(format!("Source file not found: {:?}", source));
        }
        if target.exists() {
            fs::remove_file(&target).map_err(|e| e.to_string())?;
        }
        fs::hard_link(&source, &target).map_err(|e| format!("Failed to create hardlink: {}", e))?;
    }
    Ok(())
}

/// Command: Converts chosen duplicate paths into reflinks (Copy-on-Write fallback to hardlink).
#[tauri::command]
async fn create_reflinks(pairs: Vec<(String, String)>) -> Result<(), String> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);
        if !source.exists() {
            return Err(format!("Source file not found: {:?}", source));
        }
        if target.exists() {
            fs::remove_file(&target).map_err(|e| e.to_string())?;
        }
        fs::hard_link(&source, &target).map_err(|e| format!("Failed to link file: {}", e))?;
    }
    Ok(())
}

/// Command: Extracts a chosen pixel channel in grayscale to display in comparative viewport.
///
/// Highly optimized by verifying against a thread-safe global in-memory validation cache
/// (`DECODED_CACHE`) and dynamically downscaling large targets down to 512px on first load.
#[tauri::command]
async fn get_channel_preview(path: String, channel: String) -> Result<String, String> {
    let p = PathBuf::from(&path);
    if !p.is_file() {
        return Err("Invalid file path".into());
    }

    // Retrieve file modification timestamp for validation
    let current_mtime = fs::metadata(&p)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let cache_mutex = DECODED_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache_mutex.lock().map_err(|_| "Cache lock poisoned")?;

    // Validate cached entry existence and modification timestamp consistency
    let has_valid_cache = cache
        .get(&path)
        .map(|item| item.mtime == current_mtime)
        .unwrap_or(false);

    if !has_valid_cache {
        let mut img = dds_loader::open_image_with_dds_fallback(&p).map_err(|e| e.to_string())?;

        // Fast downscaling limit optimization
        if img.width() > 512 || img.height() > 512 {
            img = img.thumbnail(512, 512);
        }

        let rgba = img.to_rgba8();
        cache.insert(
            path.clone(),
            DecodedCacheItem {
                mtime: current_mtime,
                image: rgba,
            },
        );

        // Clear out oldest cached allocations if limit exceeded
        if cache.len() > 16 {
            cache.clear();
            let mut img =
                dds_loader::open_image_with_dds_fallback(&p).map_err(|e| e.to_string())?;
            if img.width() > 512 || img.height() > 512 {
                img = img.thumbnail(512, 512);
            }
            cache.insert(
                path.clone(),
                DecodedCacheItem {
                    mtime: current_mtime,
                    image: img.to_rgba8(),
                },
            );
        }
    }

    // Retrieve verified cached image pointer
    let cached_item = cache.get(&path).ok_or("Failed to retrieve cached item")?;
    let rgba = &cached_item.image;

    let out_img = if channel == "RGB" || channel == "Composite" {
        if crate::perceptual::is_vfx_transparent_texture(rgba) {
            image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        } else {
            image::DynamicImage::ImageRgba8(rgba.clone())
        }
    } else {
        let channel_idx = match channel.as_str() {
            "R" => 0,
            "G" => 1,
            "B" => 2,
            _ => 3, // A
        };

        let mut out_rgb = image::RgbImage::new(rgba.width(), rgba.height());
        for (x, y, pixel) in rgba.enumerate_pixels() {
            let val = pixel[channel_idx];
            out_rgb.put_pixel(x, y, image::Rgb([val, val, val]));
        }
        image::DynamicImage::ImageRgb8(out_rgb)
    };

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);

    out_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;

    let b64 = general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{}", b64))
}

/// Command: Generates structured visual collage reports for duplicate groups.
#[tauri::command]
async fn generate_visualization_report(
    groups: Vec<DuplicateGroupSummary>,
) -> Result<String, String> {
    if groups.is_empty() {
        return Err("No duplicate groups to visualize".into());
    }

    let app_dir = get_portable_app_data_dir()?;
    let visuals_dir = app_dir.join("duplicate_visuals");
    fs::create_dir_all(&visuals_dir).map_err(|e| e.to_string())?;

    let card_size = 180usize;
    let padding = 12usize;
    let text_height = 40usize;
    let columns = 4usize;

    for (i, group) in groups.iter().enumerate() {
        let file_count = group.files.len();
        if file_count == 0 {
            continue;
        }

        let rows = (file_count + columns - 1) / columns;
        let canvas_w = columns * (card_size + padding) + padding;
        let canvas_h = rows * (card_size + text_height + padding) + padding + 20;

        let mut canvas = image::RgbaImage::from_pixel(
            canvas_w as u32,
            canvas_h as u32,
            image::Rgba([45, 45, 45, 255]),
        );

        for (j, file_summary) in group.files.iter().enumerate() {
            let col = j % columns;
            let row = j / columns;

            let x_offset = padding + col * (card_size + padding);
            let y_offset = padding + row * (card_size + text_height + padding);

            let p = PathBuf::from(&file_summary.path);
            if let Ok(img) = dds_loader::open_image_with_dds_fallback(&p) {
                let resized = img.resize(
                    card_size as u32,
                    card_size as u32,
                    image::imageops::FilterType::Triangle,
                );
                let rgba_resized = resized.to_rgba8();

                let inner_x = x_offset + (card_size - rgba_resized.width() as usize) / 2;
                let inner_y = y_offset + (card_size - rgba_resized.height() as usize) / 2;

                image::imageops::overlay(
                    &mut canvas,
                    &rgba_resized,
                    inner_x as i64,
                    inner_y as i64,
                );
            }
        }

        let report_path = visuals_dir.join(format!("group_{:03}.png", i + 1));
        canvas.save(&report_path).map_err(|e| e.to_string())?;
    }

    Ok(visuals_dir.to_string_lossy().to_string())
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FileMetadata {
    path: PathBuf,
    size: u64,
    width: usize,
    height: usize,
    hash: String,
}

fn discover_files(root: &Path) -> Vec<PathBuf> {
    let supported_extensions = [
        "png", "jpg", "jpeg", "tga", "dds", "exr", "hdr", "tif", "tiff",
    ];
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            if let Some(ext) = p.extension() {
                supported_extensions.contains(&ext.to_string_lossy().to_lowercase().as_str())
            } else {
                false
            }
        })
        .collect()
}

fn calculate_xxhash(path: &Path) -> Result<String, std::io::Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Xxh64::new(0);
    let mut buffer = vec![0; 8 * 1024 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.digest()))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            download_models,
            run_exact_scan,
            run_qc_scan,
            run_folder_compare,
            run_perceptual_scan,
            run_ai_duplicate_scan,
            run_ai_search,
            run_image_search,
            test_tonemap,
            calculate_diff,
            delete_files,
            create_hardlinks,
            create_reflinks,
            get_channel_preview,
            generate_visualization_report
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
