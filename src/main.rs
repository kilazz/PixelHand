mod database;
mod dds_loader;
mod downloader;
mod inference;
mod perceptual;
mod qc;
mod tonemapper;

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::Read; // Brings standard read() trait method into scope for fs::File
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use xxhash_rust::xxh64::Xxh64;

// Slint UI Runtime Imports
use slint::winit_030::WinitWindowAccessor;
use slint::{ComponentHandle, ModelRc, SharedPixelBuffer, VecModel};

// Include Slint generated module
slint::include_modules!();

static APP_HANDLE: OnceLock<slint::Weak<AppWindow>> = OnceLock::new();

#[derive(Default)]
struct AppState {
    results: Vec<ResultsRowData>,
    groups: Vec<DuplicateGroupSummary>,
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
pub struct AiSearchResultSummary {
    pub path: String,
    pub similarity: f32,
}

#[derive(Clone)]
pub struct QcScanOptions {
    pub check_npot: bool,
    pub check_mipmaps: bool,
    pub check_block_align: bool,
    pub check_bit_depth: bool,
    pub validate_normals: bool,
    pub normals_tags: String,
    pub check_solid: bool,
}

#[derive(Clone)]
pub struct FolderCompareOptions {
    pub check_size_bloat: bool,
    pub check_alpha: bool,
    pub check_color_space: bool,
    pub check_compression: bool,
    pub match_by_stem: bool,
}

/// A 100% thread-safe (Send + Sync) shadow representation of Slint's ResultsRow
#[derive(Clone)]
pub struct ResultsRowData {
    pub is_header: bool,
    pub is_qc: bool,
    pub is_ai: bool,
    pub group_index: i32,
    pub hash_or_issue: String,
    pub path: String,
    pub name: String,
    pub score_or_detail: String,
    pub size_str: String,
    pub meta_str: String,
    pub is_best: bool,
    pub is_checked: bool,
    pub thumbnail_data: Option<image::RgbaImage>, // Thread-safe raw pixels
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

/// Converts an image::RgbaImage into a native slint::Image directly in memory.
fn convert_to_slint_image(rgba_img: &image::RgbaImage) -> slint::Image {
    let buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        rgba_img.as_raw(),
        rgba_img.width(),
        rgba_img.height(),
    );
    slint::Image::from_rgba8(buffer)
}

/// Maps thread-safe ResultsRowData into UI-bound Slint ResultsRow on the main thread
fn convert_to_slint_row(rd: &ResultsRowData) -> ResultsRow {
    let thumbnail = match &rd.thumbnail_data {
        Some(rgba) => convert_to_slint_image(rgba),
        None => slint::Image::default(),
    };
    ResultsRow {
        is_header: rd.is_header,
        is_qc: rd.is_qc,
        is_ai: rd.is_ai,
        group_index: rd.group_index,
        hash_or_issue: rd.hash_or_issue.clone().into(),
        path: rd.path.clone().into(),
        name: rd.name.clone().into(),
        score_or_detail: rd.score_or_detail.clone().into(),
        size_str: rd.size_str.clone().into(),
        meta_str: rd.meta_str.clone().into(),
        is_best: rd.is_best,
        is_checked: rd.is_checked,
        thumbnail,
    }
}

/// Utility helper to load and scale down image thumbnail. Returns thread-safe Option<RgbaImage>.
fn load_thumbnail_for_path(path_str: &str) -> Option<image::RgbaImage> {
    let path = PathBuf::from(path_str);
    if let Ok(mut img) = dds_loader::open_image_with_dds_fallback(&path) {
        if img.width() > 64 || img.height() > 64 {
            img = img.thumbnail(64, 64);
        }
        return Some(img.to_rgba8());
    }
    None
}

/// Helper function to format bytes to human-readable size.
fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".to_string();
    }
    let k = 1024.0;
    let sizes = ["B", "KB", "MB", "GB"];
    let i = (bytes as f64).log(k).floor() as usize;
    format!("{:.2} {}", bytes as f64 / k.powi(i as i32), sizes[i])
}

/// Maps duplicate groups to a flat representation for the Slint results model.
fn map_groups_to_rows(groups: &[DuplicateGroupSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for (g_idx, group) in groups.iter().enumerate() {
        if group.files.is_empty() {
            continue;
        }

        // Add Group Header Row
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: false,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: group.hash.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),
            size_str: format_size(group.files.first().map(|f| f.size).unwrap_or(0)),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
        });

        // Add Child Duplicate File Rows
        for (f_idx, file) in group.files.iter().enumerate() {
            let is_best = f_idx == 0;
            // Background load thumbnail to keep execution extremely fast and asynchronous
            let thumbnail_data = load_thumbnail_for_path(&file.path);
            rows.push(ResultsRowData {
                is_header: false,
                is_qc: false,
                is_ai: false,
                group_index: g_idx as i32,
                hash_or_issue: String::new(),
                path: file.path.clone(),
                name: Path::new(&file.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                score_or_detail: if is_best {
                    "[Best]".to_string()
                } else {
                    format!("{:.1}%", file.similarity)
                },
                size_str: format_size(file.size),
                meta_str: format!(
                    "{}x{} • {} • {}b",
                    file.width, file.height, file.compression_format, file.bit_depth
                ),
                is_best,
                is_checked: false,
                thumbnail_data,
            });
        }
    }
    rows
}

/// Maps Quality Control issues list to results row representations.
fn map_qc_to_rows(issues: &[QcIssueSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for issue in issues {
        let thumbnail_data = load_thumbnail_for_path(&issue.path);
        rows.push(ResultsRowData {
            is_header: false,
            is_qc: true,
            is_ai: false,
            group_index: -1,
            hash_or_issue: issue.issue.clone(),
            path: issue.path.clone(),
            name: Path::new(&issue.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            score_or_detail: issue.issue.clone(),
            size_str: String::new(),
            meta_str: issue.details.clone(),
            is_best: false,
            is_checked: false,
            thumbnail_data,
        });
    }
    rows
}

/// Maps AI Search results list to results row representations.
fn map_ai_search_to_rows(ai_results: &[AiSearchResultSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for res in ai_results {
        let thumbnail_data = load_thumbnail_for_path(&res.path);
        rows.push(ResultsRowData {
            is_header: false,
            is_qc: false,
            is_ai: true,
            group_index: -1,
            hash_or_issue: String::new(),
            path: res.path.clone(),
            name: Path::new(&res.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            score_or_detail: format!("{:.1}%", res.similarity),
            size_str: String::new(),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data,
        });
    }
    rows
}

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

/// Cache item for storing decoded images in memory.
struct DecodedCacheItem {
    mtime: std::time::SystemTime,
    image: image::RgbaImage,
}

// Thread-safe global cache of decoded preview images.
static DECODED_CACHE: OnceLock<Mutex<HashMap<String, DecodedCacheItem>>> = OnceLock::new();

/// Extracts a chosen pixel channel in grayscale to display in the comparative viewport.
async fn get_channel_preview_image(path: &str, channel: &str) -> Option<image::RgbaImage> {
    let p = PathBuf::from(path);
    if !p.is_file() {
        return None;
    }

    let current_mtime = fs::metadata(&p)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let cache_mutex = DECODED_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache_mutex.lock().ok()?;

    let has_valid_cache = cache
        .get(path)
        .map(|item| item.mtime == current_mtime)
        .unwrap_or(false);

    if !has_valid_cache {
        let mut img = dds_loader::open_image_with_dds_fallback(&p).ok()?;
        if img.width() > 512 || img.height() > 512 {
            img = img.thumbnail(512, 512);
        }
        let rgba = img.to_rgba8();
        cache.insert(
            path.to_string(),
            DecodedCacheItem {
                mtime: current_mtime,
                image: rgba,
            },
        );

        if cache.len() > 16 {
            cache.clear();
            let mut img = dds_loader::open_image_with_dds_fallback(&p).ok()?;
            if img.width() > 512 || img.height() > 512 {
                img = img.thumbnail(512, 512);
            }
            cache.insert(
                path.to_string(),
                DecodedCacheItem {
                    mtime: current_mtime,
                    image: img.to_rgba8(),
                },
            );
        }
    }

    let cached_item = cache.get(path)?;
    let rgba = &cached_item.image;

    let out_img = if channel == "RGB" || channel == "Composite" {
        if crate::perceptual::is_vfx_transparent_texture(rgba) {
            image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        } else {
            image::DynamicImage::ImageRgba8(rgba.clone())
        }
    } else {
        let channel_idx = match channel {
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

    Some(out_img.to_rgba8())
}

/// Helper method to translate Slint checkboxes state into channel flags.
fn get_current_active_channel(ui: &AppWindow) -> &'static str {
    if ui.get_active_r() {
        "R"
    } else if ui.get_active_g() {
        "G"
    } else if ui.get_active_b() {
        "B"
    } else if ui.get_active_a() {
        "A"
    } else {
        "Composite"
    }
}

/// Direct scan implementation for Exact (xxHash64) duplicate matching.
async fn run_exact_scan(
    directory: String,
    extensions: Vec<String>,
) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path, &extensions);
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

/// Absolute technical quality control auditor scan execution.
async fn run_qc_scan_internal(
    directory: String,
    options: QcScanOptions,
    extensions: Vec<String>,
) -> Result<Vec<QcIssueSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path, &extensions);

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
                options.check_npot,
                options.check_block_align,
                options.check_mipmaps,
                options.check_bit_depth,
            );
            for issue in abs_issues {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details: String::new(),
                });
            }

            if options.check_solid
                && let Some((issue, details)) = qc::check_solid_texture(p)
            {
                file_issues.push(QcIssueSummary {
                    path: p.to_string_lossy().to_string(),
                    issue,
                    details,
                });
            }

            if options.validate_normals {
                let path_str = p.to_string_lossy().to_lowercase();
                let should_check = if options.normals_tags.is_empty() {
                    true
                } else {
                    options
                        .normals_tags
                        .split(',')
                        .map(|t| t.trim().to_lowercase())
                        .filter(|t| !t.is_empty())
                        .any(|t| path_str.contains(&t))
                };
                if should_check
                    && let Some((issue, details)) = qc::check_normal_map_integrity(p, 0.15)
                {
                    file_issues.push(QcIssueSummary {
                        path: p.to_string_lossy().to_string(),
                        issue,
                        details,
                    });
                }
            }
            file_issues
        })
        .collect();
    Ok(issues)
}

/// Compares assets between two folders to discover relative QC modifications.
async fn run_folder_compare(
    directory_a: String,
    directory_b: String,
    options: FolderCompareOptions,
    extensions: Vec<String>,
) -> Result<Vec<QcIssueSummary>, String> {
    let path_a = PathBuf::from(directory_a);
    let path_b = PathBuf::from(directory_b);
    if !path_a.is_dir() || !path_b.is_dir() {
        return Err("Both target paths must be valid directories".into());
    }

    let files_a = discover_files(&path_a, &extensions);
    let files_b = discover_files(&path_b, &extensions);

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
        map_a.insert(get_key(p, options.match_by_stem), p.clone());
    }

    let mut issues = Vec::new();

    for p_b in &files_b {
        let key = get_key(p_b, options.match_by_stem);
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
                options.check_size_bloat,
                options.check_alpha,
                options.check_color_space,
                options.check_compression,
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

/// Advanced visual hashing (pHash and dHash).
async fn run_perceptual_scan_internal(
    directory: String,
    threshold: u32,
    analysis_type: String,
    ignore_solid_channels: bool,
    use_phash: bool,
    extensions: Vec<String>,
) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path, &extensions);

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
            if let Some(dist) = perceptual::calculate_hamming_distance(&hashes[i].1, &hashes[j].1)
                && dist <= max_dist
            {
                group_members.push(hashes[j].clone());
                visited[j] = true;
            }
        }
        if group_members.len() > 1 {
            group_id += 1;
            let mut file_summaries = Vec::new();
            for (p_buf, _hash_str) in &group_members {
                let metadata = fs::metadata(p_buf).map_err(|e| e.to_string())?;
                let size = metadata.len();
                let (width, height) = match imagesize::size(p_buf) {
                    Ok(dim) => (dim.width, dim.height),
                    Err(_) => (0, 0),
                };
                let qc_meta =
                    qc::extract_qc_metadata(p_buf).unwrap_or_else(|_| qc::QcImageMetadata {
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

            let best_path = &file_summaries[0].path;
            let best_hash = group_members
                .iter()
                .find(|(pb, _)| pb.to_string_lossy() == *best_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or(&group_members[0].1);

            for file in &mut file_summaries {
                let cur_hash = group_members
                    .iter()
                    .find(|(pb, _)| pb.to_string_lossy() == file.path)
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

/// Similar duplicates matching using ONNX embeddings.
async fn run_ai_duplicate_scan(
    directory: String,
    threshold: f32,
    execution_provider: String,
    extensions: Vec<String>,
    batch_size: Option<usize>,
) -> Result<Vec<DuplicateGroupSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path, &extensions);
    let app_dir = get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = database::DatabaseService::new();
    db.initialize(&db_dir, "images", 512)
        .await
        .map_err(|e| e.to_string())?;
    let engine = inference::InferenceEngine::new(&model_dir, 512, 4, &execution_provider)
        .map_err(|e| e.to_string())?;

    let batch_sz = batch_size.unwrap_or(128);
    let mut records: Vec<database::DbRecord> = Vec::new();
    let chunks: Vec<&[PathBuf]> = paths.chunks(batch_sz).collect();

    for chunk in chunks {
        let loaded_images: Vec<(PathBuf, image::DynamicImage)> = chunk
            .par_iter()
            .filter_map(|p| {
                let img = dds_loader::open_image_with_dds_fallback(p).ok()?;
                Some((p.clone(), img))
            })
            .collect();

        let (chunk_paths, imgs): (Vec<PathBuf>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();

        let mut chunk_records: Vec<database::DbRecord> = Vec::new();
        if !imgs.is_empty()
            && let Ok(vectors) =
                engine.encode_images_batch(&imgs, &inference::PreprocessingConfig::default())
        {
            for (path, vector) in chunk_paths.into_iter().zip(vectors) {
                let id = uuid::Uuid::new_v4().to_string();
                chunk_records.push(database::DbRecord {
                    id,
                    vector,
                    path: path.to_string_lossy().to_string(),
                    channel: "Composite".to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

    db.add_batch(&records).await.map_err(|e| e.to_string())?;
    db.create_vector_index().await.map_err(|e| e.to_string())?;

    let dist_threshold = 1.0 - (threshold / 100.0);
    let mut uf = UnionFind::new();

    {
        use futures::StreamExt;

        let query_items: Vec<(String, Vec<f32>)> = records
            .iter()
            .map(|r| (r.path.clone(), r.vector.clone()))
            .collect();

        let mut query_stream = futures::stream::iter(query_items)
            .map(|(r_path, query_vec)| {
                let db_clone = db.clone();
                async move {
                    let matches = db_clone.search_similarity(&query_vec, 10).await?;
                    Ok::<_, Box<dyn std::error::Error>>((r_path, matches))
                }
            })
            .buffer_unordered(16);

        while let Some(res) = query_stream.next().await {
            let (r_path, matches) = res.map_err(|e| e.to_string())?;
            for m in matches {
                if m.path != r_path && m.distance <= dist_threshold {
                    uf.union(&r_path, &m.path);
                }
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

/// Generates a comparison difference map image.
async fn calculate_diff_internal(file1: &str, file2: &str) -> Result<String, String> {
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

/// Moves chosen paths safely to the system Trash/Recycle Bin.
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

/// Converts chosen duplicate paths into hardlinks to the source file.
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

/// Converts chosen duplicate paths into reflinks (Copy-on-Write fallback to hardlink).
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

fn discover_files(root: &Path, extensions: &[String]) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            if let Some(ext) = p.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                let lower_ext = if ext_str.starts_with('.') {
                    ext_str.clone()
                } else {
                    format!(".{}", ext_str)
                };
                extensions.contains(&lower_ext) || extensions.contains(&ext_str)
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

#[tokio::main]
async fn main() -> Result<(), slint::PlatformError> {
    let args: Vec<String> = std::env::args().collect();
    if args
        .iter()
        .any(|arg| arg == "--cli" || arg == "-c" || arg == "--help" || arg == "-h")
    {
        println!("==================================================");
        println!("           PixelHandRust CLI Mode                 ");
        println!("==================================================");

        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            println!("Usage:");
            println!("  pixelhand -c --scan-exact <directory_path>");
            println!("  pixelhand -c --scan-qc <directory_path> [options]");
            println!("\nQC Options:");
            println!("  --check-npot          Verify if dimensions are Non-Power-of-Two");
            println!("  --check-mipmaps       Verify if mipmaps are generated");
            println!("  --check-block         Verify if dimensions are 4px block aligned");
            println!("  --check-bit           Verify bit depths");
            println!("  --validate-normals    Validate typical normal maps format");
            return Ok(());
        }

        if let Some(pos) = args.iter().position(|arg| arg == "--scan-exact") {
            if pos + 1 < args.len() {
                let dir = args[pos + 1].clone();
                println!("[CLI] Running Byte-Exact Scan (xxHash64) on: {}\n", dir);
                let default_exts = vec![
                    ".png".to_string(),
                    ".jpg".to_string(),
                    ".jpeg".to_string(),
                    ".tga".to_string(),
                    ".dds".to_string(),
                    ".exr".to_string(),
                    ".hdr".to_string(),
                    ".tif".to_string(),
                    ".tiff".to_string(),
                ];
                match run_exact_scan(dir, default_exts).await {
                    Ok(results) => {
                        println!(
                            "[SUCCESS] Exact Scan Completed! Found {} duplicate groups:",
                            results.len()
                        );
                        for (idx, group) in results.iter().enumerate() {
                            println!("  Group #{} (Hash: {})", idx + 1, group.hash);
                            for file in &group.files {
                                println!(
                                    "    - {} (Size: {} bytes, Dim: {}x{})",
                                    file.path, file.size, file.width, file.height
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[ERROR] Exact Scan Failed: {}", e);
                    }
                }
            } else {
                eprintln!("[ERROR] Missing directory path for --scan-exact");
            }
            return Ok(());
        }

        if let Some(pos) = args.iter().position(|arg| arg == "--scan-qc") {
            if pos + 1 < args.len() {
                let dir = args[pos + 1].clone();
                let check_npot = args.iter().any(|arg| arg == "--check-npot");
                let check_mipmaps = args.iter().any(|arg| arg == "--check-mipmaps");
                let check_block = args.iter().any(|arg| arg == "--check-block");
                let check_bit = args.iter().any(|arg| arg == "--check-bit");
                let validate_normals = args.iter().any(|arg| arg == "--validate-normals");

                println!("[CLI] Running Technical Quality Control Scan on: {}", dir);
                println!(
                    "      Options: NPOT={}, Mipmaps={}, BlockAlign={}, BitDepth={}, Normals={}\n",
                    check_npot, check_mipmaps, check_block, check_bit, validate_normals
                );

                let default_exts = vec![
                    ".png".to_string(),
                    ".jpg".to_string(),
                    ".jpeg".to_string(),
                    ".tga".to_string(),
                    ".dds".to_string(),
                    ".exr".to_string(),
                    ".hdr".to_string(),
                    ".tif".to_string(),
                    ".tiff".to_string(),
                ];

                let qc_opts = QcScanOptions {
                    check_npot,
                    check_mipmaps,
                    check_block_align: check_block,
                    check_bit_depth: check_bit,
                    validate_normals,
                    normals_tags: "".to_string(),
                    check_solid: false,
                };

                match run_qc_scan_internal(dir, qc_opts, default_exts).await {
                    Ok(results) => {
                        println!(
                            "[SUCCESS] QC Scan Completed! Found {} issues:",
                            results.len()
                        );
                        for issue in results {
                            println!("  - File: {}", issue.path);
                            println!("    Issue: {}", issue.issue);
                            println!("    Details: {}", issue.details);
                        }
                    }
                    Err(e) => {
                        eprintln!("[ERROR] QC Scan Failed: {}", e);
                    }
                }
            } else {
                eprintln!("[ERROR] Missing directory path for --scan-qc");
            }
            return Ok(());
        }

        println!("[ERROR] Unknown CLI arguments. Use -h or --help for instructions.");
        return Ok(());
    }

    // Start logging subscriber
    tracing_subscriber::fmt::init();

    // Prevent ONNX Runtime from printing massive C++ execution provider logs (CPU fallbacks)
    unsafe {
        std::env::set_var("ORT_LOGGING_LEVEL", "WARNING");
        std::env::set_var("ORT_LOG_LEVEL", "WARNING");
    }

    // Thread-safe state holder for list updates
    let state = Arc::new(Mutex::new(AppState::default()));

    // Instantiate Slint AppWindow
    let app = AppWindow::new()?;
    let _ = APP_HANDLE.set(app.as_weak());

    // Auto-download models on startup!
    let app_weak_startup = app.as_weak();
    tokio::spawn(async move {
        let app_copy = app_weak_startup.clone();
        let app_dir = match get_portable_app_data_dir() {
            Ok(dir) => dir,
            Err(e) => {
                let _ = app_copy.upgrade_in_event_loop(move |ui| {
                    ui.set_status_text(format!("Failed to locate app dir: {}", e).into());
                });
                return;
            }
        };
        let model_dir = app_dir
            .join("models")
            .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");
        let _ = fs::create_dir_all(&model_dir);

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
                let app_copy_progress = app_copy.clone();
                let file_name = name.to_string();
                let res = downloader::download_file_with_progress(
                    move |percentage| {
                        let f_name = file_name.clone();
                        let _ = app_copy_progress.upgrade_in_event_loop(move |ui| {
                            ui.set_progress(percentage / 100.0);
                            ui.set_status_text(
                                format!("Downloading {}: {:.1}%", f_name, percentage).into(),
                            );
                        });
                    },
                    url,
                    &dest,
                    name,
                )
                .await;

                if let Err(e) = res {
                    let error_message = format!("Failed to download {}: {}", name, e);
                    let _ = app_copy.upgrade_in_event_loop(move |ui| {
                        ui.set_status_text(error_message.into());
                    });
                    return;
                }
            }
        }

        let _ = app_copy.upgrade_in_event_loop(move |ui| {
            ui.set_status_text("AI models verified. System ready.".into());
            ui.set_progress(1.0);
        });
    });

    // 1. Browse Folder A Callback
    let app_weak = app.as_weak();
    app.on_select_folder_a(move || {
        let app_copy = app_weak.clone();
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder A")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_dir_a(path_str.into());
            });
        }
    });

    // 2. Browse Folder B Callback
    let app_weak = app.as_weak();
    app.on_select_folder_b(move || {
        let app_copy = app_weak.clone();
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Select Folder B")
            .pick_folder()
        {
            let path_str = folder.to_string_lossy().to_string();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_dir_b(path_str.into());
            });
        }
    });

    // 3. Scan Runner Callback
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_run_scan(move || {
        let app_copy = app_weak.clone();
        let state_copy = state_clone.clone();

        let ui = app_copy.unwrap();
        let dir_a_str = ui.get_dir_a().to_string();
        let dir_b_str = ui.get_dir_b().to_string();
        let query_text_str = ui.get_query_text().to_string();
        let similarity = ui.get_similarity_threshold();
        let batch_sz = ui.get_batch_size() as usize;

        let qc_mode_active = ui.get_qc_mode();
        let check_npot = ui.get_qc_npot();
        let check_mipmaps = ui.get_qc_mipmaps();
        let check_block = ui.get_qc_block_align();
        let check_bit = ui.get_qc_bit_depth();
        let check_solid = ui.get_qc_solid_colors();
        let validate_normals = ui.get_qc_normals();
        let normals_tags = ui.get_qc_normals_tags().to_string();
        let method = ui.get_search_method();

        let default_exts = vec![
            ".png".to_string(),
            ".jpg".to_string(),
            ".jpeg".to_string(),
            ".tga".to_string(),
            ".dds".to_string(),
            ".exr".to_string(),
            ".hdr".to_string(),
            ".tif".to_string(),
            ".tiff".to_string(),
            ".webp".to_string(),
        ];

        ui.set_is_scanning(true);
        ui.set_status_text("Initializing graphical scan...".into());

        tokio::spawn(async move {
            let scan_result = if qc_mode_active {
                if !dir_b_str.trim().is_empty() {
                    let compare_opts = FolderCompareOptions {
                        check_size_bloat: true,
                        check_alpha: true,
                        check_color_space: true,
                        check_compression: true,
                        match_by_stem: true,
                    };
                    let res =
                        run_folder_compare(dir_a_str, dir_b_str, compare_opts, default_exts).await;
                    match res {
                        Ok(issues) => Ok((Vec::new(), issues, Vec::new())),
                        Err(e) => Err(e),
                    }
                } else {
                    let qc_opts = QcScanOptions {
                        check_npot,
                        check_mipmaps,
                        check_block_align: check_block,
                        check_bit_depth: check_bit,
                        validate_normals,
                        normals_tags,
                        check_solid,
                    };
                    let res = run_qc_scan_internal(dir_a_str, qc_opts, default_exts).await;
                    match res {
                        Ok(issues) => Ok((Vec::new(), issues, Vec::new())),
                        Err(e) => Err(e),
                    }
                }
            } else if method == 2 {
                // AI search
                if !query_text_str.trim().is_empty() {
                    let res = run_ai_search(
                        dir_a_str,
                        query_text_str,
                        "CPU".to_string(),
                        default_exts,
                        Some(batch_sz),
                    )
                    .await;
                    match res {
                        Ok(results) => Ok((Vec::new(), Vec::new(), results)),
                        Err(e) => Err(e),
                    }
                } else {
                    let res = run_ai_duplicate_scan(
                        dir_a_str,
                        similarity,
                        "CPU".to_string(),
                        default_exts,
                        Some(batch_sz),
                    )
                    .await;
                    match res {
                        Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                        Err(e) => Err(e),
                    }
                }
            } else if method == 1 {
                // Simple perceptual
                let res = run_perceptual_scan_internal(
                    dir_a_str,
                    similarity as u32,
                    "Composite".to_string(),
                    true,
                    false,
                    default_exts,
                )
                .await;
                match res {
                    Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                    Err(e) => Err(e),
                }
            } else {
                // Exact matching
                let res = run_exact_scan(dir_a_str, default_exts).await;
                match res {
                    Ok(groups) => Ok((groups, Vec::new(), Vec::new())),
                    Err(e) => Err(e),
                }
            };

            // Build dynamic row objects in background task to prevent event-loop freezing
            let rows = match &scan_result {
                Ok((groups, qc_issues, ai_results)) => {
                    if !qc_issues.is_empty() {
                        map_qc_to_rows(qc_issues)
                    } else if !ai_results.is_empty() {
                        map_ai_search_to_rows(ai_results)
                    } else {
                        map_groups_to_rows(groups)
                    }
                }
                Err(_) => Vec::new(),
            };

            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_is_scanning(false);
                match scan_result {
                    Ok((groups, _, _)) => {
                        let mut state_lock = state_copy.lock().unwrap();
                        state_lock.groups = groups.clone();
                        state_lock.results = rows.clone();

                        let slint_rows: Vec<ResultsRow> =
                            rows.iter().map(convert_to_slint_row).collect();
                        let results_model =
                            ModelRc::from(std::rc::Rc::new(VecModel::from(slint_rows)));
                        ui.set_results(results_model);
                        ui.set_status_text("Scan finished successfully!".into());
                    }
                    Err(e) => {
                        ui.set_status_text(format!("Scan failed: {}", e).into());
                    }
                }
            });
        });
    });

    // 4. File Table Checkbox Toggled Callback (Synchronous UI callback execution thread)
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_row_checkbox_toggled(move |idx| {
        let mut lock = state_clone.lock().unwrap();
        if let Some(row) = lock.results.get_mut(idx as usize) {
            row.is_checked = !row.is_checked;
        }
        if let Some(ui) = app_weak.upgrade() {
            let slint_rows: Vec<ResultsRow> =
                lock.results.iter().map(convert_to_slint_row).collect();
            let model = ModelRc::from(std::rc::Rc::new(VecModel::from(slint_rows)));
            ui.set_results(model);
        }
    });

    // 5. File Table Row Click Selection Callback
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_row_clicked(move |_, is_header, group_idx, path| {
        if is_header {
            return;
        }
        let app_copy = app_weak.clone();
        let path_str = path.to_string();

        let lock = state_clone.lock().unwrap();
        let group = match lock.groups.get(group_idx as usize) {
            Some(g) => g,
            None => return,
        };

        let original = match group.files.first() {
            Some(f) => f,
            None => return,
        };

        let duplicate = match group.files.iter().find(|f| f.path == path_str) {
            Some(f) => f,
            None => return,
        };

        let o_meta = SelectedFile {
            name: Path::new(&original.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
                .into(),
            size_str: format_size(original.size).into(),
            format: original.compression_format.clone().into(),
            resolution: format!("{}x{}", original.width, original.height).into(),
            bit_depth: format!("{}-bit", original.bit_depth).into(),
            color_space: original.color_space.clone().into(),
            mipmaps: original.mipmap_count.to_string().into(),
            alpha: (if original.has_alpha { "Yes" } else { "No" }).into(),
            similarity: "-".into(),
            path: original.path.clone().into(),
        };

        let d_meta = SelectedFile {
            name: Path::new(&duplicate.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
                .into(),
            size_str: format_size(duplicate.size).into(),
            format: duplicate.compression_format.clone().into(),
            resolution: format!("{}x{}", duplicate.width, duplicate.height).into(),
            bit_depth: format!("{}-bit", duplicate.bit_depth).into(),
            color_space: duplicate.color_space.clone().into(),
            mipmaps: duplicate.mipmap_count.to_string().into(),
            alpha: (if duplicate.has_alpha { "Yes" } else { "No" }).into(),
            similarity: format!("{:.1}%", duplicate.similarity).into(),
            path: duplicate.path.clone().into(),
        };

        let ui = app_copy.unwrap();
        ui.set_original_meta(o_meta);
        ui.set_duplicate_meta(d_meta);

        // Map duplicates list to Slint right-hand compare miniature ListView
        let mut group_files = Vec::new();
        for row in &lock.results {
            if !row.is_header && row.group_index == group_idx {
                group_files.push(convert_to_slint_row(row));
            }
        }
        let selected_model = ModelRc::from(std::rc::Rc::new(VecModel::from(group_files)));
        ui.set_selected_group_files(selected_model);

        let channel = get_current_active_channel(&ui).to_string();
        let compare_mode = ui.get_compare_mode();

        let orig_path = original.path.clone();
        let dup_path = duplicate.path.clone();
        let app_weak_clone = app_weak.clone();

        tokio::spawn(async move {
            if let Some(raw_orig) = get_channel_preview_image(&orig_path, &channel).await {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let slint_orig = convert_to_slint_image(&raw_orig);
                    ui.set_image_original(slint_orig);
                });
            }

            if let Some(raw_dup) = get_channel_preview_image(&dup_path, &channel).await {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let slint_dup = convert_to_slint_image(&raw_dup);
                    ui.set_image_duplicate(slint_dup);
                });
            }

            if compare_mode == 3
                && let Ok(diff_path) = calculate_diff_internal(&orig_path, &dup_path).await
                && let Ok(diff_img) = image::open(&diff_path)
            {
                let raw_diff = diff_img.to_rgba8();
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let slint_diff = convert_to_slint_image(&raw_diff);
                    ui.set_image_heatmap(slint_diff);
                });
            }
        });
    });

    // 6. Hardlink / Reflink / Trash Actions Handler Callback
    let app_weak = app.as_weak();
    let state_clone = state.clone();
    app.on_trigger_action(move |action_type| {
        let app_copy = app_weak.clone();
        let state_copy = state_clone.clone();
        let action = action_type.to_string();

        tokio::spawn(async move {
            let (checked_files, pairs) = {
                let lock = state_copy.lock().unwrap();
                let mut checked_files = Vec::new();
                let mut pairs = Vec::new();

                for row in &lock.results {
                    if !row.is_header && row.is_checked {
                        checked_files.push(row.path.clone());

                        // Find the original path inside the matching duplicate group
                        let group = lock.groups.get(row.group_index as usize);
                        if let Some(g) = group
                            && let Some(orig) = g.files.first()
                        {
                            pairs.push((orig.path.clone(), row.path.clone()));
                        }
                    }
                }
                (checked_files, pairs)
            };

            if checked_files.is_empty() {
                return;
            }

            // Clone the `action` string for the closure so it can still be used in the match
            let action_clone1 = action.clone();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_status_text(format!("Processing selection: {}...", action_clone1).into());
            });

            let res = match action.as_str() {
                "trash" => delete_files(checked_files).await,
                "hardlink" => create_hardlinks(pairs).await,
                "reflink" => create_reflinks(pairs).await,
                _ => Err("Unknown action type".to_string()),
            };

            let app_copy_final = app_copy.clone();
            match res {
                Ok(_) => {
                    let _ = app_copy.upgrade_in_event_loop(move |ui| {
                        ui.set_status_text(
                            format!("Successfully completed {} operation.", action).into(),
                        );
                        ui.set_results(ModelRc::from(std::rc::Rc::new(VecModel::from(Vec::new()))));
                    });

                    // Clear local memory buffers
                    let mut lock = state_copy.lock().unwrap();
                    lock.results.clear();
                    lock.groups.clear();
                }
                Err(e) => {
                    let error_msg = format!("Selection action failed: {}", e);
                    let _ = app_copy_final.upgrade_in_event_loop(move |ui| {
                        ui.set_status_text(error_msg.into());
                    });
                }
            }
        });
    });

    // 7. Channel Toggled Callback
    let app_weak = app.as_weak();
    app.on_channel_toggled(move || {
        let app_copy = app_weak.clone();
        let ui = app_copy.unwrap();

        let orig_path = ui.get_original_meta().path.to_string();
        let dup_path = ui.get_duplicate_meta().path.to_string();
        if orig_path.is_empty() || dup_path.is_empty() {
            return;
        }

        let channel = get_current_active_channel(&ui).to_string();
        let app_weak_clone = app_weak.clone();

        tokio::spawn(async move {
            // Load original
            if let Some(raw_orig) = get_channel_preview_image(&orig_path, &channel).await {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let slint_orig = convert_to_slint_image(&raw_orig);
                    ui.set_image_original(slint_orig);
                });
            }
            // Load duplicate
            if let Some(raw_dup) = get_channel_preview_image(&dup_path, &channel).await {
                let _ = app_weak_clone.upgrade_in_event_loop(move |ui| {
                    let slint_dup = convert_to_slint_image(&raw_dup);
                    ui.set_image_duplicate(slint_dup);
                });
            }
        });
    });

    // Register OS system-level Drag & Drop file handler to load image models instantly
    let app_weak_dnd = app.as_weak();
    app.window().on_winit_window_event(move |_window, event| {
        if let slint::winit_030::winit::event::WindowEvent::DroppedFile(path_buf) = event {
            let path_str = path_buf.to_string_lossy().to_string();
            let app_copy = app_weak_dnd.clone();
            let _ = app_copy.upgrade_in_event_loop(move |ui| {
                ui.set_query_text(path_str.clone().into());
                ui.set_search_method(2); // Auto switch to AI Visual Search
                ui.set_status_text(format!("Reference image loaded: {}", path_str).into());
            });
        }
        slint::winit_030::EventResult::Propagate
    });

    // Launch main UI event loop
    app.run()
}

/// Helper function required for AI semantic image search.
async fn run_ai_search(
    directory: String,
    query: String,
    execution_provider: String,
    extensions: Vec<String>,
    batch_size: Option<usize>,
) -> Result<Vec<AiSearchResultSummary>, String> {
    let path = PathBuf::from(directory);
    if !path.is_dir() {
        return Err("The specified path is not a valid directory".into());
    }
    let paths = discover_files(&path, &extensions);
    let app_dir = get_portable_app_data_dir()?;
    let db_dir = app_dir.join(".lancedb_cache");
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");

    let mut db = database::DatabaseService::new();
    db.initialize(&db_dir, "images", 512)
        .await
        .map_err(|e| e.to_string())?;
    let engine = inference::InferenceEngine::new(&model_dir, 512, 4, &execution_provider)
        .map_err(|e| e.to_string())?;

    let batch_sz = batch_size.unwrap_or(128);
    let mut records: Vec<database::DbRecord> = Vec::new();
    let chunks: Vec<&[PathBuf]> = paths.chunks(batch_sz).collect();

    for chunk in chunks {
        let loaded_images: Vec<(PathBuf, image::DynamicImage)> = chunk
            .par_iter()
            .filter_map(|p| {
                let img = dds_loader::open_image_with_dds_fallback(p).ok()?;
                Some((p.clone(), img))
            })
            .collect();

        let (chunk_paths, imgs): (Vec<PathBuf>, Vec<image::DynamicImage>) =
            loaded_images.into_iter().unzip();

        let mut chunk_records: Vec<database::DbRecord> = Vec::new();
        if !imgs.is_empty()
            && let Ok(vectors) =
                engine.encode_images_batch(&imgs, &inference::PreprocessingConfig::default())
        {
            for (path, vector) in chunk_paths.into_iter().zip(vectors) {
                let id = uuid::Uuid::new_v4().to_string();
                chunk_records.push(database::DbRecord {
                    id,
                    vector,
                    path: path.to_string_lossy().to_string(),
                    channel: "Composite".to_string(),
                });
            }
        }
        records.extend(chunk_records);
    }

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
