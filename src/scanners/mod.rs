// src/scanners/mod.rs

pub mod ai;
pub mod exact;
pub mod perceptual;
pub mod qc;

use anyhow::Result;
use moka::sync::Cache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use ustr::Ustr;
use xxhash_rust::xxh64::Xxh64;

use crate::core::perceptual::AnalysisType;
use crate::state::{DuplicateGroupSummary, ResultsRowData};

/// High-performance container holding composite and lazy-allocated channel-isolated preview buffers.
#[derive(Clone)]
pub struct CachedThumbnail {
    pub composite: image::RgbaImage,
    pub r_channel: Arc<OnceLock<image::RgbaImage>>,
    pub g_channel: Arc<OnceLock<image::RgbaImage>>,
    pub b_channel: Arc<OnceLock<image::RgbaImage>>,
    pub a_channel: Arc<OnceLock<image::RgbaImage>>,
}

impl CachedThumbnail {
    pub fn new(composite: image::RgbaImage) -> Self {
        Self {
            composite,
            r_channel: Arc::new(OnceLock::new()),
            g_channel: Arc::new(OnceLock::new()),
            b_channel: Arc::new(OnceLock::new()),
            a_channel: Arc::new(OnceLock::new()),
        }
    }

    /// Retrieves or dynamically computes a color-isolated preview on a separate thread, caching the result.
    pub fn get_channel(&self, channel: &str) -> image::RgbaImage {
        let lock = match channel {
            "R" => &self.r_channel,
            "G" => &self.g_channel,
            "B" => &self.b_channel,
            "A" => &self.a_channel,
            _ => return self.composite.clone(),
        };

        lock.get_or_init(|| {
            let idx = match channel {
                "R" => 0,
                "G" => 1,
                "B" => 2,
                _ => 3,
            };

            let width = self.composite.width();
            let height = self.composite.height();
            let mut out_rgba = image::RgbaImage::new(width, height);

            let composite_raw = self.composite.as_raw();
            let out_raw = out_rgba.as_mut();

            for (i, chunk) in composite_raw.chunks_exact(4).enumerate() {
                let val = chunk[idx];
                let dst_idx = i * 4;
                if idx == 3 {
                    out_raw[dst_idx] = val;
                    out_raw[dst_idx + 1] = val;
                    out_raw[dst_idx + 2] = val;
                    out_raw[dst_idx + 3] = val;
                } else {
                    out_raw[dst_idx] = val;
                    out_raw[dst_idx + 1] = val;
                    out_raw[dst_idx + 2] = val;
                    out_raw[dst_idx + 3] = 255;
                }
            }
            out_rgba
        })
        .clone()
    }
}

/// Fast thread-safe in-memory cache for loaded thumbnail textures to support zero-lag hover channel previews.
/// Key type migrated to Ustr to leverage O(1) comparison and pre-calculated pointer hashing.
pub static THUMBNAIL_MEMORY_CACHE: OnceLock<Cache<Ustr, CachedThumbnail>> = OnceLock::new();

pub static ENABLE_PREVIEWS: AtomicBool = AtomicBool::new(true);
pub static PREVIEW_QUALITY: AtomicI32 = AtomicI32::new(1); // 0: Fast, 1: Balanced, 2: High

/// Normalizes path representations across Windows and Unix platforms to guarantee absolute cache hit consistency.
/// Returns an interned Ustr for instant pointer-comparison (==) and zero heap allocations on hot paths.
pub fn normalize_path_key(path_str: &str) -> Ustr {
    let normalized: String = path_str
        .chars()
        .map(|c| {
            if c == '/' || c == '\\' {
                std::path::MAIN_SEPARATOR
            } else {
                c
            }
        })
        .collect();
    ustr::ustr(&normalized.to_lowercase())
}

/// Stores an image buffer in the memory cache, evicting the oldest items if total memory footprint exceeds 500 MB.
fn store_in_thumbnail_memory_cache(path: &str, img: image::RgbaImage) {
    let cache = THUMBNAIL_MEMORY_CACHE.get_or_init(|| {
        Cache::builder()
            // Limit to approximately 500 MB for thumbnail caches to prevent system memory bloat
            .max_capacity(500 * 1024 * 1024)
            // Weigh each item according to its actual decompressed pixel payload (Width * Height * 4 channels)
            .weigher(|_k, v: &CachedThumbnail| {
                let bytes = v.composite.width() as u64 * v.composite.height() as u64 * 4;
                bytes.min(u32::MAX as u64) as u32
            })
            .build()
    });

    let normalized_key = normalize_path_key(path);
    cache.insert(normalized_key, CachedThumbnail::new(img));
}

/// Data structure mapping all options gathered from the UI panel to orchestrate a scan.
#[derive(Clone)]
pub struct ScanParams {
    pub dir_a: String,
    pub dir_b: String,
    pub query_text: String,
    pub similarity: f32,
    pub batch_size: usize,
    pub search_method: i32, // 0: Exact (xxHash), 1: Perceptual (dHash), 2: AI, 3: Inventory, 4: QC
    pub execution_provider: String,
    pub qc_mode: bool,
    pub qc_npot: bool,
    pub qc_mipmaps: bool,
    pub qc_block_align: bool,
    pub qc_bit_depth: bool,
    pub qc_solid_colors: bool,
    pub qc_normals: bool,
    pub qc_normals_tags: String,
    pub extensions: Vec<String>,
    pub cancel_token: Arc<std::sync::atomic::AtomicBool>,

    // Contact Sheet configurations
    pub save_visuals: bool,
    pub visuals_columns: usize,
    pub visuals_max_count: usize,
    pub visuals_font_size: usize,
    pub visuals_scale: f32,

    // Preprocessing configurations
    pub prep_luminance: bool,
    pub prep_channels: bool,
    pub prep_r: bool,
    pub prep_g: bool,
    pub prep_b: bool,
    pub prep_a: bool,
    pub prep_tags: String,
    pub prep_ignore_solid: bool,

    // Exclude filters
    pub excluded_folders: String,
    pub qc_match_by_stem: bool,
    pub qc_hide_same_resolution: bool,

    // Relative Quality Control parameters
    pub qc_check_bloat: bool,
    pub qc_check_alpha: bool,
    pub qc_check_colorspace: bool,
    pub qc_check_compression: bool,

    pub search_precision: i32,
    pub ai_model: i32,

    #[allow(clippy::type_complexity)]
    pub on_progress: Option<Arc<dyn Fn(f32, usize, usize) + Send + Sync>>,

    pub custom_model_path: String,
    pub custom_model_arch: i32,
    pub custom_model_dim: i32,
}

impl ScanParams {
    /// Compiles Slint UI properties from the global Store singleton into thread-safe ScanParams.
    pub fn from_store(
        store: &crate::app::Store,
        cancel_token: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        let mut extensions = Vec::new();

        if store.get_ext_png() {
            extensions.push(".png".to_string());
        }
        if store.get_ext_jpg() {
            extensions.push(".jpg".to_string());
            extensions.push(".jpeg".to_string());
        }
        if store.get_ext_tga() {
            extensions.push(".tga".to_string());
        }
        if store.get_ext_dds() {
            extensions.push(".dds".to_string());
        }
        if store.get_ext_bmp() {
            extensions.push(".bmp".to_string());
        }
        if store.get_ext_exr() {
            extensions.push(".exr".to_string());
        }
        if store.get_ext_hdr() {
            extensions.push(".hdr".to_string());
        }
        if store.get_ext_tif() {
            extensions.push(".tif".to_string());
            extensions.push(".tiff".to_string());
        }
        if store.get_ext_webp() {
            extensions.push(".webp".to_string());
        }
        if store.get_ext_gif() {
            extensions.push(".gif".to_string());
        }
        if store.get_ext_psd() {
            extensions.push(".psd".to_string());
        }
        if store.get_ext_jxl() {
            extensions.push(".jxl".to_string());
        }
        if store.get_ext_heic() {
            extensions.push(".heic".to_string());
            extensions.push(".heif".to_string());
        }
        if store.get_ext_avif() {
            extensions.push(".avif".to_string());
        }

        let execution_provider = match store.get_execution_provider() {
            1 => "DirectML".to_string(),
            2 => "CUDA".to_string(),
            3 => "TensorRT".to_string(),
            4 => "CoreML".to_string(),
            _ => "CPU".to_string(),
        };

        Self {
            dir_a: store.get_dir_a().to_string(),
            dir_b: store.get_dir_b().to_string(),
            query_text: store.get_query_text().to_string(),
            similarity: store.get_similarity_threshold(),
            batch_size: store.get_batch_size() as usize,
            search_method: store.get_search_method(),
            execution_provider,

            qc_mode: store.get_search_method() == 4,
            qc_npot: store.get_qc_npot(),
            qc_mipmaps: store.get_qc_mipmaps(),
            qc_block_align: store.get_qc_block_align(),
            qc_bit_depth: store.get_qc_bit_depth(),
            qc_solid_colors: store.get_qc_solid_colors(),
            qc_normals: store.get_qc_normals(),
            qc_normals_tags: store.get_qc_normals_tags().to_string(),

            extensions,
            cancel_token,

            save_visuals: store.get_save_visuals(),
            visuals_columns: store.get_visuals_columns() as usize,
            visuals_max_count: store.get_visuals_max_count() as usize,
            visuals_font_size: store.get_visuals_font_size() as usize,
            visuals_scale: store.get_visuals_scale(),

            prep_luminance: store.get_prep_luminance(),
            prep_channels: store.get_prep_channels(),
            prep_r: store.get_prep_r(),
            prep_g: store.get_prep_g(),
            prep_b: store.get_prep_b(),
            prep_a: store.get_prep_a(),
            prep_tags: store.get_prep_tags().to_string(),
            prep_ignore_solid: store.get_prep_ignore_solid(),

            excluded_folders: store.get_excluded_folders().to_string(),
            qc_match_by_stem: store.get_qc_match_by_stem(),
            qc_hide_same_resolution: store.get_qc_hide_same_resolution(),

            qc_check_bloat: store.get_qc_check_bloat(),
            qc_check_alpha: store.get_qc_check_alpha(),
            qc_check_colorspace: store.get_qc_check_colorspace(),
            qc_check_compression: store.get_qc_check_compression(),

            search_precision: store.get_search_precision(),
            ai_model: store.get_ai_model(),

            custom_model_path: store.get_custom_model_path().to_string(),
            custom_model_arch: store.get_custom_model_arch(),
            custom_model_dim: store.get_custom_model_dim(),

            on_progress: None,
        }
    }
}

/// Represents a discrete analysis item mapped by the image channels splitter.
#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub path: PathBuf,
    pub analysis_type: AnalysisType,
}

/// Expands a flat list of file paths into distinct AnalysisItems based on pre-processing rules.
pub fn generate_analysis_items(paths: &[PathBuf], params: &ScanParams) -> Vec<AnalysisItem> {
    let mut items = Vec::new();

    let tags: Vec<String> = params
        .prep_tags
        .split(',')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    for path in paths {
        let path_str = path.to_string_lossy().to_lowercase();

        let matches_tags = if tags.is_empty() {
            true
        } else {
            tags.iter().any(|tag| path_str.contains(tag))
        };

        if params.prep_channels && matches_tags {
            if params.prep_r {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::R,
                });
            }
            if params.prep_g {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::G,
                });
            }
            if params.prep_b {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::B,
                });
            }
            if params.prep_a {
                items.push(AnalysisItem {
                    path: path.clone(),
                    analysis_type: AnalysisType::A,
                });
            }
        } else if params.prep_luminance {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Luminance,
            });
        } else {
            items.push(AnalysisItem {
                path: path.clone(),
                analysis_type: AnalysisType::Composite,
            });
        }
    }

    items
}

/// Central routing mechanism directing search configurations to proper execution engines.
pub async fn execute_scan(
    params: ScanParams,
) -> Result<(Vec<DuplicateGroupSummary>, Vec<ResultsRowData>)> {
    if params.qc_mode {
        if !params.dir_b.trim().is_empty() {
            let ex_folders: Vec<String> = params
                .excluded_folders
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();

            let issues = qc::run_folder_compare(
                params.dir_a.clone(),
                params.dir_b.clone(),
                params.extensions.clone(),
                params.qc_match_by_stem,
                params.qc_hide_same_resolution,
                ex_folders,
                params.qc_check_bloat,
                params.qc_check_alpha,
                params.qc_check_colorspace,
                params.qc_check_compression,
            )
            .await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        } else {
            let issues = qc::run_qc_scan_internal(params).await?;
            let rows = map_qc_to_rows(&issues);
            Ok((Vec::new(), rows))
        }
    } else if params.search_method == 3 {
        let mut rows = qc::run_asset_audit(params).await?;
        rows.sort_by(|a, b| a.path.cmp(&b.path));
        Ok((Vec::new(), rows))
    } else if params.search_method == 2 {
        if !params.query_text.trim().is_empty() {
            let matches = ai::run_ai_search(params).await?;
            let rows = map_ai_search_to_rows(&matches);
            Ok((Vec::new(), rows))
        } else {
            let groups = ai::run_ai_duplicate_scan(params).await?;
            let rows = map_groups_to_rows(&groups);
            Ok((groups, rows))
        }
    } else if params.search_method == 1 {
        let groups = perceptual::run_perceptual_scan_internal(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    } else {
        let groups = exact::run_exact_scan(params).await?;
        let rows = map_groups_to_rows(&groups);
        Ok((groups, rows))
    }
}

// ---------------------------------------------------------
// PRIVATE RESULT-MAPPING DATA HELPERS
// ---------------------------------------------------------

static CACHE_DIR_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Loads thumbnail textures from disk cache, generating compressed fallback files during misses.
pub(crate) fn load_thumbnail_for_path(path_str: &str) -> Option<image::RgbaImage> {
    if !ENABLE_PREVIEWS.load(Ordering::Relaxed) {
        return None;
    }

    let path = PathBuf::from(path_str);
    if !path.is_file() {
        return None;
    }

    let cache_dir = crate::utils::settings::get_portable_app_data_dir()
        .ok()
        .map(|p| p.join(".cache").join("thumbnails"));

    let cache_path = if let Some(ref dir) = cache_dir {
        CACHE_DIR_INITIALIZED.get_or_init(|| {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::error!("Failed to initialize thumbnail cache directory: {}", e);
            }
        });

        if let Ok(metadata) = std::fs::metadata(&path) {
            let size = metadata.len();
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);

            let mut hasher = Xxh64::new(0);
            hasher.update(path.to_string_lossy().as_bytes());
            hasher.update(&size.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());

            Some(dir.join(format!("{:016x}.png", hasher.digest())))
        } else {
            None
        }
    } else {
        None
    };

    // 1. Try reading the tiny pre-rendered PNG from the disk cache safely
    if let Some(ref cp) = cache_path
        && cp.is_file()
        && let Ok(img) = image::open(cp)
    {
        let rgba = img.to_rgba8();
        store_in_thumbnail_memory_cache(path_str, rgba.clone());
        return Some(rgba);
    }

    let quality = PREVIEW_QUALITY.load(Ordering::Relaxed);
    let target_size = match quality {
        0 => 64,
        1 => 128,
        _ => 256,
    };
    let filter = match quality {
        0 => image::imageops::FilterType::Nearest,
        1 => image::imageops::FilterType::Triangle,
        _ => image::imageops::FilterType::Lanczos3,
    };

    // 2. Fallback: Parse the actual texture and downscale
    if let Ok(mut img) =
        crate::format_loaders::dds_loader::open_image_with_dds_fallback(&path, Some(target_size))
    {
        if img.width() > target_size || img.height() > target_size {
            img = img.resize(target_size, target_size, filter);
        }
        let rgba = img.to_rgba8();

        store_in_thumbnail_memory_cache(path_str, rgba.clone());

        if let Some(ref cp) = cache_path {
            let _ = rgba.save(cp);
        }
        return Some(rgba);
    }
    None
}

/// Converts core rust clustering models into the Slint-friendly row representation.
pub fn map_groups_to_rows(groups: &[DuplicateGroupSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    let is_pow2 = |n: usize| n != 0 && (n & (n - 1)) == 0;

    for (g_idx, group) in groups.iter().enumerate() {
        if group.files.is_empty() {
            continue;
        }

        // 1. Add Group Cluster Header row
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: false,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: group.hash.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),

            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),

            size_str: crate::utils::helpers::format_size(
                group.files.first().map(|f| f.size).unwrap_or(0),
            ),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
            similarity: 100.0,

            size_bytes: 0,
            pixels_count: 0,

            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        });

        // 2. Add Child file rows belonging to this cluster
        for (f_idx, file) in group.files.iter().enumerate() {
            let is_best = f_idx == 0;
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

                format_str: file.compression_format.clone(),
                dimensions_str: format!("{} x {}", file.width, file.height),
                mipmaps_str: file.mipmap_count.to_string(),
                cubemap_str: if file.is_cubemap {
                    "YES".to_string()
                } else {
                    "NO".to_string()
                },

                size_str: crate::utils::helpers::format_size(file.size),
                meta_str: {
                    let size_mb = file.size as f64 / 1024.0 / 1024.0;
                    let alpha_str = if file.has_alpha { "RGBA" } else { "RGB" };
                    format!(
                        "{}x{} • {:.2} MB • {} • {} • {}-bit • {} • Mips: {}",
                        file.width,
                        file.height,
                        size_mb,
                        file.format_str.to_uppercase(),
                        file.compression_format,
                        file.bit_depth,
                        alpha_str,
                        file.mipmap_count
                    )
                },
                is_best,
                is_checked: false,
                thumbnail_data,
                similarity: file.similarity,

                size_bytes: file.size,
                pixels_count: (file.width * file.height) as u64,

                is_npot: !is_pow2(file.width) || !is_pow2(file.height),
                is_uncompressed: file
                    .compression_format
                    .to_lowercase()
                    .contains("uncompressed"),
                is_missing_mips: file.mipmap_count <= 1,
                is_cubemap_bool: file.is_cubemap,
            });
        }
    }
    rows
}

/// Converts core technical Quality Control issues into Slint row models grouped by issue category.
pub fn map_qc_to_rows(issues: &[crate::state::QcIssueSummary]) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    if issues.is_empty() {
        return rows;
    }

    let mut grouped: std::collections::HashMap<String, Vec<&crate::state::QcIssueSummary>> =
        std::collections::HashMap::new();
    for issue in issues {
        grouped.entry(issue.issue.clone()).or_default().push(issue);
    }

    let mut sorted_issue_types: Vec<String> = grouped.keys().cloned().collect();
    sorted_issue_types.sort();

    for (g_idx, issue_type) in sorted_issue_types.iter().enumerate() {
        let group_issues = &grouped[issue_type];

        // 1. Add Group Header Row for this specific issue type
        rows.push(ResultsRowData {
            is_header: true,
            is_qc: true,
            is_ai: false,
            group_index: g_idx as i32,
            hash_or_issue: issue_type.clone(),
            path: String::new(),
            name: String::new(),
            score_or_detail: String::new(),

            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),

            size_str: format!("{} files", group_issues.len()),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data: None,
            similarity: 0.0,

            size_bytes: 0,
            pixels_count: 0,

            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        });

        // 2. Add Child Rows belonging to this issue category
        for issue in group_issues {
            let thumbnail_data = load_thumbnail_for_path(&issue.path);
            rows.push(ResultsRowData {
                is_header: false,
                is_qc: true,
                is_ai: false,
                group_index: g_idx as i32,
                hash_or_issue: issue.issue.clone(),
                path: issue.path.clone(),
                name: Path::new(&issue.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                score_or_detail: issue.issue.clone(),

                format_str: String::new(),
                dimensions_str: String::new(),
                mipmaps_str: String::new(),
                cubemap_str: String::new(),

                size_str: String::new(),
                meta_str: issue.details.clone(),
                is_best: false,
                is_checked: false,
                thumbnail_data,
                similarity: 0.0,

                size_bytes: 0,
                pixels_count: 0,

                is_npot: false,
                is_uncompressed: false,
                is_missing_mips: false,
                is_cubemap_bool: false,
            });
        }
    }

    rows
}

/// Converts semantic vector database matches into Slint row models.
pub fn map_ai_search_to_rows(
    matches: &[crate::state::AiSearchResultSummary],
) -> Vec<ResultsRowData> {
    let mut rows = Vec::new();
    for res in matches {
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

            format_str: String::new(),
            dimensions_str: String::new(),
            mipmaps_str: String::new(),
            cubemap_str: String::new(),

            size_str: String::new(),
            meta_str: String::new(),
            is_best: false,
            is_checked: false,
            thumbnail_data,
            similarity: res.similarity,

            size_bytes: 0,
            pixels_count: 0,

            is_npot: false,
            is_uncompressed: false,
            is_missing_mips: false,
            is_cubemap_bool: false,
        });
    }
    rows
}
