// src/format_loaders/mod.rs

pub mod dds;
pub mod exr;
pub mod heic;
pub mod jxl;
pub mod ktx2;
pub mod psd;
pub mod raw;
pub mod standard;

use anyhow::{Result, anyhow};
use image::DynamicImage;
use std::path::Path;
use std::sync::OnceLock;

use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

// =============================================================================
// --- IMAGE FORMAT LOADER TRAIT -----------------------------------------------
// =============================================================================

/// Unified polymorphic interface for all graphics asset loaders.
pub trait ImageFormatLoader: Send + Sync {
    /// Supported lowercase file extensions (e.g. "exr", "dds", "ktx2")
    fn extensions(&self) -> &[&str];

    /// High-fidelity decoding into dynamic image memory buffers.
    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage>;

    /// Rapid technical metadata extraction from file headers.
    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata>;

    /// Decodes a specific mipmap layer if supported by the file format.
    /// Default implementation falls back to standard decoding.
    fn decode_specific_mip(
        &self,
        path: &Path,
        _mip_level: u32,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        self.decode(path, None, tonemap_config)
    }
}

// =============================================================================
// --- CENTRAL POLYMORPHIC REGISTRY --------------------------------------------
// =============================================================================

pub struct LoaderRegistry {
    loaders: Vec<Box<dyn ImageFormatLoader>>,
}

impl LoaderRegistry {
    pub fn new() -> Self {
        Self {
            loaders: vec![
                Box::new(dds::DdsLoader),
                Box::new(exr::ExrLoader),
                Box::new(psd::PsdLoader),
                Box::new(jxl::JxlLoader),
                Box::new(heic::HeicLoader),
                Box::new(ktx2::Ktx2Loader),
                Box::new(raw::RawCameraLoader),
                Box::new(standard::StandardLoader),
            ],
        }
    }

    /// Searches for a loader that maps to the requested file extension.
    pub fn find_loader(&self, ext: &str) -> Option<&dyn ImageFormatLoader> {
        self.loaders
            .iter()
            .find(|l| l.extensions().contains(&ext))
            .map(|l| l.as_ref())
    }
}

pub static REGISTRY: OnceLock<LoaderRegistry> = OnceLock::new();

/// Retrieves the central thread-safe registry singleton.
pub fn get_registry() -> &'static LoaderRegistry {
    REGISTRY.get_or_init(LoaderRegistry::new)
}

// =============================================================================
// --- CONVENIENCE CONTEXT WRAPPERS --------------------------------------------
// =============================================================================

/// Public wrapper to decode an image, falling back automatically to the correct format loader.
pub fn open_image_with_dds_fallback<P: AsRef<Path>>(
    path: P,
    target_size: Option<u32>,
    tonemap_config: Option<TonemapConfig>,
) -> Result<DynamicImage> {
    let path_ref = path.as_ref();
    let ext = path_ref
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let registry = get_registry();
    if let Some(loader) = registry.find_loader(&ext) {
        loader.decode(path_ref, target_size, tonemap_config)
    } else {
        Err(anyhow!("Unsupported file extension: {}", ext))
    }
}

/// Public wrapper to decode a specific mipmap resolution layer from supported loaders.
pub fn open_image_with_specific_mip<P: AsRef<Path>>(
    path: P,
    mip_level: u32,
    tonemap_config: Option<TonemapConfig>,
) -> Result<DynamicImage> {
    let path_ref = path.as_ref();
    let ext = path_ref
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    let registry = get_registry();
    if let Some(loader) = registry.find_loader(&ext) {
        loader.decode_specific_mip(path_ref, mip_level, tonemap_config)
    } else {
        Err(anyhow!("Unsupported file extension: {}", ext))
    }
}
