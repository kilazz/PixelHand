// src/format_loaders/mod.rs

pub mod dds_loader;

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;
use std::sync::OnceLock;

use crate::core::qc::QcImageMetadata;
use crate::core::tonemapper::{TonemapConfig, TonemapOperator};

/// Unified polymorphic interface for all graphics asset loaders
pub trait ImageFormatLoader: Send + Sync {
    /// Supported lowercase file extensions (e.g. "exr", "dds")
    fn extensions(&self) -> &[&str];

    /// High-fidelity decoding into dynamic memory buffers
    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage>;

    /// Rapid technical metadata extraction from file headers
    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata>;

    /// Optional: specific mipmap decoding fallback (overridden by DDS)
    fn decode_specific_mip(
        &self,
        path: &Path,
        _mip_level: u32,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        self.decode(path, None, tonemap_config)
    }
}

// ---------------------------------------------------------
// INDIVIDUAL FORMAT LOADER IMPLEMENTATIONS
// ---------------------------------------------------------

pub struct ExrLoader;
impl ImageFormatLoader for ExrLoader {
    fn extensions(&self) -> &[&str] {
        &["exr", "ext_exr"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let (hdr_pixels, width, height) = crate::core::tonemapper::load_exr_rgba(path)
            .context("Failed to load EXR float pixels")?;

        let config = tonemap_config.unwrap_or(TonemapConfig {
            enabled: true,
            auto_exposure: true,
            operator: TonemapOperator::AcesFilmic,
        });

        let ldr_img = crate::core::tonemapper::tonemap_hdr_to_ldr_rgba(
            &hdr_pixels,
            width,
            height,
            config,
            1.0,
        )?;
        Ok(DynamicImage::ImageRgba8(ldr_img))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let mut w = 0;
        let mut h = 0;
        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();
        let mut compression_format = "EXR".to_string();
        let mut has_alpha = false;

        if let Ok(meta) = exr::prelude::MetaData::read_from_file(path, false)
            && let Some(header) = meta.headers.first()
        {
            w = header.shared_attributes.display_window.size.width() as u32;
            h = header.shared_attributes.display_window.size.height() as u32;
            bit_depth = 16;
            color_space = "Linear".to_string();
            compression_format = format!("{:?}", header.compression);
            has_alpha = header
                .channels
                .list
                .iter()
                .any(|c| c.name.to_string() == "A" || c.name.to_string() == "alpha");
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "exr".to_string(),
            compression_format,
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::core::qc::estimate_vram(w, h, "EXR", 1, false),
        })
    }
}

pub struct PsdLoader;
impl ImageFormatLoader for PsdLoader {
    fn extensions(&self) -> &[&str] {
        &["psd"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read PSD file")?;
        let psd = psd::Psd::from_bytes(&bytes).map_err(|e| anyhow!("PSD parsing failed: {}", e))?;
        let buffer = image::RgbaImage::from_raw(psd.width(), psd.height(), psd.rgba())
            .ok_or_else(|| anyhow!("Failed to compile PSD pixel buffer"))?;
        Ok(DynamicImage::ImageRgba8(buffer))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));
        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "psd".to_string(),
            compression_format: "PSD".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::core::qc::estimate_vram(w, h, "PSD", 1, false),
        })
    }
}

pub struct JxlLoader;
impl ImageFormatLoader for JxlLoader {
    fn extensions(&self) -> &[&str] {
        &["jxl"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read JXL file")?;
        let decoder = jxl_oxide::integration::JxlDecoder::new(std::io::Cursor::new(bytes))
            .map_err(|e| anyhow!("JXL decoder initialization failed: {:?}", e))?;
        let img = DynamicImage::from_decoder(decoder)
            .map_err(|e| anyhow!("JXL decoding pipeline failed: {:?}", e))?;
        Ok(img)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));
        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "jxl".to_string(),
            compression_format: "JXL".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: false,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::core::qc::estimate_vram(w, h, "JXL", 1, false),
        })
    }
}

pub struct HeicLoader;
impl ImageFormatLoader for HeicLoader {
    fn extensions(&self) -> &[&str] {
        &["heic", "heif"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read HEIC/HEIF file")?;
        let output = heic::DecoderConfig::new()
            .decode(&bytes, heic::PixelLayout::Rgba8)
            .map_err(|e| anyhow!("HEIC decoding failed: {:?}", e))?;
        let buffer =
            image::RgbaImage::from_raw(output.width as u32, output.height as u32, output.data)
                .ok_or_else(|| anyhow!("Failed to map raw HEIC pixel data buffer"))?;
        Ok(DynamicImage::ImageRgba8(buffer))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));
        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "heic".to_string(),
            compression_format: "HEIC/HEIF".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: false,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::core::qc::estimate_vram(w, h, "HEIC", 1, false),
        })
    }
}

pub struct StandardLoader;
impl ImageFormatLoader for StandardLoader {
    fn extensions(&self) -> &[&str] {
        &[
            "png", "jpg", "jpeg", "tga", "bmp", "hdr", "tif", "tiff", "webp", "gif", "avif",
        ]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let img = image::open(path).context("Failed to decode standard image format")?;

        if ext == "hdr"
            || matches!(
                img.color(),
                image::ColorType::Rgb32F | image::ColorType::Rgba32F
            )
        {
            let float_img = img.to_rgba32f();
            let width = float_img.width();
            let height = float_img.height();
            let hdr_pixels = float_img.into_raw();

            let config = tonemap_config.unwrap_or(TonemapConfig {
                enabled: true,
                auto_exposure: true,
                operator: TonemapOperator::AcesFilmic,
            });

            let ldr_img = crate::core::tonemapper::tonemap_hdr_to_ldr_rgba(
                &hdr_pixels,
                width,
                height,
                config,
                1.0,
            )?;
            Ok(DynamicImage::ImageRgba8(ldr_img))
        } else {
            Ok(DynamicImage::ImageRgba8(img.to_rgba8()))
        }
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((0, 0));

        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();
        let mut compression_format = ext.to_uppercase();
        let mut has_alpha = false;

        if ext == "hdr" {
            bit_depth = 32;
            color_space = "Linear".to_string();
            compression_format = "Radiance HDR".to_string();
        } else if let Ok(img) = image::open(path) {
            let color = img.color();
            has_alpha = color.has_alpha();
            bit_depth = color.bits_per_pixel() as u32 / if has_alpha { 4 } else { 3 };
        }

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: ext.clone(),
            compression_format,
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::core::qc::estimate_vram(w, h, &ext, 1, false),
        })
    }
}

// ---------------------------------------------------------
// CENTRAL POLYMORPHIC REGISTRY
// ---------------------------------------------------------

pub struct LoaderRegistry {
    loaders: Vec<Box<dyn ImageFormatLoader>>,
}

impl LoaderRegistry {
    pub fn new() -> Self {
        Self {
            loaders: vec![
                Box::new(crate::format_loaders::dds_loader::DdsLoader),
                Box::new(ExrLoader),
                Box::new(PsdLoader),
                Box::new(JxlLoader),
                Box::new(HeicLoader),
                Box::new(StandardLoader),
            ],
        }
    }

    pub fn find_loader(&self, ext: &str) -> Option<&dyn ImageFormatLoader> {
        self.loaders
            .iter()
            .find(|l| l.extensions().contains(&ext))
            .map(|l| l.as_ref())
    }
}

pub static REGISTRY: OnceLock<LoaderRegistry> = OnceLock::new();

pub fn get_registry() -> &'static LoaderRegistry {
    REGISTRY.get_or_init(LoaderRegistry::new)
}

// ---------------------------------------------------------
// PUBLIC CONVENIENCE WRAPPERS (Leaves existing calls intact)
// ---------------------------------------------------------

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
