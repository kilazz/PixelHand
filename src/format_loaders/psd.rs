// src/format_loaders/psd.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct PsdLoader;

impl ImageFormatLoader for PsdLoader {
    fn extensions(&self) -> &[&str] {
        &["psd"]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read PSD file from disk")?;

        // Early validation: Check magic header bytes ("8BPS") before invoking the parser
        if bytes.len() < 26 || &bytes[0..4] != b"8BPS" {
            return Err(anyhow!("Invalid PSD header magic bytes (expected '8BPS')"));
        }

        // Safely catch potential panics from the `psd` crate (e.g., unsupported Zip-compressed layers)
        let psd_result = std::panic::catch_unwind(|| psd::Psd::from_bytes(&bytes));

        let psd = match psd_result {
            Ok(Ok(psd)) => psd,
            Ok(Err(e)) => return Err(anyhow!("PSD parsing failed: {}", e)),
            Err(_) => {
                return Err(anyhow!(
                    "PSD decoding panicked (unsupported Zip compression or feature in psd crate)"
                ));
            }
        };

        let width = psd.width();
        let height = psd.height();

        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(anyhow!(
                "Invalid or oversized PSD dimensions: {}x{}",
                width,
                height
            ));
        }

        let rgba_bytes = psd.rgba();
        let buffer = image::RgbaImage::from_raw(width, height, rgba_bytes)
            .ok_or_else(|| anyhow!("Failed to compile PSD RGBA pixel buffer"))?;

        let mut img = DynamicImage::ImageRgba8(buffer);

        // Scale down thumbnail if target_size is requested
        if let Some(target) = target_size
            && (img.width() > target || img.height() > target)
        {
            img = img.resize(target, target, image::imageops::FilterType::Triangle);
        }

        Ok(img)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut bit_depth = 8;
        let mut color_space = "sRGB".to_string();
        let mut compression_format = "PSD".to_string();

        let is_valid_psd_header = bytes.len() >= 26 && &bytes[0..4] == b"8BPS";

        if is_valid_psd_header
            && let Ok(Ok(psd)) = std::panic::catch_unwind(|| psd::Psd::from_bytes(&bytes))
        {
            w = psd.width();
            h = psd.height();

            bit_depth = match psd.depth() {
                psd::PsdDepth::One => 1,
                psd::PsdDepth::Eight => 8,
                psd::PsdDepth::Sixteen => 16,
                psd::PsdDepth::ThirtyTwo => 32,
            };

            color_space = match psd.color_mode() {
                psd::ColorMode::Rgb => "sRGB".to_string(),
                psd::ColorMode::Cmyk => "CMYK".to_string(),
                psd::ColorMode::Grayscale => "Grayscale".to_string(),
                psd::ColorMode::Bitmap => "Bitmap".to_string(),
                psd::ColorMode::Indexed => "Indexed".to_string(),
                psd::ColorMode::Lab => "Lab".to_string(),
                psd::ColorMode::Multichannel => "Multichannel".to_string(),
                psd::ColorMode::Duotone => "Duotone".to_string(),
            };

            compression_format = format!("PSD ({:?})", psd.compression());
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        let estimated_vram = crate::qc::rules::estimate_vram(w, h, "UNCOMPRESSED", 1, false);

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "psd".to_string(),
            compression_format,
            color_space,
            has_alpha: true,
            bit_depth,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram,
        })
    }
}
