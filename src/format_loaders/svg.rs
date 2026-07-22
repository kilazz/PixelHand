// src/format_loaders/svg.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use resvg::tiny_skia::Pixmap;
use resvg::usvg::{Options, Transform, Tree};
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct SvgLoader;

pub fn decode_svg_bytes(bytes: &[u8], target_size: Option<u32>) -> Result<DynamicImage> {
    let opt = Options::default();
    let tree = Tree::from_data(bytes, &opt).map_err(|e| anyhow!("SVG parsing failed: {}", e))?;

    let mut width = tree.size().width() as u32;
    let mut height = tree.size().height() as u32;

    if let Some(target) = target_size {
        let aspect = width as f32 / height as f32;
        if width > target || height > target {
            if aspect > 1.0 {
                width = target;
                height = (target as f32 / aspect) as u32;
            } else {
                height = target;
                width = (target as f32 * aspect) as u32;
            }
        }
    }

    width = width.max(16);
    height = height.max(16);

    let mut pixmap =
        Pixmap::new(width, height).ok_or_else(|| anyhow!("Failed to create SVG pixmap buffer"))?;

    let scale_x = width as f32 / tree.size().width();
    let scale_y = height as f32 / tree.size().height();
    let transform = Transform::from_scale(scale_x, scale_y);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let img = image::RgbaImage::from_raw(width, height, pixmap.take())
        .ok_or_else(|| anyhow!("Failed to compile RGBA buffer from SVG pixmap"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for SvgLoader {
    fn extensions(&self) -> &[&str] {
        &["svg"]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read SVG file")?;
        decode_svg_bytes(&bytes, target_size)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let (w, h) = imagesize::size(path)
            .map(|d| (d.width as u32, d.height as u32))
            .unwrap_or((512, 512));

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "svg".to_string(),
            compression_format: "SVG Vector".to_string(),
            color_space: "sRGB".to_string(),
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: 1,
            is_cubemap: false,
            estimated_vram: crate::qc::rules::estimate_vram(w, h, "UNCOMPRESSED", 1, false),
        })
    }
}
