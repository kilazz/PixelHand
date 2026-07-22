// src/format_loaders/svg.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use resvg::tiny_skia::Pixmap;
use resvg::usvg::{self, Options, Transform, Tree};
use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct SvgLoader;

/// Global thread-safe FontDB singleton to avoid expensive system font re-scanning on every SVG decode.
static FONT_DB: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();

fn get_font_database() -> Arc<usvg::fontdb::Database> {
    FONT_DB
        .get_or_init(|| {
            let mut db = usvg::fontdb::Database::new();
            db.load_system_fonts();
            Arc::new(db)
        })
        .clone()
}

/// Preprocesses raw SVG byte buffer, transparently handling .svgz (Gzip compressed SVG) files.
fn prepare_svg_data(bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
    // Check for Gzip magic bytes header (0x1f, 0x8b)
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let decompressed = usvg::decompress_svgz(bytes)
            .map_err(|e| anyhow!("Failed to decompress SVGZ archive: {}", e))?;
        Ok(Cow::Owned(decompressed))
    } else {
        Ok(Cow::Borrowed(bytes))
    }
}

pub fn decode_svg_bytes(
    bytes: &[u8],
    target_size: Option<u32>,
    parent_dir: Option<&Path>,
) -> Result<DynamicImage> {
    let svg_data = prepare_svg_data(bytes)?;

    let mut opt = Options {
        fontdb: get_font_database(),
        ..Options::default()
    };
    if let Some(dir) = parent_dir {
        opt.resources_dir = Some(dir.to_path_buf());
    }

    let tree =
        Tree::from_data(&svg_data, &opt).map_err(|e| anyhow!("SVG parsing failed: {}", e))?;

    let mut width = tree.size().width() as u32;
    let mut height = tree.size().height() as u32;

    if width == 0 || height == 0 {
        return Err(anyhow!("SVG document has invalid zero-area dimensions"));
    }

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
        &["svg", "svgz"]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read SVG file from disk")?;
        let parent_dir = path.parent();
        decode_svg_bytes(&bytes, target_size, parent_dir)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let (w, h) = if let Ok(svg_data) = prepare_svg_data(&bytes) {
            let opt = Options {
                fontdb: get_font_database(),
                ..Options::default()
            };
            if let Ok(tree) = Tree::from_data(&svg_data, &opt) {
                (tree.size().width() as u32, tree.size().height() as u32)
            } else {
                (512, 512)
            }
        } else {
            (512, 512)
        };

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
