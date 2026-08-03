// src/format_loaders/uasset.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use unreal_asset::{
    Asset,
    engine_version::EngineVersion,
    exports::Export,
    properties::{Property, PropertyDataTrait, int_property::BytePropertyValue},
};

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::viewer::tonemapping::TonemapConfig;

pub struct UassetLoader;

/// Helper function to open paired Unreal Engine package files (.uasset, .uexp).
fn open_uasset_files(path: &Path) -> Result<(BufReader<File>, Option<BufReader<File>>)> {
    let uasset_file =
        File::open(path).with_context(|| format!("Failed to open .uasset file: {:?}", path))?;

    let uexp_path = path.with_extension("uexp");
    let uexp_file = if uexp_path.exists() {
        File::open(&uexp_path).ok().map(BufReader::new)
    } else {
        None
    };

    Ok((BufReader::new(uasset_file), uexp_file))
}

/// Parses "64x64" or "1024x1024" strings into (width, height)
fn parse_dimensions_string(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X', '*']).collect();
    if parts.len() == 2 {
        let w = parts[0].trim().parse::<u32>().ok()?;
        let h = parts[1].trim().parse::<u32>().ok()?;
        if w > 0 && h > 0 {
            return Some((w, h));
        }
    }
    None
}

/// Scans raw binary buffer for embedded JPEG or PNG images (used as fallbacks/thumbnails)
fn scan_embedded_image(bytes: &[u8]) -> Option<DynamicImage> {
    if bytes.len() < 128 {
        return None;
    }
    let mut i = 0;
    let limit = bytes.len().saturating_sub(4);
    while i < limit {
        if (bytes[i] == 0xFF && bytes[i + 1] == 0xD8 && bytes[i + 2] == 0xFF)
            || bytes[i..].starts_with(b"\x89PNG\r\n\x1a\n")
        {
            let slice = &bytes[i..];
            if let Ok(img) = image::load_from_memory(slice) {
                return Some(img);
            }
        }
        i += 1;
    }
    None
}

/// Scans raw binary buffer for embedded JPEG or PNG image dimensions
fn scan_embedded_image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 128 {
        return None;
    }
    let mut i = 0;
    let limit = bytes.len().saturating_sub(4);
    while i < limit {
        if (bytes[i] == 0xFF && bytes[i + 1] == 0xD8 && bytes[i + 2] == 0xFF)
            || bytes[i..].starts_with(b"\x89PNG\r\n\x1a\n")
        {
            let slice = &bytes[i..];
            if let Ok(size) = imagesize::blob_size(slice)
                && size.width > 0
                && size.height > 0
            {
                return Some((size.width as u32, size.height as u32));
            }
        }
        i += 1;
    }
    None
}

/// Recursively scans properties (including nested StructProperty like FTextureSource "Source")
fn process_properties(
    properties: &[Property],
    width: &mut u32,
    height: &mut u32,
    format_str: &mut String,
) {
    for prop in properties {
        let name = prop.get_name().get_content(|s| s.to_string());
        match prop {
            Property::IntProperty(p) => {
                if name == "SizeX" || name == "ImportedSizeX" {
                    *width = p.value as u32;
                } else if name == "SizeY" || name == "ImportedSizeY" {
                    *height = p.value as u32;
                }
            }
            Property::Int64Property(p) => {
                if name == "SizeX" || name == "ImportedSizeX" {
                    *width = p.value as u32;
                } else if name == "SizeY" || name == "ImportedSizeY" {
                    *height = p.value as u32;
                }
            }
            Property::UInt32Property(p) => {
                if name == "SizeX" || name == "ImportedSizeX" {
                    *width = p.value;
                } else if name == "SizeY" || name == "ImportedSizeY" {
                    *height = p.value;
                }
            }
            Property::UInt64Property(p) => {
                if name == "SizeX" || name == "ImportedSizeX" {
                    *width = p.value as u32;
                } else if name == "SizeY" || name == "ImportedSizeY" {
                    *height = p.value as u32;
                }
            }
            Property::StructProperty(struct_prop) => {
                // Recurse into nested structs like FTextureSource ("Source")
                process_properties(&struct_prop.value, width, height, format_str);
            }
            Property::ArrayProperty(array_prop) => {
                // Recurse into array elements
                for elem in &array_prop.value {
                    process_properties(std::slice::from_ref(elem), width, height, format_str);
                }
            }
            Property::StrProperty(str_prop) => {
                if let Some(ref val) = str_prop.value {
                    if name == "Dimensions" || name == "ImportedDimensions" {
                        if let Some((w, h)) = parse_dimensions_string(val) {
                            *width = w;
                            *height = h;
                        }
                    } else if name == "PixelFormat"
                        || name == "Format"
                        || name == "TextureCompressionSettings"
                        || name == "CompressionSettings"
                    {
                        *format_str = val.clone();
                    } else if (*width == 0 || *height == 0)
                        && let Some((w, h)) = parse_dimensions_string(val)
                    {
                        *width = w;
                        *height = h;
                    }
                }
            }
            Property::NameProperty(name_prop) => {
                let val = name_prop.value.get_content(|s| s.to_string());
                if name == "Dimensions" || name == "ImportedDimensions" {
                    if let Some((w, h)) = parse_dimensions_string(&val) {
                        *width = w;
                        *height = h;
                    }
                } else if name == "PixelFormat"
                    || name == "Format"
                    || name == "TextureCompressionSettings"
                    || name == "CompressionSettings"
                {
                    *format_str = val;
                }
            }
            Property::ByteProperty(byte_prop)
                if name == "PixelFormat"
                    || name == "Format"
                    || name == "TextureCompressionSettings"
                    || name == "CompressionSettings" =>
            {
                match &byte_prop.value {
                    BytePropertyValue::Byte(b) => {
                        *format_str = format!("PF_{}", b);
                    }
                    BytePropertyValue::FName(fname) => {
                        *format_str = fname.get_content(|s| s.to_string());
                    }
                }
            }
            Property::EnumProperty(enum_prop)
                if name == "PixelFormat"
                    || name == "Format"
                    || name == "TextureCompressionSettings"
                    || name == "CompressionSettings" =>
            {
                if let Some(val) = &enum_prop.value {
                    *format_str = val.get_content(|s| s.to_string());
                }
            }
            _ => {}
        }
    }
}

/// Extracts Mip 0 payload and metadata properties from a UTexture2D or Interchange export.
fn extract_texture_data(path: &Path) -> Result<(Vec<u8>, u32, u32, String, u32, bool)> {
    let versions = [
        EngineVersion::VER_UE5_2,
        EngineVersion::VER_UE5_1,
        EngineVersion::VER_UE5_0,
        EngineVersion::VER_UE4_27,
        EngineVersion::VER_UE4_26,
        EngineVersion::UNKNOWN,
    ];

    let mut parsed_asset: Option<Asset<BufReader<File>>> = None;

    for ver in versions {
        let (uasset_r, uexp_r) = open_uasset_files(path)?;
        if let Ok(asset) = Asset::new(uasset_r, uexp_r, ver, None) {
            parsed_asset = Some(asset);
            break;
        }
    }

    if let Some(asset) = parsed_asset {
        for export in &asset.asset_data.exports {
            if let Export::NormalExport(normal_export) = export {
                let mut width = 0u32;
                let mut height = 0u32;
                let mut format_str = "PF_UNKNOWN".to_string();

                // Recursively parse top-level and nested properties (Source, etc.)
                process_properties(
                    &normal_export.properties,
                    &mut width,
                    &mut height,
                    &mut format_str,
                );

                if width > 0 && height > 0 {
                    let payload = locate_bulk_pixel_payload(path, width, height, &format_str)
                        .unwrap_or_default();
                    return Ok((payload, width, height, format_str, 1, false));
                }
            }
        }
    }

    // Fallback: Check if file contains embedded JPEG/PNG preview thumbnail
    if let Ok(bytes) = std::fs::read(path)
        && let Some((width, height)) = scan_embedded_image_dimensions(&bytes)
    {
        return Ok((
            Vec::new(),
            width,
            height,
            "Embedded Preview".to_string(),
            1,
            false,
        ));
    }

    Err(anyhow!(
        "No UTexture2D or Interchange texture properties found inside this .uasset package"
    ))
}

/// Locates the largest contiguous Mip 0 binary payload from .ubulk, .uexp, or .uasset.
fn locate_bulk_pixel_payload(
    path: &Path,
    width: u32,
    height: u32,
    format_str: &str,
) -> Result<Vec<u8>> {
    let expected_bytes = estimate_payload_bytes(width, height, format_str);

    // Try reading from .ubulk first
    let ubulk_path = path.with_extension("ubulk");
    if ubulk_path.exists()
        && let Ok(ubulk_bytes) = std::fs::read(&ubulk_path)
    {
        if ubulk_bytes.len() >= expected_bytes && expected_bytes > 0 {
            let offset = ubulk_bytes.len() - expected_bytes;
            return Ok(ubulk_bytes[offset..].to_vec());
        } else if !ubulk_bytes.is_empty() {
            return Ok(ubulk_bytes);
        }
    }

    // Try reading from .uexp next
    let uexp_path = path.with_extension("uexp");
    if uexp_path.exists()
        && let Ok(uexp_bytes) = std::fs::read(&uexp_path)
        && uexp_bytes.len() >= expected_bytes
        && expected_bytes > 0
    {
        let offset = uexp_bytes.len() - expected_bytes;
        return Ok(uexp_bytes[offset..].to_vec());
    }

    // Fallback to .uasset file itself
    let uasset_bytes = std::fs::read(path)?;
    if uasset_bytes.len() >= expected_bytes && expected_bytes > 0 {
        let offset = uasset_bytes.len() - expected_bytes;
        return Ok(uasset_bytes[offset..].to_vec());
    }

    Err(anyhow!(
        "Could not locate Mip 0 binary payload for .uasset texture"
    ))
}

/// Calculates estimated byte size of Mip 0 for a given format and resolution.
fn estimate_payload_bytes(width: u32, height: u32, format_str: &str) -> usize {
    let fmt = format_str.to_uppercase();
    let w = width as usize;
    let h = height as usize;

    if fmt.contains("BC1")
        || fmt.contains("DXT1")
        || fmt.contains("BC4")
        || fmt.contains("ATI1")
        || fmt.contains("TC_DEFAULT")
    {
        (w.div_ceil(4)) * (h.div_ceil(4)) * 8
    } else if fmt.contains("BC2")
        || fmt.contains("DXT3")
        || fmt.contains("BC3")
        || fmt.contains("DXT5")
        || fmt.contains("BC5")
        || fmt.contains("ATI2")
        || fmt.contains("BC6")
        || fmt.contains("BC7")
        || fmt.contains("ASTC")
        || fmt.contains("TC_NORMALMAP")
        || fmt.contains("TC_MASKS")
        || fmt.contains("TC_GRAYSCALE")
    {
        (w.div_ceil(4)) * (h.div_ceil(4)) * 16
    } else {
        w * h * 4
    }
}

/// Decodes raw Unreal Engine pixel payload buffers into standard DynamicImage buffers.
fn decode_unreal_pixels(
    payload: &[u8],
    width: u32,
    height: u32,
    format_str: &str,
) -> Result<DynamicImage> {
    let w = width as usize;
    let h = height as usize;
    let fmt = format_str.to_uppercase();

    let mut rgba_u32 = vec![0u32; w * h];

    match fmt.as_str() {
        // BC1 / DXT1
        f if f.contains("BC1") || f.contains("DXT1") || f.contains("TC_DEFAULT") => {
            texture2ddecoder::decode_bc1(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC1 decoding failed: {:?}", e))?;
        }
        // BC2 / DXT3
        f if f.contains("BC2") || f.contains("DXT3") => {
            texture2ddecoder::decode_bc2(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC2 decoding failed: {:?}", e))?;
        }
        // BC3 / DXT5
        f if f.contains("BC3") || f.contains("DXT5") => {
            texture2ddecoder::decode_bc3(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC3 decoding failed: {:?}", e))?;
        }
        // BC4 / ATI1
        f if f.contains("BC4") || f.contains("ATI1") || f.contains("TC_GRAYSCALE") => {
            texture2ddecoder::decode_bc4(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC4 decoding failed: {:?}", e))?;
        }
        // BC5 / ATI2 (Two-channel normal maps)
        f if f.contains("BC5")
            || f.contains("ATI2")
            || f.contains("TC_NORMALMAP")
            || f.contains("TC_MASKS") =>
        {
            texture2ddecoder::decode_bc5(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC5 decoding failed: {:?}", e))?;
        }
        // BC6H
        f if f.contains("BC6") || f.contains("TC_HDR") => {
            texture2ddecoder::decode_bc6_unsigned(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC6H decoding failed: {:?}", e))?;
        }
        // BC7
        f if f.contains("BC7") => {
            texture2ddecoder::decode_bc7(payload, w, h, &mut rgba_u32)
                .map_err(|e| anyhow!("BC7 decoding failed: {:?}", e))?;
        }
        // ASTC 2D formats
        f if f.contains("ASTC_6X6") => {
            texture2ddecoder::decode_astc(payload, w, h, 6, 6, &mut rgba_u32)
                .map_err(|e| anyhow!("ASTC 6x6 decoding failed: {:?}", e))?;
        }
        f if f.contains("ASTC_8X8") => {
            texture2ddecoder::decode_astc(payload, w, h, 8, 8, &mut rgba_u32)
                .map_err(|e| anyhow!("ASTC 8x8 decoding failed: {:?}", e))?;
        }
        f if f.contains("ASTC") => {
            texture2ddecoder::decode_astc(payload, w, h, 4, 4, &mut rgba_u32)
                .map_err(|e| anyhow!("ASTC 4x4 decoding failed: {:?}", e))?;
        }
        // Uncompressed BGRA8 / RGBA8
        _ => {
            if payload.len() >= w * h * 4 {
                let mut bgra_buf = payload.to_vec();
                crate::utils::image_processing::bgra_to_rgba_in_place(&mut bgra_buf);
                if let Some(img) = image::RgbaImage::from_raw(width, height, bgra_buf) {
                    return Ok(DynamicImage::ImageRgba8(img));
                }
            }
            Err(anyhow!("Unsupported or truncated pixel payload"))?
        }
    }

    let raw_bytes = crate::utils::image_processing::bgra_u32_to_rgba_bytes(rgba_u32);
    let img = image::RgbaImage::from_raw(width, height, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile RGBA buffer from Unreal texture payload"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for UassetLoader {
    fn extensions(&self) -> &[&str] {
        &["uasset"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let (payload, width, height, format_str, _, _) = extract_texture_data(path)?;

        // Try decoding GPU texture payload first
        if let Ok(img) = decode_unreal_pixels(&payload, width, height, &format_str) {
            return Ok(img);
        }

        // Fallback: If payload decoding fails, extract embedded JPEG/PNG thumbnail
        if let Ok(bytes) = std::fs::read(path)
            && let Some(img) = scan_embedded_image(&bytes)
        {
            return Ok(img);
        }

        Err(anyhow!("Failed to decode .uasset image payload"))
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let file_size = std::fs::metadata(path)?.len();

        let (width, height, mipmap_count, format_str, is_cubemap) =
            if let Ok((_, w, h, fmt, mips, cube)) = extract_texture_data(path) {
                (w, h, mips, fmt, cube)
            } else {
                let (w, h) = imagesize::size(path)
                    .map(|d| (d.width as u32, d.height as u32))
                    .unwrap_or((0, 0));
                (w, h, 1, "Unreal Package".to_string(), false)
            };

        let color_space = if format_str.contains("BC5") || format_str.contains("NORMAL") {
            "Linear".to_string()
        } else {
            "sRGB".to_string()
        };

        let estimated_vram =
            crate::qc::rules::estimate_vram(width, height, &format_str, mipmap_count, is_cubemap);

        Ok(QcImageMetadata {
            width,
            height,
            file_size,
            format_str: "uasset".to_string(),
            compression_format: format_str,
            color_space,
            has_alpha: true,
            bit_depth: 8,
            mipmap_count,
            is_cubemap,
            estimated_vram,
        })
    }
}
