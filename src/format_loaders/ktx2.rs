// src/format_loaders/ktx2.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::borrow::Cow;
use std::path::Path;

use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::utils::image_processing::{bgra_to_rgba_in_place, bgra_u32_to_rgba_bytes};
use crate::viewer::tonemapping::TonemapConfig;

pub struct Ktx2Loader;

/// Helper function to resolve ASTC 2D block dimensions (width, height) from Vulkan `VkFormat` raw values.
fn resolve_astc_block_size(vk_format: u32) -> Option<(usize, usize)> {
    match vk_format {
        157 | 158 | 1000066000 => Some((4, 4)),
        159 | 160 | 1000066001 => Some((5, 4)),
        161 | 162 | 1000066002 => Some((5, 5)),
        163 | 164 | 1000066003 => Some((6, 5)),
        165 | 166 | 1000066004 => Some((6, 6)),
        167 | 168 | 1000066005 => Some((8, 5)),
        169 | 170 | 1000066006 => Some((8, 6)),
        171 | 172 | 1000066007 => Some((8, 8)),
        173 | 174 | 1000066008 => Some((10, 5)),
        175 | 176 | 1000066009 => Some((10, 6)),
        177 | 178 | 1000066010 => Some((10, 8)),
        179 | 180 | 1000066011 => Some((10, 10)),
        181 | 182 | 1000066012 => Some((12, 10)),
        183 | 184 | 1000066013 => Some((12, 12)),
        _ => None,
    }
}

/// Decompresses level payload if the KTX2 container uses a supercompression scheme.
fn decompress_level_payload<'a>(
    header: &ktx2::Header,
    raw_payload: &'a [u8],
) -> Result<Cow<'a, [u8]>> {
    match header.supercompression_scheme {
        None => Ok(Cow::Borrowed(raw_payload)),
        Some(ktx2::SupercompressionScheme::Zstandard) => {
            let decompressed = zstd::decode_all(raw_payload)
                .map_err(|e| anyhow!("Failed to decompress KTX2 Zstd payload: {}", e))?;
            Ok(Cow::Owned(decompressed))
        }
        Some(ktx2::SupercompressionScheme::ZLIB) => {
            use std::io::Read;
            let mut decoder = flate2::read::ZlibDecoder::new(raw_payload);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| anyhow!("Failed to decompress KTX2 Zlib payload: {}", e))?;
            Ok(Cow::Owned(decompressed))
        }
        Some(ktx2::SupercompressionScheme::BasisLZ) => Err(anyhow!(
            "BasisLZ supercompressed KTX2 requires universal transcoding"
        )),
        Some(scheme) => Err(anyhow!(
            "Unsupported KTX2 supercompression scheme: {:?}",
            scheme
        )),
    }
}

/// Calculates the expected byte slice length for a single 2D image face/layer at Level 0.
fn calculate_single_slice_bytes(vk_format: u32, width: usize, height: usize) -> Option<usize> {
    match vk_format {
        // Uncompressed RGBA8 / BGRA8
        37 | 43 | 44 | 50 => Some(width * height * 4),
        // Uncompressed RGB8 / BGR8
        23 | 29 | 30 | 36 => Some(width * height * 3),
        // Uncompressed R8
        9 | 15 => Some(width * height),
        // Uncompressed RG8
        16 | 22 => Some(width * height * 2),

        // BC1 (DXT1), BC4, ETC1, ETC2 RGB, ETC2 RGBA1, EAC R11 (8 bytes per 4x4 block)
        131..=134 | 139 | 140 | 147..=150 | 153 | 154 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 8)
        }
        // BC2, BC3, BC5, BC6H, BC7, ETC2 RGBA8, EAC RG11 (16 bytes per 4x4 block)
        135..=138 | 141..=146 | 151 | 152 | 155 | 156 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 16)
        }
        // ASTC 2D formats (16 bytes per block)
        157..=184 | 1000066000..=1000066013 => {
            let (bx, by) = resolve_astc_block_size(vk_format)?;
            Some(width.div_ceil(bx) * height.div_ceil(by) * 16)
        }
        // PVRTC 4BPP (8 bytes per 4x4 block)
        1000054001 | 1000054003 | 1000054005 | 1000054007 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 8)
        }
        // PVRTC 2BPP (8 bytes per 8x4 block)
        1000054000 | 1000054002 | 1000054004 | 1000054006 => {
            Some(width.div_ceil(8) * height.div_ceil(4) * 8)
        }
        _ => None,
    }
}

/// Decodes KTX2 container textures natively using `ktx2` reader and `texture2ddecoder` for payload decompression.
pub fn decode_ktx2_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    // Parse KTX2 container header and level index tables securely via ktx2 crate
    let reader = ktx2::Reader::new(bytes)
        .map_err(|e| anyhow!("Invalid KTX2 header or corrupted file: {:?}", e))?;

    let header = reader.header();
    let width = header.pixel_width as usize;
    let height = header.pixel_height.max(1) as usize;

    // OOM Bounds Check: Prevent memory allocation crashes on corrupted/oversized image headers
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(anyhow!(
            "Invalid or oversized KTX2 dimensions: {}x{}",
            width,
            height
        ));
    }

    // Retrieve raw level payload for the base mipmap level (Level 0)
    let first_level = reader
        .levels()
        .next()
        .ok_or_else(|| anyhow!("KTX2 container contains no mip levels"))?;

    // Decompress payload if Zstd/Zlib container supercompression is enabled
    let decompressed_payload = decompress_level_payload(&header, first_level.data)?;
    let level_data: &[u8] = &decompressed_payload;

    let format_raw = header.format.map(|f| f.value()).unwrap_or(0);

    // Isolate single 2D slice for cubemaps (face_count == 6) or 2D array layers
    let slice_bytes =
        calculate_single_slice_bytes(format_raw, width, height).unwrap_or(level_data.len());
    let slice_data = &level_data[..level_data.len().min(slice_bytes)];

    let mut rgba_u32 = vec![0u32; width * height];

    match format_raw {
        // VK_FORMAT_R8G8B8A8_UNORM (37) | VK_FORMAT_R8G8B8A8_SRGB (43)
        37 | 43 => {
            let copy_len = slice_data.len().min(width * height * 4);
            let img = image::RgbaImage::from_raw(
                width as u32,
                height as u32,
                slice_data[..copy_len].to_vec(),
            )
            .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 UNORM data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // VK_FORMAT_B8G8R8A8_UNORM (44) | VK_FORMAT_B8G8R8A8_SRGB (50)
        44 | 50 => {
            let copy_len = slice_data.len().min(width * height * 4);
            let mut bgra_buf = slice_data[..copy_len].to_vec();
            bgra_to_rgba_in_place(&mut bgra_buf);
            let img = image::RgbaImage::from_raw(width as u32, height as u32, bgra_buf)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 BGRA data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // BC1 / DXT1 (131..=134)
        131..=134 => {
            texture2ddecoder::decode_bc1(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC1 decoding failed: {:?}", e))?;
        }

        // BC2 / DXT3 (135, 136)
        135 | 136 => {
            texture2ddecoder::decode_bc2(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC2 decoding failed: {:?}", e))?;
        }

        // BC3 / DXT5 (137, 138)
        137 | 138 => {
            texture2ddecoder::decode_bc3(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC3 decoding failed: {:?}", e))?;
        }

        // BC4 UNORM (139) / SNORM (140)
        139 | 140 => {
            texture2ddecoder::decode_bc4(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC4 decoding failed: {:?}", e))?;
        }

        // BC5 UNORM (141) / SNORM (142)
        141 | 142 => {
            texture2ddecoder::decode_bc5(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC5 decoding failed: {:?}", e))?;
        }

        // BC6H UFLOAT (143)
        143 => {
            texture2ddecoder::decode_bc6_unsigned(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC6H UFLOAT decoding failed: {:?}", e))?;
        }
        // BC6H SFLOAT (144)
        144 => {
            texture2ddecoder::decode_bc6_signed(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC6H SFLOAT decoding failed: {:?}", e))?;
        }

        // BC7 (145, 146)
        145 | 146 => {
            texture2ddecoder::decode_bc7(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC7 decoding failed: {:?}", e))?;
        }

        // ETC1 (160)
        160 => {
            texture2ddecoder::decode_etc1(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC1 decoding failed: {:?}", e))?;
        }

        // ETC2 RGB8 / RGBA1 / RGBA8 (147..=152)
        147 | 148 => {
            texture2ddecoder::decode_etc2_rgb(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGB decoding failed: {:?}", e))?;
        }
        149 | 150 => {
            texture2ddecoder::decode_etc2_rgba1(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGBA1 decoding failed: {:?}", e))?;
        }
        151 | 152 => {
            texture2ddecoder::decode_etc2_rgba8(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGBA8 decoding failed: {:?}", e))?;
        }

        // EAC R11 (153, 154)
        153 | 154 => {
            texture2ddecoder::decode_eacr(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R11 decoding failed: {:?}", e))?;
        }

        // EAC RG11 (155, 156)
        155 | 156 => {
            texture2ddecoder::decode_eacrg(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG11 decoding failed: {:?}", e))?;
        }

        // ASTC 2D formats (157..=184 | 1000066000..=1000066013)
        157..=184 | 1000066000..=1000066013 => {
            let (bx, by) = resolve_astc_block_size(format_raw).ok_or_else(|| {
                anyhow!("Unrecognized ASTC VkFormat enum identifier: {}", format_raw)
            })?;
            texture2ddecoder::decode_astc(slice_data, width, height, bx, by, &mut rgba_u32)
                .map_err(|e| anyhow!("ASTC {}x{} decoding failed: {:?}", bx, by, e))?;
        }

        // PVRTC 2BPP / 4BPP (1000054000..=1000054007)
        1000054000..=1000054007 => {
            let is_2bpp = matches!(
                format_raw,
                1000054000 | 1000054002 | 1000054004 | 1000054006
            );
            texture2ddecoder::decode_pvrtc(slice_data, width, height, &mut rgba_u32, is_2bpp)
                .map_err(|e| anyhow!("PVRTC decoding failed: {:?}", e))?;
        }

        _ => {
            return Err(anyhow!(
                "Unsupported or unhandled KTX2 VkFormat raw enum: {}",
                format_raw
            ));
        }
    }

    let raw_bytes = bgra_u32_to_rgba_bytes(rgba_u32);

    let img = image::RgbaImage::from_raw(width as u32, height as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile KTX2 RGBA buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
}

impl ImageFormatLoader for Ktx2Loader {
    fn extensions(&self) -> &[&str] {
        &["ktx2", "basis"]
    }

    fn decode(
        &self,
        path: &Path,
        _target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let bytes = std::fs::read(path).context("Failed to read KTX2 file from disk")?;
        decode_ktx2_bytes(&bytes)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        let mut w = 0;
        let mut h = 0;
        let mut mips = 1;
        let mut is_cubemap = false;
        let mut compression_format = "KTX2".to_string();
        let mut color_space = "Linear".to_string();

        if let Ok(reader) = ktx2::Reader::new(&bytes) {
            let header = reader.header();
            w = header.pixel_width;
            h = header.pixel_height.max(1);
            mips = header.level_count.max(1);
            is_cubemap = header.face_count == 6;

            if let Some(scheme) = header.supercompression_scheme {
                compression_format = format!("KTX2 ({:?})", scheme);
            } else if let Some(fmt) = header.format {
                compression_format = format!("KTX2 ({:?})", fmt);
            }

            if let Some(tf) = reader.transfer_function() {
                if tf == ktx2::TransferFunction::SRGB {
                    color_space = "sRGB".to_string();
                }
            } else if compression_format.contains("SRGB") {
                color_space = "sRGB".to_string();
            }
        } else if let Ok(dim) = imagesize::size(path) {
            w = dim.width as u32;
            h = dim.height as u32;
        }

        let estimated_vram =
            crate::qc::rules::estimate_vram(w, h, &compression_format, mips, is_cubemap);

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "ktx2".to_string(),
            compression_format,
            color_space,
            has_alpha: true,
            bit_depth: 8,
            mipmap_count: mips,
            is_cubemap,
            estimated_vram,
        })
    }
}
