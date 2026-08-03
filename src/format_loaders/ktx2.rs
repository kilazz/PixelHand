// src/format_loaders/ktx2.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::borrow::Cow;
use std::path::Path;

use crate::compression::gpu_block::{TextureBlockFormat, decode_gpu_block};
use crate::compression::stream::{decompress_zlib, decompress_zstd};
use crate::format_loaders::ImageFormatLoader;
use crate::qc::rules::QcImageMetadata;
use crate::utils::image_processing::bgra_to_rgba_in_place;
use crate::viewer::tonemapping::TonemapConfig;

pub struct Ktx2Loader;

/// Bitwise conversion of IEEE 754 half-precision float (f16) to single-precision float (f32).
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 0x0001;
    let e = (h >> 10) & 0x001f;
    let m = h & 0x03ff;

    if e == 0 {
        if m == 0 {
            if s != 0 { -0.0 } else { 0.0 }
        } else {
            let mut e_norm = 1u32;
            let mut m_norm = m as u32;
            while (m_norm & 0x0400) == 0 {
                m_norm <<= 1;
                e_norm += 1;
            }
            let shift_e = 127 - 15 - e_norm + 1;
            let shift_m = (m_norm & 0x03ff) << 13;
            f32::from_bits(((s as u32) << 31) | (shift_e << 23) | shift_m)
        }
    } else if e == 31 {
        if m == 0 {
            if s != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let shift_e = (e as u32) + 127 - 15;
        let shift_m = (m as u32) << 13;
        f32::from_bits(((s as u32) << 31) | (shift_e << 23) | shift_m)
    }
}

/// Unpacks `VK_FORMAT_B10G11R11_UFLOAT_PACK32` 32-bit word into normalized RGB [0..1].
fn unpack_b10g11r11(val: u32) -> (f32, f32, f32) {
    let r_bits = val & 0x7FF;
    let g_bits = (val >> 11) & 0x7FF;
    let b_bits = (val >> 22) & 0x3FF;

    let decode_11 = |bits: u32| -> f32 {
        let exp = (bits >> 6) & 0x1F;
        let mant = bits & 0x3F;
        if exp == 0 {
            (mant as f32) / 64.0 * (1.0 / 16384.0)
        } else {
            (1.0 + (mant as f32) / 64.0) * 2.0_f32.powi(exp as i32 - 15)
        }
    };

    let decode_10 = |bits: u32| -> f32 {
        let exp = (bits >> 5) & 0x1F;
        let mant = bits & 0x1F;
        if exp == 0 {
            (mant as f32) / 32.0 * (1.0 / 16384.0)
        } else {
            (1.0 + (mant as f32) / 32.0) * 2.0_f32.powi(exp as i32 - 15)
        }
    };

    (decode_11(r_bits), decode_11(g_bits), decode_10(b_bits))
}

/// Unpacks `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` shared-exponent 32-bit word into RGB [0..1].
fn unpack_e5b9g9r9(val: u32) -> (f32, f32, f32) {
    let r_mant = val & 0x1FF;
    let g_mant = (val >> 9) & 0x1FF;
    let b_mant = (val >> 18) & 0x1FF;
    let exp = (val >> 27) & 0x1F;

    let scale = 2.0_f32.powi(exp as i32 - 15 - 9);
    (
        r_mant as f32 * scale,
        g_mant as f32 * scale,
        b_mant as f32 * scale,
    )
}

/// Maps Vulkan `VkFormat` raw enum values to ASTC 2D block dimensions `(block_width, block_height)`.
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

/// Handles container-level supercompression schemes (Zstandard / Zlib) using central stream decoders.
fn decompress_level_payload<'a>(
    header: &ktx2::Header,
    raw_payload: &'a [u8],
) -> Result<Cow<'a, [u8]>> {
    match header.supercompression_scheme {
        None => Ok(Cow::Borrowed(raw_payload)),
        Some(ktx2::SupercompressionScheme::Zstandard) => {
            let decompressed = decompress_zstd(raw_payload)?;
            Ok(Cow::Owned(decompressed))
        }
        Some(ktx2::SupercompressionScheme::ZLIB) => {
            let decompressed = decompress_zlib(raw_payload)?;
            Ok(Cow::Owned(decompressed))
        }
        Some(ktx2::SupercompressionScheme::BasisLZ) => Ok(Cow::Borrowed(raw_payload)),
        Some(scheme) => Err(anyhow!(
            "Unsupported KTX2 supercompression scheme: {:?}",
            scheme
        )),
    }
}

/// Calculates the exact byte slice length for a single 2D face/layer at Level 0.
fn calculate_single_slice_bytes(vk_format: u32, width: usize, height: usize) -> Option<usize> {
    match vk_format {
        37 | 43 | 44 | 50 => Some(width * height * 4),
        23 | 29 | 30 | 36 => Some(width * height * 3),
        9 | 15 => Some(width * height),
        16 | 22 => Some(width * height * 2),
        97 => Some(width * height * 8),
        109 => Some(width * height * 16),
        122 | 123 => Some(width * height * 4),
        131..=134 | 139 | 140 | 147..=150 | 153 | 154 | 160 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 8)
        }
        135..=138 | 141..=146 | 151 | 152 | 155 | 156 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 16)
        }
        157..=184 | 1000066000..=1000066013 => {
            let (bx, by) = resolve_astc_block_size(vk_format)?;
            Some(width.div_ceil(bx) * height.div_ceil(by) * 16)
        }
        1000054001 | 1000054003 | 1000054005 | 1000054007 => {
            Some(width.div_ceil(4) * height.div_ceil(4) * 8)
        }
        1000054000 | 1000054002 | 1000054004 | 1000054006 => {
            Some(width.div_ceil(8) * height.div_ceil(4) * 8)
        }
        _ => None,
    }
}

/// Decodes KTX2 container textures natively using `ktx2`, `basisu`, and central `gpu_block`.
pub fn decode_ktx2_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    let reader = ktx2::Reader::new(bytes)
        .map_err(|e| anyhow!("Invalid KTX2 header or corrupted file: {:?}", e))?;

    let header = reader.header();
    let width = header.pixel_width as usize;
    let height = header.pixel_height.max(1) as usize;

    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(anyhow!(
            "Invalid or oversized KTX2 dimensions: {}x{}",
            width,
            height
        ));
    }

    let format_raw = header.format.map(|f| f.value()).unwrap_or(0);

    // BasisLZ / UASTC transcoding path
    if header.supercompression_scheme == Some(ktx2::SupercompressionScheme::BasisLZ)
        || format_raw == 0
    {
        let mut basis_payload = Cow::Borrowed(bytes);

        if header.pixel_depth > 1 {
            let mut patched = bytes.to_vec();
            patched[28..32].copy_from_slice(&0u32.to_le_bytes());
            let layers = header.layer_count.max(1);
            let new_layers = layers * header.pixel_depth;
            patched[32..36].copy_from_slice(&new_layers.to_le_bytes());
            patched[40..44].copy_from_slice(&1u32.to_le_bytes());
            basis_payload = Cow::Owned(patched);
        }

        let tex = basisu::Transcoder::new(&basis_payload)
            .map_err(|e| anyhow!("Basis Universal transcoder init failed: {:?}", e))?;

        let transcoded_rgba = tex
            .transcode_image(
                0,
                0,
                0,
                basisu::TargetFormat::Rgba32,
                basisu::DecodeFlags::NONE,
            )
            .map_err(|e| anyhow!("Basis Universal transcoding failed: {:?}", e))?;

        let img = image::RgbaImage::from_raw(width as u32, height as u32, transcoded_rgba)
            .ok_or_else(|| anyhow!("Failed to build RGBA image from Basis buffer"))?;

        return Ok(DynamicImage::ImageRgba8(img));
    }

    let first_level = reader
        .levels()
        .next()
        .ok_or_else(|| anyhow!("KTX2 container contains no mip levels"))?;

    let decompressed_payload = decompress_level_payload(&header, first_level.data)?;
    let level_data: &[u8] = &decompressed_payload;

    let slice_bytes =
        calculate_single_slice_bytes(format_raw, width, height).unwrap_or(level_data.len());
    let slice_data = &level_data[..level_data.len().min(slice_bytes)];

    let block_format = match format_raw {
        23 | 29 => {
            let mut out_rgba = vec![255u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(3).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                out_rgba[i * 4] = chunk[0];
                out_rgba[i * 4 + 1] = chunk[1];
                out_rgba[i * 4 + 2] = chunk[2];
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from RGB8 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        30 | 36 => {
            let mut out_rgba = vec![255u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(3).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                out_rgba[i * 4] = chunk[2];
                out_rgba[i * 4 + 1] = chunk[1];
                out_rgba[i * 4 + 2] = chunk[0];
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from BGR8 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        37 | 43 => {
            let copy_len = slice_data.len().min(width * height * 4);
            let img = image::RgbaImage::from_raw(
                width as u32,
                height as u32,
                slice_data[..copy_len].to_vec(),
            )
            .ok_or_else(|| anyhow!("Failed to build RGBA image from UNORM data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        44 | 50 => {
            let copy_len = slice_data.len().min(width * height * 4);
            let mut bgra_buf = slice_data[..copy_len].to_vec();
            bgra_to_rgba_in_place(&mut bgra_buf);
            let img = image::RgbaImage::from_raw(width as u32, height as u32, bgra_buf)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from BGRA data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        97 => {
            let mut out_rgba = vec![0u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(8).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                let r16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g16 = u16::from_le_bytes([chunk[2], chunk[3]]);
                let b16 = u16::from_le_bytes([chunk[4], chunk[5]]);
                let a16 = u16::from_le_bytes([chunk[6], chunk[7]]);

                out_rgba[i * 4] = (f16_to_f32(r16).clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 1] = (f16_to_f32(g16).clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 2] = (f16_to_f32(b16).clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 3] = (f16_to_f32(a16).clamp(0.0, 1.0) * 255.0) as u8;
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from f16 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        109 => {
            let mut out_rgba = vec![0u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(16).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                let rf = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let gf = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let bf = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                let af = f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);

                out_rgba[i * 4] = (rf.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 1] = (gf.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 2] = (bf.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 3] = (af.clamp(0.0, 1.0) * 255.0) as u8;
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from f32 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        122 => {
            let mut out_rgba = vec![255u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(4).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let (r, g, b) = unpack_b10g11r11(val);
                out_rgba[i * 4] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from B10G11R11 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        123 => {
            let mut out_rgba = vec![255u8; width * height * 4];
            for (i, chunk) in slice_data.chunks_exact(4).enumerate() {
                if i * 4 + 3 >= out_rgba.len() {
                    break;
                }
                let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let (r, g, b) = unpack_e5b9g9r9(val);
                out_rgba[i * 4] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                out_rgba[i * 4 + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            }
            let img = image::RgbaImage::from_raw(width as u32, height as u32, out_rgba)
                .ok_or_else(|| anyhow!("Failed to build RGBA image from E5B9G9R9 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        131..=134 => TextureBlockFormat::Bc1,
        135 | 136 => TextureBlockFormat::Bc2,
        137 | 138 => TextureBlockFormat::Bc3,
        139 | 140 => TextureBlockFormat::Bc4,
        141 | 142 => TextureBlockFormat::Bc5,
        143 => TextureBlockFormat::Bc6Unsigned,
        144 => TextureBlockFormat::Bc6Signed,
        145 | 146 => TextureBlockFormat::Bc7,
        160 => TextureBlockFormat::Etc1,
        147 | 148 => TextureBlockFormat::Etc2Rgb,
        149 | 150 => TextureBlockFormat::Etc2Rgba1,
        151 | 152 => TextureBlockFormat::Etc2Rgba8,
        153 => TextureBlockFormat::EacR,
        154 => TextureBlockFormat::EacRSigned,
        155 => TextureBlockFormat::EacRg,
        156 => TextureBlockFormat::EacRgSigned,

        157..=184 | 1000066000..=1000066013 => {
            let (bx, by) = resolve_astc_block_size(format_raw).ok_or_else(|| {
                anyhow!("Unrecognized ASTC VkFormat enum identifier: {}", format_raw)
            })?;
            TextureBlockFormat::Astc {
                block_x: bx,
                block_y: by,
            }
        }

        1000054000..=1000054007 => {
            let is_2bpp = matches!(
                format_raw,
                1000054000 | 1000054002 | 1000054004 | 1000054006
            );
            TextureBlockFormat::Pvrtc { is_2bpp }
        }

        _ => return Err(anyhow!("Unsupported VkFormat enum ID: {}", format_raw)),
    };

    decode_gpu_block(slice_data, width, height, block_format)
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
