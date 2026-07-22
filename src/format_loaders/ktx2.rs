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

/// Bitwise conversion of IEEE 754 half-precision float (f16) to single-precision float (f32).
/// Avoids external dependencies by implementing bitwise exponent bias shifting.
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 0x0001;
    let e = (h >> 10) & 0x001f;
    let m = h & 0x03ff;

    if e == 0 {
        if m == 0 {
            if s != 0 { -0.0 } else { 0.0 }
        } else {
            // Normalize subnormal float values
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
/// - Red: 11 bits (5 exponent, 6 mantissa)
/// - Green: 11 bits (5 exponent, 6 mantissa)
/// - Blue: 10 bits (5 exponent, 5 mantissa)
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
/// - Red: 9-bit mantissa
/// - Green: 9-bit mantissa
/// - Blue: 9-bit mantissa
/// - Shared Exponent: 5 bits
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

/// Handles container-level supercompression schemes (e.g. Zstandard / Zlib) for non-Basis textures.
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
        Some(ktx2::SupercompressionScheme::BasisLZ) => {
            // Handled directly via basis-universal transcoder pipeline
            Ok(Cow::Borrowed(raw_payload))
        }
        Some(scheme) => Err(anyhow!(
            "Unsupported KTX2 supercompression scheme: {:?}",
            scheme
        )),
    }
}

/// Calculates the exact byte slice length for a single 2D face/layer at Level 0.
/// Prevents extra array layers or cubemap faces (6x memory size) from overflowing 2D decoders.
fn calculate_single_slice_bytes(vk_format: u32, width: usize, height: usize) -> Option<usize> {
    match vk_format {
        // Uncompressed 8-bit RGBA / BGRA
        37 | 43 | 44 | 50 => Some(width * height * 4),
        // Uncompressed 8-bit RGB / BGR
        23 | 29 | 30 | 36 => Some(width * height * 3),
        // Uncompressed 8-bit R
        9 | 15 => Some(width * height),
        // Uncompressed 8-bit RG
        16 | 22 => Some(width * height * 2),
        // Uncompressed 16-bit Float RGBA (97 = VK_FORMAT_R16G16B16A16_SFLOAT)
        97 => Some(width * height * 8),
        // Uncompressed 32-bit Float RGBA (109 = VK_FORMAT_R32G32B32A32_SFLOAT)
        109 => Some(width * height * 16),
        // Packed 32-bit Floats (122 = B10G11R11, 123 = E5B9G9R9)
        122 | 123 => Some(width * height * 4),

        // BC1, BC4, ETC1, ETC2 RGB, ETC2 RGBA1, EAC R11 (8 bytes per 4x4 block)
        131..=134 | 139 | 140 | 147..=150 | 153 | 154 | 160 => {
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

/// Decodes KTX2 container textures natively using `ktx2`, `basisu`, and `texture2ddecoder`.
pub fn decode_ktx2_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    // Parse container header and level index tables
    let reader = ktx2::Reader::new(bytes)
        .map_err(|e| anyhow!("Invalid KTX2 header or corrupted file: {:?}", e))?;

    let header = reader.header();
    let width = header.pixel_width as usize;
    let height = header.pixel_height.max(1) as usize;

    // Safety Bounds Check: Guard against memory exhaustion from malformed or corrupted headers
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(anyhow!(
            "Invalid or oversized KTX2 dimensions: {}x{}",
            width,
            height
        ));
    }

    let format_raw = header.format.map(|f| f.value()).unwrap_or(0);

    // --- TRANSCODING PATH FOR BASISLZ (ETC1S) AND UASTC FORMATS ---
    if header.supercompression_scheme == Some(ktx2::SupercompressionScheme::BasisLZ)
        || format_raw == 0
    {
        // The basisu transcoder does not support 3D Volume textures (pixel_depth > 1),
        // rejecting them with an InvalidData error. Since we only need the first 2D slice (z=0)
        // for the thumbnail preview, we can elegantly trick the transcoder by patching the KTX2 header
        // in memory to pretend it's a 2D Texture Array, which basisu natively supports.
        let mut basis_payload = Cow::Borrowed(bytes);

        if header.pixel_depth > 1 {
            let mut patched = bytes.to_vec();
            // 1. Set pixel_depth (bytes 28..32) to 0
            patched[28..32].copy_from_slice(&0u32.to_le_bytes());

            // 2. Map Z-slices to Array Layers: new_layers = max(layers, 1) * pixel_depth
            let layers = header.layer_count.max(1);
            let new_layers = layers * header.pixel_depth;
            patched[32..36].copy_from_slice(&new_layers.to_le_bytes());

            // 3. Force level_count (bytes 40..44) to 1 to bypass mipmap size validation mismatches
            //    (since 3D mips reduce in Z-axis, but 2D Array mips do not)
            patched[40..44].copy_from_slice(&1u32.to_le_bytes());

            basis_payload = Cow::Owned(patched);
        }

        let tex = basisu::Transcoder::new(&basis_payload).map_err(|e| {
            anyhow!(
                "Basis Universal transcoder failed to initialize for KTX2 stream: {:?}",
                e
            )
        })?;

        // Decode exactly the first face of the first layer of the first mip level
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
            .ok_or_else(|| anyhow!("Failed to build RGBA image from Basis transcoded buffer"))?;

        return Ok(DynamicImage::ImageRgba8(img));
    }

    // --- STANDARD GPU FORMATS DECODING PATH ---
    let first_level = reader
        .levels()
        .next()
        .ok_or_else(|| anyhow!("KTX2 container contains no mip levels"))?;

    let decompressed_payload = decompress_level_payload(&header, first_level.data)?;
    let level_data: &[u8] = &decompressed_payload;

    // Isolate single 2D slice for 3D volumes, cubemaps (6 faces), or 2D array layers
    let slice_bytes =
        calculate_single_slice_bytes(format_raw, width, height).unwrap_or(level_data.len());
    let slice_data = &level_data[..level_data.len().min(slice_bytes)];

    let mut rgba_u32 = vec![0u32; width * height];

    match format_raw {
        // VK_FORMAT_R8G8B8_UNORM (23) | VK_FORMAT_R8G8B8_SRGB (29)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 RGB8 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // VK_FORMAT_B8G8R8_UNORM (30) | VK_FORMAT_B8G8R8_SRGB (36)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 BGR8 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

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

        // VK_FORMAT_R16G16B16A16_SFLOAT (97)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 f16 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // VK_FORMAT_R32G32B32A32_SFLOAT (109)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 f32 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // VK_FORMAT_B10G11R11_UFLOAT_PACK32 (122)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 B10G11R11 data"))?;
            return Ok(DynamicImage::ImageRgba8(img));
        }

        // VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 (123)
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
                .ok_or_else(|| anyhow!("Failed to build RGBA image from KTX2 E5B9G9R9 data"))?;
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

        // EAC R11 UNORM (153)
        153 => {
            texture2ddecoder::decode_eacr(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R11 decoding failed: {:?}", e))?;
        }

        // EAC R11 SNORM (154)
        154 => {
            texture2ddecoder::decode_eacr_signed(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R11 Signed decoding failed: {:?}", e))?;
        }

        // EAC RG11 UNORM (155)
        155 => {
            texture2ddecoder::decode_eacrg(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG11 decoding failed: {:?}", e))?;
        }

        // EAC RG11 SNORM (156)
        156 => {
            texture2ddecoder::decode_eacrg_signed(slice_data, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG11 Signed decoding failed: {:?}", e))?;
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
            return Err(anyhow!("Unsupported VkFormat enum ID: {}", format_raw));
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
