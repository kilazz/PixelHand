// src/format_loaders/dds_loader.rs

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use rayon::prelude::*;
use std::borrow::Cow;
use std::io::Cursor;
use std::path::Path;

use crate::core::qc::QcImageMetadata;
use crate::core::tonemapper::TonemapConfig;
use crate::format_loaders::ImageFormatLoader;

const FOURCC_DXT1: u32 = u32::from_le_bytes(*b"DXT1");
const FOURCC_DXT2: u32 = u32::from_le_bytes(*b"DXT2");
const FOURCC_DXT3: u32 = u32::from_le_bytes(*b"DXT3");
const FOURCC_DXT4: u32 = u32::from_le_bytes(*b"DXT4");
const FOURCC_DXT5: u32 = u32::from_le_bytes(*b"DXT5");
const FOURCC_DX10: u32 = u32::from_le_bytes(*b"DX10");
const FOURCC_ATI1: u32 = u32::from_le_bytes(*b"ATI1");
const FOURCC_BC4U: u32 = u32::from_le_bytes(*b"BC4U");
const FOURCC_BC4S: u32 = u32::from_le_bytes(*b"BC4S");
const FOURCC_ATI2: u32 = u32::from_le_bytes(*b"ATI2");
const FOURCC_BC5U: u32 = u32::from_le_bytes(*b"BC5U");
const FOURCC_BC5S: u32 = u32::from_le_bytes(*b"BC5S");

const FOURCC_CRYF: u32 = u32::from_le_bytes(*b"CRYF");
const FOURCC_FYRC: u32 = u32::from_le_bytes(*b"FYRC");

struct AnalyzedHeaderInfo {
    width: u32,
    height: u32,
    fourcc: u32,
    is_swizzled: bool,
    is_xbox: bool,
    pixel_data_offset: usize,
    block_bytes: usize,
    pitch: usize,
    slice_pitch: usize,
    is_compressed: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MipSelectionStrategy {
    Auto(Option<u32>),
    Specific(u32),
}

pub struct DdsPayloadProcessor;

impl DdsPayloadProcessor {
    pub fn unswizzle_block_linear(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        block_bytes: usize,
    ) -> Result<()> {
        let block_width = width.div_ceil(4);
        let block_height = height.div_ceil(4);
        let required_size = (block_width as usize) * (block_height as usize) * block_bytes;

        if src.len() < required_size || dst.len() < required_size {
            return Err(anyhow!(
                "Source or destination buffer is too small for unswizzle operation"
            ));
        }

        let log_w = if block_width > 1 {
            (block_width - 1).ilog2() + 1
        } else {
            0
        };
        let log_h = if block_height > 1 {
            (block_height - 1).ilog2() + 1
        } else {
            0
        };
        let min_log = std::cmp::min(log_w, log_h);
        let row_bytes = (block_width as usize) * block_bytes;

        dst.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(y_idx, row_chunk)| {
                let y = y_idx as u32;
                for x in 0..block_width {
                    let mut swizzled_index = 0u32;
                    for i in 0..min_log {
                        swizzled_index |= ((x >> i) & 1) << (2 * i);
                        swizzled_index |= ((y >> i) & 1) << (2 * i + 1);
                    }
                    if log_w > log_h {
                        for i in min_log..log_w {
                            swizzled_index |= ((x >> i) & 1) << (i + min_log);
                        }
                    } else {
                        for i in min_log..log_h {
                            swizzled_index |= ((y >> i) & 1) << (i + min_log);
                        }
                    }

                    let src_offset = (swizzled_index as usize) * block_bytes;
                    let dst_offset = (x as usize) * block_bytes;

                    if let Some(src_slice) = src.get(src_offset..src_offset + block_bytes)
                        && let Some(dst_slice) =
                            row_chunk.get_mut(dst_offset..dst_offset + block_bytes)
                    {
                        dst_slice.copy_from_slice(src_slice);
                    }
                }
            });
        Ok(())
    }

    pub fn perform_xbox_endian_swap(data: &mut [u8], format_fourcc: u32) {
        if matches!(
            format_fourcc,
            FOURCC_DXT1 | FOURCC_ATI1 | FOURCC_BC4U | FOURCC_BC4S
        ) {
            data.par_chunks_mut(8).for_each(|chunk| {
                if chunk.len() >= 4 {
                    chunk.swap(0, 1);
                    chunk.swap(2, 3);
                }
            });
        } else if matches!(
            format_fourcc,
            FOURCC_DXT2 | FOURCC_DXT3 | FOURCC_DXT4 | FOURCC_DXT5
        ) {
            let is_bc3 = format_fourcc == FOURCC_DXT4 || format_fourcc == FOURCC_DXT5;
            data.par_chunks_mut(16).for_each(|chunk| {
                if chunk.len() >= 12 {
                    chunk.swap(8, 9);
                    chunk.swap(10, 11);
                    if is_bc3 && chunk.len() >= 8 {
                        chunk.swap(2, 3);
                        chunk.swap(4, 5);
                        chunk.swap(6, 7);
                    }
                }
            });
        } else if matches!(format_fourcc, FOURCC_ATI2 | FOURCC_BC5U | FOURCC_BC5S) {
            data.par_chunks_mut(16).for_each(|chunk| {
                if chunk.len() >= 16 {
                    chunk.swap(2, 3);
                    chunk.swap(4, 5);
                    chunk.swap(6, 7);
                    chunk.swap(10, 11);
                    chunk.swap(12, 13);
                    chunk.swap(14, 15);
                }
            });
        } else {
            data.par_chunks_mut(2).for_each(|chunk| {
                if chunk.len() >= 2 {
                    chunk.swap(0, 1);
                }
            });
        }
    }
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes
        .get(offset..offset + 4)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}

fn is_xbox_360_format(fourcc: u32) -> bool {
    matches!(
        fourcc,
        0x44585431
            | 0x44585433
            | 0x44585435
            | 0x44583130
            | 0x41544931
            | 0x41544932
            | 0x42433455
            | 0x42433555
    )
}

fn xbox_360_to_standard_fourcc(fourcc: u32) -> u32 {
    match fourcc {
        0x44585431 => FOURCC_DXT1,
        0x44585433 => FOURCC_DXT3,
        0x44585435 => FOURCC_DXT5,
        0x44583130 => FOURCC_DX10,
        0x41544931 => FOURCC_ATI1,
        0x41544932 => FOURCC_ATI2,
        0x42433455 => FOURCC_BC4U,
        0x42433555 => FOURCC_BC5U,
        _ => fourcc,
    }
}

fn analyze_header(dds_bytes: &[u8]) -> Result<AnalyzedHeaderInfo> {
    if dds_bytes.len() < 128 {
        return Err(anyhow!("Input data too small for a DDS file"));
    }
    if dds_bytes.get(0..4) != Some(b"DDS ") {
        return Err(anyhow!("Not a DDS file"));
    }

    let dw_size = read_u32_le(dds_bytes, 4).context("Failed to read header size")?;
    if dw_size != 124 {
        return Err(anyhow!("Invalid DDS header size"));
    }

    let height = read_u32_le(dds_bytes, 12).context("Failed to read height")?;
    let width = read_u32_le(dds_bytes, 16).context("Failed to read width")?;
    let pf_size = read_u32_le(dds_bytes, 76).context("Failed to read pixel format size")?;
    if pf_size != 32 {
        return Err(anyhow!("Invalid DDS_PIXELFORMAT size"));
    }

    let pf_flags = read_u32_le(dds_bytes, 80).context("Failed to read pixel format flags")?;
    let mut pf_fourcc = read_u32_le(dds_bytes, 84).context("Failed to read pixel format FourCC")?;
    let pf_rgb_bitcount =
        read_u32_le(dds_bytes, 88).context("Failed to read pixel format bitcount")?;

    let is_xbox = is_xbox_360_format(pf_fourcc);
    if is_xbox {
        pf_fourcc = xbox_360_to_standard_fourcc(pf_fourcc);
    }

    let mut is_compressed = false;
    let mut block_bytes = 16;
    let mut header_size = 128;

    let ddpf_fourcc = 0x00000004;
    let ddpf_rgb = 0x00000040;

    if (pf_flags & ddpf_fourcc) != 0 {
        if pf_fourcc == FOURCC_DX10 {
            header_size += 20;
            if dds_bytes.len() < header_size {
                return Err(anyhow!("DX10 header out of bounds"));
            }
            let dxgi = read_u32_le(dds_bytes, 128).context("Failed to read DX10 DXGI format")?;
            if matches!(dxgi, 71 | 72 | 80 | 81) {
                is_compressed = true;
                block_bytes = 8;
            } else if matches!(
                dxgi,
                74 | 75 | 77 | 78 | 83 | 84 | 94 | 95 | 96 | 97 | 98 | 99
            ) {
                is_compressed = true;
                block_bytes = 16;
            }
        } else {
            if matches!(
                pf_fourcc,
                FOURCC_DXT1 | FOURCC_ATI1 | FOURCC_BC4U | FOURCC_BC4S
            ) {
                is_compressed = true;
                block_bytes = 8;
            } else if matches!(
                pf_fourcc,
                FOURCC_DXT2
                    | FOURCC_DXT3
                    | FOURCC_DXT4
                    | FOURCC_DXT5
                    | FOURCC_ATI2
                    | FOURCC_BC5U
                    | FOURCC_BC5S
            ) {
                is_compressed = true;
                block_bytes = 16;
            }
        }
    }

    let mut data_ptr_offset = header_size;
    let mut is_swizzled = false;

    if data_ptr_offset + 4 <= dds_bytes.len() {
        let marker = read_u32_le(dds_bytes, data_ptr_offset).context("Failed to read marker")?;
        if marker == FOURCC_CRYF || marker == FOURCC_FYRC {
            is_swizzled = marker == FOURCC_CRYF;
            data_ptr_offset += if marker == FOURCC_CRYF { 8 } else { 4 };
            if (pf_flags & ddpf_rgb) != 0 {
                is_compressed = true;
                block_bytes = 8;
            }
        }
    }

    if data_ptr_offset > dds_bytes.len() {
        return Err(anyhow!("DDS offset out of bounds"));
    }

    let (pitch, slice_pitch) = if is_compressed {
        let num_blocks_wide = width.div_ceil(4).max(1) as usize;
        let num_blocks_high = height.div_ceil(4).max(1) as usize;
        let p = num_blocks_wide * block_bytes;
        (p, p * num_blocks_high)
    } else {
        let bpp = pf_rgb_bitcount as usize;
        let p = (width as usize * bpp).div_ceil(8);
        (p, p * height as usize)
    };

    Ok(AnalyzedHeaderInfo {
        width,
        height,
        fourcc: pf_fourcc,
        is_swizzled,
        is_xbox,
        pixel_data_offset: data_ptr_offset,
        block_bytes,
        pitch,
        slice_pitch,
        is_compressed,
    })
}

fn map_dxgi_format_to_string(dxgi_format: u32) -> String {
    match dxgi_format {
        2 => "RGBA32 Float".to_string(),
        10 => "RGBA16 Float".to_string(),
        11 => "RGBA16 UNORM".to_string(),
        24 => "RGB10A2 UNORM".to_string(),
        28 => "RGBA8 UNORM".to_string(),
        29 => "RGBA8 sRGB".to_string(),
        54 => "R16 Float".to_string(),
        56 => "R16 UNORM".to_string(),
        61 => "R8 UNORM".to_string(),
        70 | 71 => "BC1 (DXT1)".to_string(),
        72 => "BC1 sRGB (DXT1)".to_string(),
        73 | 74 => "BC2 (DXT3)".to_string(),
        75 => "BC2 sRGB (DXT3)".to_string(),
        76 | 77 => "BC3 (DXT5)".to_string(),
        78 => "BC3 sRGB (DXT5)".to_string(),
        79 | 80 => "BC4 (ATI1)".to_string(),
        81 => "BC4 SNORM".to_string(),
        82 | 83 => "BC5 (ATI2)".to_string(),
        84 => "BC5 SNORM".to_string(),
        87 => "BGRA8 UNORM".to_string(),
        91 => "BGRA8 sRGB".to_string(),
        94 | 95 => "BC6H (UF16)".to_string(),
        96 => "BC6H (SF16)".to_string(),
        97 | 98 => "BC7".to_string(),
        99 => "BC7 sRGB".to_string(),
        _ => format!("DX10 (DXGI {})", dxgi_format),
    }
}

fn decode_dds_internal(dds_bytes: &[u8], strategy: MipSelectionStrategy) -> Result<DynamicImage> {
    let info = analyze_header(dds_bytes)?;

    let clean_dds_bytes = if info.is_swizzled || info.is_xbox {
        let mut pixel_data = if info.is_swizzled {
            let mut unswizzled = vec![0u8; info.slice_pitch];
            DdsPayloadProcessor::unswizzle_block_linear(
                &dds_bytes[info.pixel_data_offset..],
                &mut unswizzled,
                info.width,
                info.height,
                info.block_bytes,
            )?;
            unswizzled
        } else {
            let limit = dds_bytes
                .len()
                .min(info.pixel_data_offset + info.slice_pitch);
            let mut raw_pixels = vec![0u8; info.slice_pitch];
            let bytes_to_copy = limit.saturating_sub(info.pixel_data_offset);

            if let Some(src_slice) =
                dds_bytes.get(info.pixel_data_offset..info.pixel_data_offset + bytes_to_copy)
            {
                raw_pixels[0..bytes_to_copy].copy_from_slice(src_slice);
            }
            raw_pixels
        };

        if info.is_xbox {
            DdsPayloadProcessor::perform_xbox_endian_swap(&mut pixel_data, info.fourcc);
        }

        let mut clean = Vec::with_capacity(128 + pixel_data.len());
        clean.extend_from_slice(b"DDS ");

        let mut dds_header = [0u8; 124];
        dds_header[0..4].copy_from_slice(&124u32.to_le_bytes());

        let mut dw_flags: u32 = 0x1 | 0x2 | 0x4 | 0x1000;
        dw_flags |= if info.is_compressed { 0x80000 } else { 0x8 };
        dds_header[4..8].copy_from_slice(&dw_flags.to_le_bytes());
        dds_header[8..12].copy_from_slice(&info.height.to_le_bytes());
        dds_header[12..16].copy_from_slice(&info.width.to_le_bytes());

        let dw_pitch = if info.is_compressed {
            info.slice_pitch as u32
        } else {
            info.pitch as u32
        };
        dds_header[16..20].copy_from_slice(&dw_pitch.to_le_bytes());
        dds_header[20..24].copy_from_slice(&1u32.to_le_bytes());
        dds_header[24..28].copy_from_slice(&1u32.to_le_bytes());

        dds_header[72..76].copy_from_slice(&32u32.to_le_bytes());
        let pf_flags: u32 = if info.is_compressed {
            0x00000004
        } else {
            0x00000040 | 0x00000001
        };
        dds_header[76..80].copy_from_slice(&pf_flags.to_le_bytes());

        let pf_fourcc = if info.is_compressed { info.fourcc } else { 0 };
        dds_header[80..84].copy_from_slice(&pf_fourcc.to_le_bytes());

        let pf_rgb_bitcount: u32 = if info.is_compressed { 0 } else { 32 };
        dds_header[84..88].copy_from_slice(&pf_rgb_bitcount.to_le_bytes());
        if !info.is_compressed {
            dds_header[88..92].copy_from_slice(&0x00ff0000u32.to_le_bytes());
            dds_header[92..96].copy_from_slice(&0x0000ff00u32.to_le_bytes());
            dds_header[96..100].copy_from_slice(&0x000000ffu32.to_le_bytes());
            dds_header[100..104].copy_from_slice(&0xff000000u32.to_le_bytes());
        }

        dds_header[104..108].copy_from_slice(&0x1000u32.to_le_bytes());
        clean.extend_from_slice(&dds_header);

        if info.fourcc == FOURCC_DX10
            && let Some(dx10_header) = dds_bytes.get(128..148)
        {
            clean.extend_from_slice(dx10_header);
        }

        clean.extend_from_slice(&pixel_data);
        Cow::Owned(clean)
    } else {
        Cow::Borrowed(dds_bytes)
    };

    let dds = image_dds::ddsfile::Dds::read(Cursor::new(&*clean_dds_bytes))
        .context("Failed to parse DDS file structure")?;

    let level = match strategy {
        MipSelectionStrategy::Auto(target_size) => {
            if let Some(target) = target_size {
                let mut level = 0;
                let mut w = dds.header.width;
                let mut h = dds.header.height;
                for mip in 0..dds.header.mip_map_count.unwrap_or(1) {
                    if w >= target && h >= target {
                        level = mip;
                    } else {
                        break;
                    }
                    w /= 2;
                    h /= 2;
                }
                level
            } else {
                0
            }
        }
        MipSelectionStrategy::Specific(mip_level) => {
            let max_mips = dds.header.mip_map_count.unwrap_or(1);
            mip_level.min(max_mips.saturating_sub(1))
        }
    };

    let rgba_img = image_dds::image_from_dds(&dds, level)
        .context("Failed to decode compressed DDS payload")?;
    Ok(DynamicImage::ImageRgba8(rgba_img))
}

pub fn decode_dds_bytes(dds_bytes: &[u8], target_size: Option<u32>) -> Result<DynamicImage> {
    decode_dds_internal(dds_bytes, MipSelectionStrategy::Auto(target_size))
}

pub fn decode_dds_bytes_specific_mip(dds_bytes: &[u8], mip_level: u32) -> Result<DynamicImage> {
    decode_dds_internal(dds_bytes, MipSelectionStrategy::Specific(mip_level))
}

// ---------------------------------------------------------
// POLYMORPHIC LOADER INTEGRATION
// ---------------------------------------------------------

pub struct DdsLoader;
impl ImageFormatLoader for DdsLoader {
    fn extensions(&self) -> &[&str] {
        &["dds"]
    }

    fn decode(
        &self,
        path: &Path,
        target_size: Option<u32>,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let file = std::fs::File::open(path).context("Failed to open DDS file")?;
        let mmap = unsafe { memmap2::Mmap::map(&file).context("Failed to memory map DDS file")? };
        decode_dds_bytes(&mmap, target_size)
    }

    fn decode_specific_mip(
        &self,
        path: &Path,
        mip_level: u32,
        _tonemap_config: Option<TonemapConfig>,
    ) -> Result<DynamicImage> {
        let file = std::fs::File::open(path).context("Failed to open DDS file")?;
        let mmap = unsafe { memmap2::Mmap::map(&file).context("Failed to memory map DDS file")? };
        decode_dds_bytes_specific_mip(&mmap, mip_level)
    }

    fn extract_metadata(&self, path: &Path) -> Result<QcImageMetadata> {
        let size = std::fs::metadata(path)?.len();
        let bytes = std::fs::read(path)?;

        if bytes.len() < 128 {
            return Err(anyhow!("DDS is too small to contain a valid header"));
        }
        if bytes.get(0..4) != Some(b"DDS ") {
            return Err(anyhow!("Missing or invalid DDS magic bytes"));
        }

        let dw_size = read_u32_le(&bytes, 4).unwrap_or(0);
        if dw_size != 124 {
            return Err(anyhow!("Invalid standard DDS header size field"));
        }

        let h = read_u32_le(&bytes, 12).unwrap_or(0);
        let w = read_u32_le(&bytes, 16).unwrap_or(0);
        let mips = read_u32_le(&bytes, 28).unwrap_or(0);
        let mipmap_count = if mips > 0 { mips } else { 1 };

        let dw_caps2 = read_u32_le(&bytes, 112).unwrap_or(0);
        let is_cubemap = (dw_caps2 & 0x0000FE00) != 0;

        let mut has_alpha = false;
        let pf_flags = read_u32_le(&bytes, 80).unwrap_or(0);
        if (pf_flags & 0x1) != 0 || (pf_flags & 0x2) != 0 {
            has_alpha = true;
        }

        let mut compression_format = "DDS".to_string();
        let mut color_space = "sRGB".to_string();
        let mut bit_depth = 8;

        let pf_fourcc = read_u32_le(&bytes, 84).unwrap_or(0);
        if pf_fourcc != 0 {
            if let Some(fourcc_slice) = bytes.get(84..88) {
                compression_format = String::from_utf8_lossy(fourcc_slice).trim().to_string();
            }
            if compression_format == "DXT3" || compression_format == "DXT5" {
                has_alpha = true;
            }

            if pf_fourcc == u32::from_le_bytes(*b"DX10") && bytes.len() >= 148 {
                let dxgi_format = read_u32_le(&bytes, 128).unwrap_or(0);
                compression_format = map_dxgi_format_to_string(dxgi_format);
                if matches!(dxgi_format, 74 | 75 | 77 | 78 | 98 | 99) {
                    has_alpha = true;
                }
                if matches!(dxgi_format, 71 | 74 | 77 | 80 | 83 | 98) {
                    color_space = "Linear".to_string();
                }
            }
        } else {
            compression_format = "Uncompressed".to_string();
            let rgb_bit_count = read_u32_le(&bytes, 88).unwrap_or(0);
            bit_depth = if rgb_bit_count / 4 == 0 {
                8
            } else {
                rgb_bit_count / 4
            };
        }

        let estimated_vram =
            crate::core::qc::estimate_vram(w, h, &compression_format, mipmap_count, is_cubemap);

        Ok(QcImageMetadata {
            width: w,
            height: h,
            file_size: size,
            format_str: "dds".to_string(),
            compression_format,
            color_space,
            has_alpha,
            bit_depth,
            mipmap_count,
            is_cubemap,
            estimated_vram,
        })
    }
}
