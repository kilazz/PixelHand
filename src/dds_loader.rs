use image::DynamicImage;
use std::io::Cursor;

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

// CryEngine markers
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
        0x44585431 | // "1TXD"
        0x44585433 | // "3TXD"
        0x44585435 | // "5TXD"
        0x44583130 | // "01XD"
        0x41544931 | // "1ITA"
        0x41544932 | // "2ITA"
        0x42433455 | // "U4CB"
        0x42433555 // "U5CB"
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

fn unswizzle_block_linear(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    block_bytes: usize,
) -> Result<(), String> {
    let block_width = width.div_ceil(4);
    let block_height = height.div_ceil(4);
    let required_size = (block_width as usize) * (block_height as usize) * block_bytes;
    if src.len() < required_size || dst.len() < required_size {
        return Err("Buffer too small for unswizzle operation".to_string());
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

    use rayon::prelude::*;
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
                if src_offset + block_bytes <= src.len()
                    && dst_offset + block_bytes <= row_chunk.len()
                {
                    row_chunk[dst_offset..dst_offset + block_bytes]
                        .copy_from_slice(&src[src_offset..src_offset + block_bytes]);
                }
            }
        });
    Ok(())
}

fn perform_xbox_endian_swap(data: &mut [u8], format_fourcc: u32) {
    use rayon::prelude::*;
    if format_fourcc == FOURCC_DXT1
        || format_fourcc == FOURCC_ATI1
        || format_fourcc == FOURCC_BC4U
        || format_fourcc == FOURCC_BC4S
    {
        let block_size = 8;
        data.par_chunks_mut(block_size).for_each(|chunk| {
            if chunk.len() >= 4 {
                chunk.swap(0, 1);
                chunk.swap(2, 3);
            }
        });
    } else if format_fourcc == FOURCC_DXT2
        || format_fourcc == FOURCC_DXT3
        || format_fourcc == FOURCC_DXT4
        || format_fourcc == FOURCC_DXT5
    {
        let block_size = 16;
        let is_bc3 = format_fourcc == FOURCC_DXT4 || format_fourcc == FOURCC_DXT5;
        data.par_chunks_mut(block_size).for_each(|chunk| {
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
    } else if format_fourcc == FOURCC_ATI2
        || format_fourcc == FOURCC_BC5U
        || format_fourcc == FOURCC_BC5S
    {
        let block_size = 16;
        data.par_chunks_mut(block_size).for_each(|chunk| {
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

fn analyze_header(dds_bytes: &[u8]) -> Result<AnalyzedHeaderInfo, String> {
    if dds_bytes.len() < 128 {
        return Err("Input data too small for a DDS file".to_string());
    }
    if dds_bytes.get(0..4) != Some(b"DDS ") {
        return Err("Not a DDS file (missing 'DDS ' magic number)".to_string());
    }

    let dw_size = read_u32_le(dds_bytes, 4).ok_or("Failed to read header size")?;
    if dw_size != 124 {
        return Err("Invalid DDS header size".to_string());
    }

    let height = read_u32_le(dds_bytes, 12).ok_or("Failed to read height")?;
    let width = read_u32_le(dds_bytes, 16).ok_or("Failed to read width")?;

    // Pixel format (offset 76)
    let pf_size = read_u32_le(dds_bytes, 76).ok_or("Failed to read pixel format size")?;
    if pf_size != 32 {
        return Err("Invalid DDS_PIXELFORMAT size".to_string());
    }

    let pf_flags = read_u32_le(dds_bytes, 80).ok_or("Failed to read pixel format flags")?;
    let mut pf_fourcc = read_u32_le(dds_bytes, 84).ok_or("Failed to read pixel format FourCC")?;
    let pf_rgb_bitcount =
        read_u32_le(dds_bytes, 88).ok_or("Failed to read pixel format bitcount")?;

    let is_xbox = is_xbox_360_format(pf_fourcc);
    if is_xbox {
        pf_fourcc = xbox_360_to_standard_fourcc(pf_fourcc);
    }

    let mut is_compressed = false;
    let mut block_bytes = 16;

    let ddpf_fourcc = 0x00000004;
    let ddpf_rgb = 0x00000040;

    let mut header_size = 128;

    if (pf_flags & ddpf_fourcc) != 0 {
        if pf_fourcc == FOURCC_DX10 {
            header_size += 20;
            if dds_bytes.len() < header_size {
                return Err("DX10 header out of bounds".to_string());
            }
            let dxgi = read_u32_le(dds_bytes, 128).ok_or("Failed to read DX10 DXGI format")?;
            // DXGI Formats: BC1=71, BC2=74, BC3=77, BC4=80, BC5=83
            if dxgi == 71 || dxgi == 72 || dxgi == 80 || dxgi == 81 {
                is_compressed = true;
                block_bytes = 8;
            } else if dxgi == 74
                || dxgi == 75
                || dxgi == 77
                || dxgi == 78
                || dxgi == 83
                || dxgi == 84
            {
                is_compressed = true;
                block_bytes = 16;
            }
        } else {
            if pf_fourcc == FOURCC_DXT1
                || pf_fourcc == FOURCC_ATI1
                || pf_fourcc == FOURCC_BC4U
                || pf_fourcc == FOURCC_BC4S
            {
                is_compressed = true;
                block_bytes = 8;
            } else if pf_fourcc == FOURCC_DXT2
                || pf_fourcc == FOURCC_DXT3
                || pf_fourcc == FOURCC_DXT4
                || pf_fourcc == FOURCC_DXT5
                || pf_fourcc == FOURCC_ATI2
                || pf_fourcc == FOURCC_BC5U
                || pf_fourcc == FOURCC_BC5S
            {
                is_compressed = true;
                block_bytes = 16;
            }
        }
    }

    let mut data_ptr_offset = header_size;
    let mut is_swizzled = false;

    if data_ptr_offset + 4 <= dds_bytes.len() {
        let marker = read_u32_le(dds_bytes, data_ptr_offset).ok_or("Failed to read marker")?;
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
        return Err("DDS offset out of bounds".to_string());
    }

    let (pitch, slice_pitch) = if is_compressed {
        let num_blocks_wide = width.div_ceil(4).max(1) as usize;
        let num_blocks_high = height.div_ceil(4).max(1) as usize;
        let p = num_blocks_wide * block_bytes;
        let sp = p * num_blocks_high;
        (p, sp)
    } else {
        let bpp = pf_rgb_bitcount as usize;
        let p = (width as usize * bpp).div_ceil(8);
        let sp = p * height as usize;
        (p, sp)
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

pub fn decode_dds_bytes(dds_bytes: &[u8]) -> Result<DynamicImage, String> {
    let info = analyze_header(dds_bytes)?;

    use std::borrow::Cow;
    let clean_dds_bytes = if info.is_swizzled || info.is_xbox {
        let mut pixel_data = if info.is_swizzled {
            let mut unswizzled = vec![0u8; info.slice_pitch];
            unswizzle_block_linear(
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
            perform_xbox_endian_swap(&mut pixel_data, info.fourcc);
        }

        // Construct a clean, standard DDS file buffer in memory
        let mut clean = Vec::with_capacity(128 + pixel_data.len());
        clean.extend_from_slice(b"DDS ");

        let mut dds_header = [0u8; 124];
        dds_header[0..4].copy_from_slice(&124u32.to_le_bytes());

        // dwFlags: CAPS, HEIGHT, WIDTH, PIXELFORMAT + LINEARSIZE or PITCH
        let mut dw_flags: u32 = 0x1 | 0x2 | 0x4 | 0x1000;
        if info.is_compressed {
            dw_flags |= 0x80000;
        } else {
            dw_flags |= 0x8;
        }
        dds_header[4..8].copy_from_slice(&dw_flags.to_le_bytes());

        dds_header[8..12].copy_from_slice(&info.height.to_le_bytes());
        dds_header[12..16].copy_from_slice(&info.width.to_le_bytes());

        let dw_pitch = if info.is_compressed {
            info.slice_pitch as u32
        } else {
            info.pitch as u32
        };
        dds_header[16..20].copy_from_slice(&dw_pitch.to_le_bytes());
        dds_header[20..24].copy_from_slice(&1u32.to_le_bytes()); // dwDepth
        dds_header[24..28].copy_from_slice(&1u32.to_le_bytes()); // dwMipMapCount

        // ddspf
        dds_header[72..76].copy_from_slice(&32u32.to_le_bytes()); // pf_size
        let pf_flags: u32 = if info.is_compressed {
            0x00000004
        } else {
            0x00000040 | 0x00000001
        }; // FOURCC or RGB|ALPHAPIXELS
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

        dds_header[104..108].copy_from_slice(&0x1000u32.to_le_bytes()); // dwCaps (DDSCAPS_TEXTURE)

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
        .map_err(|e| format!("Ddsfile parse error: {}", e))?;

    let rgba_img =
        image_dds::image_from_dds(&dds, 0).map_err(|e| format!("image_dds decode error: {}", e))?;

    Ok(DynamicImage::ImageRgba8(rgba_img))
}

pub fn open_image_with_dds_fallback<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<DynamicImage, String> {
    let path_ref = path.as_ref();

    // Extract the file extension and convert it to lower case
    let ext = path_ref
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if ext == "dds" {
        // Parse DDS (handles standard, Xbox 360, and CryEngine specific unswizzling)
        let file = std::fs::File::open(path_ref).map_err(|e| e.to_string())?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| e.to_string())? };
        decode_dds_bytes(&mmap)
    } else if ext == "exr" {
        // Custom pipeline for EXR: Read raw float pixels and apply ACES Filmic Tonemapping
        let (hdr_pixels, width, height) = crate::tonemapper::load_exr_rgba(path_ref)
            .map_err(|e| format!("EXR Load error: {}", e))?;

        let ldr_img = crate::tonemapper::tonemap_hdr_to_ldr_rgba(
            &hdr_pixels,
            width,
            height,
            crate::tonemapper::TonemapOperator::AcesFilmic,
            1.0, // Default exposure value
        )
        .map_err(|e| format!("Tonemap error: {}", e))?;

        Ok(DynamicImage::ImageRgba8(ldr_img))
    } else {
        // Standard fallback for PNG, JPEG, TGA, BMP, TIFF, WEBP etc.
        image::open(path_ref).map_err(|e| e.to_string())
    }
}
