// src/compression/gpu_block.rs

use anyhow::{Result, anyhow};
use image::DynamicImage;

#[derive(Debug, Clone, Copy)]
pub enum TextureBlockFormat {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6Unsigned,
    Bc6Signed,
    Bc7,
    Etc1,
    Etc2Rgb,
    Etc2Rgba1,
    Etc2Rgba8,
    EacR,
    EacRSigned,
    EacRg,
    EacRgSigned,
    Astc { block_x: usize, block_y: usize },
    Pvrtc { is_2bpp: bool },
    AtcRgb4,
    AtcRgba8,
}

/// Decodes GPU block-compressed raw pixel data into an RGBA8 DynamicImage.
pub fn decode_gpu_block(
    payload: &[u8],
    width: usize,
    height: usize,
    format: TextureBlockFormat,
) -> Result<DynamicImage> {
    if width == 0 || height == 0 {
        return Err(anyhow!("Invalid zero texture dimensions"));
    }

    let mut rgba_u32 = vec![0u32; width * height];

    match format {
        TextureBlockFormat::Bc1 => {
            texture2ddecoder::decode_bc1(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC1 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc2 => {
            texture2ddecoder::decode_bc2(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC2 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc3 => {
            texture2ddecoder::decode_bc3(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC3 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc4 => {
            texture2ddecoder::decode_bc4(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC4 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc5 => {
            texture2ddecoder::decode_bc5(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC5 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc6Unsigned => {
            texture2ddecoder::decode_bc6_unsigned(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC6H unsigned decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc6Signed => {
            texture2ddecoder::decode_bc6_signed(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC6H signed decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Bc7 => {
            texture2ddecoder::decode_bc7(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("BC7 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Etc1 => {
            texture2ddecoder::decode_etc1(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC1 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Etc2Rgb => {
            texture2ddecoder::decode_etc2_rgb(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGB decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Etc2Rgba1 => {
            texture2ddecoder::decode_etc2_rgba1(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGBA1 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Etc2Rgba8 => {
            texture2ddecoder::decode_etc2_rgba8(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ETC2 RGBA8 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::EacR => {
            texture2ddecoder::decode_eacr(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R decode failed: {:?}", e))?;
        }
        TextureBlockFormat::EacRSigned => {
            texture2ddecoder::decode_eacr_signed(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC R signed decode failed: {:?}", e))?;
        }
        TextureBlockFormat::EacRg => {
            texture2ddecoder::decode_eacrg(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG decode failed: {:?}", e))?;
        }
        TextureBlockFormat::EacRgSigned => {
            texture2ddecoder::decode_eacrg_signed(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("EAC RG signed decode failed: {:?}", e))?;
        }
        TextureBlockFormat::Astc { block_x, block_y } => {
            texture2ddecoder::decode_astc(payload, width, height, block_x, block_y, &mut rgba_u32)
                .map_err(|e| anyhow!("ASTC {}x{} decode failed: {:?}", block_x, block_y, e))?;
        }
        TextureBlockFormat::Pvrtc { is_2bpp } => {
            texture2ddecoder::decode_pvrtc(payload, width, height, &mut rgba_u32, is_2bpp)
                .map_err(|e| anyhow!("PVRTC decode failed: {:?}", e))?;
        }
        TextureBlockFormat::AtcRgb4 => {
            texture2ddecoder::decode_atc_rgb4(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ATC RGB4 decode failed: {:?}", e))?;
        }
        TextureBlockFormat::AtcRgba8 => {
            texture2ddecoder::decode_atc_rgba8(payload, width, height, &mut rgba_u32)
                .map_err(|e| anyhow!("ATC RGBA8 decode failed: {:?}", e))?;
        }
    }

    let raw_bytes = crate::utils::image_processing::bgra_u32_to_rgba_bytes(rgba_u32);
    let img = image::RgbaImage::from_raw(width as u32, height as u32, raw_bytes)
        .ok_or_else(|| anyhow!("Failed to compile RGBA image buffer"))?;

    Ok(DynamicImage::ImageRgba8(img))
}
