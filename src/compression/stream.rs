// src/compression/stream.rs

use anyhow::{Context, Result};
use std::io::Read;

pub fn decompress_zstd(compressed: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(compressed).context("Failed to decompress Zstd stream")
}

pub fn decompress_zlib(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::ZlibDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .context("Failed to decompress Zlib stream")?;
    Ok(decompressed)
}

pub fn decompress_gzip(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::GzDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .context("Failed to decompress Gzip stream")?;
    Ok(decompressed)
}
