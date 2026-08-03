// src/compression/oodle.rs

use anyhow::{Result, anyhow};
use oozextract::Extractor;

/// Decompresses an Oodle LZ payload (Kraken, Leviathan, Mermaid, Selkie) in pure Rust using oozextract.
pub fn decompress_oodle_lz(compressed: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    if compressed.is_empty() || uncompressed_size == 0 {
        return Err(anyhow!("Empty compression input or zero expected size"));
    }

    let mut decompressed = vec![0u8; uncompressed_size];

    Extractor::new()
        .read_from_slice(compressed, &mut decompressed)
        .map_err(|e| anyhow!("Oodle LZ decompression failed: {:?}", e))?;

    Ok(decompressed)
}
