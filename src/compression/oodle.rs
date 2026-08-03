// src/compression/oodle.rs

use anyhow::{Result, anyhow};
use oozextract::Extractor;

/// Decompresses an Oodle LZ payload (Kraken, Leviathan, Mermaid, Selkie) in pure Rust using oozextract.
pub fn decompress_oodle_lz(compressed: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    // Safety check against capacity overflow / OOM and non-Oodle header bytes
    if compressed.is_empty()
        || uncompressed_size == 0
        || uncompressed_size > 256 * 1024 * 1024
        || compressed.starts_with(&[0xC1, 0x83, 0x2A, 0x9E])
    {
        return Err(anyhow!("Invalid, non-Oodle or oversized input payload"));
    }

    // Isolate panics from oozextract on malformed buffers
    let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut decompressed = vec![0u8; uncompressed_size];
        let mut extractor = Extractor::new();

        extractor
            .read_from_slice(compressed, &mut decompressed)
            .map_err(|e| anyhow!("Oodle LZ decompression failed: {:?}", e))?;

        Ok::<_, anyhow::Error>(decompressed)
    }));

    match panic_res {
        Ok(res) => res,
        Err(_) => Err(anyhow!("Oodle decompression panicked on malformed input")),
    }
}
