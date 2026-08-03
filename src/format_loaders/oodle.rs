// src/format_loaders/oodle.rs

use libloading::Library;
use std::sync::OnceLock;

type FnOodleLZDecompress = unsafe extern "system" fn(
    comp_buf: *const u8,
    comp_buf_size: usize,
    raw_buf: *mut u8,
    raw_len: usize,
    fuzz_safe: u32, // 1 = Yes
    check_crc: u32, // 0 = No
    verbosity: u32, // 0 = None
    dec_buf_base: *mut u8,
    dec_buf_size: usize,
    fp_callback: *mut std::ffi::c_void,
    callback_user_data: *mut std::ffi::c_void,
    decoder_memory: *mut std::ffi::c_void,
    decoder_memory_size: usize,
    thread_phase: u32, // 3 = Unthreaded
) -> usize;

struct OodleLibrary {
    _lib: Library,
    decompress: FnOodleLZDecompress,
}

static OODLE_LIB: OnceLock<Option<OodleLibrary>> = OnceLock::new();

/// Searches for `oo2core_win64.dll` in the executable directory, working directory, and system paths.
fn load_oodle_library() -> Option<OodleLibrary> {
    let candidate_names = [
        "oo2core_win64.dll",
        "oo2core_9_win64.dll",
        "oo2core_8_win64.dll",
        "oo2core_7_win64.dll",
        "liboo2corelinux64.so",
        "liboo2coremac64.dylib",
    ];

    // Check executable directory first
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        for &name in &candidate_names {
            let p = exe_dir.join(name);
            if p.is_file()
                && let Ok(lib) = unsafe { Library::new(&p) }
                && let Ok(decompress) =
                    unsafe { lib.get::<FnOodleLZDecompress>(b"OodleLZ_Decompress\0") }
            {
                tracing::info!("[Oodle] Successfully loaded library from: {}", p.display());
                return Some(OodleLibrary {
                    decompress: *decompress,
                    _lib: lib,
                });
            }
        }
    }

    // Check working directory / system PATH
    for &name in &candidate_names {
        if let Ok(lib) = unsafe { Library::new(name) }
            && let Ok(decompress) =
                unsafe { lib.get::<FnOodleLZDecompress>(b"OodleLZ_Decompress\0") }
        {
            tracing::info!("[Oodle] Successfully loaded library: {}", name);
            return Some(OodleLibrary {
                decompress: *decompress,
                _lib: lib,
            });
        }
    }

    tracing::warn!(
        "[Oodle] oo2core_win64.dll not found. Oodle-compressed UE5/6 4K textures will use embedded preview fallbacks."
    );
    None
}

fn get_oodle() -> Option<&'static OodleLibrary> {
    OODLE_LIB.get_or_init(load_oodle_library).as_ref()
}

/// Decompresses an Oodle LZ payload (Kraken/Leviathan/Mermaid/Selkie) into uncompressed texture bytes.
pub fn decompress_oodle_lz(compressed: &[u8], uncompressed_size: usize) -> Option<Vec<u8>> {
    let oodle = get_oodle()?;
    if compressed.is_empty() || uncompressed_size == 0 {
        return None;
    }

    let mut decompressed = vec![0u8; uncompressed_size];

    let result = unsafe {
        (oodle.decompress)(
            compressed.as_ptr(),
            compressed.len(),
            decompressed.as_mut_ptr(),
            uncompressed_size,
            1, // fuzz_safe: Yes
            0, // check_crc: No
            0, // verbosity: None
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            3, // thread_phase: Unthreaded
        )
    };

    if result == uncompressed_size {
        Some(decompressed)
    } else {
        tracing::warn!(
            "[Oodle] Decompression output size mismatch: got {} bytes, expected {} bytes",
            result,
            uncompressed_size
        );
        None
    }
}
