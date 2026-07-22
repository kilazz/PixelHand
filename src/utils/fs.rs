// src/utils/fs.rs

use crate::state::AppState;
use std::fs;
use std::path::PathBuf;
use ustr::Ustr;

/// Standardizes a path string to a lowercased, slash-unified format to ensure cross-platform compatibility.
/// Replaces Windows backslashes with forward slashes to prevent platform discrepancies.
pub fn normalize_path(path_str: &str) -> String {
    path_str
        .chars()
        .map(|c| if c == '\\' { '/' } else { c })
        .collect::<String>()
        .to_lowercase()
}

/// Returns an interned Ustr representing the normalized path to support O(1) matching and zero-allocation lookups.
pub fn normalize_path_key(path_str: &str) -> Ustr {
    ustr::ustr(&normalize_path(path_str))
}

/// Extracts checked file paths for deletion and resolves original-duplicate pairs for link transformations.
pub fn extract_selected_files(lock: &AppState) -> (Vec<String>, Vec<(String, String)>) {
    let mut checked_files = Vec::new();
    let mut pairs = Vec::new();

    // Check duplicate groups
    for group in &lock.groups {
        for file in &group.files {
            if lock.checked_paths.contains(&file.path) {
                checked_files.push(file.path.clone());
                if let Some(orig) = group.files.first() {
                    pairs.push((orig.path.clone(), file.path.clone()));
                }
            }
        }
    }

    // Check QC issues
    for issue in &lock.qc_issues {
        if lock.checked_paths.contains(&issue.path) {
            checked_files.push(issue.path.clone());
        }
    }

    // Check Asset Inventory files
    for file in &lock.inventory_files {
        if lock.checked_paths.contains(&file.path) {
            checked_files.push(file.path.clone());
        }
    }

    (checked_files, pairs)
}

/// Routes filesystem deduplication or deletion operations to the appropriate handler.
/// Offloads synchronous blocking IO operations to a dedicated blocking thread pool.
pub async fn execute_file_action(
    action: &str,
    files: Vec<String>,
    pairs: Vec<(String, String)>,
) -> anyhow::Result<()> {
    let action_owned = action.to_string();

    // Run blocking filesystem operations inside the dedicated Tokio blocking pool
    tokio::task::spawn_blocking(move || match action_owned.as_str() {
        "trash" => delete_files_sync(files),
        "hardlink" => create_hardlinks_sync(pairs),
        "reflink" => create_reflinks_sync(pairs),
        _ => Err(anyhow::anyhow!("Unknown filesystem action type requested")),
    })
    .await?
}

/// Safely moves targeted duplicate files to the operating system recycle bin.
fn delete_files_sync(paths: Vec<String>) -> anyhow::Result<()> {
    let files_to_delete: Vec<PathBuf> = paths
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
    if !files_to_delete.is_empty() {
        trash::delete_all(&files_to_delete)?;
    }
    Ok(())
}

/// Transforms duplicates into hard links pointing to the master source file safely.
/// Uses atomic creation to guarantee target file protection if hardlinking fails (e.g., cross-drive EXDEV).
fn create_hardlinks_sync(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    let mut error_count = 0;

    for (source_str, target_str) in pairs {
        let source = PathBuf::from(&source_str);
        let target = PathBuf::from(&target_str);

        // Guard against linking a master file to itself
        if source == target {
            continue;
        }

        if !source.exists() {
            tracing::warn!("Deduplication source does not exist: {}", source.display());
            continue;
        }

        if !target.exists() {
            tracing::warn!("Deduplication target does not exist: {}", target.display());
            continue;
        }

        // Create temporary hard link alongside target path with safely appended suffix
        let target_filename = target.file_name().unwrap_or_default().to_string_lossy();
        let temp_target = target.with_file_name(format!("{}.ph_tmp_hardlink", target_filename));

        if temp_target.exists() {
            let _ = fs::remove_file(&temp_target);
        }

        if let Err(e) = fs::hard_link(&source, &temp_target) {
            tracing::error!(
                "Failed to generate hard link from '{}' to temp target '{}': {}. Original file preserved.",
                source.display(),
                temp_target.display(),
                e
            );
            error_count += 1;
            continue;
        }

        // Atomically replace the original target file with the validated hard link
        if let Err(e) = fs::rename(&temp_target, &target) {
            tracing::error!(
                "Failed to atomically replace target '{}' with created hard link: {}",
                target.display(),
                e
            );
            let _ = fs::remove_file(&temp_target);
            error_count += 1;
        }
    }

    if error_count > 0 {
        Err(anyhow::anyhow!(
            "Failed to process {} hard link operations. Ensure source and target are on the same disk partition.",
            error_count
        ))
    } else {
        Ok(())
    }
}

/// Transforms duplicates into reflinks pointing to the master source file safely.
/// Falls back to deep copies if the underlying filesystem does not support reflinking.
/// Uses atomic creation to guarantee target file protection if operations fail.
fn create_reflinks_sync(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    let mut error_count = 0;

    for (source_str, target_str) in pairs {
        let source = PathBuf::from(&source_str);
        let target = PathBuf::from(&target_str);

        // Guard against linking a master file to itself
        if source == target {
            continue;
        }

        if !source.exists() {
            tracing::warn!("Deduplication source does not exist: {}", source.display());
            continue;
        }

        if !target.exists() {
            tracing::warn!("Deduplication target does not exist: {}", target.display());
            continue;
        }

        // Create temporary reflink alongside target path with safely appended suffix
        let target_filename = target.file_name().unwrap_or_default().to_string_lossy();
        let temp_target = target.with_file_name(format!("{}.ph_tmp_reflink", target_filename));

        if temp_target.exists() {
            let _ = fs::remove_file(&temp_target);
        }

        if let Err(e) = reflink_copy::reflink_or_copy(&source, &temp_target) {
            tracing::error!(
                "Failed to generate reflink clone from '{}' to temp target '{}': {}. Original file preserved.",
                source.display(),
                temp_target.display(),
                e
            );
            error_count += 1;
            continue;
        }

        // Atomically replace the original target file with the validated reflink
        if let Err(e) = fs::rename(&temp_target, &target) {
            tracing::error!(
                "Failed to atomically replace target '{}' with created reflink: {}",
                target.display(),
                e
            );
            let _ = fs::remove_file(&temp_target);
            error_count += 1;
        }
    }

    if error_count > 0 {
        Err(anyhow::anyhow!(
            "Failed to process {} reflink/copy operations. Check file permissions or disk space.",
            error_count
        ))
    } else {
        Ok(())
    }
}
