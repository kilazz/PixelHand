// src/utils/fs.rs

use crate::state::AppState;
use std::fs;
use std::path::PathBuf;

/// Extracts checked file paths for deletion and resolves original-duplicate pairs for link transformations.
pub fn extract_selected_files(lock: &AppState) -> (Vec<String>, Vec<(String, String)>) {
    let mut checked_files = Vec::new();
    let mut pairs = Vec::new();

    for row in &lock.results {
        if !row.is_header && row.is_checked {
            checked_files.push(row.path.clone());

            // Resolve the corresponding master file within the target duplicate group
            if let Some(group) = lock.groups.get(row.group_index as usize)
                && let Some(orig) = group.files.first()
            {
                pairs.push((orig.path.clone(), row.path.clone()));
            }
        }
    }
    (checked_files, pairs)
}

/// Routes filesystem deduplication or deletion operations to the appropriate handler.
pub async fn execute_file_action(
    action: &str,
    files: Vec<String>,
    pairs: Vec<(String, String)>,
) -> anyhow::Result<()> {
    match action {
        "trash" => delete_files(files).await,
        "hardlink" => create_hardlinks(pairs).await,
        "reflink" => create_reflinks(pairs).await,
        _ => Err(anyhow::anyhow!("Unknown filesystem action type requested")),
    }
}

/// Safely moves targeted duplicate files to the operating system recycle bin.
async fn delete_files(paths: Vec<String>) -> anyhow::Result<()> {
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

/// Transforms duplicates into hard links pointing to the master source file.
/// Processes files sequentially and logs isolated lock/permission errors to prevent batch interruption.
async fn create_hardlinks(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);

        if !source.exists() {
            tracing::warn!("Deduplication source does not exist: {}", source.display());
            continue;
        }

        if target.exists()
            && let Err(e) = fs::remove_file(&target)
        {
            tracing::error!(
                "Failed to remove target duplicate '{}' before linking: {}",
                target.display(),
                e
            );
            continue;
        }

        if let Err(e) = fs::hard_link(&source, &target) {
            tracing::error!(
                "Failed to generate hard link from '{}' to '{}': {}",
                source.display(),
                target.display(),
                e
            );
        }
    }
    Ok(())
}

/// Transforms duplicates into reflinks (Copy-on-Write clones) pointing to the master source file.
/// Falls back to deep copies if the underlying filesystem does not support reflinking.
async fn create_reflinks(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);

        if !source.exists() {
            tracing::warn!("Deduplication source does not exist: {}", source.display());
            continue;
        }

        if target.exists()
            && let Err(e) = fs::remove_file(&target)
        {
            tracing::error!(
                "Failed to remove target duplicate '{}' before linking: {}",
                target.display(),
                e
            );
            continue;
        }

        if let Err(e) = reflink_copy::reflink_or_copy(&source, &target) {
            tracing::error!(
                "Failed to generate reflink clone from '{}' to '{}': {}",
                source.display(),
                target.display(),
                e
            );
        }
    }
    Ok(())
}
