// src/utils/fs.rs
use crate::state::AppState;
use std::fs;
use std::path::PathBuf;

/// Extracts a tuple of (paths_to_delete, hardlink_source_target_pairs) based on UI checkbox state
pub fn extract_selected_files(lock: &AppState) -> (Vec<String>, Vec<(String, String)>) {
    let mut checked_files = Vec::new();
    let mut pairs = Vec::new();

    for row in &lock.results {
        if !row.is_header && row.is_checked {
            checked_files.push(row.path.clone());

            // Find the original path inside the matching duplicate group
            if let Some(group) = lock.groups.get(row.group_index as usize)
                && let Some(orig) = group.files.first()
            {
                pairs.push((orig.path.clone(), row.path.clone()));
            }
        }
    }
    (checked_files, pairs)
}

/// Routes the requested filesystem operation to the proper handler
pub async fn execute_file_action(
    action: &str,
    files: Vec<String>,
    pairs: Vec<(String, String)>,
) -> anyhow::Result<()> {
    match action {
        "trash" => delete_files(files).await,
        "hardlink" => create_hardlinks(pairs).await,
        "reflink" => create_reflinks(pairs).await,
        _ => Err(anyhow::anyhow!("Unknown action type")),
    }
}

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

async fn create_hardlinks(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);

        if !source.exists() {
            continue;
        }
        if target.exists() {
            fs::remove_file(&target)?;
        }
        fs::hard_link(&source, &target)?;
    }
    Ok(())
}

async fn create_reflinks(pairs: Vec<(String, String)>) -> anyhow::Result<()> {
    for (source_str, target_str) in pairs {
        let source = PathBuf::from(source_str);
        let target = PathBuf::from(target_str);

        if !source.exists() {
            continue;
        }
        if target.exists() {
            fs::remove_file(&target)?;
        }
        reflink_copy::reflink_or_copy(&source, &target)?;
    }
    Ok(())
}
