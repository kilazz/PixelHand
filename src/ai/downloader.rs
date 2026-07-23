// src/ai/downloader.rs

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use reqwest::Client;
use slint::ComponentHandle;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex as AsyncMutex;

use crate::state::models::AiModelType;

// Global lock to prevent conflicts between concurrent download requests and manual scan threads
static DOWNLOAD_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();

/// Evaluates expected minimum byte threshold based on file type to prevent caching 404/Error HTML pages.
fn get_expected_min_bytes(file_name: &str) -> u64 {
    if file_name.ends_with(".onnx") {
        10 * 1024 * 1024 // ONNX weights must be at least 10 MB
    } else if file_name.ends_with(".json") {
        10 * 1024 // Tokenizer configs must be at least 10 KB
    } else {
        1024
    }
}

/// Validates file existence and verifies that size matches expected minimum byte thresholds.
pub async fn verify_file_integrity(path: &Path) -> bool {
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return false;
    };

    let file_name = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    let min_expected = get_expected_min_bytes(&file_name);

    if meta.len() < min_expected {
        tracing::warn!(
            "Model file '{}' failed integrity check (Found {} bytes, expected >= {} bytes). Preparing re-download...",
            path.display(),
            meta.len(),
            min_expected
        );
        return false;
    }

    true
}

/// Downloads a file from a specified URL to the local destination path with progress reporting and cooperative cancellation.
pub async fn download_file_with_progress<F>(
    progress_callback: F,
    url: &str,
    dest_path: &Path,
    cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    tracing::info!("Building secure HTTP client with safety timeouts...");

    // Configure reqwest client with timeouts to prevent infinite hangs on slow networks
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300)) // 5 minutes overall timeout limit
        .build()
        .context("Failed to build secure HTTP client")?;

    tracing::info!("Connecting to CDN: sending GET request to {}", url);

    let response = client.get(url).send().await.context(
        "HTTP request failed. Connection could be blocked by network settings or requires a VPN.",
    )?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Server returned an HTTP error status code: {}",
            response.status()
        ));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    tracing::info!(
        "Connection established. Total file size to download: {} bytes",
        total_bytes
    );

    let mut file = File::create(dest_path)
        .await
        .context("Failed to create destination file on disk")?;

    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;

    while let Some(chunk_result) = stream.next().await {
        // Cooperative cancellation check during stream processing loop
        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow!("Download cancelled by user"));
        }

        let chunk = chunk_result?;
        file.write_all(&chunk)
            .await
            .context("Failed to write downloaded chunks to disk")?;

        bytes_downloaded += chunk.len() as u64;
        let percentage = if total_bytes > 0 {
            (bytes_downloaded as f32 / total_bytes as f32) * 100.0
        } else {
            0.0
        };
        progress_callback(percentage);
    }

    file.flush()
        .await
        .context("Failed to flush file buffers to disk")?;

    // Validate that the full payload was received matching the server Content-Length
    if total_bytes > 0 && bytes_downloaded != total_bytes {
        return Err(anyhow!(
            "Incomplete download for '{}': received {} of {} expected bytes",
            dest_path.display(),
            bytes_downloaded,
            total_bytes
        ));
    }

    Ok(())
}

/// Verifies if model weight files are cached locally on disk.
/// If files are missing or corrupted, downloads them directly from Hugging Face CDN.
pub async fn verify_and_download_models(
    app_weak: slint::Weak<crate::app::AppWindow>,
    model: AiModelType,
    cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    if model == AiModelType::Custom {
        return Ok(());
    }

    tracing::info!("Verifying AI model weights for architecture: {:?}", model);

    // Acquire lock to prevent overlapping downloads if triggered multiple times
    let lock = DOWNLOAD_LOCK.get_or_init(|| AsyncMutex::new(()));

    tracing::info!("Acquiring download queue lock...");
    let _guard = lock.lock().await;
    tracing::info!("Download lock successfully acquired.");

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let folder_name = model.folder_name();

    let files = match model {
        AiModelType::ClipVitL14 => vec![
            (
                "tokenizer.json",
                "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/text_model.onnx",
            ),
            (
                "visual.onnx",
                "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/vision_model.onnx",
            ),
        ],
        AiModelType::SiglipBase => vec![
            (
                "tokenizer.json",
                "https://huggingface.co/Xenova/siglip-base-patch16-384/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/Xenova/siglip-base-patch16-384/resolve/main/onnx/text_model.onnx",
            ),
            (
                "visual.onnx",
                "https://huggingface.co/Xenova/siglip-base-patch16-384/resolve/main/onnx/vision_model.onnx",
            ),
        ],
        AiModelType::SiglipLarge => vec![
            (
                "tokenizer.json",
                "https://huggingface.co/Xenova/siglip-large-patch16-384/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/Xenova/siglip-large-patch16-384/resolve/main/onnx/text_model.onnx",
            ),
            (
                "visual.onnx",
                "https://huggingface.co/Xenova/siglip-large-patch16-384/resolve/main/onnx/vision_model.onnx",
            ),
        ],
        AiModelType::DinoV2Base => vec![(
            "visual.onnx",
            "https://huggingface.co/Xenova/dinov2-base/resolve/main/onnx/model.onnx",
        )],
        AiModelType::Siglip2Base => vec![
            (
                "tokenizer.json",
                "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/text_model.onnx",
            ),
            (
                "visual.onnx",
                "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/vision_model.onnx",
            ),
        ],
        AiModelType::Llm2ClipBase => vec![
            // microsoft/LLM2CLIP-Openai-B-16 lacks tokenizer files. Standard CLIP-B/32 is compatible.
            (
                "tokenizer.json",
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/microsoft/LLM2CLIP-Openai-B-16/resolve/7a66a9239794caa50824f8c737366abc34d328aa/onnx/model.onnx",
            ),
        ],
        AiModelType::ClipVitB32 => vec![
            (
                "tokenizer.json",
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
            ),
            (
                "text.onnx",
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
            ),
            (
                "visual.onnx",
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
            ),
        ],
        AiModelType::Custom => unreachable!(),
    };

    let model_dir = app_dir.join("models").join(folder_name);
    fs::create_dir_all(&model_dir)
        .await
        .context("Failed to create cache directory for models")?;

    for (name, url) in files {
        let dest = model_dir.join(name);

        // Verification step: checks file existence and ensures it meets minimum byte thresholds
        let is_valid = verify_file_integrity(&dest).await;

        if is_valid {
            tracing::info!(
                "Model file '{}' verified successfully (already cached on disk).",
                name
            );
        } else {
            tracing::info!(
                "Model file '{}' is missing or invalid. Preparing background download...",
                name
            );
            // Download to temporary .tmp files first to prevent corrupt file mappings on interrupted downloads
            let tmp_dest = model_dir.join(format!("{}.tmp", name));

            let app_copy = app_weak.clone();
            let file_name = name.to_string();

            // Set scanning state on the UI Diagnostics singleton to show progress bar
            let _ = app_copy.upgrade_in_event_loop(|ui| {
                let diag = ui.global::<crate::app::Diagnostics>();
                diag.set_is_scanning(true);
            });

            let download_res = download_file_with_progress(
                move |percentage| {
                    let f_name = file_name.clone();
                    let _ = app_copy.upgrade_in_event_loop(move |ui| {
                        let diag = ui.global::<crate::app::Diagnostics>();
                        diag.set_progress(percentage / 100.0);
                        diag.set_status_text(
                            format!("Downloading {}: {:.1}%", f_name, percentage).into(),
                        );
                    });
                },
                url,
                &tmp_dest,
                cancel_token.clone(),
            )
            .await;

            // In case of error or user cancellation, remove the temporary file to keep disk workspace clean
            if let Err(e) = download_res {
                tracing::error!(
                    "Download of '{}' failed or was cancelled: {}. Cleaning temporary files...",
                    name,
                    e
                );
                let _ = tokio::fs::remove_file(&tmp_dest).await;
                return Err(e);
            }

            // Verify integrity of downloaded temporary file before moving into place
            if !verify_file_integrity(&tmp_dest).await {
                let _ = tokio::fs::remove_file(&tmp_dest).await;
                return Err(anyhow!(
                    "Downloaded file '{}' failed integrity check (incomplete or corrupted download)",
                    name
                ));
            }

            // Atomically finalize downloaded file on success
            fs::rename(&tmp_dest, &dest)
                .await
                .context("Failed to finalize downloaded file")?;
            tracing::info!(
                "Model file '{}' successfully downloaded and validated.",
                name
            );
        }
    }

    // Optimization for single-file models like LLM2CLIP to save 1.2 GB of duplicate download
    if model == AiModelType::Llm2ClipBase {
        let visual_dest = model_dir.join("visual.onnx");
        let text_dest = model_dir.join("text.onnx");

        let is_visual_valid = verify_file_integrity(&visual_dest).await;
        if !is_visual_valid && verify_file_integrity(&text_dest).await {
            tracing::info!(
                "Duplicating single-file LLM2CLIP model on disk to prepare visual model..."
            );
            tokio::fs::copy(&text_dest, &visual_dest)
                .await
                .context("Failed to duplicate single-file ONNX model")?;
            tracing::info!("LLM2CLIP visual model successfully duplicated on disk.");
        }
    }

    Ok(())
}
