// src/core/downloader.rs

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use slint::ComponentHandle;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex as AsyncMutex;

// Global lock to prevent conflicts between background download and manual scan threads
static DOWNLOAD_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();

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

    // Configure client with safety timeouts to avoid infinite hangs on blocked networks
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300)) // 5 minutes overall timeout
        .build()
        .context("Failed to build secure HTTP client")?;

    tracing::info!("Connecting to Hugging Face CDN (sending GET request)...");

    let response = client.get(url).send().await.context(
        "HTTP request failed. Hugging Face may be blocked by your ISP or a VPN is required.",
    )?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Server returned HTTP error status: {}",
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
        .context("Failed to create destination model file on disk")?;

    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;

    while let Some(chunk_result) = stream.next().await {
        // Safe cooperative cancellation check inside the stream loop
        if cancel_token.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Download cancelled by user"));
        }

        let chunk = chunk_result?;
        file.write_all(&chunk)
            .await
            .context("Failed to write model chunk to disk asynchronously")?;

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
        .context("Failed to flush compiled file buffers to disk")?;

    Ok(())
}

pub async fn verify_and_download_models(
    app_weak: slint::Weak<crate::app::AppWindow>,
    model_idx: i32,
    cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    if model_idx == 7 {
        return Ok(());
    }

    tracing::info!("Verifying AI models for model index: {}", model_idx);

    // Acquire lock to avoid concurrent downloads if triggered multiple times
    let lock = DOWNLOAD_LOCK.get_or_init(|| AsyncMutex::new(()));

    tracing::info!("Checking download queue state...");
    let _guard = lock.lock().await;
    tracing::info!("Download lock successfully acquired.");

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;

    let (folder_name, files) = match model_idx {
        1 => (
            "clip_vit_l14",
            vec![
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
        ),
        2 => (
            "siglip_base",
            vec![
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
        ),
        3 => (
            "siglip_large",
            vec![
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
        ),
        4 => (
            "dinov2_base",
            vec![(
                "visual.onnx",
                "https://huggingface.co/Xenova/dinov2-base/resolve/main/onnx/model.onnx",
            )],
        ),
        5 => (
            "siglip2_base",
            vec![
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
        ),
        6 => (
            "llm2clip_base",
            vec![
                // microsoft/LLM2CLIP-Openai-B-16 lacks tokenizer files. We redirect to standard CLIP-B/32 since vocabularies are identical.
                (
                    "tokenizer.json",
                    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
                ),
                // The ONNX model is unmerged on the main branch, so we download it from PR #3 (refs/pr/3) branch
                (
                    "text.onnx",
                    "https://huggingface.co/microsoft/LLM2CLIP-Openai-B-16/resolve/refs/pr/3/onnx/model.onnx",
                ),
            ],
        ),
        _ => (
            "clip_vit_b32",
            vec![
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
        ),
    };

    let model_dir = app_dir.join("models").join(folder_name);
    fs::create_dir_all(&model_dir)
        .await
        .context("Failed to build directory path for target models")?;

    for (name, url) in files {
        let dest = model_dir.join(name);

        // Checks if the file exists and is not corrupted / empty (< 1KB)
        let is_valid = match tokio::fs::metadata(&dest).await {
            Ok(meta) => meta.len() > 1024,
            Err(_) => false,
        };

        if is_valid {
            tracing::info!(
                "Model file '{}' verified successfully (already cached on disk).",
                name
            );
        } else {
            tracing::info!(
                "Model file '{}' is missing or corrupted. Preparing download...",
                name
            );
            // Download to temporary .tmp files first to prevent corruption on interruption
            let tmp_dest = model_dir.join(format!("{}.tmp", name));

            let app_copy = app_weak.clone();
            let file_name = name.to_string();

            // Force scanning banner to show so the user sees progress
            let _ = app_copy.upgrade_in_event_loop(|ui| {
                let store = ui.global::<crate::app::Store>();
                store.set_is_scanning(true);
            });

            let download_res = download_file_with_progress(
                move |percentage| {
                    let f_name = file_name.clone();
                    let _ = app_copy.upgrade_in_event_loop(move |ui| {
                        let store = ui.global::<crate::app::Store>();
                        store.set_progress(percentage / 100.0);
                        store.set_status_text(
                            format!("Downloading {}: {:.1}%", f_name, percentage).into(),
                        );
                    });
                },
                url,
                &tmp_dest,
                cancel_token.clone(),
            )
            .await;

            // If download was cancelled or failed, remove the temporary file to keep disk clean
            if let Err(e) = download_res {
                tracing::error!(
                    "Download of '{}' failed or cancelled: {}. Cleaning up...",
                    name,
                    e
                );
                let _ = tokio::fs::remove_file(&tmp_dest).await;
                return Err(e);
            }

            // Atomically finalize downloaded file on success
            fs::rename(&tmp_dest, &dest)
                .await
                .context("Failed to finalize downloaded file")?;
            tracing::info!(
                "Model file '{}' successfully downloaded and verified.",
                name
            );
        }
    }

    // Special optimization for single-file ONNX models like LLM2CLIP to save 1.2 GB of download
    if model_idx == 6 {
        let visual_dest = model_dir.join("visual.onnx");
        let text_dest = model_dir.join("text.onnx");
        if text_dest.exists() && !visual_dest.exists() {
            tracing::info!(
                "Duplicating single-file LLM2CLIP model on disk to save 1.2 GB of download..."
            );
            tokio::fs::copy(&text_dest, &visual_dest)
                .await
                .context("Failed to duplicate single-file ONNX model")?;
            tracing::info!("LLM2CLIP visual model successfully duplicated.");
        }
    }

    Ok(())
}
