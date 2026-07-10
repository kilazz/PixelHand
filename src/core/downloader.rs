// src/core/downloader.rs

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use std::path::Path;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;

/// Progress-aware asynchronous HTTP download utility.
/// Streams bytes directly to disk asynchronously and updates UI percentage indicators safely.
pub async fn download_file_with_progress<F>(
    progress_callback: F,
    url: &str,
    dest_path: &Path,
) -> Result<()>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to send HTTP download request")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Server returned HTTP error status: {}",
            response.status()
        ));
    }

    let total_bytes = response.content_length().unwrap_or(0);

    // Use tokio's asynchronous file handler to prevent blocking the async reactor
    let mut file = File::create(dest_path)
        .await
        .context("Failed to create destination model file on disk")?;

    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;

    while let Some(chunk_result) = stream.next().await {
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

/// Orchestrates the verification and download of the selected AI model weights.
/// Automatically handles vision-only (DINOv2) vs multimodal (CLIP/SigLIP) layout boundaries.
pub async fn verify_and_download_models(
    app_weak: slint::Weak<crate::app::AppWindow>,
    model_idx: i32,
) -> Result<()> {
    // If the index corresponds to a Custom Local Model, skip downloading entirely
    if model_idx == 5 {
        return Ok(());
    }

    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;

    // Map selected model indexes to HuggingFace Xenova ONNX repositories
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
        if !dest.exists() {
            let app_copy = app_weak.clone();
            let file_name = name.to_string();

            download_file_with_progress(
                move |percentage| {
                    let f_name = file_name.clone();
                    let _ = app_copy.upgrade_in_event_loop(move |ui| {
                        ui.set_progress(percentage / 100.0);
                        ui.set_status_text(
                            format!("Downloading {}: {:.1}%", f_name, percentage).into(),
                        );
                    });
                },
                url,
                &dest,
            )
            .await?;
        }
    }
    Ok(())
}
