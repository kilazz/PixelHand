// src/core/downloader.rs
use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

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
        .context("Failed to send HTTP request")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Server returned HTTP error status: {}",
            response.status()
        ));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    let mut file = File::create(dest_path)?;
    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk)?;

        bytes_downloaded += chunk.len() as u64;
        let percentage = if total_bytes > 0 {
            (bytes_downloaded as f32 / total_bytes as f32) * 100.0
        } else {
            0.0
        };
        progress_callback(percentage);
    }
    Ok(())
}

/// Orchestrates the startup model verification and downloads.
/// (We accept the `slint::Weak<AppWindow>` here just to cleanly bridge UI updates).
pub async fn verify_and_download_models(
    app_weak: slint::Weak<crate::app::AppWindow>,
) -> Result<()> {
    let app_dir = crate::utils::settings::get_portable_app_data_dir()?;
    let model_dir = app_dir
        .join("models")
        .join("CLIP-ViT-B-32-laion2B-s34B-b79K_fp16");
    fs::create_dir_all(&model_dir)?;

    let files = [
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
    ];

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
