// src/downloader.rs
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tauri::{AppHandle, Emitter, Runtime}; // The Emitter trait is used in Tauri v2 for firing frontend events

/// Event payload containing progress metrics emitted to the frontend
#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub file_name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub percentage: f32,
}

/// Downloads a file from a remote URL while streaming real-time progress events to the Tauri GUI
pub async fn download_file_with_progress<R: Runtime>(
    app: &AppHandle<R>,
    url: &str,
    dest_path: &Path,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the HTTP Client and send the request
    let client = Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Server returned HTTP error status: {}", response.status()).into());
    }

    // 2. Read the total file size from the Content-Length header
    let total_bytes = response.content_length().unwrap_or(0);
    let mut file = File::create(dest_path)?;
    let mut stream = response.bytes_stream();
    let mut bytes_downloaded = 0u64;

    // 3. Process the incoming HTTP byte stream chunk by chunk (reduces RAM usage)
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk)?;

        bytes_downloaded += chunk.len() as u64;

        // Calculate current progress percentage
        let percentage = if total_bytes > 0 {
            (bytes_downloaded as f32 / total_bytes as f32) * 100.0
        } else {
            0.0
        };

        // 4. Emit progress update event payload to the Tauri frontend
        app.emit(
            "download-progress",
            DownloadProgress {
                file_name: file_name.to_string(),
                bytes_downloaded,
                total_bytes,
                percentage,
            },
        )?;
    }

    Ok(())
}
