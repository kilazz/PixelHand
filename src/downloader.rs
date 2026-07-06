use futures_util::StreamExt;
use reqwest::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Downloads a file from a remote URL while streaming real-time progress callbacks
pub async fn download_file_with_progress<F>(
    progress_callback: F,
    url: &str,
    dest_path: &Path,
    _file_name: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    let client = Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Server returned HTTP error status: {}", response.status()).into());
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
