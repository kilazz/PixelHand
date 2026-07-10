// src/core/inference.rs

use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::{Array, Array4};
use ort::inputs;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_size: (u32, u32),
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PreprocessingConfig {
    /// Returns the appropriate preprocessing configuration based on the selected AI model index.
    /// Supports the 5 standard models plus dynamic custom local model configuration.
    pub fn for_model(model_idx: i32, custom_arch: i32) -> Self {
        if model_idx == 5 {
            match custom_arch {
                1 => Self {
                    // SigLIP-type architectures (384x384, Symmetric normalization: [-1, 1])
                    target_size: (384, 384),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                2 => Self {
                    // DINOv2-type backbones (224x224, standard ImageNet normalization)
                    target_size: (224, 224),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                _ => Self {
                    // CLIP-type architectures (224x224, OpenAI CLIP normalization)
                    target_size: (224, 224),
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.2613026, 0.2757771],
                },
            }
        } else {
            match model_idx {
                1 => Self {
                    // CLIP-L/14 (224x224, OpenAI CLIP normalization)
                    target_size: (224, 224),
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.2613026, 0.2757771],
                },
                2 => Self {
                    // SigLIP-B (384x384, Symmetric normalization: [-1, 1])
                    target_size: (384, 384),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                3 => Self {
                    // SigLIP-L (384x384, Symmetric normalization: [-1, 1])
                    target_size: (384, 384),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                4 => Self {
                    // DINOv2-B (224x224, standard ImageNet normalization)
                    target_size: (224, 224),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                _ => Self {
                    // CLIP-B/32 (224x224, OpenAI CLIP normalization)
                    target_size: (224, 224),
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.2613026, 0.2757771],
                },
            }
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self::for_model(0, 0)
    }
}

pub fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    vec
}

/// Identifies the output extraction mode for different ONNX model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualOutputMode {
    /// Extract "image_embeds" directly (Standard for CLIP)
    ImageEmbeds,
    /// Extract "pooler_output" directly (Standard for SigLIP ONNX models)
    PoolerOutput,
    /// Slice "last_hidden_state" to extract the first token ([CLS]) (Required for DINOv2 backbone)
    ClsToken,
    /// Safe fallback to the very first output of the ONNX model graph
    FirstOutput,
}

pub struct InferenceEngine {
    visual_session: Mutex<Session>,
    text_session: Option<Mutex<Session>>,
    tokenizer: Option<Tokenizer>,
    pub model_dim: usize,
    visual_output_mode: VisualOutputMode,
    fallback_output_name: String,
}

impl InferenceEngine {
    /// Loads the compiled visual and text ONNX models along with the native tokenizer
    pub fn new(
        model_dir: &Path,
        model_dim: usize,
        threads_per_worker: usize,
        execution_provider: &str,
    ) -> Result<Self> {
        let visual_path = model_dir.join("visual.onnx");
        let text_path = model_dir.join("text.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Map non-Send/Sync ort errors explicitly to anyhow strings
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to initialize Session builder: {:?}", e))?
            .with_intra_threads(threads_per_worker)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {:?}", e))?;

        match execution_provider {
            "DirectML" => {
                match builder.with_execution_providers([ort::ep::DirectML::default().build()]) {
                    Ok(b) => {
                        builder = b;
                    }
                    Err(e) => {
                        eprintln!(
                            "DirectML EP failed to register, falling back to CPU: {:?}",
                            e
                        );
                        builder = Session::builder()
                            .map_err(|eb| anyhow::anyhow!("Fallback error: {:?}", eb))?
                            .with_intra_threads(threads_per_worker)
                            .map_err(|eb| anyhow::anyhow!("Fallback thread error: {:?}", eb))?;
                    }
                }
            }
            "CUDA" => match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                Ok(b) => {
                    builder = b;
                }
                Err(e) => {
                    eprintln!("CUDA EP failed to register, falling back to CPU: {:?}", e);
                    builder = Session::builder()
                        .map_err(|eb| anyhow::anyhow!("Fallback error: {:?}", eb))?
                        .with_intra_threads(threads_per_worker)
                        .map_err(|eb| anyhow::anyhow!("Fallback thread error: {:?}", eb))?;
                }
            },
            "TensorRT" => {
                match builder.with_execution_providers([ort::ep::TensorRT::default().build()]) {
                    Ok(b) => {
                        builder = b;
                    }
                    Err(e) => {
                        eprintln!(
                            "TensorRT EP failed to register, falling back to CPU: {:?}",
                            e
                        );
                        builder = Session::builder()
                            .map_err(|eb| anyhow::anyhow!("Fallback error: {:?}", eb))?
                            .with_intra_threads(threads_per_worker)
                            .map_err(|eb| anyhow::anyhow!("Fallback thread error: {:?}", eb))?;
                    }
                }
            }
            "CoreML" => {
                match builder.with_execution_providers([ort::ep::CoreML::default().build()]) {
                    Ok(b) => {
                        builder = b;
                    }
                    Err(e) => {
                        eprintln!("CoreML EP failed to register, falling back to CPU: {:?}", e);
                        builder = Session::builder()
                            .map_err(|eb| anyhow::anyhow!("Fallback error: {:?}", eb))?
                            .with_intra_threads(threads_per_worker)
                            .map_err(|eb| anyhow::anyhow!("Fallback thread error: {:?}", eb))?;
                    }
                }
            }
            _ => {}
        }

        let visual_session = builder.commit_from_file(&visual_path)?;

        // Inspect the ONNX model outputs dynamically to detect the output architecture
        let output_names: Vec<String> = visual_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let visual_output_mode = if output_names.iter().any(|n| n == "image_embeds") {
            VisualOutputMode::ImageEmbeds
        } else if output_names.iter().any(|n| n == "pooler_output") {
            VisualOutputMode::PoolerOutput
        } else if output_names.iter().any(|n| n == "last_hidden_state") {
            VisualOutputMode::ClsToken
        } else {
            VisualOutputMode::FirstOutput
        };

        let fallback_output_name = output_names
            .first()
            .cloned()
            .unwrap_or_else(|| "image_embeds".to_string());

        tracing::info!(
            "ONNX Visual Model loaded successfully. Output names detected: {:?}. Strategy chosen: {:?}. Fallback node name: '{}'",
            output_names,
            visual_output_mode,
            fallback_output_name
        );

        let text_session = if text_path.exists() {
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("Failed to initialize Text Session builder: {:?}", e))?
                .with_intra_threads(threads_per_worker)
                .map_err(|e| anyhow::anyhow!("Failed to set text threads: {:?}", e))?
                .commit_from_file(&text_path)?;
            Some(Mutex::new(session))
        } else {
            None
        };

        // Load the local HuggingFace tokenizer config natively in Rust
        let tokenizer = if tokenizer_path.exists() {
            let tok = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer.json: {}", e))?;
            Some(tok)
        } else {
            None
        };

        Ok(Self {
            visual_session: Mutex::new(visual_session),
            text_session,
            tokenizer,
            model_dim,
            visual_output_mode,
            fallback_output_name,
        })
    }

    pub fn preprocess_image(
        &self,
        img: &DynamicImage,
        config: &PreprocessingConfig,
    ) -> Array4<f32> {
        let (tw, th) = config.target_size;
        let resized = img.resize_exact(tw, th, image::imageops::FilterType::Triangle);
        let raw = resized.to_rgb8().into_raw();

        let mut tensor = Array::zeros((1, 3, th as usize, tw as usize));
        let plane_len = (tw * th) as usize;

        if let Some(slice) = tensor.as_slice_mut() {
            let (r_plane, rest) = slice.split_at_mut(plane_len);
            let (g_plane, b_plane) = rest.split_at_mut(plane_len);

            for i in 0..plane_len {
                r_plane[i] = ((raw[i * 3] as f32 / 255.0) - config.mean[0]) / config.std[0];
                g_plane[i] = ((raw[i * 3 + 1] as f32 / 255.0) - config.mean[1]) / config.std[1];
                b_plane[i] = ((raw[i * 3 + 2] as f32 / 255.0) - config.mean[2]) / config.std[2];
            }
        }
        tensor
    }

    pub fn encode_images_batch(
        &self,
        images: &[DynamicImage],
        config: &PreprocessingConfig,
    ) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let (tw, th) = config.target_size;
        let batch_size = images.len();

        use rayon::prelude::*;
        let preprocessed_tensors: Vec<Array4<f32>> = images
            .par_iter()
            .map(|img| self.preprocess_image(img, config))
            .collect();

        let mut batch_tensor = Array::zeros((batch_size, 3, th as usize, tw as usize));
        let img_bytes_len = 3 * (tw * th) as usize;

        if let Some(batch_slice) = batch_tensor.as_slice_mut() {
            for (idx, tensor) in preprocessed_tensors.iter().enumerate() {
                if let Some(src_slice) = tensor.as_slice() {
                    let dst_start = idx * img_bytes_len;
                    batch_slice[dst_start..dst_start + img_bytes_len].copy_from_slice(src_slice);
                }
            }
        }

        let input_tensor = Value::from_array(batch_tensor)?;
        let mut session = self
            .visual_session
            .lock()
            .map_err(|_| anyhow::anyhow!("Session lock poisoned"))?;

        let outputs = session.run(inputs!["pixel_values" => &input_tensor])?;

        // Dynamically slice or read the correct outputs based on the model signature
        let (shape, raw_slice): (Vec<i64>, std::borrow::Cow<'_, [f32]>) = match self
            .visual_output_mode
        {
            VisualOutputMode::ImageEmbeds => {
                let (sh, slice) = outputs["image_embeds"].try_extract_tensor::<f32>()?;
                (sh.to_vec(), std::borrow::Cow::Borrowed(slice))
            }
            VisualOutputMode::PoolerOutput => {
                let (sh, slice) = outputs["pooler_output"].try_extract_tensor::<f32>()?;
                (sh.to_vec(), std::borrow::Cow::Borrowed(slice))
            }
            VisualOutputMode::ClsToken => {
                let (raw_shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
                let b_size = raw_shape[0] as usize;
                let seq_len = raw_shape[1] as usize;
                let hidden_size = raw_shape[2] as usize;

                // DINOv2 returns [batch, seq_len, 768]. Slice the CLS token (sequence index 0)
                let mut cls_embeddings = Vec::with_capacity(b_size * hidden_size);
                for b in 0..b_size {
                    let start_idx = b * seq_len * hidden_size;
                    let cls_token_slice = &data[start_idx..start_idx + hidden_size];
                    cls_embeddings.extend_from_slice(cls_token_slice);
                }
                (
                    vec![b_size as i64, hidden_size as i64],
                    std::borrow::Cow::Owned(cls_embeddings),
                )
            }
            VisualOutputMode::FirstOutput => {
                let (sh, slice) =
                    outputs[self.fallback_output_name.as_str()].try_extract_tensor::<f32>()?;
                (sh.to_vec(), std::borrow::Cow::Borrowed(slice))
            }
        };

        if shape[0] as usize != batch_size || shape[1] as usize != self.model_dim {
            return Err(anyhow::anyhow!(
                "Output shape mismatch! Expected [{}, {}], got {:?}",
                batch_size,
                self.model_dim,
                shape
            ));
        }

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.model_dim;
            results.push(normalize_vector(
                raw_slice[start..start + self.model_dim].to_vec(),
            ));
        }

        Ok(results)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer.as_ref().context("Tokenizer is not loaded")?;
        let text_session_mutex = self
            .text_session
            .as_ref()
            .context("Text model is not loaded (DINOv2 does not support text search)")?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask_vec: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        let length = input_ids_vec.len();
        let input_ids_arr = Array::from_shape_vec((1, length), input_ids_vec)?.into_dyn();
        let attention_mask_arr = Array::from_shape_vec((1, length), attention_mask_vec)?.into_dyn();

        let mut text_session = text_session_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Session lock poisoned"))?;
        let outputs = text_session.run(inputs![
            "input_ids" => &Value::from_array(input_ids_arr)?,
            "attention_mask" => &Value::from_array(attention_mask_arr)?
        ])?;

        let (_, raw_slice) = outputs["text_embeds"].try_extract_tensor::<f32>()?;
        Ok(normalize_vector(raw_slice.to_vec()))
    }
}
