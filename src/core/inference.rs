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

use crate::state::models::AiModelType;

#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_size: (u32, u32),
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PreprocessingConfig {
    /// Resolves the correct image scaling and normalization parameters based on the chosen AI architecture.
    pub fn for_model(model: AiModelType, custom_arch: i32) -> Self {
        match model {
            AiModelType::Custom => match custom_arch {
                1 => Self {
                    // SigLIP-style symmetric normalization [-1, 1] at 384x384 resolution
                    target_size: (384, 384),
                    mean: [0.5, 0.5, 0.5],
                    std: [0.5, 0.5, 0.5],
                },
                2 => Self {
                    // DINOv2 standard ImageNet normalization
                    target_size: (224, 224),
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225],
                },
                _ => Self {
                    // OpenAI CLIP normalization coefficients
                    target_size: (224, 224),
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.2613026, 0.2757771],
                },
            },
            AiModelType::ClipVitB32 | AiModelType::ClipVitL14 => Self {
                target_size: (224, 224),
                mean: [0.48145466, 0.4578275, 0.40821073],
                std: [0.26862954, 0.2613026, 0.2757771],
            },
            AiModelType::SiglipBase | AiModelType::SiglipLarge => Self {
                target_size: (384, 384),
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            },
            AiModelType::DinoV2Base => Self {
                target_size: (224, 224),
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            AiModelType::Siglip2Base => Self {
                target_size: (224, 224),
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            },
            AiModelType::Llm2ClipBase => Self {
                target_size: (224, 224),
                mean: [0.48145466, 0.4578275, 0.40821073],
                std: [0.26862954, 0.2613026, 0.2757771],
            },
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self::for_model(AiModelType::ClipVitB32, 0)
    }
}

/// Computes L2 vector normalization.
pub fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    vec
}

/// Identifies the output tensor node extraction strategy for various model graph shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualOutputMode {
    /// Extract 'image_embeds' node (typical of CLIP models)
    ImageEmbeds,
    /// Extract 'pooler_output' node (typical of SigLIP models)
    PoolerOutput,
    /// Slice 'last_hidden_state' to pull the CLS token (typical of raw ViT / DINOv2 backbones)
    ClsToken,
    /// Fallback to the first output of the graph
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
    /// Builds and configures runtime visual and text sessions.
    pub fn new(
        model_dir: &Path,
        model_dim: usize,
        threads: usize,
        execution_provider: &str,
    ) -> Result<Self> {
        let visual_path = model_dir.join("visual.onnx");
        let text_path = model_dir.join("text.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Self-healing: If compilation fails, automatically delete the corrupted file via inspect_err to trigger a clean re-download next time
        let visual_session = create_session(&visual_path, threads, execution_provider)
            .inspect_err(|e| {
                tracing::warn!(
                    "Corrupted visual model detected. Auto-deleting '{}' to heal. Error: {:?}",
                    visual_path.display(),
                    e
                );
                let _ = std::fs::remove_file(&visual_path);
            })?;

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
            "ONNX Session initialized. Graph output nodes: {:?}. Strategy: {:?}.",
            output_names,
            visual_output_mode
        );

        let text_session = if text_path.exists() {
            let session =
                create_session(&text_path, threads, execution_provider).inspect_err(|e| {
                    tracing::warn!(
                        "Corrupted text model detected. Auto-deleting '{}' to heal. Error: {:?}",
                        text_path.display(),
                        e
                    );
                    let _ = std::fs::remove_file(&text_path);
                })?;
            Some(Mutex::new(session))
        } else {
            None
        };

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

    /// Converts raw RGB buffer into structured planar CHW floats mapping to the normalization coefficients.
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
                let r_val = raw[i * 3] as f32 / 255.0;
                let g_val = raw[i * 3 + 1] as f32 / 255.0;
                let b_val = raw[i * 3 + 2] as f32 / 255.0;

                r_plane[i] = (r_val - config.mean[0]) / config.std[0];
                g_plane[i] = (g_val - config.mean[1]) / config.std[1];
                b_plane[i] = (b_val - config.mean[2]) / config.std[2];
            }
        }
        tensor
    }

    /// Preprocesses image frames in parallel, aggregates them into a batch tensor, and executes inference.
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

        // Retrieve input names to detect whether this is a unified model expecting text inputs
        let inputs_names: std::collections::HashSet<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        // Provide dummy input tensors if the model expects text inputs alongside images (e.g. LLM2CLIP)
        let outputs = if inputs_names.contains("input_ids") {
            let dummy_input_ids =
                Value::from_array(Array::<i64, _>::zeros((batch_size, 1)).into_dyn())?;
            if inputs_names.contains("attention_mask") {
                let dummy_attention_mask =
                    Value::from_array(Array::<i64, _>::ones((batch_size, 1)).into_dyn())?;
                session.run(inputs![
                    "pixel_values" => &input_tensor,
                    "input_ids" => &dummy_input_ids,
                    "attention_mask" => &dummy_attention_mask
                ])?
            } else {
                session.run(inputs![
                    "pixel_values" => &input_tensor,
                    "input_ids" => &dummy_input_ids
                ])?
            }
        } else {
            session.run(inputs!["pixel_values" => &input_tensor])?
        };

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

                // Slice CLS token (sequence index 0) for ViT backbones (e.g. DINOv2)
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

    /// Tokenizes query text and runs inference inside the text-encoder session.
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer.as_ref().context("Tokenizer is not loaded")?;
        let text_session_mutex = self.text_session.as_ref().context(
            "Text model is not loaded (selected architecture does not support text query)",
        )?;

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

/// Helper function to create an ONNX Runtime session cleanly, handling fallback steps gracefully.
fn create_session(model_path: &Path, threads: usize, provider: &str) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to instantiate session builder: {:?}", e))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("Failed to set thread pool limit: {:?}", e))?;

    builder = match provider {
        "DirectML" => builder.with_execution_providers([ort::ep::DirectML::default().build()]),
        "CUDA" => builder.with_execution_providers([ort::ep::CUDA::default().build()]),
        "TensorRT" => builder.with_execution_providers([ort::ep::TensorRT::default().build()]),
        "CoreML" => builder.with_execution_providers([ort::ep::CoreML::default().build()]),
        _ => Ok(builder),
    }
    .unwrap_or_else(|e| {
        tracing::warn!(
            "Failed to bind hardware EP '{}', falling back to CPU: {:?}",
            provider,
            e
        );
        // Fall back to clean CPU builder state
        Session::builder()
            .unwrap()
            .with_intra_threads(threads)
            .unwrap()
    });

    builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Model compilation run failed: {:?}", e))
}
