// src-tauri/src/inference.rs
use image::DynamicImage;
use ndarray::{Array, Array4};
use ort::inputs;
use ort::session::Session;
use ort::value::Value;
use std::error::Error;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Normalization parameters used to prepare image tensors for AI models
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_size: (u32, u32),
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_size: (224, 224),
            mean: [0.48145466, 0.4578275, 0.40821073], // CLIP default normalization
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

/// Normalizes a floating-point vector to unit length (L2 norm) for Cosine Similarity search
pub fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    vec
}

/// Standalone AI Inference Engine wrapping ONNX Sessions in thread-safe Mutexes
pub struct InferenceEngine {
    visual_session: Mutex<Session>,
    text_session: Option<Mutex<Session>>,
    tokenizer: Option<Tokenizer>,
    model_dim: usize,
}

impl InferenceEngine {
    /// Loads the compiled visual and text ONNX models along with the native tokenizer
    pub fn new(
        model_dir: &Path,
        model_dim: usize,
        threads_per_worker: usize,
        execution_provider: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let visual_path = model_dir.join("visual.onnx");
        let text_path = model_dir.join("text.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let mut builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
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
                        builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
                    }
                }
            }
            "CUDA" => match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                Ok(b) => {
                    builder = b;
                }
                Err(e) => {
                    eprintln!("CUDA EP failed to register, falling back to CPU: {:?}", e);
                    builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
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
                        builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
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
                        builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
                    }
                }
            }
            _ => {}
        }

        let visual_session = builder.commit_from_file(&visual_path)?;

        let text_session = if text_path.exists() {
            let mut text_builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
            match execution_provider {
                "DirectML" => {
                    match text_builder
                        .with_execution_providers([ort::ep::DirectML::default().build()])
                    {
                        Ok(b) => {
                            text_builder = b;
                        }
                        Err(e) => {
                            eprintln!(
                                "DirectML text EP failed to register, falling back to CPU: {:?}",
                                e
                            );
                            text_builder =
                                Session::builder()?.with_intra_threads(threads_per_worker)?;
                        }
                    }
                }
                "CUDA" => {
                    match text_builder.with_execution_providers([ort::ep::CUDA::default().build()])
                    {
                        Ok(b) => {
                            text_builder = b;
                        }
                        Err(e) => {
                            eprintln!(
                                "CUDA text EP failed to register, falling back to CPU: {:?}",
                                e
                            );
                            text_builder =
                                Session::builder()?.with_intra_threads(threads_per_worker)?;
                        }
                    }
                }
                "TensorRT" => {
                    match text_builder
                        .with_execution_providers([ort::ep::TensorRT::default().build()])
                    {
                        Ok(b) => {
                            text_builder = b;
                        }
                        Err(e) => {
                            eprintln!(
                                "TensorRT text EP failed to register, falling back to CPU: {:?}",
                                e
                            );
                            text_builder =
                                Session::builder()?.with_intra_threads(threads_per_worker)?;
                        }
                    }
                }
                "CoreML" => {
                    match text_builder
                        .with_execution_providers([ort::ep::CoreML::default().build()])
                    {
                        Ok(b) => {
                            text_builder = b;
                        }
                        Err(e) => {
                            eprintln!(
                                "CoreML text EP failed to register, falling back to CPU: {:?}",
                                e
                            );
                            text_builder =
                                Session::builder()?.with_intra_threads(threads_per_worker)?;
                        }
                    }
                }
                _ => {}
            }
            let session = text_builder.commit_from_file(&text_path)?;
            Some(Mutex::new(session))
        } else {
            None
        };

        // Load the local HuggingFace tokenizer config natively in Rust
        let tokenizer = if tokenizer_path.exists() {
            let tok = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("Failed to parse tokenizer.json: {}", e))?;
            Some(tok)
        } else {
            None
        };

        Ok(Self {
            visual_session: Mutex::new(visual_session),
            text_session,
            tokenizer,
            model_dim,
        })
    }

    /// Preprocesses a dynamic image to NCHW format, scales pixels, and applies normalization
    pub fn preprocess_image(
        &self,
        img: &DynamicImage,
        config: &PreprocessingConfig,
    ) -> Array4<f32> {
        let (tw, th) = config.target_size;

        // Fast resize using a Bilinear filter
        let resized = img.resize_exact(tw, th, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let raw = rgb.as_raw();

        let mut tensor = Array::zeros((1, 3, th as usize, tw as usize));
        let tw_u = tw as usize;
        let th_u = th as usize;
        let plane_len = tw_u * th_u;

        // Use Ndarray flat slices mapping to avoid nested multidx cost and allow autovectorization
        if let Some(slice) = tensor.as_slice_mut() {
            let (r_plane, rest) = slice.split_at_mut(plane_len);
            let (g_plane, b_plane) = rest.split_at_mut(plane_len);

            for i in 0..plane_len {
                let r_val = raw[i * 3];
                let g_val = raw[i * 3 + 1];
                let b_val = raw[i * 3 + 2];

                r_plane[i] = ((r_val as f32 / 255.0) - config.mean[0]) / config.std[0];
                g_plane[i] = ((g_val as f32 / 255.0) - config.mean[1]) / config.std[1];
                b_plane[i] = ((b_val as f32 / 255.0) - config.mean[2]) / config.std[2];
            }
        }

        tensor
    }

    /// Generates L2-normalized embeddings for a given image
    pub fn encode_image(
        &self,
        img: &DynamicImage,
        config: &PreprocessingConfig,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let tensor = self.preprocess_image(img, config);

        let input_tensor = Value::from_array(tensor)?;

        let mut session = self
            .visual_session
            .lock()
            .map_err(|_| "Visual session lock poisoned")?;

        // Run the visual model on the locked mutable session
        let outputs = session.run(inputs![
            "pixel_values" => &input_tensor
        ])?;

        // Retrieve the output tensor (image_embeds)
        let output_tensor = outputs["image_embeds"].try_extract_tensor::<f32>()?;

        // Extract slice, verify model dimension, and normalize
        let (_shape, raw_slice) = output_tensor;
        let raw_vector: Vec<f32> = raw_slice.to_vec();

        if raw_vector.len() != self.model_dim {
            return Err(format!(
                "Output shape mismatch! Expected size {}, got {}.",
                self.model_dim,
                raw_vector.len()
            )
            .into());
        }

        Ok(normalize_vector(raw_vector))
    }

    /// Generates L2-normalized embeddings for a batch of images simultaneously
    pub fn encode_images_batch(
        &self,
        images: &[DynamicImage],
        config: &PreprocessingConfig,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let tw = config.target_size.0 as usize;
        let th = config.target_size.1 as usize;
        let batch_size = images.len();

        // 1. Preprocess all images in parallel using rayon
        use rayon::prelude::*;
        let preprocessed_tensors: Vec<Array4<f32>> = images
            .par_iter()
            .map(|img| self.preprocess_image(img, config))
            .collect();

        // 2. Concatenate all tensors into a single [B, 3, H, W] tensor
        let mut batch_tensor = Array::zeros((batch_size, 3, th, tw));
        let plane_len = tw * th;
        let img_bytes_len = 3 * plane_len;

        if let Some(batch_slice) = batch_tensor.as_slice_mut() {
            for (idx, tensor) in preprocessed_tensors.iter().enumerate() {
                if let Some(src_slice) = tensor.as_slice() {
                    let dst_start = idx * img_bytes_len;
                    let dst_end = dst_start + img_bytes_len;
                    batch_slice[dst_start..dst_end].copy_from_slice(src_slice);
                }
            }
        }

        let input_tensor = Value::from_array(batch_tensor)?;

        let mut session = self
            .visual_session
            .lock()
            .map_err(|_| "Visual session lock poisoned")?;

        // Run the visual model on the locked mutable session
        let outputs = session.run(inputs![
            "pixel_values" => &input_tensor
        ])?;

        // Retrieve the output tensor (image_embeds)
        let output_tensor = outputs["image_embeds"].try_extract_tensor::<f32>()?;

        // Extract shape and raw slice
        let (shape, raw_slice) = output_tensor;
        let out_batch_size = shape[0] as usize;
        let out_dim = shape[1] as usize;

        if out_batch_size != batch_size || out_dim != self.model_dim {
            return Err(format!(
                "Output shape mismatch! Expected [{}, {}], got {:?}",
                batch_size, self.model_dim, shape
            )
            .into());
        }

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.model_dim;
            let end = start + self.model_dim;
            let vec = raw_slice[start..end].to_vec();
            results.push(normalize_vector(vec));
        }

        Ok(results)
    }

    /// Encodes a text query into an L2-normalized semantic vector for text-to-image search
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer is not loaded")?;
        let text_session_mutex = self
            .text_session
            .as_ref()
            .ok_or("Text ONNX model is not loaded")?;

        // Encode query string using Hugging Face's tokenizer logic
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask_vec: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();

        let length = input_ids_vec.len();

        // Convert token arrays to dynamic 2D tensors with a batch size of 1
        let input_ids_arr = Array::from_shape_vec((1, length), input_ids_vec)?.into_dyn();
        let attention_mask_arr = Array::from_shape_vec((1, length), attention_mask_vec)?.into_dyn();

        let input_ids_val = Value::from_array(input_ids_arr)?;
        let attention_mask_val = Value::from_array(attention_mask_arr)?;

        let mut text_session = text_session_mutex
            .lock()
            .map_err(|_| "Text session lock poisoned")?;

        // Run the text ONNX session
        let outputs = text_session.run(inputs![
            "input_ids" => &input_ids_val,
            "attention_mask" => &attention_mask_val
        ])?;

        let output_tensor = outputs["text_embeds"].try_extract_tensor::<f32>()?;

        // Extract slice, verify model dimension, and normalize
        let (_shape, raw_slice) = output_tensor;
        let raw_vector: Vec<f32> = raw_slice.to_vec();

        if raw_vector.len() != self.model_dim {
            return Err(format!(
                "Output shape mismatch! Expected size {}, got {}.",
                self.model_dim,
                raw_vector.len()
            )
            .into());
        }

        Ok(normalize_vector(raw_vector))
    }
}
