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
        if execution_provider == "DirectML" {
            builder = builder.with_execution_providers([ort::ep::DirectML::default().build()])?;
        }

        let visual_session = builder.commit_from_file(&visual_path)?;

        let text_session = if text_path.exists() {
            let mut text_builder = Session::builder()?.with_intra_threads(threads_per_worker)?;
            if execution_provider == "DirectML" {
                text_builder = text_builder
                    .with_execution_providers([ort::ep::DirectML::default().build()])?;
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

        let mut tensor = Array::zeros((1, 3, th as usize, tw as usize));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            // Map pixel range [0, 255] to [0.0, 1.0] and apply standard mean/std normalization
            let r = ((pixel[0] as f32 / 255.0) - config.mean[0]) / config.std[0];
            let g = ((pixel[1] as f32 / 255.0) - config.mean[1]) / config.std[1];
            let b = ((pixel[2] as f32 / 255.0) - config.mean[2]) / config.std[2];

            tensor[[0, 0, y as usize, x as usize]] = r;
            tensor[[0, 1, y as usize, x as usize]] = g;
            tensor[[0, 2, y as usize, x as usize]] = b;
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
