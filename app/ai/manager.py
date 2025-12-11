# app/ai/manager.py
"""
Contains worker functions for parallel computation (Preprocessing and Inference)
and manages the Model lifecycle via a Singleton manager.
"""

import gc
import logging
import multiprocessing
import os
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from PIL import Image

# Pre-load libraries to avoid import lock contention
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoProcessor  # noqa: F401
except ImportError:
    pass

from app.imaging.image_io import load_image
from app.shared.constants import (
    APP_DATA_DIR,
    DEEP_LEARNING_AVAILABLE,
    FP16_MODEL_SUFFIX,
    MODELS_DIR,
)

if TYPE_CHECKING:
    from app.domain.data_models import AnalysisItem


app_logger = logging.getLogger("PixelHand.ai.manager")


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes a batch of vectors in-place using NumPy (L2 Norm)."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


class InferenceEngine:
    """Wrapper around ONNX Runtime for running AI models."""

    def __init__(self, model_name: str, device: str = "CPUExecutionProvider", threads_per_worker: int = 1):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")

        from transformers import AutoProcessor

        model_dir = MODELS_DIR / model_name
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.is_fp16 = FP16_MODEL_SUFFIX in model_name.lower()

        # Extract input size
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
        self.input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)

        self.text_session = None
        self.text_input_names = set()

        self._load_onnx_model(model_dir, device, threads_per_worker)

    def _load_onnx_model(self, model_dir: Path, device: str, threads_per_worker: int):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if device not in available_providers:
            app_logger.warning(f"Requested provider '{device}' not found. Fallback to CPU.")
            device = "CPUExecutionProvider"

        providers = [device]
        if device != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        if device == "DmlExecutionProvider" and self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
            opts.enable_float16_for_dml = True

        if device == "CPUExecutionProvider":
            total_cores = multiprocessing.cpu_count()
            if threads_per_worker <= 1:
                estimated_workers = 4
                threads_to_use = max(1, total_cores // estimated_workers)
            else:
                threads_to_use = threads_per_worker

            opts.intra_op_num_threads = threads_to_use
            opts.inter_op_num_threads = 1
            if threads_to_use > 1:
                opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            else:
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        onnx_file = model_dir / "visual.onnx"
        self.visual_session = ort.InferenceSession(str(onnx_file), opts, providers=providers)

        text_model_path = model_dir / "text.onnx"
        if text_model_path.exists():
            self.text_session = ort.InferenceSession(str(text_model_path), opts, providers=providers)
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}

        app_logger.info(f"ONNX Loaded. Device: {device}, FP16: {self.is_fp16}")

    def get_text_features(self, text: str) -> np.ndarray:
        if not self.text_session:
            raise RuntimeError("Text model not loaded.")
        inputs = self.processor.tokenizer(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
            return_tensors="np",
            return_attention_mask=True,
        )
        onnx_inputs = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in self.text_input_names:
            onnx_inputs["attention_mask"] = inputs["attention_mask"]
        outputs = self.text_session.run(None, onnx_inputs)
        return normalize_vectors_numpy(outputs[0]).flatten()


class ModelManager:
    """Singleton to manage the lifecycle of AI models."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.inference_engine: InferenceEngine | None = None
        self.preprocessor: Any | None = None
        self.current_model_name: str | None = None
        self.access_lock = threading.RLock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, config: dict):
        requested_model = config["model_name"]
        with self.access_lock:
            if self.inference_engine and self.current_model_name == requested_model:
                return

            app_logger.info(f"ModelManager: Switching model {self.current_model_name} -> {requested_model}")
            self.release_resources()
            gc.collect()

            try:
                self.inference_engine = InferenceEngine(
                    model_name=requested_model,
                    device=config.get("device", "CPUExecutionProvider"),
                    threads_per_worker=config.get("threads_per_worker", 1),
                )
                from transformers import AutoProcessor

                model_dir = MODELS_DIR / requested_model
                self.preprocessor = AutoProcessor.from_pretrained(str(model_dir))
                self.current_model_name = requested_model
            except Exception as e:
                app_logger.error(f"Failed to load model {requested_model}: {e}")
                self.release_resources()
                raise e

    def release_resources(self):
        with self.access_lock:
            if self.inference_engine:
                self.inference_engine.visual_session = None
                self.inference_engine.text_session = None
                self.inference_engine = None
            self.preprocessor = None
            self.current_model_name = None
        gc.collect()

    def get_input_size(self) -> tuple[int, int]:
        with self.access_lock:
            if self.preprocessor:
                image_proc = getattr(self.preprocessor, "image_processor", self.preprocessor)
                size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
                return (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
            return (224, 224)

    def get_models_snapshot(self) -> tuple[InferenceEngine | None, Any | None]:
        with self.access_lock:
            return self.inference_engine, self.preprocessor


# --- Helper Functions ---
def cleanup_worker():
    ModelManager.instance().release_resources()


def init_worker(config: dict):
    try:
        ModelManager.instance().load_model(config)
    except Exception as e:
        _log_worker_crash(e, "init_worker")


def init_preprocessor_worker(config: dict):
    init_worker(config)


def get_model_input_size() -> tuple[int, int]:
    return ModelManager.instance().get_input_size()


# --- Optimized Processing Logic ---
def _read_and_process_batch_for_ai(
    items: list["AnalysisItem"], input_size: tuple[int, int], simple_config: dict
) -> tuple[list[Image.Image], list[tuple[str, str | None]], list[tuple[str, str]]]:
    """Reads images, handles resizing/channels, prepares for AI."""
    images = []
    successful_paths = []
    skipped = []
    ignore_solid = simple_config.get("ignore_solid_channels", True)

    for item in items:
        path = Path(item.path)
        try:
            # Heuristic shrink
            file_size = path.stat().st_size
            shrink = 1
            if file_size > 50 * 1024 * 1024:
                shrink = 8
            elif file_size > 10 * 1024 * 1024:
                shrink = 4
            elif file_size > 2 * 1024 * 1024:
                shrink = 2

            pil_image = load_image(path, shrink=shrink)
            if not pil_image:
                skipped.append((str(path), "Image loading failed"))
                continue

            if pil_image.mode not in ("RGB", "RGBA", "L"):
                pil_image = pil_image.convert("RGBA")

            processed_image = None
            channel_name = None

            if item.analysis_type in ("R", "G", "B", "A"):
                if pil_image.mode != "RGBA":
                    pil_image = pil_image.convert("RGBA")
                channels = pil_image.split()
                idx = {"R": 0, "G": 1, "B": 2, "A": 3}[item.analysis_type]
                if idx < len(channels):
                    ch = channels[idx]
                    # Fixed SIM102 and E701
                    if ignore_solid and ch.getextrema()[1] < 5:
                        continue
                    processed_image = Image.merge("RGB", (ch, ch, ch))
                    channel_name = item.analysis_type

            elif item.analysis_type == "Luminance":
                processed_image = pil_image.convert("L").convert("RGB")
            else:
                if pil_image.mode == "RGBA":
                    # VFX Fix: Alpha=0 but RGB>0
                    ext = pil_image.getextrema()
                    if ext[3][1] == 0 and (ext[0][1] > 0 or ext[1][1] > 0 or ext[2][1] > 0):
                        processed_image = pil_image.convert("RGB")
                    else:
                        bg = Image.new("RGB", pil_image.size, (0, 0, 0))
                        bg.paste(pil_image, mask=pil_image.split()[3])
                        processed_image = bg
                else:
                    processed_image = pil_image.convert("RGB")

            if processed_image:
                if processed_image.size != input_size:
                    processed_image = processed_image.resize(input_size, Image.Resampling.BILINEAR)
                images.append(processed_image)
                successful_paths.append((str(path), channel_name))
            del pil_image

        except Exception as e:
            skipped.append((str(path), str(e)))

    return images, successful_paths, skipped


def worker_preprocess_threaded(
    items: list["AnalysisItem"], input_size: tuple[int, int], dtype: np.dtype, simple_config: dict, output_queue: Any
):
    _, preprocessor = ModelManager.instance().get_models_snapshot()
    if not preprocessor:
        output_queue.put((None, [], [(str(i.path), "Model not initialized") for i in items]))
        return

    images, success_paths, skipped = _read_and_process_batch_for_ai(items, input_size, simple_config)
    if not images:
        output_queue.put((None, [], skipped))
        return

    try:
        batch_dict = preprocessor(images=images, return_tensors="np")
        pixel_values = batch_dict.pixel_values.astype(dtype)
        del images
        gc.collect()
        output_queue.put((pixel_values, success_paths, skipped))
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess")
        output_queue.put((None, [], [(str(i.path), f"Preproc: {e}") for i in items]))


def run_inference_direct(pixel_values: np.ndarray, paths_with_channels: list) -> tuple[dict, list]:
    engine, _ = ModelManager.instance().get_models_snapshot()
    if not engine:
        return {}, [(p, "Engine missing") for p, _ in paths_with_channels]

    try:
        io_binding = engine.visual_session.io_binding()
        io_binding.bind_cpu_input("pixel_values", pixel_values)
        io_binding.bind_output("image_embeds")
        engine.visual_session.run_with_iobinding(io_binding)
        embeddings = io_binding.get_outputs()[0].numpy()
        del pixel_values

        embeddings = normalize_vectors_numpy(embeddings)
        # Fixed B905 (zip strict)
        results = {tuple(pc): vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
        return results, []
    except Exception as e:
        _log_worker_crash(e, "inference")
        return {}, [(p, f"Infer: {e}") for p, _ in paths_with_channels]


def worker_get_single_vector(image_path_str: str) -> np.ndarray | None:
    engine, preproc = ModelManager.instance().get_models_snapshot()
    # Fixed E701
    if not engine or not preproc:
        return None
    try:
        from app.domain.data_models import AnalysisItem

        items = [AnalysisItem(path=Path(image_path_str), analysis_type="Composite")]
        images, _, _ = _read_and_process_batch_for_ai(items, engine.input_size, {})
        if images:
            px = preproc(images=images, return_tensors="np").pixel_values
            # Fixed E701
            if engine.is_fp16:
                px = px.astype(np.float16)
            io = engine.visual_session.io_binding()
            io.bind_cpu_input("pixel_values", px)
            io.bind_output("image_embeds")
            engine.visual_session.run_with_iobinding(io)
            return normalize_vectors_numpy(io.get_outputs()[0].numpy()).flatten()
    except Exception:
        pass
    return None


def worker_get_text_vector(text: str) -> np.ndarray | None:
    engine, _ = ModelManager.instance().get_models_snapshot()
    return engine.get_text_features(text) if engine else None


def _log_worker_crash(e: Exception, context: str):
    pid = os.getpid()
    tid = threading.get_ident()
    log_file = APP_DATA_DIR / "crash_logs" / f"worker_error_{pid}_{tid}_{int(time.time())}.txt"
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"Context: {context}\nError: {e}\n{traceback.format_exc()}")
    app_logger.error(f"Worker Error in {context}: {e}")
