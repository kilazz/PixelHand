# app/ai/manager.py
"""
AI Model Manager & Inference Engine.
"""

import gc
import logging
import multiprocessing
import os
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

# Pre-load libraries to avoid import lock contention in threads
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

from app.imaging.image_io import load_image
from app.shared.constants import (
    APP_DATA_DIR,
    DEEP_LEARNING_AVAILABLE,
    FP16_MODEL_SUFFIX,
)

if TYPE_CHECKING:
    from app.domain.data_models import AnalysisItem

logger = logging.getLogger("PixelHand.ai.manager")

# --- Module-Level State for Worker Processes ---
# These global variables hold the model instance INSIDE a worker process.
# They are initialized via 'init_worker' called by the ProcessPoolExecutor.
_WORKER_ENGINE: Optional["InferenceEngine"] = None
_WORKER_PREPROCESSOR: Any = None


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes a batch of vectors in-place using NumPy (L2 Norm)."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    np.divide(embeddings, norms, out=embeddings, where=norms > 1e-12)
    return embeddings


class InferenceEngine:
    """
    Wrapper around ONNX Runtime for running AI models.
    Handles Tokenizers, Processors, and ONNX Sessions.
    """

    def __init__(
        self, model_name: str, models_dir: Path, device: str = "CPUExecutionProvider", threads_per_worker: int = 1
    ):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")

        self.model_dir = models_dir / model_name

        # Load HuggingFace Processor (Tokenizer + Image Config)
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to load processor from {self.model_dir}: {e}") from e

        self.is_fp16 = FP16_MODEL_SUFFIX in model_name.lower()

        # Extract input size from config
        # Handle various config structures (CLIP vs SigLIP vs DINOv2)
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})

        if "height" in size_cfg:
            self.input_size = (size_cfg["width"], size_cfg["height"])
        elif "shortest_edge" in size_cfg:
            # Approximation for DINOv2 / older configs
            s = size_cfg["shortest_edge"]
            self.input_size = (s, s)
        else:
            self.input_size = (224, 224)

        self.text_session = None
        self.text_input_names = set()
        self.visual_session = None

        self._load_onnx_model(device, threads_per_worker)

    def _load_onnx_model(self, device: str, threads_per_worker: int):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Provider Logic
        available_providers = ort.get_available_providers()
        if device not in available_providers:
            logger.warning(f"Requested provider '{device}' not found. Fallback to CPU.")
            device = "CPUExecutionProvider"

        providers = [device]
        if device != "CPUExecutionProvider":
            # Append CPU as fallback
            providers.append("CPUExecutionProvider")

        # FP16 specifics for DirectML
        if device == "DmlExecutionProvider" and self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
            opts.enable_float16_for_dml = True

        # Threading Configuration
        if device == "CPUExecutionProvider":
            total_cores = multiprocessing.cpu_count()
            if threads_per_worker <= 1:
                # Heuristic: split cores among workers
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

        # Load Visual Model
        onnx_file = self.model_dir / "visual.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"Model file missing: {onnx_file}")

        self.visual_session = ort.InferenceSession(str(onnx_file), opts, providers=providers)

        # Load Text Model (Optional)
        text_model_path = self.model_dir / "text.onnx"
        if text_model_path.exists():
            self.text_session = ort.InferenceSession(str(text_model_path), opts, providers=providers)
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}

        logger.info(f"ONNX Loaded. Device: {device}, FP16: {self.is_fp16}, Input: {self.input_size}")

    def get_text_features(self, text: str) -> np.ndarray:
        if not self.text_session:
            raise RuntimeError("Text model not available for this architecture.")

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
    """
    Manages AI resources in the MAIN process.
    Acts as a coordinator for single-shot inference (e.g., text search query)
    and holds configuration for where models are stored.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.current_engine: InferenceEngine | None = None
        self.current_model_name: str | None = None
        self._lock = threading.RLock()

    def ensure_model_loaded(self, config: dict[str, Any]):
        """
        Loads the model in the main process if needed (e.g., for single text query).
        Args:
            config: Dict containing 'model_name' and optionally 'device'.
        """
        requested_model = config["model_name"]

        with self._lock:
            if self.current_engine and self.current_model_name == requested_model:
                return

            logger.info(f"ModelManager: Switching model to {requested_model}")
            self.release_resources()

            try:
                self.current_engine = InferenceEngine(
                    model_name=requested_model,
                    models_dir=self.models_dir,
                    device=config.get("device", "CPUExecutionProvider"),
                    threads_per_worker=1,  # Main thread usually doesn't need heavy parallel ops
                )
                self.current_model_name = requested_model
            except Exception as e:
                logger.error(f"Failed to load model {requested_model}: {e}")
                self.release_resources()
                raise e

    def release_resources(self):
        with self._lock:
            if self.current_engine:
                self.current_engine.visual_session = None
                self.current_engine.text_session = None
                self.current_engine = None
            self.current_model_name = None
        gc.collect()

    def get_input_size(self) -> tuple[int, int]:
        with self._lock:
            if self.current_engine:
                return self.current_engine.input_size
            return (224, 224)


# --- Worker Functions (Run in separate processes) ---


def init_worker(config: dict[str, Any]):
    """
    Initializes the global model state inside a worker process.
    Called by ProcessPoolExecutor (initializer).
    """
    global _WORKER_ENGINE, _WORKER_PREPROCESSOR

    try:
        # Determine models directory
        models_dir = config.get("models_dir")
        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        elif not models_dir:
            # Fallback if not provided (though Pipeline should provide it)
            models_dir = APP_DATA_DIR / "models"

        _WORKER_ENGINE = InferenceEngine(
            model_name=config["model_name"],
            models_dir=models_dir,
            device=config.get("device", "CPUExecutionProvider"),
            threads_per_worker=config.get("threads_per_worker", 1),
        )
        _WORKER_PREPROCESSOR = _WORKER_ENGINE.processor
        logger.info(f"Worker process {os.getpid()} initialized.")

    except Exception as e:
        _log_worker_crash(e, "init_worker")
        raise e


def init_preprocessor_worker(config: dict[str, Any]):
    """Alias for init_worker, strictly for preprocessing threads if separated."""
    init_worker(config)


def get_model_input_size() -> tuple[int, int]:
    """Returns input size from the loaded worker engine."""
    if _WORKER_ENGINE:
        return _WORKER_ENGINE.input_size
    return (224, 224)


def _read_and_process_batch_for_ai(
    items: list["AnalysisItem"], input_size: tuple[int, int], simple_config: dict
) -> tuple[list[Image.Image], list[tuple[str, str]], list[tuple[str, str]]]:
    """Reads images, handles resizing/channels, prepares for AI."""
    images = []
    successful_paths = []
    skipped = []
    ignore_solid = simple_config.get("ignore_solid_channels", True)

    for item in items:
        path = Path(item.path)
        try:
            # Heuristic shrink for performance
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

            # Channel Logic
            if item.analysis_type in ("R", "G", "B", "A"):
                if pil_image.mode != "RGBA":
                    pil_image = pil_image.convert("RGBA")
                channels = pil_image.split()
                # Map channel name to index
                idx = {"R": 0, "G": 1, "B": 2, "A": 3}[item.analysis_type]

                if idx < len(channels):
                    ch = channels[idx]
                    # Skip solid black alpha/channels if requested
                    if ignore_solid and ch.getextrema()[1] < 5:
                        continue
                    processed_image = Image.merge("RGB", (ch, ch, ch))
                    channel_name = item.analysis_type

            elif item.analysis_type == "Luminance":
                processed_image = pil_image.convert("L").convert("RGB")

            else:  # "Composite"
                if pil_image.mode == "RGBA":
                    # VFX Fix: Check if Alpha=0 but RGB has data
                    ext = pil_image.getextrema()
                    # ext[3] is Alpha min/max
                    if ext[3][1] == 0 and (ext[0][1] > 0 or ext[1][1] > 0 or ext[2][1] > 0):
                        processed_image = pil_image.convert("RGB")
                    else:
                        # Composite onto black
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

            # Explicit cleanup
            del pil_image

        except Exception as e:
            skipped.append((str(path), str(e)))

    return images, successful_paths, skipped


def worker_preprocess_threaded(
    items: list["AnalysisItem"], input_size: tuple[int, int], dtype: np.dtype, simple_config: dict, output_queue: Any
):
    """
    Worker task: loads images, pre-processes them into numpy tensors, puts into queue.
    """
    global _WORKER_PREPROCESSOR
    if not _WORKER_PREPROCESSOR:
        output_queue.put((None, [], [(str(i.path), "Model not initialized in worker") for i in items]))
        return

    images, success_paths, skipped = _read_and_process_batch_for_ai(items, input_size, simple_config)

    if not images:
        output_queue.put((None, [], skipped))
        return

    try:
        # HuggingFace Processor
        batch_dict = _WORKER_PREPROCESSOR(images=images, return_tensors="np")
        pixel_values = batch_dict.pixel_values.astype(dtype)

        # Cleanup PIL images to free RAM
        del images
        gc.collect()

        output_queue.put((pixel_values, success_paths, skipped))
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess")
        output_queue.put((None, [], [(str(i.path), f"Preproc: {e}") for i in items]))


def run_inference_direct(pixel_values: np.ndarray, paths_with_channels: list) -> tuple[dict, list]:
    """
    Runs inference on pre-calculated pixel_values using the worker's engine.
    """
    global _WORKER_ENGINE
    if not _WORKER_ENGINE:
        return {}, [(p, "Engine missing") for p, _ in paths_with_channels]

    try:
        # Use IOBinding for efficiency
        io_binding = _WORKER_ENGINE.visual_session.io_binding()
        io_binding.bind_cpu_input("pixel_values", pixel_values)
        io_binding.bind_output("image_embeds")

        _WORKER_ENGINE.visual_session.run_with_iobinding(io_binding)
        embeddings = io_binding.get_outputs()[0].numpy()

        del pixel_values

        embeddings = normalize_vectors_numpy(embeddings)

        # Map results back to paths
        # zip(paths_with_channels, embeddings) creates tuples of ((path, channel), vector)
        results = {tuple(pc): vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
        return results, []
    except Exception as e:
        _log_worker_crash(e, "inference")
        return {}, [(p, f"Infer: {e}") for p, _ in paths_with_channels]


def worker_get_single_vector(image_path_str: str) -> np.ndarray | None:
    """Runs a single image inference (used for Sample Search)."""
    global _WORKER_ENGINE, _WORKER_PREPROCESSOR
    if not _WORKER_ENGINE or not _WORKER_PREPROCESSOR:
        return None
    try:
        # Local import to avoid circular dependency
        from app.domain.data_models import AnalysisItem

        items = [AnalysisItem(path=Path(image_path_str), analysis_type="Composite")]
        images, _, _ = _read_and_process_batch_for_ai(items, _WORKER_ENGINE.input_size, {})

        if images:
            px = _WORKER_PREPROCESSOR(images=images, return_tensors="np").pixel_values
            if _WORKER_ENGINE.is_fp16:
                px = px.astype(np.float16)

            io = _WORKER_ENGINE.visual_session.io_binding()
            io.bind_cpu_input("pixel_values", px)
            io.bind_output("image_embeds")

            _WORKER_ENGINE.visual_session.run_with_iobinding(io)
            return normalize_vectors_numpy(io.get_outputs()[0].numpy()).flatten()
    except Exception:
        pass
    return None


def worker_get_text_vector(text: str) -> np.ndarray | None:
    """Runs text inference (used for Text Search)."""
    global _WORKER_ENGINE
    if _WORKER_ENGINE:
        return _WORKER_ENGINE.get_text_features(text)
    return None


def _log_worker_crash(e: Exception, context: str):
    """Logs worker errors to a file since stderr might be swallowed."""
    pid = os.getpid()
    tid = threading.get_ident()
    log_file = APP_DATA_DIR / "crash_logs" / f"worker_error_{pid}_{tid}_{int(time.time())}.txt"
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"Context: {context}\nError: {e}\n{traceback.format_exc()}")
    logger.error(f"Worker Error in {context}: {e}")
