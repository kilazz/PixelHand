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

# Pre-load libraries to avoid import lock contention in threads
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

from app.ai.preprocessing import ImageBatchPreprocessor
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
        self,
        model_name: str,
        models_dir: Path,
        device: str = "CPUExecutionProvider",
        threads_per_worker: int = 1,
        model_dim: int = 512,
    ):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")

        self.model_name = model_name  # Store name for state validation
        self.model_dir = models_dir / model_name
        self.device = device
        self.model_dim = model_dim
        self.threads_per_worker = threads_per_worker

        # Load HuggingFace Processor (Tokenizer + Image Config)
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to load processor from {self.model_dir}: {e}") from e

        self.is_fp16 = FP16_MODEL_SUFFIX in model_name.lower()

        # Extract input size from config
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})

        if "height" in size_cfg:
            self.input_size = (size_cfg["width"], size_cfg["height"])
        elif "shortest_edge" in size_cfg:
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

        # MEMORY OPTIMIZATION: Disable Memory Arena
        # This prevents ORT from holding onto large memory blocks for future reuse.
        # It slightly increases allocation overhead but drastically reduces peak RAM usage
        # in scenarios with variable workloads.
        opts.enable_cpu_mem_arena = False

        # Provider Logic
        available_providers = ort.get_available_providers()
        if device not in available_providers:
            logger.warning(f"Requested provider '{device}' not found. Fallback to CPU.")
            device = "CPUExecutionProvider"

        providers = [device]
        if device != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        # FP16 specifics for DirectML
        if device == "DmlExecutionProvider" and self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
            opts.enable_float16_for_dml = True

        # Threading Configuration (CPU only)
        if device == "CPUExecutionProvider":
            total_cores = multiprocessing.cpu_count()
            if threads_per_worker <= 1:
                estimated_workers = 4
                threads_to_use = max(1, total_cores // estimated_workers)
            else:
                threads_to_use = threads_per_worker

            opts.intra_op_num_threads = threads_to_use
            opts.inter_op_num_threads = 1
            opts.execution_mode = (
                ort.ExecutionMode.ORT_PARALLEL if threads_to_use > 1 else ort.ExecutionMode.ORT_SEQUENTIAL
            )

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

        self.device = device
        logger.info(f"ONNX Loaded. Device: {device}, FP16: {self.is_fp16}, Input: {self.input_size}")

    def encode_visual_batch(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Runs the visual model on a batch of pixel values.
        Includes output validation.
        """
        if not self.visual_session:
            raise RuntimeError("Visual session not initialized")

        onnx_inputs = {"pixel_values": pixel_values}
        embeddings = self.visual_session.run(["image_embeds"], onnx_inputs)[0]

        # Validation: Check if output dimension matches expected model dimension
        if embeddings.ndim != 2 or embeddings.shape[1] != self.model_dim:
            msg = (
                f"Model shape mismatch! Expected (N, {self.model_dim}), got {embeddings.shape}. "
                "The model file appears to be corrupted or incompatible."
            )
            logger.critical(msg)
            # Raise error to stop processing this batch, but do NOT delete files.
            raise RuntimeError(msg)

        return normalize_vectors_numpy(embeddings)

    def get_text_features(self, text: str) -> np.ndarray:
        """Runs the text model on a string query."""
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
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.current_engine: InferenceEngine | None = None
        self.current_model_name: str | None = None
        self._lock = threading.RLock()

    def ensure_model_loaded(self, config: dict[str, Any]):
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
                    threads_per_worker=1,
                    model_dim=config.get("model_dim", 512),
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
    """Initializes global model state in worker process."""
    global _WORKER_ENGINE, _WORKER_PREPROCESSOR
    try:
        models_dir = config.get("models_dir")
        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        elif not models_dir:
            models_dir = APP_DATA_DIR / "models"

        _WORKER_ENGINE = InferenceEngine(
            model_name=config["model_name"],
            models_dir=models_dir,
            device=config.get("device", "CPUExecutionProvider"),
            threads_per_worker=config.get("threads_per_worker", 1),
            model_dim=config.get("model_dim", 512),
        )
        _WORKER_PREPROCESSOR = _WORKER_ENGINE.processor
        logger.info(f"Worker process {os.getpid()} initialized.")
    except Exception as e:
        _log_worker_crash(e, "init_worker")
        raise e


def init_preprocessor_worker(config: dict[str, Any]):
    init_worker(config)


def get_model_input_size() -> tuple[int, int]:
    if _WORKER_ENGINE:
        return _WORKER_ENGINE.input_size
    return (224, 224)


def worker_preprocess_task(
    items: list["AnalysisItem"], input_size: tuple[int, int], dtype: np.dtype, simple_config: dict
):
    """
    Worker task: loads images via Preprocessor, creates numpy tensors.
    Returns (pixel_values, success_paths, skipped_items).
    This function is picklable and suitable for ProcessPoolExecutor.
    """
    global _WORKER_PREPROCESSOR, _WORKER_ENGINE

    req_config = simple_config.get("init_config")

    # State Validation:
    # Ensure worker has initialized, AND that it is initialized with the CORRECT model.
    # ProcessPool workers persist across scans, so we must detect model switches.
    needs_init = False
    if not _WORKER_PREPROCESSOR or (_WORKER_ENGINE and req_config and _WORKER_ENGINE.model_name != req_config["model_name"]):
        needs_init = True

    if needs_init and req_config:
        try:
            init_worker(req_config)
        except Exception as e:
            return None, [], [(str(i.path), f"Worker Init Failed: {e}") for i in items]

    if not _WORKER_PREPROCESSOR:
        return None, [], [(str(i.path), "Model not initialized in worker") for i in items]

    # Use extracted preprocessing logic
    images, success_paths, skipped = ImageBatchPreprocessor.prepare_batch(
        items, input_size, ignore_solid_channels=simple_config.get("ignore_solid_channels", True)
    )

    if not images:
        return None, [], skipped

    try:
        # Request PyTorch tensors first ("pt"), then convert to NumPy.
        batch_dict = _WORKER_PREPROCESSOR(images=images, return_tensors="pt")

        # Detach from graph (if any), move to CPU, convert to numpy, then cast to target dtype
        pixel_values = batch_dict.pixel_values.detach().cpu().numpy().astype(dtype)

        # Cleanup PIL images to free RAM
        del images
        gc.collect()

        return pixel_values, success_paths, skipped
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess")
        return None, [], [(str(i.path), f"Preproc: {e}") for i in items]


def run_inference_direct(pixel_values: np.ndarray, paths_with_channels: list) -> tuple[dict, list]:
    """
    Runs inference on pre-calculated pixel_values using the worker's engine.
    """
    global _WORKER_ENGINE
    if not _WORKER_ENGINE:
        return {}, [(p, "Engine missing") for p, _ in paths_with_channels]

    try:
        # Use class method for encapsulated logic
        embeddings = _WORKER_ENGINE.encode_visual_batch(pixel_values)
        del pixel_values

        # Map results back to paths
        results = {tuple(pc): vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
        return results, []
    except Exception as e:
        _log_worker_crash(e, "inference")
        return {}, [(p, f"Infer Error: {e}") for p, _ in paths_with_channels]


def _log_worker_crash(e: Exception, context: str):
    """Logs worker errors to a file."""
    pid = os.getpid()
    tid = threading.get_ident()
    log_file = APP_DATA_DIR / "crash_logs" / f"worker_error_{pid}_{tid}_{int(time.time())}.txt"
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"Context: {context}\nError: {e}\n{traceback.format_exc()}")
    logger.error(f"Worker Error in {context}: {e}")
