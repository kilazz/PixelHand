# app/ui/background_tasks.py
"""
Contains QRunnable tasks for performing background operations without freezing the GUI.
"""

import gc
import inspect
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import send2trash
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QSemaphore, Signal

from app.ai.optimizer import optimize_model_post_export
from app.domain.data_models import FileOperation
from app.imaging.image_io import get_image_metadata, load_image
from app.imaging.processing import is_vfx_transparent_texture
from app.infrastructure.cache import get_thumbnail_cache_key, thumbnail_cache
from app.shared.constants import (
    DEEP_LEARNING_AVAILABLE,
    LANCEDB_AVAILABLE,
    MAX_CONCURRENT_IMAGE_LOADS,
    MODELS_DIR,
    QuantizationMode,
    TonemapMode,
)
from app.shared.utils import get_model_folder_name

if TYPE_CHECKING:
    from app.infrastructure.db_service import DatabaseService

app_logger = logging.getLogger("PixelHand.ui.tasks")


class ModelConverter(QRunnable):
    """
    A task to download, convert, and cache a HuggingFace model to ONNX format.
    """

    class Signals(QObject):
        finished = Signal(bool, str)
        log = Signal(str, str)

    def __init__(
        self,
        hf_model_name: str,
        onnx_name_base: str,
        quant_mode: QuantizationMode,
        model_info: dict,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.hf_model_name = hf_model_name
        self.onnx_name_base = onnx_name_base
        self.quant_mode = quant_mode
        self.model_info = model_info
        self.signals = self.Signals()

    def run(self):
        if not DEEP_LEARNING_AVAILABLE:
            self.signals.finished.emit(False, "Deep learning libraries not found.")
            return

        try:
            import torch
            from PIL import Image

            from app.ai.adapters import get_model_adapter

            adapter = get_model_adapter(self.hf_model_name)

            original_progress_bar_setting = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            target_dir = self._setup_directories(MODELS_DIR, adapter)

            if (target_dir / "visual.onnx").exists():
                self.signals.log.emit(f"Model '{target_dir.name}' already exists.", "info")
                self.signals.finished.emit(True, "Model already exists.")
                return

            self.signals.log.emit(f"Downloading model '{self.hf_model_name}'...", "info")

            ProcessorClass = adapter.get_processor_class()
            ModelClass = adapter.get_model_class()

            processor = ProcessorClass.from_pretrained(self.hf_model_name)
            model = ModelClass.from_pretrained(self.hf_model_name)
            model.eval()  # Ensure base model is in inference mode

            # Do NOT convert the whole model to half() here globally.
            # We will handle FP16 conversion specifically for the vision component
            # during the export phase to avoid LayerNorm errors in the text encoder.
            export_fp16 = self.quant_mode == QuantizationMode.FP16

            visual_out_path = target_dir / "visual.onnx"
            text_out_path = target_dir / "text.onnx"

            self.signals.log.emit("Exporting to ONNX (Dynamo Backend)...", "info")

            # Perform export
            self._export_model(model, processor, target_dir, torch, Image, adapter, export_fp16)

            # AGGRESSIVE MEMORY CLEANUP
            # PyTorch's trace graph is huge. We must free it immediately.
            self.signals.log.emit("Cleaning up conversion memory...", "info")
            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.quant_mode == QuantizationMode.INT8:
                self.signals.log.emit("Optimizing and Quantizing to INT8...", "info")
                optimize_model_post_export(visual_out_path, text_out_path, self.quant_mode)
                # Cleanup again after optimization tools run
                gc.collect()

            self.signals.finished.emit(True, "Model prepared successfully.")

        except Exception as e:
            msg = f"Failed to prepare model: {e}"
            app_logger.error(msg, exc_info=True)
            self.signals.finished.emit(False, str(e))
        finally:
            if original_progress_bar_setting is None:
                if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                    del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_progress_bar_setting
            # Final safety GC
            gc.collect()

    def _setup_directories(self, models_dir: Path, adapter) -> Path:
        target_dir_name = get_model_folder_name(self.onnx_name_base, self.quant_mode)
        target_dir = models_dir / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        ProcessorClass = adapter.get_processor_class()
        processor = ProcessorClass.from_pretrained(self.hf_model_name)
        processor.save_pretrained(target_dir)
        return target_dir

    def _export_model(self, model, processor, target_dir, torch, Image, adapter, is_fp16):
        """
        Exports the model using TorchDynamo (dynamo=True).
        This captures the full graph (including pooling/projection) accurately.
        """
        # Suppress Dynamo verbosity/errors that don't block export
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        from torch.export import Dim

        # Opset 18 is required for better compatibility with Dynamo optimizations (avoids InlinePass errors)
        opset_version = 18

        # --- 1. Export Vision Model ---
        vision_wrapper = adapter.get_vision_wrapper(model, torch)
        vision_wrapper.eval() # Ensure wrapper is in eval mode

        input_size = adapter.get_input_size(processor)

        # Use batch size > 1 to help Dynamo identify the batch dimension as dynamic
        dummy_input = processor(images=[Image.new("RGB", input_size), Image.new("RGB", input_size)], return_tensors="pt")

        # Use simple batch for tracing input
        pixel_values = dummy_input["pixel_values"]

        if is_fp16:
            self.signals.log.emit("Converting Vision component to FP16...", "info")
            vision_wrapper.half()
            pixel_values = pixel_values.half()
        else:
            vision_wrapper.float()
            pixel_values = pixel_values.float()

        # Use dynamic_shapes instead of dynamic_axes for Dynamo
        batch_dim = Dim("batch_size", min=1)
        dynamic_shapes = {"pixel_values": {0: batch_dim}}

        torch.onnx.export(
            vision_wrapper,
            (pixel_values,),
            str(target_dir / "visual.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_shapes=dynamic_shapes,
            opset_version=opset_version,
            dynamo=True,  # Force Dynamo backend
        )

        # Free vision wrapper immediately
        del vision_wrapper
        del pixel_values
        del dummy_input
        gc.collect()

        # --- 2. Export Text Model (if applicable) ---
        if adapter.has_text_model():
            self.signals.log.emit("Exporting Text component (FP32)...", "info")
            text_wrapper = adapter.get_text_wrapper(model, torch)
            text_wrapper.eval() # Ensure wrapper is in eval mode

            # Keep Text Model FP32 for stability with LayerNorms
            text_wrapper.float()

            # Create text dummy input with batch > 1
            dummy_text_input = processor.tokenizer(
                text=["a test query", "another query"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=processor.tokenizer.model_max_length,
                return_attention_mask=True,
            )

            # Prepare inputs based on signature (some models might not use attention_mask in wrapper)
            sig = inspect.signature(text_wrapper.forward)

            # Define dims for text
            seq_dim = Dim("sequence", min=1)

            if "attention_mask" in sig.parameters:
                onnx_inputs = (dummy_text_input["input_ids"], dummy_text_input["attention_mask"])
                input_names = ["input_ids", "attention_mask"]
                text_dynamic_shapes = {
                    "input_ids": {0: batch_dim, 1: seq_dim},
                    "attention_mask": {0: batch_dim, 1: seq_dim}
                }
            else:
                onnx_inputs = (dummy_text_input["input_ids"],)
                input_names = ["input_ids"]
                text_dynamic_shapes = {
                    "input_ids": {0: batch_dim, 1: seq_dim}
                }

            torch.onnx.export(
                text_wrapper,
                onnx_inputs,
                str(target_dir / "text.onnx"),
                input_names=input_names,
                output_names=["text_embeds"],
                dynamic_shapes=text_dynamic_shapes,
                opset_version=opset_version,
                dynamo=True,  # Force Dynamo backend
            )

            # Free text wrapper
            del text_wrapper
            del dummy_text_input
            gc.collect()


class ImageLoader(QRunnable):
    """
    A cancellable task to load an image in a background thread.
    Uses a semaphore to limit concurrent heavy I/O operations (OOM protection).
    """

    _semaphore = QSemaphore(MAX_CONCURRENT_IMAGE_LOADS)

    class Signals(QObject):
        result = Signal(str, object)
        error = Signal(str, str)

    def __init__(
        self,
        path_str: str,
        mtime: float,
        target_size: int | None,
        tonemap_mode: str = TonemapMode.ENABLED.value,
        use_cache: bool = True,
        channel_to_load: str | None = None,
        ui_key: str | None = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.path_str = path_str
        self.mtime = mtime
        self.target_size = target_size
        self.tonemap_mode = tonemap_mode
        self.use_cache = use_cache
        self._is_cancelled = False
        self.channel_to_load = channel_to_load
        self.ui_key = ui_key or path_str
        self.signals = self.Signals()

    def cancel(self):
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        try:
            if self.is_cancelled:
                return

            pil_img = None
            cache_key = get_thumbnail_cache_key(
                self.path_str,
                self.mtime,
                self.target_size,
                self.tonemap_mode,
                self.channel_to_load,
            )

            # 1. Check Cache (Fast, does not require semaphore)
            if self.use_cache and self.target_size and not self.is_cancelled:
                cached_data = thumbnail_cache.get(cache_key)
                if cached_data:
                    try:
                        pil_img = Image.open(io.BytesIO(cached_data))
                        pil_img.load()
                        self.signals.result.emit(self.ui_key, pil_img)
                        return
                    except Exception:
                        pil_img = None

            if self.is_cancelled:
                return

            # 2. Heavy Load from Disk (Protected by Semaphore)
            self._semaphore.acquire()
            try:
                if self.is_cancelled:
                    return

                if pil_img is None:
                    path_obj = Path(self.path_str)
                    if not path_obj.exists():
                        self.signals.error.emit(self.ui_key, "File not found")
                        return

                    metadata = get_image_metadata(path_obj)
                    shrink = 1
                    if metadata and self.target_size:
                        width, height = metadata["resolution"]
                        if width > self.target_size * 1.5 or height > self.target_size * 1.5:
                            shrink_w = width / (self.target_size * 1.5)
                            shrink_h = height / (self.target_size * 1.5)
                            shrink = max(1, int(min(shrink_w, shrink_h)))
                            if shrink > 1:
                                shrink = 1 << (shrink - 1).bit_length()

                    if self.is_cancelled:
                        return

                    pil_img = load_image(self.path_str, tonemap_mode=self.tonemap_mode, shrink=shrink)

                    if self.is_cancelled:
                        return

                    if pil_img:
                        if self.channel_to_load:
                            pil_img = pil_img.convert("RGBA")
                            channels = pil_img.split()
                            idx = {"R": 0, "G": 1, "B": 2, "A": 3}.get(self.channel_to_load)
                            if idx is not None and idx < len(channels):
                                ch = channels[idx]
                                pil_img = Image.merge("RGB", (ch, ch, ch))
                        else:
                            pil_img = pil_img.convert("RGBA")

                        # VFX Fix: Detect if Alpha=0 but RGB has data
                        is_vfx = is_vfx_transparent_texture(pil_img)
                        pil_img.info["is_vfx"] = is_vfx

                        if self.target_size:
                            if is_vfx:
                                pil_img = pil_img.convert("RGB")
                                pil_img.info["is_vfx"] = True
                            pil_img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
                            if self.use_cache and not self.is_cancelled:
                                try:
                                    buffer = io.BytesIO()
                                    pil_img.save(buffer, "PNG")
                                    thumbnail_cache.put(cache_key, buffer.getvalue())
                                except Exception:
                                    pass

                if self.is_cancelled:
                    return

                if pil_img:
                    # Safety check if info was lost (e.g. during resize if not propagated)
                    if "is_vfx" not in pil_img.info and pil_img.mode == "RGBA":
                        pil_img.info["is_vfx"] = is_vfx_transparent_texture(pil_img)
                    self.signals.result.emit(self.ui_key, pil_img)
                else:
                    self.signals.error.emit(self.ui_key, "Image loader returned None")

            finally:
                self._semaphore.release()

        except Exception as e:
            if not self.is_cancelled:
                self.signals.error.emit(self.ui_key, f"Loader error: {e}")


class LanceDBGroupFetcherTask(QRunnable):
    """
    Fetches children items for a specific group ID from the database.
    Requires dependency injection of db_service.
    """

    class Signals(QObject):
        finished = Signal(list, int)
        error = Signal(str)

    def __init__(self, group_id: int, db_service: "DatabaseService"):
        super().__init__()
        self.setAutoDelete(True)
        self.group_id = group_id
        self.db_service = db_service
        self.signals = self.Signals()

    def run(self):
        if not LANCEDB_AVAILABLE:
            self.signals.error.emit("LanceDB not available.")
            return

        try:
            # Delegate db access to the injected service
            rows = self.db_service.get_files_by_group(self.group_id)
            self.signals.finished.emit(rows, self.group_id)
        except Exception as e:
            app_logger.error(f"Background group fetch failed: {e}", exc_info=True)
            self.signals.error.emit(str(e))


class LanceDBBestPathFetcherTask(QRunnable):
    """
    Background task to fetch ONLY the 'best' file path for a given group ID.
    Used for Grid View thumbnails to avoid loading entire groups.
    """

    class Signals(QObject):
        finished = Signal(int, str)  # group_id, path

    def __init__(self, group_id: int, db_service: "DatabaseService"):
        super().__init__()
        self.setAutoDelete(True)
        self.group_id = group_id
        self.db_service = db_service
        self.signals = self.Signals()

    def run(self):
        try:
            # We use the db_service connection directly.
            if not self.db_service.db or "scan_results" not in self.db_service.db.table_names():
                return

            tbl = self.db_service.db.open_table("scan_results")
            # We only need the path of the best file
            res = tbl.search().where(f"group_id = {self.group_id} AND is_best = true").limit(1).to_list()

            if res:
                self.signals.finished.emit(self.group_id, str(res[0]["path"]))
            else:
                # Fallback: if no 'best' flagged (rare), take the first one
                res_fallback = tbl.search().where(f"group_id = {self.group_id}").limit(1).to_list()
                if res_fallback:
                    self.signals.finished.emit(self.group_id, str(res_fallback[0]["path"]))

        except Exception as e:
            app_logger.error(f"Best path fetch failed for group {self.group_id}: {e}")


class FileOperationTask(QRunnable):
    class Signals(QObject):
        finished = Signal(list, int, int)
        log = Signal(str, str)
        progress_updated = Signal(str, int, int)

    def __init__(
        self, operation: FileOperation, paths: list[Path] | None = None, link_map: dict[Path, Path] | None = None
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.operation = operation
        self.paths = paths or []
        self.link_map = link_map or {}
        self.signals = self.Signals()

    def run(self):
        if self.operation == FileOperation.DELETING:
            self._delete_worker(self.paths)
        elif self.operation in [FileOperation.HARDLINKING, FileOperation.REFLINKING]:
            method = "reflink" if self.operation == FileOperation.REFLINKING else "hardlink"
            self._link_worker(self.link_map, method)
        else:
            self.signals.log_message.emit(f"Unknown mode: {self.operation.name}", "error")

    def _delete_worker(self, paths: list[Path]):
        moved, failed = [], 0
        total = len(paths)
        for i, path in enumerate(paths, 1):
            self.signals.progress_updated.emit(f"Deleting: {path.name}", i, total)
            try:
                if path.exists():
                    send2trash.send2trash(str(path))
                    moved.append(path)
                else:
                    moved.append(path)
            except Exception:
                failed += 1
        self.signals.finished.emit(moved, len(moved), failed)

    def _link_worker(self, link_map: dict[Path, Path], method: str):
        replaced, failed, failed_list = 0, 0, []
        affected = list(link_map.keys())
        total = len(affected)
        can_reflink = hasattr(os, "reflink")

        for i, (link_path, source_path) in enumerate(link_map.items(), 1):
            self.signals.progress_updated.emit(f"Linking: {link_path.name}", i, total)
            try:
                if not (link_path.exists() and source_path.exists()):
                    raise FileNotFoundError(f"Source or dest missing for {link_path.name}")
                if os.name == "nt" and link_path.drive.lower() != source_path.drive.lower():
                    raise OSError("Cross-drive link not supported")
                os.remove(link_path)
                if method == "reflink" and can_reflink:
                    os.reflink(source_path, link_path)
                else:
                    os.link(source_path, link_path)
                replaced += 1
            except Exception as e:
                failed += 1
                failed_list.append(f"{link_path.name} ({type(e).__name__})")

        if failed_list:
            self.signals.log.emit(f"Failed to link: {', '.join(failed_list)}", "error")
        self.signals.finished.emit(affected, replaced, failed)
