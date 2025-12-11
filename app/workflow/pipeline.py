# app/workflow/pipeline.py
"""
Contains the PipelineManager, which orchestrates the scanning pipeline.
This manager runs multi-threaded image preprocessing, ONNX model inference,
and vector database writing (LanceDB).
"""

import contextlib
import copy
import gc
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from PySide6.QtCore import QObject

from app.ai import manager as ai_manager  # Updated import
from app.domain.data_models import ScanConfig, ScanState
from app.infrastructure.cache import CacheManager
from app.shared.constants import FP16_MODEL_SUFFIX, LANCEDB_AVAILABLE, POLARS_AVAILABLE
from app.shared.signal_bus import SignalBus

if TYPE_CHECKING:
    from app.workflow.stages import ScanContext

if POLARS_AVAILABLE:
    import polars as pl

app_logger = logging.getLogger("PixelHand.workflow.pipeline")


class PipelineManager(QObject):
    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        vector_db_writer: Any,
        table_name: str,
        stop_event: "threading.Event",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.vector_db_writer = vector_db_writer
        self.table_name = table_name
        self.stop_event = stop_event
        self._internal_stop_event = threading.Event()

        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWriter")
        self.preproc_executor = ThreadPoolExecutor(
            max_workers=self.config.perf.num_workers, thread_name_prefix="Preprocessor"
        )
        self.tensor_queue = queue.Queue(maxsize=4)
        self.results_queue = queue.Queue()

    def run(self, context: "ScanContext") -> tuple[bool, list[str]]:
        items_to_process = context.items_to_process
        if not items_to_process:
            return True, []

        cache = CacheManager(self.config.folder_path, self.config.model_name, lancedb_table=self.vector_db_writer)
        all_skipped = []

        try:
            # Init AI workers
            ai_manager.init_worker(
                {
                    "model_name": self.config.model_name,
                    "device": self.config.device,
                    "threads_per_worker": self.config.perf.num_workers,
                }
            )
            ai_manager.init_preprocessor_worker({"model_name": self.config.model_name})

            input_size = ai_manager.get_model_input_size()
            is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
            dtype = np.float16 if is_fp16 else np.float32

            inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceThread")
            inference_thread.start()

            monitor_thread = threading.Thread(
                target=self._monitor_preprocessing,
                args=(items_to_process, input_size, dtype),
                daemon=True,
                name="MonitorThread",
            )
            monitor_thread.start()

            all_skipped = self._collect_results(cache, context)

            inference_thread.join(timeout=1.0)
            monitor_thread.join(timeout=1.0)

            return True, all_skipped

        except Exception as e:
            app_logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]
        finally:
            self._cleanup(cache)

    def _monitor_preprocessing(self, items, input_size, dtype):
        batch_size = min(self.config.perf.batch_size, 128)
        simple_config = {"ignore_solid_channels": self.config.ignore_solid_channels}
        max_pending_tasks = self.config.perf.num_workers * 2
        semaphore = threading.Semaphore(max_pending_tasks)

        def task_done_callback(_):
            semaphore.release()

        try:
            for i in range(0, len(items), batch_size):
                if self.stop_event.is_set() or self._internal_stop_event.is_set():
                    break

                semaphore.acquire()
                batch = items[i : i + batch_size]
                future = self.preproc_executor.submit(
                    ai_manager.worker_preprocess_threaded,
                    items=batch,
                    input_size=input_size,
                    dtype=dtype,
                    simple_config=simple_config,
                    output_queue=self.tensor_queue,
                )
                future.add_done_callback(task_done_callback)

                if i % (batch_size * 10) == 0:
                    time.sleep(0.01)

        except Exception as e:
            app_logger.error(f"Preprocessing monitor crashed: {e}")

        self.preproc_executor.shutdown(wait=True)
        self.tensor_queue.put(None)

    def _inference_loop(self):
        while True:
            try:
                item = self.tensor_queue.get()
            except queue.Empty:
                continue

            if item is None:
                self.results_queue.put(None)
                self.tensor_queue.task_done()
                break

            pixel_values, paths_with_channels, skipped_tuples = item

            if pixel_values is not None:
                results, inf_skipped = ai_manager.run_inference_direct(pixel_values, paths_with_channels)
                del pixel_values
                gc.collect()
                self.results_queue.put((results, skipped_tuples + inf_skipped))
            else:
                self.results_queue.put(({}, skipped_tuples))

            self.tensor_queue.task_done()

    def _collect_results(self, cache: CacheManager, context: "ScanContext") -> list[str]:
        fps_to_cache_buffer = []
        db_buffer = []
        all_skipped = []
        unique_paths_processed = set()
        unique_paths_total = len({item.path for item in context.items_to_process})
        gc_trigger_counter = 0
        WRITE_THRESHOLD = 512

        while True:
            if self.stop_event.is_set():
                break

            try:
                result_batch = self.results_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if result_batch is None:
                self.results_queue.task_done()
                break

            batch_data, batch_skipped = result_batch
            self._handle_batch_results(
                batch_data,
                batch_skipped,
                fps_to_cache_buffer,
                db_buffer,
                context,
                unique_paths_processed,
            )

            all_skipped.extend([str(p) for p, _ in batch_skipped])

            if len(db_buffer) >= WRITE_THRESHOLD:
                self._add_to_lancedb(db_buffer)
                db_buffer.clear()

            if len(fps_to_cache_buffer) >= WRITE_THRESHOLD:
                cache.put_many(fps_to_cache_buffer)
                fps_to_cache_buffer.clear()

            gc_trigger_counter += len(batch_data) + len(batch_skipped)
            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            if gc_trigger_counter >= 500:
                gc.collect()
                gc_trigger_counter = 0

        if db_buffer:
            self._add_to_lancedb(db_buffer)
        if fps_to_cache_buffer:
            cache.put_many(fps_to_cache_buffer)
        gc.collect()
        return all_skipped

    def _handle_batch_results(self, batch_results, skipped, fps_buffer, db_buffer, context, unique_paths):
        for (path_str, channel), vector in batch_results.items():
            path_key = Path(path_str)
            if p_obj := context.all_image_fps.get(path_key):
                unique_paths.add(path_key)
                fp_orig = p_obj
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

                db_buffer.append({"path": str(fp_orig.path), "channel": channel, "vector": vector_list})

                fp_copy = copy.copy(fp_orig)
                fp_copy.hashes = vector
                fp_copy.channel = channel
                fps_buffer.append(fp_copy)

        for path_str, _ in skipped:
            unique_paths.add(Path(path_str))

    def _add_to_lancedb(self, data_dicts: list[dict]):
        if not data_dicts or not LANCEDB_AVAILABLE:
            return
        self.db_executor.submit(self._write_lancedb_task, data_dicts[:])

    def _write_lancedb_task(self, data_dicts: list[dict]):
        if not data_dicts:
            return
        try:
            data = [
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, d["path"] + (d["channel"] or ""))),
                    "vector": d["vector"],
                    "path": d["path"],
                    "channel": d["channel"],
                }
                for d in data_dicts
            ]

            arrow_table = pl.DataFrame(data).to_arrow() if POLARS_AVAILABLE else pa.Table.from_pylist(data)

            self.vector_db_writer.add(data=arrow_table)
            del data, arrow_table
            gc.collect()
        except Exception as e:
            app_logger.error(f"LanceDB batch write failed: {e}")

    def _cleanup(self, cache):
        self._internal_stop_event.set()
        while not self.tensor_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.tensor_queue.get_nowait()
        self.db_executor.shutdown(wait=True)
        cache.close()
