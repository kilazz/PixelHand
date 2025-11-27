# app/core/pipeline.py
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

from app.cache import POLARS_AVAILABLE, CacheManager
from app.constants import FP16_MODEL_SUFFIX, LANCEDB_AVAILABLE
from app.services.signal_bus import SignalBus

from . import worker

if TYPE_CHECKING:
    from app.core.scan_stages import ScanContext
    from app.data_models import ScanConfig, ScanState


if LANCEDB_AVAILABLE:
    pass

if POLARS_AVAILABLE:
    import polars as pl

app_logger = logging.getLogger("PixelHand.pipeline")


class PipelineManager(QObject):
    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        # Accepts LanceDB Table directly
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

        self.is_lancedb_mode = LANCEDB_AVAILABLE
        self.stop_event = stop_event

        self._internal_stop_event = threading.Event()

        # Executors
        # Single thread for DB writes to avoid locking issues
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWriter")

        # Workers for image resizing and preprocessing
        self.preproc_executor = ThreadPoolExecutor(
            max_workers=self.config.perf.num_workers, thread_name_prefix="Preprocessor"
        )

        # Queues:
        # Reduced queue size to strictly limit RAM usage.
        # Limits pending batches between Preprocessing -> Inference.
        self.tensor_queue = queue.Queue(maxsize=4)
        self.results_queue = queue.Queue()

    def run(self, context: "ScanContext") -> tuple[bool, list[str]]:
        items_to_process = context.items_to_process
        if not items_to_process:
            return True, []

        unique_files_count = len({item.path for item in items_to_process})
        app_logger.info(
            f"Starting Pipeline. Items: {len(items_to_process)} "
            f"({unique_files_count} unique files). Device: {self.config.device}."
        )

        # CacheManager works with the LanceDB table
        cache = CacheManager(self.config.folder_path, self.config.model_name, lancedb_table=self.vector_db_writer)
        all_skipped = []

        try:
            # 1. Init Global Workers
            # Pass num_workers so the worker can calculate optimal CPU thread allocation per ONNX session
            worker.init_worker(
                {
                    "model_name": self.config.model_name,
                    "device": self.config.device,
                    "threads_per_worker": self.config.perf.num_workers,
                }
            )
            worker.init_preprocessor_worker({"model_name": self.config.model_name})

            input_size = worker.get_model_input_size()
            is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
            dtype = np.float16 if is_fp16 else np.float32

            # 2. Start Inference Thread (Consumer)
            inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceThread")
            inference_thread.start()

            # 3. Start Preprocessing Monitor (Producer)
            monitor_thread = threading.Thread(
                target=self._monitor_preprocessing,
                args=(items_to_process, input_size, dtype),
                daemon=True,
                name="MonitorThread",
            )
            monitor_thread.start()

            # 4. Collect Results (Main Thread)
            all_skipped = self._collect_results(cache, context)

            # Clean shutdown
            inference_thread.join(timeout=1.0)
            monitor_thread.join(timeout=1.0)

            return True, all_skipped

        except Exception as e:
            app_logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]
        finally:
            self._cleanup(cache)

    def _monitor_preprocessing(self, items, input_size, dtype):
        """
        Submits tasks to the thread pool and waits for them to finish.
        Uses a Semaphore to implement Backpressure, preventing memory overflow.
        """
        # Cap batch size to prevent massive RAM usage if user set it too high
        batch_size = min(self.config.perf.batch_size, 128)
        simple_config = {"ignore_solid_channels": self.config.ignore_solid_channels}

        # Semaphore limits the number of tasks pending in the executor queue.
        # 2 * workers is a reasonable buffer to keep CPUs busy without overloading RAM.
        max_pending_tasks = self.config.perf.num_workers * 2
        semaphore = threading.Semaphore(max_pending_tasks)

        def task_done_callback(_):
            """Release semaphore when a task is finished by the worker."""
            semaphore.release()

        try:
            for i in range(0, len(items), batch_size):
                if self.stop_event.is_set() or self._internal_stop_event.is_set():
                    break

                # Block here if too many tasks are pending
                semaphore.acquire()

                batch = items[i : i + batch_size]
                future = self.preproc_executor.submit(
                    worker.worker_preprocess_threaded,
                    items=batch,
                    input_size=input_size,
                    dtype=dtype,
                    simple_config=simple_config,
                    output_queue=self.tensor_queue,
                )

                # Attach callback to release semaphore
                future.add_done_callback(task_done_callback)

                # Small sleep to allow Python's GC to run in the main loop context
                if i % (batch_size * 10) == 0:
                    time.sleep(0.01)

        except Exception as e:
            app_logger.error(f"Preprocessing monitor crashed: {e}")

        # Wait for all submitted tasks to complete
        self.preproc_executor.shutdown(wait=True)
        # Send sentinel to stop inference loop
        self.tensor_queue.put(None)

    def _inference_loop(self):
        """Consumes tensors from queue, runs ONNX inference, produces embeddings."""
        while True:
            try:
                # Blocking get() allows this thread to sleep when queue is empty
                item = self.tensor_queue.get()
            except queue.Empty:
                continue

            if item is None:
                self.results_queue.put(None)
                self.tensor_queue.task_done()
                break

            pixel_values, paths_with_channels, skipped_tuples = item

            if pixel_values is not None:
                results, inf_skipped = worker.run_inference_direct(pixel_values, paths_with_channels)

                # Explicitly delete large input tensor and trigger GC
                del pixel_values
                gc.collect()

                self.results_queue.put((results, skipped_tuples + inf_skipped))
            else:
                self.results_queue.put(({}, skipped_tuples))

            self.tensor_queue.task_done()

    def _collect_results(self, cache: CacheManager, context: "ScanContext") -> list[str]:
        """
        Collects results from the queue and batches DB writes.
        """
        fps_to_cache_buffer = []  # Buffer for CacheManager (metadata update)
        db_buffer = []  # Buffer for LanceDB (vector insertion)
        all_skipped = []

        unique_paths_processed = set()
        unique_paths_total = len({item.path for item in context.items_to_process})

        gc_trigger_counter = 0

        # Reduced thresholds to flush data to disk faster and free RAM
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

            count_in_batch = len(batch_data) + len(batch_skipped)
            gc_trigger_counter += count_in_batch

            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            # Frequent GC to prevent memory creep
            if gc_trigger_counter >= 500:
                gc.collect()
                gc_trigger_counter = 0

        # Final flush
        if db_buffer:
            self._add_to_lancedb(db_buffer)

        if fps_to_cache_buffer:
            cache.put_many(fps_to_cache_buffer)

        gc.collect()

        return all_skipped

    def _handle_batch_results(
        self,
        batch_results: dict,
        skipped_items: list,
        fps_to_cache_buffer: list,
        db_buffer: list,
        context: "ScanContext",
        unique_paths_processed: set,
    ):
        """
        Matches raw vectors back to File Paths/Channels using context lookups.
        """
        for (path_str, channel), vector in batch_results.items():
            # Lookup key in context
            # Assuming context.all_image_fps keys match path_str format or are Path objects
            path_key = None
            p_obj = context.all_image_fps.get(Path(path_str))

            # If direct lookup failed (e.g. string vs Path mismatch), iterate (slower fallback)
            if not p_obj:
                path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)
            else:
                path_key = Path(path_str)

            if path_key:
                unique_paths_processed.add(path_key)
                fp_orig = context.all_image_fps[path_key]

                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

                # Data for LanceDB insertion
                db_buffer.append({"path": str(fp_orig.path), "channel": channel, "vector": vector_list})

                # Data for Cache/Metadata update
                fp_copy = copy.copy(fp_orig)
                fp_copy.hashes = vector
                fp_copy.channel = channel
                fps_to_cache_buffer.append(fp_copy)
            else:
                app_logger.warning(f"Path mismatch during collection: {path_str}")

        for path_str, _ in skipped_items:
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)
            if path_key:
                unique_paths_processed.add(path_key)

    def _add_to_lancedb(self, data_dicts: list[dict]):
        """Submits a batch write task to the single-threaded DB Executor."""
        if not data_dicts or not LANCEDB_AVAILABLE:
            return
        # Copy list to ensure thread safety when clearing buffer in main loop
        data_copy = data_dicts[:]
        self.db_executor.submit(self._write_lancedb_task, data_copy)

    def _write_lancedb_task(self, data_dicts: list[dict]):
        """Running inside the DB executor thread."""
        if not data_dicts:
            return

        try:
            data_for_lancedb = [
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, d["path"] + (d["channel"] or ""))),
                    "vector": d["vector"],
                    "path": d["path"],
                    "channel": d["channel"],
                }
                for d in data_dicts
            ]

            if POLARS_AVAILABLE:
                polars_df = pl.DataFrame(data_for_lancedb)
                arrow_table = polars_df.to_arrow()
            else:
                arrow_table = pa.Table.from_pylist(data_for_lancedb)

            self.vector_db_writer.add(data=arrow_table)

            # Clean up explicitly within the thread
            del data_for_lancedb
            del arrow_table
            gc.collect()

        except Exception as e:
            app_logger.error(f"LanceDB batch write failed: {e}")

    def _cleanup(self, cache):
        self._internal_stop_event.set()

        # Drain queues to unblock threads
        while not self.tensor_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.tensor_queue.get_nowait()

        self.db_executor.shutdown(wait=True)
        cache.close()
        app_logger.info("Pipeline resources cleaned up.")
