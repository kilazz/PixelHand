# app/workflow/pipeline.py
"""
AI Pipeline Manager.
"""

import gc
import logging
import queue
import threading
from concurrent.futures import wait
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject

# AI Manager functions are still used directly as they operate on
# module-level globals specific to the worker process context.
from app.ai import manager as ai_manager
from app.shared.constants import (
    DB_WRITE_BATCH_SIZE,
    FP16_MODEL_SUFFIX,
    GC_COLLECT_INTERVAL_ITEMS,
)

if TYPE_CHECKING:
    from app.domain.config import ScanConfig
    from app.domain.data_models import ScanState
    from app.infrastructure.container import ServiceContainer
    from app.shared.signal_bus import SignalBus
    from app.workflow.stages import ScanContext

logger = logging.getLogger("PixelHand.workflow.pipeline")


class PipelineManager(QObject):
    """
    Manages the flow of data: Disk -> Preprocessing -> Inference -> Database.
    Implements a Producer-Consumer pattern with bounded queues to manage memory.
    """

    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: "SignalBus",
        stop_event: threading.Event,
        services: "ServiceContainer",  # Injected Dependencies
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.stop_event = stop_event
        self.services = services

        # Thread-safe queues
        # Tensor queue is limited to prevent Preprocessor from flooding RAM with heavy tensors
        self.tensor_queue = queue.Queue(maxsize=4)
        # Results queue has no limit to ensure Inference never blocks waiting for DB writing
        self.results_queue = queue.Queue()

    def run(self, context: "ScanContext") -> tuple[bool, list[str]]:
        """
        Executes the pipeline. Returns (success_bool, list_of_skipped_files).
        """
        items_to_process = context.items_to_process
        if not items_to_process:
            return True, []

        try:
            # 1. Initialize AI Workers
            # We initialize the global state in this process/thread context
            # so that worker functions can access the model efficiently.
            worker_config = {
                "model_name": self.config.ai.model_name,
                "device": self.config.ai.device,
                "threads_per_worker": self.config.perf.num_workers,
                "models_dir": self.services.models_dir,
            }

            ai_manager.init_worker(worker_config)
            ai_manager.init_preprocessor_worker(worker_config)

            # 2. Start Background Threads

            # Producer: Reads images, creates tensors (Heavy CPU/IO)
            monitor_thread = threading.Thread(
                target=self._producer_preprocessing,
                args=(items_to_process,),
                daemon=True,
                name="ProducerThread",
            )

            # Worker: Runs Inference (Heavy CPU/GPU)
            inference_thread = threading.Thread(target=self._worker_inference, daemon=True, name="InferenceThread")

            monitor_thread.start()
            inference_thread.start()

            # 3. Run Consumer Logic (Blocking Main Thread)
            # Collects results and writes to DB
            total_items = len(items_to_process)
            all_skipped = self._consumer_collect_results(context, total_items)

            # 4. Cleanup
            monitor_thread.join()
            inference_thread.join()

            return True, all_skipped

        except Exception as e:
            logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]

    def _producer_preprocessing(self, items):
        """
        [Thread 1] Iterates over items, submits preprocessing tasks to the
        unified TaskManager, and pushes results to tensor_queue.
        """
        # Get input size from the loaded model
        input_size = ai_manager.get_model_input_size()
        batch_size = min(self.config.perf.batch_size, 128)

        # Determine DataType based on model name suffix
        is_fp16 = FP16_MODEL_SUFFIX in self.config.ai.model_name.lower()
        dtype = np.float16 if is_fp16 else np.float32

        simple_config = {"ignore_solid_channels": self.config.hashing.ignore_solid_channels}
        futures = []

        try:
            for i in range(0, len(items), batch_size):
                if self.stop_event.is_set():
                    break

                batch = items[i : i + batch_size]

                # Submit task to the unified CPU pool.
                # The worker will put the result into output_queue (self.tensor_queue).
                future = self.services.task_manager.submit_cpu_bound(
                    ai_manager.worker_preprocess_threaded,
                    items=batch,
                    input_size=input_size,
                    dtype=dtype,
                    simple_config=simple_config,
                    output_queue=self.tensor_queue,
                )
                futures.append(future)

        except Exception as e:
            logger.error(f"Producer thread crashed: {e}")

        # Wait for all preprocessing tasks to finish filling the queue
        # before sending the termination signal.
        if futures:
            wait(futures)

        # Signal End of Stream
        self.tensor_queue.put(None)

    def _worker_inference(self):
        """
        [Thread 2] Consumes batches from tensor_queue, runs ONNX inference,
        and pushes embeddings to results_queue.
        """
        while True:
            try:
                # Blocks until tensor is available
                batch_data = self.tensor_queue.get()
            except queue.Empty:
                continue

            # End of Stream check
            if batch_data is None:
                self.results_queue.put(None)
                self.tensor_queue.task_done()
                break

            pixel_values, paths_with_channels, skipped_tuples = batch_data

            if pixel_values is not None:
                # Run Inference using the ai_manager wrapper
                results, inf_skipped = ai_manager.run_inference_direct(pixel_values, paths_with_channels)

                # Immediately free heavy tensor memory
                del pixel_values

                self.results_queue.put((results, skipped_tuples + inf_skipped))
            else:
                # Pass through preprocessing errors
                self.results_queue.put(({}, skipped_tuples))

            self.tensor_queue.task_done()

    def _consumer_collect_results(self, context: "ScanContext", total_items: int) -> list[str]:
        """
        [Main Thread] Consumes inference results, batches them, and writes
        to LanceDB via the injected DatabaseService. Updates UI progress.
        """
        db_buffer = []
        all_skipped = []
        processed_count = 0

        # Track unique paths to report progress accurately
        unique_paths_processed = set()
        unique_paths_total = len({item.path for item in context.items_to_process})

        while True:
            if self.stop_event.is_set():
                break

            try:
                item = self.results_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.results_queue.task_done()
                break

            results_map, skipped_batch = item

            # Handle Skipped Files
            for path_str, _ in skipped_batch:
                all_skipped.append(path_str)
                unique_paths_processed.add(path_str)

            # Handle Successful Vectors
            for (path_str, channel), vector in results_map.items():
                unique_paths_processed.add(path_str)

                # Convert vector to list for JSON/LanceDB compatibility
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

                # Prepare record for DB
                db_buffer.append(
                    {
                        "path": path_str,
                        "channel": channel,
                        "vector": vector_list,
                    }
                )

            processed_count += len(results_map) + len(skipped_batch)

            # Flush to DB if buffer is full
            if len(db_buffer) >= DB_WRITE_BATCH_SIZE:
                self._enrich_and_flush_buffer(db_buffer, context)
                db_buffer.clear()

            # Update UI Progress
            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            # Periodic GC to prevent RAM creep
            if processed_count % GC_COLLECT_INTERVAL_ITEMS == 0:
                gc.collect()

        # Flush remaining items
        if db_buffer:
            self._enrich_and_flush_buffer(db_buffer, context)

        gc.collect()
        return all_skipped

    def _enrich_and_flush_buffer(self, db_buffer: list, context: "ScanContext"):
        """
        Enriches the raw vector data with metadata from context.all_image_fps
        before sending to the DatabaseService.
        """
        enriched_batch = []

        for item in db_buffer:
            path_str = item["path"]
            channel = item["channel"]
            vector = item["vector"]

            # Lookup metadata
            # Path keys in all_image_fps are Path objects
            path_obj = context.all_image_fps.get(item.get("path_obj") or Path(path_str))
            if not path_obj:
                # Try string match as fallback
                for p, fp in context.all_image_fps.items():
                    if str(p) == path_str:
                        path_obj = fp
                        break

            if path_obj:
                # Convert ImageFingerprint to dict and inject vector
                record = path_obj.to_lancedb_dict(channel=channel)
                record["vector"] = vector
                record["channel"] = channel
                enriched_batch.append(record)
            else:
                # Fallback if metadata missing (should be rare)
                item["id"] = ""
                enriched_batch.append(item)

        # Use injected DB Service
        self.services.db_service.add_batch(enriched_batch)
