# app/workflow/pipeline.py
"""
AI Pipeline Manager.
"""

import gc
import logging
import queue
import threading
from concurrent.futures import as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject

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
    Uses TaskManager for thread orchestration.
    """

    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: "SignalBus",
        stop_event: threading.Event,
        services: "ServiceContainer",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.stop_event = stop_event
        self.services = services

        # Tensor queue is limited to prevent Preprocessor from flooding RAM
        self.tensor_queue = queue.Queue(maxsize=4)
        # Results queue has no limit to ensure Inference never blocks
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
            worker_config = {
                "model_name": self.config.ai.model_name,
                "model_dim": self.config.ai.model_dim,
                "device": self.config.ai.device,
                "threads_per_worker": self.config.perf.num_workers,
                "models_dir": self.services.models_dir,
            }

            ai_manager.init_worker(worker_config)
            ai_manager.init_preprocessor_worker(worker_config)

            # 2. Start Background Threads via Task Manager
            monitor_thread = self.services.task_manager.start_thread(
                target=self._producer_preprocessing,
                name="ProducerThread",
                args=(items_to_process, worker_config),
            )

            inference_thread = self.services.task_manager.start_thread(
                target=self._worker_inference,
                name="InferenceThread",
            )

            # 3. Run Consumer Logic (Blocking Main Thread)
            total_items = len(items_to_process)
            all_skipped = self._consumer_collect_results(context, total_items)

            # 4. Cleanup
            monitor_thread.join()
            inference_thread.join()

            return True, all_skipped

        except Exception as e:
            logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]

    def _producer_preprocessing(self, items, worker_config):
        """
        [Thread 1] Iterates over items, submits preprocessing tasks to the
        unified TaskManager (ProcessPool), collects results, and pushes to tensor_queue.
        """
        input_size = ai_manager.get_model_input_size()
        batch_size = min(self.config.perf.batch_size, 128)
        is_fp16 = FP16_MODEL_SUFFIX in self.config.ai.model_name.lower()
        dtype = np.float16 if is_fp16 else np.float32

        process_config = {
            "ignore_solid_channels": self.config.hashing.ignore_solid_channels,
            "init_config": {
                "model_name": worker_config["model_name"],
                "model_dim": worker_config["model_dim"],
                "device": worker_config["device"],
                "threads_per_worker": 1,
                "models_dir": str(worker_config["models_dir"]),
            },
        }

        futures = []

        try:
            for i in range(0, len(items), batch_size):
                if self.stop_event.is_set():
                    break

                batch = items[i : i + batch_size]
                future = self.services.task_manager.submit_cpu_bound(
                    ai_manager.worker_preprocess_task,
                    items=batch,
                    input_size=input_size,
                    dtype=dtype,
                    simple_config=process_config,
                )
                futures.append(future)

            for future in as_completed(futures):
                if self.stop_event.is_set():
                    break
                try:
                    result = future.result()
                    if result:
                        self.tensor_queue.put(result)
                except Exception as e:
                    logger.error(f"Preprocessing task failed: {e}")

        except Exception as e:
            logger.error(f"Producer thread crashed: {e}")

        self.tensor_queue.put(None)

    def _worker_inference(self):
        """
        [Thread 2] Consumes batches from tensor_queue, runs ONNX inference,
        and pushes embeddings to results_queue.
        """
        while True:
            try:
                batch_data = self.tensor_queue.get()
            except queue.Empty:
                continue

            if batch_data is None:
                self.results_queue.put(None)
                self.tensor_queue.task_done()
                break

            pixel_values, paths_with_channels, skipped_tuples = batch_data

            if pixel_values is not None:
                results, inf_skipped = ai_manager.run_inference_direct(pixel_values, paths_with_channels)
                del pixel_values
                self.results_queue.put((results, skipped_tuples + inf_skipped))
            else:
                self.results_queue.put(({}, skipped_tuples))

            self.tensor_queue.task_done()

    def _consumer_collect_results(self, context: "ScanContext", total_items: int) -> list[str]:
        """
        [Main Thread] Consumes inference results, batches them, and writes
        to LanceDB via the injected DatabaseService.
        """
        db_buffer = []
        all_skipped = []
        processed_count = 0
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

            for path_str, _ in skipped_batch:
                all_skipped.append(path_str)
                unique_paths_processed.add(path_str)

            for (path_str, channel), vector in results_map.items():
                unique_paths_processed.add(path_str)
                if vector is None or len(vector) == 0:
                    continue

                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                db_buffer.append(
                    {
                        "path": path_str,
                        "channel": channel,
                        "vector": vector_list,
                    }
                )

            processed_count += len(results_map) + len(skipped_batch)

            if len(db_buffer) >= DB_WRITE_BATCH_SIZE:
                self._enrich_and_flush_buffer(db_buffer, context)
                db_buffer.clear()

            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            if processed_count % GC_COLLECT_INTERVAL_ITEMS == 0:
                gc.collect()

        if db_buffer:
            self._enrich_and_flush_buffer(db_buffer, context)

        gc.collect()
        return all_skipped

    def _enrich_and_flush_buffer(self, db_buffer: list, context: "ScanContext"):
        enriched_batch = []
        for item in db_buffer:
            path_str = item["path"]
            channel = item["channel"]
            vector = item["vector"]

            path_obj = context.all_image_fps.get(item.get("path_obj") or Path(path_str))
            if not path_obj:
                for p, fp in context.all_image_fps.items():
                    if str(p) == path_str:
                        path_obj = fp
                        break

            if path_obj:
                record = path_obj.to_lancedb_dict(channel=channel)
                record["vector"] = vector
                record["channel"] = channel
                enriched_batch.append(record)
            else:
                item["id"] = ""
                enriched_batch.append(item)

        self.services.db_service.add_batch(enriched_batch)
