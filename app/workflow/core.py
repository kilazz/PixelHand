# app/workflow/core.py
"""
Core Scanner Logic (Headless).
Independent of UI frameworks.
"""

import logging
import threading
import time

from app.domain.config import ScanConfig
from app.domain.data_models import ScanMode, ScanState
from app.infrastructure.cache import setup_caches, teardown_caches
from app.infrastructure.container import ServiceContainer

# Strategies
from app.workflow.strategies import (
    FindDuplicatesStrategy,
    FolderComparisonStrategy,
    SearchStrategy,
    SingleFolderQCStrategy,
)

logger = logging.getLogger("PixelHand.workflow.core")


class ScannerCore:
    """
    The worker object that runs the business logic.
    It is a pure Python class, agnostic to the UI framework (Qt).
    It communicates purely through the EventBus in the ServiceContainer.
    """

    def __init__(self, config: ScanConfig, state: ScanState, services: ServiceContainer):
        self.config = config
        self.state = state
        self.services = services  # Injected Dependencies: DB, AI, EventBus

        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []

    def run(self, stop_event: threading.Event):
        """
        Main entry point for the scanner logic.
        """
        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()

        try:
            # 1. Setup lightweight caches (thumbnails)
            setup_caches(self.config)

            # 2. Initialize Database Service via Container
            if not self.services.db_service.initialize(self.config):
                self.services.event_bus.scan_error.emit("Failed to initialize Database Service.")
                return

            # 3. Select Strategy
            strategy_map = {
                ScanMode.DUPLICATES: FindDuplicatesStrategy,
                ScanMode.TEXT_SEARCH: SearchStrategy,
                ScanMode.SAMPLE_SEARCH: SearchStrategy,
                ScanMode.FOLDER_COMPARE: FolderComparisonStrategy,
                ScanMode.SINGLE_FOLDER_QC: SingleFolderQCStrategy,
            }

            strategy_class = strategy_map.get(self.config.scan_mode)

            if strategy_class:
                # Instantiate Strategy with Dependencies
                strategy = strategy_class(
                    config=self.config,
                    state=self.state,
                    signals=self.services.event_bus,  # Pass abstract bus protocol
                    scanner_core=self,
                    services=self.services,
                )

                strategy.execute(stop_event, start_time)
            else:
                msg = f"Unknown scan mode: {self.config.scan_mode}"
                logger.error(msg)
                self.services.event_bus.log_message.emit(msg, "error")
                self._finalize_scan(None, 0, None, 0, [])

        except Exception as e:
            if not stop_event.is_set():
                logger.critical(f"Critical scan error: {e}", exc_info=True)
                self.services.event_bus.scan_error.emit(f"Scan aborted: {e}")
        finally:
            # 4. Cleanup
            teardown_caches()

            total_duration = time.time() - start_time
            logger.info(f"Scan logic finished in {total_duration:.2f}s")

            # Ensure UI gets a finished signal even if interrupted/cancelled
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan(None, 0, None, total_duration, self.all_skipped_files)

    def _check_stop_or_empty(self, stop_event, collection, mode, payload, start_time) -> bool:
        """
        Helper for strategies to check early exit conditions (cancelled or empty).
        """
        duration = time.time() - start_time

        if stop_event.is_set():
            self.state.set_phase("Scan cancelled.", 0.0)
            self._finalize_scan(None, 0, None, duration, self.all_skipped_files)
            return True

        if not collection:
            self.state.set_phase("Finished! No items found.", 0.0)

            # Check if we have groups from a previous pass or payload
            groups = payload.get("groups_data", {}) if payload else {}
            num_found = sum(len(dups) for dups in groups.values()) if groups else 0

            self._finalize_scan(payload, num_found, mode, duration, self.all_skipped_files)
            return True

        return False

    def _finalize_scan(self, payload, num_found, mode, duration, skipped_files):
        """
        Emits the final results signal to the UI via the EventBus.
        """
        if not self.scan_has_finished:
            # Send results to UI/CLI
            self.services.event_bus.scan_finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True
