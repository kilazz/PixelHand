# app/workflow/scanner.py
"""
Main orchestrator for the scanning process.
Manages the lifecycle of the scan thread, initializes the Database Service,
and selects the appropriate strategy to execute.
"""

import logging
import threading
import time

from PySide6.QtCore import QObject, QThread, Slot

from app.domain.data_models import ScanConfig, ScanMode, ScanState
from app.infrastructure.cache import setup_caches, teardown_caches
from app.infrastructure.db_service import DB_SERVICE
from app.shared.signal_bus import APP_SIGNAL_BUS

# Strategies
from app.workflow.strategies import (
    FindDuplicatesStrategy,
    FolderComparisonStrategy,
    SearchStrategy,
    SingleFolderQCStrategy,
)

app_logger = logging.getLogger("PixelHand.workflow.scanner")


class ScannerCore(QObject):
    """
    The main business logic orchestrator for the entire scanning process.
    It runs inside a QThread, preventing the GUI from freezing.
    """

    def __init__(self, config: ScanConfig, state: ScanState):
        super().__init__()
        self.config = config
        self.state = state
        self.vectors_table_name = "images"
        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []
        self.all_image_fps = {}

    def run(self, stop_event: threading.Event):
        """
        Main entry point for the scanner logic, executed in a separate thread.
        """
        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()
        self.all_image_fps.clear()

        try:
            # 1. Setup lightweight caches (thumbnails)
            setup_caches(self.config)

            # 2. Initialize the Singleton Database Service
            # This ensures we have a valid connection and schema before proceeding.
            if not DB_SERVICE.initialize(self.config):
                APP_SIGNAL_BUS.scan_error.emit("Failed to initialize Database Service.")
                return

            # 3. Select Strategy based on configuration
            strategy_map = {
                ScanMode.DUPLICATES: FindDuplicatesStrategy,
                ScanMode.TEXT_SEARCH: SearchStrategy,
                ScanMode.SAMPLE_SEARCH: SearchStrategy,
                ScanMode.FOLDER_COMPARE: FolderComparisonStrategy,
                ScanMode.SINGLE_FOLDER_QC: SingleFolderQCStrategy,
            }
            strategy_class = strategy_map.get(self.config.scan_mode)

            if strategy_class:
                # We pass 'self' (ScannerCore) to the strategy so it can call _finalize_scan
                # or access shared state if necessary.
                strategy = strategy_class(self.config, self.state, APP_SIGNAL_BUS, self)
                strategy.execute(stop_event, start_time)
            else:
                APP_SIGNAL_BUS.log_message.emit(f"Unknown scan mode: {self.config.scan_mode}", "error")
                self._finalize_scan(None, 0, None, 0, [])

        except Exception as e:
            if not stop_event.is_set():
                app_logger.error(f"Critical scan error: {e}", exc_info=True)
                APP_SIGNAL_BUS.scan_error.emit(f"Scan aborted due to critical error: {e}")
        finally:
            # 4. Cleanup Resources
            # NOTE: We DO NOT close DB_SERVICE here.
            # It must remain open so the UI can fetch group details/thumbnails
            # after the scan finishes. Closing happens in main_window.py closeEvent.

            teardown_caches()

            total_duration = time.time() - start_time
            app_logger.info("Scan process finished.")

            # Ensure the UI gets a finished signal even if interrupted
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan(None, 0, None, total_duration, self.all_skipped_files)

    def _check_stop_or_empty(self, stop_event, collection, mode, payload, start_time) -> bool:
        """
        Helper used by strategies to check if they should exit early (cancelled or no files found).
        """
        duration = time.time() - start_time
        if stop_event.is_set():
            self.state.set_phase("Scan cancelled.", 0.0)
            self._finalize_scan(None, 0, None, duration, self.all_skipped_files)
            return True
        if not collection:
            self.state.set_phase("Finished! No new images to process.", 0.0)
            num_found = (
                sum(len(dups) for dups in payload.get("groups_data", {}).values()) if payload.get("groups_data") else 0
            )
            self._finalize_scan(payload, num_found, mode, duration, self.all_skipped_files)
            return True
        return False

    def _finalize_scan(self, payload, num_found, mode, duration, skipped_files):
        """
        Emits the final signal to the UI to display results.
        """
        if not self.scan_has_finished:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "<1s"
            log_msg = "Scan cancelled." if not mode else f"Found {num_found} items in {time_str}."
            app_logger.info(log_msg)

            # Send results to UI
            APP_SIGNAL_BUS.scan_finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True


class ScannerController(QObject):
    """
    Manages the lifecycle of the scanner thread.
    Acts as the bridge between the UI thread and the worker thread.
    """

    def __init__(self):
        super().__init__()
        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.scan_state: ScanState = ScanState()
        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None

        # UI Signals
        APP_SIGNAL_BUS.scan_requested.connect(self.start_scan)
        APP_SIGNAL_BUS.scan_cancellation_requested.connect(self.cancel_scan)

    def is_running(self) -> bool:
        return self.scan_thread is not None and self.scan_thread.isRunning()

    @Slot(object)
    def start_scan(self, config: ScanConfig):
        """Prepares and starts the background scan thread."""
        if self.is_running():
            return

        self.config = config
        self.scan_state.reset()
        self.stop_event = threading.Event()

        self.scan_thread = QThread()
        self.scanner_core = ScannerCore(config, self.scan_state)
        self.scanner_core.moveToThread(self.scan_thread)

        # Wire up signals
        APP_SIGNAL_BUS.scan_finished.connect(self.scan_thread.quit)
        APP_SIGNAL_BUS.scan_error.connect(self.scan_thread.quit)

        # Start logic
        self.scan_thread.started.connect(lambda: self.scanner_core.run(self.stop_event))
        self.scan_thread.finished.connect(self._on_scan_thread_finished)

        self.scan_thread.start()
        app_logger.info("New scan thread started.")

    def cancel_scan(self):
        """Sets the stop event to signal the worker thread to exit gracefully."""
        if self.is_running():
            APP_SIGNAL_BUS.log_message.emit("Cancellation requested...", "warning")
            self.stop_event.set()
            # We do NOT force quit() here; we wait for the logic to see the flag
            # and clean up in the finally block.

    @Slot()
    def _on_scan_thread_finished(self):
        """Cleanup QObjects after thread execution."""
        if self.scanner_core:
            self.scanner_core.deleteLater()
        if self.scan_thread:
            self.scan_thread.deleteLater()
        self.scanner_core, self.scan_thread = None, None
