# app/workflow/scanner.py
"""
Scanner Orchestrator.
"""

import logging
import sys
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

# --- Conditional Environment Setup ---

# Detect CLI mode to avoid QObject dependency even if PySide6 is installed.
# This prevents the class from looking for a QApplication event loop that doesn't exist.
IS_CLI_MODE = "cli.py" in sys.argv[0]

# Conditional import: Use Qt classes ONLY if not in CLI mode and library is available
QT_AVAILABLE = False
if not IS_CLI_MODE:
    try:
        from PySide6.QtCore import QObject, QThread, Slot

        QT_AVAILABLE = True
    except ImportError:
        pass

# If not available or disabled via CLI mode, create dummy classes so the file parses
if not QT_AVAILABLE:

    class QObject:
        pass

    class QThread:
        pass

    def Slot(*args):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger("PixelHand.workflow.scanner")


class ScannerCore(QObject):
    """
    The worker object that runs the business logic in a background thread.
    It is agnostic to the UI, communicating purely through the EventBus in the ServiceContainer.
    """

    def __init__(self, config: ScanConfig, state: ScanState, services: ServiceContainer):
        super().__init__()
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
            # We currently keep this global setup for ImageLoader compatibility,
            # but ideally, this would also move to the container.
            setup_caches(self.config)

            # 2. Initialize Database Service via Container
            # This ensures we have a valid connection and schema before starting.
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
                # Strategies receive the full ServiceContainer to access DB, AI, etc.
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

            # Note: We do NOT close db_service here.
            # The UI needs it to fetch results/thumbnails after the scan finishes.
            # The ServiceContainer.shutdown() method handles closing on app exit.

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


class ScannerController(QObject):
    """
    Bridge between UI and the Background Thread.
    Holds the ServiceContainer reference and manages the QThread lifecycle.

    This class is primarily used in GUI mode.
    """

    def __init__(self, services: ServiceContainer):
        super().__init__()
        self.services = services  # Dependency Injection

        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.scan_state = ScanState()

        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None

        # Listen for global start requests via the EventBus
        self.services.event_bus.scan_requested.connect(self.start_scan)
        self.services.event_bus.scan_cancellation_requested.connect(self.cancel_scan)

    def is_running(self) -> bool:
        return self.scan_thread is not None and self.scan_thread.isRunning()

    @Slot(object)
    def start_scan(self, config: ScanConfig):
        """
        Initializes and starts the background scan thread.
        """
        if self.is_running():
            logger.warning("Scan start requested but already running.")
            return

        if not QT_AVAILABLE:
            logger.error("ScannerController requires PySide6. Use HeadlessRunner for CLI.")
            return

        self.config = config
        self.scan_state.reset()
        self.stop_event.clear()

        # Create QThread
        self.scan_thread = QThread()

        # Create Worker with Dependencies
        self.scanner_core = ScannerCore(config, self.scan_state, self.services)
        self.scanner_core.moveToThread(self.scan_thread)

        # Wire Signals to Thread Management
        self.services.event_bus.scan_finished.connect(self.scan_thread.quit)
        self.services.event_bus.scan_error.connect(self.scan_thread.quit)

        # Connect Thread Start to Worker Run
        self.scan_thread.started.connect(lambda: self.scanner_core.run(self.stop_event))
        self.scan_thread.finished.connect(self._on_scan_thread_finished)

        # Launch
        self.scan_thread.start()
        logger.info(f"Scan thread started: {config.scan_mode.name} on {config.folder_path}")

    @Slot()
    def cancel_scan(self):
        """Signals the worker to stop processing."""
        if self.is_running():
            self.services.event_bus.log_message.emit("Cancellation requested...", "warning")
            self.stop_event.set()
            # The thread will exit naturally when strategies check stop_event

    @Slot()
    def _on_scan_thread_finished(self):
        """Cleanup QObjects after thread execution."""
        logger.debug("Scan thread finished. Cleaning up.")
        if self.scanner_core:
            self.scanner_core.deleteLater()
        if self.scan_thread:
            self.scan_thread.deleteLater()
        self.scanner_core = None
        self.scan_thread = None
