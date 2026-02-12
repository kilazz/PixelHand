# app/workflow/scanner.py
"""
Scanner UI Controller (Qt).
Bridges the GUI with the pure Python ScannerCore.
"""

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.domain.config import ScanConfig
from app.domain.data_models import ScanState
from app.infrastructure.container import ServiceContainer
from app.workflow.core import ScannerCore

logger = logging.getLogger("PixelHand.workflow.scanner")


class ScannerQtWorker(QObject):
    """
    A QObject wrapper that allows the pure-Python ScannerCore
    to run inside a QThread using the moveToThread mechanism.
    """

    finished = Signal()

    def __init__(self, core: ScannerCore, stop_event: threading.Event):
        super().__init__()
        self.core = core
        self.stop_event = stop_event

    @Slot()
    def run(self):
        """Slot to start the core logic."""
        self.core.run(self.stop_event)
        self.finished.emit()


class ScannerController(QObject):
    """
    Bridge between UI and the Background Thread.
    Manages the QThread lifecycle and wraps the core logic.
    """

    def __init__(self, services: ServiceContainer):
        super().__init__()
        self.services = services  # Dependency Injection

        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.worker: ScannerQtWorker | None = None
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

        self.config = config
        self.scan_state.reset()
        self.stop_event.clear()

        # Create QThread
        self.scan_thread = QThread()

        # Create pure logic Core
        self.scanner_core = ScannerCore(config, self.scan_state, self.services)

        # Create Qt Worker Wrapper
        self.worker = ScannerQtWorker(self.scanner_core, self.stop_event)
        self.worker.moveToThread(self.scan_thread)

        # Wire Signals
        # 1. When the scan logic finishes (via EventBus signal), stop the thread
        self.services.event_bus.scan_finished.connect(self.scan_thread.quit)
        self.services.event_bus.scan_error.connect(self.scan_thread.quit)

        # 2. When thread starts, run the worker
        self.scan_thread.started.connect(self.worker.run)

        # 3. Cleanup when thread finishes
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

    @Slot()
    def _on_scan_thread_finished(self):
        """Cleanup QObjects after thread execution."""
        logger.debug("Scan thread finished. Cleaning up.")
        if self.worker:
            self.worker.deleteLater()
        if self.scan_thread:
            self.scan_thread.deleteLater()

        # Reset references
        self.worker = None
        self.scanner_core = None
        self.scan_thread = None
