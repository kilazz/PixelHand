# main.py
"""
PixelHand Application Entry Point.
"""

import contextlib
import ctypes
import faulthandler
import logging
import multiprocessing
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# --- SCRIPT PATH SETUP ---
# Ensure the application root is in sys.path to allow imports from 'app.'
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))

# --- ENVIRONMENT CONFIGURATION ---
# Fix for OpenCV IO issues with EXR files if used by dependencies
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from PySide6.QtWidgets import QApplication, QMessageBox

# Import Refactored Components
from app.infrastructure.container import ServiceContainer
from app.shared.constants import APP_DATA_DIR, CRASH_LOG_DIR
from app.shared.logger import setup_logging
from app.shared.signal_bus import APP_SIGNAL_BUS
from app.ui.main_window import App

IS_DEBUG_MODE = "--debug" in sys.argv


def log_global_crash(exc_type, exc_value, exc_traceback):
    """
    A global exception hook to catch and log any unhandled exceptions
    that would otherwise crash the application silently or to console.
    """
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_message = f"--- CRITICAL UNHANDLED ERROR ---\n{tb_info}"

    # Try to log to internal logger
    with contextlib.suppress(Exception):
        logging.getLogger("PixelHand.main").critical(error_message)

    try:
        CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Use UTC for consistent log filenames
        ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = CRASH_LOG_DIR / f"crash_report_{ts}.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_message)

        # Show blocking dialog to user if GUI is alive
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Critical Error",
                f"An unhandled error occurred.\nDetails have been saved to:\n{log_file.resolve()}",
            )
    except Exception as e:
        print(f"Failed to save crash report or show dialog: {e}", file=sys.stderr)
    finally:
        sys.exit(1)


def run_application():
    """
    Initializes and runs the Qt application with Dependency Injection.
    """
    # Windows-specific initialization for COM/OLE
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            ctypes.windll.ole32.CoInitialize()

    # Hook global exceptions
    sys.excepthook = log_global_crash

    app = QApplication(sys.argv)

    # 1. Setup Logging
    setup_logging(APP_SIGNAL_BUS, force_debug=IS_DEBUG_MODE)
    logger = logging.getLogger("PixelHand.main")
    logger.info("Starting PixelHand application...")

    services = None
    exit_code = 0

    try:
        # 2. Initialize Dependency Injection Container
        # This spins up the TaskManager (Threads), Database Connection, etc.
        logger.info("Initializing core services...")

        services = ServiceContainer.create(
            app_data_dir=APP_DATA_DIR,
            headless=False,  # We are in GUI mode, use QThread where possible
            max_workers=4,  # Default threads for heavy lifting
        )

        # 3. Inject Services into Main Window
        # The Main Window will distribute these services to Controllers
        main_window = App(services)
        main_window.show()
        logger.info("Main window displayed.")

        # 4. Run Event Loop
        exit_code = app.exec()

    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{e}")
        exit_code = 1

    finally:
        # 5. Graceful Shutdown
        # Ensures threads are stopped and DB connection closed cleanly
        if services:
            logger.info("Shutting down services...")
            services.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    # Multiprocessing guards for Windows/PyInstaller
    multiprocessing.freeze_support()

    # Ensure 'spawn' context for ML libraries safety
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    # Enable fault handler for low-level segfaults (C/C++ crashes)
    faulthandler.enable()

    try:
        run_application()
    except Exception:
        # Catch any errors that happen before sys.excepthook is set
        log_global_crash(*sys.exc_info())
