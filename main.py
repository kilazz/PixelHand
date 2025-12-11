# main.py
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
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))

# Enable OpenEXR support for OpenCV if used internally by libraries
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from PySide6.QtWidgets import QApplication, QMessageBox

from app.shared.constants import CRASH_LOG_DIR
from app.shared.logger import setup_logging
from app.shared.signal_bus import APP_SIGNAL_BUS
from app.ui.main_window import App

IS_DEBUG_MODE = "--debug" in sys.argv


def log_global_crash(exc_type, exc_value, exc_traceback):
    """A global exception hook to catch and log any unhandled exceptions."""
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_message = f"--- CRITICAL UNHANDLED ERROR ---\n{tb_info}"

    with contextlib.suppress(Exception):
        logging.getLogger("PixelHand.main").critical(error_message)

    try:
        CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Use UTC for consistent log filenames
        ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = CRASH_LOG_DIR / f"crash_report_{ts}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_message)

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
    """Initializes and runs the Qt application."""
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            ctypes.windll.ole32.CoInitialize()

    sys.excepthook = log_global_crash

    app = QApplication(sys.argv)

    setup_logging(APP_SIGNAL_BUS, force_debug=IS_DEBUG_MODE)

    app_logger = logging.getLogger("PixelHand.main")
    app_logger.info("Starting PixelHand application...")

    main_window = App()
    main_window.show()
    app_logger.info("Main window displayed.")

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    faulthandler.enable()

    try:
        run_application()
    except Exception:
        log_global_crash(*sys.exc_info())
