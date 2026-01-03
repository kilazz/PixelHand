# cli.py
"""
Command Line Interface (Headless Mode) for PixelHand.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# NOTE: We do NOT import PySide6 here.
# The ServiceContainer (headless=True) ensures we use pure Python events.
from app.domain.config import (
    AIConfig,
    HashingConfig,
    OutputConfig,
    PerformanceConfig,
    QCConfig,
    ScanConfig,
)
from app.domain.data_models import ScanMode, ScanState
from app.infrastructure.container import ServiceContainer
from app.shared.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    APP_DATA_DIR,
    SUPPORTED_MODELS,
    QuantizationMode,
)
from app.shared.utils import get_model_folder_name
from app.workflow.scanner import ScannerCore

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PixelHand.CLI")


class HeadlessRunner:
    """
    Orchestrates the scan process in a headless environment.
    """

    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()  # Signal to main loop that work is done
        self.exit_code = 0

        # 1. Initialize Dependency Injection Container
        logger.info("Initializing services...")
        self.services = ServiceContainer.create(
            app_data_dir=APP_DATA_DIR,
            headless=True,
            max_workers=self.args.workers,
        )

        # 2. Build Configuration
        self.config = self._build_config()
        self.state = ScanState()

        # 3. Initialize Scanner Logic
        self.scanner = ScannerCore(self.config, self.state, self.services)

    def _build_config(self) -> ScanConfig:
        """
        Maps command-line arguments to the structured ScanConfig domain object.
        """
        folder_path = Path(self.args.folder).resolve()
        if not folder_path.exists():
            logger.error(f"Target folder does not exist: {folder_path}")
            sys.exit(1)

        # Determine Mode
        if self.args.qc:
            mode = ScanMode.SINGLE_FOLDER_QC
        elif self.args.text_search:
            mode = ScanMode.TEXT_SEARCH
        else:
            mode = ScanMode.DUPLICATES

        # --- AI Configuration Logic ---
        # 1. Get Model Info from Constants
        model_key = self.args.model
        # Fallback if key is invalid
        if model_key not in SUPPORTED_MODELS:
            logger.warning(f"Model '{model_key}' not found. Using default: 'Fastest (OpenCLIP ViT-B/32)'")
            model_key = "Fastest (OpenCLIP ViT-B/32)"

        model_info = SUPPORTED_MODELS[model_key]

        # 2. Determine Quantization Mode
        quant_mode = QuantizationMode.INT8 if self.args.fast else QuantizationMode.FP16

        # 3. Resolve ONNX Folder Name (Fixes "Repo id" error)
        onnx_name = get_model_folder_name(model_info["onnx_name"], quant_mode)

        # AI Settings
        ai_config = AIConfig(
            model_name=onnx_name,
            model_dim=model_info["dim"],
            device="CPUExecutionProvider",
            use_ai=not self.args.no_ai,
            quantization_mode=quant_mode.name,
            supports_text_search=model_info.get("supports_text_search", True),
            supports_image_search=model_info.get("supports_image_search", True),
        )

        # QC Settings
        qc_config = QCConfig(
            check_alpha=True,
            check_normal_maps=True,
            check_npot=self.args.strict,
            check_mipmaps=self.args.strict,
            check_solid_color=False,
        )

        # Hashing Settings
        hashing_config = HashingConfig(
            find_exact=True,
            find_simple=True,
            find_perceptual=True,
            dhash_threshold=self.args.threshold,
        )

        # Performance Settings
        perf_config = PerformanceConfig(
            num_workers=self.args.workers,
            batch_size=32,
        )

        # Output Settings
        output_config = OutputConfig(
            save_visuals=self.args.save_visuals,
            max_visuals=50,
        )

        return ScanConfig(
            folder_path=folder_path,
            scan_mode=mode,
            ai=ai_config,
            qc=qc_config,
            hashing=hashing_config,
            perf=perf_config,
            output=output_config,
            search_query=self.args.text_search,
            excluded_folders=self.args.exclude.split(",") if self.args.exclude else [],
            # Important: Must provide extensions (Fixes "Found 0 files")
            selected_extensions=ALL_SUPPORTED_EXTENSIONS,
        )

    def start(self):
        """Starts the scanning thread and connects signals."""
        logger.info("=" * 60)
        logger.info(f"Starting Scan: {self.config.scan_mode.name}")
        logger.info(f"Target:        {self.config.folder_path}")
        logger.info(f"AI Enabled:    {self.config.ai.use_ai}")
        logger.info(f"Model Dir:     {self.config.ai.model_name}")  # Log resolved name
        logger.info("=" * 60)

        # Connect Core Signals
        self.services.event_bus.scan_finished.connect(self._on_finished)
        self.services.event_bus.scan_error.connect(self._on_error)

        # Run logic in a background thread
        self.scan_thread = threading.Thread(
            target=self.scanner.run,
            args=(self.stop_event,),
            daemon=True,
        )
        self.scan_thread.start()

        # Main Loop: Wait for finish signal or interrupt
        try:
            while not self.finished_event.is_set():
                # Safety Check: If thread died unexpectedly
                if not self.scan_thread.is_alive():
                    time.sleep(2.0)
                    if not self.finished_event.is_set():
                        logger.warning("Scan thread finished but did not signal completion. Forcing exit.")
                        self.exit_code = 0
                        break

                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.warning("\nInterrupt received. Stopping...")
            self.stop_event.set()
            self.scan_thread.join(timeout=5.0)
            self.exit_code = 130
        finally:
            self._shutdown()
            sys.exit(self.exit_code)

    def _on_finished(self, payload, num_found, mode, duration, skipped):
        """Called when scanning completes successfully."""
        logger.info("-" * 60)
        logger.info(f"SCAN COMPLETE in {duration:.2f}s")
        logger.info(f"Items Found:   {num_found}")
        logger.info(f"Files Skipped: {len(skipped)}")

        if skipped:
            logger.warning(f"First 5 skipped files: {skipped[:5]}")

        if self.args.fail_on_error and len(skipped) > 0:
            logger.error("FAILURE: Errors occurred during processing.")
            self.exit_code = 1
        elif self.args.fail_on_found and num_found > 0:
            logger.error(f"FAILURE: Found {num_found} issues/duplicates.")
            self.exit_code = 1
        else:
            logger.info("SUCCESS: Scan passed.")
            self.exit_code = 0

        self.finished_event.set()

    def _on_error(self, message):
        """Called when a critical error occurs."""
        logger.critical(f"FATAL ERROR: {message}")
        self.exit_code = 1
        self.finished_event.set()

    def _shutdown(self):
        """Clean up services."""
        logger.info("Shutting down services...")
        self.services.shutdown()


def main():
    parser = argparse.ArgumentParser(description="PixelHand Headless Scanner")

    # Target
    parser.add_argument("folder", help="Path to the folder to scan")

    # Modes
    parser.add_argument("--qc", action="store_true", help="Run QC checks")
    parser.add_argument("--text-search", type=str, default=None, help="Run Semantic Search")

    # Behavior Flags
    parser.add_argument("--no-ai", action="store_true", help="Disable AI")
    parser.add_argument("--fast", action="store_true", help="Use INT8 quantization")
    parser.add_argument("--strict", action="store_true", help="Enable strict QC")
    parser.add_argument("--save-visuals", action="store_true", help="Generate comparison images")

    # Configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Fastest (OpenCLIP ViT-B/32)",
        help="Model key name",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--exclude", type=str, default="")

    # CI/CD Controls
    parser.add_argument("--fail-on-found", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")

    args = parser.parse_args()

    # Create Runner and Start
    runner = HeadlessRunner(args)
    runner.start()


if __name__ == "__main__":
    main()
