# app/ui/controllers.py
"""
UI Controllers.
"""

import logging
import os
from pathlib import Path

import send2trash
from PySide6.QtCore import QObject, Signal

from app.domain.data_models import FileOperation
from app.infrastructure.container import ServiceContainer

logger = logging.getLogger("PixelHand.controllers")


class ResultsController(QObject):
    """
    Manages the business logic for the Results Panel.

    Responsibilities:
    - Executing file operations (Delete, Hardlink, Reflink) in background threads.
    - Synchronizing the Database state when files are modified.
    - Emitting status updates to the View.
    """

    # Signals to update the UI
    operation_started = Signal(str)  # Operation Name (e.g., "Deleting...")
    operation_progress = Signal(str, int, int)  # Message, current, total
    operation_finished = Signal(list)  # List of successfully affected paths
    status_message = Signal(str, str)  # Message, Level (info, error, success)
    error_occurred = Signal(str, str)  # Title, Message

    def __init__(self, services: ServiceContainer):
        super().__init__()
        self.services = services
        self._is_busy = False

    def request_deletion(self, paths: list[Path]):
        """
        Starts a background task to move files to the trash.
        """
        if self._is_busy:
            return

        if not paths:
            return

        self._is_busy = True
        self.operation_started.emit("Deleting...")
        logger.info(f"Controller: Requesting deletion of {len(paths)} files.")

        # Use the TaskManager from the ServiceContainer
        self.services.task_manager.start_background_task(
            func=self._worker_delete_files,
            on_finish=self._on_operation_complete,
            on_error=self._on_operation_error,
            paths=paths,
        )

    def request_linking(self, link_map: dict[Path, Path], operation: FileOperation):
        """
        Starts a background task to replace files with links (Hardlink or Reflink).

        Args:
            link_map: Dictionary mapping {Target_File_To_Replace: Source_File_To_Point_To}
            operation: FileOperation.HARDLINKING or FileOperation.REFLINKING
        """
        if self._is_busy:
            return

        if not link_map:
            return

        method = "reflink" if operation == FileOperation.REFLINKING else "hardlink"
        op_name = "Reflinking..." if method == "reflink" else "Hardlinking..."

        self._is_busy = True
        self.operation_started.emit(op_name)
        logger.info(f"Controller: Requesting {method} for {len(link_map)} files.")

        self.services.task_manager.start_background_task(
            func=self._worker_link_files,
            on_finish=self._on_operation_complete,
            on_error=self._on_operation_error,
            link_map=link_map,
            method=method,
        )

    # --- Background Workers (Run in ThreadPool) ---

    def _worker_delete_files(self, paths: list[Path]) -> dict:
        """
        Actual deletion logic running in a background thread.
        Returns a summary dict to be passed to on_finish.
        """
        affected = []
        failed = []
        total = len(paths)

        for _, path in enumerate(paths, 1):
            # We can't emit Qt signals directly from a non-Qt thread safely if using standard threads,
            # but TaskManager ensures thread safety or we use signals if wrapper supports it.
            # ideally, TaskManager handles this, but for simplicity we return the result at the end.
            try:
                if path.exists():
                    send2trash.send2trash(str(path))
                    affected.append(path)
                else:
                    # File already gone, count as handled
                    affected.append(path)
            except Exception as e:
                failed.append((path, str(e)))

        return {
            "type": "delete",
            "affected": affected,
            "failed": failed,
            "total": total,
        }

    def _worker_link_files(self, link_map: dict[Path, Path], method: str) -> dict:
        """
        Actual linking logic running in a background thread.
        """
        affected = []
        failed = []
        total = len(link_map)

        # Check if OS supports reflink if requested
        can_reflink = hasattr(os, "reflink") if method == "reflink" else False

        for target, source in link_map.items():
            try:
                if not source.exists():
                    raise FileNotFoundError(f"Source missing: {source}")

                # Windows cross-drive check
                if os.name == "nt" and target.drive.lower() != source.drive.lower():
                    raise OSError("Cross-drive links not supported.")

                # Atomic-ish replacement: Unlink target, then link source to target
                if target.exists():
                    os.remove(target)

                if method == "reflink" and can_reflink:
                    os.reflink(source, target)
                else:
                    # Fallback to hardlink or standard hardlink
                    os.link(source, target)

                affected.append(target)
            except Exception as e:
                failed.append((target, str(e)))

        return {
            "type": "link",
            "method": method,
            "affected": affected,
            "failed": failed,
            "total": total,
        }

    # --- Callbacks (Run on Main Thread) ---

    def _on_operation_complete(self, result: dict):
        """
        Called when the background worker finishes.
        Updates the DB and notifies the View.
        """
        affected_paths = result["affected"]
        failed_items = result["failed"]
        op_type = result["type"]

        # 1. Update Database (Remove processed files from index)
        if affected_paths:
            try:
                # We need to convert Paths to strings for the DB service
                paths_str = [str(p) for p in affected_paths]
                # Assuming DBService has a delete_paths method (added in refactoring)
                if hasattr(self.services.db_service, "delete_paths"):
                    self.services.db_service.delete_paths(paths_str)
            except Exception as e:
                logger.error(f"Failed to update DB after file op: {e}")

        # 2. Log Results
        if failed_items:
            logger.warning(f"Operation partially failed. {len(failed_items)} errors.")
            for p, err in failed_items:
                logger.debug(f"Failed {p}: {err}")
            self.status_message.emit(f"Completed with {len(failed_items)} errors.", "warning")
        else:
            action = "Moved" if op_type == "delete" else "Linked"
            msg = f"Successfully {action} {len(affected_paths)} files."
            self.status_message.emit(msg, "success")

        self._is_busy = False

        # 3. Notify View to update UI (remove rows)
        self.operation_finished.emit(affected_paths)

    def _on_operation_error(self, exception: Exception):
        """
        Called if the worker crashes unexpectedly.
        """
        logger.error(f"Critical error in file operation: {exception}", exc_info=True)
        self._is_busy = False
        self.error_occurred.emit("Operation Failed", str(exception))
        self.operation_finished.emit([])
