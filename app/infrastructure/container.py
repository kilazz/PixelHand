# app/infrastructure/container.py
"""
Service Container / Dependency Injection Root.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

# Import Core Services
from app.ai.manager import ModelManager
from app.infrastructure.cache import (
    AbstractThumbnailCache,
    DummyThumbnailCache,
    LanceDBThumbnailCache,
)
from app.infrastructure.db_service import DatabaseService
from app.infrastructure.task_manager import TaskManager
from app.shared.constants import LANCEDB_AVAILABLE
from app.shared.signal_bus import APP_SIGNAL_BUS, SignalBus

logger = logging.getLogger("PixelHand.container")


@dataclass
class ServiceContainer:
    """
    Holds references to application-wide services.
    Passed to Controllers and Workers to provide access to infrastructure.
    """

    db_service: DatabaseService
    model_manager: ModelManager
    task_manager: TaskManager
    thumbnail_cache: AbstractThumbnailCache
    event_bus: SignalBus

    # Context Paths
    app_data_dir: Path
    cache_dir: Path
    models_dir: Path
    temp_dir: Path

    @classmethod
    def create(cls, app_data_dir: Path, headless: bool = False, max_workers: int = 4) -> "ServiceContainer":
        """
        Factory method to initialize the application environment and services.

        Args:
            app_data_dir: Root directory for application data.
            headless: Mode flag (Used for logging/task manager config).
            max_workers: Number of workers for the TaskManager.
        """
        # 1. Define and Create Directories
        app_data_dir = app_data_dir.resolve()
        cache_dir = app_data_dir / ".cache"
        models_dir = app_data_dir / "models"
        temp_dir = app_data_dir / "temp"

        for path in [app_data_dir, cache_dir, models_dir, temp_dir]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing Service Container. Root: {app_data_dir}")
        logger.info(f"Mode: {'Headless/CLI' if headless else 'GUI'}")

        # 2. Initialize Infrastructure Services

        # Init UI Thumbnail Cache
        if headless:
            thumbnail_cache = DummyThumbnailCache()
        else:
            thumbnail_cache = LanceDBThumbnailCache(in_memory=False) if LANCEDB_AVAILABLE else DummyThumbnailCache()

        # Task Manager: Handles threading abstraction
        task_manager = TaskManager(headless=headless, max_workers=max_workers)

        # Database Service: Handles Vector DB connection
        db_service = DatabaseService(storage_path=cache_dir)

        # Model Manager: Handles AI Model loading and inference context
        model_manager = ModelManager(models_dir=models_dir)

        # Event Bus: Always use the unified SignalBus (requires QCoreApplication in CLI)
        event_bus = APP_SIGNAL_BUS

        # 3. Return the populated container
        return cls(
            db_service=db_service,
            model_manager=model_manager,
            task_manager=task_manager,
            thumbnail_cache=thumbnail_cache,
            event_bus=event_bus,
            app_data_dir=app_data_dir,
            cache_dir=cache_dir,
            models_dir=models_dir,
            temp_dir=temp_dir,
        )

    def shutdown(self):
        """
        Gracefully shuts down all services, releasing resources and threads.
        """
        logger.info("Shutting down services...")

        # 1. Close Database Connections
        if self.db_service:
            try:
                self.db_service.close()
            except Exception as e:
                logger.error(f"Error closing DB: {e}")

        # 2. Release AI Resources (ONNX Sessions)
        if self.model_manager:
            try:
                self.model_manager.release_resources()
            except Exception as e:
                logger.error(f"Error releasing AI models: {e}")

        # 3. Stop Thread Pools
        if self.task_manager:
            self.task_manager.shutdown()

        # 4. Close Thumbnail Cache
        if self.thumbnail_cache:
            try:
                self.thumbnail_cache.close()
            except Exception as e:
                logger.error(f"Error closing cache: {e}")

        # 5. Clean Temp Directory (Optional)
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not clean temp dir: {e}")

        logger.info("Services shutdown complete.")
