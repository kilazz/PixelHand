# app/imaging/loaders/base_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class BaseLoader(ABC):
    """Abstract base class defining the interface for all image loaders."""

    @abstractmethod
    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        """
        Loads image data into a Pillow Image object.

        Args:
            path: Path to the image file.
            tonemap_mode: Strategy for handling HDR data ('enabled' or 'none').
            shrink: Downscaling factor (1 = full size, 2 = half size, etc.).
        """
        pass

    @abstractmethod
    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        """
        Extracts metadata from an image file.

        Args:
            path: Path to the image file.
            stat_result: Cached os.stat result to avoid extra I/O.

        Returns:
            Dictionary containing metadata or None if extraction fails.
        """
        pass
