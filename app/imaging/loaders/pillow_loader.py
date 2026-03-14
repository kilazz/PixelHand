# app/imaging/loaders/pillow_loader.py
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.shared.constants import PILLOW_AVAILABLE

from .base_loader import BaseLoader

app_logger = logging.getLogger("PixelHand.pillow_loader")


class PillowLoader(BaseLoader):
    """Fallback loader for common image formats using Pillow."""

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not PILLOW_AVAILABLE:
            return None

        try:
            with Image.open(path) as img:
                if shrink > 1:
                    target_w = max(1, img.width // shrink)
                    target_h = max(1, img.height // shrink)
                    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

                # Force loading data into memory so the file can be closed
                img.load()
                return img
        except Exception:
            return None

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not PILLOW_AVAILABLE:
            return None

        try:
            with Image.open(path) as img:
                # Load header only
                format_str = img.format or path.suffix.strip(".").upper()
                compression_format = img.info.get("fourcc", img.mode) if format_str == "DDS" else format_str

                return {
                    "resolution": img.size,
                    "file_size": stat_result.st_size,
                    "mtime": stat_result.st_mtime,
                    "format_str": format_str,
                    "compression_format": compression_format,
                    "format_details": img.mode,
                    "has_alpha": "A" in img.getbands(),
                    "capture_date": None,
                    "bit_depth": 8,
                    "mipmap_count": 1,
                    "texture_type": "2D",
                    "color_space": "sRGB" if "icc_profile" in img.info else "Unknown",
                }
        except Exception:
            return None
