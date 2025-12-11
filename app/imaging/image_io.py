# app/imaging/image_io.py
"""
Handles all image loading and metadata extraction for the application.
This module acts as a manager, orchestrating a cascade of specialized loaders
to handle a wide variety of image formats with high robustness.
"""

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.imaging.loaders.directxtex_loader import DirectXTexLoader
from app.imaging.loaders.oiio_loader import OIIOLoader
from app.imaging.loaders.pillow_loader import PillowLoader
from app.shared.constants import TonemapMode

app_logger = logging.getLogger("PixelHand.image_io")

# --- Loader Instantiation and Prioritization ---
# DirectXTex is preferred for DDS as it handles modern BC compression and cubemaps best.
DDS_LOADERS = [DirectXTexLoader(), OIIOLoader(), PillowLoader()]

# OIIO is now the primary high-performance loader for general formats.
GENERAL_LOADERS = [OIIOLoader(), PillowLoader()]

ALL_LOADERS = {
    "directx": DDS_LOADERS[0],
    "oiio": OIIOLoader(),
    "pillow": PillowLoader(),
}


def load_image(
    path_or_buffer: str | Path | io.BytesIO,
    tonemap_mode: str = TonemapMode.ENABLED.value,
    shrink: int = 1,
) -> Image.Image | None:
    """
    Loads an image from a path or an in-memory buffer, trying a cascade of loaders.
    """
    # If we receive an in-memory buffer, use Pillow directly.
    if isinstance(path_or_buffer, io.BytesIO):
        try:
            with Image.open(path_or_buffer) as img:
                img.load()
                return img.convert("RGBA") if img.mode != "RGBA" else img
        except Exception as e:
            app_logger.error(f"Pillow failed to load image from memory buffer: {e}")
            return None

    # Logic for handling file paths
    try:
        path = Path(path_or_buffer)
    except (TypeError, ValueError):
        app_logger.error(f"Invalid path provided to load_image: {path_or_buffer}")
        return None

    if not path.exists() or not path.is_file():
        return None

    loaders_to_try = DDS_LOADERS if path.suffix.lower() == ".dds" else GENERAL_LOADERS

    for loader in loaders_to_try:
        try:
            pil_image = loader.load(path, tonemap_mode, shrink=shrink)
            if pil_image:
                return pil_image.convert("RGBA") if pil_image.mode != "RGBA" else pil_image
        except Exception:
            continue

    app_logger.error(f"All available loaders failed for '{path.name}'.")
    return None


def get_image_metadata(path: Path, precomputed_stat: Any = None) -> dict | None:
    """Extracts image metadata using a cascade of loaders, with special enrichment for DDS."""
    try:
        # If the file is gone (deleted), stat() will fail with FileNotFoundError
        stat_result = precomputed_stat or path.stat()
    except (FileNotFoundError, OSError):
        return None

    is_dds = path.suffix.lower() == ".dds"
    loaders_to_try = DDS_LOADERS if is_dds else GENERAL_LOADERS

    for loader in loaders_to_try:
        try:
            metadata = loader.get_metadata(path, stat_result)
            if metadata:
                # If using DirectXTex for DDS, try to enrich color space info from OIIO if possible
                if is_dds and isinstance(loader, DirectXTexLoader):
                    try:
                        oiio_loader = ALL_LOADERS.get("oiio")
                        if oiio_loader:
                            oiio_meta = oiio_loader.get_metadata(path, stat_result)
                            if oiio_meta and (cs := oiio_meta.get("color_space")):
                                metadata["color_space"] = cs
                    except Exception:
                        pass

                return metadata
        except Exception:
            continue

    # Fallback to minimal metadata from stat if everything fails
    try:
        return {
            "resolution": (0, 0),
            "file_size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "format_str": path.suffix.strip(".").upper(),
            "compression_format": "Unknown",
            "format_details": "METADATA FAILED",
            "has_alpha": False,
            "capture_date": None,
            "bit_depth": 0,
            "mipmap_count": 1,
            "texture_type": "2D",
            "color_space": "Unknown",
        }
    except Exception:
        pass

    return None
