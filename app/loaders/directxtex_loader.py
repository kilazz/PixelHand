# app/loaders/directxtex_loader.py
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.constants import DIRECTXTEX_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder

app_logger = logging.getLogger("PixelHand.dds_loader")


class DirectXTexLoader(BaseLoader):
    """Loader for DDS files using the directxtex_decoder library."""

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not DIRECTXTEX_AVAILABLE:
            return None

        try:
            decoded = directxtex_decoder.decode_dds(path.read_bytes())
            numpy_array, dtype = decoded["data"], decoded["data"].dtype

            pil_image = None
            if np.issubdtype(dtype, np.floating):
                # Handle Float (HDR)
                if tonemap_mode == TonemapMode.ENABLED.value:
                    pil_image = Image.fromarray(tonemap_float_array(numpy_array.astype(np.float32)))
                else:
                    pil_image = Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
            elif np.issubdtype(dtype, np.uint16):
                # Handle 16-bit Int
                pil_image = Image.fromarray((numpy_array // 257).astype(np.uint8))
            elif np.issubdtype(dtype, np.signedinteger):
                # Handle Signed Int
                info = np.iinfo(dtype)
                norm = (numpy_array.astype(np.float32) - info.min) / (info.max - info.min)
                pil_image = Image.fromarray((norm * 255).astype(np.uint8))
            elif np.issubdtype(dtype, np.uint8):
                # Handle Standard 8-bit
                pil_image = Image.fromarray(numpy_array)

            if pil_image is None:
                raise TypeError(f"Unhandled NumPy dtype from DirectXTex decoder: {dtype}")

            if shrink > 1:
                # Ensure dimensions never collapse to 0
                target_w = max(1, pil_image.width // shrink)
                target_h = max(1, pil_image.height // shrink)

                # Use LANCZOS for high quality downscaling
                pil_image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

            return self._handle_alpha_logic(pil_image)

        except Exception as e:
            app_logger.warning(f"DirectXTex decode failed for {path.name}: {e}")
            return None

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not DIRECTXTEX_AVAILABLE:
            return None

        try:
            dxt_meta = directxtex_decoder.get_dds_metadata(path.read_bytes())
            return {
                "resolution": (dxt_meta["width"], dxt_meta["height"]),
                "file_size": stat_result.st_size,
                "mtime": stat_result.st_mtime,
                "format_str": "DDS",
                "compression_format": dxt_meta["format_str"],
                "format_details": "DXGI",
                "has_alpha": self._get_alpha_from_format_str(dxt_meta["format_str"]),
                "capture_date": None,
                "bit_depth": 8,
                "mipmap_count": dxt_meta["mip_levels"],
                "texture_type": "Cubemap" if dxt_meta["is_cubemap"] else ("3D" if dxt_meta["is_3d"] else "2D"),
                "color_space": "sRGB",
            }
        except Exception:
            return None

    def _get_alpha_from_format_str(self, format_str: str) -> bool:
        fmt = format_str.upper()
        return any(s in fmt for s in ["A8", "A16", "A32", "BC2", "BC3", "BC7"]) or (
            "A" in fmt and any(s in fmt for s in ["R8G8B8A8", "R16G16B16A16", "B8G8R8A8"])
        )

    def _handle_alpha_logic(self, pil_image: Image.Image) -> Image.Image:
        """
        Smart Alpha Handling (Simplified/Fast):
        1. If Alpha is fully opaque (255), discard it to save memory.
        2. If Alpha is fully transparent (0), but RGB has data (Emission/VFX),
           discard Alpha to make the content visible.
        """
        if pil_image.mode != "RGBA":
            return pil_image

        # Use PIL's C-based getextrema() for speed.
        # This is much faster than converting to numpy for large images.
        try:
            extrema = pil_image.getextrema()
            # RGBA extrema format: [(Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax), (Amin, Amax)]
            if len(extrema) >= 4:
                alpha_min, alpha_max = extrema[3]

                # Case 1: Alpha is fully opaque (255). Drop alpha channel.
                if alpha_min >= 255:
                    return pil_image.convert("RGB")

                # Case 2: Alpha is fully transparent (0).
                if alpha_max <= 0:
                    r_max = extrema[0][1]
                    g_max = extrema[1][1]
                    b_max = extrema[2][1]

                    # If RGB contains data, dropping Alpha makes it visible (Opaque).
                    # This fixes "Black Preview" for additive/emission textures.
                    if r_max > 0 or g_max > 0 or b_max > 0:
                        return pil_image.convert("RGB")

        except Exception:
            pass

        return pil_image
