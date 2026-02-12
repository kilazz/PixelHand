# app/imaging/processing.py
"""
Contains shared image processing utility functions to avoid circular dependencies.
"""

import logging

import numpy as np
from PIL import Image

from app.shared.constants import OCIO_AVAILABLE

app_logger = logging.getLogger("PixelHand.imaging.processing")

# --- OCIO Setup ---
TONE_MAPPER = None
if OCIO_AVAILABLE:
    try:
        from simple_ocio import ToneMapper

        TONE_MAPPER = ToneMapper(view="Khronos PBR Neutral")
    except Exception as e:
        app_logger.error(f"Failed to initialize simple-ocio ToneMapper: {e}")
        OCIO_AVAILABLE = False


def set_active_tonemap_view(view_name: str) -> bool:
    """Dynamically sets the active view on the global tone mapper."""
    global TONE_MAPPER
    if TONE_MAPPER and view_name in TONE_MAPPER.available_views and TONE_MAPPER.view != view_name:
        TONE_MAPPER.view = view_name
        return True
    return False


def tonemap_float_array(float_array: np.ndarray) -> np.ndarray:
    """Applies a tonemapping operator to a floating-point NumPy array."""
    if float_array.ndim == 2:
        rgb = np.stack([float_array] * 3, axis=-1)
    elif float_array.shape[-1] > 3:
        rgb = float_array[..., :3]
    else:
        rgb = float_array

    alpha = float_array[..., 3:4] if float_array.ndim > 2 and float_array.shape[-1] > 3 else None
    rgb = np.maximum(rgb, 0.0)

    if TONE_MAPPER:
        try:
            # Apply exposure and tonemap
            rgb_tonemapped = TONE_MAPPER.hdr_to_ldr((rgb * 1.5).astype(np.float32), clip=True)
            final_rgb = (rgb_tonemapped * 255).astype(np.uint8)
        except Exception as e:
            app_logger.error(f"simple-ocio tonemapping failed: {e}")
            final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        # Fallback if OCIO is not available
        final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    if alpha is not None:
        final_alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        if final_rgb.ndim == 3:
            return np.concatenate([final_rgb, final_alpha], axis=-1)
    return final_rgb


def is_vfx_transparent_texture(pil_image: Image.Image) -> bool:
    """
    Detects if an image has a fully transparent alpha channel (Max Alpha <= 0),
    but contains valid visual data in the RGB channels (Max RGB > 0).
    Common in GameDev/VFX for packed textures (e.g. Emission/Roughness in RGB).
    """
    if pil_image.mode != "RGBA":
        return False

    try:
        # getextrema returns [(Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax), (Amin, Amax)]
        extrema = pil_image.getextrema()
        if len(extrema) < 4:
            return False

        alpha_max = extrema[3][1]

        # If Alpha has any opacity, it's a standard image, not a "hidden data" texture.
        if alpha_max > 0:
            return False

        # Alpha is fully transparent. Check if RGB has any data.
        r_max = extrema[0][1]
        g_max = extrema[1][1]
        b_max = extrema[2][1]

        return r_max > 0 or g_max > 0 or b_max > 0

    except Exception:
        return False
