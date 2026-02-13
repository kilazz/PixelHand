# app/imaging/processing.py
"""
Contains shared image processing utility functions to avoid circular dependencies.
"""

import logging

import numpy as np
from PIL import Image, ImageChops, ImageOps

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


def extract_channel_as_rgb(pil_image: Image.Image, channel: str) -> Image.Image | None:
    """
    Extracts a specific channel (R, G, B, or A) and returns it as a grayscale RGB image.
    """
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
    idx = channel_map.get(channel)

    if idx is not None:
        bands = pil_image.split()
        if idx < len(bands):
            band = bands[idx]
            return Image.merge("RGB", (band, band, band))
    return None


def process_image_channels(pil_image: Image.Image, active_channels: dict[str, bool]) -> Image.Image:
    """
    Returns a new image with only the active channels visible.
    Inactive channels are zeroed out (black), except Alpha which becomes fully opaque if disabled
    to allow seeing the RGB data.
    """
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    # Optimization: If only 1 channel is active, show it as grayscale
    active_keys = [k for k, v in active_channels.items() if v]
    if len(active_keys) == 1:
        extracted = extract_channel_as_rgb(pil_image, active_keys[0])
        if extracted:
            # Ensure it has an alpha channel for consistency
            extracted.putalpha(Image.new("L", extracted.size, 255))
            return extracted

    # Standard mixing
    r, g, b, a = pil_image.split()

    if not active_channels.get("R", True):
        r = r.point(lambda _: 0)
    if not active_channels.get("G", True):
        g = g.point(lambda _: 0)
    if not active_channels.get("B", True):
        b = b.point(lambda _: 0)
    if not active_channels.get("A", True):
        # If Alpha is disabled, we usually want to see the RGB data fully opaque
        a = a.point(lambda _: 255)

    return Image.merge("RGBA", (r, g, b, a))


def calculate_diff_image(img1: Image.Image, img2: Image.Image, channels: dict[str, bool], heatmap: bool = False) -> Image.Image:
    """
    Calculates the difference between two images based on active channels.
    """
    # 1. Resize to match largest dimensions
    if img1.size != img2.size:
        ts = (max(img1.width, img2.width), max(img1.height, img2.height))
        img1 = img1.resize(ts, Image.Resampling.LANCZOS)
        img2 = img2.resize(ts, Image.Resampling.LANCZOS)

    if img1.mode != "RGBA":
        img1 = img1.convert("RGBA")
    if img2.mode != "RGBA":
        img2 = img2.convert("RGBA")

    r1, g1, b1, a1 = img1.split()
    r2, g2, b2, a2 = img2.split()

    # Calculate diff per channel if active
    r_diff = ImageChops.difference(r1, r2) if channels.get("R", True) else Image.new("L", img1.size, 0)
    g_diff = ImageChops.difference(g1, g2) if channels.get("G", True) else Image.new("L", img1.size, 0)
    b_diff = ImageChops.difference(b1, b2) if channels.get("B", True) else Image.new("L", img1.size, 0)
    a_diff = ImageChops.difference(a1, a2) if channels.get("A", True) else Image.new("L", img1.size, 0)

    if heatmap:
        # Sum differences and colorize
        diff_sum = ImageChops.add(r_diff, g_diff)
        diff_sum = ImageChops.add(diff_sum, b_diff)
        diff_sum = ImageChops.add(diff_sum, a_diff)
        # Blue = No Diff, Red = Max Diff
        return ImageOps.colorize(diff_sum, black="blue", white="red")
    else:
        # Return composite difference
        # Alpha is set to opaque so we can see the RGB difference clearly
        return Image.merge("RGBA", (r_diff, g_diff, b_diff, Image.new("L", img1.size, 255)))
