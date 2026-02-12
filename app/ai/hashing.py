# app/ai/hashing.py
"""
Contains lightweight, standalone worker functions for hashing and metadata extraction.
These functions are designed to be run in a ProcessPoolExecutor (CPU-bound) or
ThreadPoolExecutor (I/O-bound).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import xxhash

from app.imaging.image_io import get_image_metadata, load_image
from app.imaging.processing import is_vfx_transparent_texture

if TYPE_CHECKING:
    from app.domain.data_models import AnalysisItem

try:
    import imagehash
    from PIL import Image, ImageStat

    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    Image = None
    IMAGEHASH_AVAILABLE = False


def worker_calculate_hashes_and_meta(path: Path) -> dict[str, Any] | None:
    """
    A lightweight worker that collects basic file metadata and a full-file xxHash.
    Optimized to fail early if metadata cannot be read, avoiding unnecessary I/O.
    """
    try:
        # 1. Try to read metadata first (Fast header read).
        # If the file is corrupt or the format is unsupported, we stop here.
        meta = get_image_metadata(path)
        if not meta:
            return None

        # 2. Calculate xxHash (Heavy I/O).
        # We use a larger buffer (8MB) to optimize for modern SSDs.
        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                hasher.update(chunk)
        xxh = hasher.hexdigest()

        return {
            "path": path,
            "meta": meta,
            "xxhash": xxh,
        }
    except (OSError, PermissionError):
        # Common file system errors (locked file, permission denied)
        return None
    except Exception:
        # Catch-all for corrupt files or library errors to prevent worker crash.
        # We avoid printing stack traces to keep the console clean in production.
        return None


def worker_calculate_perceptual_hashes(item: "AnalysisItem", ignore_solid_channels: bool) -> dict[str, Any] | None:
    """
    Calculates perceptual hashes (dHash, pHash, wHash) for an image or a specific channel.
    """
    path = item.path
    analysis_type = item.analysis_type

    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        # Optimization: We don't need full resolution for perceptual hashing.
        # shrink=4 significantly reduces memory usage and decode time.
        original_pil_img = load_image(path, shrink=4)

        if not original_pil_img:
            return None

        # Further downscale for hashing algorithms
        base_size = (128, 128)
        if original_pil_img.width > 128 or original_pil_img.height > 128:
            original_pil_img.thumbnail(base_size, Image.Resampling.NEAREST)

        image_for_hashing = None

        if analysis_type == "Luminance":
            image_for_hashing = original_pil_img.convert("L")

        elif analysis_type in ("R", "G", "B", "A"):
            rgba_img = original_pil_img.convert("RGBA") if original_pil_img.mode != "RGBA" else original_pil_img
            channels = rgba_img.split()
            channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
            channel_index = channel_map.get(analysis_type)

            if channel_index is not None and len(channels) > channel_index:
                channel_to_check = channels[channel_index]

                if ignore_solid_channels:
                    # Check if channel is solid color (e.g., all black alpha)
                    min_val, max_val = channel_to_check.getextrema()
                    # Tolerance for compression artifacts
                    if max_val < 5 or min_val > 250:
                        return None

                    # Statistical check for near-solid colors
                    stat = ImageStat.Stat(channel_to_check)
                    if stat.mean[0] < 5 or stat.mean[0] > 250:
                        return None

                image_for_hashing = channel_to_check
            else:
                return None
        else:  # "Composite" (Default)
            if original_pil_img.mode == "RGBA":
                if is_vfx_transparent_texture(original_pil_img):
                    # If VFX (Alpha=0 but data in RGB), ignore alpha and hash the RGB data
                    image_for_hashing = original_pil_img.convert("RGB")
                else:
                    # Standard compositing: Paste onto black background
                    bg = Image.new("RGB", original_pil_img.size, (0, 0, 0))
                    bg.paste(original_pil_img, mask=original_pil_img.split()[3])
                    image_for_hashing = bg
            else:
                image_for_hashing = original_pil_img.convert("RGB")

        if image_for_hashing:
            dhash = imagehash.dhash(image_for_hashing)
            phash = imagehash.phash(image_for_hashing)
            whash = imagehash.whash(image_for_hashing)

            precise_meta = {
                "resolution": original_pil_img.size,
                "format_details": f"{original_pil_img.mode}",
                "has_alpha": "A" in original_pil_img.getbands(),
            }

            # Query real metadata to update DB with actual dimensions, not thumbnail dimensions
            real_meta = get_image_metadata(path)
            if real_meta:
                precise_meta["resolution"] = real_meta["resolution"]

            return {
                "path": path,
                "analysis_type": analysis_type,
                "dhash": dhash,
                "phash": phash,
                "whash": whash,
                "precise_meta": precise_meta,
            }
        return None
    except Exception:
        # Return None on error so the pipeline simply skips this item
        return None
