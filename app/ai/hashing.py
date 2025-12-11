# app/ai/hashing.py
"""
Contains lightweight, standalone worker functions for hashing and metadata extraction.
"""

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xxhash

from app.imaging.image_io import get_image_metadata, load_image

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
    """
    try:
        meta = get_image_metadata(path)
        if not meta:
            return None

        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                hasher.update(chunk)
        xxh = hasher.hexdigest()

        return {
            "path": path,
            "meta": meta,
            "xxhash": xxh,
        }
    except OSError:
        return None
    except Exception as e:
        print(f"!!! XXHASH WORKER CRASH on {path.name}: {e}")
        traceback.print_exc()
        return None


def worker_calculate_perceptual_hashes(item: "AnalysisItem", ignore_solid_channels: bool) -> dict[str, Any] | None:
    """
    Calculates perceptual hashes (dHash, pHash, wHash).
    """
    path = item.path
    analysis_type = item.analysis_type

    try:
        # Optimization: We don't need full resolution for hashing.
        original_pil_img = load_image(path, shrink=4)

        if not original_pil_img:
            return None

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
                    min_val, max_val = channel_to_check.getextrema()
                    if max_val < 5 or min_val > 250:
                        return None
                    stat = ImageStat.Stat(channel_to_check)
                    if stat.mean[0] < 5 or stat.mean[0] > 250:
                        return None

                image_for_hashing = channel_to_check
            else:
                return None
        else:  # "Composite"
            if original_pil_img.mode == "RGBA":
                is_vfx = False
                try:
                    extrema = original_pil_img.getextrema()
                    if (
                        len(extrema) >= 4
                        and extrema[3][1] == 0
                        and (extrema[0][1] > 0 or extrema[1][1] > 0 or extrema[2][1] > 0)
                    ):
                        is_vfx = True
                except Exception:
                    pass

                if is_vfx:
                    image_for_hashing = original_pil_img.convert("RGB")
                else:
                    bg = Image.new("RGB", original_pil_img.size, (0, 0, 0))
                    bg.paste(original_pil_img, mask=original_pil_img.split()[3])
                    image_for_hashing = bg
            else:
                image_for_hashing = original_pil_img.convert("RGB")

        if image_for_hashing and IMAGEHASH_AVAILABLE:
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
    except Exception as e:
        print(f"Hash Worker Error on {path}: {e}")
        return None
