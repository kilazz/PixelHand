# app/ai/preprocessing.py
"""
Handles preparation of raw image files into formats suitable for AI models.
Includes logic for smart-shrinking large files, channel extraction, and resizing.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from app.imaging.image_io import load_image
from app.imaging.processing import is_vfx_transparent_texture

if TYPE_CHECKING:
    from app.domain.data_models import AnalysisItem


class ImageBatchPreprocessor:
    """
    Stateless helper to read images from disk and prepare them for the AI Processor.
    """

    @staticmethod
    def prepare_batch(
        items: list["AnalysisItem"],
        target_size: tuple[int, int],
        ignore_solid_channels: bool = True,
    ) -> tuple[list[Image.Image], list[tuple[str, str]], list[tuple[str, str]]]:
        """
        Reads a list of AnalysisItems and converts them to PIL Images ready for the model.

        Returns:
            (valid_images, valid_paths_with_channels, skipped_items)
        """
        images = []
        successful_items = []  # (path_str, channel_name)
        skipped_items = []  # (path_str, error_reason)

        for item in items:
            path = Path(item.path)
            try:
                # 1. Heuristic Shrink: Don't load a 8K texture at full res just to resize it to 224x224.
                file_size = path.stat().st_size
                shrink = 1
                if file_size > 50 * 1024 * 1024:
                    shrink = 8
                elif file_size > 10 * 1024 * 1024:
                    shrink = 4
                elif file_size > 2 * 1024 * 1024:
                    shrink = 2

                pil_image = load_image(path, shrink=shrink)
                if not pil_image:
                    skipped_items.append((str(path), "Image loading failed"))
                    continue

                processed_image, channel_name = ImageBatchPreprocessor._process_single_image(
                    pil_image, item.analysis_type, ignore_solid_channels
                )

                if processed_image:
                    # Final resize to model input size (e.g. 224x224 or 384x384)
                    if processed_image.size != target_size:
                        processed_image = processed_image.resize(target_size, Image.Resampling.BILINEAR)

                    images.append(processed_image)
                    successful_items.append((str(path), channel_name))
                else:
                    # Logic returned None (e.g. solid alpha channel ignored)
                    pass

                # Explicit cleanup to help GC with large textures
                del pil_image

            except Exception as e:
                skipped_items.append((str(path), str(e)))

        return images, successful_items, skipped_items

    @staticmethod
    def _process_single_image(
        pil_image: Image.Image,
        analysis_type: str,
        ignore_solid: bool,
    ) -> tuple[Image.Image | None, str | None]:
        """
        Handles channel splitting (R/G/B/A), Luminance conversion, or Composite logic.
        """
        # Ensure working mode
        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGBA")

        # 1. Specific Channel Analysis
        if analysis_type in ("R", "G", "B", "A"):
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            channels = pil_image.split()
            idx = {"R": 0, "G": 1, "B": 2, "A": 3}[analysis_type]

            if idx < len(channels):
                ch = channels[idx]
                # Skip solid black channels if requested (usually empty Alpha)
                if ignore_solid and ch.getextrema()[1] < 5:
                    return None, None
                # Merge channel into grayscale RGB representation
                return Image.merge("RGB", (ch, ch, ch)), analysis_type

        # 2. Luminance Analysis
        elif analysis_type == "Luminance":
            return pil_image.convert("L").convert("RGB"), None

        # 3. Composite (Standard) Analysis
        else:
            if pil_image.mode == "RGBA":
                if is_vfx_transparent_texture(pil_image):
                    # Keep RGB data if alpha is 0
                    return pil_image.convert("RGB"), None

                # Standard compositing onto black background
                bg = Image.new("RGB", pil_image.size, (0, 0, 0))
                bg.paste(pil_image, mask=pil_image.split()[3])
                return bg, None
            else:
                return pil_image.convert("RGB"), None

        return None, None
