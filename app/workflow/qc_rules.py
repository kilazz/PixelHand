# app/workflow/qc_rules.py
"""
Centralized logic for Quality Control (QC) rules.
Separates checks into 'Absolute' (single file) and 'Relative' (comparison) categories.
"""

import logging

import numpy as np
from PIL import Image

from app.domain.config import ScanConfig
from app.domain.data_models import ImageFingerprint

app_logger = logging.getLogger("PixelHand.workflow.qc_rules")


def is_power_of_two(n: int) -> bool:
    """Checks if an integer is a power of two."""
    return (n != 0) and ((n & (n - 1)) == 0)


class QCRules:
    """
    Static container for QC validation logic.
    """

    @staticmethod
    def check_normal_map_integrity(pil_image: Image.Image, threshold: float = 0.15) -> tuple[str, str] | None:
        """
        Checks if pixels form normalized vectors (length ~ 1.0).
        Optimized using NumPy. Expects Tangent Space Normal Map.

        Args:
            pil_image: The loaded PIL image.
            threshold: Allowed deviation from length 1.0 (default 0.15 for compression tolerance).

        Returns:
            Tuple (Group Name, Detail String) if validation fails, otherwise None.
        """
        try:
            # Optimization: Work on a smaller copy for speed (errors are usually systemic).
            # We use NEAREST resampling to preserve specific pixel values rather than
            # averaging them, which would artificially shorten the vectors.
            img_proc = pil_image.resize((512, 512), Image.Resampling.NEAREST) if pil_image.width > 512 else pil_image

            if img_proc.mode != "RGB":
                img_proc = img_proc.convert("RGB")

            # Convert to float array
            arr = np.array(img_proc, dtype=np.float32)

            # Transform [0, 255] -> [-1.0, 1.0]
            # Formula: (Value / 255.0) * 2.0 - 1.0
            # Simplified: Value * (2.0 / 255.0) - 1.0
            vectors = arr * (2.0 / 255.0) - 1.0

            # --- Detect "Drop Blue" / "White Z" optimization ---
            # If Blue channel is mostly White (255), the vector length will mathematically be > 1.0
            # (because Z=1 and X,Y!=0). This is valid for game engines (CryEngine/Unity).
            mean_z = np.mean(vectors[..., 2])

            if mean_z > 0.98:
                # "Drop Blue" Mode detected.
                # Instead of checking total length, we must check if X/Y fit inside the unit circle.
                # If X^2 + Y^2 > 1.0, then Z cannot be reconstructed (sqrt of negative).
                len_xy = np.sqrt(vectors[..., 0] ** 2 + vectors[..., 1] ** 2)

                # Check if XY exceeds 1.0 (allowing for compression noise via threshold)
                diff_xy = np.maximum(0, len_xy - 1.0)
                bad_pixels_xy = np.mean(diff_xy > threshold)

                if bad_pixels_xy > 0.10:
                    return (
                        "Bad Normal Map (XY Clip)",
                        f"XY > 1.0 ({bad_pixels_xy:.0%})",
                    )

                # If XY is valid, the file is fine (Drop Blue format)
                return None

            # --- Standard Normal Map Check ---
            # Calculate vector magnitude: sqrt(x^2 + y^2 + z^2)
            # axis=2 because shape is (Height, Width, Channels)
            norms = np.linalg.norm(vectors, axis=2)

            # Calculate deviation from 1.0
            diff = np.abs(norms - 1.0)

            # Count bad pixels (deviation > threshold)
            # Threshold 0.15 accounts for DXT/BC compression artifacts
            bad_pixels_mean = np.mean(diff > threshold)

            if bad_pixels_mean > 0.10:  # If >10% pixels are bad
                # Return tuple: (Group Name for UI grouping, Detail string for metadata)
                return "Bad Normal Map (Integrity)", f"{bad_pixels_mean:.0%}"

            # Optional: Check if it's actually a Normal Map (Z channel should be prominent)
            # In Tangent space, Blue (Z) is usually facing up (approx 1.0)
            mean_vec = np.mean(vectors, axis=(0, 1))
            if mean_vec[2] < 0.2:  # If avg Z is negative or too small, it's likely not a valid NM
                return "Inverted Z / Not Tangent", "Z-Axis issue"

            return None

        except Exception as e:
            app_logger.warning(f"Normal map validation failed: {e}")
            return None

    @staticmethod
    def check_absolute(fp: ImageFingerprint, config: ScanConfig) -> list[str]:
        """
        Runs checks that depend only on the file itself (Single Folder QC).
        """
        issues = []
        w, h = fp.resolution
        qc = config.qc

        # 1. Non-Power-Of-Two (NPOT) Check
        if qc.check_npot and not (is_power_of_two(w) and is_power_of_two(h)):
            issues.append("Non-Power-Of-Two (NPOT)")

        # 2. Block Alignment Check (DXT/BC)
        if qc.check_block_align and (w > 0 and h > 0 and (w % 4 != 0 or h % 4 != 0)):
            issues.append("Bad Alignment (Not divisible by 4)")

        # 3. Mipmaps Check
        if qc.check_mipmaps and min(w, h) >= 64 and fp.mipmap_count <= 1:
            issues.append("Missing Mipmaps")

        # 4. Bit Depth Check
        if qc.check_bit_depth and fp.bit_depth > 8:
            issues.append(f"High Bit Depth ({fp.bit_depth}-bit)")

        return issues

    @staticmethod
    def check_relative(
        fp_source: ImageFingerprint,
        fp_target: ImageFingerprint,
        config: ScanConfig,
    ) -> list[str]:
        """
        Runs checks that compare a Source file against a Target file (Folder Compare).
        """
        issues = []
        qc = config.qc

        # 1. Resolution Downgrade Check
        area_a = fp_source.resolution[0] * fp_source.resolution[1]
        area_b = fp_target.resolution[0] * fp_target.resolution[1]
        if area_b < area_a:
            issues.append("Resolution Downgrade")

        # 2. Size Bloat Check
        if qc.check_size_bloat and fp_target.file_size > (fp_source.file_size * 1.5):
            issues.append("Size Bloat (>1.5x)")

        # 3. Alpha Channel Mismatch
        if qc.check_alpha:
            if fp_source.has_alpha and not fp_target.has_alpha:
                issues.append("Lost Alpha Channel")
            elif not fp_source.has_alpha and fp_target.has_alpha:
                issues.append("Added Empty Alpha")

        # 4. Color Space Mismatch
        if qc.check_color_space:
            cs_a = fp_source.color_space or "Unknown"
            cs_b = fp_target.color_space or "Unknown"
            if cs_a != cs_b and cs_a != "Unknown" and cs_b != "Unknown":
                issues.append(f"Color Space Diff ({cs_a}->{cs_b})")

        # 5. Compression / Format Change
        if qc.check_compression:
            fmt_a = str(fp_source.compression_format).upper()
            fmt_b = str(fp_target.compression_format).upper()
            lossless_formats = ["PNG", "TGA", "BMP", "PSD", "TIFF", "TIF"]

            if fmt_a != fmt_b and "BC" in fmt_a and "BC" in fmt_b:
                issues.append(f"Format Change ({fmt_a}->{fmt_b})")
            elif ("BC" in fmt_a or "DXT" in fmt_a) and any(f in fmt_b for f in lossless_formats):
                issues.append(f"Decompressed ({fmt_a}->{fmt_b})")

        return issues
