# app/core/qc_rules.py
"""
Centralized logic for Quality Control (QC) rules.
Separates checks into 'Absolute' (single file) and 'Relative' (comparison) categories.
"""

from app.data_models import ImageFingerprint, ScanConfig


def is_power_of_two(n: int) -> bool:
    """Checks if an integer is a power of two (e.g., 256, 512, 1024)."""
    return (n != 0) and ((n & (n - 1)) == 0)


class QCRules:
    """
    Static container for QC validation logic.
    """

    @staticmethod
    def check_absolute(fp: ImageFingerprint, config: ScanConfig) -> list[str]:
        """
        Runs checks that depend only on the file itself (Single Folder QC).
        """
        issues = []
        w, h = fp.resolution

        # 1. Non-Power-Of-Two (NPOT) Check
        # Important for game engines (mipmaps, streaming)
        if config.qc_check_npot and not (is_power_of_two(w) and is_power_of_two(h)):
            issues.append("Non-Power-Of-Two (NPOT)")

        # 2. Block Alignment Check
        # DXT/BC compression requires dimensions divisible by 4
        if config.qc_check_block_align and (w > 0 and h > 0 and (w % 4 != 0 or h % 4 != 0)):
            issues.append("Bad Alignment (Not divisible by 4)")

        # 3. Mipmaps Check
        # Large textures without mipmaps cause aliasing and cache misses
        # Ignore very small textures (<64px) where mips might be overkill/irrelevant
        if config.qc_check_mipmaps and min(w, h) >= 64 and fp.mipmap_count <= 1:
            issues.append("Missing Mipmaps")

        # 4. Bit Depth Check
        # Example: Flagging accidental 16-bit saves for standard textures
        if config.qc_check_bit_depth and fp.bit_depth > 8:
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

        # 1. Resolution Downgrade Check
        area_a = fp_source.resolution[0] * fp_source.resolution[1]
        area_b = fp_target.resolution[0] * fp_target.resolution[1]
        if area_b < area_a:
            issues.append("Resolution Downgrade")

        # 2. Size Bloat Check
        # Flag if target is significantly larger (>1.5x) than source
        if config.qc_check_size_bloat and fp_target.file_size > (fp_source.file_size * 1.5):
            issues.append("Size Bloat (>1.5x)")

        # 3. Alpha Channel Mismatch
        if config.qc_check_alpha:
            if fp_source.has_alpha and not fp_target.has_alpha:
                issues.append("Lost Alpha Channel")
            elif not fp_source.has_alpha and fp_target.has_alpha:
                issues.append("Added Empty Alpha")

        # 4. Color Space Mismatch
        if config.qc_check_color_space:
            cs_a = fp_source.color_space or "Unknown"
            cs_b = fp_target.color_space or "Unknown"
            # Ignore if both are unknown, otherwise flag difference
            if cs_a != cs_b and cs_a != "Unknown" and cs_b != "Unknown":
                issues.append(f"Color Space Diff ({cs_a}->{cs_b})")

        # 5. Compression / Format Change
        if config.qc_check_compression:
            fmt_a = str(fp_source.compression_format).upper()
            fmt_b = str(fp_target.compression_format).upper()
            lossless_formats = ["PNG", "TGA", "BMP", "PSD", "TIFF", "TIF"]

            # Check if format changed between compressed types (e.g., BC1 -> BC3)
            if fmt_a != fmt_b and "BC" in fmt_a and "BC" in fmt_b:
                issues.append(f"Format Change ({fmt_a}->{fmt_b})")

            # Check if Source was compressed but Target is uncompressed (likely accidental decompression)
            elif ("BC" in fmt_a or "DXT" in fmt_a) and any(f in fmt_b for f in lossless_formats):
                issues.append(f"Decompressed ({fmt_a}->{fmt_b})")

        return issues
