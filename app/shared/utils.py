# app/shared/utils.py
"""
General utility functions for file system operations, data handling, and
application-specific helpers.
"""

import errno
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.shared.constants import (
    APP_DATA_DIR,
    CACHE_DIR,
    FP16_MODEL_SUFFIX,
    MODELS_DIR,
    SUPPORTED_MODELS,
    QuantizationMode,
)

utils_logger = logging.getLogger("PixelHand.utils")


class UnionFind:
    """
    A simple and efficient Union-Find (Disjoint Set Union) implementation.
    Replaces scipy.sparse.csgraph.connected_components for grouping duplicates.
    """

    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        path = []
        root = i
        while self.parent[root] != root:
            path.append(root)
            root = self.parent[root]
        for node in path:
            self.parent[node] = root
        return root

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i

    def get_groups(self) -> dict:
        """Returns a dictionary mapping representative ID to list of member IDs."""
        groups = {}
        for i in self.parent:
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


def format_result_metadata(node: Any) -> str:
    """
    Helper to create the detailed metadata string from a ResultNode-like object.
    Used in UI Delegates and Models.
    """
    if getattr(node, "path", "") == "loading_dummy":
        return ""

    res_w = getattr(node, "resolution_w", 0)
    res_h = getattr(node, "resolution_h", 0)
    res = f"{res_w}x{res_h}"

    size_mb = (getattr(node, "file_size", 0) or 0) / (1024**2)
    size_str = f"{size_mb:.2f} MB"

    bit_depth = getattr(node, "bit_depth", 0)
    bit_depth_str = f"{bit_depth}-bit" if bit_depth else ""

    parts = [
        res,
        size_str,
        getattr(node, "format_str", ""),
        getattr(node, "compression_format", ""),
        getattr(node, "color_space", ""),
        bit_depth_str,
        getattr(node, "format_details", ""),
        getattr(node, "texture_type", ""),
        f"Mips: {getattr(node, 'mipmap_count', 0)}",
    ]

    return " • ".join(filter(None, parts))


def find_best_in_group(group: list) -> any:
    """Heuristically finds the 'best' file in a group."""
    if not group:
        raise ValueError("Input group cannot be empty.")

    def get_format_score(fp) -> int:
        fmt = str(getattr(fp, "format_str", "")).upper()
        if fmt in ["PNG", "BMP", "TIFF", "TIF", "EXR"]:
            return 2
        if fmt in ["JPEG", "JPG", "WEBP", "AVIF", "TGA"]:
            return 1
        return 0

    return max(
        group,
        key=lambda fp: (
            getattr(fp, "resolution", (0, 0))[0] * getattr(fp, "resolution", (0, 0))[1],
            get_format_score(fp),
            getattr(fp, "file_size", 0),
            -len(str(fp.path.name)),
            -(getattr(fp, "capture_date", 0) or 0),
        ),
    )


def find_common_base_name(paths: list[Path]) -> str:
    if not paths:
        return ""
    stems = [p.stem for p in paths]
    if len(stems) < 2:
        return stems[0] if stems else ""

    shortest = min(stems, key=len)
    for i, char in enumerate(shortest):
        if any(other[i] != char for other in stems):
            last_sep = max(shortest.rfind(s, 0, i) for s in "_- ")
            return shortest[:last_sep] if last_sep != -1 else shortest[:i]

    return shortest


def is_onnx_model_cached(onnx_model_name: str) -> bool:
    model_path = MODELS_DIR / onnx_model_name
    if not (model_path.exists() and (model_path / "visual.onnx").exists()):
        return False
    cfg = next((c for c in SUPPORTED_MODELS.values() if onnx_model_name.startswith(c["onnx_name"])), None)
    return not (cfg and cfg.get("supports_text_search") and not (model_path / "text.onnx").exists())


def _clear_directory(dir_path: Path) -> bool:
    if not dir_path.exists():
        return True
    try:
        shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        utils_logger.error(f"Failed to clear directory {dir_path}: {e}")
        return False


def clear_scan_cache() -> bool:
    return _clear_directory(CACHE_DIR)


def clear_models_cache() -> bool:
    return _clear_directory(MODELS_DIR)


def clear_all_app_data() -> bool:
    return _clear_directory(APP_DATA_DIR)


def check_link_support(folder_path: Path) -> dict[str, bool]:
    support = {"hardlink": True, "reflink": False}
    if not (folder_path.is_dir() and hasattr(os, "reflink")):
        return support

    source = folder_path / f"__reflink_test_{uuid.uuid4()}"
    dest = folder_path / f"__reflink_test_{uuid.uuid4()}"
    try:
        source.write_text("test")
        os.reflink(source, dest)
        support["reflink"] = True
    except OSError as e:
        if e.errno != errno.EOPNOTSUPP:
            utils_logger.warning(f"Could not confirm reflink support due to OS error: {e}")
    except Exception as e:
        utils_logger.error(f"An unexpected error occurred during reflink check: {e}")
    finally:
        source.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)

    return support


def get_model_folder_name(onnx_base_name: str, quant_mode: QuantizationMode) -> str:
    """
    Centralized logic for model folder naming based on quantization mode.
    """
    if quant_mode == QuantizationMode.FP16:
        return f"{onnx_base_name}{FP16_MODEL_SUFFIX}"
    elif quant_mode == QuantizationMode.INT8:
        return f"{onnx_base_name}_int8"
    return onnx_base_name
