# app/domain/config.py
"""
Contains configuration dataclasses used to parameterize the scanning process.
Splitting the configuration allows specific strategies and workers to receive
only the settings they need, adhering to the Interface Segregation Principle.
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.domain.data_models import ScanMode


@dataclass
class AIConfig:
    """Configuration related to the Neural Network model and inference."""

    model_name: str
    model_dim: int
    device: str  # e.g., "CPUExecutionProvider", "DmlExecutionProvider"
    use_ai: bool
    quantization_mode: str  # "FP16", "INT8", "FP32"

    # Model capabilities
    supports_text_search: bool = True
    supports_image_search: bool = True


@dataclass
class PerformanceConfig:
    """Runtime performance settings."""

    num_workers: int = 4
    batch_size: int = 64
    run_at_low_priority: bool = True
    search_precision: str = "Balanced (Default)"  # LanceDB search settings preset


@dataclass
class QCConfig:
    """Settings for Quality Control validation rules."""

    check_alpha: bool = False
    check_npot: bool = False  # Non-Power-Of-Two
    check_mipmaps: bool = False
    check_normal_maps: bool = False
    check_solid_color: bool = False
    check_size_bloat: bool = False
    check_color_space: bool = False
    check_bit_depth: bool = False
    check_compression: bool = False
    check_block_align: bool = False

    # Specific filtering for Normal Map checks
    normal_maps_tags: list[str] = field(default_factory=list)

    # Contextual flags
    hide_same_resolution_groups: bool = False
    match_by_stem: bool = False


@dataclass
class HashingConfig:
    """Settings for standard image hashing algorithms."""

    # Algorithm Toggles
    find_exact: bool = True  # xxHash
    find_simple: bool = True  # dHash
    find_perceptual: bool = True  # pHash
    find_structural: bool = False  # wHash

    # Thresholds (Hamming Distance)
    dhash_threshold: int = 8
    phash_threshold: int = 8
    whash_threshold: int = 2

    # Image Pre-processing for Hashing
    active_channels: list[str] = field(default_factory=lambda: ["R", "G", "B", "A"])
    ignore_solid_channels: bool = True
    compare_by_luminance: bool = False
    compare_by_channel: bool = False
    channel_split_tags: list[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Settings for generating results and visualizations."""

    save_visuals: bool = False
    max_visuals: int = 100
    visuals_columns: int = 6
    tonemap_visuals: bool = False
    tonemap_view: str = "Khronos PBR Neutral"


@dataclass
class ScanConfig:
    """
    The Aggregate Configuration Root.
    Holds all context required to perform a scan operation.
    """

    # --- Context ---
    folder_path: Path
    scan_mode: ScanMode

    # --- Sub-Configurations ---
    ai: AIConfig
    perf: PerformanceConfig
    qc: QCConfig
    hashing: HashingConfig
    output: OutputConfig

    # --- File Filtering ---
    excluded_folders: list[str] = field(default_factory=list)
    selected_extensions: list[str] = field(default_factory=list)

    # --- Search / Comparison Context ---
    # Used only when scan_mode is TEXT_SEARCH
    search_query: str | None = None

    # Used only when scan_mode is SAMPLE_SEARCH
    sample_path: Path | None = None

    # Used only when scan_mode is FOLDER_COMPARE
    comparison_folder_path: Path | None = None

    # Threshold for AI Similarity Search (0-100)
    similarity_threshold: int = 70

    @property
    def is_comparison_mode(self) -> bool:
        return self.scan_mode == ScanMode.FOLDER_COMPARE

    @property
    def is_search_mode(self) -> bool:
        return self.scan_mode in (ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH)
