# app/infrastructure/configuration.py
"""
Contains the ScanConfigBuilder class, responsible for constructing a valid
ScanConfig object from the application's settings and current scan context.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.data_models import AppSettings, PerformanceConfig, ScanConfig, ScanMode
from app.shared.constants import SUPPORTED_MODELS, QuantizationMode
from app.shared.utils import get_model_folder_name

if TYPE_CHECKING:
    pass


class ScanConfigBuilder:
    """A builder class that centralizes the logic for creating a ScanConfig
    from the application's settings and current scan context.
    """

    def __init__(
        self,
        settings: AppSettings,
        scan_mode: ScanMode,
        search_query: str | None,
        sample_path: Path | None,
        comparison_folder_path: Path | None = None,
    ):
        self.settings = settings
        self.scan_mode = scan_mode
        self.search_query = search_query
        self.sample_path = sample_path
        self.comparison_folder_path = comparison_folder_path

    def build(self) -> ScanConfig:
        """Constructs and validates a ScanConfig object."""
        folder_path = self._validate_folder_path()
        self._validate_search_inputs()

        # --- Validation for Folder Compare Mode ---
        if self.scan_mode == ScanMode.FOLDER_COMPARE:
            if not self.comparison_folder_path:
                raise ValueError("Please select a second folder for comparison.")
            if not self.comparison_folder_path.exists() or not self.comparison_folder_path.is_dir():
                raise ValueError("The comparison folder path is invalid or does not exist.")

            if self.comparison_folder_path.resolve() == folder_path.resolve():
                raise ValueError("Source and Comparison folders must be different.")

        model_info, onnx_name = self._get_model_details()
        performance_config = self._build_performance_config()

        # --- Build Active Channels List ---
        channels = []
        if self.settings.hashing.channel_r:
            channels.append("R")
        if self.settings.hashing.channel_g:
            channels.append("G")
        if self.settings.hashing.channel_b:
            channels.append("B")
        if self.settings.hashing.channel_a:
            channels.append("A")

        if not channels:
            channels = ["R", "G", "B", "A"]

        return ScanConfig(
            folder_path=folder_path,
            similarity_threshold=int(self.settings.threshold),
            excluded_folders=[p.strip() for p in self.settings.exclude.split(",") if p.strip()],
            model_name=onnx_name,
            model_dim=model_info["dim"],
            selected_extensions=self.settings.selected_extensions,
            perf=performance_config,
            search_precision=self.settings.performance.search_precision,
            scan_mode=self.scan_mode,
            device=self.settings.performance.device,
            use_ai=self.settings.hashing.use_ai,
            find_exact_duplicates=self.settings.hashing.find_exact,
            find_simple_duplicates=self.settings.hashing.find_simple,
            dhash_threshold=self.settings.hashing.dhash_threshold,
            find_perceptual_duplicates=self.settings.hashing.find_perceptual,
            phash_threshold=self.settings.hashing.phash_threshold,
            find_structural_duplicates=self.settings.hashing.find_structural,
            whash_threshold=self.settings.hashing.whash_threshold,
            compare_by_luminance=self.settings.hashing.compare_by_luminance,
            compare_by_channel=self.settings.hashing.compare_by_channel,
            # --- Disable In-Memory mode permanently for scalability ---
            lancedb_in_memory=False,
            save_visuals=self.settings.visuals.save,
            max_visuals=int(self.settings.visuals.max_count),
            visuals_columns=self.settings.visuals.columns,
            tonemap_visuals=self.settings.visuals.tonemap_enabled,
            tonemap_view=self.settings.viewer.tonemap_view,
            # Fields with default values
            ignore_solid_channels=self.settings.hashing.ignore_solid_channels,
            active_channels=channels,
            channel_split_tags=[
                tag.strip().lower() for tag in self.settings.hashing.channel_split_tags.split(",") if tag.strip()
            ],
            model_info=model_info,
            sample_path=self.sample_path,
            search_query=self.search_query if self.scan_mode == ScanMode.TEXT_SEARCH else None,
            comparison_folder_path=self.comparison_folder_path,
            # QC Flags
            hide_same_resolution_groups=self.settings.hashing.hide_same_resolution_groups,
            qc_check_alpha=self.settings.hashing.qc_check_alpha,
            qc_check_npot=self.settings.hashing.qc_check_npot,
            qc_check_mipmaps=self.settings.hashing.qc_check_mipmaps,
            qc_check_size_bloat=self.settings.hashing.qc_check_size_bloat,
            qc_check_solid_color=self.settings.hashing.qc_check_solid_color,
            qc_check_color_space=self.settings.hashing.qc_check_color_space,
            qc_check_bit_depth=self.settings.hashing.qc_check_bit_depth,
            match_by_stem=self.settings.hashing.match_by_stem,
            # New QC Flags
            qc_check_compression=self.settings.hashing.qc_check_compression,
            qc_check_block_align=self.settings.hashing.qc_check_block_align,
            # --- Normal Map QC ---
            qc_check_normal_maps=self.settings.hashing.qc_check_normal_maps,
            # Parse comma-separated tags into a list of lowercase strings
            qc_normal_maps_tags=[
                t.strip().lower() for t in self.settings.hashing.qc_normal_maps_tags.split(",") if t.strip()
            ],
        )

    def _validate_folder_path(self) -> Path:
        """Validates that the selected folder path exists and is a directory."""
        folder_path_str = self.settings.folder_path
        if not folder_path_str:
            raise ValueError("Please select a folder to scan.")
        folder_path = Path(folder_path_str)
        if not folder_path.is_dir():
            raise ValueError(f"The selected path is not a valid folder:\n{folder_path}")
        return folder_path

    def _validate_search_inputs(self):
        """Validates inputs specific to text or sample search modes."""
        if self.scan_mode == ScanMode.TEXT_SEARCH and not (self.search_query and self.search_query.strip()):
            raise ValueError("Please enter a text search query.")

        if self.scan_mode == ScanMode.SAMPLE_SEARCH and not (self.sample_path and self.sample_path.is_file()):
            raise ValueError("Please select a valid sample image for the search.")

    def _get_model_details(self) -> tuple[dict, str]:
        """Determines the correct ONNX model name based on UI selections."""
        model_info = SUPPORTED_MODELS.get(self.settings.model_key, next(iter(SUPPORTED_MODELS.values())))
        quant_mode_str = self.settings.performance.quantization_mode

        # Safe conversion from string to Enum
        quant_mode = next(
            (q for q in QuantizationMode if q.value == quant_mode_str),
            QuantizationMode.FP16,
        )

        onnx_name = get_model_folder_name(model_info["onnx_name"], quant_mode)

        return model_info, onnx_name

    def _build_performance_config(self) -> PerformanceConfig:
        """Constructs the PerformanceConfig dataclass from UI settings."""
        try:
            batch_size = int(self.settings.performance.batch_size)
            if batch_size <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("Batch size must be a positive integer.") from None

        try:
            num_workers = int(self.settings.performance.num_workers)
            if num_workers <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("Number of workers must be a positive integer.") from None

        return PerformanceConfig(
            num_workers=num_workers,
            run_at_low_priority=self.settings.performance.low_priority,
            batch_size=batch_size,
        )
