# app/infrastructure/configuration.py
"""
Configuration Builder.
"""

from pathlib import Path

from app.domain.config import (
    AIConfig,
    HashingConfig,
    OutputConfig,
    PerformanceConfig,
    QCConfig,
    ScanConfig,
)
from app.domain.data_models import AppSettings, ScanMode
from app.shared.constants import SUPPORTED_MODELS, QuantizationMode
from app.shared.utils import get_model_folder_name


class ScanConfigBuilder:
    """
    Builder pattern to construct a validated ScanConfig object.
    Separates the complexity of config creation from the UI logic.
    """

    def __init__(
        self,
        settings: AppSettings,
        scan_mode: ScanMode,
        search_query: str | None = None,
        sample_path: Path | None = None,
        comparison_folder_path: Path | None = None,
    ):
        self.settings = settings
        self.scan_mode = scan_mode
        self.search_query = search_query
        self.sample_path = sample_path
        self.comparison_folder_path = comparison_folder_path

    def build(self) -> ScanConfig:
        """
        Constructs and validates the final ScanConfig object.
        Raises ValueError if paths are invalid or required inputs are missing.
        """
        # 1. Validate Context
        folder_path = self._validate_folder_path()
        self._validate_search_inputs()
        self._validate_comparison_inputs(folder_path)

        # 2. Build Sub-Configurations
        ai_config = self._build_ai_config()
        perf_config = self._build_performance_config()
        qc_config = self._build_qc_config()
        hashing_config = self._build_hashing_config()
        output_config = self._build_output_config()

        # 3. Assemble Root Config
        return ScanConfig(
            folder_path=folder_path,
            scan_mode=self.scan_mode,
            ai=ai_config,
            perf=perf_config,
            qc=qc_config,
            hashing=hashing_config,
            output=output_config,
            excluded_folders=[p.strip() for p in self.settings.exclude.split(",") if p.strip()],
            selected_extensions=self.settings.selected_extensions,
            search_query=self.search_query,
            sample_path=self.sample_path,
            comparison_folder_path=self.comparison_folder_path,
        )

    def _validate_folder_path(self) -> Path:
        """Ensures the main scan folder exists."""
        folder_path_str = self.settings.folder_path
        if not folder_path_str:
            raise ValueError("Please select a folder to scan.")

        folder_path = Path(folder_path_str)
        if not folder_path.exists():
            raise ValueError(f"The selected path does not exist:\n{folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"The selected path is not a directory:\n{folder_path}")

        return folder_path

    def _validate_search_inputs(self):
        """Validates inputs specific to text or sample search modes."""
        if self.scan_mode == ScanMode.TEXT_SEARCH and (not self.search_query or not self.search_query.strip()):
            raise ValueError("Please enter a text search query.")

        if self.scan_mode == ScanMode.SAMPLE_SEARCH and (not self.sample_path or not self.sample_path.is_file()):
            raise ValueError("Please select a valid sample image for the search.")

    def _validate_comparison_inputs(self, source_path: Path):
        """Validates inputs specific to Folder Comparison mode."""
        if self.scan_mode == ScanMode.FOLDER_COMPARE:
            if not self.comparison_folder_path:
                raise ValueError("Please select a second folder for comparison.")
            if not self.comparison_folder_path.exists():
                raise ValueError("The comparison folder path is invalid.")
            if self.comparison_folder_path.resolve() == source_path.resolve():
                raise ValueError("Source and Comparison folders must be different.")

    def _build_ai_config(self) -> AIConfig:
        """Constructs the AI settings."""
        # Retrieve model metadata from constants
        model_key = self.settings.model_key
        # Fallback to default if key invalid
        model_info = SUPPORTED_MODELS.get(model_key, next(iter(SUPPORTED_MODELS.values())))

        # Parse Quantization Mode
        quant_mode_str = self.settings.performance.quantization_mode
        # Find enum by value, default to FP16
        quant_mode_enum = next(
            (q for q in QuantizationMode if q.value == quant_mode_str),
            QuantizationMode.FP16,
        )

        # Generate actual onnx folder name (e.g., "clip_b32_fp16")
        onnx_name = get_model_folder_name(model_info["onnx_name"], quant_mode_enum)

        return AIConfig(
            model_name=onnx_name,
            model_dim=model_info["dim"],
            device=self.settings.performance.device,
            use_ai=self.settings.hashing.use_ai,
            quantization_mode=quant_mode_enum.name,  # Store as "FP16" string in config
            supports_text_search=model_info.get("supports_text_search", True),
            supports_image_search=model_info.get("supports_image_search", True),
        )

    def _build_performance_config(self) -> PerformanceConfig:
        """Constructs the Performance settings."""
        try:
            batch_size = int(self.settings.performance.batch_size)
            if batch_size <= 0:
                raise ValueError
        except (ValueError, TypeError):
            batch_size = 64

        try:
            num_workers = int(self.settings.performance.num_workers)
            if num_workers <= 0:
                raise ValueError
        except (ValueError, TypeError):
            num_workers = 4

        return PerformanceConfig(
            num_workers=num_workers,
            batch_size=batch_size,
            run_at_low_priority=self.settings.performance.low_priority,
            search_precision=self.settings.performance.search_precision,
        )

    def _build_qc_config(self) -> QCConfig:
        """Constructs the Quality Control settings."""
        h = self.settings.hashing  # Accessing legacy location in AppSettings

        # Parse tags string into list
        nm_tags = [t.strip().lower() for t in h.qc_normal_maps_tags.split(",") if t.strip()]

        return QCConfig(
            check_alpha=h.qc_check_alpha,
            check_npot=h.qc_check_npot,
            check_mipmaps=h.qc_check_mipmaps,
            check_normal_maps=h.qc_check_normal_maps,
            check_solid_color=h.qc_check_solid_color,
            check_size_bloat=h.qc_check_size_bloat,
            check_color_space=h.qc_check_color_space,
            check_bit_depth=h.qc_check_bit_depth,
            check_compression=h.qc_check_compression,
            check_block_align=h.qc_check_block_align,
            normal_maps_tags=nm_tags,
            hide_same_resolution_groups=h.hide_same_resolution_groups,
            match_by_stem=h.match_by_stem,
        )

    def _build_hashing_config(self) -> HashingConfig:
        """Constructs the Image Hashing settings."""
        h = self.settings.hashing

        # Construct active channels list
        channels = []
        if h.channel_r:
            channels.append("R")
        if h.channel_g:
            channels.append("G")
        if h.channel_b:
            channels.append("B")
        if h.channel_a:
            channels.append("A")

        # Default to all if none selected (UI logic safeguard)
        if not channels:
            channels = ["R", "G", "B", "A"]

        split_tags = [t.strip().lower() for t in h.channel_split_tags.split(",") if t.strip()]

        return HashingConfig(
            find_exact=h.find_exact,
            find_simple=h.find_simple,
            find_perceptual=h.find_perceptual,
            find_structural=h.find_structural,
            dhash_threshold=h.dhash_threshold,
            phash_threshold=h.phash_threshold,
            whash_threshold=h.whash_threshold,
            active_channels=channels,
            ignore_solid_channels=h.ignore_solid_channels,
            compare_by_luminance=h.compare_by_luminance,
            compare_by_channel=h.compare_by_channel,
            channel_split_tags=split_tags,
        )

    def _build_output_config(self) -> OutputConfig:
        """Constructs the Output/Visualization settings."""
        v = self.settings.visuals

        try:
            max_vis = int(v.max_count)
        except (ValueError, TypeError):
            max_vis = 100

        return OutputConfig(
            save_visuals=v.save,
            max_visuals=max_vis,
            visuals_columns=v.columns,
            tonemap_visuals=v.tonemap_enabled,
            tonemap_view=self.settings.viewer.tonemap_view,
        )
