# app/ui/panels/qc.py
import logging

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QWidget,
)

from app.domain.data_models import AppSettings, ScanMode
from app.infrastructure.settings import SettingsManager

app_logger = logging.getLogger("PixelHand.ui.panels.qc")


class QCPanel(QGroupBox):
    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Quality Control (QC) Options")
        self.settings_manager = settings_manager
        self.setEnabled(False)
        self._init_ui()
        self.load_settings(settings_manager.settings)
        self._connect_signals()

    def _init_ui(self):
        qc_grid = QGridLayout(self)
        qc_grid.setColumnStretch(0, 1)
        qc_grid.setColumnStretch(1, 1)

        self.hide_same_res_check = QCheckBox("Hide matches (Show diffs only)")
        self.hide_same_res_check.setToolTip("Ignore files where name and resolution match exactly.")
        self.hide_same_res_check.setStyleSheet("color: #FF9800; font-weight: bold;")

        self.match_stem_check = QCheckBox("Match by stem (Ignore ext)")
        self.match_stem_check.setToolTip("Matches 'file.tga' with 'file.dds'.")

        self.check_alpha_check = QCheckBox("Check Alpha Mismatch")
        self.check_compression_check = QCheckBox("Check Compression Change")
        self.check_bloat_check = QCheckBox("Check Size Bloat")
        self.check_colorspace_check = QCheckBox("Check Color Space")

        qc_grid.addWidget(self.hide_same_res_check, 0, 0)
        qc_grid.addWidget(self.match_stem_check, 1, 0)
        qc_grid.addWidget(self.check_alpha_check, 2, 0)
        qc_grid.addWidget(self.check_compression_check, 3, 0)
        qc_grid.addWidget(self.check_bloat_check, 4, 0)
        qc_grid.addWidget(self.check_colorspace_check, 5, 0)

        self.check_npot_check = QCheckBox("Check NPOT (Power of 2)")
        self.check_mipmaps_check = QCheckBox("Check Mip-Maps")
        self.check_align_check = QCheckBox("Check Block Alignment (4px)")
        self.check_bitdepth_check = QCheckBox("Check Bit Depth")
        self.check_solid_check = QCheckBox("Check Solid Color (Slow)")

        qc_grid.addWidget(self.check_npot_check, 0, 1)
        qc_grid.addWidget(self.check_mipmaps_check, 1, 1)
        qc_grid.addWidget(self.check_align_check, 2, 1)
        qc_grid.addWidget(self.check_bitdepth_check, 3, 1)
        qc_grid.addWidget(self.check_solid_check, 4, 1)

        # --- Normal Map Validation Widgets ---
        self.check_normals_check = QCheckBox("Validate Normal Maps")
        self.check_normals_check.setToolTip("Checks if vectors are normalized (length ~1.0).")

        self.normals_tags_entry = QLineEdit()
        self.normals_tags_entry.setPlaceholderText("Tags: _norm, _ddn (empty = check all)")
        self.normals_tags_entry.setToolTip("If empty, checks ALL files. If set, checks only matching files.")

        # Place them side-by-side in row 6
        qc_grid.addWidget(self.check_normals_check, 6, 0)
        qc_grid.addWidget(self.normals_tags_entry, 6, 1)

        self.relative_widgets = [
            self.hide_same_res_check,
            self.match_stem_check,
            self.check_alpha_check,
            self.check_compression_check,
            self.check_bloat_check,
            self.check_colorspace_check,
        ]
        self.absolute_widgets = [
            self.check_npot_check,
            self.check_mipmaps_check,
            self.check_align_check,
            self.check_bitdepth_check,
            self.check_solid_check,
            self.check_normals_check,
            self.normals_tags_entry,
        ]

    def set_mode_context(self, mode_name: str):
        is_single_qc = mode_name == ScanMode.SINGLE_FOLDER_QC.name
        is_compare = mode_name == ScanMode.FOLDER_COMPARE.name

        if not (is_single_qc or is_compare):
            self.setEnabled(False)
            return

        self.setEnabled(True)
        for w in self.absolute_widgets:
            w.setEnabled(True)
            if isinstance(w, QWidget):  # Ensure it's a widget before styling
                w.setStyleSheet("")

        # The tags entry is only enabled if the validation check is active
        self.normals_tags_entry.setEnabled(self.check_normals_check.isChecked())

        for w in self.relative_widgets:
            w.setEnabled(is_compare)
            if not is_compare:
                w.setStyleSheet("color: gray;")
            else:
                w.setStyleSheet("color: #FF9800; font-weight: bold;" if w == self.hide_same_res_check else "")

    def _connect_signals(self):
        self.hide_same_res_check.toggled.connect(self.settings_manager.set_hide_same_resolution_groups)
        self.match_stem_check.toggled.connect(self.settings_manager.set_match_by_stem)
        self.check_alpha_check.toggled.connect(self.settings_manager.set_qc_check_alpha)
        self.check_npot_check.toggled.connect(self.settings_manager.set_qc_check_npot)
        self.check_mipmaps_check.toggled.connect(self.settings_manager.set_qc_check_mipmaps)
        self.check_bloat_check.toggled.connect(self.settings_manager.set_qc_check_size_bloat)
        self.check_solid_check.toggled.connect(self.settings_manager.set_qc_check_solid_color)
        self.check_colorspace_check.toggled.connect(self.settings_manager.set_qc_check_color_space)
        self.check_bitdepth_check.toggled.connect(self.settings_manager.set_qc_check_bit_depth)
        self.check_compression_check.toggled.connect(self.settings_manager.set_qc_check_compression)
        self.check_align_check.toggled.connect(self.settings_manager.set_qc_check_block_align)

        # Normal Map Signals
        self.check_normals_check.toggled.connect(self.settings_manager.set_qc_check_normal_maps)
        self.check_normals_check.toggled.connect(self.normals_tags_entry.setEnabled)
        self.normals_tags_entry.textChanged.connect(self.settings_manager.set_qc_normal_maps_tags)

    def load_settings(self, s: AppSettings):
        self.hide_same_res_check.setChecked(s.hashing.hide_same_resolution_groups)
        self.match_stem_check.setChecked(s.hashing.match_by_stem)
        self.check_alpha_check.setChecked(s.hashing.qc_check_alpha)
        self.check_npot_check.setChecked(s.hashing.qc_check_npot)
        self.check_mipmaps_check.setChecked(s.hashing.qc_check_mipmaps)
        self.check_bloat_check.setChecked(s.hashing.qc_check_size_bloat)
        self.check_solid_check.setChecked(s.hashing.qc_check_solid_color)
        self.check_colorspace_check.setChecked(s.hashing.qc_check_color_space)
        self.check_bitdepth_check.setChecked(s.hashing.qc_check_bit_depth)
        self.check_compression_check.setChecked(s.hashing.qc_check_compression)
        self.check_align_check.setChecked(s.hashing.qc_check_block_align)

        # Normal Map Settings
        self.check_normals_check.setChecked(s.hashing.qc_check_normal_maps)
        self.normals_tags_entry.setText(s.hashing.qc_normal_maps_tags)
        self.normals_tags_entry.setEnabled(s.hashing.qc_check_normal_maps)
