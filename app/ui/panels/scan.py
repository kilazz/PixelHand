# app/ui/panels/scan.py
import logging
import webbrowser

from PySide6.QtCore import Slot
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.domain.data_models import AppSettings
from app.infrastructure.settings import SettingsManager
from app.shared.constants import VISUALS_DIR, UIConfig

app_logger = logging.getLogger("PixelHand.ui.panels.scan")


class ScanOptionsPanel(QGroupBox):
    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Scan and Output Options")
        self.settings_manager = settings_manager
        self._init_ui()
        self.load_settings(settings_manager.settings)
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.qc_mode_check = QCheckBox("Enable Quality Control (QC) Mode")
        self.qc_mode_check.setStyleSheet("color: #FF9800; font-weight: bold;")
        layout.addWidget(self.qc_mode_check)
        self.use_ai_check = QCheckBox("Use AI for similarity search")
        layout.addWidget(self.use_ai_check)

        self.hashing_options_group = QGroupBox("Duplicate Finding Methods")
        hashing_grid = QGridLayout(self.hashing_options_group)
        self.exact_duplicates_check = QCheckBox("Find exact duplicates (xxHash)")
        hashing_grid.addWidget(self.exact_duplicates_check, 0, 0, 1, 3)
        self.simple_duplicates_check = QCheckBox("Find simple duplicates (dHash)")
        self.dhash_threshold_spin = QSpinBox()
        self.dhash_threshold_spin.setRange(0, 64)
        hashing_grid.addWidget(self.simple_duplicates_check, 1, 0, 1, 2)
        hashing_grid.addWidget(self.dhash_threshold_spin, 1, 2)
        self.perceptual_duplicates_check = QCheckBox("Find near-identical images (pHash)")
        self.phash_threshold_spin = QSpinBox()
        self.phash_threshold_spin.setRange(0, 64)
        hashing_grid.addWidget(self.perceptual_duplicates_check, 2, 0, 1, 2)
        hashing_grid.addWidget(self.phash_threshold_spin, 2, 2)
        self.structural_duplicates_check = QCheckBox("Find structural duplicates (wHash)")
        self.whash_threshold_spin = QSpinBox()
        self.whash_threshold_spin.setRange(0, 64)
        hashing_grid.addWidget(self.structural_duplicates_check, 3, 0, 1, 2)
        hashing_grid.addWidget(self.whash_threshold_spin, 3, 2)

        self.image_prep_group = QGroupBox("Image Preparation Options")
        prep_layout = QVBoxLayout(self.image_prep_group)
        self.luminance_check = QCheckBox("Compare by luminance only (Grayscale)")
        prep_layout.addWidget(self.luminance_check)
        self.channel_group_widget = QWidget()
        channel_main_layout = QVBoxLayout(self.channel_group_widget)
        self.channel_check = QCheckBox("Compare by specific channels:")
        self.channels_layout = QHBoxLayout()
        self.cb_r = QCheckBox("R")
        self.cb_r.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.cb_g = QCheckBox("G")
        self.cb_g.setStyleSheet("color: #55FF55; font-weight: bold;")
        self.cb_b = QCheckBox("B")
        self.cb_b.setStyleSheet("color: #55AAFF; font-weight: bold;")
        self.cb_a = QCheckBox("A")
        self.cb_a.setStyleSheet("color: #AAAAAA; font-weight: bold;")
        self.channels_layout.addWidget(self.cb_r)
        self.channels_layout.addWidget(self.cb_g)
        self.channels_layout.addWidget(self.cb_b)
        self.channels_layout.addWidget(self.cb_a)
        self.channels_layout.addStretch()
        channel_main_layout.addWidget(self.channel_check)
        channel_main_layout.addLayout(self.channels_layout)
        prep_layout.addWidget(self.channel_group_widget)
        self.channel_tags_entry = QLineEdit()
        self.channel_tags_entry.setPlaceholderText("Filter by filename tag (e.g. _diff, _norm)...")
        channel_tags_layout = QHBoxLayout()
        channel_tags_layout.addWidget(QLabel("Filename Filter:"))
        channel_tags_layout.addWidget(self.channel_tags_entry)
        prep_layout.addLayout(channel_tags_layout)
        self.ignore_solid_check = QCheckBox("Ignore solid black/white channels")
        prep_layout.addWidget(self.ignore_solid_check)

        layout.addWidget(self.hashing_options_group)
        layout.addWidget(self.image_prep_group)

        self.visuals_layout = QHBoxLayout()
        self.save_visuals_check = QCheckBox("Save visuals")
        self.visuals_tonemap_check = QCheckBox("HDR")
        self.max_visuals_entry = QLineEdit()
        self.max_visuals_entry.setValidator(QIntValidator(0, 9999))
        self.max_visuals_entry.setFixedWidth(UIConfig.Sizes.MAX_VISUALS_ENTRY_WIDTH)
        self.visuals_columns_spinbox = QSpinBox()
        self.visuals_columns_spinbox.setRange(2, 12)
        self.visuals_columns_spinbox.setFixedWidth(UIConfig.Sizes.VISUALS_COLUMNS_SPINBOX_WIDTH)
        self.open_visuals_folder_button = QPushButton("📂")
        self.open_visuals_folder_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        self.visuals_layout.addWidget(self.save_visuals_check)
        self.visuals_layout.addWidget(self.visuals_tonemap_check)
        self.visuals_layout.addStretch()
        self.visuals_layout.addWidget(QLabel("Cols:"))
        self.visuals_layout.addWidget(self.visuals_columns_spinbox)
        self.visuals_layout.addWidget(QLabel("Max:"))
        self.visuals_layout.addWidget(self.max_visuals_entry)
        self.visuals_layout.addWidget(self.open_visuals_folder_button)
        layout.addLayout(self.visuals_layout)

    def _connect_signals(self):
        self.use_ai_check.toggled.connect(self.settings_manager.set_use_ai)
        self.exact_duplicates_check.toggled.connect(self.settings_manager.set_find_exact)
        self.simple_duplicates_check.toggled.connect(self.settings_manager.set_find_simple)
        self.perceptual_duplicates_check.toggled.connect(self.settings_manager.set_find_perceptual)
        self.structural_duplicates_check.toggled.connect(self.settings_manager.set_find_structural)
        self.use_ai_check.toggled.connect(self._update_dependent_ui_state)
        self.simple_duplicates_check.toggled.connect(self._update_dependent_ui_state)
        self.perceptual_duplicates_check.toggled.connect(self._update_dependent_ui_state)
        self.structural_duplicates_check.toggled.connect(self._update_dependent_ui_state)
        self.dhash_threshold_spin.valueChanged.connect(self.settings_manager.set_dhash_threshold)
        self.phash_threshold_spin.valueChanged.connect(self.settings_manager.set_phash_threshold)
        self.whash_threshold_spin.valueChanged.connect(self.settings_manager.set_whash_threshold)
        self.luminance_check.toggled.connect(self.settings_manager.set_compare_by_luminance)
        self.channel_check.toggled.connect(self.settings_manager.set_compare_by_channel)
        self.cb_r.toggled.connect(self.settings_manager.set_channel_r)
        self.cb_g.toggled.connect(self.settings_manager.set_channel_g)
        self.cb_b.toggled.connect(self.settings_manager.set_channel_b)
        self.cb_a.toggled.connect(self.settings_manager.set_channel_a)
        self.channel_tags_entry.textChanged.connect(self.settings_manager.set_channel_split_tags)
        self.ignore_solid_check.toggled.connect(self.settings_manager.set_ignore_solid_channels)
        self.channel_check.toggled.connect(self._update_dependent_ui_state)
        self.save_visuals_check.toggled.connect(self.settings_manager.set_save_visuals)
        self.visuals_tonemap_check.toggled.connect(self.settings_manager.set_visuals_tonemap)
        self.max_visuals_entry.textChanged.connect(self.settings_manager.set_max_visuals)
        self.visuals_columns_spinbox.valueChanged.connect(self.settings_manager.set_visuals_columns)
        self.save_visuals_check.toggled.connect(self.toggle_visuals_option)
        self.open_visuals_folder_button.clicked.connect(self._open_visuals_folder)

    @Slot()
    def _update_dependent_ui_state(self):
        is_ai_on = self.use_ai_check.isChecked()
        is_phash_any_on = (
            self.simple_duplicates_check.isChecked()
            or self.perceptual_duplicates_check.isChecked()
            or self.structural_duplicates_check.isChecked()
        )
        is_prep_enabled = is_ai_on or is_phash_any_on
        self.image_prep_group.setEnabled(is_prep_enabled)
        is_channel_check_on = self.channel_check.isChecked()
        for cb in [self.cb_r, self.cb_g, self.cb_b, self.cb_a]:
            cb.setEnabled(is_prep_enabled and is_channel_check_on)
        self.channel_tags_entry.setEnabled(is_prep_enabled and is_channel_check_on)
        self.ignore_solid_check.setEnabled(is_prep_enabled and is_channel_check_on)
        self.dhash_threshold_spin.setEnabled(self.simple_duplicates_check.isChecked())
        self.phash_threshold_spin.setEnabled(self.perceptual_duplicates_check.isChecked())
        self.whash_threshold_spin.setEnabled(self.structural_duplicates_check.isChecked())
        if self.window() and hasattr(self.window(), "options_panel"):
            self.window().performance_panel.setEnabled(is_ai_on)
            self.window().options_panel.model_combo.setEnabled(is_ai_on)
            self.window().options_panel.threshold_spinbox.setEnabled(is_ai_on)
            self.window().options_panel._update_scan_context()

    def toggle_visuals_option(self, is_checked):
        for i in range(2, self.visuals_layout.count()):
            item = self.visuals_layout.itemAt(i)
            if w := item.widget():
                w.setVisible(is_checked)

    def _open_visuals_folder(self):
        if not VISUALS_DIR.exists():
            QMessageBox.information(self, "Folder Not Found", "The visualizations folder does not exist yet.")
            return
        try:
            webbrowser.open(VISUALS_DIR.resolve().as_uri())
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")

    def load_settings(self, s: AppSettings):
        self.use_ai_check.setChecked(s.hashing.use_ai)
        self.exact_duplicates_check.setChecked(s.hashing.find_exact)
        self.simple_duplicates_check.setChecked(s.hashing.find_simple)
        self.perceptual_duplicates_check.setChecked(s.hashing.find_perceptual)
        self.structural_duplicates_check.setChecked(s.hashing.find_structural)
        self.dhash_threshold_spin.setValue(s.hashing.dhash_threshold)
        self.phash_threshold_spin.setValue(s.hashing.phash_threshold)
        self.whash_threshold_spin.setValue(s.hashing.whash_threshold)
        self.luminance_check.setChecked(s.hashing.compare_by_luminance)
        self.channel_check.setChecked(s.hashing.compare_by_channel)
        self.cb_r.setChecked(s.hashing.channel_r)
        self.cb_g.setChecked(s.hashing.channel_g)
        self.cb_b.setChecked(s.hashing.channel_b)
        self.cb_a.setChecked(s.hashing.channel_a)
        self.channel_tags_entry.setText(s.hashing.channel_split_tags)
        self.ignore_solid_check.setChecked(s.hashing.ignore_solid_channels)
        self.save_visuals_check.setChecked(s.visuals.save)
        self.visuals_tonemap_check.setChecked(s.visuals.tonemap_enabled)
        self.max_visuals_entry.setText(s.visuals.max_count)
        self.visuals_columns_spinbox.setValue(s.visuals.columns)
        self.toggle_visuals_option(s.visuals.save)
