# app/ui/options.py

import logging
import multiprocessing
import webbrowser
from pathlib import Path

import onnxruntime
from PySide6.QtCore import QPoint, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.domain.data_models import AppSettings, ScanMode
from app.infrastructure.settings import SettingsManager
from app.shared.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    DEEP_LEARNING_AVAILABLE,
    DEFAULT_SEARCH_PRECISION,
    ROOT_DIR,
    SEARCH_PRECISION_PRESETS,
    SUPPORTED_MODELS,
    VISUALS_DIR,
    QuantizationMode,
    UIConfig,
)
from app.ui.dialogs import FileTypesDialog

app_logger = logging.getLogger("PixelHand.ui.options")


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
            w.setStyleSheet("")

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


class OptionsPanel(QGroupBox):
    scan_requested = Signal()
    clear_scan_cache_requested = Signal()
    clear_models_cache_requested = Signal()
    clear_all_data_requested = Signal()
    log_message = Signal(str, str)
    scan_context_changed = Signal(str)
    qc_mode_toggled = Signal(bool)

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Scan Configuration")
        self.settings_manager = settings_manager
        settings = settings_manager.settings
        self.selected_extensions = list(settings.selected_extensions)
        self.current_scan_mode = ScanMode.DUPLICATES
        self._sample_path: Path | None = None
        self._init_ui()
        self._connect_signals()
        self.load_settings(settings)
        self._on_model_changed()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form_layout.setSpacing(8)

        self._create_path_widgets()
        self._create_search_widgets()
        self._create_core_scan_widgets()
        main_layout.addLayout(self.form_layout)

        self.theme_menu = self._create_theme_menu()
        self._create_action_buttons(main_layout)

    def _create_path_widgets(self):
        self.folder_path_entry = QLineEdit()
        self.browse_folder_button = QPushButton("...")
        self.browse_folder_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.folder_path_entry)
        folder_layout.addWidget(self.browse_folder_button)
        self.form_layout.addRow("Folder A:", folder_layout)

        self.folder_b_entry = QLineEdit()
        self.folder_b_entry.setPlaceholderText("Select second folder to compare resolutions...")
        self.browse_folder_b_button = QPushButton("...")
        self.browse_folder_b_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        folder_b_layout = QHBoxLayout()
        folder_b_layout.setContentsMargins(0, 0, 0, 0)
        folder_b_layout.addWidget(self.folder_b_entry)
        folder_b_layout.addWidget(self.browse_folder_b_button)
        self.folder_b_label = QLabel("Folder B:")
        self.folder_b_container = QWidget()
        self.folder_b_container.setLayout(folder_b_layout)
        self.folder_b_label.setVisible(False)
        self.folder_b_container.setVisible(False)
        self.form_layout.addRow(self.folder_b_label, self.folder_b_container)

    def _create_search_widgets(self):
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Enter text to search, or leave blank for duplicates...")
        self.browse_sample_button = QPushButton("🖼️")
        self.browse_sample_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_entry)
        search_layout.addWidget(self.browse_sample_button)
        self.form_layout.addRow("Find:", search_layout)
        self.sample_path_label = QLabel("Sample: None")
        self.sample_path_label.setStyleSheet("font-style: italic; color: #9E9E9E;")
        self.clear_sample_button = QPushButton("❌")
        self.clear_sample_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        sample_layout = QHBoxLayout()
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(self.sample_path_label)
        sample_layout.addStretch()
        sample_layout.addWidget(self.clear_sample_button)
        self.form_layout.addRow("", sample_layout)

    def _create_core_scan_widgets(self):
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 100)
        self.threshold_spinbox.setSuffix("%")
        self.form_layout.addRow("Similarity:", self.threshold_spinbox)
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_MODELS.keys())
        self.form_layout.addRow("Model:", self.model_combo)
        self.exclude_entry = QLineEdit()
        self.exclude_entry.setPlaceholderText("e.g., .cache, previews, temp_files")
        self.form_layout.addRow("Exclude Folders:", self.exclude_entry)

    def _create_action_buttons(self, main_layout: QVBoxLayout):
        top_action_layout = QHBoxLayout()
        top_action_layout.setSpacing(5)
        self.file_types_button = QPushButton("File Types...")
        self.clear_scan_cache_button = QPushButton("Clear Scan Cache")
        self.clear_models_cache_button = QPushButton("Clear AI Models")
        # RESTORED ID
        self.clear_models_cache_button.setObjectName("clear_models_button")

        self.clear_all_data_button = QPushButton("Clear All Data")
        # RESTORED ID
        self.clear_all_data_button.setObjectName("clear_all_data_button")

        self.theme_button = QPushButton("🎨")
        self.theme_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        top_action_layout.addWidget(self.file_types_button)
        top_action_layout.addWidget(self.clear_scan_cache_button)
        top_action_layout.addWidget(self.clear_models_cache_button)
        top_action_layout.addWidget(self.clear_all_data_button)
        top_action_layout.addStretch()
        top_action_layout.addWidget(self.theme_button)
        main_layout.addLayout(top_action_layout)

        self.scan_button = QPushButton("Scan for Duplicates")
        # RESTORED ID
        self.scan_button.setObjectName("scan_button")
        main_layout.addWidget(self.scan_button)

    def _connect_signals(self):
        self.folder_path_entry.textChanged.connect(self.settings_manager.set_folder_path)
        self.threshold_spinbox.valueChanged.connect(self.settings_manager.set_threshold)
        self.exclude_entry.textChanged.connect(self.settings_manager.set_exclude_folders)
        self.model_combo.currentTextChanged.connect(self.settings_manager.set_model_key)
        self.browse_folder_button.clicked.connect(self._browse_for_folder)
        self.browse_folder_b_button.clicked.connect(self._browse_for_folder_b)
        self.browse_sample_button.clicked.connect(self._browse_for_sample)
        self.clear_sample_button.clicked.connect(self._clear_sample)
        self.clear_sample_button.clicked.connect(self._update_scan_context)
        self.search_entry.textChanged.connect(self._update_scan_context)
        self.folder_b_entry.textChanged.connect(self._update_scan_context)
        self.scan_button.clicked.connect(self.on_scan_button_clicked)
        self.clear_scan_cache_button.clicked.connect(self.clear_scan_cache_requested.emit)
        self.clear_models_cache_button.clicked.connect(self.clear_models_cache_requested.emit)
        self.clear_all_data_button.clicked.connect(self.clear_all_data_requested.emit)
        self.file_types_button.clicked.connect(self._open_file_types_dialog)
        self.theme_button.clicked.connect(self._show_theme_menu)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

    def _create_theme_menu(self) -> QMenu:
        theme_menu = QMenu(self)
        theme_action_group = QActionGroup(self)
        theme_action_group.setExclusive(True)
        styles_dir = ROOT_DIR / "app/ui/styles"
        if styles_dir.is_dir():
            for theme_dir in sorted(p for p in styles_dir.iterdir() if p.is_dir()):
                theme_id = theme_dir.name
                if (theme_dir / f"{theme_id}.qss").is_file():
                    theme_name = theme_id.replace("_", " ").title()
                    action = QAction(theme_name, self, checkable=True)
                    action.triggered.connect(lambda checked, t_id=theme_id: self.window().load_theme(theme_id=t_id))
                    theme_menu.addAction(action)
                    theme_action_group.addAction(action)
        current_theme = getattr(self.settings_manager.settings, "theme", "Dark")
        for action in theme_action_group.actions():
            if action.text() == current_theme:
                action.setChecked(True)
                break
        return theme_menu

    @Slot()
    def _show_theme_menu(self):
        self.theme_menu.exec(self.theme_button.mapToGlobal(QPoint(0, self.theme_button.height())))

    @Slot()
    def _update_scan_context(self):
        is_ai_on = self.settings_manager.settings.hashing.use_ai
        model_info = self.get_selected_model_info()
        supports_text = model_info.get("supports_text_search", True)
        supports_image = model_info.get("supports_image_search", True)
        has_folder_b = bool(self.folder_b_entry.text().strip())
        has_sample = bool(self._sample_path)
        has_text_query = bool(self.search_entry.text().strip())
        is_qc_mode_checked = False
        if self.window() and hasattr(self.window(), "scan_options_panel"):
            is_qc_mode_checked = self.window().scan_options_panel.qc_mode_check.isChecked()

        self.folder_b_label.setVisible(is_qc_mode_checked)
        self.folder_b_container.setVisible(is_qc_mode_checked)

        if is_qc_mode_checked:
            if has_folder_b:
                self.current_scan_mode = ScanMode.FOLDER_COMPARE
                self.scan_button_text = "Run QC Compare"
            else:
                self.current_scan_mode = ScanMode.SINGLE_FOLDER_QC
                self.scan_button_text = "Run QC Scan"
            self.search_entry.setEnabled(False)
            self.browse_sample_button.setEnabled(False)
            self.clear_sample_button.setEnabled(False)
            self.threshold_spinbox.setEnabled(False)
            self.qc_mode_toggled.emit(True)
        elif is_ai_on and has_sample and supports_image:
            self.current_scan_mode = ScanMode.SAMPLE_SEARCH
            self.scan_button_text = "Search by Sample"
            self.search_entry.setEnabled(True)
            self.browse_sample_button.setEnabled(True)
            self.clear_sample_button.setVisible(True)
            self.threshold_spinbox.setEnabled(True)
            self.qc_mode_toggled.emit(False)
        elif is_ai_on and has_text_query and supports_text:
            self.current_scan_mode = ScanMode.TEXT_SEARCH
            self.scan_button_text = "Search by Text"
            self.search_entry.setEnabled(True)
            self.browse_sample_button.setEnabled(True)
            self.clear_sample_button.setVisible(False)
            self.threshold_spinbox.setEnabled(True)
            self.qc_mode_toggled.emit(False)
        else:
            self.current_scan_mode = ScanMode.DUPLICATES
            self.scan_button_text = "Scan for Duplicates"
            self.search_entry.setEnabled(True)
            self.browse_sample_button.setEnabled(is_ai_on and supports_image)
            self.clear_sample_button.setVisible(False)
            self.threshold_spinbox.setEnabled(is_ai_on)
            self.qc_mode_toggled.emit(False)

        if not is_ai_on:
            self.search_entry.setPlaceholderText("AI is disabled")
            self.search_entry.setEnabled(False)
        elif not supports_text and not is_qc_mode_checked:
            self.search_entry.setPlaceholderText("Text search not supported by this model")
        elif not is_qc_mode_checked:
            self.search_entry.setPlaceholderText("Enter text to search...")
        else:
            self.search_entry.setPlaceholderText("Search disabled in QC mode")

        if not (is_ai_on and supports_image) and not is_qc_mode_checked:
            self._clear_sample()
        is_ai_search = self.current_scan_mode in (ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH)
        label = self.form_layout.labelForField(self.threshold_spinbox)
        if label:
            label.setText("Similarity:" if not is_ai_search else "Similarity (N/A):")
        if is_ai_search or not is_ai_on:
            self.threshold_spinbox.setValue(0)

        self.scan_button.setText(self.scan_button_text)
        self.scan_context_changed.emit(self.current_scan_mode.name)
        if self.window() and hasattr(self.window(), "qc_panel"):
            self.window().qc_panel.set_mode_context(self.current_scan_mode.name)

    @Slot()
    def _on_model_changed(self):
        self._update_scan_context()

    def _browse_for_folder(self):
        path = self._browse_generic(self.folder_path_entry.text())
        if path:
            self.folder_path_entry.setText(path)

    def _browse_for_folder_b(self):
        path = self._browse_generic(self.folder_b_entry.text())
        if path:
            self.folder_b_entry.setText(path)

    def _browse_generic(self, start_path: str) -> str:
        try:
            return QFileDialog.getExistingDirectory(self, "Select Folder", start_path)
        except Exception:
            return QFileDialog.getExistingDirectory(
                self, "Select Folder", start_path, options=QFileDialog.Option.DontUseNativeDialog
            )

    def _browse_for_sample(self):
        path_str, _ = "", ""
        try:
            path_str, _ = QFileDialog.getOpenFileName(
                self,
                "Select Sample Image",
                self.folder_path_entry.text(),
                f"Images ({' '.join(['*' + e for e in ALL_SUPPORTED_EXTENSIONS])})",
            )
        except Exception:
            path_str, _ = QFileDialog.getOpenFileName(
                self,
                "Select Sample Image",
                self.folder_path_entry.text(),
                f"Images ({' '.join(['*' + e for e in ALL_SUPPORTED_EXTENSIONS])})",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
        if path_str:
            self._sample_path = Path(path_str)
            self.search_entry.clear()
            self.folder_b_entry.clear()
            self.sample_path_label.setText(f"Sample: {self._sample_path.name}")
            self._update_scan_context()

    @Slot()
    def _clear_sample(self):
        self._sample_path = None
        self.sample_path_label.setText("Sample: None")

    def set_scan_button_state(self, is_scanning: bool):
        self.scan_button.setText("Cancel Scan" if is_scanning else getattr(self, "scan_button_text", "Scan"))
        self.scan_button.setEnabled(True)

    @Slot()
    def on_scan_button_clicked(self):
        if "Cancel" in self.scan_button.text():
            if "Cancelling" not in self.scan_button.text():
                self.scan_button.setText("Cancelling...")
                if self.window():
                    self.window().controller.cancel_scan()
        else:
            self.scan_requested.emit()

    def _open_file_types_dialog(self):
        dialog = FileTypesDialog(self.selected_extensions, self)
        if dialog.exec():
            self.selected_extensions = dialog.get_selected_extensions()
            self.log_message.emit(f"Selected {len(self.selected_extensions)} file type(s).", "info")
            self.settings_manager.set_selected_extensions(self.selected_extensions)

    def get_selected_model_info(self) -> dict:
        return SUPPORTED_MODELS.get(self.model_combo.currentText(), next(iter(SUPPORTED_MODELS.values())))

    def load_settings(self, s: AppSettings):
        self.folder_path_entry.setText(s.folder_path)
        self.threshold_spinbox.setValue(int(s.threshold))
        self.exclude_entry.setText(s.exclude)
        self.model_combo.setCurrentText(s.model_key)
        self.selected_extensions = list(s.selected_extensions)


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
        self.visuals_tonemap_check = QCheckBox("TM HDR")
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


class PerformancePanel(QGroupBox):
    log_message = Signal(str, str)
    device_changed = Signal(bool)

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Performance and AI Model")
        self.settings_manager = settings_manager
        self._init_ui()
        self._detect_and_setup_devices()
        self._connect_signals()
        self.load_settings(settings_manager.settings)

    def _init_ui(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.device_combo = QComboBox()
        layout.addRow("Device:", self.device_combo)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems([q.value for q in QuantizationMode])
        layout.addRow("Model Precision:", self.quant_combo)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 4096)
        self.batch_size_spin.setSingleStep(16)
        layout.addRow("Batch Size:", self.batch_size_spin)
        self.search_precision_combo = QComboBox()
        layout.addRow("Search Precision:", self.search_precision_combo)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, (multiprocessing.cpu_count() or 1) * 4)
        layout.addRow("Preprocessing Workers:", self.num_workers_spin)

    def _connect_signals(self):
        self.device_combo.activated.connect(self._on_device_selection_changed)
        self.quant_combo.currentTextChanged.connect(self.settings_manager.set_quantization_mode)
        self.batch_size_spin.valueChanged.connect(self.settings_manager.set_batch_size)
        self.search_precision_combo.currentTextChanged.connect(self.settings_manager.set_search_precision)
        self.num_workers_spin.valueChanged.connect(self.settings_manager.set_num_workers)

    def _detect_and_setup_devices(self):
        self.device_combo.clear()
        if not DEEP_LEARNING_AVAILABLE:
            self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
            return
        try:
            providers = onnxruntime.get_available_providers()
            if "CPUExecutionProvider" in providers:
                self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
            for pid in providers:
                if pid != "CPUExecutionProvider":
                    self.device_combo.addItem(pid, pid)
        except Exception as e:
            self.log_message.emit(f"Error detecting ONNX providers: {e}", "error")
            if self.device_combo.count() == 0:
                self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
        self._on_device_change(self.device_combo.currentData())

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        if pid := self.device_combo.itemData(index):
            self.settings_manager.set_device(pid)
            self._on_device_change(pid)

    @Slot(str)
    def _on_device_change(self, provider_id: str):
        self.device_changed.emit(provider_id == "CPUExecutionProvider")

    @Slot(str)
    def update_precision_presets(self, scan_mode_name: str):
        self.search_precision_combo.blockSignals(True)
        self.search_precision_combo.clear()
        presets = list(SEARCH_PRECISION_PRESETS.keys())
        self.search_precision_combo.addItems(presets)
        current = self.settings_manager.settings.performance.search_precision
        self.search_precision_combo.setCurrentText(current if current in presets else DEFAULT_SEARCH_PRECISION)
        self.search_precision_combo.blockSignals(False)

    def load_settings(self, s: AppSettings):
        self._detect_and_setup_devices()
        idx = self.device_combo.findData(s.performance.device)
        self.device_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.quant_combo.setCurrentText(s.performance.quantization_mode)
        self.batch_size_spin.setValue(int(s.performance.batch_size))
        self.search_precision_combo.setCurrentText(s.performance.search_precision)
        self.num_workers_spin.setValue(int(s.performance.num_workers))
        self._on_device_change(self.device_combo.currentData())

    def get_selected_quantization(self) -> QuantizationMode:
        return next((q for q in QuantizationMode if q.value == self.quant_combo.currentText()), QuantizationMode.FP16)
