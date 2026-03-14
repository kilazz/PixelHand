# app/ui/panels/general.py
import logging
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.domain.data_models import AppSettings, ScanMode
from app.infrastructure.settings import SettingsManager
from app.shared.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    ROOT_DIR,
    SUPPORTED_MODELS,
    UIConfig,
)
from app.ui.dialogs import FileTypesDialog
from app.ui.widgets import DragDropLabel, DragDropLineEdit

app_logger = logging.getLogger("PixelHand.ui.panels.general")


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
        # Use custom DragDropLineEdit for folder paths
        self.folder_path_entry = DragDropLineEdit()
        self.browse_folder_button = QPushButton("...")
        self.browse_folder_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.folder_path_entry)
        folder_layout.addWidget(self.browse_folder_button)
        self.form_layout.addRow("Folder A:", folder_layout)

        self.folder_b_entry = DragDropLineEdit()
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

        # Use DragDropLabel for the sample image area
        self.sample_path_label = DragDropLabel("Sample: None")
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
        self.clear_models_cache_button.setObjectName("clear_models_button")

        self.clear_all_data_button = QPushButton("Clear All Data")
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

        # Handle drop on sample label
        self.sample_path_label.file_dropped.connect(self._on_sample_dropped)

        # Connect clear sample directly to the robust logic
        self.clear_sample_button.clicked.connect(self._clear_sample)

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
            self.clear_sample_button.setEnabled(True)  # Ensure it is re-enabled
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

        if not (is_ai_on and supports_image) and not is_qc_mode_checked and has_sample:
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
            self._set_sample(path_str)

    @Slot(str)
    def _on_sample_dropped(self, path: str):
        self._set_sample(path)

    def _set_sample(self, path_str: str):
        self._sample_path = Path(path_str)
        self.search_entry.clear()
        self.folder_b_entry.clear()
        self.sample_path_label.setText(f"Sample: {self._sample_path.name}")
        self._update_scan_context()

    @Slot()
    def _clear_sample(self):
        self._sample_path = None
        self.sample_path_label.setText("Sample: None")
        self.search_entry.setEnabled(True)  # Ensure search entry is re-enabled if it was disabled
        self._update_scan_context()

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
