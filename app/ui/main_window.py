# app/ui/main_window.py
"""
Main Application Window.
"""

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.domain.config import ScanConfig
from app.domain.data_models import FileOperation, ScanMode
from app.infrastructure.configuration import ScanConfigBuilder
from app.infrastructure.container import ServiceContainer
from app.infrastructure.settings import SettingsManager
from app.shared.constants import (
    DEEP_LEARNING_AVAILABLE,
    ROOT_DIR,
    VISUALS_DIR,
)
from app.shared.signal_bus import APP_SIGNAL_BUS
from app.shared.utils import (
    check_link_support,
    clear_all_app_data,
    clear_models_cache,
    clear_scan_cache,
    is_onnx_model_cached,
)

# Controllers & Tasks
from app.ui.controllers import ResultsController

# Dialogs
from app.ui.dialogs import (
    ModelConversionDialog,
    ScanStatisticsDialog,
    SkippedFilesDialog,
)

# Panels
from app.ui.options import (
    OptionsPanel,
    PerformancePanel,
    QCPanel,
    ScanOptionsPanel,
)
from app.ui.results import ResultsPanel
from app.ui.status import LogPanel, SystemStatusPanel
from app.ui.viewer import ImageViewerPanel
from app.workflow.auxiliary import VisualizationTask
from app.workflow.scanner import ScannerController

logger = logging.getLogger("PixelHand.ui.main")


class App(QMainWindow):
    """
    The main window of the PixelHand application.
    Acts as the Composition Root for UI components.
    """

    def __init__(self, services: ServiceContainer):
        """
        Args:
            services: The DI container holding core infrastructure (DB, AI, Tasks).
        """
        super().__init__()
        self.setWindowTitle("PixelHand")
        self.setGeometry(100, 100, 1600, 900)
        self.setStatusBar(QStatusBar(self))

        # 1. Store Dependencies
        self.services = services

        # 2. Initialize Logic Components
        self.settings_manager = SettingsManager(self)
        self.scanner_controller = ScannerController(self.services)
        self.results_controller = ResultsController(self.services)

        self.stats_dialog: ScanStatisticsDialog | None = None

        # 3. Setup UI
        self._setup_ui()
        self._create_menu_bar()
        self._connect_signals()

        # 4. Initial State Setup
        self.scan_options_panel._update_dependent_ui_state()
        self.options_panel._update_scan_context()
        self._log_system_status()
        self._apply_initial_theme()

    def _setup_ui(self):
        SPACING = 6

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QHBoxLayout(self.main_widget)
        main_layout.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        main_layout.setSpacing(0)

        # --- Left Splitter (Options / Log) ---
        self.left_v_splitter = QSplitter(Qt.Orientation.Vertical)

        self.top_left_container = QWidget()
        top_left_layout = QVBoxLayout(self.top_left_container)
        top_left_layout.setSpacing(SPACING)
        top_left_layout.setContentsMargins(SPACING, SPACING, SPACING, SPACING)

        # Instantiate Configuration Panels
        self.options_panel = OptionsPanel(self.settings_manager)
        self.qc_panel = QCPanel(self.settings_manager)
        self.scan_options_panel = ScanOptionsPanel(self.settings_manager)
        self.performance_panel = PerformancePanel(self.settings_manager)
        self.system_status_panel = SystemStatusPanel()

        top_left_layout.addWidget(self.options_panel)
        top_left_layout.addWidget(self.qc_panel)
        top_left_layout.addWidget(self.scan_options_panel)
        top_left_layout.addWidget(self.performance_panel)
        top_left_layout.addWidget(self.system_status_panel)
        top_left_layout.addStretch(1)

        self.log_container = QWidget()
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_panel = LogPanel()
        log_layout.addWidget(self.log_panel)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setWidget(self.top_left_container)

        self.left_v_splitter.addWidget(self.settings_scroll)
        self.left_v_splitter.addWidget(self.log_container)
        self.left_v_splitter.setStretchFactor(0, 0)
        self.left_v_splitter.setStretchFactor(1, 1)

        # --- Right Splitter (Results / Viewer) ---
        self.results_viewer_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Instantiate View Panels
        # Note: We pass the results_controller (formerly file_op_manager) to handle file actions
        self.results_panel = ResultsPanel(self.results_controller)

        # ImageViewerPanel needs QThreadPool for image loading tasks.
        # We also pass results_controller to handle context menu deletions.
        self.viewer_panel = ImageViewerPanel(
            self.settings_manager,
            QThreadPool.globalInstance(),
            self.results_controller,
        )

        self.results_viewer_splitter.addWidget(self.results_panel)
        self.viewer_panel_container = QWidget()
        viewer_layout = QVBoxLayout(self.viewer_panel_container)
        viewer_layout.setContentsMargins(SPACING, 0, 0, 0)
        viewer_layout.addWidget(self.viewer_panel)
        self.results_viewer_splitter.addWidget(self.viewer_panel_container)

        # --- Main Splitter ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_v_splitter)
        self.main_splitter.addWidget(self.results_viewer_splitter)

        main_layout.addWidget(self.main_splitter)

        # Initial Sizes
        self.main_splitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.75)])
        self.results_viewer_splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.35)])

    def _create_menu_bar(self):
        self.menuBar().setVisible(False)

    def _apply_initial_theme(self):
        theme_name = self.settings_manager.settings.theme
        for action in self.options_panel.theme_menu.actions():
            if action.text() == theme_name:
                action.setChecked(True)
                action.trigger()
                return
        if self.options_panel.theme_menu.actions():
            self.options_panel.theme_menu.actions()[0].trigger()

    def load_theme(self, theme_id: str):
        qss_file = ROOT_DIR / "app/ui/styles" / theme_id / f"{theme_id}.qss"
        if not qss_file.is_file():
            APP_SIGNAL_BUS.log_message.emit(f"Error: Theme file not found at '{qss_file}'", "error")
            return
        try:
            with open(qss_file, encoding="utf-8") as f:
                qss_content = f.read()
            self.setProperty("searchPaths", f"file:///{qss_file.parent.as_posix()}")
            self.apply_qss_string(qss_content, theme_id.replace("_", " ").title())
        except OSError as e:
            APP_SIGNAL_BUS.log_message.emit(f"Error loading theme '{theme_id}': {e}", "error")

    def apply_qss_string(self, qss: str, theme_name: str):
        if app := QApplication.instance():
            app.setStyleSheet(qss)
        if self.settings_manager.settings.theme != theme_name:
            self.settings_manager.settings.theme = theme_name
            self.settings_manager.save()

    def _connect_signals(self):
        # Global Event Bus
        APP_SIGNAL_BUS.scan_finished.connect(self.on_scan_complete)
        APP_SIGNAL_BUS.scan_error.connect(self.on_scan_error)
        APP_SIGNAL_BUS.log_message.connect(self.log_panel.log_message)

        # Locking Logic
        APP_SIGNAL_BUS.lock_ui.connect(lambda: self.set_ui_scan_state(is_busy=True))
        APP_SIGNAL_BUS.unlock_ui.connect(lambda: self.set_ui_scan_state(is_busy=False))

        # Status Bar
        APP_SIGNAL_BUS.status_message_updated.connect(self.statusBar().showMessage)

        # Options Panel Interactions
        self.options_panel.scan_requested.connect(self._start_scan)
        self.options_panel.clear_scan_cache_requested.connect(self._clear_scan_cache)
        self.options_panel.clear_models_cache_requested.connect(self._clear_models_cache)
        self.options_panel.clear_all_data_requested.connect(self._clear_app_data)
        self.options_panel.scan_context_changed.connect(self.performance_panel.update_precision_presets)
        self.options_panel.qc_mode_toggled.connect(self.qc_panel.setEnabled)

        self.scan_options_panel.qc_mode_check.toggled.connect(self.options_panel._update_scan_context)
        self.scan_options_panel.use_ai_check.toggled.connect(self.options_panel._update_scan_context)

        # Results & Viewer Interactions
        # When user clicks a row in results, update the viewer
        self.results_panel.results_view.selectionModel().selectionChanged.connect(self._on_results_selection_changed)

        # When results are filtered, update viewer items
        self.results_panel.visible_results_changed.connect(self.viewer_panel.display_results)

        # When fetching lazy data completes, check if selection needs update
        self.results_panel.results_model.fetch_completed.connect(self._on_group_fetch_finished)

        # Viewer feedback loops
        self.viewer_panel.group_became_empty.connect(self.results_panel.results_model.remove_group_by_id)
        self.viewer_panel.group_became_empty.connect(self.results_panel._update_summary)
        self.viewer_panel.file_missing_detected.connect(self._on_external_file_missing)

        # Results Controller Signals (File Operations)
        self.results_controller.operation_started.connect(self._on_file_op_started)
        self.results_controller.operation_finished.connect(self._on_file_op_finished)
        self.results_controller.status_message.connect(self._on_controller_status)
        self.results_controller.error_occurred.connect(self._on_controller_error)

    def _log_system_status(self):
        logger.info("Application initialized. System capabilities checked.")

    # --- Scan Lifecycle ---

    @Slot()
    def _start_scan(self):
        if self.scanner_controller.is_running():
            return

        config = self._get_config()
        if not config:
            return

        # Auto-cleanup if model changed (legacy logic, could be moved to controller)
        last_model = self.settings_manager.settings.hashing.last_model_name
        current_model = config.ai.model_name

        if last_model and last_model != current_model:
            APP_SIGNAL_BUS.log_message.emit("Model changed. Clearing cache...", "warning")
            clear_scan_cache()
            self.results_panel.clear_results()
            self.viewer_panel.clear_viewer()
            self.settings_manager.settings.hashing.last_model_name = current_model
            self.settings_manager.save()

        # Check/Convert Model
        if (
            config.ai.use_ai
            and DEEP_LEARNING_AVAILABLE
            and not is_onnx_model_cached(config.ai.model_name)
            and not self._run_model_conversion(config)
        ):
            return

        # Prepare UI
        self.results_panel.clear_results()
        self.viewer_panel.clear_viewer()
        self.log_panel.clear()
        self.set_ui_scan_state(is_busy=True)

        logger.info(f"Starting scan in '{config.folder_path}'")

        # Show Progress Dialog
        self.stats_dialog = ScanStatisticsDialog(self.scanner_controller.scan_state, APP_SIGNAL_BUS, self)
        self.stats_dialog.show()

        # Trigger Controller
        APP_SIGNAL_BUS.scan_requested.emit(config)

    def _get_config(self) -> ScanConfig | None:
        try:
            sample_path = self.options_panel._sample_path
            comp_path_str = self.options_panel.folder_b_entry.text().strip()
            comp_path = Path(comp_path_str) if comp_path_str else None

            builder = ScanConfigBuilder(
                settings=self.settings_manager.settings,
                scan_mode=self.options_panel.current_scan_mode,
                search_query=self.options_panel.search_entry.text(),
                sample_path=sample_path,
                comparison_folder_path=comp_path,
            )
            return builder.build()
        except ValueError as e:
            APP_SIGNAL_BUS.log_message.emit(f"Configuration Error: {e}", "error")
            self.on_scan_end()
            return None

    def _run_model_conversion(self, config: ScanConfig) -> bool:
        # Get metadata for UI dialog
        model_key = self.options_panel.model_combo.currentText()
        model_info = self.options_panel.get_selected_model_info()

        # Map string to Enum for dialog (legacy compatibility)
        from app.shared.constants import QuantizationMode

        quant_str = self.performance_panel.get_selected_quantization().value
        quant_mode = next(q for q in QuantizationMode if q.value == quant_str)

        dialog = ModelConversionDialog(
            model_key,
            model_info["hf_name"],
            model_info["onnx_name"],
            quant_mode,
            model_info,
            self,
        )
        return bool(dialog.exec())

    def set_ui_scan_state(self, is_busy: bool):
        """Locks/Unlocks UI elements during operations."""
        for panel in [
            self.options_panel,
            self.qc_panel,
            self.performance_panel,
            self.scan_options_panel,
            self.system_status_panel,
            self.results_panel,
            self.viewer_panel,
        ]:
            panel.setEnabled(not is_busy)

        if not is_busy:
            self.scan_options_panel._update_dependent_ui_state()
            self.options_panel._update_scan_context()

        self.options_panel.set_scan_button_state(is_busy)
        QApplication.processEvents()

    @Slot(object, int, object, float, list)
    def on_scan_complete(self, payload, num_found, mode, duration, skipped):
        if not mode:
            logger.warning("Scan cancelled.")
            self.on_scan_end()
            return

        time_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        log_msg = f"Finished! Found {num_found} items in {time_str}."
        APP_SIGNAL_BUS.log_message.emit(log_msg, "success")

        # Pass data to Results Panel (View)
        # Note: payload contains 'db_path' and 'groups_summary'
        self.results_panel.display_results(payload, num_found, mode)

        # Check Filesystem Capabilities if duplicates found
        if num_found > 0 and mode == ScanMode.DUPLICATES and self.scanner_controller.config:
            folder = self.scanner_controller.config.folder_path
            link_support = check_link_support(folder)
            self.results_panel.hardlink_available = link_support.get("hardlink", False)
            self.results_panel.reflink_available = link_support.get("reflink", False)
            if link_support.get("reflink"):
                APP_SIGNAL_BUS.log_message.emit("Filesystem supports Reflinks (CoW).", "success")

        self.results_panel.set_enabled_state(num_found > 0)

        if skipped:
            APP_SIGNAL_BUS.log_message.emit(f"{len(skipped)} files skipped.", "warning")
            SkippedFilesDialog(skipped, self).exec()

        # Handle Visualization Output
        if self.scanner_controller.config and self.scanner_controller.config.output.save_visuals:
            groups_data = payload.get("groups_data")  # Might be None if summarized
            db_uri = payload.get("db_path")

            if groups_data or (num_found > 0 and db_uri):
                if self.stats_dialog:
                    self.stats_dialog.switch_to_visualization_mode()
                self._start_visualization_task(groups_data, db_uri)
            else:
                self._close_stats_dialog(payload, num_found, mode, duration, skipped)
        else:
            self._close_stats_dialog(payload, num_found, mode, duration, skipped)

    def _close_stats_dialog(self, *args):
        if self.stats_dialog:
            self.stats_dialog.scan_finished(*args)
        self.on_scan_end()

    @Slot(str)
    def on_scan_error(self, message: str):
        if self.stats_dialog:
            self.stats_dialog.scan_error(message)
        self.on_scan_end()

    def on_scan_end(self):
        if self.stats_dialog:
            if not self.stats_dialog.close():
                self.stats_dialog.deleteLater()
            self.stats_dialog = None
        self.set_ui_scan_state(is_busy=False)

        # Ensure enabled state reflects data
        has_data = self.results_panel.results_model.rowCount() > 0
        self.results_panel.set_enabled_state(has_data)

    # --- Visualization ---

    def _start_visualization_task(self, groups_data, db_uri):
        APP_SIGNAL_BUS.log_message.emit("Generating visualization report...", "info")
        config = self.scanner_controller.config

        # Reusing the existing VisualizationTask but passing new config structure
        # (Assuming VisualizationTask has been updated or config adapter used)
        task = VisualizationTask(groups_data, db_uri, config)

        if self.stats_dialog:
            task.signals.progress.connect(self.stats_dialog.update_visualization_progress)
        task.signals.finished.connect(self._on_save_visuals_finished)
        QThreadPool.globalInstance().start(task)

    @Slot()
    def _on_save_visuals_finished(self):
        APP_SIGNAL_BUS.log_message.emit(f"Report saved to '{VISUALS_DIR.resolve()}'", "success")
        self.on_scan_end()

    # --- File Operations (Via Controller) ---

    @Slot(str)
    def _on_file_op_started(self, op_name: str):
        self.set_ui_scan_state(is_busy=True)
        try:
            op_enum = FileOperation[op_name.upper().replace("...", "")]
            self.results_panel.set_operation_in_progress(op_enum)
        except KeyError:
            pass

    @Slot(list)
    def _on_file_op_finished(self, affected_paths: list[Path]):
        self.statusBar().clearMessage()
        self.results_panel.clear_operation_in_progress()

        if affected_paths:
            # Update Views
            self.results_panel.results_view.selectionModel().blockSignals(True)
            self.viewer_panel.list_view.blockSignals(True)

            # The Controller updated the DB, now we update the UI Model
            self.results_panel.update_after_deletion(affected_paths)
            self.viewer_panel.clear_viewer()

            QApplication.processEvents()
            self.results_panel.results_view.selectionModel().blockSignals(False)
            self.viewer_panel.list_view.blockSignals(False)

        self.set_ui_scan_state(is_busy=False)

    @Slot(str, str)
    def _on_controller_status(self, msg: str, level: str):
        APP_SIGNAL_BUS.log_message.emit(msg, level)

    @Slot(str, str)
    def _on_controller_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    @Slot(Path)
    def _on_external_file_missing(self, path: Path):
        logger.info(f"External deletion detected: {path}")
        self.results_panel.update_after_deletion([path])

    # --- UI Interactions ---

    @Slot()
    def _on_results_selection_changed(self):
        """Syncs selected result group with the Image Viewer."""
        proxy_indexes = self.results_panel.results_view.selectionModel().selectedRows()
        if not proxy_indexes:
            return

        # Map Proxy -> Source
        source_index = self.results_panel.proxy_model.mapToSource(proxy_indexes[0])
        if not source_index.isValid():
            return

        node = source_index.internalPointer()
        if not node:
            return

        # Handle Lazy Loading
        from app.domain.data_models import GroupNode, ResultNode

        results_model = self.results_panel.results_model

        if isinstance(node, GroupNode) and not node.fetched:
            results_model.fetchMore(source_index)
            return

        # Show in Viewer
        group_id = node.group_id
        scroll_to_path = Path(node.path) if isinstance(node, ResultNode) else None

        items = results_model.get_group_children(group_id)
        self.viewer_panel.show_image_group(items, group_id, scroll_to_path)

    @Slot(object)  # QModelIndex
    def _on_group_fetch_finished(self, index):
        # Re-trigger selection logic if the fetched group was selected
        self._on_results_selection_changed()

    # --- Maintenance Actions ---

    def _confirm_action(self, title: str, text: str) -> bool:
        if self.scanner_controller.is_running():
            APP_SIGNAL_BUS.log_message.emit("Action disabled during scan.", "warning")
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _clear_scan_cache(self):
        if self._confirm_action("Clear Scan Cache", "Delete all temporary scan data?"):
            # We must close DB connections before deleting files
            self.services.db_service.close()

            success = clear_scan_cache()
            if success:
                APP_SIGNAL_BUS.log_message.emit("Cache cleared.", "success")
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()
            else:
                APP_SIGNAL_BUS.log_message.emit("Failed to clear cache.", "error")

    def _clear_models_cache(self):
        if self._confirm_action("Clear Models", "Delete downloaded AI models?"):
            self.services.model_manager.release_resources()
            clear_models_cache()
            APP_SIGNAL_BUS.log_message.emit("Models cleared.", "success")

    def _clear_app_data(self):
        if self._confirm_action("Reset App", "Delete ALL settings, logs, and caches?"):
            self.services.shutdown()
            logging.shutdown()
            try:
                clear_all_app_data()
                QMessageBox.information(self, "Reset", "Application data cleared. Please restart.")
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Reset failed: {e}")

    # --- Cleanup ---

    def closeEvent(self, event):
        self.settings_manager.save()

        if QThreadPool.globalInstance().activeThreadCount() > 0:
            QMessageBox.warning(self, "Wait", "Background tasks are still running.")
            event.ignore()
            return

        if self.scanner_controller.is_running():
            if self._confirm_action("Exit", "Scan in progress. Stop and exit?"):
                self.scanner_controller.cancel_scan()
                event.accept()
            else:
                event.ignore()
        else:
            # Shutdown services is handled in main.py finally block
            event.accept()
