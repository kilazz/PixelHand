# app/ui/viewer.py
"""
Image Viewer Panel (View).
"""

import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageChops
from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QModelIndex,
    QPoint,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCursor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.domain.data_models import AppSettings
from app.domain.view_models import ImageComparerState
from app.imaging.processing import TONE_MAPPER, set_active_tonemap_view
from app.infrastructure.settings import SettingsManager
from app.shared.constants import CompareMode, TonemapMode, UIConfig
from app.ui.delegates import ImageItemDelegate
from app.ui.qt_models import ImagePreviewModel
from app.ui.widgets import AlphaBackgroundWidget, ImageCompareWidget, ResizedListView

if TYPE_CHECKING:
    from app.ui.controllers import ResultsController

logger = logging.getLogger("PixelHand.ui.viewer")


def create_file_context_menu(parent) -> tuple[QMenu, QAction, QAction, QAction]:
    """Creates a standard context menu for file operations."""
    context_menu = QMenu(parent)
    open_action = QAction("Open File", parent)
    show_action = QAction("Show in Explorer", parent)
    delete_action = QAction("Move to Trash", parent)
    context_menu.addAction(open_action)
    context_menu.addAction(show_action)
    context_menu.addSeparator()
    context_menu.addAction(delete_action)
    return context_menu, open_action, show_action, delete_action


class ImageViewerPanel(QGroupBox):
    """
    The right-hand panel for viewing and comparing images.
    """

    # Signals
    log_message = Signal(str, str)
    group_became_empty = Signal(int)
    file_missing_detected = Signal(Path)

    def __init__(
        self,
        settings_manager: SettingsManager,
        thread_pool: QThreadPool,
        controller: "ResultsController",
    ):
        super().__init__("Image Viewer")
        self.settings_manager = settings_manager
        self.thread_pool = thread_pool
        self.controller = controller  # Replaces FileOperationManager

        # Internal State
        self.state = ImageComparerState(thread_pool)
        self.is_transparency_enabled = settings_manager.settings.viewer.show_transparency
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)

        self._init_state()
        self._init_ui()
        self._init_context_menu()
        self._connect_signals()

        self.load_settings(settings_manager.settings)
        self.clear_viewer()

    def _init_state(self):
        self.current_group_id: int | None = None
        self.channel_buttons: dict[str, QPushButton] = {}
        self.channel_states = {"R": True, "G": True, "B": True, "A": True}

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 8, 4, 4)

        # Container 1: List/Grid View
        self.list_container = QWidget()
        list_layout = QVBoxLayout(self.list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(4)
        self._create_list_view_controls(list_layout)

        # Container 2: Comparison View
        self.compare_container = QWidget()
        compare_layout = QVBoxLayout(self.compare_container)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        self._create_compare_view_controls(compare_layout)

        main_layout.addWidget(self.list_container)
        main_layout.addWidget(self.compare_container)

    def _create_list_view_controls(self, parent_layout):
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)

        # View Mode Buttons
        self.viewer_mode_group = QButtonGroup(self)
        self.btn_view_list = QPushButton("☰")
        self.btn_view_list.setCheckable(True)
        self.btn_view_list.setChecked(True)
        self.btn_view_list.setFixedSize(30, 26)
        self.btn_view_list.setToolTip("List View")

        self.btn_view_grid = QPushButton("⊞")
        self.btn_view_grid.setCheckable(True)
        self.btn_view_grid.setFixedSize(30, 26)
        self.btn_view_grid.setToolTip("Grid View")

        self.viewer_mode_group.addButton(self.btn_view_list)
        self.viewer_mode_group.addButton(self.btn_view_grid)

        top_bar.addWidget(self.btn_view_list)
        top_bar.addWidget(self.btn_view_grid)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setFixedHeight(16)
        top_bar.addWidget(line)

        # Size Slider
        top_bar.addWidget(QLabel("Size:"))
        self.preview_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_size_slider.setRange(UIConfig.Sizes.PREVIEW_MIN_SIZE, UIConfig.Sizes.PREVIEW_MAX_SIZE)
        self.preview_size_slider.setFixedWidth(120)
        top_bar.addWidget(self.preview_size_slider)

        # Spacer
        top_bar.addSpacing(10)

        # Background Controls
        self.bg_alpha_check = QCheckBox("BG:")
        top_bar.addWidget(self.bg_alpha_check)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(255)
        self.alpha_slider.setFixedWidth(80)
        top_bar.addWidget(self.alpha_slider)

        self.alpha_label = QLabel("255")
        self.alpha_label.setFixedWidth(28)
        top_bar.addWidget(self.alpha_label)

        # HDR Controls
        top_bar.addSpacing(10)
        self.thumbnail_tonemap_check = QCheckBox("HDR")
        top_bar.addWidget(self.thumbnail_tonemap_check)

        self.thumbnail_tonemap_combo = QComboBox()
        self.thumbnail_tonemap_combo.setMinimumWidth(100)
        if TONE_MAPPER and TONE_MAPPER.available_views:
            self.thumbnail_tonemap_combo.addItems(TONE_MAPPER.available_views)
        else:
            self.thumbnail_tonemap_combo.setEnabled(False)
        top_bar.addWidget(self.thumbnail_tonemap_combo)

        top_bar.addStretch(1)
        parent_layout.addLayout(top_bar)

        # Compare Button
        self.compare_button = QPushButton("Compare (0)")
        parent_layout.addWidget(self.compare_button)

        # List View Setup
        self.model = ImagePreviewModel(self.thread_pool, self)
        self.model.file_missing.connect(self.file_missing_detected.emit)

        self.delegate = ImageItemDelegate(self.settings_manager.settings.viewer.preview_size, self.state, self)

        self.list_view = ResizedListView(self)
        self.list_view.set_preview_size(self.settings_manager.settings.viewer.preview_size)
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.channel_hovered.connect(self.delegate.set_hover_channel)
        self.list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setSpacing(5)

        parent_layout.addWidget(self.list_view)

    def _create_compare_view_controls(self, parent_layout):
        # Top Bar
        top_controls = QHBoxLayout()
        self.back_button = QPushButton("< Back to List")
        self.compare_type_combo = QComboBox()
        self.compare_type_combo.addItems([e.value for e in CompareMode])

        self.reset_view_button = QPushButton("Fit / Reset")
        self.reset_view_button.setToolTip("Reset Zoom and Pan")
        self.reset_view_button.clicked.connect(self._reset_compare_views)

        top_controls.addWidget(self.back_button)
        top_controls.addWidget(self.compare_type_combo)
        top_controls.addWidget(self.reset_view_button)

        self.tiling_check = QCheckBox("Tile Check (3x3)")
        top_controls.addWidget(self.tiling_check)
        top_controls.addStretch()

        # --- HDR & Channel Controls ---

        # 1. HDR Checkbox (Moved before View dropdown)
        self.compare_tonemap_check = QCheckBox("HDR")
        top_controls.addWidget(self.compare_tonemap_check)

        # 2. View Label & Combo
        self.tonemap_view_label = QLabel("View:")
        self.tonemap_view_combo = QComboBox()
        if TONE_MAPPER and TONE_MAPPER.available_views:
            self.tonemap_view_combo.addItems(TONE_MAPPER.available_views)
        else:
            self.tonemap_view_combo.setEnabled(False)
            self.tonemap_view_label.setEnabled(False)

        top_controls.addWidget(self.tonemap_view_label)
        top_controls.addWidget(self.tonemap_view_combo)

        # 3. Channels
        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(2)
        for channel in ["R", "G", "B", "A"]:
            btn = QPushButton(channel)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(UIConfig.Sizes.CHANNEL_BUTTON_SIZE, UIConfig.Sizes.CHANNEL_BUTTON_SIZE)
            btn.toggled.connect(self._on_channel_toggled)
            self.channel_buttons[channel] = btn
            self._update_channel_button_style(btn, True)
            channel_layout.addWidget(btn)
        top_controls.addLayout(channel_layout)

        parent_layout.addLayout(top_controls)

        # Bottom Controls
        bottom_controls = QHBoxLayout()
        overlay_widget, bg_widget = QWidget(), QWidget()
        overlay_layout, bg_layout = (
            QHBoxLayout(overlay_widget),
            QHBoxLayout(bg_widget),
        )

        self.overlay_alpha_label = QLabel("Overlay Alpha:")
        self.overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_alpha_slider.setRange(0, 255)
        self.overlay_alpha_slider.setValue(128)
        overlay_layout.addWidget(self.overlay_alpha_label)
        overlay_layout.addWidget(self.overlay_alpha_slider)

        self.compare_bg_alpha_check = QCheckBox("BG Alpha:")
        self.compare_bg_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_bg_alpha_slider.setRange(0, 255)
        self.compare_bg_alpha_slider.setValue(255)
        bg_layout.addWidget(self.compare_bg_alpha_check)
        bg_layout.addWidget(self.compare_bg_alpha_slider)

        bottom_controls.addWidget(overlay_widget)
        bottom_controls.addWidget(bg_widget)
        parent_layout.addLayout(bottom_controls)

        # Comparison Stack
        self.compare_stack = QStackedWidget()

        # 1. Side-by-Side
        sbs_view = QWidget()
        sbs_layout = QHBoxLayout(sbs_view)
        sbs_layout.setContentsMargins(0, 0, 0, 0)
        sbs_layout.setSpacing(1)
        self.compare_view_1, self.compare_view_2 = (
            AlphaBackgroundWidget(),
            AlphaBackgroundWidget(),
        )
        sbs_layout.addWidget(self.compare_view_1, 1)
        sbs_layout.addWidget(self.compare_view_2, 1)

        # 2. Compare Widget (Wipe/Overlay)
        self.compare_widget = ImageCompareWidget()

        # 3. Difference View
        self.diff_view = AlphaBackgroundWidget()

        self.compare_stack.addWidget(sbs_view)
        self.compare_stack.addWidget(self.compare_widget)
        self.compare_stack.addWidget(self.diff_view)

        parent_layout.addWidget(self.compare_stack, 1)

        # Connect Synchronous Zoom
        self.compare_view_1.view_changed.connect(self._sync_views)
        self.compare_view_2.view_changed.connect(self._sync_views)
        self.compare_widget.view_changed.connect(self._sync_views)
        self.diff_view.view_changed.connect(self._sync_views)

    def _init_context_menu(self):
        self.context_menu_path: Path | None = None
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)

        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _connect_signals(self):
        # Modes
        self.btn_view_list.clicked.connect(lambda: self._set_viewer_mode(is_grid=False))
        self.btn_view_grid.clicked.connect(lambda: self._set_viewer_mode(is_grid=True))

        # Controls
        self.alpha_slider.valueChanged.connect(self._on_alpha_change)

        # --- PREVIEW SIZE SLIDER PERFORMANCE FIX ---
        # 1. While sliding: Update visuals only (Fast)
        self.preview_size_slider.valueChanged.connect(self._on_preview_size_sliding)
        # 2. On Release: Reload data (Slow, clears cache)
        self.preview_size_slider.sliderReleased.connect(self._on_preview_size_confirmed)

        self.bg_alpha_check.toggled.connect(self._on_transparency_toggled)
        self.compare_bg_alpha_check.toggled.connect(self._on_transparency_toggled)

        # HDR
        self.thumbnail_tonemap_check.toggled.connect(self.settings_manager.set_thumbnail_tonemap_enabled)
        self.thumbnail_tonemap_check.toggled.connect(self._on_thumbnail_tonemap_toggled)
        self.thumbnail_tonemap_combo.currentTextChanged.connect(self._on_thumbnail_view_changed)

        # List View Events
        self.list_view.verticalScrollBar().valueChanged.connect(self.update_timer.start)
        self.list_view.resized.connect(self.update_timer.start)
        self.update_timer.timeout.connect(self._update_visible_previews)
        self.list_view.clicked.connect(self._on_item_clicked)

        # Context Menu
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

        # Comparison Logic Signals (State Pattern)
        self.compare_button.clicked.connect(self._show_comparison_view)
        self.back_button.clicked.connect(self._back_to_list_view)
        self.compare_type_combo.currentTextChanged.connect(self._on_compare_mode_change)
        self.overlay_alpha_slider.valueChanged.connect(self._on_overlay_alpha_change)
        self.compare_bg_alpha_slider.valueChanged.connect(self._on_alpha_change)

        self.state.candidates_changed.connect(self._update_compare_button)
        self.state.candidate_updated.connect(self._on_candidate_updated)
        self.state.image_loaded.connect(self._on_full_res_image_loaded)
        self.state.load_complete.connect(self._on_load_complete)
        self.state.load_error.connect(self.log_message.emit)

        self.compare_tonemap_check.toggled.connect(self.settings_manager.set_compare_tonemap_enabled)
        self.compare_tonemap_check.toggled.connect(self._on_compare_tonemap_changed)

        self.tonemap_view_combo.currentTextChanged.connect(self.settings_manager.set_tonemap_view)
        self.tonemap_view_combo.currentTextChanged.connect(self._on_tonemap_view_changed)

        self.tiling_check.toggled.connect(self._on_tiling_toggled)

    def _set_viewer_mode(self, is_grid: bool):
        self.list_view.setUpdatesEnabled(False)
        if is_grid:
            self.list_view.setViewMode(QListView.ViewMode.IconMode)
            self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
            self.list_view.setSpacing(10)
            self.delegate.set_grid_mode(True)
        else:
            self.list_view.setViewMode(QListView.ViewMode.ListMode)
            self.list_view.setResizeMode(QListView.ResizeMode.Fixed)
            self.list_view.setSpacing(5)
            self.delegate.set_grid_mode(False)
        self.list_view.updateGeometries()
        self.list_view.setUpdatesEnabled(True)

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu(self, pos):
        if (item := self.get_item_at_pos(pos)) and (path_str := item.path):
            self.context_menu_path = Path(path_str)
            self.context_menu.exec(QCursor.pos())

    @Slot()
    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    @Slot()
    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        """Delegates deletion to the ResultsController."""
        if (
            self.context_menu_path
            and QMessageBox.question(
                self,
                "Confirm Move",
                f"Move '{self.context_menu_path.name}' to trash?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            # Using Controller instead of direct file op manager
            self.controller.request_deletion([self.context_menu_path])

    @Slot(list)
    def display_results(self, items: list):
        self.model.set_items_from_list(items)
        self._back_to_list_view()
        QTimer.singleShot(50, self.update_timer.start)

    # --- Tonemapping Logic ---

    @Slot(bool)
    def _on_thumbnail_tonemap_toggled(self, checked: bool):
        self.thumbnail_tonemap_combo.setEnabled(checked)
        mode = TonemapMode.ENABLED.value if checked else TonemapMode.NONE.value
        self.model.set_tonemap_mode(mode)

    @Slot(str)
    def _on_thumbnail_view_changed(self, view_name: str):
        if set_active_tonemap_view(view_name):
            self.settings_manager.set_tonemap_view(view_name)
            self.model.clear_cache()
            self.list_view.viewport().update()

    @Slot(bool)
    def _on_compare_tonemap_changed(self, checked: bool):
        self.tonemap_view_combo.setEnabled(checked)
        self.tonemap_view_label.setEnabled(checked)
        if self.compare_container.isVisible() and len(self.state.get_candidates()) == 2:
            self._show_comparison_view()

    @Slot(str)
    def _on_tonemap_view_changed(self, view_name: str):
        if set_active_tonemap_view(view_name):
            self.model.clear_cache()
            if self.compare_container.isVisible():
                self._show_comparison_view()

    # --- Load/Save Settings ---

    def load_settings(self, settings: AppSettings):
        self.preview_size_slider.setValue(settings.viewer.preview_size)
        self.bg_alpha_check.setChecked(settings.viewer.show_transparency)
        self.thumbnail_tonemap_check.setChecked(settings.viewer.thumbnail_tonemap_enabled)
        self.thumbnail_tonemap_combo.setEnabled(settings.viewer.thumbnail_tonemap_enabled)
        self.compare_tonemap_check.setChecked(settings.viewer.compare_tonemap_enabled)

        current_view = settings.viewer.tonemap_view
        if (self.tonemap_view_combo.isEnabled() or current_view) and (
            (idx := self.tonemap_view_combo.findText(current_view)) >= 0
        ):
            self.tonemap_view_combo.setCurrentIndex(idx)

        if (self.thumbnail_tonemap_combo.isEnabled() or current_view) and (
            (idx := self.thumbnail_tonemap_combo.findText(current_view)) >= 0
        ):
            self.thumbnail_tonemap_combo.setCurrentIndex(idx)

        self._on_transparency_toggled(settings.viewer.show_transparency)

    def clear_viewer(self):
        self.update_timer.stop()
        self.model.set_items_from_list([])
        self.current_group_id = None
        self._back_to_list_view()

    # --- Core Logic ---

    @Slot(list, int, object)
    def show_image_group(self, items: list, group_id: int, scroll_to_path: Path | None):
        if self.current_group_id == group_id and not scroll_to_path:
            return

        self._back_to_list_view()
        self.current_group_id = group_id
        self.state.clear_candidates()
        self.model.set_items_from_list(items)
        self.model.group_id = group_id

        if self.model.rowCount() == 0 and self.current_group_id is not None:
            self.group_became_empty.emit(self.current_group_id)
            self.current_group_id = None
            return

        if self.model.rowCount() > 0:
            self.list_view.scrollToTop()
            # Force size update to current slider value
            self.delegate.set_preview_size(self.preview_size_slider.value())
            self.list_view.set_preview_size(self.preview_size_slider.value())

            self._on_thumbnail_tonemap_toggled(self.thumbnail_tonemap_check.isChecked())
            QTimer.singleShot(50, self.update_timer.start)
            if scroll_to_path:
                QTimer.singleShot(0, lambda: self._scroll_to_file(scroll_to_path))

    def _scroll_to_file(self, file_path: Path):
        if (row := self.model.get_row_for_path(file_path)) is not None:
            self.list_view.scrollTo(
                self.model.index(row, 0),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    @Slot(int)
    def _on_preview_size_sliding(self, value: int):
        """
        Visual Update Only (Fast).
        Called continuously while dragging the slider.
        """
        # Updates the paint delegate (rendering size)
        self.delegate.set_preview_size(value)
        # Updates the list view grid size hints
        self.list_view.set_preview_size(value)
        # Forces re-layout of the grid without reloading data
        self.list_view.doItemsLayout()
        self.list_view.viewport().update()

    @Slot()
    def _on_preview_size_confirmed(self):
        """
        Data Reload (Slow).
        Called when slider is released.
        """
        new_size = self.preview_size_slider.value()
        self.settings_manager.set_preview_size(new_size)
        # This triggers clear_cache() inside the model
        self.model.set_target_size(new_size)
        self.list_view.viewport().update()

    @Slot()
    def _update_visible_previews(self):
        if self.model.rowCount() > 0 and self.list_container.isVisible():
            self.list_view.viewport().update()

    @Slot(QModelIndex)
    def _on_item_clicked(self, index):
        if item := self.model.data(index, Qt.ItemDataRole.UserRole):
            self.state.toggle_candidate(item)

    @Slot(str, str)
    def _on_candidate_updated(self, added_path: str, removed_path: str):
        paths_to_update = {p for p in [added_path, removed_path] if p}
        for path_str in paths_to_update:
            if (row := self.model.get_row_for_path(path_str)) is not None:
                self.list_view.update(self.model.index(row, 0))

    @Slot(int)
    def _update_compare_button(self, count):
        self.compare_button.setText(f"Compare ({count})")
        self.compare_button.setEnabled(count == 2)

    # --- Comparison Logic ---

    def _show_comparison_view(self):
        if len(self.state.get_candidates()) != 2:
            return
        self._set_view_mode(is_list=False)
        tonemap_mode = TonemapMode.ENABLED.value if self.compare_tonemap_check.isChecked() else TonemapMode.NONE.value
        self.compare_view_1.setPixmap(QPixmap())
        self.compare_view_2.setPixmap(QPixmap())
        self.state.load_full_res_images(tonemap_mode)

    @Slot(str, QPixmap)
    def _on_full_res_image_loaded(self, path_str: str, pixmap: QPixmap):
        candidates = self.state.get_candidates()
        if len(candidates) < 2:
            return
        paths = [str(node.path) for node in candidates]
        if path_str == paths[0]:
            self.compare_view_1.setPixmap(pixmap)
        elif path_str == paths[1]:
            self.compare_view_2.setPixmap(pixmap)

        self.compare_widget.setPixmaps(self.compare_view_1.pixmap, self.compare_view_2.pixmap)

    @Slot()
    def _on_load_complete(self):
        self._update_channel_controls_based_on_images()
        self._update_compare_views()

    def _update_channel_controls_based_on_images(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return
        act1 = self._get_channel_activity(images[0])
        act2 = self._get_channel_activity(images[1])

        for channel, button in self.channel_buttons.items():
            is_active = act1.get(channel, False) or act2.get(channel, False)
            button.setEnabled(is_active)
            button.setChecked(is_active)
            self._update_channel_button_style(button, is_active)

    def _back_to_list_view(self):
        for button in self.channel_buttons.values():
            button.setEnabled(True)
            button.setChecked(True)
            self._update_channel_button_style(button, True)

        self.state.stop_loaders()
        self.tiling_check.setChecked(False)

        self._reset_compare_views()
        self._set_view_mode(is_list=True)
        self.list_view.viewport().update()

        if self.model.rowCount() > 0:
            self.update_timer.start()

    def _set_view_mode(self, is_list: bool):
        self.list_container.setVisible(is_list)
        self.compare_container.setVisible(not is_list)
        if not is_list:
            self._on_compare_mode_change(self.compare_type_combo.currentText())

    def _on_compare_mode_change(self, text: str):
        mode = CompareMode(text)
        is_overlay = mode == CompareMode.OVERLAY
        is_diff = mode == CompareMode.DIFF

        self.compare_stack.setCurrentIndex(
            2 if is_diff else (1 if mode in [CompareMode.WIPE, CompareMode.OVERLAY] else 0)
        )
        self.overlay_alpha_slider.parentWidget().setVisible(is_overlay)
        if mode in [CompareMode.WIPE, CompareMode.OVERLAY]:
            self.compare_widget.setMode(mode)
        self._update_compare_views()

    def get_item_at_pos(self, pos) -> object | None:
        if (index := self.list_view.indexAt(pos)).isValid():
            return index.data(Qt.ItemDataRole.UserRole)
        return None

    @Slot(bool)
    def _on_transparency_toggled(self, state: bool):
        self.is_transparency_enabled = state
        self.bg_alpha_check.setChecked(state)
        self.compare_bg_alpha_check.setChecked(state)
        for w in [self.alpha_slider, self.alpha_label, self.compare_bg_alpha_slider]:
            w.setEnabled(state)

        self.delegate.set_transparency_enabled(state)
        for view in [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]:
            view.set_transparency_enabled(state)

        self.settings_manager.set_show_transparency(state)
        self.list_view.viewport().update()
        self.compare_container.update()

    @Slot(int)
    def _on_alpha_change(self, value: int):
        self.alpha_label.setText(str(value))
        self.alpha_slider.setValue(value)
        self.compare_bg_alpha_slider.setValue(value)
        self.delegate.set_bg_alpha(value)
        self.list_view.viewport().update()

        for view in [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]:
            view.set_alpha(value)

    @Slot(int)
    def _on_overlay_alpha_change(self, value: int):
        self.compare_widget.setOverlayAlpha(value)

    @Slot(bool)
    def _on_channel_toggled(self, is_checked):
        if sender := self.sender():
            self.channel_states[sender.text()] = is_checked
            self._update_channel_button_style(sender, is_checked)
            self._update_compare_views()

    def _update_channel_button_style(self, button: QPushButton, is_checked: bool):
        channel = button.text()
        if not button.isEnabled():
            button.setStyleSheet("background-color: #3c3c3c; color: #7f8c8d;")
            return
        color = {"R": "red", "G": "lime", "B": "deepskyblue", "A": "white"}[channel]
        if is_checked:
            button.setStyleSheet(f"background-color: {color}; color: black; font-weight: bold;")
        else:
            bc = {"R": "#c0392b", "G": "#27ae60", "B": "#2980b9", "A": "#bdc3c7"}[channel]
            button.setStyleSheet(f"background-color: #2c3e50; border: 1px solid {bc}; color: {bc};")

    def _update_compare_views(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return

        active_channels = [ch for ch, is_active in self.channel_states.items() if is_active]
        num_active = len(active_channels)

        def get_processed_pixmap(pil_image: Image.Image) -> QPixmap:
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            if num_active == 1:
                ch = pil_image.getchannel(active_channels[0])
                gray = Image.merge("RGB", (ch, ch, ch))
                gray.putalpha(Image.new("L", gray.size, 255))
                return QPixmap.fromImage(ImageQt(gray))
            else:
                r, g, b, a = pil_image.split()
                # If only RGB is selected and Alpha is 0 but has RGB data, force full alpha for view
                # (Simplified check)

                if not self.channel_states["R"]:
                    r = r.point(lambda _: 0)
                if not self.channel_states["G"]:
                    g = g.point(lambda _: 0)
                if not self.channel_states["B"]:
                    b = b.point(lambda _: 0)
                if not self.channel_states["A"]:
                    a = a.point(lambda _: 255)

                return QPixmap.fromImage(ImageQt(Image.merge("RGBA", (r, g, b, a))))

        current_mode = CompareMode(self.compare_type_combo.currentText())
        if current_mode == CompareMode.DIFF:
            self.diff_view.setPixmap(self._calculate_diff_pixmap())
            return

        p1 = get_processed_pixmap(images[0])
        p2 = get_processed_pixmap(images[1])

        if current_mode == CompareMode.SIDE_BY_SIDE:
            self.compare_view_1.setPixmap(p1)
            self.compare_view_2.setPixmap(p2)
        else:
            self.compare_widget.setPixmaps(p1, p2)

    def _get_channel_activity(self, img: Image.Image) -> dict[str, bool]:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        activity = {}
        # Channel Indices: 0=R, 1=G, 2=B, 3=A
        names = ["R", "G", "B", "A"]
        for i, name in enumerate(names):
            try:
                extrema = img.getchannel(i).getextrema()
                activity[name] = extrema[0] != extrema[1]
            except Exception:
                activity[name] = False
        return activity

    def _calculate_diff_pixmap(self) -> QPixmap | None:
        images = self.state.get_pil_images()
        if len(images) != 2:
            return None
        img1, img2 = images[0], images[1]

        # Resize largest
        if img1.size != img2.size:
            ts = (max(img1.width, img2.width), max(img1.height, img2.height))
            img1 = img1.resize(ts, Image.Resampling.LANCZOS)
            img2 = img2.resize(ts, Image.Resampling.LANCZOS)

        if img1.mode != "RGBA":
            img1 = img1.convert("RGBA")
        if img2.mode != "RGBA":
            img2 = img2.convert("RGBA")

        r1, g1, b1, a1 = img1.split()
        r2, g2, b2, a2 = img2.split()

        r_diff = ImageChops.difference(r1, r2) if self.channel_states["R"] else Image.new("L", img1.size, 0)
        g_diff = ImageChops.difference(g1, g2) if self.channel_states["G"] else Image.new("L", img1.size, 0)
        b_diff = ImageChops.difference(b1, b2) if self.channel_states["B"] else Image.new("L", img1.size, 0)
        a_diff = ImageChops.difference(a1, a2) if self.channel_states["A"] else Image.new("L", img1.size, 255)

        return QPixmap.fromImage(ImageQt(Image.merge("RGBA", (r_diff, g_diff, b_diff, a_diff))))

    # --- Zoom & Pan Sync ---

    @Slot(bool)
    def _on_tiling_toggled(self, checked: bool):
        for v in [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]:
            v.set_tiling_enabled(checked)

    @Slot(float, QPoint)
    def _sync_views(self, zoom: float, offset: QPoint):
        """Synchronizes zoom/pan across all compare widgets."""
        sender = self.sender()
        widgets = [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]
        for w in widgets:
            if w != sender:
                w.blockSignals(True)
                w.set_sync_view(zoom, offset)
                w.blockSignals(False)

    @Slot()
    def _reset_compare_views(self):
        """Resets zoom and pan to default (Fit)."""
        widgets = [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]
        for w in widgets:
            w.blockSignals(True)
            w.reset_view()
            w.blockSignals(False)
