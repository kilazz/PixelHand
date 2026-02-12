# app/ui/results.py
"""
Results Panel (View).
"""

import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.domain.data_models import FileOperation, GroupNode, ResultNode, ScanMode
from app.shared.constants import UIConfig
from app.ui.delegates import GroupGridDelegate
from app.ui.qt_models import ResultsProxyModel, ResultsTreeModel
from app.ui.widgets import create_file_context_menu

if TYPE_CHECKING:
    from app.ui.controllers import ResultsController

logger = logging.getLogger("PixelHand.ui.results")


class ResultsPanel(QGroupBox):
    """
    Displays scan results and provides controls for file actions.
    """

    # Signal emitted when the visible set of items changes (e.g. filtering)
    # Used by the ImageViewer to update its thumbnail list.
    visible_results_changed = Signal(list)

    def __init__(self, controller: "ResultsController"):
        super().__init__("Results")
        self.controller = controller  # Business Logic & Service Access

        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300)

        # UI State
        self.hardlink_available = False
        self.reflink_available = False
        self.current_operation = FileOperation.NONE
        self.context_menu_path: Path | None = None

        self._init_ui()
        self._setup_models()
        self._init_context_menu()
        self._connect_signals()

        self.set_enabled_state(is_enabled=False)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Header Controls (Search, View Mode, Filter)
        self._create_header_controls(layout)

        # 2. Main Views (Tree / Grid)
        self.view_stack = QStackedWidget()

        self.results_view = QTreeView()
        self.results_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_view.setAlternatingRowColors(True)
        self.results_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_view.setSortingEnabled(True)

        self.results_grid = QListView()
        self.results_grid.setViewMode(QListView.ViewMode.IconMode)
        self.results_grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_grid.setSpacing(12)
        self.results_grid.setUniformItemSizes(False)
        self.results_grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.view_stack.addWidget(self.results_view)
        self.view_stack.addWidget(self.results_grid)

        layout.addWidget(self.view_stack)

        # 3. Action Buttons (Select All, Delete, etc.)
        self._create_action_buttons(layout)

        # 4. Bottom Operation Buttons
        bottom_buttons_layout = QHBoxLayout()
        self.hardlink_button = QPushButton("Replace with Hardlink")
        self.reflink_button = QPushButton("Replace with Reflink")
        self.delete_button = QPushButton("Move to Trash")

        # IDs for styling
        self.hardlink_button.setObjectName("hardlink_button")
        self.reflink_button.setObjectName("reflink_button")
        self.delete_button.setObjectName("delete_button")

        self.hardlink_button.setToolTip("Replaces duplicates with a pointer to the best file's data.")
        self.reflink_button.setToolTip("Creates a space-saving, independent copy (Copy-on-Write).")

        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(self.hardlink_button)
        bottom_buttons_layout.addWidget(self.reflink_button)
        bottom_buttons_layout.addWidget(self.delete_button)
        layout.addLayout(bottom_buttons_layout)

    def _setup_models(self):
        # Pass the DB service from the controller's container to the model
        self.results_model = ResultsTreeModel(self.controller.services.db_service, self)

        self.proxy_model = ResultsProxyModel(self)
        self.proxy_model.setSourceModel(self.results_model)

        self.results_view.setModel(self.proxy_model)
        self.results_grid.setModel(self.proxy_model)

        # Grid Delegate needs ThreadPool for async thumbnail loading
        # QThreadPool.globalInstance() is used for UI rendering tasks
        self.grid_delegate = GroupGridDelegate(QThreadPool.globalInstance(), self)
        self.results_grid.setItemDelegate(self.grid_delegate)

    def _init_context_menu(self):
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)

        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _create_header_controls(self, layout):
        top_controls_layout = QHBoxLayout()
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Filter results by name...")
        top_controls_layout.addWidget(self.search_entry)

        self.view_mode_group = QButtonGroup(self)
        self.btn_list_view = QPushButton("☰")
        self.btn_list_view.setCheckable(True)
        self.btn_list_view.setChecked(True)
        self.btn_list_view.setFixedWidth(30)
        self.btn_list_view.setToolTip("List View (Details)")

        self.btn_grid_view = QPushButton("⊞")
        self.btn_grid_view.setCheckable(True)
        self.btn_grid_view.setFixedWidth(30)
        self.btn_grid_view.setToolTip("Grid View (Folders)")

        self.view_mode_group.addButton(self.btn_list_view)
        self.view_mode_group.addButton(self.btn_grid_view)

        mode_btn_layout = QHBoxLayout()
        mode_btn_layout.setSpacing(4)
        mode_btn_layout.setContentsMargins(0, 0, 0, 0)
        mode_btn_layout.addWidget(self.btn_list_view)
        mode_btn_layout.addWidget(self.btn_grid_view)

        mode_widget = QWidget()
        mode_widget.setLayout(mode_btn_layout)
        top_controls_layout.addWidget(mode_widget)

        # --- Grid Size Slider with Label ---
        self.grid_size_container = QWidget()
        self.grid_size_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        grid_size_layout = QHBoxLayout(self.grid_size_container)
        grid_size_layout.setContentsMargins(0, 0, 0, 0)
        grid_size_layout.setSpacing(4)

        self.grid_size_label = QLabel("Size:")
        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_size_slider.setRange(100, 350)
        self.grid_size_slider.setValue(130)
        self.grid_size_slider.setFixedWidth(80)
        self.grid_size_slider.setToolTip("Grid Card Size")
        self.grid_size_slider.valueChanged.connect(self._on_grid_size_changed)

        grid_size_layout.addWidget(self.grid_size_label)
        grid_size_layout.addWidget(self.grid_size_slider)

        self.grid_size_container.setVisible(False)  # Hidden initially (List view default)
        top_controls_layout.addWidget(self.grid_size_container)

        self.expand_button = QPushButton("Expand All")
        self.collapse_button = QPushButton("Collapse All")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(UIConfig.ResultsView.SORT_OPTIONS)

        top_controls_layout.addWidget(self.expand_button)
        top_controls_layout.addWidget(self.collapse_button)
        top_controls_layout.addWidget(self.sort_combo)

        layout.addLayout(top_controls_layout)

        # Filter controls (Similarity Slider)
        filter_controls_layout = QHBoxLayout()
        self.similarity_filter_label = QLabel("Min Similarity:")
        self.similarity_filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_filter_slider.setRange(0, 100)
        self.similarity_filter_slider.setValue(0)
        self.similarity_filter_value_label = QLabel("0%")
        self.similarity_filter_value_label.setFixedWidth(UIConfig.Sizes.SIMILARITY_LABEL_WIDTH)

        filter_controls_layout.addWidget(self.similarity_filter_label)
        filter_controls_layout.addWidget(self.similarity_filter_slider)
        filter_controls_layout.addWidget(self.similarity_filter_value_label)

        self.filter_widget = QWidget()
        self.filter_widget.setLayout(filter_controls_layout)
        self.filter_widget.setVisible(False)
        layout.addWidget(self.filter_widget)

    def _create_action_buttons(self, layout):
        self.selection_group_box = QGroupBox("Selection")
        actions_layout = QGridLayout(self.selection_group_box)
        actions_layout.setContentsMargins(4, 4, 4, 4)

        self.select_all_button = QPushButton("Select All")
        self.deselect_all_button = QPushButton("Deselect All")
        self.select_except_best_button = QPushButton("Select All Except Best")
        self.invert_selection_button = QPushButton("Invert Selection")

        actions_layout.addWidget(self.select_all_button, 0, 0)
        actions_layout.addWidget(self.deselect_all_button, 0, 1)
        actions_layout.addWidget(self.select_except_best_button, 1, 0)
        actions_layout.addWidget(self.invert_selection_button, 1, 1)

        layout.addWidget(self.selection_group_box)

    def _connect_signals(self):
        # View Switching
        self.btn_list_view.clicked.connect(lambda: self._set_results_view_mode(is_grid=False))
        self.btn_grid_view.clicked.connect(lambda: self._set_results_view_mode(is_grid=True))

        # Grid Selection Sync
        self.results_grid.selectionModel().selectionChanged.connect(self._sync_grid_selection_to_tree)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)

        # Selection Helpers
        self.select_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Checked))
        self.deselect_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Unchecked))
        self.select_except_best_button.clicked.connect(self.results_model.select_all_except_best)
        self.invert_selection_button.clicked.connect(self.results_model.invert_selection)

        # File Operations -> Controller
        self.delete_button.clicked.connect(self._request_deletion)
        self.hardlink_button.clicked.connect(self._request_hardlink)
        self.reflink_button.clicked.connect(self._request_reflink)

        # Tree Expansions
        self.expand_button.clicked.connect(self.results_view.expandAll)
        self.collapse_button.clicked.connect(self.results_view.collapseAll)

        # Filtering
        self.search_entry.textChanged.connect(self.search_timer.start)
        self.search_timer.timeout.connect(self._on_search_triggered)
        self.similarity_filter_slider.valueChanged.connect(self._on_similarity_filter_changed)
        self.similarity_filter_slider.sliderReleased.connect(self._emit_visible_results)

        # Model Loading
        self.results_model.fetch_completed.connect(self._on_fetch_completed)

        # Context Menus
        self.results_view.customContextMenuRequested.connect(self._show_context_menu_tree)
        self.results_grid.customContextMenuRequested.connect(self._show_context_menu_grid)

        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

    def _set_results_view_mode(self, is_grid: bool):
        self.view_stack.setCurrentWidget(self.results_grid if is_grid else self.results_view)
        self.expand_button.setEnabled(not is_grid)
        self.collapse_button.setEnabled(not is_grid)
        self.selection_group_box.setEnabled(not is_grid)
        self.grid_size_container.setVisible(is_grid)

    @Slot(int)
    def _on_grid_size_changed(self, value: int):
        if hasattr(self, "grid_delegate"):
            self.grid_delegate.set_base_size(value)
            self.results_grid.doItemsLayout()

    @Slot(object, object)
    def _sync_grid_selection_to_tree(self, selected, deselected):
        if not selected.indexes():
            return
        index = selected.indexes()[0]
        self.results_view.selectionModel().select(
            index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
        )
        self.results_view.scrollTo(index)

    @Slot()
    def _update_summary(self):
        self.setTitle(f"Results {self.results_model.get_summary_text()}")

    # --- Interaction Handlers ---

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu_tree(self, pos):
        self._handle_context_menu(self.results_view, pos)

    @Slot(QPoint)
    def _show_context_menu_grid(self, pos):
        self._handle_context_menu(self.results_grid, pos)

    def _handle_context_menu(self, view, pos):
        proxy_idx = view.indexAt(pos)
        if not proxy_idx.isValid():
            return
        source_idx = self.proxy_model.mapToSource(proxy_idx)
        if not source_idx.isValid():
            return
        node = source_idx.internalPointer()

        if isinstance(node, ResultNode) and node.path != "loading_dummy":
            self.context_menu_path = Path(node.path)
            self.context_menu.exec(QCursor.pos())
        elif isinstance(node, GroupNode):
            # Special menu for groups
            group_menu = QMenu(self)
            del_action = QAction(f"Move Group to Trash ({node.count} files)", self)
            del_action.triggered.connect(lambda: self._context_delete_group(node))
            font = del_action.font()
            font.setBold(True)
            del_action.setFont(font)
            group_menu.addAction(del_action)
            group_menu.exec(QCursor.pos())

    def _context_delete_group(self, node: GroupNode):
        """Prepares group deletion and delegates to controller."""
        msg = f"Move {node.count} files from group '{node.name}' to trash?"
        if QMessageBox.question(self, "Delete Group", msg) != QMessageBox.StandardButton.Yes:
            return

        # Gather paths synchronously (might trigger fetch if not loaded, handle gracefully)
        paths = self.results_model.get_paths_for_group_sync(node.group_id)
        if not paths:
            QMessageBox.warning(self, "Error", "Could not find files for this group.")
            return

        self.controller.request_deletion(paths)

    @Slot()
    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    @Slot()
    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        if (
            self.context_menu_path
            and QMessageBox.question(
                self,
                "Confirm Move",
                f"Move '{self.context_menu_path.name}' to trash?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.controller.request_deletion([self.context_menu_path])

    # --- Button Actions -> Controller ---

    def _request_deletion(self):
        to_move = self.results_model.get_checked_paths()
        if not to_move:
            QMessageBox.warning(self, "No Selection", "No files selected to move.")
            return

        # UI Check: Calculate affected groups for confirmation message
        affected_group_ids = {self.results_model.path_to_group_id.get(str(p)) for p in to_move}
        affected_group_ids.discard(None)
        group_str = "group" if len(affected_group_ids) == 1 else "groups"
        msg = f"Move {len(to_move)} files from {len(affected_group_ids)} {group_str} to the system trash?"

        if QMessageBox.question(self, "Confirm Move", msg) == QMessageBox.StandardButton.Yes:
            self.controller.request_deletion(to_move)

    def _request_hardlink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected to replace.")
            return

        # Model Logic: Determine Target -> Source mapping based on "Best" file logic
        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(
                self,
                "No Action",
                "Selected files did not contain any duplicates to replace.",
            )
            return

        msg = f"This will replace {len(link_map)} duplicate files with hardlinks.\n\n⚠️ Original data is shared. Editing one affects all."
        if QMessageBox.question(self, "Confirm Hardlink", msg) == QMessageBox.StandardButton.Yes:
            self.controller.request_linking(link_map, FileOperation.HARDLINKING)

    @Slot()
    def _request_reflink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected.")
            return

        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(self, "No Action", "Selected files did not contain duplicates.")
            return

        msg = f"This will replace {len(link_map)} duplicate files with reflinks (CoW)."
        if QMessageBox.question(self, "Confirm Reflink", msg) == QMessageBox.StandardButton.Yes:
            self.controller.request_linking(link_map, FileOperation.REFLINKING)

    # --- Updates from Controller/Model ---

    def update_after_deletion(self, deleted_paths: list[Path]):
        """
        Called by MainWindow when Controller finishes a deletion/move.
        Updates the internal tree model to reflect changes.
        """
        expanded = self._get_expanded_group_ids()
        self.results_model.remove_deleted_paths(deleted_paths)

        # Reset proxy to ensure filtering is re-applied correctly
        self.proxy_model.sourceModel().beginResetModel()
        self.proxy_model.sourceModel().endResetModel()

        self._update_summary()
        self._restore_expanded_group_ids(expanded)

        if self.results_model.rowCount() == 0:
            self.set_enabled_state(is_enabled=False)
        self._emit_visible_results()

    def set_operation_in_progress(self, operation: FileOperation):
        """Updates UI text to show busy state."""
        self.current_operation = operation
        if operation == FileOperation.DELETING:
            self.delete_button.setText("Deleting...")
        elif operation == FileOperation.HARDLINKING:
            self.hardlink_button.setText("Linking...")
        elif operation == FileOperation.REFLINKING:
            self.reflink_button.setText("Linking...")

    def clear_operation_in_progress(self):
        """Restores UI text after operation."""
        self.current_operation = FileOperation.NONE
        self.delete_button.setText("Move to Trash")
        self.hardlink_button.setText("Replace with Hardlink")
        self.reflink_button.setText("Replace with Reflink")

    # --- View Logic Helpers ---

    @Slot(QModelIndex)
    def _on_fetch_completed(self, parent_index: QModelIndex):
        if self.results_model.mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]:
            self._emit_visible_results()
        self._update_summary()

    def _emit_visible_results(self):
        if self.results_model.mode not in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]:
            return
        visible_items = []
        if self.proxy_model.rowCount() > 0:
            group_proxy_index = self.proxy_model.index(0, 0)
            # Flatten visible children for the viewer
            for row in range(self.proxy_model.rowCount(group_proxy_index)):
                child_proxy_index = self.proxy_model.index(row, 0, group_proxy_index)
                source_index = self.proxy_model.mapToSource(child_proxy_index)
                if source_index.isValid() and (node := source_index.internalPointer()):
                    visible_items.append(node)
        self.visible_results_changed.emit(visible_items)

    @Slot(int)
    def _on_similarity_filter_changed(self, value: int):
        self.similarity_filter_value_label.setText(f"{value}%")
        self.proxy_model.set_similarity_filter(value)

        # Update title count
        visible_count = 0
        if self.proxy_model.rowCount() > 0:
            group_index = self.proxy_model.index(0, 0)
            if group_index.isValid():
                visible_count = self.proxy_model.rowCount(group_index)
            self.setTitle(f"Results ({visible_count} items shown)")
        else:
            self.setTitle("Results")

    @Slot()
    def _on_search_triggered(self):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.filter(self.search_entry.text())
        self._update_summary()
        self._restore_expanded_group_ids(expanded_ids)
        self._emit_visible_results()

    def set_enabled_state(self, is_enabled: bool):
        """Enables/Disables controls based on app state (Scanning vs Idle)."""
        has_results = self.results_model.rowCount() > 0
        is_duplicate_mode = self.results_model.mode == ScanMode.DUPLICATES
        enable_general = is_enabled and has_results

        for w in [
            self.search_entry,
            self.expand_button,
            self.collapse_button,
            self.btn_list_view,
            self.btn_grid_view,
            self.grid_size_slider,
        ]:
            w.setEnabled(enable_general)

        self.results_view.setEnabled(enable_general)
        self.results_grid.setEnabled(enable_general)
        self.sort_combo.setEnabled(enable_general and is_duplicate_mode)
        self.filter_widget.setEnabled(enable_general and not is_duplicate_mode)

        enable_duplicate_controls = enable_general and is_duplicate_mode
        for w in [
            self.select_all_button,
            self.deselect_all_button,
            self.select_except_best_button,
            self.invert_selection_button,
            self.delete_button,
        ]:
            w.setEnabled(enable_duplicate_controls)

        self.hardlink_button.setEnabled(enable_duplicate_controls and self.hardlink_available)
        self.reflink_button.setEnabled(enable_duplicate_controls and self.reflink_available)

    def clear_results(self):
        self.hardlink_available = False
        self.reflink_available = False
        self.search_entry.clear()
        self.results_model.clear()
        self.setTitle("Results")
        self.set_enabled_state(is_enabled=False)

    def display_results(self, payload, num_found, mode):
        self.search_entry.clear()
        self.results_model.load_data(payload, mode)
        is_search_mode = mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]
        self.filter_widget.setVisible(is_search_mode)
        self.similarity_filter_slider.setValue(0)
        self._update_summary()

        is_duplicate_mode = self.results_model.mode == ScanMode.DUPLICATES
        for widget in [
            self.sort_combo,
            self.select_all_button,
            self.deselect_all_button,
            self.select_except_best_button,
            self.invert_selection_button,
            self.hardlink_button,
            self.reflink_button,
            self.delete_button,
        ]:
            widget.setVisible(is_duplicate_mode)

        if num_found > 0:
            if is_duplicate_mode:
                self.results_model.sort_results(self.sort_combo.currentText())
            header = self.results_view.header()
            header.setSectionsMovable(True)
            for i in range(header.count()):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self.set_enabled_state(num_found > 0)
        if is_search_mode and self.proxy_model.rowCount() > 0:
            self.results_view.expandAll()
            # Sort by Score desc
            QTimer.singleShot(100, lambda: self.results_view.sortByColumn(1, Qt.SortOrder.DescendingOrder))
        else:
            self.visible_results_changed.emit([])

    def _get_expanded_group_ids(self) -> set[int]:
        expanded_ids = set()
        for i in range(self.proxy_model.rowCount()):
            proxy_index = self.proxy_model.index(i, 0)
            if self.results_view.isExpanded(proxy_index):
                source_index = self.proxy_model.mapToSource(proxy_index)
                node = source_index.internalPointer()
                if node and hasattr(node, "group_id"):
                    expanded_ids.add(node.group_id)
        return expanded_ids

    def _restore_expanded_group_ids(self, gids: set[int]):
        for i in range(self.proxy_model.rowCount()):
            proxy_index = self.proxy_model.index(i, 0)
            source_index = self.proxy_model.mapToSource(proxy_index)
            if source_index.isValid() and source_index.internalPointer().group_id in gids:
                self.results_view.expand(proxy_index)

    @Slot(str)
    def _on_sort_changed(self, sort_key: str):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.sort_results(sort_key)
        self._restore_expanded_group_ids(expanded_ids)
