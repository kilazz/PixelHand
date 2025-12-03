# app/gui/results_panel.py

import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QItemSelectionModel, QModelIndex, QPoint, Qt, QThreadPool, QTimer, Signal, Slot
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
    QSlider,
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.constants import (
    DB_TABLE_NAME,
    LANCEDB_AVAILABLE,
    UIConfig,
)
from app.data_models import FileOperation, GroupNode, ResultNode, ScanMode
from app.gui.delegates import GroupGridDelegate
from app.gui.models import ResultsProxyModel, ResultsTreeModel

if TYPE_CHECKING:
    from app.services.file_operation_manager import FileOperationManager

app_logger = logging.getLogger("PixelHand.gui.results")


def create_file_context_menu(parent) -> tuple[QMenu, QAction, QAction, QAction]:
    """Creates a standardized context menu for file operations."""
    context_menu = QMenu(parent)
    open_action = QAction("Open File", parent)
    show_action = QAction("Show in Explorer", parent)
    delete_action = QAction("Move to Trash", parent)

    context_menu.addAction(open_action)
    context_menu.addAction(show_action)
    context_menu.addSeparator()
    context_menu.addAction(delete_action)

    return context_menu, open_action, show_action, delete_action


class ResultsPanel(QGroupBox):
    """
    Displays scan results in a Tree View (Details) or Grid View (Folders).
    Provides filtering, sorting, and file operations.
    """

    visible_results_changed = Signal(list)

    def __init__(self, file_op_manager: "FileOperationManager"):
        super().__init__("Results")
        self.file_op_manager = file_op_manager
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300)
        self.hardlink_available = False
        self.reflink_available = False
        self.current_operation = FileOperation.NONE

        self._init_ui()
        self._setup_models()
        self._init_context_menu()
        self._connect_signals()

        self.set_enabled_state(is_enabled=False)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self._create_header_controls(layout)

        # --- Main View Stack (Tree vs Grid) ---
        self.view_stack = QStackedWidget()

        # 1. Tree View (Table/Details)
        self.results_view = QTreeView()
        self.results_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_view.setAlternatingRowColors(True)
        self.results_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_view.setSortingEnabled(True)

        # 2. Grid View (Icons/Folders)
        self.results_grid = QListView()
        self.results_grid.setViewMode(QListView.ViewMode.IconMode)
        self.results_grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_grid.setSpacing(12)
        self.results_grid.setUniformItemSizes(True)
        self.results_grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.view_stack.addWidget(self.results_view)
        self.view_stack.addWidget(self.results_grid)

        layout.addWidget(self.view_stack)

        self._create_action_buttons(layout)

        # Footer buttons
        bottom_buttons_layout = QHBoxLayout()
        self.hardlink_button = QPushButton("Replace with Hardlink")
        self.reflink_button = QPushButton("Replace with Reflink")
        self.delete_button = QPushButton("Move to Trash")

        self.hardlink_button.setObjectName("hardlink_button")
        self.hardlink_button.setToolTip("Replaces duplicates with a pointer to the best file's data.")
        self.reflink_button.setObjectName("reflink_button")
        self.reflink_button.setToolTip("Creates a space-saving, independent copy (Copy-on-Write).")
        self.delete_button.setObjectName("delete_button")

        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(self.hardlink_button)
        bottom_buttons_layout.addWidget(self.reflink_button)
        bottom_buttons_layout.addWidget(self.delete_button)
        layout.addLayout(bottom_buttons_layout)

    def _setup_models(self):
        self.results_model = ResultsTreeModel(self)
        self.proxy_model = ResultsProxyModel(self)
        self.proxy_model.setSourceModel(self.results_model)

        # Apply model to Tree
        self.results_view.setModel(self.proxy_model)

        # Apply model to Grid
        self.results_grid.setModel(self.proxy_model)

        # Setup Grid Delegate (Custom Painting)
        self.grid_delegate = GroupGridDelegate(QThreadPool.globalInstance(), self)
        self.results_grid.setItemDelegate(self.grid_delegate)

    def _init_context_menu(self):
        self.context_menu_path: Path | None = None
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)

        # Enable context menus on both views
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _create_header_controls(self, layout):
        top_controls_layout = QHBoxLayout()
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Filter results by name...")
        top_controls_layout.addWidget(self.search_entry)

        # --- View Mode Switcher ---
        self.view_mode_group = QButtonGroup(self)

        # List Button
        self.btn_list_view = QPushButton("☰")
        self.btn_list_view.setCheckable(True)
        self.btn_list_view.setChecked(True)
        self.btn_list_view.setFixedWidth(30)
        self.btn_list_view.setToolTip("List View (Details)")

        # Grid Button
        self.btn_grid_view = QPushButton("⊞")
        self.btn_grid_view.setCheckable(True)
        self.btn_grid_view.setFixedWidth(30)
        self.btn_grid_view.setToolTip("Grid View (Folders)")

        self.view_mode_group.addButton(self.btn_list_view)
        self.view_mode_group.addButton(self.btn_grid_view)

        # Layout for buttons to stick together
        mode_btn_layout = QHBoxLayout()
        mode_btn_layout.setSpacing(4)
        mode_btn_layout.setContentsMargins(0, 0, 0, 0)
        mode_btn_layout.addWidget(self.btn_list_view)
        mode_btn_layout.addWidget(self.btn_grid_view)

        mode_widget = QWidget()
        mode_widget.setLayout(mode_btn_layout)
        top_controls_layout.addWidget(mode_widget)

        self.expand_button = QPushButton("Expand All")
        self.collapse_button = QPushButton("Collapse All")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(UIConfig.ResultsView.SORT_OPTIONS)

        top_controls_layout.addWidget(self.expand_button)
        top_controls_layout.addWidget(self.collapse_button)
        top_controls_layout.addWidget(self.sort_combo)

        layout.addLayout(top_controls_layout)

        # Filter Controls
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
        # Reduce margins inside group box
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

        # View Sync: Clicking in Grid selects row in Tree (which triggers MainWindow preview)
        self.results_grid.selectionModel().selectionChanged.connect(self._sync_grid_selection_to_tree)

        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)

        # Selection Actions
        self.select_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Checked))
        self.deselect_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Unchecked))
        self.select_except_best_button.clicked.connect(self.results_model.select_all_except_best)
        self.invert_selection_button.clicked.connect(self.results_model.invert_selection)

        # Operations
        self.delete_button.clicked.connect(self._request_deletion)
        self.hardlink_button.clicked.connect(self._request_hardlink)
        self.reflink_button.clicked.connect(self._request_reflink)

        # Tree Controls
        self.expand_button.clicked.connect(self.results_view.expandAll)
        self.collapse_button.clicked.connect(self.results_view.collapseAll)

        # Filters
        self.search_entry.textChanged.connect(self.search_timer.start)
        self.search_timer.timeout.connect(self._on_search_triggered)
        self.similarity_filter_slider.valueChanged.connect(self._on_similarity_filter_changed)
        self.similarity_filter_slider.sliderReleased.connect(self._emit_visible_results)

        # Model Signals
        self.results_model.fetch_completed.connect(self._on_fetch_completed)

        # Context Menus
        self.results_view.customContextMenuRequested.connect(self._show_context_menu_tree)
        self.results_grid.customContextMenuRequested.connect(self._show_context_menu_grid)

        # File Context Actions
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

    def _set_results_view_mode(self, is_grid: bool):
        """Switches the view stack and toggles enabled state of buttons."""
        if is_grid:
            self.view_stack.setCurrentWidget(self.results_grid)
            # Disable Tree-only operations
            self.expand_button.setEnabled(False)
            self.collapse_button.setEnabled(False)
            # Disable File Selection buttons (Grid works on groups, check-boxes are in tree)
            self.selection_group_box.setEnabled(False)
        else:
            self.view_stack.setCurrentWidget(self.results_view)
            self.expand_button.setEnabled(True)
            self.collapse_button.setEnabled(True)
            self.selection_group_box.setEnabled(True)

    @Slot(object, object)
    def _sync_grid_selection_to_tree(self, selected, deselected):
        """
        When items are selected in the grid (which are Groups),
        mirror that selection to the Tree View (select the Group row).
        This allows the existing logic in MainWindow to work without modification.
        """
        if not selected.indexes():
            return

        # Map Proxy index (Grid) -> Proxy index (Tree) is 1:1 for the Group Nodes
        index = selected.indexes()[0]

        self.results_view.selectionModel().select(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        )
        # Ensure the selected item is visible in the tree
        self.results_view.scrollTo(index)

    @Slot()
    def _update_summary(self):
        self.setTitle(f"Results {self.results_model.get_summary_text()}")

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu_tree(self, pos):
        self._handle_context_menu(self.results_view, pos)

    @Slot(QPoint)
    def _show_context_menu_grid(self, pos):
        self._handle_context_menu(self.results_grid, pos)

    def _handle_context_menu(self, view, pos):
        """
        Determines which context menu to show based on the item under the cursor.
        """
        proxy_idx = view.indexAt(pos)
        if not proxy_idx.isValid():
            return

        source_idx = self.proxy_model.mapToSource(proxy_idx)
        if not source_idx.isValid():
            return

        node = source_idx.internalPointer()

        # --- CASE 1: ResultNode (Individual File - Only in Tree View) ---
        if isinstance(node, ResultNode) and node.path != "loading_dummy":
            self.context_menu_path = Path(node.path)
            self.context_menu.exec(QCursor.pos())

        # --- CASE 2: GroupNode (Folder - Grid View or Tree Parent) ---
        elif isinstance(node, GroupNode):
            group_menu = QMenu(self)

            # Action: Delete entire group
            del_action = QAction(f"Move Group to Trash ({node.count} files)", self)
            del_action.triggered.connect(lambda: self._context_delete_group(node))

            # Stylize the destructive action
            font = del_action.font()
            font.setBold(True)
            del_action.setFont(font)

            group_menu.addAction(del_action)
            group_menu.exec(QCursor.pos())

    def _context_delete_group(self, node: GroupNode):
        """
        Handles the deletion of an entire group.
        Uses the model to fetch all paths (synchronously from DB if needed).
        """
        msg = (
            f"Are you sure you want to move the ENTIRE group '{node.name}' to trash?\n\n"
            f"This will delete {node.count} files.\n"
            "This action cannot be undone via this app."
        )
        if QMessageBox.question(self, "Delete Group", msg) != QMessageBox.StandardButton.Yes:
            return

        # Fetch paths from model (handles lazy loaded data transparently)
        paths = self.results_model.get_paths_for_group_sync(node.group_id)

        if not paths:
            QMessageBox.warning(self, "Error", "Could not find files for this group.")
            return

        self.file_op_manager.request_deletion(paths)

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
            and QMessageBox.question(self, "Confirm Move", f"Move '{self.context_menu_path.name}' to trash?")
            == QMessageBox.StandardButton.Yes
        ):
            self.file_op_manager.request_deletion([self.context_menu_path])

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
        if self.proxy_model.rowCount() > 0:
            group_index = self.proxy_model.index(0, 0)
            visible_count = (
                self.proxy_model.rowCount(group_index) if group_index.isValid() else self.proxy_model.rowCount()
            )
            self.setTitle(f"Results ({visible_count} items shown)")
        else:
            self.setTitle("Results")

    def set_operation_in_progress(self, operation: FileOperation):
        self.current_operation = operation
        if operation == FileOperation.DELETING:
            self.delete_button.setText("Deleting...")
        elif operation == FileOperation.HARDLINKING:
            self.hardlink_button.setText("Linking...")
        elif operation == FileOperation.REFLINKING:
            self.reflink_button.setText("Linking...")

    def clear_operation_in_progress(self):
        self.current_operation = FileOperation.NONE
        self.delete_button.setText("Move to Trash")
        self.hardlink_button.setText("Replace with Hardlink")
        self.reflink_button.setText("Replace with Reflink")

    @Slot()
    def _on_search_triggered(self):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.filter(self.search_entry.text())
        self._update_summary()
        self._restore_expanded_group_ids(expanded_ids)
        self._emit_visible_results()

    def set_enabled_state(self, is_enabled: bool):
        has_results = self.results_model.rowCount() > 0
        is_duplicate_mode = self.results_model.mode == ScanMode.DUPLICATES

        enable_general = is_enabled and has_results

        # General controls
        for w in [self.search_entry, self.expand_button, self.collapse_button, self.btn_list_view, self.btn_grid_view]:
            w.setEnabled(enable_general)

        # Views
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
            QTimer.singleShot(
                100,
                lambda: self.results_view.sortByColumn(1, Qt.SortOrder.DescendingOrder),
            )
        else:
            self.visible_results_changed.emit([])

    def _request_deletion(self):
        to_move = self.results_model.get_checked_paths()
        if not to_move:
            QMessageBox.warning(self, "No Selection", "No files selected to move.")
            return

        affected_group_ids = {self.results_model.path_to_group_id.get(str(p)) for p in to_move}
        affected_group_ids.discard(None)

        group_count = len(affected_group_ids)
        file_count = len(to_move)

        group_str = "group" if group_count == 1 else "groups"
        msg = f"Move {file_count} files from {group_count} {group_str} to the system trash?"

        if QMessageBox.question(self, "Confirm Move", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_deletion(to_move)

    def _request_hardlink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected to replace.")
            return

        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(
                self,
                "No Action",
                "Selected files did not contain any duplicates to replace.",
            )
            return

        msg = (
            f"This will replace {len(link_map)} duplicate files with hardlinks to the best file.\n\n"
            "⚠️ IMPORTANT:\n• The original file data will be preserved.\n• Duplicate files will become pointers to the same data.\n"
            "• If you edit any linked file, ALL linked copies will change.\n• This operation cannot be undone.\n\nAre you sure you want to continue?"
        )
        if QMessageBox.question(self, "Confirm Hardlink Replacement", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_hardlink(link_map)

    @Slot()
    def _request_reflink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected to replace.")
            return

        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(
                self,
                "No Action",
                "Selected files did not contain any duplicates to replace.",
            )
            return

        msg = (
            f"This will replace {len(link_map)} duplicate files with reflinks (Copy-on-Write).\n\n"
            "INFO:\n• Creates space-saving copies that share data blocks.\n"
            "• When you edit a file, only the changed blocks are duplicated.\n"
            "• Safer than hardlinks but requires filesystem support (APFS, Btrfs, XFS, ReFS).\n• This operation cannot be undone.\n\nAre you sure you want to continue?"
        )
        if QMessageBox.question(self, "Confirm Reflink Replacement", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_reflink(link_map)

    def remove_items_from_results_db(self, paths_to_delete: list[Path]):
        """
        Removes deleted items from the vector database to keep it in sync.
        """
        if not self.results_model.db_path or not paths_to_delete:
            return

        if not LANCEDB_AVAILABLE:
            return

        try:
            import lancedb

            # Escape single quotes for SQL query
            path_list_str = ", ".join(f"'{str(p).replace("'", "''")}'" for p in paths_to_delete)

            # Connect to LanceDB (db_path is the folder URI)
            db = lancedb.connect(str(self.results_model.db_path))

            # Open the table and delete rows
            if DB_TABLE_NAME in db.table_names():
                table = db.open_table(DB_TABLE_NAME)
                table.delete(f"path IN ({path_list_str})")
                app_logger.info(f"Removed {len(paths_to_delete)} items from LanceDB.")
            else:
                app_logger.warning(f"Table '{DB_TABLE_NAME}' not found in DB, skipping DB deletion.")

        except Exception as e:
            # Log error but don't crash, so UI can still update
            app_logger.error(f"Failed to remove items from LanceDB: {e}")

    def update_after_deletion(self, deleted_paths: list[Path]):
        expanded = self._get_expanded_group_ids()
        self.results_model.remove_deleted_paths(deleted_paths)

        self.proxy_model.sourceModel().beginResetModel()
        self.proxy_model.sourceModel().endResetModel()

        self._update_summary()
        self._restore_expanded_group_ids(expanded)
        if self.results_model.rowCount() == 0:
            self.set_enabled_state(is_enabled=False)
        self._emit_visible_results()

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
            if (
                source_index.isValid()
                and hasattr(source_index.internalPointer(), "group_id")
                and source_index.internalPointer().group_id in gids
            ):
                self.results_view.expand(proxy_index)

    @Slot(str)
    def _on_sort_changed(self, sort_key: str):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.sort_results(sort_key)
        self._restore_expanded_group_ids(expanded_ids)
