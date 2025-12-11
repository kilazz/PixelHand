# app/ui/qt_models.py
"""
Contains Qt Item Models for managing and providing data to views.
Refactored to use the Singleton DB_SERVICE for thread-safe database access.
"""

import logging
import os
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QAbstractItemModel,
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap

from app.domain.data_models import GroupNode, ResultNode, ScanMode
from app.infrastructure.db_service import DB_SERVICE
from app.shared.constants import (
    BEST_FILE_METHOD_NAME,
    METHOD_DISPLAY_NAMES,
    UIConfig,
)
from app.ui.background_tasks import ImageLoader, LanceDBGroupFetcherTask

app_logger = logging.getLogger("PixelHand.ui.models")

# Custom roles
SortRole = Qt.ItemDataRole.UserRole + 1
VfxRole = Qt.ItemDataRole.UserRole + 2


def _format_metadata_string(node: ResultNode) -> str:
    if node.path == "loading_dummy":
        return ""
    res = f"{node.resolution_w}x{node.resolution_h}"
    size_mb = (node.file_size or 0) / (1024**2)
    size_str = f"{size_mb:.2f} MB"
    bit_depth_str = f"{node.bit_depth}-bit"
    parts = [
        res,
        size_str,
        node.format_str,
        node.compression_format,
        node.color_space,
        bit_depth_str,
        node.format_details,
        node.texture_type,
        f"Mips: {node.mipmap_count}",
    ]
    return " | ".join(filter(None, parts))


class ResultsTreeModel(QAbstractItemModel):
    fetch_completed = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode: ScanMode | str = ""
        self.groups_data: dict[int, GroupNode] = {}
        self.sorted_group_ids: list[int] = []
        self.check_states: dict[str, Qt.CheckState] = {}
        self.filter_text = ""
        self.path_to_group_id: dict[str, int] = {}
        self.group_id_to_best_path: dict[int, str] = {}
        self.pending_fetches: dict[int, QPersistentModelIndex] = {}

    def clear(self):
        self.beginResetModel()
        self.mode = ""
        self.groups_data.clear()
        self.sorted_group_ids.clear()
        self.check_states.clear()
        self.filter_text = ""
        self.path_to_group_id.clear()
        self.group_id_to_best_path.clear()
        self.pending_fetches.clear()
        self.endResetModel()

    def filter(self, text: str):
        self.beginResetModel()
        self.filter_text = text
        self.endResetModel()

    def _create_dummy_child(self, group_id: int) -> ResultNode:
        return ResultNode(
            path="loading_dummy",
            is_best=False,
            group_id=group_id,
            resolution_w=0,
            resolution_h=0,
            file_size=0,
            mtime=0,
            capture_date=0,
            distance=0,
            format_str="",
            compression_format="",
            format_details="",
            has_alpha=False,
            bit_depth=0,
            mipmap_count=0,
            texture_type="",
            color_space="",
            found_by="Loading...",
        )

    def load_data(self, payload: dict, mode: ScanMode):
        self.clear()
        self.beginResetModel()
        self.mode = mode

        lazy_summary = payload.get("lazy_summary")
        full_groups = payload.get("groups_data")

        if lazy_summary:
            for item in lazy_summary:
                gid = item["group_id"]
                gn = GroupNode(
                    name=item["name"],
                    count=item["count"],
                    total_size=item["total_size"],
                    group_id=gid,
                    fetched=False,
                    children=[self._create_dummy_child(gid)],
                )
                self.groups_data[gid] = gn
            self.sorted_group_ids = sorted(self.groups_data.keys())
        elif full_groups:
            self._process_full_groups(full_groups)
        self.endResetModel()

    def _process_full_groups(self, raw_groups: dict):
        group_id_counter = 1
        for best_fp, dups in raw_groups.items():
            total_size = (best_fp.file_size or 0) + sum((fp.file_size or 0) for fp, _, _ in dups)
            count = len(dups) + 1
            group_name = best_fp.path.stem
            if best_fp.channel:
                group_name += f" ({best_fp.channel})"

            group_node = GroupNode(
                name=group_name,
                count=count,
                total_size=total_size,
                group_id=group_id_counter,
                fetched=True,
                children=[],
            )
            children = []
            best_node = ResultNode(
                path=str(best_fp.path),
                is_best=True,
                group_id=group_id_counter,
                resolution_w=best_fp.resolution[0] if best_fp.resolution else 0,
                resolution_h=best_fp.resolution[1] if best_fp.resolution else 0,
                file_size=best_fp.file_size or 0,
                mtime=best_fp.mtime or 0.0,
                capture_date=best_fp.capture_date,
                format_str=best_fp.format_str or "",
                compression_format=best_fp.compression_format or "",
                format_details=best_fp.format_details or "",
                has_alpha=bool(best_fp.has_alpha),
                bit_depth=best_fp.bit_depth or 0,
                mipmap_count=best_fp.mipmap_count or 1,
                texture_type=best_fp.texture_type or "2D",
                color_space=best_fp.color_space or "sRGB",
                found_by=BEST_FILE_METHOD_NAME,
                channel=best_fp.channel,
                distance=0,
            )
            children.append(best_node)
            self.path_to_group_id[str(best_fp.path)] = group_id_counter
            self.group_id_to_best_path[group_id_counter] = str(best_fp.path)

            for dup_fp, score, method in dups:
                dup_node = ResultNode(
                    path=str(dup_fp.path),
                    is_best=False,
                    group_id=group_id_counter,
                    resolution_w=dup_fp.resolution[0] if dup_fp.resolution else 0,
                    resolution_h=dup_fp.resolution[1] if dup_fp.resolution else 0,
                    file_size=dup_fp.file_size or 0,
                    mtime=dup_fp.mtime or 0.0,
                    capture_date=dup_fp.capture_date,
                    format_str=dup_fp.format_str or "",
                    compression_format=dup_fp.compression_format or "",
                    format_details=dup_fp.format_details or "",
                    has_alpha=bool(dup_fp.has_alpha),
                    bit_depth=dup_fp.bit_depth or 0,
                    mipmap_count=dup_fp.mipmap_count or 1,
                    texture_type=dup_fp.texture_type or "2D",
                    color_space=dup_fp.color_space or "sRGB",
                    found_by=method,
                    distance=score,
                    channel=dup_fp.channel,
                )
                children.append(dup_node)
                self.path_to_group_id[str(dup_fp.path)] = group_id_counter

            children.sort(key=lambda n: (n.is_best, n.distance), reverse=True)
            group_node.children = children
            self.groups_data[group_id_counter] = group_node
            group_id_counter += 1
        self.sorted_group_ids = list(self.groups_data.keys())

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return len(self.sorted_group_ids)
        node = parent.internalPointer()
        if isinstance(node, GroupNode):
            return len(node.children)
        return 0

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 4

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if isinstance(node, GroupNode):
            return QModelIndex()
        if isinstance(node, ResultNode):
            group_id = node.group_id
            if group_id in self.groups_data:
                try:
                    row = self.sorted_group_ids.index(group_id)
                    return self.createIndex(row, 0, self.groups_data[group_id])
                except ValueError:
                    pass
        return QModelIndex()

    def index(self, row, col, parent: QModelIndex | None = None):
        parent = parent or QModelIndex()
        if not parent.isValid():
            if 0 <= row < len(self.sorted_group_ids):
                group_id = self.sorted_group_ids[row]
                return self.createIndex(row, col, self.groups_data[group_id])
        else:
            parent_node = parent.internalPointer()
            if isinstance(parent_node, GroupNode) and 0 <= row < len(parent_node.children):
                return self.createIndex(row, col, parent_node.children[row])
        return QModelIndex()

    def hasChildren(self, parent: QModelIndex | None = None) -> bool:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return bool(self.groups_data)
        node = parent.internalPointer()
        if isinstance(node, GroupNode):
            return len(node.children) > 0
        return False

    def canFetchMore(self, parent):
        if not parent.isValid():
            return False
        node = parent.internalPointer()
        return isinstance(node, GroupNode) and not node.fetched and node.group_id not in self.pending_fetches

    def fetchMore(self, parent):
        if not self.canFetchMore(parent):
            return
        node = parent.internalPointer()
        group_id = node.group_id
        self.pending_fetches[group_id] = QPersistentModelIndex(parent)

        # Use the global DB_SERVICE indirectly via the task (no DB path needed)
        task = LanceDBGroupFetcherTask(group_id)
        task.signals.finished.connect(self._on_fetch_finished)
        task.signals.error.connect(self._on_fetch_error)
        QThreadPool.globalInstance().start(task)

    @Slot(list, int)
    def _on_fetch_finished(self, children_dicts: list[dict], group_id: int):
        if group_id not in self.pending_fetches:
            return
        persistent_index = self.pending_fetches.pop(group_id)
        if not persistent_index.isValid():
            return
        parent_index = QModelIndex(persistent_index)
        node = parent_index.internalPointer()
        if not isinstance(node, GroupNode):
            return

        children = []
        for c in children_dicts:
            try:
                c["group_id"] = int(c.get("group_id", group_id))
                c["is_best"] = bool(c.get("is_best", False))
                c["distance"] = int(float(c.get("distance", 0)))
                c["file_size"] = c.get("file_size") or 0
                c["resolution_w"] = c.get("resolution_w") or 0
                c["resolution_h"] = c.get("resolution_h") or 0
                children.append(ResultNode.from_dict(c))
            except Exception:
                pass

        children.sort(key=lambda n: (n.is_best, n.distance), reverse=True)
        for child in children:
            self.path_to_group_id[child.path] = group_id
            if child.is_best:
                self.group_id_to_best_path[group_id] = child.path

        if not children:
            self.beginRemoveRows(parent_index, 0, 0)
            node.children = []
            node.count = 0
            node.fetched = True
            self.endRemoveRows()
            self.fetch_completed.emit(parent_index)
            return

        node.children[0] = children[0]
        self.dataChanged.emit(self.index(0, 0, parent_index), self.index(0, self.columnCount() - 1, parent_index))
        if len(children) > 1:
            self.beginInsertRows(parent_index, 1, len(children) - 1)
            node.children.extend(children[1:])
            self.endInsertRows()

        node.count = len(node.children)
        node.fetched = True
        self.fetch_completed.emit(parent_index)

    @Slot(str)
    def _on_fetch_error(self, error_msg: str):
        app_logger.error(f"Fetch Error: {error_msg}")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if role == Qt.ItemDataRole.UserRole:
            return node
        if role == SortRole:
            if isinstance(node, GroupNode):
                return -1
            if node.path == "loading_dummy":
                return -999
            if node.is_best:
                return 102
            method = node.found_by
            if method == "xxHash":
                return 101
            if method == "dHash":
                return 100
            if method == "pHash":
                return 99
            return node.distance

        if role == Qt.ItemDataRole.DisplayRole:
            return self._get_display_data(index, node)

        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                return None
            if isinstance(node, GroupNode):
                if not node.fetched:
                    return Qt.CheckState.Unchecked
                checked_count = sum(1 for c in node.children if self.check_states.get(c.path) == Qt.CheckState.Checked)
                if checked_count == 0:
                    return Qt.CheckState.Unchecked
                elif checked_count == len(node.children):
                    return Qt.CheckState.Checked
                else:
                    return Qt.CheckState.PartiallyChecked
            else:
                return self.check_states.get(node.path, Qt.CheckState.Unchecked)

        if role == Qt.ItemDataRole.FontRole and index.column() == 0:
            if isinstance(node, GroupNode) or (isinstance(node, ResultNode) and node.is_best):
                font = QFont()
                font.setBold(True)
                return font
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                font = QFont()
                font.setItalic(True)
                return font

        if role == Qt.ItemDataRole.BackgroundRole and isinstance(node, ResultNode) and node.is_best:
            return QBrush(QColor(UIConfig.Colors.BEST_FILE_BG))
        return None

    def _get_display_data(self, index, node):
        if isinstance(node, GroupNode):
            if index.column() != 0:
                return ""
            is_search = self.mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]
            return (
                f"{node.name} ({node.count} results)" if is_search else f"Group: {node.name} ({node.count} duplicates)"
            )
        if node.path == "loading_dummy":
            return "Loading data..." if index.column() == 0 else ""
        path = Path(node.path)
        col = index.column()
        if col == 0:
            display_name = path.name
            if node.channel:
                display_name += f" ({node.channel})"
            return display_name
        elif col == 1:
            if node.is_best:
                return f"[{BEST_FILE_METHOD_NAME}]"
            return METHOD_DISPLAY_NAMES.get(node.found_by, f"{node.distance}%")
        elif col == 2:
            return str(path.parent)
        elif col == 3:
            return _format_metadata_string(node)
        return ""

    def setData(self, index, value, role):
        if not (role == Qt.ItemDataRole.CheckStateRole and index.column() == 0):
            return super().setData(index, value, role)
        node = index.internalPointer()
        if not node or (isinstance(node, ResultNode) and node.path == "loading_dummy"):
            return False
        new_state = Qt.CheckState(value)

        if isinstance(node, GroupNode):
            state_to_apply = Qt.CheckState.Checked if new_state == Qt.CheckState.PartiallyChecked else new_state
            if node.fetched:
                for child in node.children:
                    self.check_states[child.path] = state_to_apply
                if node.children:
                    self.dataChanged.emit(
                        self.index(0, 0, index),
                        self.index(len(node.children) - 1, 0, index),
                        [Qt.ItemDataRole.CheckStateRole],
                    )
            else:
                self.fetchMore(index)
        else:
            self.check_states[node.path] = new_state
            parent = self.parent(index)
            if parent.isValid():
                self.dataChanged.emit(parent, parent, [Qt.ItemDataRole.CheckStateRole])
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        return True

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            node = index.internalPointer()
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                return flags
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return UIConfig.ResultsView.HEADERS[section]
        return None

    @Slot(int)
    def remove_group_by_id(self, group_id: int):
        if group_id in self.sorted_group_ids:
            row = self.sorted_group_ids.index(group_id)
            self.beginRemoveRows(QModelIndex(), row, row)
            self.sorted_group_ids.pop(row)
            if group_id in self.groups_data:
                del self.groups_data[group_id]
            self.endRemoveRows()

    def get_group_children(self, group_id: int) -> list[ResultNode]:
        if group_id in self.groups_data:
            node = self.groups_data[group_id]
            if node.fetched:
                return node.children
        return []

    def sort_results(self, sort_key: str):
        if not self.groups_data:
            return
        self.beginResetModel()
        if sort_key == UIConfig.ResultsView.SORT_OPTIONS[0]:
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].count, reverse=True)
        elif sort_key == UIConfig.ResultsView.SORT_OPTIONS[1]:
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].total_size, reverse=True)
        else:
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].name)
        self.endResetModel()

    def set_all_checks(self, state: Qt.CheckState):
        self._set_check_state_for_all(lambda n: state)

    def select_all_except_best(self):
        self._set_check_state_for_all(lambda n: Qt.CheckState.Checked if not n.is_best else Qt.CheckState.Unchecked)

    def invert_selection(self):
        self._set_check_state_for_all(
            lambda n: Qt.CheckState.Unchecked
            if self.check_states.get(n.path) == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )

    def _set_check_state_for_all(self, logic):
        if not self.groups_data:
            return
        for gid in self.sorted_group_ids:
            node = self.groups_data[gid]
            if node.fetched:
                for child in node.children:
                    self.check_states[child.path] = logic(child)
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def get_link_map_for_paths(self, paths: list[Path]) -> dict[Path, Path]:
        link_map = {}
        for path in paths:
            path_str = str(path)
            if (
                (gid := self.path_to_group_id.get(path_str))
                and (best := self.group_id_to_best_path.get(gid))
                and path_str != best
            ):
                link_map[path] = Path(best)
        return link_map

    def get_checked_paths(self) -> list[Path]:
        return [Path(p) for p, s in self.check_states.items() if s == Qt.CheckState.Checked]

    def get_summary_text(self) -> str:
        num = len(self.sorted_group_ids)
        total = sum(max(0, d.count - 1) for d in self.groups_data.values())
        return (
            f"({num} Groups, ~{total} duplicates)"
            if self.mode == ScanMode.DUPLICATES
            else f"({sum(d.count for d in self.groups_data.values())} results)"
        )

    def remove_deleted_paths(self, paths: list[Path]):
        deleted = {os.path.normcase(str(p)) for p in paths}
        to_remove = []
        for gid, data in self.groups_data.items():
            if not data.fetched:
                continue
            orig = len(data.children)
            data.children = [f for f in data.children if os.path.normcase(str(f.path)) not in deleted]
            data.count = len(data.children)
            if data.count != orig:
                min_items = 2 if self.mode == ScanMode.DUPLICATES else 1
                if data.count < min_items:
                    to_remove.append(gid)
                elif self.mode == ScanMode.DUPLICATES and not any(f.is_best for f in data.children) and data.children:
                    old = data.children[0]
                    new = ResultNode(
                        **{f.name: getattr(old, f.name) for f in fields(ResultNode) if f.name != "is_best"},
                        is_best=True,
                    )
                    data.children[0] = new
                    self.group_id_to_best_path[gid] = new.path
        for gid in to_remove:
            if gid in self.groups_data:
                del self.groups_data[gid]
        self.sorted_group_ids = [g for g in self.sorted_group_ids if g not in to_remove]
        for p in list(self.check_states.keys()):
            if os.path.normcase(p) in deleted:
                del self.check_states[p]

    def get_paths_for_group_sync(self, group_id: int) -> list[Path]:
        """
        Synchronously retrieves all file paths associated with a group ID.
        Uses the thread-safe DB_SERVICE to avoid file locks.
        """
        if group_id in self.groups_data:
            node = self.groups_data[group_id]
            if node.fetched:
                return [Path(c.path) for c in node.children]

        # Use Global Singleton
        try:
            rows = DB_SERVICE.get_files_by_group(group_id)
            return [Path(r["path"]) for r in rows]
        except Exception:
            return []

    def get_best_path_for_group_lazy(self, group_id: int) -> str | None:
        """
        Retrieves the path of the 'best' file in a group.
        If not cached, returns None to avoid blocking the UI thread with a DB query.
        The GroupDelegate will handle the lazy load via fetchMore or display a placeholder.
        """
        if group_id in self.group_id_to_best_path:
            return self.group_id_to_best_path[group_id]

        # Optimization: We do not query DB_SERVICE here to prevent micro-stutters
        # in the UI rendering loop. If the data isn't cached, we let the
        # fetchMore mechanism populate it eventually.
        return None


class ImagePreviewModel(QAbstractListModel):
    file_missing = Signal(Path)

    def __init__(self, thread_pool: QThreadPool, parent=None):
        super().__init__(parent)
        self.group_id = -1
        self.items = []
        self.pixmap_cache = OrderedDict()
        self.vfx_flags = {}
        self.CACHE_SIZE_LIMIT = 200
        self.thread_pool = thread_pool
        self.loading_paths = set()
        self.active_tasks = {}
        self.tonemap_mode = "none"
        self.target_size = 250
        self.error_paths = {}

    def set_tonemap_mode(self, mode: str):
        if self.tonemap_mode != mode:
            self.tonemap_mode = mode
            self.clear_cache()

    def set_target_size(self, size: int):
        if self.target_size != size:
            self.target_size = size
            self.clear_cache()

    def clear_cache(self):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.pixmap_cache.clear()
        self.vfx_flags.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.endResetModel()

    def cancel_all_tasks(self):
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()

    def set_items_from_list(self, items: list[ResultNode]):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.group_id = -1
        self.items = items
        self.pixmap_cache.clear()
        self.vfx_flags.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is None:
            parent = QModelIndex()
        return len(self.items) if not parent.isValid() else 0

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return None
        item = self.items[index.row()]
        ui_key = f"{item.path}_{item.channel or 'full'}"

        if role == Qt.ItemDataRole.UserRole:
            return item
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{item.path}\nChannel: {item.channel or 'Full'}"
        if role == VfxRole:
            return self.vfx_flags.get(ui_key, False)

        if role == Qt.ItemDataRole.DecorationRole:
            if ui_key in self.pixmap_cache:
                return self.pixmap_cache[ui_key]
            if ui_key not in self.loading_paths:
                self.loading_paths.add(ui_key)
                loader = ImageLoader(
                    item.path, item.mtime, self.target_size, self.tonemap_mode, True, item.channel, ui_key
                )
                loader.signals.result.connect(self._on_image_loaded)
                loader.signals.error.connect(self._on_image_error)
                self.active_tasks[ui_key] = loader
                self.thread_pool.start(loader)
            return None
        return None

    @Slot(str, object)
    def _on_image_loaded(self, ui_key: str, pil_img: object):
        if ui_key in self.loading_paths:
            self.loading_paths.remove(ui_key)
            if ui_key in self.active_tasks:
                del self.active_tasks[ui_key]
            if pil_img:
                try:
                    if hasattr(pil_img, "info"):
                        self.vfx_flags[ui_key] = pil_img.info.get("is_vfx", False)
                    self.pixmap_cache[ui_key] = QPixmap.fromImage(ImageQt(pil_img))
                    if len(self.pixmap_cache) > self.CACHE_SIZE_LIMIT:
                        self.pixmap_cache.popitem(last=False)
                except Exception:
                    pass
            self._emit_data_changed(ui_key)

    @Slot(str, str)
    def _on_image_error(self, ui_key: str, error_msg: str):
        if ui_key in self.loading_paths:
            self.loading_paths.remove(ui_key)
            if ui_key in self.active_tasks:
                del self.active_tasks[ui_key]
            if "File not found" in error_msg:
                for item in self.items:
                    if str(item.path) in ui_key:
                        self.file_missing.emit(Path(item.path))
                        return
            self.error_paths[ui_key] = error_msg
            self._emit_data_changed(ui_key)

    def _emit_data_changed(self, ui_key):
        for i, item in enumerate(self.items):
            if f"{item.path}_{item.channel or 'full'}" == ui_key:
                idx = self.index(i, 0)
                self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])
                break

    def get_row_for_path(self, path: Path) -> int | None:
        s = str(path)
        for i, item in enumerate(self.items):
            if item.path == s:
                return i
        return None


class ResultsProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_similarity = 0

    def set_similarity_filter(self, value: int):
        if self._min_similarity != value:
            self._min_similarity = value
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if self._min_similarity == 0:
            return True
        if not parent.isValid():
            return True
        idx = self.sourceModel().index(row, 0, parent)
        if not idx.isValid():
            return False
        node = idx.internalPointer()
        if not node or isinstance(node, GroupNode):
            return True
        if node.path == "loading_dummy":
            return True
        if node.is_best or node.found_by in METHOD_DISPLAY_NAMES:
            return True
        return node.distance >= self._min_similarity

    def lessThan(self, left, right):
        l_data = self.sourceModel().data(left, SortRole)
        r_data = self.sourceModel().data(right, SortRole)
        if l_data is None or r_data is None:
            return super().lessThan(left, right)
        try:
            return float(l_data) < float(r_data)
        except (ValueError, TypeError):
            return super().lessThan(left, right)
