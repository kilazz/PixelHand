# app/ui/delegates.py
"""
Contains custom QStyledItemDelegate classes for rendering items in views.
Handles custom painting for Group Grid View and Image Viewer (List/Grid).
"""

import logging
from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QModelIndex, QRect, QSize, Qt, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QStyle, QStyledItemDelegate

from app.domain.data_models import GroupNode, ResultNode
from app.imaging.processing import extract_channel_as_rgb
from app.shared.constants import BEST_FILE_METHOD_NAME, METHOD_DISPLAY_NAMES, UIConfig
from app.shared.utils import format_result_metadata
from app.ui.background_tasks import ImageLoader
from app.ui.qt_models import VfxRole
from app.ui.widgets import PaintUtilsMixin

app_logger = logging.getLogger("PixelHand.ui.delegates")


class GroupGridDelegate(QStyledItemDelegate):
    """
    Renders duplicate groups as "cards" or "folders" in a Grid View.
    Displays the best file's thumbnail, group name, and item count.
    """

    def __init__(self, thread_pool, parent=None):
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.padding = 10
        self.text_height = 60
        self.set_base_size(130)
        self.cache: dict[str, QPixmap] = {}
        self.loading_paths: set[str] = set()

    def set_base_size(self, size: int):
        self.thumb_size = size
        self.card_width = self.thumb_size + (self.padding * 2)
        self.card_height = self.thumb_size + (self.padding * 2) + self.text_height

    def sizeHint(self, option, index):
        return QSize(self.card_width, self.card_height)

    def paint(self, painter: QPainter, option, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        node = index.data(Qt.ItemDataRole.UserRole)
        rect = option.rect
        bg_rect = rect.adjusted(4, 4, -4, -4)

        if option.state & QStyle.State_Selected:
            painter.setBrush(QColor("#2A5075"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_rect, 8, 8)
        elif option.state & QStyle.State_MouseOver:
            painter.setBrush(QColor("#3A3A3A"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_rect, 8, 8)

        if not isinstance(node, GroupNode):
            painter.restore()
            return

        best_file_path = self._get_best_file_path(index, node)
        img_x = rect.x() + (rect.width() - self.thumb_size) // 2
        img_y = rect.y() + self.padding
        img_rect = QRect(img_x, img_y, self.thumb_size, self.thumb_size)

        pixmap_to_draw = None
        if best_file_path:
            if best_file_path in self.cache:
                pixmap_to_draw = self.cache[best_file_path]
            elif best_file_path not in self.loading_paths:
                self._start_loading(best_file_path)

        if pixmap_to_draw:
            scaled = pixmap_to_draw.scaled(
                img_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            x_pos = img_rect.x() + (img_rect.width() - scaled.width()) // 2
            y_pos = img_rect.y() + (img_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x_pos, y_pos, scaled)
        else:
            painter.setBrush(QColor("#202020"))
            painter.setPen(QColor("#404040"))
            painter.drawRoundedRect(img_rect, 6, 6)
            painter.setPen(QColor("#606060"))
            status_text = "Loading..." if best_file_path else "No Image"
            painter.drawText(img_rect, Qt.AlignmentFlag.AlignCenter, status_text)

        text_y_start = img_rect.bottom() + 10
        text_rect = QRect(rect.x() + 5, text_y_start, rect.width() - 10, 20)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor("#E0E0E0"))

        metrics = QFontMetrics(font)
        elided_name = metrics.elidedText(node.name, Qt.TextElideMode.ElideMiddle, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, elided_name)

        sub_text_rect = QRect(rect.x() + 5, text_rect.bottom(), rect.width() - 10, 15)
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)

        count_color = QColor("#FF9800") if node.count > 2 else QColor("#AAAAAA")
        painter.setPen(count_color)

        item_text = f"{node.count} files"
        size_mb = node.total_size / (1024**2)
        if size_mb > 0:
            item_text += f" • {size_mb:.1f} MB"

        painter.drawText(sub_text_rect, Qt.AlignmentFlag.AlignCenter, item_text)
        painter.restore()

    def _get_best_file_path(self, index: QModelIndex, node: GroupNode) -> str | None:
        model = index.model()
        while hasattr(model, "sourceModel"):
            model = model.sourceModel()

        if hasattr(model, "group_id_to_best_path"):
            path = model.group_id_to_best_path.get(node.group_id)
            if path:
                return path

        if node.fetched and node.children:
            return str(node.children[0].path)

        if hasattr(model, "get_best_path_for_group_lazy"):
            return model.get_best_path_for_group_lazy(node.group_id)
        return None

    def _start_loading(self, path_str: str):
        self.loading_paths.add(path_str)
        loader = ImageLoader(
            path_str=path_str,
            mtime=0,
            target_size=250,
            tonemap_mode="none",
            use_cache=True,
            ui_key=path_str,
        )
        loader.signals.result.connect(self._on_image_loaded)
        loader.signals.error.connect(self._on_image_error)
        self.thread_pool.start(loader)

    @Slot(str, object)
    def _on_image_loaded(self, key: str, pil_img: object):
        if key in self.loading_paths:
            self.loading_paths.remove(key)
        if pil_img:
            try:
                self.cache[key] = QPixmap.fromImage(ImageQt(pil_img))
                if self.parent() and isinstance(self.parent(), QAbstractItemView):
                    self.parent().viewport().update()
            except Exception as e:
                app_logger.error(f"Failed to convert grid thumbnail: {e}")

    @Slot(str, str)
    def _on_image_error(self, key: str, msg: str):
        if key in self.loading_paths:
            self.loading_paths.remove(key)


class ImageItemDelegate(QStyledItemDelegate, PaintUtilsMixin):
    """
    Renders individual files in the Right Panel (Image Viewer).
    """

    def __init__(self, preview_size: int, state, parent=None):
        super().__init__(parent)
        self.preview_size = preview_size
        self.state = state
        self.bg_alpha = 255
        self.is_transparency_enabled = True
        self.is_grid_mode = False
        self.bold_font = QFont()
        self.bold_font.setBold(True)
        self.bold_font_metrics = QFontMetrics(self.bold_font)
        self.regular_font = QFont()
        self.regular_font_metrics = QFontMetrics(self.regular_font)
        self._hover_index_key = None
        self._hover_channel = None
        self._channel_cache = {}
        self._CACHE_SIZE = 50

    def set_grid_mode(self, enabled: bool):
        self.is_grid_mode = enabled

    def set_bg_alpha(self, alpha: int):
        self.bg_alpha = alpha

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state

    def set_preview_size(self, size: int):
        self.preview_size = size
        self._channel_cache.clear()

    def sizeHint(self, option, index):
        if self.is_grid_mode:
            width = self.preview_size + 10
            # INCREASED: Give more vertical space for 3-line filenames + metadata
            text_area_height = 180
            return QSize(width, self.preview_size + text_area_height)
        else:
            return QSize(self.preview_size + 250, self.preview_size + 10)

    @Slot(QModelIndex, object)
    def set_hover_channel(self, index: QModelIndex, channel: str | None):
        item_key = None
        if index.isValid() and (item_data := index.data(Qt.ItemDataRole.UserRole)):
            item_key = id(item_data)

        if item_key != self._hover_index_key or channel != self._hover_channel:
            self._hover_index_key = item_key
            self._hover_channel = channel
            if self.parent():
                if hasattr(self.parent(), "list_view"):
                    self.parent().list_view.viewport().update()
                elif hasattr(self.parent(), "viewport"):
                    self.parent().viewport().update()

    def paint(self, painter: QPainter, option, index: QModelIndex):
        painter.save()
        try:
            painter.setClipRect(option.rect)
            item_data: ResultNode = index.data(Qt.ItemDataRole.UserRole)
            if not item_data:
                return
            self._draw_background(painter, option, item_data)
            self._draw_thumbnail(painter, option, index, item_data)
            self._draw_text_info(painter, option, item_data)
        finally:
            painter.restore()

    def _draw_background(self, painter, option, item_data):
        painter.fillRect(option.rect, option.palette.base())
        if option.state & QStyle.State_Selected:
            c = option.palette.highlight().color()
            c.setAlpha(80)
            painter.fillRect(option.rect, c)
        if self.state.is_candidate(item_data):
            painter.setPen(QPen(QColor(UIConfig.Colors.HIGHLIGHT), 2))
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

    def _draw_thumbnail(self, painter, option, index, item_data):
        if self.is_grid_mode:
            x = option.rect.x() + (option.rect.width() - self.preview_size) // 2
            y = option.rect.y() + 5
            thumb_rect = QRect(x, y, self.preview_size, self.preview_size)
        else:
            thumb_rect = option.rect.adjusted(5, 5, -(option.rect.width() - self.preview_size - 5), -5)

        if self.is_transparency_enabled:
            painter.drawPixmap(thumb_rect.topLeft(), self.get_checkered_pixmap(thumb_rect.size(), self.bg_alpha))

        raw_pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        is_vfx = index.data(VfxRole)
        pixmap_to_draw = raw_pixmap
        is_hovered = id(item_data) == self._hover_index_key

        if is_hovered and self._hover_channel:
            # Special VFX Handling for Alpha hover
            if self._hover_channel == "A" and is_vfx and raw_pixmap:
                key = f"{item_data.path}_BLACK_PLACEHOLDER"
                if key in self._channel_cache:
                    pixmap_to_draw = self._channel_cache[key]
                else:
                    black = QPixmap(raw_pixmap.size())
                    black.fill(QColor("black"))
                    self._channel_cache[key] = black
                    pixmap_to_draw = black
            elif raw_pixmap and not raw_pixmap.isNull():
                key = (item_data.path, self._hover_channel)
                if key in self._channel_cache:
                    pixmap_to_draw = self._channel_cache[key]
                else:
                    # REFACTOR: Use centralized logic from processing.py
                    pixmap_to_draw = self._generate_channel_pixmap_safe(raw_pixmap, self._hover_channel)
                    if pixmap_to_draw:
                        self._channel_cache[key] = pixmap_to_draw
                        if len(self._channel_cache) > self._CACHE_SIZE:
                            self._channel_cache.popitem()

        error_msg = None
        if self.parent() and hasattr(self.parent(), "model"):
            cache_key = f"{item_data.path}_{item_data.channel or 'full'}"
            error_msg = self.parent().model.error_paths.get(cache_key)

        if pixmap_to_draw and not pixmap_to_draw.isNull():
            scaled = pixmap_to_draw.scaled(
                thumb_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            x = thumb_rect.x() + (thumb_rect.width() - scaled.width()) // 2
            y = thumb_rect.y() + (thumb_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw channel label in the correct corner
            if is_hovered and self._hover_channel:
                self._draw_channel_label(painter, thumb_rect, self._hover_channel)
        elif error_msg:
            painter.save()
            painter.setPen(QColor(UIConfig.Colors.ERROR))
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "Error")
            painter.restore()
        else:
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "Loading...")

    def _generate_channel_pixmap_safe(self, qpixmap, channel):
        """Helper to safely bridge QPixmap -> PIL -> Logic -> QPixmap."""
        try:
            from PIL.ImageQt import ImageQt, fromqimage
            qimg = qpixmap.toImage()
            if qimg.format() != QImage.Format.Format_RGBA8888:
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)

            pil_img = fromqimage(qimg)

            # Use centralized logic
            processed_pil = extract_channel_as_rgb(pil_img, channel)

            if processed_pil:
                return QPixmap.fromImage(ImageQt(processed_pil))
        except Exception:
            pass
        return qpixmap

    def _draw_channel_label(self, painter, thumb_rect, ch):
        c = {"R": "#F55", "G": "#5F5", "B": "#5AF", "A": "#FFF"}.get(ch, "#CCC")

        # Determine position based on channel logic (Corners)
        size = 20
        padding = 5

        if ch == "R":
            x = thumb_rect.left() + padding
            y = thumb_rect.top() + padding
        elif ch == "G":
            x = thumb_rect.right() - size - padding
            y = thumb_rect.top() + padding
        elif ch == "B":
            x = thumb_rect.left() + padding
            y = thumb_rect.bottom() - size - padding
        elif ch == "A":
            x = thumb_rect.right() - size - padding
            y = thumb_rect.bottom() - size - padding
        else:
            # Fallback to Top-Left
            x = thumb_rect.left() + padding
            y = thumb_rect.top() + padding

        r = QRect(x, y, size, size)

        painter.save()
        painter.setBrush(QColor(c))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r, 5, 5)
        painter.setPen(QColor("black"))
        painter.setFont(self.bold_font)
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, ch)
        painter.restore()

    def _draw_text_info(self, painter, option, item_data):
        path = Path(item_data.path)
        filename = path.name
        if item_data.channel:
            filename += f" ({item_data.channel})"

        main_c = (
            option.palette.highlightedText().color()
            if option.state & QStyle.State_Selected
            else option.palette.text().color()
        )
        sec_c = QColor(main_c)
        sec_c.setAlpha(150)

        dist_text = (
            f"[{BEST_FILE_METHOD_NAME}]"
            if item_data.is_best
            else (METHOD_DISPLAY_NAMES.get(item_data.found_by) or f"{item_data.distance}%")
        )

        # Use common metadata logic with dots
        meta_text = format_result_metadata(item_data)

        if self.is_grid_mode:
            padding = 4
            x = option.rect.x() + padding
            w = option.rect.width() - (padding * 2)
            y = option.rect.y() + self.preview_size + 8

            # --- 1. Draw Filename (Bold) ---
            painter.setFont(self.bold_font)
            painter.setPen(main_c)

            flags = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap

            # Calculate dynamic height of filename
            bounding_rect = painter.boundingRect(QRect(x, y, w, 100), flags, filename)
            name_height = min(bounding_rect.height(), 60)  # Cap at roughly 3 lines

            painter.drawText(QRect(x, y, w, int(name_height)), flags, filename)

            # --- 2. Draw Metadata (Regular) ---
            # Move Y down based on filename height
            y += name_height + 4

            painter.setFont(self.regular_font)
            painter.setPen(sec_c)

            # Combine Score/Best with Metadata using dot separator
            full_grid_meta = f"{dist_text} • {meta_text}" if dist_text else meta_text

            remaining_height = option.rect.bottom() - y - padding

            if remaining_height > 10:
                painter.drawText(
                    QRect(x, int(y), w, int(remaining_height)),
                    flags,
                    full_grid_meta,
                )
        else:
            # --- List View ---
            r = option.rect.adjusted(self.preview_size + 15, 5, -5, -5)
            x, y = r.left(), r.top() + self.bold_font_metrics.ascent()
            painter.setFont(self.bold_font)
            painter.setPen(main_c)
            painter.drawText(x, y, self.bold_font_metrics.elidedText(filename, Qt.TextElideMode.ElideRight, r.width()))

            y += self.regular_font_metrics.height()
            painter.setFont(self.regular_font)
            painter.setPen(sec_c)

            # Combine Score/Best with Metadata using dot separator
            full_meta = f"{dist_text} • {meta_text}" if dist_text else meta_text

            painter.drawText(
                x, y, self.regular_font_metrics.elidedText(full_meta, Qt.TextElideMode.ElideRight, r.width())
            )

            y += self.regular_font_metrics.height()
            painter.drawText(
                x, y, self.regular_font_metrics.elidedText(str(path.parent), Qt.TextElideMode.ElideRight, r.width())
            )
