# app/ui/widgets.py
"""
Contains small, reusable custom QWidget subclasses used throughout the GUI.
"""

from collections import OrderedDict
from typing import ClassVar

# Added QModelIndex to fix F821
from PySide6.QtCore import QModelIndex, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QListView, QWidget

from app.shared.constants import CompareMode, UIConfig


class PaintUtilsMixin:
    """Reusable painting logic for transparency and tiling."""

    _checkered_cache: ClassVar[OrderedDict] = OrderedDict()
    _CACHE_MAX = 64

    def get_checkered_pixmap(self, size, alpha) -> QPixmap:
        key = (size.width(), size.height(), alpha)
        if key in self._checkered_cache:
            self._checkered_cache.move_to_end(key)
            return self._checkered_cache[key]

        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        with QPainter(pixmap) as painter:
            light = QColor(200, 200, 200, alpha)
            dark = QColor(150, 150, 150, alpha)
            tile_size = 10
            cols = (size.width() // tile_size) + 1
            rows = (size.height() // tile_size) + 1

            for y in range(rows):
                for x in range(cols):
                    painter.fillRect(
                        x * tile_size, y * tile_size, tile_size, tile_size, light if (x + y) % 2 == 0 else dark
                    )

        self._checkered_cache[key] = pixmap
        if len(self._checkered_cache) > self._CACHE_MAX:
            self._checkered_cache.popitem(last=False)
        return pixmap

    def get_draw_rects(self, base_rect: QRect, bounds: QRect, is_tiling: bool) -> list[QRect]:
        if not is_tiling:
            return [base_rect]
        rects = []
        w, h = base_rect.width(), base_rect.height()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tiled_rect = base_rect.translated(dx * w, dy * h)
                if tiled_rect.intersects(bounds):
                    rects.append(tiled_rect)
        return rects

    def draw_tile_borders(self, painter: QPainter, rects: list[QRect]):
        painter.setPen(QPen(QColor(255, 255, 0, 80), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r in rects:
            painter.drawRect(r)


class AlphaBackgroundWidget(QWidget, PaintUtilsMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.error_message = ""
        self.bg_alpha = 255
        self.is_transparency_enabled = True
        self.is_tiling_enabled = False

    def set_alpha(self, alpha: int):
        self.bg_alpha = alpha
        self.update()

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state
        self.update()

    def set_tiling_enabled(self, state: bool):
        self.is_tiling_enabled = state
        self.update()

    def setPixmap(self, pixmap: QPixmap):
        self.pixmap, self.error_message = pixmap, ""
        self.update()

    def setError(self, message: str):
        self.error_message, self.pixmap = message, QPixmap()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.is_transparency_enabled:
                painter.drawPixmap(self.rect(), self.get_checkered_pixmap(self.size(), self.bg_alpha))
            else:
                painter.fillRect(self.rect(), self.palette().base())

            if self.error_message:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.error_message)
            elif not self.pixmap.isNull():
                scaled = self.pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                base_rect = QRect(
                    (self.width() - scaled.width()) // 2,
                    (self.height() - scaled.height()) // 2,
                    scaled.width(),
                    scaled.height(),
                )
                draw_rects = self.get_draw_rects(base_rect, self.rect(), self.is_tiling_enabled)
                for r in draw_rects:
                    painter.drawPixmap(r, scaled)
                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)


class ImageCompareWidget(QWidget, PaintUtilsMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap1, self.pixmap2 = QPixmap(), QPixmap()
        self.mode = CompareMode.WIPE
        self.wipe_x = self.width() // 2
        self.overlay_alpha = 128
        self.bg_alpha = 255
        self.is_dragging = False
        self.is_transparency_enabled = True
        self.is_tiling_enabled = False
        self.setMouseTracking(True)

    def setPixmaps(self, p1: QPixmap, p2: QPixmap):
        self.pixmap1, self.pixmap2 = p1, p2
        self.wipe_x = self.width() // 2
        self.update()

    def setMode(self, mode: CompareMode):
        self.mode = mode
        self.update()

    def setOverlayAlpha(self, alpha: int):
        self.overlay_alpha = alpha
        if self.mode == CompareMode.OVERLAY:
            self.update()

    def set_alpha(self, alpha: int):
        self.bg_alpha = alpha
        self.update()

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state
        self.update()

    def set_tiling_enabled(self, state: bool):
        self.is_tiling_enabled = state
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.is_transparency_enabled:
                painter.drawPixmap(self.rect(), self.get_checkered_pixmap(self.size(), self.bg_alpha))
            else:
                painter.fillRect(self.rect(), self.palette().base())

            if self.pixmap1.isNull() or self.pixmap2.isNull():
                return
            base_rect = self._calculate_scaled_rect(self.pixmap1)
            draw_rects = self.get_draw_rects(base_rect, self.rect(), self.is_tiling_enabled)

            if self.mode == CompareMode.WIPE:
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap2)
                painter.setClipRect(QRect(0, 0, self.wipe_x, self.height()))
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap1)
                painter.setClipping(False)
                painter.setPen(QPen(QColor(UIConfig.Colors.DIVIDER), 2))
                painter.drawLine(self.wipe_x, 0, self.wipe_x, self.height())
                painter.setBrush(QColor(UIConfig.Colors.DIVIDER))
                painter.drawEllipse(QPoint(self.wipe_x, self.height() // 2), 8, 8)
                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)
            elif self.mode == CompareMode.OVERLAY:
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap1)
                painter.setOpacity(self.overlay_alpha / 255.0)
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap2)
                painter.setOpacity(1.0)
                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)

    def _calculate_scaled_rect(self, pixmap: QPixmap) -> QRect:
        scaled = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        return QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.mode == CompareMode.WIPE
            and abs(event.pos().x() - self.wipe_x) < 10
        ):
            self.is_dragging = True
            self.setCursor(Qt.CursorShape.SplitHCursor)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.wipe_x = max(0, min(self.width(), event.pos().x()))
            self.update()
        elif self.mode == CompareMode.WIPE and abs(event.pos().x() - self.wipe_x) < 10:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)


class ResizedListView(QListView):
    resized = Signal()
    channel_hovered = Signal(QModelIndex, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._last_hovered_index = QModelIndex()
        self._last_channel = None
        self.preview_size = 250

    def set_preview_size(self, size: int):
        self.preview_size = size

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        index = self.indexAt(event.pos())

        def clear_hover():
            if self._last_channel is not None:
                self._last_channel = None
                self._last_hovered_index = QModelIndex()
                self.channel_hovered.emit(QModelIndex(), None)

        if not index.isValid():
            clear_hover()
            return

        rect = self.visualRect(index)
        thumb_rect = QRect(rect.x() + 5, rect.y() + 5, self.preview_size, self.preview_size)

        if not thumb_rect.contains(event.pos()):
            clear_hover()
            return

        local_pos = event.pos() - thumb_rect.topLeft()
        rel_x = local_pos.x() / thumb_rect.width()
        rel_y = local_pos.y() / thumb_rect.height()
        channel = ("R" if rel_x < 0.5 else "G") if rel_y < 0.5 else ("B" if rel_x < 0.5 else "A")

        if index != self._last_hovered_index or channel != self._last_channel:
            self._last_hovered_index = index
            self._last_channel = channel
            self.channel_hovered.emit(index, channel)

    def leaveEvent(self, event):
        if self._last_channel is not None:
            self._last_channel = None
            self.channel_hovered.emit(QModelIndex(), None)
        super().leaveEvent(event)
