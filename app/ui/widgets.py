# app/ui/widgets.py
"""
Contains small, reusable custom QWidget subclasses used throughout the GUI.
"""

from collections import OrderedDict
from typing import ClassVar

from PySide6.QtCore import QModelIndex, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QSizePolicy,
    QWidget,
)

from app.shared.constants import CompareMode, UIConfig


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


class DragDropLineEdit(QLineEdit):
    """A QLineEdit that accepts file/folder drops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.setText(path)
                event.acceptProposedAction()


class DragDropLabel(QLabel):
    """A QLabel that accepts image file drops for sample search."""

    file_dropped = Signal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)
                event.acceptProposedAction()


class ResizedListView(QListView):
    """
    A QListView that emits a signal when resized and tracks mouse hover.
    Required by ImageViewerPanel to update layouts and handle channel previews.
    """

    resized = Signal()
    channel_hovered = Signal(QModelIndex, object)  # index, channel_str (or None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._preview_size = 250

    def set_preview_size(self, size: int):
        self._preview_size = size

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        index = self.indexAt(event.pos())

        if not index.isValid():
            self.channel_hovered.emit(QModelIndex(), None)
            return

        # Calculate if mouse is over a channel "hotspot" (corners)
        channel = None

        # Get the visual rectangle of the item in the view
        rect = self.visualRect(index)
        local_pos = event.pos() - rect.topLeft()

        # Define the thumbnail area based on View Mode
        # Matches logic in ImageItemDelegate.paint
        thumb_size = self._preview_size
        thumb_rect = QRect()

        if self.viewMode() == QListView.ViewMode.IconMode: # Grid View
            # In Grid, thumb is centered horizontally, padded 5px from top
            x = (rect.width() - thumb_size) // 2
            y = 5
            thumb_rect = QRect(x, y, thumb_size, thumb_size)
        else: # List View
            # In List, thumb is left-aligned with padding
            thumb_rect = QRect(5, 5, thumb_size, thumb_size)

        # Check corners if mouse is inside the thumbnail area
        if thumb_rect.contains(local_pos):
            CORNER_SIZE = 50 # Size of the hotspot in pixels
            rel_x = local_pos.x() - thumb_rect.x()
            rel_y = local_pos.y() - thumb_rect.y()

            w, h = thumb_rect.width(), thumb_rect.height()

            # Top-Left = Red
            if rel_x < CORNER_SIZE and rel_y < CORNER_SIZE:
                channel = "R"
            # Top-Right = Green
            elif rel_x > w - CORNER_SIZE and rel_y < CORNER_SIZE:
                channel = "G"
            # Bottom-Left = Blue
            elif rel_x < CORNER_SIZE and rel_y > h - CORNER_SIZE:
                channel = "B"
            # Bottom-Right = Alpha
            elif rel_x > w - CORNER_SIZE and rel_y > h - CORNER_SIZE:
                channel = "A"

        self.channel_hovered.emit(index, channel)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.channel_hovered.emit(QModelIndex(), None)


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
        # Draw 3x3 grid around center
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tiled_rect = base_rect.translated(dx * w, dy * h)
                # Optimization: only draw if visible
                if tiled_rect.intersects(bounds):
                    rects.append(tiled_rect)
        return rects

    def draw_tile_borders(self, painter: QPainter, rects: list[QRect]):
        painter.setPen(QPen(QColor(255, 255, 0, 80), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r in rects:
            painter.drawRect(r)


class InteractiveViewMixin:
    """
    Mixin to handle Zoom and Pan logic.
    Emits signals to sync with other widgets.
    """

    view_changed = Signal(float, QPoint)  # zoom, offset

    def __init__(self):
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._is_panning = False
        self._last_mouse_pos = QPoint()

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self.update()

    def set_sync_view(self, zoom: float, offset: QPoint):
        """Called by external controller/signal to sync state."""
        self._zoom = zoom
        self._offset = offset
        self.update()

    def _calculate_geometry(self, pixmap_size: QSize, widget_size: QSize) -> QRect:
        if pixmap_size.isEmpty():
            return QRect()

        # 1. Base Fit (Aspect Ratio)
        scaled_size = pixmap_size.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)

        # 2. Apply Zoom
        final_w = int(scaled_size.width() * self._zoom)
        final_h = int(scaled_size.height() * self._zoom)

        # 3. Center
        center_x = (widget_size.width() - final_w) // 2
        center_y = (widget_size.height() - final_h) // 2

        # 4. Apply Pan Offset
        x = center_x + self._offset.x()
        y = center_y + self._offset.y()

        return QRect(x, y, final_w, final_h)

    # --- Event Handlers (must be hooked or called by QWidget subclass) ---

    def handle_wheel_event(self, event: QWheelEvent):
        # Zoom logic
        delta = event.angleDelta().y()
        zoom_factor = 1.15

        if delta > 0:
            self._zoom *= zoom_factor
        else:
            self._zoom /= zoom_factor
            # Limit min zoom
            if self._zoom < 0.1:
                self._zoom = 0.1

        # Notify sync
        if hasattr(self, "view_changed"):
            self.view_changed.emit(self._zoom, self._offset)
        self.update()

    def handle_mouse_press(self, event: QMouseEvent):
        # Middle button OR Left+Ctrl for Pan
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            if hasattr(self, "setCursor"):
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def handle_mouse_move(self, event: QMouseEvent):
        if self._is_panning:
            delta = event.pos() - self._last_mouse_pos
            self._offset += delta
            self._last_mouse_pos = event.pos()
            if hasattr(self, "view_changed"):
                self.view_changed.emit(self._zoom, self._offset)
            self.update()

    def handle_mouse_release(self, event: QMouseEvent):
        if self._is_panning:
            self._is_panning = False
            if hasattr(self, "setCursor"):
                self.setCursor(Qt.CursorShape.ArrowCursor)


class AlphaBackgroundWidget(QWidget, PaintUtilsMixin, InteractiveViewMixin):
    view_changed = Signal(float, QPoint)  # Re-declare signal for QWidget meta-object

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        InteractiveViewMixin.__init__(self)
        self.pixmap = QPixmap()
        self.error_message = ""
        self.bg_alpha = 255
        self.is_transparency_enabled = True
        self.is_tiling_enabled = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        with QPainter(self) as painter:
            # Draw Background
            if self.is_transparency_enabled:
                painter.drawPixmap(self.rect(), self.get_checkered_pixmap(self.size(), self.bg_alpha))
            else:
                painter.fillRect(self.rect(), self.palette().base())

            if self.error_message:
                painter.setPen(self.palette().text().color())
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.error_message)
            elif not self.pixmap.isNull():
                # Use Mixin calculation
                target_rect = self._calculate_geometry(self.pixmap.size(), self.size())

                # Get draw rects (handles tiling)
                draw_rects = self.get_draw_rects(target_rect, self.rect(), self.is_tiling_enabled)

                # Use Smooth Transform if zooming
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom != 1.0)

                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap)

                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)

    # --- Input Events ---
    def wheelEvent(self, event):
        self.handle_wheel_event(event)

    def mousePressEvent(self, event):
        # Allow simple left drag for this widget if no modifiers
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self.handle_mouse_press(event)

    def mouseMoveEvent(self, event):
        self.handle_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.handle_mouse_release(event)


class ImageCompareWidget(QWidget, PaintUtilsMixin, InteractiveViewMixin):
    view_changed = Signal(float, QPoint)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        InteractiveViewMixin.__init__(self)
        self.pixmap1, self.pixmap2 = QPixmap(), QPixmap()
        self.mode = CompareMode.WIPE
        self.wipe_x = self.width() // 2
        self.overlay_alpha = 128
        self.bg_alpha = 255
        self.is_dragging_wipe = False  # Specific for wipe handle
        self.is_transparency_enabled = True
        self.is_tiling_enabled = False
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        with QPainter(self) as painter:
            if self.is_transparency_enabled:
                painter.drawPixmap(self.rect(), self.get_checkered_pixmap(self.size(), self.bg_alpha))
            else:
                painter.fillRect(self.rect(), self.palette().base())

            if self.pixmap1.isNull() or self.pixmap2.isNull():
                return

            # Geometry calculation (based on pixmap1)
            target_rect = self._calculate_geometry(self.pixmap1.size(), self.size())
            draw_rects = self.get_draw_rects(target_rect, self.rect(), self.is_tiling_enabled)

            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom != 1.0)

            if self.mode == CompareMode.WIPE:
                # Layer 2 (Bottom)
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap2)

                # Layer 1 (Top) - Clipped
                painter.setClipRect(QRect(0, 0, self.wipe_x, self.height()))
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap1)

                painter.setClipping(False)

                # Draw Wipe Handle
                painter.setPen(QPen(QColor(UIConfig.Colors.DIVIDER), 2))
                painter.drawLine(self.wipe_x, 0, self.wipe_x, self.height())
                painter.setBrush(QColor(UIConfig.Colors.DIVIDER))
                painter.drawEllipse(QPoint(self.wipe_x, self.height() // 2), 8, 8)

                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)

            elif self.mode == CompareMode.OVERLAY:
                # Layer 1
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap1)

                # Layer 2 (Transparent)
                painter.setOpacity(self.overlay_alpha / 255.0)
                for r in draw_rects:
                    painter.drawPixmap(r, self.pixmap2)
                painter.setOpacity(1.0)

                if self.is_tiling_enabled:
                    self.draw_tile_borders(painter, draw_rects)

    # --- Input Events ---
    def wheelEvent(self, event):
        self.handle_wheel_event(event)

    def mousePressEvent(self, event):
        # Priority: Wipe Handle -> Pan
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.mode == CompareMode.WIPE
            and abs(event.pos().x() - self.wipe_x) < 15
        ):
            self.is_dragging_wipe = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            # Allow left drag for pan in Wipe mode IF far from handle
            if event.button() == Qt.MouseButton.LeftButton and (
                self.mode != CompareMode.WIPE or not self.is_dragging_wipe
            ):
                self._is_panning = True
                self._last_mouse_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self.handle_mouse_press(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging_wipe:
            self.wipe_x = max(0, min(self.width(), event.pos().x()))
            self.update()
        else:
            self.handle_mouse_move(event)

            # Cursor update for hover
            if not self._is_panning and self.mode == CompareMode.WIPE:
                if abs(event.pos().x() - self.wipe_x) < 15:
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging_wipe:
            self.is_dragging_wipe = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.handle_mouse_release(event)
