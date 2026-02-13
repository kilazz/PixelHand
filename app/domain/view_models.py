# app/domain/view_models.py
"""
Contains View-Model classes that manage UI state and logic, separating it from the view widgets.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QObject, QPoint, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap

from app.domain.data_models import ResultNode

if TYPE_CHECKING:
    from app.infrastructure.cache import AbstractThumbnailCache
    from app.ui.background_tasks import ImageLoader


@dataclass
class ViewerStateData:
    """Dataclass holding purely the state of the viewer settings."""
    zoom: float = 1.0
    pan_offset: QPoint = field(default_factory=lambda: QPoint(0, 0))
    channels: dict[str, bool] = field(default_factory=lambda: {"R": True, "G": True, "B": True, "A": True})
    current_group_id: int | None = None
    transparency_enabled: bool = True
    bg_alpha: int = 255


class ViewerState(QObject):
    """
    Manages the state for the ImageViewerPanel (Channels, Zoom, Group ID).
    """
    state_changed = Signal()  # Generic signal when state updates

    def __init__(self):
        super().__init__()
        self._data = ViewerStateData()

    @property
    def channels(self) -> dict[str, bool]:
        return self._data.channels

    def set_channel(self, channel: str, enabled: bool):
        if self._data.channels.get(channel) != enabled:
            self._data.channels[channel] = enabled
            self.state_changed.emit()

    def set_all_channels(self, enabled: bool):
        changed = False
        for k in self._data.channels:
            if self._data.channels[k] != enabled:
                self._data.channels[k] = enabled
                changed = True
        if changed:
            self.state_changed.emit()

    @property
    def current_group_id(self) -> int | None:
        return self._data.current_group_id

    @current_group_id.setter
    def current_group_id(self, value: int | None):
        if self._data.current_group_id != value:
            self._data.current_group_id = value
            # Reset zoom/pan on group change usually handled by view,
            # but we can reset state here too
            self.reset_transform()
            self.state_changed.emit()

    def reset_transform(self):
        self._data.zoom = 1.0
        self._data.pan_offset = QPoint(0, 0)


class ImageComparerState(QObject):
    """Manages the state and logic for the image comparison view."""

    candidates_changed = Signal(int)
    candidate_updated = Signal(str, str)
    images_loading = Signal()
    image_loaded = Signal(str, QPixmap)
    load_complete = Signal()
    load_error = Signal(str, str)

    def __init__(self, thread_pool: QThreadPool, cache_provider: "AbstractThumbnailCache"):
        """Initializes the state manager.

        Args:
            thread_pool: The shared QThreadPool.
            cache_provider: Injected cache service.
        """
        super().__init__()
        self.thread_pool = thread_pool
        self.cache_provider = cache_provider
        self._candidates: OrderedDict[ResultNode, None] = OrderedDict()
        self._pil_images: dict[str, Image.Image] = {}
        self._active_loaders: dict[str, ImageLoader] = {}

    def toggle_candidate(self, item_data: ResultNode):
        """Adds or removes an item from the comparison candidates list."""
        is_currently_candidate = item_data in self._candidates

        added_node, removed_node = None, None

        if is_currently_candidate:
            del self._candidates[item_data]
            removed_node = item_data
        else:
            self._candidates[item_data] = None
            added_node = item_data
            if len(self._candidates) > 2:
                removed_node, _ = self._candidates.popitem(last=False)

        self.candidates_changed.emit(len(self._candidates))

        added_path = str(added_node.path) if added_node else ""
        removed_path = str(removed_node.path) if removed_node else ""
        self.candidate_updated.emit(added_path, removed_path)

    def is_candidate(self, item_data: ResultNode) -> bool:
        return item_data in self._candidates

    def get_candidates(self) -> list[ResultNode]:
        return list(self._candidates.keys())

    def clear_candidates(self):
        if not self._candidates:
            return

        removed_nodes = self.get_candidates()
        self._candidates.clear()
        self.candidates_changed.emit(0)

        for node in removed_nodes:
            self.candidate_updated.emit("", str(node.path))

    def load_full_res_images(self, tonemap_mode: str):
        self.stop_loaders()
        self._pil_images.clear()

        candidates = self.get_candidates()
        if len(candidates) != 2:
            return

        self.images_loading.emit()
        from app.ui.background_tasks import ImageLoader

        for node in candidates:
            path_str = str(node.path)
            loader = ImageLoader(
                path_str=path_str,
                mtime=node.mtime,
                target_size=None,
                cache_provider=self.cache_provider,
                tonemap_mode=tonemap_mode,
                use_cache=False,
                channel_to_load=None,
            )
            loader.signals.result.connect(self._on_image_loaded)
            loader.signals.error.connect(self._on_load_error)
            self._active_loaders[path_str] = loader
            self.thread_pool.start(loader)

    @Slot(str, object)
    def _on_image_loaded(self, path_str: str, pil_img: object):
        if path_str not in self._active_loaders:
            return
        del self._active_loaders[path_str]

        if pil_img:
            self._pil_images[path_str] = pil_img
            try:
                q_img = ImageQt(pil_img)
                pixmap = QPixmap.fromImage(q_img)
                self.image_loaded.emit(path_str, pixmap)
            except Exception as e:
                self.load_error.emit(f"Display Error {Path(path_str).name}", str(e))

        if not self._active_loaders:
            self.load_complete.emit()

    @Slot(str, str)
    def _on_load_error(self, path_str: str, error_msg: str):
        if path_str in self._active_loaders:
            del self._active_loaders[path_str]
        self.load_error.emit(f"Failed to load {Path(path_str).name}", error_msg)
        if not self._active_loaders:
            self.load_complete.emit()

    def get_pil_images(self) -> list[Image.Image]:
        return [
            self._pil_images[str(node.path)] for node in self.get_candidates() if str(node.path) in self._pil_images
        ]

    def stop_loaders(self):
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
