import numpy as np
import PyOpenColorIO as OCIO

from app.shared.constants import OCIO_DIR


class ToneMapper:
    """
    Dynamically loads all .ocio configs from OCIO_DIR.
    """

    def __init__(self, default_view: str | None = None):
        self.configs = {}  # name -> OCIO.Config
        self.displays = {} # name -> default display
        self._available_views = [] # list of "[ConfigName] ViewName"
        self._view = None
        self.cpu = None
        self.xform = None

        self._load_configs()

        if self._available_views:
            # Try to set the requested view, or fallback to the first available
            if default_view and default_view in self._available_views:
                self.view = default_view
            else:
                self.view = self._available_views[0]

    def reload_configs(self):
        """Reloads all OCIO configs from disk."""
        current_view = self._view
        self._load_configs()
        if current_view in self._available_views:
            self.view = current_view
        elif self._available_views:
            self.view = self._available_views[0]
        else:
            self._view = None
            self.cpu = None
            self.xform = None

    def _load_configs(self):
        self.configs.clear()
        self.displays.clear()
        self._available_views.clear()

        if not OCIO_DIR.exists():
            return

        for ocio_file in OCIO_DIR.glob("*.ocio"):
            try:
                config_name = ocio_file.stem
                config = OCIO.Config.CreateFromFile(str(ocio_file))
                display = config.getDefaultDisplay()

                self.configs[config_name] = config
                self.displays[config_name] = display

                for view in config.getViews(display):
                    self._available_views.append(f"[{config_name}] {view}")
            except Exception as e:
                print(f"Failed to load OCIO config {ocio_file.name}: {e}")

    @property
    def view(self) -> str:
        return self._view

    @view.setter
    def view(self, value: str):
        if value not in self._available_views:
            return

        self._view = value

        # Parse "[ConfigName] ViewName"
        config_name = value[1:].split("] ", 1)[0]
        view_name = value.split("] ", 1)[1]

        config = self.configs[config_name]
        display = self.displays[config_name]

        OCIO.SetCurrentConfig(config)
        self.xform = OCIO.DisplayViewTransform(OCIO.ROLE_SCENE_LINEAR, display, view_name)
        self.cpu = config.getProcessor(self.xform).getDefaultCPUProcessor()

    @property
    def available_views(self) -> list[str]:
        return self._available_views

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def hdr_to_ldr(self, hdr: np.ndarray, clip: bool = True) -> np.ndarray:
        if not self.cpu:
            return np.clip(hdr, 0.0, 1.0) if clip else hdr

        channels = hdr.shape[-1]
        if channels not in [3, 4]:
            raise ValueError("Input must be RGB or RGBA array")

        arr = np.array(hdr, dtype=np.float32, copy=True, order='C')

        if channels == 4:
            self.cpu.applyRGBA(arr)
        else:
            self.cpu.applyRGB(arr)

        if clip:
            np.clip(arr, 0.0, 1.0, out=arr)
        return arr
