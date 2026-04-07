from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO


def _default_cfg_path() -> str:
    """Locate bundled `config.ocio` relative to this file."""
    return str(Path(__file__).parent / "ocio_data" / "cg-config-v4.0.0_aces-v2.0_ocio-v2.5.ocio")

class ToneMapper:
    """
    Parameters
    ----------
    view : str
        OCIO *view* (e.g. "Un-tone-mapped", "ACES 2.0 - SDR 100 nits (Rec.709)"). Default = "Un-tone-mapped".
    ocio_cfg : str | Path | None
        Path to an OCIO `config.ocio`. If *None*, use bundled ACES config.
    """

    def __init__(
        self,
        view: str = "Un-tone-mapped",
        ocio_cfg: str | Path | None = None,
    ):
        ocio_cfg = ocio_cfg or _default_cfg_path()
        self.config = OCIO.Config.CreateFromFile(str(ocio_cfg))
        OCIO.SetCurrentConfig(self.config)

        self.display = self.config.getDefaultDisplay()
        self._view = view

        self.xform = OCIO.DisplayViewTransform(OCIO.ROLE_SCENE_LINEAR, self.display, self.view)
        self.cpu = self.config.getProcessor(self.xform).getDefaultCPUProcessor()

    @property
    def view(self) -> str:
        return self._view

    @view.setter
    def view(self, value: str):
        self._view = value
        self.xform.setView(value)
        del self.cpu
        self.cpu = self.config.getProcessor(self.xform).getDefaultCPUProcessor()

    @property
    def available_views(self) -> list[str]:
        return list(self.config.getViews(self.display))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def hdr_to_ldr(self, hdr: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Tone-map HDR -> LDR (both float32, NumPy).

        Accepts RGB or RGBA arrays.  Values are expected in linear scene-referred
        [0,+infinity).  Output will be display-referred [0,1] (clipped if `clip`).

        Notes
        -----
        `CPUProcessor.applyRGB` works *in place* and requires:
        * contiguous buffer
        * dtype float32 (or matching OCIO bit-depth)
        """

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
