# app/imaging/loaders/oiio_loader.py
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app.imaging.processing import tonemap_float_array
from app.shared.constants import OIIO_AVAILABLE, TonemapMode

from .base_loader import BaseLoader

if OIIO_AVAILABLE:
    import OpenImageIO as oiio

    # Limit OIIO's global tile cache to 512 MB.
    try:
        oiio.attribute("max_memory_MB", 512.0)
        oiio.attribute("autotile", 64)  # Smaller tiles = less memory fragmentation
    except Exception:
        pass

app_logger = logging.getLogger("PixelHand.oiio_loader")


class OIIOLoader(BaseLoader):
    """
    High-performance loader using OpenImageIO (OIIO).

    Optimizations:
    1. Uses `ImageInput` instead of `ImageBuf` for precise control over IO.
    2. Implements MIP-Map seeking (`seek_subimage`) to read thumbnail-sized
       data directly from formats like EXR, TIFF, DDS, and TX.
    """

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not OIIO_AVAILABLE:
            return None

        # Use ImageInput to open the file without reading pixels immediately
        inp = oiio.ImageInput.open(str(path))
        if not inp:
            return None

        try:
            spec = inp.spec()

            # --- SMART-SHRINK LOGIC ---
            nsubimages = getattr(spec, "nsubimages", 1)

            if shrink > 1 and nsubimages > 1:
                target_width = spec.width // shrink
                best_subimage = 0

                for i in range(nsubimages):
                    mipspec = inp.spec_dimensions(i)
                    if mipspec.width >= target_width:
                        best_subimage = i
                    else:
                        break

                if best_subimage > 0:
                    inp.seek_subimage(best_subimage, 0)
                    spec = inp.spec()

            # --- DATA READING ---
            is_float_source = spec.format.basetype in (oiio.FLOAT, oiio.HALF, oiio.DOUBLE)
            is_hdr_requested = tonemap_mode == TonemapMode.ENABLED.value

            if is_float_source and is_hdr_requested:
                # Read as FLOAT
                data = inp.read_image(format=oiio.FLOAT)

                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(2)

                if data.ndim == 3 and data.shape[2] > 4:
                    data = data[:, :, :4]

                pil_image = Image.fromarray(tonemap_float_array(data))

            else:
                # Read as UINT8
                data = inp.read_image(format=oiio.UINT8)

                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(2)

                # Handle channel configurations
                if spec.nchannels not in (1, 3, 4):
                    if spec.nchannels == 2:
                        pil_image = Image.fromarray(data, mode="LA").convert("RGBA")
                    elif spec.nchannels > 4:
                        # Crop extra channels (Z-depth, IDs, etc.)
                        pil_image = Image.fromarray(data[:, :, :4])
                    else:
                        pil_image = Image.fromarray(data)
                else:
                    pil_image = Image.fromarray(data)

            return pil_image

        except Exception as e:
            app_logger.error(f"OIIO load failed for {path}: {e}")
            return None
        finally:
            inp.close()

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not OIIO_AVAILABLE:
            return None

        try:
            buf = oiio.ImageBuf(str(path))
            if buf.has_error:
                # Explicitly retrieve the error to prevent OIIO destructor from
                # logging "An ImageBuf was destroyed with a pending error message" to stderr.
                _ = buf.geterror()
                return None

            spec = buf.spec()

            bit_depth_map = {
                oiio.UINT8: 8,
                oiio.INT8: 8,
                oiio.UINT16: 16,
                oiio.INT16: 16,
                oiio.HALF: 16,
                oiio.UINT32: 32,
                oiio.INT32: 32,
                oiio.FLOAT: 32,
                oiio.DOUBLE: 64,
            }
            bit_depth = bit_depth_map.get(spec.format.basetype, 8)

            ch_count = spec.nchannels
            ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(ch_count, f"{ch_count}ch")

            format_str = buf.file_format_name.upper()
            dds_fmt = spec.get_string_attribute("dds:format") or spec.get_string_attribute("compression")
            compression_format = dds_fmt.upper() if dds_fmt and format_str == "DDS" else format_str

            capture_date = None
            if dt := spec.get_string_attribute("DateTime"):
                with contextlib.suppress(ValueError, TypeError):
                    capture_date = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()

            mipmap_count = getattr(buf, "nsubimages", 1)

            return {
                "resolution": (spec.width, spec.height),
                "file_size": stat_result.st_size,
                "mtime": stat_result.st_mtime,
                "format_str": format_str,
                "compression_format": compression_format,
                "format_details": ch_str,
                "has_alpha": spec.alpha_channel != -1,
                "capture_date": capture_date,
                "bit_depth": bit_depth,
                "mipmap_count": max(1, mipmap_count),
                "texture_type": "2D",
                "color_space": spec.get_string_attribute("oiio:ColorSpace") or "sRGB",
            }
        except Exception as e:
            app_logger.warning(f"OIIO metadata error for {path}: {e}")
            return None
