# renormalize_tool.py
"""
Batch Normal Map Renormalizer.
"""

import shutil
import sys
from pathlib import Path

try:
    import numpy as np
    import OpenImageIO as oiio
except ImportError:
    print("Error: Missing dependencies.")
    print("Please pip install OpenImageIO numpy")
    sys.exit(1)

# ================= CONFIGURATION =================

# Filename tags to identify Normal Maps
TARGET_TAGS = ["_ddn", "_ddna"]

# Supported extensions
EXTENSIONS = {".tif", ".tiff"}

# Create a .bak copy of the original file?
MAKE_BACKUP = True

# "WHITE" = Standard for CryEngine/Unity/Unreal (Z channel = 1.0)
# "RECONSTRUCT" = Calculates the perfect mathematical Z
# "PRESERVE" = Normalizes existing X/Y/Z vector
BLUE_CHANNEL_MODE = "WHITE"

# Re-process everything to ensure metadata and math are perfect
FORCE_ALL = True

# Error threshold (ignored when FORCE_ALL is True)
ERROR_THRESHOLD = 0.005  # Lower threshold because we use float 0..1

# =================================================


def normalize_vector_array_float(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Performs vector math using Linear -1..1 space.
    Input and Output are float32 in range [0.0, 1.0].
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Extract RGB (first 3 channels)
    rgb = arr[..., :3]

    # Map [0..1] -> [-1..1] directly
    vx = rgb[..., 0] * 2.0 - 1.0
    vy = rgb[..., 1] * 2.0 - 1.0

    if mode == "WHITE":
        # Calculate length squared in XY plane
        len_xy_sq = vx**2 + vy**2

        # Clamp vectors outside the unit circle
        invalid_mask = len_xy_sq > 1.0
        if np.any(invalid_mask):
            len_xy = np.sqrt(len_xy_sq[invalid_mask])
            # Normalize X and Y to fit exactly on edge (length 1.0)
            vx[invalid_mask] /= len_xy
            vy[invalid_mask] /= len_xy

        # Force Z to 1.0 (Full White)
        vz = np.ones_like(vx)
        vectors_norm = np.stack([vx, vy, vz], axis=2)

    elif mode == "RECONSTRUCT":
        # Clamp XY
        len_xy_sq = vx**2 + vy**2
        scale_mask = len_xy_sq > 1.0
        if np.any(scale_mask):
            scale_factor = np.sqrt(len_xy_sq[scale_mask])
            vx[scale_mask] /= scale_factor
            vy[scale_mask] /= scale_factor

        # Calculate Z = sqrt(1 - x^2 - y^2)
        vz = np.sqrt(np.maximum(0, 1.0 - vx**2 - vy**2))
        vectors_norm = np.stack([vx, vy, vz], axis=2)

    else:  # PRESERVE
        vz = rgb[..., 2] * 2.0 - 1.0
        vectors = np.stack([vx, vy, vz], axis=2)
        lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
        # Avoid division by zero
        lengths[lengths == 0] = 1.0
        vectors_norm = vectors / lengths

    # Back to [0..1]
    rgb_norm = (vectors_norm + 1.0) * 0.5

    # Clip just in case (floating point errors)
    rgb_final = np.clip(rgb_norm, 0.0, 1.0)

    # Handle Alpha channel (copy from source)
    if arr.shape[2] >= 4:
        alpha = arr[..., 3:4]
        return np.concatenate([rgb_final, alpha], axis=2)

    return rgb_final


def calculate_error(arr: np.ndarray) -> float:
    """Calculates average deviation from vector length 1.0."""
    rgb = arr[..., :3]
    vectors = rgb * 2.0 - 1.0
    lengths = np.linalg.norm(vectors, axis=2)
    diff = np.abs(lengths - 1.0)
    return float(np.mean(diff))


def process_file(path: Path):
    path_str = str(path)

    # 1. READ with OIIO
    inp = oiio.ImageInput.open(path_str)
    if not inp:
        print(f"[ERROR] Could not open {path.name}")
        return

    spec = inp.spec()

    # Detect Bit Depth
    is_16bit = spec.format == oiio.UINT16 or spec.format == oiio.USHORT
    bit_depth_str = "16-bit" if is_16bit else "8-bit"

    # Read as float (0.0 - 1.0)
    pixels = inp.read_image(format=oiio.FLOAT)
    inp.close()

    if pixels is None:
        print(f"[ERROR] Could not read pixels {path.name}")
        return

    # Logic to skip if FORCE_ALL is False
    if not FORCE_ALL:
        if BLUE_CHANNEL_MODE == "WHITE":
            # Check if Blue is already > 0.98 (approx 250/255)
            if np.mean(pixels[..., 2]) > 0.98:
                print(f"[SKIP] {path.name}")
                return
        elif calculate_error(pixels) < ERROR_THRESHOLD:
            print(f"[SKIP] {path.name}")
            return

    print(f"[FIXING] {path.name} ({bit_depth_str})...")

    # 2. PROCESS (Normalize & Clamp)
    # Perform math in float space (0..1)
    fixed_pixels_float = normalize_vector_array_float(pixels, BLUE_CHANNEL_MODE)

    # 3. PREPARE OUTPUT DATA
    # Convert back to target bit depth
    if is_16bit:
        out_type = oiio.UINT16
        # Scale 0..1 to 0..65535
        fixed_pixels_out = (fixed_pixels_float * 65535.0).astype(np.uint16)
    else:
        out_type = oiio.UINT8
        # Scale 0..1 to 0..255
        fixed_pixels_out = (fixed_pixels_float * 255.0).astype(np.uint8)

    # 4. SETUP OUTPUT METADATA
    out_spec = oiio.ImageSpec(spec.width, spec.height, spec.nchannels, out_type)

    # --- Force Linear Color Space ---
    out_spec.attribute("oiio:ColorSpace", "Linear")
    out_spec.attribute("color_space", "Linear")

    # Handle Compression based on file type
    ext = path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        out_spec.attribute("compression", "lzw")
    # For PNG/TGA we rely on default OIIO settings or add specific ones if needed

    # Clean up metadata
    out_spec.erase_attribute("ICCProfile")
    out_spec.erase_attribute("icc_profile")
    out_spec.erase_attribute("gamma")
    out_spec.erase_attribute("DateTime")

    # 5. BACKUP
    if MAKE_BACKUP:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    # 6. WRITE
    out = oiio.ImageOutput.create(path_str)
    if not out:
        print(f"[ERROR] Could not create output {path.name}")
        return

    try:
        out.open(path_str, out_spec)
        out.write_image(fixed_pixels_out)
        out.close()
        print("    -> Done! (Fixed XY, Z & Metadata)")
    except Exception as e:
        print(f"[ERROR] Write failed: {e}")


def main():
    print("==========================================")
    print("   Normal Map Renormalizer                ")
    print(f"  Mode: {BLUE_CHANNEL_MODE}              ")
    print("   ColorSpace: Forced Linear              ")
    print("==========================================\n")

    target_dir_str = input("Enter folder path: ").strip().strip('"')
    if not target_dir_str:
        return

    path_obj = Path(target_dir_str)
    if not path_obj.exists():
        print("Path does not exist.")
        return

    files_to_process = []
    print("Scanning...")
    for p in path_obj.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            fname = p.name.lower()
            if any(tag in fname for tag in TARGET_TAGS):
                files_to_process.append(p)

    print(f"Found {len(files_to_process)} files.")
    if not files_to_process:
        return

    if input("Start? (y/n): ").lower() != "y":
        return

    print("\n--- Starting ---")
    for p in files_to_process:
        process_file(p)

    print("\n--- Complete ---")


if __name__ == "__main__":
    main()
