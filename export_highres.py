import shutil
import sys
from pathlib import Path

# ==========================================
#              CONFIGURATION
# ==========================================

# Folder A: Original / Low Resolution (The baseline)
FOLDER_A = r"E:\Games\_Crysis 3 PAK\C3"

# Folder B: Remastered / High Resolution (Source to copy from)
FOLDER_B = r"E:\Games\_Crysis 3 Remastered PAK\C3"

# Output: Where the higher resolution files will be copied
OUTPUT_DIR = r"E:\CE\Tools\_Edit\PixelHand\Crysis3_HighRes_Export"

# Match files by filename stem (ignoring extension)?
# Example: 'texture.tif' in A matches 'texture.dds' in B if True.
MATCH_BY_STEM = True

# Copy files that exist in B but NOT in A?
COPY_NEW_FILES = False

# ==========================================

# Setup path to import PixelHand modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    # We use the app's internal image loader because standard Python libraries
    # often fail to read compressed DDS headers (DirectXTex) correctly.
    from app.constants import ALL_SUPPORTED_EXTENSIONS
    from app.image_io import get_image_metadata

    print("[OK] PixelHand modules loaded successfully.")
except ImportError:
    print("[ERROR] Could not import PixelHand modules.")
    print("Please ensure this script is placed in the same folder as 'main.py' and the 'app' directory.")
    input("Press Enter to exit...")
    sys.exit(1)


def get_relative_identifier(file_path: Path, root_folder: Path) -> str:
    """
    Generates a unique ID for a file based on its structure relative to the root.
    If MATCH_BY_STEM is True, extensions are ignored.
    Example: 'Objects/Weapons/Rifle.dds' -> 'objects/weapons/rifle'
    """
    try:
        rel_path = file_path.relative_to(root_folder)
        if MATCH_BY_STEM:
            # Remove suffix and convert to lower case for case-insensitive matching
            return str(rel_path.with_suffix("")).replace("\\", "/").lower()
        else:
            return str(rel_path).replace("\\", "/").lower()
    except ValueError:
        return str(file_path.name).lower()


def main():
    root_a = Path(FOLDER_A)
    root_b = Path(FOLDER_B)
    output_root = Path(OUTPUT_DIR)

    if not root_a.exists():
        print(f"[ERROR] Folder A does not exist: {root_a}")
        return
    if not root_b.exists():
        print(f"[ERROR] Folder B does not exist: {root_b}")
        return

    print("\n--- Starting High-Res Export ---")
    print(f"Source A (Old): {root_a}")
    print(f"Source B (New): {root_b}")
    print(f"Output:         {output_root}\n")

    # 1. Index Folder A
    print("1. Indexing Folder A...")
    index_a = {}
    count_a = 0

    # Using rglob to recursively find files
    for f_path in root_a.rglob("*"):
        if f_path.is_file() and f_path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
            key = get_relative_identifier(f_path, root_a)
            index_a[key] = f_path
            count_a += 1
            if count_a % 1000 == 0:
                print(f"   Indexed {count_a} files...", end="\r")

    print(f"   Done. Indexed {count_a} files in Folder A.\n")

    # 2. Process Folder B
    print("2. Comparing Folder B against A...")

    stats = {"processed": 0, "upgraded": 0, "skipped_same_res": 0, "skipped_smaller": 0, "new_files": 0, "errors": 0}

    for f_path_b in root_b.rglob("*"):
        if not f_path_b.is_file() or f_path_b.suffix.lower() not in ALL_SUPPORTED_EXTENSIONS:
            continue

        stats["processed"] += 1
        key = get_relative_identifier(f_path_b, root_b)

        # Print progress every 500 files
        if stats["processed"] % 500 == 0:
            print(f"   Processing: {stats['processed']} files... (Upgraded: {stats['upgraded']})", end="\r")

        # Logic: Does this file exist in A?
        if key in index_a:
            f_path_a = index_a[key]

            # Read metadata (resolution)
            meta_a = get_image_metadata(f_path_a)
            meta_b = get_image_metadata(f_path_b)

            if not meta_a or not meta_b:
                stats["errors"] += 1
                continue

            res_a = meta_a["resolution"]
            res_b = meta_b["resolution"]

            area_a = res_a[0] * res_a[1]
            area_b = res_b[0] * res_b[1]

            # Compare Resolution
            if area_b > area_a:
                # B is bigger -> Copy
                copy_file(f_path_b, root_b, output_root)
                # Optional: Detailed log
                # print(f"[UPGRADE] {f_path_b.name}: {res_a} -> {res_b}")
                stats["upgraded"] += 1
            elif area_b == area_a:
                stats["skipped_same_res"] += 1
            else:
                stats["skipped_smaller"] += 1

        else:
            # File exists in B but not in A (New file)
            if COPY_NEW_FILES:
                copy_file(f_path_b, root_b, output_root)
                stats["new_files"] += 1

    print("\n" + "=" * 40)
    print("SCAN COMPLETE")
    print("=" * 40)
    print(f"Total Files Scanned in B: {stats['processed']}")
    print(f"files Upgraded (Copied):  {stats['upgraded']}")
    print(f"Files Same Resolution:    {stats['skipped_same_res']}")
    print(f"Files Smaller (Downgrade):{stats['skipped_smaller']}")
    if COPY_NEW_FILES:
        print(f"New Files Copied:         {stats['new_files']}")
    print(f"Read Errors:              {stats['errors']}")
    print(f"\nFiles saved to: {output_root}")
    input("Press Enter to close...")


def copy_file(file_path: Path, root_src: Path, root_dst: Path):
    """
    Copies a file maintaining the directory structure.
    """
    try:
        relative_path = file_path.relative_to(root_src)
        destination_path = root_dst / relative_path

        # Create parent directories if they don't exist
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file with metadata
        shutil.copy2(file_path, destination_path)
    except Exception as e:
        print(f"\n[COPY ERROR] Could not copy {file_path.name}: {e}")


if __name__ == "__main__":
    main()
