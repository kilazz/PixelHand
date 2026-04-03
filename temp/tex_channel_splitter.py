import os
from pathlib import Path
from PIL import Image

def split_directory_textures(input_dir: str, split_dir: str, process_mode: str):
    """
    Splits TGA, DDS, and TIF files into PNGs.
    process_mode = 'channels': separate R, G, B, A
    process_mode = 'rgb_a': separate RGB and A
    """
    input_path = Path(input_dir)
    split_path = Path(split_dir)

    print("\n[PROCESS] STARTING SPLIT...")

    if not input_path.exists():
        print(f"[ERROR] Folder '{input_dir}' not found. Please create it and place TGA/DDS/TIF files inside.")
        return

    found_files = False
    # Added .tif and .tiff support here
    valid_extensions = ['.tga', '.dds', '.tif', '.tiff']

    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            found_files = True
            rel_path = file_path.relative_to(input_path)

            out_dir = split_path / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            base_name = file_path.stem

            try:
                img = Image.open(file_path)

                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")

                # Mode 1: RGB + Alpha
                if process_mode == 'rgb_a':
                    if img.mode == "RGBA":
                        rgb_img = img.convert("RGB")
                        alpha_channel = img.split()[3] # R, G, B, A -> index 3

                        rgb_img.save(out_dir / f"{base_name}_RGB.png")

                        # Check if alpha is completely white (min and max are both 255)
                        if alpha_channel.getextrema() == (255, 255):
                            print(f"[SUCCESS] Split: {rel_path} -> RGB (Solid White Alpha Ignored)")
                        else:
                            alpha_channel.save(out_dir / f"{base_name}_A.png")
                            print(f"[SUCCESS] Split: {rel_path} -> RGB + A")
                    else:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(out_dir / f"{base_name}_RGB.png")
                        print(f"[SUCCESS] Split: {rel_path} -> RGB (No Alpha)")

                # Mode 2: Individual Channels (R, G, B, A)
                else:
                    mode = img.mode
                    channels = img.split()
                    saved_channels =[]

                    for i, channel_name in enumerate(mode):
                        channel_img = channels[i]

                        # Skip saving the Alpha channel if it's completely solid white
                        if channel_name == 'A' and channel_img.getextrema() == (255, 255):
                            continue

                        out_file = out_dir / f"{base_name}_{channel_name}.png"
                        channel_img.save(out_file)
                        saved_channels.append(channel_name)

                    print(f"[SUCCESS] Split: {rel_path} -> {''.join(saved_channels)}")

            except Exception as e:
                print(f"[ERROR] Failed to split {file_path}: {e}")

    if not found_files:
        print(f"[INFO] No supported texture files found in '{input_dir}'.")


def merge_directory_pngs(split_dir: str, merged_dir: str, output_ext: str):
    """
    Auto-detects and merges PNG channels back into the chosen format (.tga or .tif).
    Automatically handles both '_RGB.png' + '_A.png' and '_R.png' + '_G.png' + '_B.png' + '_A.png'
    """
    split_path = Path(split_dir)
    merged_path = Path(merged_dir)

    print("\n[PROCESS] STARTING AUTO-MERGE...")

    if not split_path.exists():
        print(f"[ERROR] Folder '{split_dir}' not found. Nothing to merge.")
        return

    found_files = False

    # 1. First, search for and merge textures of type RGB + A
    for rgb_file in split_path.rglob('*_RGB.png'):
        found_files = True
        rel_path = rgb_file.relative_to(split_path)

        out_dir = merged_path / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = rgb_file.name.rsplit('_RGB.png', 1)[0]
        out_file = out_dir / f"{base_name}{output_ext}"
        a_path = rgb_file.with_name(f"{base_name}_A.png")

        try:
            merged_img = Image.open(rgb_file).convert("RGB")
            mode = "RGB"

            if a_path.exists():
                a_img = Image.open(a_path).convert("L")

                if merged_img.size != a_img.size:
                    print(f"[ERROR] Dimension mismatch for '{base_name}': {merged_img.size} vs {a_img.size}. Cannot merge.")
                    continue

                merged_img.putalpha(a_img)
                mode = "RGBA"

            # For TIFF, Pillow supports LZW compression natively to save space
            if output_ext == '.tif':
                merged_img.save(out_file, compression="tiff_lzw")
            else:
                merged_img.save(out_file)

            print(f"[SUCCESS] Merged (RGB+A): {rel_path.parent / base_name}{output_ext} ({mode})")

        except Exception as e:
            print(f"[ERROR] Failed to merge {base_name}: {e}")

    # 2. Next, search for and merge textures split into individual channels (R, G, B, A)
    for r_file in split_path.rglob('*_R.png'):
        found_files = True
        rel_path = r_file.relative_to(split_path)

        out_dir = merged_path / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = r_file.name.rsplit('_R.png', 1)[0]
        out_file = out_dir / f"{base_name}{output_ext}"

        r_path = r_file
        g_path = r_file.with_name(f"{base_name}_G.png")
        b_path = r_file.with_name(f"{base_name}_B.png")
        a_path = r_file.with_name(f"{base_name}_A.png")

        if not (g_path.exists() and b_path.exists()):
            print(f"[SKIP] '{base_name}': Missing G or B channel.")
            continue

        try:
            images_to_merge =[
                Image.open(r_path).convert("L"),
                Image.open(g_path).convert("L"),
                Image.open(b_path).convert("L")
            ]

            mode = "RGB"

            if a_path.exists():
                images_to_merge.append(Image.open(a_path).convert("L"))
                mode = "RGBA"

            sizes = [img.size for img in images_to_merge]
            if len(set(sizes)) > 1:
                print(f"[ERROR] Dimension mismatch for '{base_name}': {sizes}. Cannot merge.")
                continue

            merged_img = Image.merge(mode, tuple(images_to_merge))

            if output_ext == '.tif':
                merged_img.save(out_file, compression="tiff_lzw")
            else:
                merged_img.save(out_file)

            print(f"[SUCCESS] Merged (Channels): {rel_path.parent / base_name}{output_ext} ({mode})")

        except Exception as e:
            print(f"[ERROR] Failed to merge {base_name}: {e}")

    if not found_files:
        print(f"[INFO] No files to merge found in '{split_dir}' (looking for *_RGB.png or *_R.png files).")


def get_split_mode():
    """Helper function to ask user for the split mode."""
    print("\n" + "-"*45)
    print(" Select splitting mode:")
    print(" 1. Individual Channels (creates _R, _G, _B, _A)")
    print(" 2. RGB + Alpha (creates _RGB, _A)")
    print("-" * 45)

    choice = input("Choose mode (1-2): ").strip()
    if choice == '1':
        return 'channels'
    elif choice == '2':
        return 'rgb_a'
    else:
        print("[ERROR] Invalid mode selected. Returning to main menu.")
        return None

def get_output_extension():
    """Helper function to ask user for the target merge format."""
    print("\n" + "-"*45)
    print(" Select output format for merging:")
    print(" 1. .tga (Standard gaming format)")
    print(" 2. .tif (TIFF with LZW compression)")
    print("-" * 45)

    choice = input("Choose format (1-2): ").strip()
    if choice == '2':
        return '.tif'
    else:
        return '.tga' # Default to TGA if input is 1 or invalid


def main():
    INPUT_FOLDER = "1_original_textures"
    SPLIT_FOLDER = "2_split_pngs"
    MERGED_FOLDER = "3_restored_tgas"

    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    Path(SPLIT_FOLDER).mkdir(exist_ok=True)
    Path(MERGED_FOLDER).mkdir(exist_ok=True)

    while True:
        print("\n" + "="*45)
        print("     AI TEXTURE PREPARATION UTILITY")
        print("="*45)
        print(f" 1. Split Textures (TGA/DDS/TIF) into PNGs")
        print(f" 2. Merge PNGs back into TGA or TIF")
        print(f" 3. Run both (Test)")
        print(f" 0. Exit")
        print("="*45)

        choice = input("Choose an action (0-3): ").strip()

        if choice == '0':
            print("Exiting program.")
            break
        elif choice == '1':
            mode = get_split_mode()
            if mode:
                split_directory_textures(INPUT_FOLDER, SPLIT_FOLDER, mode)
        elif choice == '2':
            ext = get_output_extension()
            merge_directory_pngs(SPLIT_FOLDER, MERGED_FOLDER, ext)
        elif choice == '3':
            mode = get_split_mode()
            if mode:
                ext = get_output_extension()
                split_directory_textures(INPUT_FOLDER, SPLIT_FOLDER, mode)
                merge_directory_pngs(SPLIT_FOLDER, MERGED_FOLDER, ext)
        else:
            print("[ERROR] Invalid input! Please enter a number from 0 to 3.")

if __name__ == "__main__":
    main()