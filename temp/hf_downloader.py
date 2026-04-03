# hf_downloader.py
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
FILENAME = "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

if len(sys.argv) == 3:
    REPO_ID, FILENAME = sys.argv[1], sys.argv[2]
elif len(sys.argv) > 1:
    print("Usage: python hf_downloader.py [repo_id] [filename]")
    sys.exit(1)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAVE_DIR = Path(__file__).parent.resolve()

print("=" * 60)
print("HUGGING FACE UNIVERSAL DOWNLOADER (zero warnings)")
print("=" * 60)
print(f"Repo   → {REPO_ID}")
print(f"File   → {FILENAME}")
print(f"Folder → {SAVE_DIR}")
print("Starting download... (Ctrl+C → re-run to resume)\n")

try:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=SAVE_DIR,
        cache_dir=None,
    )
    print("\nSUCCESS! Downloaded:")
    print(f"{path}")

except KeyboardInterrupt:
    print("\nStopped. Run again → resumes automatically.")
except Exception as e:
    print(f"\nERROR: {e}")
    if "401" in str(e) or "403" in str(e):
        print("→ Private repo? Run: huggingface-cli login")
