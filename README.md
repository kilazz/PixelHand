# PixelHand
**A local tool for finding duplicate images, visual similarity search, and quality control.**

> **Note: This is a Prototype**

## Features
*   🔍 **Duplicate Detection**
    Finds everything from exact byte-matches (**xxHash**) to visual lookalikes (**dHash/pHash**) and conceptual duplicates via AI models (**CLIP, SigLIP**).
*   🧠 **Smart Search**
    *   **Text Search**: Find images by description (e.g., "concrete wall").
    *   **Image Search**: Drop a reference image to find similar assets.
*   📐 **Quality Control (QC)**
    Useful for GameDev & VFX. Automates checks for:
    *   **NPOT** (Non-Power-of-Two) dimensions.
    *   Missing **Mip-maps**.
    *   **Solid color** textures and Alpha channel issues.
*   📂 **Compare Modes**
    *   **Folder vs Folder**: Compare a source directory against a target (e.g., Source vs. Build).
    *   **Channel Analysis**: Find duplicates based on specific channels (**R, G, B, A**) or Luminance.
*   🖼️ **Viewer**
    Built-in comparison tools (**Side-by-Side, Wipe, Overlay, Diff**) with **HDR** tone mapping support (OCIO).
*   💾 **Format Support**
    Handles standard formats (`PNG`, `JPG`) and professional ones (`DDS`, `EXR`, `TIF`, `PSD`).
*   🔗 **Actions**
    Save space using **Hardlinks** or **Reflinks** (Copy-on-Write). Deletions are safely moved to the Recycle Bin.

## Tech Stack
*   **Core**: Python 3.13, PySide6 (Qt)
*   **AI/DB**: ONNXRuntime, LanceDB
*   **Imaging**: OpenImageIO, DirectXTex, Pillow

## Quick Start
1.  Clone the repository.
2.  Run `run.bat`.