# PixelHand WIP ~
A local desktop tool for visual duplicate detection, semantic AI search, technical quality control auditing, and texture comparison.

## Key Features
### 🔍 Duplicate Detection & AI Search
*   **Multi-Tier Scan**: Find duplicates by exact byte matches (**xxHash64**), visual similarity (**dHash**), or local AI embeddings (**LanceDB**).
*   **Offline AI Search**: Image-to-Image reference search and Text-to-Image natural language queries.
*   **Supported AI Models**: CLIP (B/32, L/14), SigLIP (Base, Large, SigLIP 2), LLM2CLIP, DINOv2-B, or custom local ONNX models.

### 📐 Technical Quality Control (QC)
*   **Standalone Audits**: Checks Non-Power-of-Two (NPOT), missing mipmaps, 4px block alignment, bit depth, solid/empty channels, and resolution limits.
*   **Normal Map Integrity**: Validates normal vectors, BC5 limits, and **DirectX (-Y) vs OpenGL (+Y)** orientation.
*   **Data Maps & Tiling**: Detects sRGB misconfigurations on packed maps (`_norm`, `_orm`) and checks **seamless edge tiling**.
*   **VRAM Estimation**: Calculates estimated GPU memory usage alongside disk size.
*   **Folder A vs B Compare**: Audits resolution downgrades, lost alpha, colorspace shifts, and size bloat between directories.

### 🖼️ Interactive Comparative Viewer
*   **5 Compare Modes**: Side-by-Side, Wipe, Overlay, Difference Heatmap, and **Flicker Mode**.
*   **Animation Player**: Dual-mode **Flipbook / Sequence Player** supporting **Spritesheet Grid** subdivision and **Group Files Sequence** playback (e.g. `_01.dds`..`_08.dds` image series) with frame blending and FPS controls.
*   **Inspection Tools**: Adjustable **Magnifier Loupe (2x–16x)**, real-time Tangent-Space Normal Vector calculator, pixel delta ($\Delta E$), luminance histograms, and RGBA channel isolation.
*   **HDR Tonemapping**: **ACES Filmic**, **ACES 2.0 Fit**, **Khronos PBR Neutral**, **ICtCp BT.2446c / Lumina**, and False Color visualization.

### 📊 Deduplication & Reporting
*   **Actions**: Move to Trash, atomic Hardlink replacement, or Reflink/Copy creation.
*   **Exports**: Printable HTML/PDF audit reports, visual contact sheets (`.png`), and CSV inventories.

## Technical Stack & Formats
*   **GUI**: [Slint UI](https://slint.dev/)
*   **AI Inference**: [ONNX Runtime](https://onnxruntime.ai/)
*   **Vector DB**: [LanceDB](https://lancedb.com/)
*   **Supported Formats**: `.dds`, `.ktx2/.basis`, `.astc`, `.atc`, `.crn`, `.pvr`, `.exr`, `.hdr`, `.psd`, `.jxl`, `.heic/heif`, `.raw`, `.qoi`, `.svg/svgz`, `.png`, `.jpg/jpeg`, `.tga`, `.bmp`, `.tiff`, `.webp`, `.avif`, `.ico`.

## Development
```
# Run GUI
cargo run

# Run CLI exact scan
cargo run -- -c --scan-exact <directory_path>

# Run CLI technical QC audit
cargo run -- -c --scan-qc <directory_path> --check-npot --validate-normals

# Build production release binary
cargo build --release
```
