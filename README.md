# PixelHand WIP ~

A local, hardware-accelerated desktop tool written in Rust and Slint for visual duplicate detection, semantic AI search, and technical quality control auditing of graphics and textures.

## Key Features

### 🔍 Duplicate Detection & AI Search
*   **Multi-Tier Scan**: Find duplicates via byte-exact matches (**xxHash64**), perceptual similarity (**dHash / pHash**), or local AI embeddings.
*   **Smart AI Search**: Search visually similar assets (Image-to-Image) or query with natural language (Text-to-Image), completely offline.
*   **Supported AI Models**: CLIP-B/32, CLIP-L/14, SigLIP-B, SigLIP-L, DINOv2-B, or load your own local custom ONNX models.
*   **Lazy DDS Mip Loading**: Dynamic resolution matching (256px or 384px) that decodes only the necessary mipmap level directly from DDS files, reducing RAM footprint during AI scans.

### 📐 Pipeline Technical QC
*   **Absolute Audits**: Non-Power-of-Two (NPOT) verification, missing Mip-maps, block compression alignment (4px), and bit depth analysis.
*   **Normal Map Integrity**: Tangent-space normal vector validation and automatic **DirectX vs OpenGL Y-axis check** (shading orientation detection).
*   **ORM/RMA Empty Channel Detection**: Auto-detects empty, missing, or flat-color masks (R, G, B, A) in packed textures.
*   **GPU VRAM Estimation**: Displays estimated GPU memory footprint (with block padding and mip pyramids calculated) alongside disk size.
*   **Relative Compare**: Folder A vs Folder B auditing to track resolution downgrades, lost alpha, colorspace shifts, and size bloat.

### 🖼️ Interactive Comparative Viewer
*   **5 Compare Modes**: Side-by-Side, Wipe, Overlay, Difference Heatmap, and **Flicker/Blink Mode** (toggle between original and duplicate at custom intervals to spot sub-pixel artifacts).
*   **Luminance Histogram**: Overlapping real-time slate graphs showing gamma, exposure, and contrast mismatches.
*   **RGBA Isolator**: Quick toggle to analyze isolated color channels (**R, G, B, A**) with zero-lag debounced cursor hovering.
*   **HDR Tonemapping**: Selectable realtime operators (**ACES Filmic**, BT.2446c **ICtCp Perceptual**, and **Khronos PBR Neutral**) for linear float textures.

## Technical Stack & Formats
*   **GUI**: [Slint UI](https://slint.dev/)
*   **AI Inference**: [ONNX Runtime](https://onnxruntime.ai/)
*   **Vector DB**: [LanceDB](https://lancedb.com/)
*   **Supported Formats**: `.dds`, `.exr`, `.hdr`, `.psd`, `.jxl`, `.heic`/`.heif`, `.png`, `.jpg`/`.jpeg`, `.tga`, `.bmp`, `.tiff`, `.webp`

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
