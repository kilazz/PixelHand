# PixelHand WIP ~
**A lightning-fast, local, and hardware-accelerated tool for visual duplicate detection, semantic AI search, and technical quality control auditing of graphics and textures.**

## Features
*   🔍 **Multi-Tier Duplicate Detection**
    Identify duplicates starting from byte-exact matches (**xxHash64**) to perceptual similarities (**dHash / pHash**) and deep conceptual lookalikes using vector embeddings.
*   🧠 **Smart Local AI Search**
    *   **Text-to-Image (Semantic)**: Search your local drives using natural language prompts (e.g., "damaged concrete wall").
    *   **Image-to-Image (Sample Search)**: Drag and drop a reference image to retrieve visually similar assets, powered by fully offline visual encoders.
    *   **Flexible Model Registry**: Out-of-the-box support for 5 remote models (`CLIP-B/32`, `CLIP-L/14`, `SigLIP-B`, `SigLIP-L`, and `DINOv2-B`) or **load your own local custom ONNX models** with adjustable dimension and architecture configurations.
*   📐 **Quality Control (QC) & Pipeline Validation**
    Specifically tailored for GameDev & VFX pipelines. Automates routine graphics asset integrity checks:
    *   **Absolute Checks**: NPOT (Non-Power-of-Two) dimensions validation, missing Mip-maps inspection, block compression alignment (4px) verification, and bit depth analysis.
    *   **Surface Inspection**: Constant flat color detection and tangent-space **Normal Map vector integrity validation** (identifies non-normalized vectors and inverted axes).
    *   **Relative Compare Checks**: Real-time evaluation of resolution downgrades, file size bloat (>1.5x), lost alpha channels, color space discrepancies (sRGB vs Linear), and unexpected compression format transitions.
*   📂 **Advanced Comparative Modes**
    *   **Folder A vs Folder B**: Audit build directories or asset releases against source repositories, with option to hide identical resolution files to isolate regressions.
    *   **Channel Toggle Analysis**: Isolate and inspect individual RGBA color channels (**R, G, B, A**) to analyze packed channel masks.
*   🖼️ **High-Fidelity Interactive Viewer**
    Pixel-perfect comparative canvas tools with synchronized panning and zooming:
    *   **Side-by-Side** layout.
    *   **Wipe Mode** with adjustable splitter bar.
    *   **Overlay Blend** with alpha transparency controls.
    *   **Difference Heatmap** generator to visually isolate tiny color channel variations.
    *   **Professional HDR-to-SDR Tonemapping**: Selectable realtime operators supporting **ACES Filmic**, per-channel **ICtCp Perceptual (BT.2446c)**, and **Khronos PBR Neutral** to display linear float textures accurately without hue shifting.
*   ⚡ **Fluid UI & Live Feedback**
    *   A level, start-aligned **4-Column Responsive Grid View** with a realtime card-resizing slider.
    *   **Live progress bar updating** during CPU-bound hashing and GPU-bound ML inference loops.
    *   Instant `Expand All` and `Collapse All` duplicate group triggers.

## Format Support
*   **Pro Formats**: Direct memory-mapped `.dds` (standard & swizzled Xbox 360 / CryEngine Morton curves), 32-bit floating-point `.exr` (OpenEXR) and `.hdr` (Radiance HDR).
*   **Production Source Formats**: Adobe Photoshop `.psd` layered composites, next-generation `.jxl` (JPEG XL), and Apple `.heic` / `.heif` native decodes.
*   **Standard Formats**: `.png`, `.jpg`, `.jpeg`, `.tga`, `.bmp`, `.tiff`, and `.webp`.

## Tech Stack
*   **GUI Engine**: [Slint UI](https://slint.dev/) (GPU-accelerated, native compiled)
*   **Machine Learning**: [ONNX Runtime](https://onnxruntime.ai/)
*   **Vector Database**: [LanceDB](https://lancedb.com/) (disk-backed transactional storage, isolated schemas per-model to prevent collisions)
*   **Concurrencies**: Thread-safe background tasks built with `Rayon` (data-parallelism CPU threads) and `Tokio` (asynchronous state runtimes)

## Hardware Acceleration
Leverages cutting-edge ONNX Runtime hardware acceleration providers dynamically chosen based on your host environment:
*   **Windows**: DirectML & CUDA / TensorRT
*   **macOS**: CoreML
*   **Linux / Fallback**: Highly-optimized multithreaded CPU execution

## Development

```
# Run GUI
cargo run

# Run the command-line (CLI) auditor mode
cargo run -- -c --scan-exact <directory_path>
cargo run -- -c --scan-qc <directory_path> --check-npot --validate-normals

# Clean security audit (Zero vulnerabilities / unmaintained warning-free)
cargo audit

# Compile optimized production release binaries
cargo build --release
```
