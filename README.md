# PixelHand WIP
**A lightning-fast, local tool for finding duplicate images, visual similarity search, and quality control.**

## Features
*   🔍 **Duplicate Detection**
    Identify duplicates starting from exact byte-matches (**xxHash**) up to conceptual and visual lookalikes using AI embeddings.
*   🧠 **Smart AI Search**
    *   **Text-to-Image**: Find images using natural language prompts (e.g., "concrete wall").
    *   **Image-to-Image**: Drag and drop a reference image to retrieve visually similar assets.
*   📐 **Quality Control (QC) & Validation**
    Tailored for GameDev & VFX. Automates routine integrity checks:
    *   **NPOT** (Non-Power-of-Two) dimensions validation.
    *   Checking for proper **Mip-map** generation.
    *   **BlockAlignment (4px)** verification.
    *   **Bit Depth** and **Normal Map** analysis.
*   📂 **Compare Modes**
    *   **Folder vs Folder**: Compare a source directory against a target build directory.
    *   **Channel Toggle Analysis**: Inspect and compare individual image channels (**R, G, B, A**).
*   🖼️ **Advanced Interactive Viewer**
    Built-in pixel-perfect comparison tools:
    *   **Side-by-Side** view
    *   **Wipe** (slider) blending
    *   **Overlay** with transparency control
    *   **Difference Heatmap** computation
*   💻 **Hardware Accelerated**
    *   Leverages **DirectML** for fast GPU-accelerated AI model inference on Windows.
    *   Falls back smoothly to highly-optimized local CPU execution.

## Tech Stack
*   **Core GUI Engine**: [Slint UI](https://slint.dev/)
*   **Machine Learning**: [ONNX Runtime](https://onnxruntime.ai/)
*   **Vector Database**: [LanceDB](https://lancedb.com/)
*   **Image Processing & Format Parsing**: Native Rust `image`, `image_dds`, `ddsfile`, and `exr` crates with custom hardware-mapped unswizzlers (Morton Z-order curve decoding).

## Development
To run the PixelHand development environment:

```
# 1. Run GUI
cargo run

# 2. Run the command-line (CLI) auditor mode
cargo run -- -c --scan-exact <directory_path>
cargo run -- -c --scan-qc <directory_path> --check-npot --validate-normals

# 3. Compile optimized production release binaries
cargo build --release
```
