import { For, Show, createSignal, createEffect, onCleanup } from "solid-js";
import ThumbImage from "./ThumbImage";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";

const formatSize = (bytes) => {
  if (bytes === undefined || bytes === null || isNaN(bytes)) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(3)} MB`;
};

export default function ViewerPanel(props) {
  const [bgAlpha, setBgAlpha] = createSignal(255);
  const [viewerGridSize, setViewerGridSize] = createSignal(110);

  const [compareMode, setCompareMode] = createSignal("side-by-side");
  const [wipeSlider, setWipeSlider] = createSignal(50);
  const [activeChannel, setActiveChannel] = createSignal(null);

  const [srcOriginal, setSrcOriginal] = createSignal("");
  const [srcDuplicate, setSrcDuplicate] = createSignal("");

  const [heatmapSrc, setHeatmapSrc] = createSignal("");
  const [loadingHeatmap, setLoadingHeatmap] = createSignal(false);

  let wipeViewportRef;
  const [isWiping, setIsWiping] = createSignal(false);
  const [isSpecsCollapsed, setIsSpecsCollapsed] = createSignal(false);

  const handleWipeStart = (e) => {
    if (e.button !== 0) return;
    setIsWiping(true);
    updateWipeFromEvent(e);
  };

  const updateWipeFromEvent = (e) => {
    if (!wipeViewportRef) return;
    const rect = wipeViewportRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(
      0,
      Math.min(100, Math.round((x / rect.width) * 100))
    );
    setWipeSlider(percentage);
  };

  createEffect(() => {
    if (isWiping()) {
      const onMouseMove = (e) => {
        updateWipeFromEvent(e);
      };
      const onMouseUp = () => {
        setIsWiping(false);
      };
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
      onCleanup(() => {
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      });
    }
  });

  const activeGroup = () => props.scanResults[props.selectedGroupIndex];

  const originalFile = () => {
    if (!activeGroup() || !activeGroup().files) return null;
    return activeGroup().files.find((f) => f.path === props.activeOriginalPath);
  };

  const duplicateFile = () => {
    if (!activeGroup() || !activeGroup().files) return null;
    return activeGroup().files.find(
      (f) => f.path === props.activeDuplicatePath
    );
  };

  createEffect(async () => {
    if (
      props.viewerCanvasOpen &&
      props.activeOriginalPath &&
      props.activeDuplicatePath
    ) {
      setLoadingHeatmap(true);
      // load images
      const channelOrSrc = async (path) => {
        if (activeChannel()) {
          try {
            return await invoke("get_channel_preview", {
              path,
              channel: activeChannel(),
            });
          } catch (e) {}
        }
        const ext = path.split(".").pop().toLowerCase();
        if (
          ["hdr", "dds", "exr", "tga", "tif", "tiff"].includes(ext) &&
          !activeChannel()
        ) {
          return await invoke("get_channel_preview", {
            path,
            channel: "Composite",
          });
        }
        return convertFileSrc(path);
      };

      if (compareMode() === "heatmap") {
        try {
          const diff = await invoke("calculate_diff", {
            file1: props.activeOriginalPath,
            file2: props.activeDuplicatePath,
          });
          setHeatmapSrc(`${convertFileSrc(diff)}?t=${Date.now()}`);
        } catch (e) {
          console.error(e);
        }
      } else {
        const s1 = await channelOrSrc(props.activeOriginalPath);
        const s2 = await channelOrSrc(props.activeDuplicatePath);
        setSrcOriginal(s1);
        setSrcDuplicate(s2);
      }
      setLoadingHeatmap(false);
    }
  });

  return (
    <div class="viewer-panel" style={{ width: `${props.width}px` }}>
      <div class="section-title" style="margin: 0; border: none; padding: 0;">
        Image Viewer
      </div>

      <Show when={!props.viewerCanvasOpen}>
        <div style="display:flex; justify-content:space-between; align-items:center; gap: 8px; width: 100%;">
          <div class="view-toggles" style="flex-shrink:0;">
            <button
              class={!props.viewerGridMode ? "active-toggle" : ""}
              onClick={() => props.setViewerGridMode(false)}
            >
              ☰
            </button>
            <button
              class={props.viewerGridMode ? "active-toggle" : ""}
              onClick={() => props.setViewerGridMode(true)}
            >
              ⊞
            </button>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin-left: auto;">
            <div style="display:flex; align-items:center; gap:6px;">
              <label style="color:var(--text-secondary); font-size:8pt;">
                Size:
              </label>
              <input
                type="range"
                min="90"
                max="220"
                value={viewerGridSize()}
                onInput={(e) => setViewerGridSize(e.target.value)}
                style="width: 70px;"
              />
            </div>
            <div style="display:flex; align-items:center; gap:6px; flex-shrink:0;">
              <span
                style={{
                  width: "12px",
                  height: "12px",
                  "border-radius": "2px",
                  display: "inline-block",
                  "background-color": "var(--brand-blue)",
                  opacity: bgAlpha() / 255.0,
                }}
              ></span>
              <label style="color:var(--text-secondary); font-size:8pt;">
                BG:
              </label>
              <input
                type="range"
                min="0"
                max="255"
                value={bgAlpha()}
                onInput={(e) => setBgAlpha(e.target.value)}
                style="width: 60px;"
              />
            </div>
          </div>
        </div>
      </Show>

      <Show when={props.viewerCanvasOpen}>
        <div style="display:flex; flex-direction:column; gap:6px; width: 100%; margin-bottom: 4px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap: 8px; width: 100%;">
            <div style="display:flex; gap:8px; align-items:center;">
              <button
                onClick={() => props.setViewerCanvasOpen(false)}
                style="font-size: 8.5pt; padding: 4px 8px; background: rgba(255,255,255,0.08); border: 1px solid var(--border-gray); color: var(--text-primary); border-radius: 4px; cursor: pointer;"
              >
                &lt; Back
              </button>
              <select
                style="width:115px; font-size:8.5pt; height:28px;"
                value={compareMode()}
                onChange={(e) => setCompareMode(e.target.value)}
              >
                <option value="side-by-side">Side-by-Side</option>
                <option value="wipe">Wipe</option>
                <option value="overlay">Overlay</option>
                <option value="heatmap">Difference</option>
              </select>
            </div>

            <div style="display:flex; align-items:center; gap:12px; margin-left: auto;">
              <div style="display:flex; align-items:center; gap:4px; flex-shrink:0;">
                <span
                  style={{
                    width: "10px",
                    height: "10px",
                    "border-radius": "2px",
                    display: "inline-block",
                    "background-color": "var(--brand-blue)",
                    opacity: bgAlpha() / 255.0,
                  }}
                ></span>
                <label style="color:var(--text-secondary); font-size:7.5pt;">
                  BG:
                </label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={bgAlpha()}
                  onInput={(e) => setBgAlpha(e.target.value)}
                  style="width: 50px;"
                />
              </div>

              <div class="channel-toggles" style="display: flex; gap: 4px;">
                <button
                  class="channel-btn"
                  style={{
                    "background-color": "var(--danger-red)",
                    color: "black",
                    "font-size": "8pt",
                    "font-weight": "bold",
                    width: "22px",
                    height: "22px",
                    display: "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "border-radius": "3px",
                    border:
                      activeChannel() === "R"
                        ? "2px solid white"
                        : "2px solid transparent",
                  }}
                  onClick={() =>
                    setActiveChannel(activeChannel() === "R" ? null : "R")
                  }
                >
                  R
                </button>
                <button
                  class="channel-btn"
                  style={{
                    "background-color": "var(--success-green)",
                    color: "black",
                    "font-size": "8pt",
                    "font-weight": "bold",
                    width: "22px",
                    height: "22px",
                    display: "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "border-radius": "3px",
                    border:
                      activeChannel() === "G"
                        ? "2px solid white"
                        : "2px solid transparent",
                  }}
                  onClick={() =>
                    setActiveChannel(activeChannel() === "G" ? null : "G")
                  }
                >
                  G
                </button>
                <button
                  class="channel-btn"
                  style={{
                    "background-color": "var(--brand-blue)",
                    color: "black",
                    "font-size": "8pt",
                    "font-weight": "bold",
                    width: "22px",
                    height: "22px",
                    display: "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "border-radius": "3px",
                    border:
                      activeChannel() === "B"
                        ? "2px solid white"
                        : "2px solid transparent",
                  }}
                  onClick={() =>
                    setActiveChannel(activeChannel() === "B" ? null : "B")
                  }
                >
                  B
                </button>
                <button
                  class="channel-btn"
                  style={{
                    "background-color": "#ffffff",
                    color: "black",
                    "font-size": "8pt",
                    "font-weight": "bold",
                    width: "22px",
                    height: "22px",
                    display: "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "border-radius": "3px",
                    border:
                      activeChannel() === "A"
                        ? "2px solid #2563eb"
                        : "2px solid transparent",
                  }}
                  onClick={() =>
                    setActiveChannel(activeChannel() === "A" ? null : "A")
                  }
                >
                  A
                </button>
              </div>
            </div>
          </div>

          <Show when={compareMode() === "wipe" || compareMode() === "overlay"}>
            <div style="display:flex; align-items:center; gap:8px; background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-gray);">
              <label style="font-size: 8pt; color: var(--text-secondary); white-space: nowrap;">
                {compareMode() === "wipe" ? "Wipe Shift:" : "Blend Opacity:"}
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={wipeSlider()}
                onInput={(e) => setWipeSlider(e.target.value)}
                style="flex: 1;"
              />
              <span style="font-size: 8pt; color: var(--text-primary); min-width: 30px; text-align: right;">
                {wipeSlider()}%
              </span>
            </div>
          </Show>
        </div>
      </Show>

      <Show when={!props.viewerCanvasOpen}>
        <button
          disabled={!activeGroup() || activeGroup().files?.length < 2}
          onClick={() => props.setViewerCanvasOpen(true)}
          style="width: 100%;"
        >
          Compare ({activeGroup()?.files ? activeGroup().files.length - 1 : 0})
        </button>

        <div
          class={`viewer-list-container ${
            props.viewerGridMode ? "grid-mode" : ""
          }`}
          style={{
            "--viewer-grid-size": `${viewerGridSize()}px`,
          }}
        >
          <Show when={activeGroup()?.files}>
            <For each={activeGroup().files}>
              {(file, idx) => (
                <div
                  class={`viewer-item ${
                    file.path === props.activeDuplicatePath ||
                    (idx === 0 && file.path === props.activeOriginalPath)
                      ? "active"
                      : ""
                  }`}
                  onClick={() => {
                    if (idx === 0) props.setActiveOriginalPath(file.path);
                    else props.setActiveDuplicatePath(file.path);
                  }}
                >
                  <ThumbImage
                    path={file.path}
                    showCorners={true}
                    wrapperClass="viewer-item-thumb-wrapper"
                    imgClass="viewer-item-thumb"
                    onCornerClick={(channel) => {
                      setActiveChannel(
                        activeChannel() === channel ? null : channel
                      );
                      if (idx === 0) props.setActiveOriginalPath(file.path);
                      else props.setActiveDuplicatePath(file.path);
                      props.setViewerCanvasOpen(true);
                    }}
                  />
                  <div
                    class="viewer-item-details"
                    style="display:flex; flex-direction:column; gap:2px; overflow:hidden; flex:1; width:100%;"
                  >
                    <span
                      class="viewer-item-title"
                      style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
                    >
                      {file.path.split(/[\\/]/).pop()}
                    </span>
                    <span
                      class="viewer-item-meta"
                      style="font-size:7.5pt; color:var(--text-secondary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
                      title={`${file.width}x${file.height} • ${
                        file.compression_format ||
                        file.format_str ||
                        "Uncompressed"
                      } • ${file.bit_depth || 8}-bit • ${
                        file.color_space || "sRGB"
                      } • Mips: ${file.mipmap_count || 1} • ${
                        file.has_alpha ? "Alpha" : "No Alpha"
                      } • ${formatSize(file.size)}`}
                    >
                      {(typeof idx === "function" ? idx() : idx) === 0
                        ? "[Best]"
                        : file.similarity !== undefined && file.similarity < 100
                        ? `${file.similarity.toFixed(1)}%`
                        : "Dup"}{" "}
                      • {file.width}x{file.height} •{" "}
                      {file.compression_format ||
                        file.format_str ||
                        "Uncompressed"}{" "}
                      • {file.bit_depth || 8}b • {file.color_space} • Mips:{" "}
                      {file.mipmap_count || 1} •{" "}
                      {file.has_alpha ? "Alpha" : "No-Alpha"} •{" "}
                      {formatSize(file.size)}
                    </span>
                  </div>
                </div>
              )}
            </For>
          </Show>
        </div>
      </Show>

      <Show when={props.viewerCanvasOpen}>
        <div class="viewer-viewport checkered-backdrop">
          <div id="bg-overlay" style={{ opacity: bgAlpha() / 255.0 }}></div>
          <Show when={compareMode() === "side-by-side"}>
            <div style="display:flex; width:100%; height:100%; gap:5px; padding:5px; z-index:2; position:relative;">
              <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; overflow:hidden;">
                <span style="font-size:7.5pt; color:var(--text-secondary); margin-bottom:4px; flex-shrink:0;">
                  ORIGINAL
                </span>
                <div
                  class="compare-square-container checkered-backdrop"
                  style="max-height: calc(100% - 18px);"
                >
                  <img
                    src={srcOriginal()}
                    style="width:100%; height:100%; object-fit:contain;"
                  />
                </div>
              </div>
              <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; overflow:hidden;">
                <span style="font-size:7.5pt; color:var(--text-secondary); margin-bottom:4px; flex-shrink:0;">
                  DUPLICATE
                </span>
                <div
                  class="compare-square-container checkered-backdrop"
                  style="max-height: calc(100% - 18px);"
                >
                  <img
                    src={srcDuplicate()}
                    style="width:100%; height:100%; object-fit:contain;"
                  />
                </div>
              </div>
            </div>
          </Show>

          <Show when={compareMode() === "wipe" || compareMode() === "overlay"}>
            <div
              class="compare-square-container checkered-backdrop"
              style="z-index: 2;"
            >
              <div
                ref={wipeViewportRef}
                onMouseDown={handleWipeStart}
                class="wipe-viewport"
                style="width: 100%; height: 100%;"
              >
                <div class="wipe-layer">
                  <img src={srcOriginal()} />
                </div>
                <div
                  class="wipe-layer"
                  style={
                    compareMode() === "wipe"
                      ? {
                          "clip-path": `polygon(0 0, ${wipeSlider()}% 0, ${wipeSlider()}% 100%, 0 100%)`,
                        }
                      : { opacity: wipeSlider() / 100.0 }
                  }
                >
                  <img src={srcDuplicate()} />
                </div>
                <Show when={compareMode() === "wipe"}>
                  <div
                    id="wipe-line"
                    style={{ left: `${wipeSlider()}%` }}
                  ></div>
                </Show>
              </div>
            </div>
          </Show>

          <Show when={compareMode() === "heatmap"}>
            <div class="compare-square-container" style="z-index: 2;">
              <Show
                when={loadingHeatmap()}
                fallback={
                  <img
                    src={heatmapSrc()}
                    style="width:100%; height:100%; object-fit:contain; border-radius: 4px;"
                  />
                }
              >
                <div style="color: #949ba4; text-align: center;">
                  Evaluating difference map...
                </div>
              </Show>
            </div>
          </Show>
        </div>
      </Show>

      <Show when={originalFile() || duplicateFile()}>
        <div
          style={{
            margin: "6px -12px -12px -12px",
            "border-top": "2px solid var(--border-gray)",
            background: "var(--darkest)",
            display: "flex",
            "flex-direction": "column",
            transition: "max-height 0.2s ease-in-out",
            "max-height": isSpecsCollapsed() ? "36px" : "600px",
            overflow: "hidden",
          }}
        >
          <div
            onClick={() => setIsSpecsCollapsed(!isSpecsCollapsed())}
            style={{
              "font-size": "8pt",
              "font-weight": "bold",
              "text-transform": "uppercase",
              "border-bottom": isSpecsCollapsed()
                ? "none"
                : "1px solid var(--border-gray)",
              padding: "0 12px",
              display: "flex",
              "justify-content": "space-between",
              "align-items": "center",
              "user-select": "none",
              height: "36px",
              "background-color": "var(--darkest)",
              cursor: "pointer",
            }}
          >
            <div style="display: flex; align-items: center; gap: 8px;">
              <button
                class="detailed-log-btn"
                style="padding: 2px 6px; font-weight: bold;"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsSpecsCollapsed(!isSpecsCollapsed());
                }}
              >
                {isSpecsCollapsed() ? "▲ EXPAND" : "▼ COLLAPSE"}
              </button>
              <span style="color: var(--text-primary); letter-spacing: 0.5px;">
                Format Specs Comparison
              </span>
            </div>
            <span style="font-size:7pt; font-weight: normal; text-transform: none; color: #888;">
              Complete Image Metadata
            </span>
          </div>
          <div
            style={{
              padding: "10px",
              display: "flex",
              "flex-direction": "column",
              gap: "8px",
            }}
          >
            <table style="width: 100%; border-collapse: collapse; font-size: 8pt; text-align: left; table-layout: fixed;">
              <thead>
                <tr style="border-bottom: 1px solid var(--border-gray); color: var(--text-secondary); font-size:7.5pt;">
                  <th style="padding: 3px 0; width: 25%;">Property</th>
                  <th style="padding: 3px 0; width: 37.5%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    Original
                  </th>
                  <th style="padding: 3px 0; width: 37.5%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    Duplicate
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    File Name
                  </td>
                  <td
                    style="padding: 5px 0; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    title={originalFile()?.path.split(/[\\/]/).pop()}
                  >
                    {originalFile()?.path.split(/[\\/]/).pop() || "-"}
                  </td>
                  <td
                    style="padding: 5px 0; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    title={duplicateFile()?.path.split(/[\\/]/).pop()}
                  >
                    {duplicateFile()?.path.split(/[\\/]/).pop() || "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    File Size
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile() ? formatSize(originalFile().size) : "-"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile() ? formatSize(duplicateFile().size) : "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Format / Codec
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()?.compression_format ||
                      originalFile()?.format_str ||
                      "-"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.compression_format ||
                      duplicateFile()?.format_str ||
                      "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Resolution
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()
                      ? `${originalFile().width}x${originalFile().height}`
                      : "-"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()
                      ? `${duplicateFile().width}x${duplicateFile().height}`
                      : "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Bit Depth
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()?.bit_depth
                      ? `${originalFile().bit_depth}-bit`
                      : "-"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.bit_depth
                      ? `${duplicateFile().bit_depth}-bit`
                      : "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Color Space
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()?.color_space || "-"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.color_space || "-"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Mipmaps
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()?.mipmap_count || "1"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.mipmap_count || "1"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Alpha Channel
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {originalFile()?.has_alpha ? "Yes" : "No"}
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.has_alpha ? "Yes" : "No"}
                  </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary);">
                    Similarity
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    [Best match]
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary);">
                    {duplicateFile()?.similarity !== undefined
                      ? `${duplicateFile().similarity.toFixed(2)}%`
                      : "Duplicate"}
                  </td>
                </tr>
                <tr>
                  <td style="padding: 5px 0; font-weight: 500; color: var(--text-secondary); vertical-align: top;">
                    Full Path
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary); font-size:7.5pt; font-family: monospace;">
                    <div style="display: flex; flex-direction: column; gap: 4px; overflow: hidden;">
                      <span
                        style="user-select: all; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;"
                        title={originalFile()?.path}
                      >
                        {originalFile()?.path}
                      </span>
                      <button
                        onClick={() =>
                          navigator.clipboard.writeText(
                            originalFile()?.path || ""
                          )
                        }
                        style="font-size:7pt; padding: 2px 4px; align-self: flex-start; cursor: pointer; background: rgba(255,255,255,0.07); border: 1px solid var(--border-gray); color: var(--text-primary); border-radius: 3px;"
                      >
                        Copy Path
                      </button>
                    </div>
                  </td>
                  <td style="padding: 5px 0; color: var(--text-primary); font-size:7.5pt; font-family: monospace;">
                    <div style="display: flex; flex-direction: column; gap: 4px; overflow: hidden;">
                      <span
                        style="user-select: all; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;"
                        title={duplicateFile()?.path}
                      >
                        {duplicateFile()?.path}
                      </span>
                      <button
                        onClick={() =>
                          navigator.clipboard.writeText(
                            duplicateFile()?.path || ""
                          )
                        }
                        style="font-size:7pt; padding: 2px 4px; align-self: flex-start; cursor: pointer; background: rgba(255,255,255,0.07); border: 1px solid var(--border-gray); color: var(--text-primary); border-radius: 3px;"
                      >
                        Copy Path
                      </button>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </Show>
    </div>
  );
}
