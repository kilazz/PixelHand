import { For, Show, createSignal, createEffect } from "solid-js";
import ThumbImage from "./ThumbImage";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";

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

  const activeGroup = () => props.scanResults[props.selectedGroupIndex];

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
        if (["hdr", "dds", "exr"].includes(ext) && !activeChannel()) {
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
          <Show when={props.viewerGridMode}>
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
          </Show>
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
          style={
            props.viewerGridMode
              ? { "--viewer-grid-size": `${viewerGridSize()}px` }
              : {}
          }
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
                  />
                  <Show when={!props.viewerGridMode}>
                    <div class="viewer-item-details">
                      <span class="viewer-item-title">
                        {file.path.split(/[\\/]/).pop()}
                      </span>
                      <span class="viewer-item-meta">
                        {idx === 0 ? "[Best]" : "Dup"}
                      </span>
                    </div>
                  </Show>
                  <Show when={props.viewerGridMode}>
                    <span class="viewer-item-title">
                      {file.path.split(/[\\/]/).pop()}
                    </span>
                  </Show>
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
              <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
                <span style="font-size:7.5pt; color:var(--text-secondary);">
                  ORIGINAL
                </span>
                <img
                  src={srcOriginal()}
                  style="max-width:100%; max-height:90%; object-fit:contain;"
                />
              </div>
              <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
                <span style="font-size:7.5pt; color:var(--text-secondary);">
                  DUPLICATE
                </span>
                <img
                  src={srcDuplicate()}
                  style="max-width:100%; max-height:90%; object-fit:contain;"
                />
              </div>
            </div>
          </Show>

          <Show when={compareMode() === "wipe" || compareMode() === "overlay"}>
            <div class="wipe-viewport">
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
                <div id="wipe-line" style={{ left: `${wipeSlider()}%` }}></div>
              </Show>
            </div>
          </Show>

          <Show when={compareMode() === "heatmap"}>
            <Show
              when={loadingHeatmap()}
              fallback={
                <img
                  src={heatmapSrc()}
                  style="max-width:100%; max-height:100%; object-fit:contain; border-radius: 4px; position:relative; z-index:2;"
                />
              }
            >
              <div style="color: #949ba4; text-align: center; z-index: 5; position: relative;">
                Evaluating difference map...
              </div>
            </Show>
          </Show>
        </div>

        <div class="viewer-controls">
          <div style="display:flex; gap:8px; align-items:center;">
            <button onClick={() => props.setViewerCanvasOpen(false)}>
              &lt; Back
            </button>
            <select
              style="width:120px;"
              value={compareMode()}
              onChange={(e) => setCompareMode(e.target.value)}
            >
              <option value="side-by-side">Side-by-Side</option>
              <option value="wipe">Wipe</option>
              <option value="overlay">Overlay</option>
              <option value="heatmap">Difference</option>
            </select>
            <div class="channel-toggles" style="margin-left:auto;">
              <button
                class="channel-btn"
                style={{
                  "background-color": "var(--danger-red)",
                  color: "black",
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
                  "background-color": "var(--text-primary)",
                  color: "black",
                  border:
                    activeChannel() === "A"
                      ? "2px solid white"
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

          <Show when={compareMode() === "wipe" || compareMode() === "overlay"}>
            <div class="form-group" style="margin: 0;">
              <label>Wipe Handle Offset:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={wipeSlider()}
                onInput={(e) => setWipeSlider(e.target.value)}
              />
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}
