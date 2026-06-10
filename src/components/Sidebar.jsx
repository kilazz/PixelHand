import { For, Show, createSignal, createEffect } from "solid-js";

export default function Sidebar(props) {
  let logConsoleRef;

  createEffect(() => {
    if (logConsoleRef && props.logs.length > 0) {
      logConsoleRef.scrollTop = logConsoleRef.scrollHeight;
    }
  });

  return (
    <div class="sidebar-panel" style={{ width: `${props.width}px` }}>
      <div class="panel-section">
        <div class="section-title">Scan Configuration</div>
        <div class="form-group row-layout">
          <label>Folder A:</label>
          <div style="display:flex; gap:4px; flex:1;">
            <input
              type="text"
              placeholder="C:\\Path\\To\\Target"
              value={props.dirA()}
              onInput={(e) => props.setDirA(e.target.value)}
              style="flex:1;"
            />
            <button
              onClick={() =>
                props.setDirA("C:\\Users\\Zed\\Pictures\\Screenshots")
              }
              style="padding: 4px 8px;"
            >
              ...
            </button>
          </div>
        </div>

        <Show when={props.qcModeCheck()}>
          <div class="form-group">
            <label>Compare Folder B:</label>
            <div style="display:flex; gap:4px; width:100%;">
              <input
                type="text"
                placeholder="C:\\Path\\To\\Build"
                value={props.dirB()}
                onInput={(e) => props.setDirB(e.target.value)}
                style="flex:1;"
              />
              <button
                onClick={() =>
                  props.setDirB("C:\\Users\\Zed\\Pictures\\BuildScreenshots")
                }
                style="padding: 4px 8px;"
              >
                ...
              </button>
            </div>
          </div>
        </Show>

        <div class="form-group">
          <label>Find (Search AI):</label>
          <div style="display:flex; gap:4px; width:100%;">
            <input
              type="text"
              placeholder="Enter text or drag image..."
              value={props.semanticQuery()}
              onInput={(e) => props.setSemanticQuery(e.target.value)}
              style="flex:1;"
            />
          </div>
          <Show when={props.selectedSamplePath()}>
            <div style="font-size:7.5pt; color:var(--text-secondary); margin-top:2px;">
              Sample: {props.selectedSamplePath()?.split(/[\\/]/).pop()}
            </div>
          </Show>
        </div>

        <div class="form-group row-layout">
          <label>Similarity Threshold:</label>
          <div style="display:flex; align-items:center; gap:4px; width: 120px;">
            <input
              type="number"
              min="0"
              max="100"
              value={props.similarityThreshold()}
              onInput={(e) => props.setSimilarityThreshold(e.target.value)}
              style="flex:1;"
            />
            <span style="color:var(--text-secondary); font-size:9pt;">%</span>
          </div>
        </div>

        <button
          class="success-btn"
          disabled={props.isScanning()}
          onClick={props.onScan}
          style="width:100%; height:32px; font-size:9.5pt; font-weight:bold; margin-top:8px;"
        >
          {props.isScanning() ? "Processing Assets..." : "Start Scan"}
        </button>
      </div>

      <div class="panel-section">
        <div class="section-title">Quality Control (QC) Options</div>
        <label
          class="checkbox-container"
          style="color: var(--warning-yellow); font-weight: bold; margin-bottom: 6px;"
        >
          <input
            type="checkbox"
            checked={props.qcModeCheck()}
            onChange={(e) => props.setQcModeCheck(e.target.checked)}
          />
          Enable Technical QC Mode
        </label>

        <div
          class="checkbox-grid"
          style={{
            opacity: props.qcModeCheck() ? "1" : "0.5",
            "pointer-events": props.qcModeCheck() ? "auto" : "none",
          }}
        >
          <label class="checkbox-container">
            <input
              type="checkbox"
              checked={props.qcNpotCheck()}
              onChange={(e) => props.setQcNpotCheck(e.target.checked)}
            />
            Check NPOT (Power of 2)
          </label>
          <label class="checkbox-container">
            <input
              type="checkbox"
              checked={props.qcMipmapsCheck()}
              onChange={(e) => props.setQcMipmapsCheck(e.target.checked)}
            />
            Check Mip-Maps
          </label>
          <label class="checkbox-container">
            <input
              type="checkbox"
              checked={props.qcBlockAlignCheck()}
              onChange={(e) => props.setQcBlockAlignCheck(e.target.checked)}
            />
            Check BlockAlignment (4px)
          </label>
          <label class="checkbox-container">
            <input
              type="checkbox"
              checked={props.qcBitDepthCheck()}
              onChange={(e) => props.setQcBitDepthCheck(e.target.checked)}
            />
            Check Bit Depth
          </label>
          <label class="checkbox-container" style="margin-top: 6px;">
            <input
              type="checkbox"
              checked={props.qcNormalMapsCheck()}
              onChange={(e) => props.setQcNormalMapsCheck(e.target.checked)}
            />
            Validate Normal Maps
          </label>
          <input
            type="text"
            placeholder="Tags: _norm, _ddn (empty = check all)"
            value={props.qcNormalsTags()}
            onInput={(e) => props.setQcNormalsTags(e.target.value)}
            disabled={!props.qcNormalMapsCheck()}
            style="margin-top:4px;"
          />
        </div>
      </div>

      <div class="panel-section">
        <div class="section-title">Search Methods (Non-QC)</div>
        <div class="sub-group-container">
          <label class="checkbox-container">
            <input
              type="radio"
              name="searchMethod"
              value="exact"
              checked={props.searchMethod() === "exact"}
              onChange={() => props.setSearchMethod("exact")}
            />
            Exact Match (xxHash)
          </label>
          <label class="checkbox-container">
            <input
              type="radio"
              name="searchMethod"
              value="simple"
              checked={props.searchMethod() === "simple"}
              onChange={() => props.setSearchMethod("simple")}
            />
            Simple Duplicates (dHash)
          </label>
          <label class="checkbox-container">
            <input
              type="radio"
              name="searchMethod"
              value="ai"
              checked={props.searchMethod() === "ai"}
              onChange={() => props.setSearchMethod("ai")}
            />
            AI Semantic/Visual Search
          </label>
        </div>
      </div>

      <div
        class="panel-section"
        style="flex: 1; display: flex; flex-direction: column; min-height: 140px; max-height: 300px; padding: 6px;"
      >
        <div class="section-title" style="margin-bottom: 4px;">
          Log
        </div>
        <div
          ref={logConsoleRef}
          style="flex: 1; background-color: var(--darkest); border: 1px solid var(--border-gray); border-radius: 3px; padding: 4px; font-family: Consolas, Monaco, monospace; font-size: 7.5pt; overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: var(--text-secondary);"
        >
          <For each={props.logs}>
            {(log) => (
              <div class={`log-line ${log.level}`}>
                <span style="color: #7c808a;">[{log.timestamp}]</span>
                <span
                  style={{
                    color:
                      log.level === "error"
                        ? "#da373c"
                        : log.level === "warning"
                        ? "#f0b132"
                        : log.level === "success"
                        ? "#23a55a"
                        : "#949ba4",
                  }}
                >
                  {" "}
                  {log.message}
                </span>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}
