import { For, Show, createSignal, createEffect } from "solid-js";

export default function Sidebar(props) {
  let logConsoleRef;

  const [showExtModal, setShowExtModal] = createSignal(false);
  const [modalPos, setModalPos] = createSignal({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = createSignal(false);
  let dragStart = { x: 0, y: 0 };
  let modalStart = { x: 0, y: 0 };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return; // Only left click
    e.preventDefault();
    setIsDragging(true);
    dragStart = { x: e.clientX, y: e.clientY };
    modalStart = { ...modalPos() };

    const handleMouseMove = (moveEv) => {
      moveEv.preventDefault();
      const dx = moveEv.clientX - dragStart.x;
      const dy = moveEv.clientY - dragStart.y;
      setModalPos({
        x: modalStart.x + dx,
        y: modalStart.y + dy,
      });
    };

    const handleMouseUp = (upEv) => {
      upEv.preventDefault();
      setIsDragging(false);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  };

  const toggleExt = (ext) => {
    let curr = props.selectedExtensions();
    if (curr.includes(ext)) {
      props.setSelectedExtensions(curr.filter((e) => e !== ext));
    } else {
      props.setSelectedExtensions([...curr, ext]);
    }
  };

  createEffect(() => {
    if (logConsoleRef && props.logs?.length > 0) {
      logConsoleRef.scrollTop = logConsoleRef.scrollHeight;
    }
  });

  return (
    <div class="sidebar-panel" style={{ width: `${props.width}px` }}>
      <Show when={showExtModal()}>
        <div style="position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.75); z-index:9999; display:flex; align-items:center; justify-content:center;">
          <div
            style={`background-color: #2b2d31 !important; border: 1px solid var(--border-gray); padding: 20px; border-radius: 8px; width: 450px; max-height: 80vh; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.8); transform: translate(${
              modalPos().x
            }px, ${modalPos().y}px);`}
          >
            <h3
              style="margin:0 0 16px 0; color:white; font-size: 14pt; cursor: move; user-select: none; padding-bottom: 8px; border-bottom: 1px solid var(--border-gray);"
              onMouseDown={handleMouseDown}
            >
              Select File Types
              <span style="font-size: 8pt; font-weight: normal; color: var(--text-secondary); margin-left: 10px;">
                (Drag here to move)
              </span>
            </h3>
            <div style="flex:1; overflow-y:auto; display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-bottom:20px;">
              <For each={props.allExtensions}>
                {(ext) => (
                  <label
                    class="checkbox-container"
                    style="margin:0; font-size:9pt; cursor: pointer; user-select: none;"
                  >
                    <input
                      type="checkbox"
                      checked={props.selectedExtensions().includes(ext)}
                      onChange={() => toggleExt(ext)}
                    />
                    {ext}
                  </label>
                )}
              </For>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div style="display:flex; gap:8px;">
                <button
                  class="primary-btn"
                  style="background:var(--button-gray); border:1px solid var(--border-gray); color:var(--text-primary);"
                  onClick={() =>
                    props.setSelectedExtensions([...props.allExtensions])
                  }
                >
                  Select All
                </button>
                <button
                  class="primary-btn"
                  style="background:var(--button-gray); border:1px solid var(--border-gray); color:var(--text-primary);"
                  onClick={() => props.setSelectedExtensions([])}
                >
                  Select None
                </button>
              </div>
              <button
                class="primary-btn"
                onClick={() => setShowExtModal(false)}
                style="min-width: 80px; font-weight: bold;"
              >
                OK
              </button>
            </div>
          </div>
        </div>
      </Show>
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
                props.onSelectDirA
                  ? props.onSelectDirA()
                  : props.setDirA("C:\\Users\\Zed\\Pictures\\Screenshots")
              }
              style="padding: 4px 8px; font-weight: bold; cursor: pointer;"
              title="Select Folder"
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
                  props.onSelectDirB
                    ? props.onSelectDirB()
                    : props.setDirB(
                        "C:\\Users\\Zed\\Pictures\\BuildScreenshots"
                      )
                }
                style="padding: 4px 8px; font-weight: bold; cursor: pointer;"
                title="Select Folder"
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

        <Show when={props.searchMethod() === "ai"}>
          <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 4px;">
            <div
              class="form-group row-layout"
              style="margin: 0; display: flex; align-items: center; justify-content: space-between;"
            >
              <label style="font-size: 8.5pt;">
                AI Batch Size (even recommended):
              </label>
              <input
                type="number"
                min="1"
                step="2"
                value={props.aiBatchSize()}
                onInput={(e) => {
                  const val = parseInt(e.target.value, 10);
                  if (!isNaN(val) && val > 0) {
                    props.setAiBatchSize(val);
                  }
                }}
                style="width: 80px; padding: 4px; text-align: center; background: var(--darkest); color: var(--text-primary); border: 1px solid var(--border-gray); border-radius: 4px; font-size: 9pt; height: 26px; box-sizing: border-box;"
              />
            </div>
            {/* Quick-select optimal presets */}
            <div style="display: flex; flex-direction: column; gap: 4px; margin-left: 2px;">
              <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-top: 2px;">
                <For each={[32, 64, 128, 256, 512]}>
                  {(preset) => (
                    <button
                      onClick={() => props.setAiBatchSize(preset)}
                      style={{
                        padding: "2px 6px",
                        "font-size": "7.5pt",
                        "font-family": "var(--font-mono)",
                        background:
                          props.aiBatchSize() === preset
                            ? "rgba(40, 167, 69, 0.25)"
                            : "var(--button-gray)",
                        color:
                          props.aiBatchSize() === preset
                            ? "#28a745"
                            : "var(--text-secondary)",
                        border:
                          props.aiBatchSize() === preset
                            ? "1px solid #28a745"
                            : "1px solid var(--border-gray)",
                        "border-radius": "3px",
                        cursor: "pointer",
                        transition: "all 0.1s",
                      }}
                    >
                      {preset}
                    </button>
                  )}
                </For>
              </div>
              <div style="font-size: 7.5pt; color: var(--text-secondary); margin-top: 2px; display: flex; align-items: center; gap: 4px;">
                <span
                  style={{
                    color:
                      props.aiBatchSize() % 2 === 0 &&
                      (props.aiBatchSize() & (props.aiBatchSize() - 1)) === 0
                        ? "#28a745"
                        : "#e0a800",
                  }}
                >
                  ●
                </span>
                <span>
                  {props.aiBatchSize() % 2 === 0 &&
                  (props.aiBatchSize() & (props.aiBatchSize() - 1)) === 0
                    ? "Optimal power of 2 for GPU"
                    : "Not power of 2. Lower speed."}
                </span>
              </div>
            </div>
          </div>
        </Show>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
          <button
            onClick={() => setShowExtModal(true)}
            style="padding: 4px 12px; font-size: 8.5pt; font-weight: bold; background: var(--brand-blue); color: white; border: 1px solid var(--brand-blue); border-radius: 4px; cursor: pointer; height: 26px; display: inline-flex; align-items: center; justify-content: center; transition: background 0.1s; box-shadow: 0 2px 4px rgba(0,0,0,0.15);"
            onMouseEnter={(e) =>
              (e.target.style.background = "var(--brand-blue-hover)")
            }
            onMouseLeave={(e) =>
              (e.target.style.background = "var(--brand-blue)")
            }
          >
            File Types...
          </button>
          <span style="font-size: 8pt; color: var(--text-secondary);">
            {props.selectedExtensions().length} types selected
          </span>
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
          <label class="checkbox-container">
            <input
              type="checkbox"
              checked={props.qcSolidColorCheck()}
              onChange={(e) => props.setQcSolidColorCheck(e.target.checked)}
            />
            Check Solid Colors
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
      <div class="panel-section">
        <div class="section-title">Execution Provider</div>
        <div class="form-group row-layout">
          <label>Backend:</label>
          <div style="display:flex; align-items:center; gap:4px; flex:1;">
            <select
              value={props.executionProvider()}
              onChange={(e) => props.setExecutionProvider(e.target.value)}
              style="flex: 1; padding: 4px; background: var(--bg-primary); color: var(--text-primary); border: 1px solid var(--border-color); border-radius: 4px; font-size: 9pt;"
            >
              <option value="CPU">CPU Mode (Fallback)</option>
              <option value="DirectML">GPU (DirectML - DX12)</option>
              <option value="CUDA">GPU (NVIDIA CUDA)</option>
              <option value="TensorRT">GPU (NVIDIA TensorRT)</option>
              <option value="CoreML">GPU (Apple CoreML)</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
