import {
  createSignal,
  createEffect,
  onMount,
  onCleanup,
  Show,
  For,
} from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Import parts
import Sidebar from "./components/Sidebar";
import ResultsPanel from "./components/ResultsPanel";
import ViewerPanel from "./components/ViewerPanel";

export default function App() {
  const [loading, setLoading] = createSignal(true);
  const [progress, setProgress] = createSignal(0);
  const [statusText, setStatusText] = createSignal(
    "Locating existing models..."
  );

  const [dirA, setDirA] = createSignal("");
  const [dirB, setDirB] = createSignal("");

  const [qcModeCheck, setQcModeCheck] = createSignal(false);
  const [qcNpotCheck, setQcNpotCheck] = createSignal(true);
  const [qcMipmapsCheck, setQcMipmapsCheck] = createSignal(true);
  const [qcBlockAlignCheck, setQcBlockAlignCheck] = createSignal(true);
  const [qcBitDepthCheck, setQcBitDepthCheck] = createSignal(true);
  const [qcNormalMapsCheck, setQcNormalMapsCheck] = createSignal(true);
  const [qcNormalsTags, setQcNormalsTags] = createSignal("");

  const [executionProvider, setExecutionProvider] = createSignal("CPU");

  const [searchMethod, setSearchMethod] = createSignal("exact");
  const [semanticQuery, setSemanticQuery] = createSignal("");
  const [selectedSamplePath, setSelectedSamplePath] = createSignal(null);
  const [similarityThreshold, setSimilarityThreshold] = createSignal(90);

  const [scanResults, setScanResults] = createSignal([]);
  const [checkedFiles, setCheckedFiles] = createSignal([]);
  const [selectedGroupIndex, setSelectedGroupIndex] = createSignal(null);
  const [isGridMode, setIsGridMode] = createSignal(false);
  const [gridSize, setGridSize] = createSignal(100);

  const [activeOriginalPath, setActiveOriginalPath] = createSignal(null);
  const [activeDuplicatePath, setActiveDuplicatePath] = createSignal(null);

  const [isScanning, setIsScanning] = createSignal(false);

  const [viewerCanvasOpen, setViewerCanvasOpen] = createSignal(false);
  const [viewerGridMode, setViewerGridMode] = createSignal(false);

  const [detailedLogs, setDetailedLogs] = createSignal([]);
  const [isLogCollapsed, setIsLogCollapsed] = createSignal(false);
  const [logFilter, setLogFilter] = createSignal("all");
  const [logSearchQuery, setLogSearchQuery] = createSignal("");
  let detailedLogRef;

  // Function to add detailed log entries with level (info, success, warning, error)
  const appendDetailedLog = (message, level = "info") => {
    const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
    setDetailedLogs((prev) => [...prev, { timestamp, message, level }]);

    // Auto scroll down to the bottom
    if (detailedLogRef) {
      setTimeout(() => {
        detailedLogRef.scrollTop = detailedLogRef.scrollHeight;
      }, 30);
    }
  };

  const clearDetailedLogs = () => {
    setDetailedLogs([]);
  };

  const exportDetailedLogs = () => {
    try {
      const textContent = detailedLogs()
        .map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`)
        .join("\r\n");
      const blob = new Blob([textContent], {
        type: "text/plain;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `pixelhand_log_${new Date()
        .toISOString()
        .slice(0, 10)}.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      appendDetailedLog("Logs exported successfully", "success");
    } catch (e) {
      appendDetailedLog(`Failed to export logs: ${e}`, "error");
    }
  };

  const handleSelectDirA = async () => {
    try {
      appendDetailedLog("Opening system folder selector for Folder A...");
      const selected = await invoke("select_folder");
      if (selected) {
        setDirA(selected);
        appendDetailedLog(
          `Successfully selected Folder A: ${selected}`,
          "success"
        );
      } else {
        appendDetailedLog("Folder selection A canceled", "warning");
      }
    } catch (err) {
      appendDetailedLog(`Failed to select directory option A: ${err}`, "error");
    }
  };

  const handleSelectDirB = async () => {
    try {
      appendDetailedLog("Opening system folder selector for Folder B...");
      const selected = await invoke("select_folder");
      if (selected) {
        setDirB(selected);
        appendDetailedLog(
          `Successfully selected Folder B: ${selected}`,
          "success"
        );
      } else {
        appendDetailedLog("Folder selection B canceled", "warning");
      }
    } catch (err) {
      appendDetailedLog(`Failed to select directory option B: ${err}`, "error");
    }
  };

  onMount(async () => {
    appendDetailedLog("Initializing application architecture...");
    setStatusText("Verifying neural network models...");
    let unlisten;
    try {
      unlisten = await listen("download-progress", (event) => {
        const { file_name, percentage } = event.payload;
        setProgress(percentage);
        setStatusText(`Downloading ${file_name}: ${percentage.toFixed(1)}%`);
      });
      await invoke("download_models");
      setStatusText("Initialization complete!");
      setProgress(100);
      appendDetailedLog("ONNX Loaded. System ready.", "success");
      setTimeout(() => setLoading(false), 800);
    } catch (error) {
      appendDetailedLog(`Failed to initiate models: ${error}`, "error");
      setTimeout(() => setLoading(false), 1000); // let them in anyway
    } finally {
      if (unlisten) unlisten();
    }

    // Listen to drag-drop
    const cleanupDrop = await listen("tauri://drag-drop", (event) => {
      const paths = event.payload.paths;
      if (paths && paths.length > 0) {
        setSelectedSamplePath(paths[0]);
        setSearchMethod("ai");
        const fileName = paths[0].split(/[\\/]/).pop();
        appendDetailedLog(`Reference image loaded: ${fileName}`);
      }
    });

    const cleanupLog = await listen("backend-log", (event) => {
      const { message, level } = event.payload;
      appendDetailedLog(message, level);
    });

    onCleanup(() => {
      cleanupDrop();
      cleanupLog();
    });
  });

  const handleScan = async () => {
    if (!dirA().trim()) {
      alert("Please specify Target Folder A!");
      return;
    }
    setIsScanning(true);
    setCheckedFiles([]);
    setScanResults([]);
    setSelectedGroupIndex(null);
    setViewerCanvasOpen(false);

    appendDetailedLog(
      `Starting scan on: ${dirA()} (Method: ${searchMethod()})`
    );
    const startTime = performance.now();

    try {
      let backendCommand = "";
      let params = {};

      if (qcModeCheck()) {
        if (dirB().trim()) {
          backendCommand = "run_folder_compare";
          params = {
            directoryA: dirA().trim(),
            directoryB: dirB().trim(),
            checkSizeBloat: true,
            checkAlpha: true,
            checkColorSpace: true,
            checkCompression: true,
            matchByStem: true,
          };
        } else {
          backendCommand = "run_qc_scan";
          params = {
            directory: dirA().trim(),
            checkNpot: qcNpotCheck(),
            checkMipmaps: qcMipmapsCheck(),
            checkBlockAlign: qcBlockAlignCheck(),
            checkBitDepth: qcBitDepthCheck(),
            validateNormals: qcNormalMapsCheck(),
            normalsTags: qcNormalsTags().trim(),
          };
        }
      } else if (searchMethod() === "ai") {
        if (selectedSamplePath()) {
          backendCommand = "run_image_search";
          params = {
            directory: dirA().trim(),
            referenceImage: selectedSamplePath(),
            executionProvider: executionProvider(),
          };
        } else if (semanticQuery().trim().length > 0) {
          backendCommand = "run_ai_search";
          params = {
            directory: dirA().trim(),
            query: semanticQuery().trim(),
            executionProvider: executionProvider(),
          };
        } else {
          backendCommand = "run_ai_duplicate_scan";
          params = {
            directory: dirA().trim(),
            threshold: parseFloat(similarityThreshold()),
            executionProvider: executionProvider(),
          };
        }
      } else if (searchMethod() === "simple") {
        backendCommand = "run_perceptual_scan";
        params = {
          directory: dirA().trim(),
          threshold: parseInt(similarityThreshold(), 10),
          analysisType: "Composite",
          ignoreSolidChannels: true,
          usePhash: false,
        };
      } else {
        backendCommand = "run_exact_scan";
        params = { directory: dirA().trim() };
      }

      const results = await invoke(backendCommand, params);

      let finalRes = [];
      if (
        backendCommand === "run_exact_scan" ||
        backendCommand === "run_ai_duplicate_scan" ||
        backendCommand === "run_perceptual_scan"
      ) {
        finalRes = results.map((g) => ({ type: "duplicate", ...g }));
      } else if (
        backendCommand === "run_qc_scan" ||
        backendCommand === "run_folder_compare"
      ) {
        finalRes = results.map((i) => ({ type: "qc", issue: i }));
      } else if (
        backendCommand === "run_ai_search" ||
        backendCommand === "run_image_search"
      ) {
        finalRes = results.map((m) => ({ type: "ai", match: m }));
      }

      setScanResults(finalRes);

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      appendDetailedLog(
        `Scan completed: found ${finalRes.length} records in ${elapsed}s`,
        "success"
      );
    } catch (error) {
      appendDetailedLog(`Scan failed: ${error}`, "error");
      setScanResults([{ type: "error", error: String(error) }]);
    } finally {
      setIsScanning(false);
    }
  };

  // Drag resizers
  // Simple width adjustments for solid (could be complex, let's keep it simple style bindings or refs)
  const [sidebarWidth, setSidebarWidth] = createSignal(320);
  const [viewerWidth, setViewerWidth] = createSignal(450);

  let isDraggingLeft = false;
  let isDraggingRight = false;

  const handleMouseMove = (e) => {
    if (isDraggingLeft) {
      if (e.clientX < 80) setSidebarWidth(0);
      else setSidebarWidth(Math.max(280, Math.min(450, e.clientX)));
    }
    if (isDraggingRight) {
      const w = window.innerWidth - e.clientX;
      if (w < 80) setViewerWidth(0);
      else setViewerWidth(Math.max(350, Math.min(650, w)));
    }
  };
  const handleMouseUp = () => {
    isDraggingLeft = false;
    isDraggingRight = false;
  };

  onMount(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    onCleanup(() => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    });
  });

  return (
    <>
      <Show when={loading()}>
        <div id="loader-view">
          <div class="loader-title">Initializing AI Models</div>
          <div class="progress-bar-container">
            <div
              id="progress-bar-fill"
              style={{ width: `${progress()}%` }}
            ></div>
          </div>
          <div id="status-text">{statusText()}</div>
        </div>
      </Show>

      <Show when={!loading()}>
        <div id="main-app-layout" style={{ display: "flex" }}>
          <Show when={sidebarWidth() > 0}>
            <Sidebar
              width={sidebarWidth()}
              dirA={dirA}
              setDirA={setDirA}
              dirB={dirB}
              setDirB={setDirB}
              onSelectDirA={handleSelectDirA}
              onSelectDirB={handleSelectDirB}
              qcModeCheck={qcModeCheck}
              setQcModeCheck={setQcModeCheck}
              qcNpotCheck={qcNpotCheck}
              setQcNpotCheck={setQcNpotCheck}
              qcMipmapsCheck={qcMipmapsCheck}
              setQcMipmapsCheck={setQcMipmapsCheck}
              qcBlockAlignCheck={qcBlockAlignCheck}
              setQcBlockAlignCheck={setQcBlockAlignCheck}
              qcBitDepthCheck={qcBitDepthCheck}
              setQcBitDepthCheck={setQcBitDepthCheck}
              qcNormalMapsCheck={qcNormalMapsCheck}
              setQcNormalMapsCheck={setQcNormalMapsCheck}
              qcNormalsTags={qcNormalsTags}
              setQcNormalsTags={setQcNormalsTags}
              executionProvider={executionProvider}
              setExecutionProvider={setExecutionProvider}
              searchMethod={searchMethod}
              setSearchMethod={setSearchMethod}
              semanticQuery={semanticQuery}
              setSemanticQuery={setSemanticQuery}
              selectedSamplePath={selectedSamplePath}
              similarityThreshold={similarityThreshold}
              setSimilarityThreshold={setSimilarityThreshold}
              isScanning={isScanning}
              onScan={handleScan}
            />
          </Show>
          <div
            class="splitter-handle"
            onMouseDown={() => {
              isDraggingLeft = true;
            }}
          ></div>

          <div style="flex: 1; display: flex; flex-direction: column; height: 100%; overflow: hidden;">
            <div style="flex: 1; min-height: 0; display: flex; flex-direction: column;">
              <ResultsPanel
                results={scanResults()}
                checkedFiles={checkedFiles()}
                setCheckedFiles={setCheckedFiles}
                isGridMode={isGridMode()}
                setIsGridMode={setIsGridMode}
                gridSize={gridSize()}
                setGridSize={setGridSize}
                onSelectGroup={(idx, path) => {
                  setSelectedGroupIndex(idx);
                  setViewerCanvasOpen(false);
                  const group = scanResults()[idx];
                  if (group && group.files && group.files.length > 0) {
                    const first = group.files[0].path;
                    setActiveOriginalPath(first);
                    const dup =
                      path ||
                      (group.files.length > 1 ? group.files[1].path : null);
                    setActiveDuplicatePath(
                      dup === first && group.files.length > 1
                        ? group.files[1].path
                        : dup
                    );
                  }
                }}
              />
            </div>

            {/* Detailed Log Window (under Results panel) */}
            <div
              class="detailed-log-panel"
              style={{ height: isLogCollapsed() ? "36px" : "220px" }}
            >
              <div class="detailed-log-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <button
                    class="detailed-log-btn"
                    style="padding: 2px 6px; font-weight: bold;"
                    onClick={() => setIsLogCollapsed(!isLogCollapsed())}
                  >
                    {isLogCollapsed() ? "▲ EXPAND LOG" : "▼ COLLAPSE LOG"}
                  </button>
                  <span style="color: var(--text-primary); letter-spacing: 0.5px;">
                    SYSTEM & TECHNICAL CONSOLE LOGS
                  </span>
                </div>
                <div class="detailed-log-controls">
                  <span style="color: var(--text-secondary); font-size: 7.5pt;">
                    Filter:
                  </span>
                  <select
                    class="detailed-log-filter-select"
                    value={logFilter()}
                    onChange={(e) => setLogFilter(e.target.value)}
                  >
                    <option value="all">All levels</option>
                    <option value="info">Info</option>
                    <option value="success">Success</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                  </select>

                  <input
                    type="text"
                    placeholder="Search logs..."
                    class="detailed-log-search"
                    style="margin-left: 4px;"
                    value={logSearchQuery()}
                    onInput={(e) => setLogSearchQuery(e.target.value)}
                  />

                  <button
                    class="detailed-log-btn"
                    style="margin-left: 4px;"
                    onClick={clearDetailedLogs}
                  >
                    Clear
                  </button>
                  <button class="detailed-log-btn" onClick={exportDetailedLogs}>
                    Export (.txt)
                  </button>
                </div>
              </div>

              <Show when={!isLogCollapsed()}>
                <div ref={detailedLogRef} class="detailed-log-content">
                  <For
                    each={detailedLogs().filter((l) => {
                      const matchesFilter =
                        logFilter() === "all" || l.level === logFilter();
                      const matchesSearch =
                        !logSearchQuery() ||
                        l.message
                          .toLowerCase()
                          .includes(logSearchQuery().toLowerCase());
                      return matchesFilter && matchesSearch;
                    })}
                  >
                    {(log) => (
                      <div class="detailed-log-line">
                        <span class="detailed-log-time">[{log.timestamp}]</span>
                        <span
                          class="detailed-log-msg"
                          style={{
                            color:
                              log.level === "error"
                                ? "var(--danger-red)"
                                : log.level === "warning"
                                ? "var(--warning-yellow)"
                                : log.level === "success"
                                ? "var(--success-green)"
                                : "var(--text-primary)",
                          }}
                        >
                          [{log.level.toUpperCase()}] {log.message}
                        </span>
                      </div>
                    )}
                  </For>
                </div>
              </Show>
            </div>
          </div>

          <div
            class="splitter-handle"
            onMouseDown={() => {
              isDraggingRight = true;
            }}
          ></div>

          <Show when={viewerWidth() > 0}>
            <ViewerPanel
              width={viewerWidth()}
              scanResults={scanResults()}
              selectedGroupIndex={selectedGroupIndex()}
              viewerCanvasOpen={viewerCanvasOpen()}
              setViewerCanvasOpen={setViewerCanvasOpen}
              activeOriginalPath={activeOriginalPath()}
              setActiveOriginalPath={setActiveOriginalPath}
              activeDuplicatePath={activeDuplicatePath()}
              setActiveDuplicatePath={setActiveDuplicatePath}
              viewerGridMode={viewerGridMode()}
              setViewerGridMode={setViewerGridMode}
            />
          </Show>
        </div>
      </Show>
    </>
  );
}
