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

  const [logs, setLogs] = createSignal([]);
  const addLog = (message, level = "info") => {
    const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
    setLogs((prev) => [...prev, { timestamp, message, level }]);
  };

  const [dirA, setDirA] = createSignal("");
  const [dirB, setDirB] = createSignal("");

  const [qcModeCheck, setQcModeCheck] = createSignal(false);
  const [qcNpotCheck, setQcNpotCheck] = createSignal(true);
  const [qcMipmapsCheck, setQcMipmapsCheck] = createSignal(true);
  const [qcBlockAlignCheck, setQcBlockAlignCheck] = createSignal(true);
  const [qcBitDepthCheck, setQcBitDepthCheck] = createSignal(true);
  const [qcNormalMapsCheck, setQcNormalMapsCheck] = createSignal(true);
  const [qcNormalsTags, setQcNormalsTags] = createSignal("");

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

  onMount(async () => {
    addLog("Initializing application...");
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
      addLog("ONNX Loaded. System ready.", "success");
      setTimeout(() => setLoading(false), 800);
    } catch (error) {
      addLog(`Failed to initiate models: ${error}`, "error");
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
        addLog(`Reference image loaded: ${fileName}`);
      }
    });
    onCleanup(() => cleanupDrop());
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

    addLog(`Starting scan on: ${dirA()}`);
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
          };
        } else if (semanticQuery().trim().length > 0) {
          backendCommand = "run_ai_search";
          params = { directory: dirA().trim(), query: semanticQuery().trim() };
        } else {
          backendCommand = "run_ai_duplicate_scan";
          params = {
            directory: dirA().trim(),
            threshold: parseFloat(similarityThreshold()),
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
      addLog(`Finished in ${elapsed}s`, "success");
    } catch (error) {
      addLog(`Scan failed: ${error}`, "error");
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
              logs={logs()}
              dirA={dirA}
              setDirA={setDirA}
              dirB={dirB}
              setDirB={setDirB}
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
                  path || (group.files.length > 1 ? group.files[1].path : null);
                setActiveDuplicatePath(
                  dup === first && group.files.length > 1
                    ? group.files[1].path
                    : dup
                );
              }
            }}
          />

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
