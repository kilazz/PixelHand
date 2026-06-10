// src/main.js
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// Loader view selectors
const loaderView = document.getElementById('loader-view');
const mainAppLayout = document.getElementById('main-app-layout');
const progressBar = document.getElementById('progress-bar-fill');
const statusText = document.getElementById('status-text');

// Layout panel splitters
const sidebarPanel = document.getElementById('sidebar-panel');
const viewerPanel = document.getElementById('viewer-panel');
const leftSplitter = document.getElementById('splitter-left');
const rightSplitter = document.getElementById('splitter-right');

// Path configurations and inputs
const dirInputA = document.getElementById('dir-input-a');
const browseBtnA = document.getElementById('browse-btn-a');
const dirInputB = document.getElementById('dir-input-b');
const browseBtnB = document.getElementById('browse-btn-b');
const folderBGroup = document.getElementById('folder-b-group');
const semanticQuery = document.getElementById('semantic-query');
const samplePathLabel = document.getElementById('sample-path-label');

const similarityInput = document.getElementById('similarity-input');
const scanBtn = document.getElementById('scan-btn');
const logConsole = document.getElementById('log-console');

// QC option checkboxes
const qcModeCheck = document.getElementById('qc-mode-check');
const qcOptionsContainer = document.getElementById('qc-options-container');
const qcNpotCheck = document.getElementById('qc-npot-check');
const qcMipmapsCheck = document.getElementById('qc-mipmaps-check');
const qcBlockAlignCheck = document.getElementById('qc-block-align-check');
const qcBitDepthCheck = document.getElementById('qc-bit-depth-check');
const qcNormalMapsCheck = document.getElementById('qc-normal-maps-check');
const qcNormalsTags = document.getElementById('qc-normals-tags');

// Search method toggles & advanced perceptual options
const advancedHashingSection = document.getElementById('advanced-hashing-section');
const searchMethods = document.querySelectorAll('input[name="searchMethod"]');
const rSimple = document.querySelector('input[value="simple"]');
const rExact = document.querySelector('input[value="exact"]');
const rAi = document.querySelector('input[value="ai"]');
const perceptualUsePhash = document.getElementById('perceptual-use-phash');
const perceptualIgnoreSolid = document.getElementById('perceptual-ignore-solid');
const perceptualChannelSelect = document.getElementById('perceptual-channel-select');

// Results view and action panels
const resultsCountTitle = document.getElementById('results-count-title');
const resultsViewport = document.getElementById('results-list-viewport');
const resultsPlaceholderText = document.getElementById('results-placeholder-text');

const expandAllBtn = document.getElementById('expand-all-btn');
const collapseAllBtn = document.getElementById('collapse-all-btn');
const selectAllBtn = document.getElementById('select-all-btn');
const deselectAllBtn = document.getElementById('deselect-all-btn');
const selectExceptBestBtn = document.getElementById('select-except-best-btn');
const invertSelectionBtn = document.getElementById('invert-selection-btn');

const btnGenerateReport = document.getElementById('btn-generate-report');
const btnHardlink = document.getElementById('btn-hardlink');
const btnReflink = document.getElementById('btn-reflink');
const btnTrash = document.getElementById('btn-trash');

const btnListView = document.getElementById('btn-list-view');
const btnGridView = document.getElementById('btn-grid-view');

// Interactive viewer panel elements
const btnViewerListMode = document.getElementById('btn-viewer-list-mode');
const btnViewerGridMode = document.getElementById('btn-viewer-grid-mode');
const viewerBgSwatch = document.getElementById('viewer-bg-swatch');
const viewerBgAlphaSlider = document.getElementById('viewer-bg-alpha-slider');
const bgOverlay = document.getElementById('bg-overlay');
const btnCompareTrigger = document.getElementById('btn-compare-trigger');
const viewerListContainer = document.getElementById('viewer-list-container');

// Comparison canvases
const viewerCanvas = document.getElementById('viewer-canvas');
const viewerCompareControls = document.getElementById('viewer-compare-controls');
const compareWipeViewport = document.getElementById('compare-wipe-viewport');
const wipeImgBg = document.getElementById('wipe-img-bg');
const wipeImgFg = document.getElementById('wipe-img-fg');
const wipeLayerFg = document.getElementById('wipe-layer-fg');
const wipeLine = document.getElementById('wipe-line');
const viewerPlaceholder = document.getElementById('viewer-placeholder');
const compareModeSelect = document.getElementById('compare-mode-select');
const wipeSliderGroup = document.getElementById('wipe-slider-group');
const wipeSlider = document.getElementById('wipe-slider');
const btnBackToList = document.getElementById('btn-back-to-list');
const channelButtons = document.querySelectorAll('.channel-btn');

// Size sliders for preview grids
const resultsSliderGroup = document.getElementById('results-slider-group');
const resultsGridSizeSlider = document.getElementById('results-grid-size-slider');
const viewerSizeSliderGroup = document.getElementById('viewer-size-slider-group');
const viewerGridSizeSlider = document.getElementById('viewer-grid-size-slider');

// State Management
let currentScanResults = [];
let selectedGroupIndex = null;
let checkedFiles = [];
let selectedSamplePath = null;
let isGridMode = false;
let isViewerGridMode = false;
let activeOriginalPath = null;
let activeDuplicatePath = null;
let activeChannel = null;

const ROW_HEIGHT = 44;
let collapsedGroups = new Set();
let flattenedVisualRows = [];

// =======================================================================================
// Logging & UI Behavior
// =======================================================================================
function log(message, level = "info") {
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour12: false
    });
    const logItem = document.createElement('div');
    logItem.className = `log-line ${level}`;
    let color = "#949ba4";
    if (level === "error")
        color = "#da373c";
    if (level === "warning")
        color = "#f0b132";
    if (level === "success")
        color = "#23a55a";
    logItem.innerHTML = `<span style="color: #7c808a;">[${timestamp}]</span> <span style="color: ${color};">${message}</span>`;
    logConsole.appendChild(logItem);
    logConsole.scrollTop = logConsole.scrollHeight;
}

// Collapsing pane splitter behaviors with native collapse features
function initSplitters() {
    let isDraggingLeft = false,
    isDraggingRight = false;

    leftSplitter.addEventListener('mousedown', () => {
        isDraggingLeft = true;
        leftSplitter.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });

    rightSplitter.addEventListener('mousedown', () => {
        isDraggingRight = true;
        rightSplitter.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', (e) => {
        if (isDraggingLeft) {
            if (e.clientX < 80) {
                sidebarPanel.style.width = '0px';
                sidebarPanel.style.minWidth = '0px';
                sidebarPanel.style.display = 'none';
            } else {
                sidebarPanel.style.display = 'flex';
                sidebarPanel.style.minWidth = '280px';
                sidebarPanel.style.width = `${Math.max(280, Math.min(450, e.clientX))}px`;
            }
        }
        if (isDraggingRight) {
            const widthFromRight = window.innerWidth - e.clientX;
            if (widthFromRight < 80) {
                viewerPanel.style.width = '0px';
                viewerPanel.style.minWidth = '0px';
                viewerPanel.style.display = 'none';
            } else {
                viewerPanel.style.display = 'flex';
                viewerPanel.style.minWidth = '350px';
                viewerPanel.style.width = `${Math.max(350, Math.min(650, widthFromRight))}px`;
            }
        }
    });

    window.addEventListener('mouseup', () => {
        isDraggingLeft = isDraggingRight = false;
        leftSplitter.classList.remove('dragging');
        rightSplitter.classList.remove('dragging');
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto';
    });
}

// =======================================================================================
// Initialization
// =======================================================================================
async function initializeModels() {
    log("Initializing application...");
    statusText.textContent = "Verifying neural network models...";
    const unlisten = await listen('download-progress', (event) => {
        const { file_name, percentage } = event.payload;
        progressBar.style.width = `${percentage}%`;
        statusText.textContent = `Downloading ${file_name}: ${percentage.toFixed(1)}%`;
    });
    try {
        await invoke('download_models');
        statusText.textContent = "Initialization complete!";
        progressBar.style.width = "100%";
        log("ONNX Loaded. System ready.", "info");
        setTimeout(() => {
            loaderView.style.display = 'none';
            mainAppLayout.style.display = 'flex';
            initSplitters();
        }, 800);
    } catch (error) {
        log(`Failed to initiate models: ${error}`, "error");
    } finally {
        unlisten();
    }

    await listen('tauri://drag-drop', (event) => {
        const paths = event.payload.paths;
        if (paths && paths.length > 0) {
            selectedSamplePath = paths[0];
            const fileName = selectedSamplePath.split(/[\\/]/).pop();
            samplePathLabel.style.display = 'block';
            samplePathLabel.textContent = `Sample: ${fileName}`;
            document.querySelector('input[value="ai"]').checked = true;
            updateHashingSectionVisibility();
            log(`Reference image loaded: ${fileName}`);
        }
    });
}

function updateHashingSectionVisibility() {
    const selectedMethod = document.querySelector('input[name="searchMethod"]:checked').value;
    if (selectedMethod === 'simple') {
        advancedHashingSection.style.display = 'block';
    } else {
        advancedHashingSection.style.display = 'none';
    }
}

searchMethods.forEach(el => {
    el.addEventListener('change', updateHashingSectionVisibility);
});

qcModeCheck.addEventListener('change', () => {
    qcOptionsContainer.style.opacity = qcModeCheck.checked ? '1.0' : '0.5';
    qcOptionsContainer.style.pointerEvents = qcModeCheck.checked ? 'auto' : 'none';
    folderBGroup.style.display = qcModeCheck.checked ? 'block' : 'none';
});

qcNormalMapsCheck.addEventListener('change', () => {
    qcNormalsTags.disabled = !qcNormalMapsCheck.checked;
});

browseBtnA.addEventListener('click', () => {
    dirInputA.value = "C:\\Users\\Zed\\Pictures\\Screenshots";
    log("Target Directory Folder A configured.");
});

browseBtnB.addEventListener('click', () => {
    dirInputB.value = "C:\\Users\\Zed\\Pictures\\BuildScreenshots";
    log("Comparison Directory Folder B configured.");
});

// =======================================================================================
// Scan Trigger & Rust Invocation
// =======================================================================================
scanBtn.addEventListener('click', async() => {
    const dirA = dirInputA.value.trim();
    if (!dirA) {
        alert("Please specify Target Folder A!");
        return;
    }

    scanBtn.disabled = true;
    scanBtn.textContent = "Processing Assets...";
    resultsPlaceholderText.style.display = "none";
    checkedFiles = [];
    updateActionButtonsState();

    log(`Starting scan on: ${dirA}`);
    const startTime = performance.now();

    try {
        let backendCommand = '';
        let params = {};
        const selectedMethod = document.querySelector('input[name="searchMethod"]:checked').value;

        if (qcModeCheck.checked) {
            const dirB = dirInputB.value.trim();
            if (dirB) {
                // Run comparative Folder Comparison (Relative QC)
                backendCommand = 'run_folder_compare';
                params = {
                    directoryA: dirA,
                    directoryB: dirB,
                    checkSizeBloat: true,
                    checkAlpha: true,
                    checkColorSpace: true,
                    checkCompression: true,
                    matchByStem: true
                };
            } else {
                // Run absolute Single Folder QC Scan
                backendCommand = 'run_qc_scan';
                params = {
                    directory: dirA,
                    checkNpot: qcNpotCheck.checked,
                    checkMipmaps: qcMipmapsCheck.checked,
                    checkBlockAlign: qcBlockAlignCheck.checked,
                    checkBitDepth: qcBitDepthCheck.checked,
                    validateNormals: qcNormalMapsCheck.checked,
                    normalsTags: qcNormalsTags.value.trim()
                };
            }
        } else if (selectedMethod === 'ai') {
            if (selectedSamplePath) {
                backendCommand = 'run_image_search';
                params = {
                    directory: dirA,
                    referenceImage: selectedSamplePath
                };
            } else if (semanticQuery.value.trim().length > 0) {
                backendCommand = 'run_ai_search';
                params = {
                    directory: dirA,
                    query: semanticQuery.value.trim()
                };
            } else {
                backendCommand = 'run_ai_duplicate_scan';
                params = {
                    directory: dirA,
                    threshold: parseFloat(similarityInput.value)
                };
            }
        } else if (selectedMethod === 'simple') {
            backendCommand = 'run_perceptual_scan';
            params = {
                directory: dirA,
                threshold: parseInt(similarityInput.value),
                analysisType: perceptualChannelSelect.value,
                ignoreSolidChannels: perceptualIgnoreSolid.checked,
                usePhash: perceptualUsePhash.checked
            };
        } else {
            backendCommand = 'run_exact_scan';
            params = {
                directory: dirA
            };
        }

        const results = await invoke(backendCommand, params);

        if (backendCommand === 'run_exact_scan' || backendCommand === 'run_ai_duplicate_scan') {
            currentScanResults = results;
            renderDuplicateResults(results);
        } else if (backendCommand === 'run_perceptual_scan') {
            renderPerceptualResults(results);
        } else if (backendCommand === 'run_qc_scan' || backendCommand === 'run_folder_compare') {
            renderQcResults(results);
        } else if (backendCommand === 'run_ai_search' || backendCommand === 'run_image_search') {
            renderAiResults(results);
        }

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        log(`Finished in ${elapsed}s`, "success");

    } catch (error) {
        resultsViewport.innerHTML = `<div style="text-align: center; color: var(--danger-red); margin-top: 100px;">Error: ${error}</div>`;
        log(`Scan failed: ${error}`, "error");
    } finally {
        scanBtn.disabled = false;
        scanBtn.textContent = "Start Scan";
    }
});

// =======================================================================================
// Virtual Scroll
// =======================================================================================
function setupVirtualContainer() {
    resultsViewport.innerHTML = `
        <div class="results-columns-header" id="list-columns-header">
          <span class="col-file">File</span><span class="col-score">Score</span><span class="col-path">Path</span><span class="col-meta">Metadata</span>
        </div>
        <div id="virtual-runway" style="position: absolute; top: 25px; left: 0; right: 0; height: 0px; pointer-events: none;"></div>
        <div id="virtual-content" style="position: absolute; top: 25px; left: 0; right: 0; will-change: transform;"></div>
    `;
}

function renderVirtualRows() {
    if (isGridMode)
        return;
    const runway = document.getElementById('virtual-runway');
    const content = document.getElementById('virtual-content');
    if (!runway || !content)
        return;

    runway.style.height = `${flattenedVisualRows.length * ROW_HEIGHT}px`;
    const scrollTop = resultsViewport.scrollTop;
    const viewportHeight = resultsViewport.clientHeight - 25;

    let startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 5);
    let endIndex = Math.min(flattenedVisualRows.length, Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT) + 5);

    content.style.transform = `translateY(${startIndex * ROW_HEIGHT}px)`;
    content.innerHTML = '';
    for (let i = startIndex; i < endIndex; i++) {
        content.appendChild(createRowElement(flattenedVisualRows[i]));
    }
}

function size_formatter(bytes) {
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

resultsViewport.addEventListener('scroll', () => {
    if (!isGridMode)
        renderVirtualRows();
});

function createRowElement(row) {
    if (row.type === 'group-header') {
        const d = document.createElement('div');
        d.className = 'group-card-header-row';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.justifyContent = 'space-between';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.setAttribute('data-group-idx', row.groupIndex);
        d.setAttribute('data-row-type', 'header');
        d.innerHTML = `<span>Group #${row.groupIndex + 1} (Hash: ${row.group.hash.substring(0, 8)}) ${row.collapsed ? '▸' : '▾'}</span><span>${size_formatter(row.group.files[0].size)}</span>`;
        return d;
    }
    if (row.type === 'file-row') {
        const d = document.createElement('div');
        d.className = `duplicate-item ${row.isBest ? 'best-match' : ''}`;
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.setAttribute('data-path', row.file.path);
        d.setAttribute('data-group-idx', row.groupIndex);
        d.setAttribute('data-row-type', 'file');
        const isChecked = checkedFiles.includes(row.file.path);
        const fileName = row.file.path.split(/[\\/]/).pop();
        d.innerHTML = `
          <span class="col-file" style="display:flex; align-items:center; gap:8px;">
            <input type="checkbox" class="file-check" data-path="${row.file.path}" ${row.isBest ? 'disabled' : ''} ${isChecked ? 'checked' : ''} style="margin:0;" />
            <div class="thumb-wrapper">
              <img class="row-thumb" src="${convertFileSrc(row.file.path)}" />
              <div class="corner-zone tl" data-channel="R" data-path="${row.file.path}"></div>
              <div class="corner-zone tr" data-channel="G" data-path="${row.file.path}"></div>
              <div class="corner-zone bl" data-channel="B" data-path="${row.file.path}"></div>
              <div class="corner-zone br" data-channel="A" data-path="${row.file.path}"></div>
            </div>
            <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${fileName}">${fileName}</span>
          </span>
          <span class="col-score" style="color: ${row.isBest ? 'var(--warning-yellow)' : 'var(--text-primary)'};">${row.isBest ? '[Best]' : 'Dup'}</span>
          <span class="col-path" title="${row.file.path}">${row.file.path}</span>
          <span class="col-meta">${row.file.width}x${row.file.height} • ${size_formatter(row.file.size)}</span>
        `;
        return d;
    }
    if (row.type === 'simple-row') {
        const d = document.createElement('div');
        d.className = 'duplicate-item';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.setAttribute('data-path', row.path);
        d.setAttribute('data-row-type', 'simple-file');
        d.innerHTML = `
          <span class="col-file" style="width:100%; display:flex; align-items:center; gap:8px;">
            <div class="thumb-wrapper">
              <img class="row-thumb" src="${convertFileSrc(row.path)}" />
              <div class="corner-zone tl" data-channel="R" data-path="${row.path}"></div>
              <div class="corner-zone tr" data-channel="G" data-path="${row.path}"></div>
              <div class="corner-zone bl" data-channel="B" data-path="${row.path}"></div>
              <div class="corner-zone br" data-channel="A" data-path="${row.path}"></div>
            </div>
            <span>${row.path.split(/[\\/]/).pop()}</span>
          </span>
        `;
        return d;
    }
    if (row.type === 'perceptual-header') {
        const d = document.createElement('div');
        d.className = 'group-card-header-row';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.style.background = 'var(--dark)';
        d.innerHTML = `<span>Visual Similarity Group #${row.group.group_id}</span>`;
        return d;
    }
    if (row.type === 'perceptual-row') {
        const d = document.createElement('div');
        d.className = 'duplicate-item';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.setAttribute('data-path', row.path);
        d.setAttribute('data-row-type', 'simple-file');
        d.innerHTML = `
          <span class="col-file" style="width:100%; display:flex; align-items:center; gap:8px;">
            <div class="thumb-wrapper">
              <img class="row-thumb" src="${convertFileSrc(row.path)}" />
              <div class="corner-zone tl" data-channel="R" data-path="${row.path}"></div>
              <div class="corner-zone tr" data-channel="G" data-path="${row.path}"></div>
              <div class="corner-zone bl" data-channel="B" data-path="${row.path}"></div>
              <div class="corner-zone br" data-channel="A" data-path="${row.path}"></div>
            </div>
            <span>${row.path.split(/[\\/]/).pop()}</span>
          </span>
        `;
        return d;
    }
    if (row.type === 'ai-row') {
        const d = document.createElement('div');
        d.className = 'duplicate-item';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.setAttribute('data-path', row.match.path);
        d.setAttribute('data-row-type', 'simple-file');
        d.innerHTML = `
          <span class="col-file" style="width:50%; display:flex; align-items:center; gap:8px;">
            <div class="thumb-wrapper">
              <img class="row-thumb" src="${convertFileSrc(row.match.path)}" />
              <div class="corner-zone tl" data-channel="R" data-path="${row.match.path}"></div>
              <div class="corner-zone tr" data-channel="G" data-path="${row.match.path}"></div>
              <div class="corner-zone bl" data-channel="B" data-path="${row.match.path}"></div>
              <div class="corner-zone br" data-channel="A" data-path="${row.match.path}"></div>
            </div>
            <span>${row.match.path.split(/[\\/]/).pop()}</span>
          </span>
          <span class="col-score" style="color: var(--success-green); font-weight:bold; width:50%; text-align:right;">${row.match.similarity.toFixed(1)}%</span>
        `;
        return d;
    }
    if (row.type === 'qc-row') {
        const d = document.createElement('div');
        d.className = 'duplicate-item';
        d.style.height = `${ROW_HEIGHT}px`;
        d.style.display = 'flex';
        d.style.alignItems = 'center';
        d.style.padding = '0 10px';
        d.style.borderBottom = '1px solid var(--border-gray)';
        d.style.borderLeft = '3px solid var(--danger-red)';
        d.setAttribute('data-path', row.issue.path);
        d.setAttribute('data-row-type', 'simple-file');
        d.innerHTML = `
          <span class="col-file" style="width:50%; display:flex; align-items:center; gap:8px;">
            <div class="thumb-wrapper">
              <img class="row-thumb" src="${convertFileSrc(row.issue.path)}" />
              <div class="corner-zone tl" data-channel="R" data-path="${row.issue.path}"></div>
              <div class="corner-zone tr" data-channel="G" data-path="${row.issue.path}"></div>
              <div class="corner-zone bl" data-channel="B" data-path="${row.issue.path}"></div>
              <div class="corner-zone br" data-channel="A" data-path="${row.issue.path}"></div>
            </div>
            <span>${row.issue.path.split(/[\\/]/).pop()}</span>
          </span>
          <span class="col-meta" style="color: var(--danger-red); font-weight:bold; width:50%; text-align:right;">${row.issue.issue} ${row.issue.details ? `(${row.issue.details})` : ''}</span>
        `;
        return d;
    }
    return null;
}

function buildFlattenedDuplicateRows() {
    flattenedVisualRows = [];
    currentScanResults.forEach((group, gIdx) => {
        flattenedVisualRows.push({
            type: 'group-header',
            groupIndex: gIdx,
            group: group,
            collapsed: collapsedGroups.has(gIdx)
        });
        if (!collapsedGroups.has(gIdx)) {
            group.files.forEach((file, fIdx) => {
                flattenedVisualRows.push({
                    type: 'file-row',
                    groupIndex: gIdx,
                    fileIndex: fIdx,
                    file: file,
                    isBest: fIdx === 0
                });
            });
        }
    });
}

function renderDuplicateResults(groups) {
    const totalFiles = groups.reduce((acc, g) => acc + g.files.length - 1, 0);
    resultsCountTitle.textContent = `Results (${groups.length} Groups, ~${totalFiles} duplicates)`;
    resultsViewport.innerHTML = "";
    collapsedGroups.clear();

    if (groups.length === 0) {
        resultsViewport.innerHTML = `<div style="text-align: center; color: var(--text-secondary); margin-top: 100px;">No duplicates found.</div>`;
        return;
    }

    if (isGridMode) {
        resultsViewport.classList.add('grid-mode');
        resultsSliderGroup.style.display = 'flex'; // Ensure size slider is visible on grid mode
        groups.forEach((group, index) => {
            const card = document.createElement('div');
            card.className = 'group-card';
            const sizeMb = (group.files[0].size / (1024 * 1024)).toFixed(2);
            card.innerHTML = `
                <div class="thumb-wrapper" style="width: var(--results-grid-size); height: var(--results-grid-size);">
                  <img class="grid-cover" src="${convertFileSrc(group.files[0].path)}" data-group-idx="${index}" data-path="${group.files[0].path}" />
                  <div class="corner-zone tl" data-channel="R" data-path="${group.files[0].path}"></div>
                  <div class="corner-zone tr" data-channel="G" data-path="${group.files[0].path}"></div>
                  <div class="corner-zone bl" data-channel="B" data-path="${group.files[0].path}"></div>
                  <div class="corner-zone br" data-channel="A" data-path="${group.files[0].path}"></div>
                </div>
                <div class="group-card-header"><span>Group #${index + 1}</span><span style="color: var(--text-secondary);">${sizeMb} MB</span></div>
            `;
            resultsViewport.appendChild(card);
        });
    } else {
        resultsViewport.classList.remove('grid-mode');
        resultsSliderGroup.style.display = 'none'; // Hide size slider on list mode
        setupVirtualContainer();
        buildFlattenedDuplicateRows();
        renderVirtualRows();
    }
}

function renderPerceptualResults(groups) {
    resultsCountTitle.textContent = `Results (${groups.length} visually similar groups)`;
    resultsViewport.classList.remove('grid-mode');
    resultsSliderGroup.style.display = 'none';
    setupVirtualContainer();
    if (groups.length === 0) {
        resultsViewport.innerHTML = `<div style="text-align: center; color: var(--text-secondary); margin-top: 100px;">No visually similar groups found.</div>`;
        return;
    }
    flattenedVisualRows = [];
    groups.forEach((group) => {
        flattenedVisualRows.push({
            type: 'perceptual-header',
            group
        });
        group.files.forEach(p => {
            flattenedVisualRows.push({
                type: 'perceptual-row',
                path: p
            });
        });
    });
    renderVirtualRows();
}

function renderAiResults(matches) {
    resultsCountTitle.textContent = `Semantic Matches (${matches.length} matches)`;
    resultsViewport.classList.remove('grid-mode');
    resultsSliderGroup.style.display = 'none';
    setupVirtualContainer();
    if (matches.length === 0) {
        resultsViewport.innerHTML = `<div style="text-align: center; color: var(--text-secondary); margin-top: 100px;">No semantic matches found.</div>`;
        return;
    }
    flattenedVisualRows = [];
    matches.forEach((match, index) => {
        flattenedVisualRows.push({
            type: 'ai-row',
            match,
            index
        });
    });
    renderVirtualRows();
}

function renderQcResults(issues) {
    resultsCountTitle.textContent = `Results (${issues.length} technical issues detected)`;
    resultsViewport.classList.remove('grid-mode');
    resultsSliderGroup.style.display = 'none';
    setupVirtualContainer();
    if (issues.length === 0) {
        resultsViewport.innerHTML = `<div style="text-align: center; color: var(--success-green); margin-top: 100px;">All assets are technically healthy!</div>`;
        return;
    }
    flattenedVisualRows = [];
    issues.forEach((issue) => {
        flattenedVisualRows.push({
            type: 'qc-row',
            issue
        });
    });
    renderVirtualRows();
}

// Tree view controls
expandAllBtn.addEventListener('click', () => {
    collapsedGroups.clear();
    buildFlattenedDuplicateRows();
    renderVirtualRows();
});
collapseAllBtn.addEventListener('click', () => {
    currentScanResults.forEach((_, idx) => collapsedGroups.add(idx));
    buildFlattenedDuplicateRows();
    renderVirtualRows();
});
selectAllBtn.addEventListener('click', () => {
    checkedFiles = [];
    currentScanResults.forEach(group => group.files.forEach((file, fIdx) => {
            if (fIdx > 0)
                checkedFiles.push(file.path);
        }));
    renderVirtualRows();
    updateActionButtonsState();
});
deselectAllBtn.addEventListener('click', () => {
    checkedFiles = [];
    renderVirtualRows();
    updateActionButtonsState();
});
selectExceptBestBtn.addEventListener('click', () => {
    checkedFiles = [];
    currentScanResults.forEach(group => group.files.forEach((file, fIdx) => {
            if (fIdx > 0)
                checkedFiles.push(file.path);
        }));
    renderVirtualRows();
    updateActionButtonsState();
});
invertSelectionBtn.addEventListener('click', () => {
    let newSelection = [];
    currentScanResults.forEach(group => group.files.forEach((file, fIdx) => {
            if (fIdx > 0 && !checkedFiles.includes(file.path))
                newSelection.push(file.path);
        }));
    checkedFiles = newSelection;
    renderVirtualRows();
    updateActionButtonsState();
});

// =======================================================================================
// Right Panel: Image Viewer
// =======================================================================================
function convertFileSrc(filePath) {
    return window.__TAURI__.core.convertFileSrc(filePath);
}

function selectGroupIndex(groupIdx, targetPath = null) {
    selectedGroupIndex = groupIdx;
    const group = currentScanResults[groupIdx];
    if (!group)
        return;
    const activePath = targetPath || group.files[0].path;
    viewerListContainer.innerHTML = "";
    if (isViewerGridMode) {
        viewerListContainer.classList.add('grid-mode');
        viewerSizeSliderGroup.style.display = 'flex'; // Ensure size slider is visible on grid mode
    } else {
        viewerListContainer.classList.remove('grid-mode');
        viewerSizeSliderGroup.style.display = 'none'; // Hide size slider on list mode
    }
    group.files.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = `viewer-item ${file.path === activePath ? 'active' : ''}`;
        item.setAttribute('data-path', file.path);
        const label = index === 0 ? '[Best]' : 'Dup';

        const thumbHtml = `
          <div class="viewer-item-thumb-wrapper">
            <img class="viewer-item-thumb" src="${convertFileSrc(file.path)}" />
            <div class="corner-zone tl" data-channel="R" data-path="${file.path}"></div>
            <div class="corner-zone tr" data-channel="G" data-path="${file.path}"></div>
            <div class="corner-zone bl" data-channel="B" data-path="${file.path}"></div>
            <div class="corner-zone br" data-channel="A" data-path="${file.path}"></div>
          </div>
        `;

        if (isViewerGridMode)
            item.innerHTML = `${thumbHtml}<span class="viewer-item-title">${file.path.split(/[\\/]/).pop()}</span>`;
        else
            item.innerHTML = `${thumbHtml}<div class="viewer-item-details"><span class="viewer-item-title">${file.path.split(/[\\/]/).pop()}</span><span class="viewer-item-meta">${label} • ${file.width}x${file.height} • ${size_formatter(file.size)}</span></div>`;
        viewerListContainer.appendChild(item);
    });
    const candidates = group.files.filter(f => f.path !== group.files[0].path).map(f => f.path);
    if (candidates.length > 0) {
        activeOriginalPath = group.files[0].path;
        activeDuplicatePath = activePath !== activeOriginalPath ? activePath : candidates[0];
    }
    btnCompareTrigger.disabled = group.files.length < 2;
    btnCompareTrigger.textContent = `Compare (${group.files.length - 1})`;
}

btnViewerListMode.addEventListener('click', () => {
    isViewerGridMode = false;
    btnViewerListMode.classList.add('active-toggle');
    btnViewerGridMode.classList.remove('active-toggle');
    viewerListContainer.classList.remove('grid-mode');
    viewerSizeSliderGroup.style.display = 'none'; // Hide size slider on list mode
    if (selectedGroupIndex !== null)
        selectGroupIndex(selectedGroupIndex);
});
btnViewerGridMode.addEventListener('click', () => {
    isViewerGridMode = true;
    btnViewerGridMode.classList.add('active-toggle');
    btnViewerListMode.classList.remove('active-toggle');
    viewerListContainer.classList.add('grid-mode');
    viewerSizeSliderGroup.style.display = 'flex'; // Ensure size slider is visible on grid mode
    if (selectedGroupIndex !== null)
        selectGroupIndex(selectedGroupIndex);
});

// BG SLIDER logic
viewerBgAlphaSlider.addEventListener('input', () => {
    const val = viewerBgAlphaSlider.value;
    viewerBgSwatch.style.opacity = (val / 255.0).toString();
    bgOverlay.style.opacity = (val / 255.0).toString();
});

btnCompareTrigger.addEventListener('click', () => {
    if (!activeOriginalPath || !activeDuplicatePath)
        return;
    viewerListContainer.style.display = 'none';
    btnCompareTrigger.style.display = 'none';
    viewerCanvas.style.display = 'flex';
    viewerCompareControls.style.display = 'flex';
    updateComparisonDisplay();
});

btnBackToList.addEventListener('click', () => {
    viewerCanvas.style.display = 'none';
    viewerCompareControls.style.display = 'none';
    viewerListContainer.style.display = 'block';
    btnCompareTrigger.style.display = 'block';
});

async function updateComparisonDisplay() {
    const mode = compareModeSelect.value;
    const src1 = await resolveImageSrc(activeOriginalPath);
    const src2 = await resolveImageSrc(activeDuplicatePath);

    compareWipeViewport.style.display = 'none';
    viewerPlaceholder.style.display = 'none';

    // Clear out side-by-side elements but keep overlay and wipe containers
    Array.from(viewerCanvas.children).forEach(c => {
        if (c.id !== 'bg-overlay' && c.id !== 'compare-wipe-viewport' && c.id !== 'viewer-placeholder')
            c.remove();
    });

    if (mode === 'side-by-side') {
        wipeSliderGroup.style.display = 'none';
        const sbs = document.createElement('div');
        sbs.style.cssText = 'display:flex; width:100%; height:100%; gap:5px; padding:5px; z-index:2; position:relative;';
        sbs.innerHTML = `<div style="flex:1; display:flex; flex-direction:column; align-items:center;"><span style="font-size:7.5pt; color:var(--text-secondary);">ORIGINAL</span><img src="${src1}" style="max-width:100%; max-height:90%; object-fit:contain;"/></div><div style="flex:1; display:flex; flex-direction:column; align-items:center;"><span style="font-size:7.5pt; color:var(--text-secondary);">DUPLICATE</span><img src="${src2}" style="max-width:100%; max-height:90%; object-fit:contain;"/></div>`;
        viewerCanvas.appendChild(sbs);
    } else if (mode === 'wipe' || mode === 'overlay') {
        wipeSliderGroup.style.display = 'block';
        compareWipeViewport.style.display = 'block';

        wipeImgBg.src = src1;
        wipeImgFg.src = src2;

        if (mode === 'wipe') {
            wipeLayerFg.style.clipPath = `polygon(0 0, ${wipeSlider.value}% 0, ${wipeSlider.value}% 100%, 0 100%)`;
            wipeLayerFg.style.opacity = "1.0";
            wipeLine.style.display = 'block';
            wipeLine.style.left = `${wipeSlider.value}%`;
        } else {
            wipeLayerFg.style.clipPath = 'none';
            wipeLayerFg.style.opacity = (wipeSlider.value / 100.0).toString();
            wipeLine.style.display = 'none';
        }
    } else if (mode === 'heatmap') {
        wipeSliderGroup.style.display = 'none';
        viewerPlaceholder.style.display = 'block';
        viewerPlaceholder.textContent = "Evaluating difference map...";

        try {
            const diffMapPath = await invoke('calculate_diff', {
                file1: activeOriginalPath,
                file2: activeDuplicatePath
            });
            // Added cache busting query parameter ?t=timestamp so the browser updates the image properly
            viewerPlaceholder.innerHTML = `<img src="${convertFileSrc(diffMapPath)}?t=${Date.now()}" style="max-width:100%; max-height:100%; object-fit:contain; border-radius: 4px; position:relative; z-index:2;"/>`;
        } catch (e) {
            viewerPlaceholder.textContent = `Failed to generate diff: ${e}`;
        }
    }
}

async function resolveImageSrc(path) {
    if (activeChannel) {
        try {
            return await invoke('get_channel_preview', {
                path,
                channel: activeChannel
            });
        } catch (e) {
            console.error(e);
        }
    }
    return convertFileSrc(path);
}

wipeSlider.addEventListener('input', () => {
    const mode = compareModeSelect.value;
    if (mode === 'wipe') {
        wipeLayerFg.style.clipPath = `polygon(0 0, ${wipeSlider.value}% 0, ${wipeSlider.value}% 100%, 0 100%)`;
        wipeLine.style.left = `${wipeSlider.value}%`;
    } else if (mode === 'overlay') {
        wipeLayerFg.style.opacity = (wipeSlider.value / 100.0).toString();
    }
});

compareModeSelect.addEventListener('change', updateComparisonDisplay);

channelButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const channel = btn.getAttribute('data-channel');
        if (activeChannel === channel) {
            activeChannel = null;
            btn.style.border = '2px solid transparent';
        } else {
            activeChannel = channel;
            channelButtons.forEach(b => b.style.border = '2px solid transparent');
            btn.style.border = '2px solid white';
        }
        updateComparisonDisplay();
    });
});

resultsViewport.addEventListener('click', (event) => {
    const cornerZone = event.target.closest('.corner-zone');
    if (cornerZone) {
        const gridCover = cornerZone.parentElement.querySelector('.grid-cover');
        if (gridCover) {
            selectGroupIndex(parseInt(gridCover.getAttribute('data-group-idx')), gridCover.getAttribute('data-path'));
            return;
        }
        const fileRow = cornerZone.closest('[data-row-type="file"]');
        if (fileRow) {
            selectGroupIndex(parseInt(fileRow.getAttribute('data-group-idx')), fileRow.getAttribute('data-path'));
            return;
        }
        const simpleRow = cornerZone.closest('[data-row-type="simple-file"]');
        if (simpleRow) {
            const path = simpleRow.getAttribute('data-path');
            selectedGroupIndex = null;
            activeOriginalPath = path;
            activeDuplicatePath = path;
            viewerListContainer.style.display = 'none';
            btnCompareTrigger.style.display = 'none';
            viewerCanvas.style.display = 'flex';
            viewerCompareControls.style.display = 'flex';
            updateComparisonDisplay();
            return;
        }
    }
    const gridCover = event.target.closest('.grid-cover');
    if (gridCover) {
        selectGroupIndex(parseInt(gridCover.getAttribute('data-group-idx')), gridCover.getAttribute('data-path'));
        return;
    }
    const headerRow = event.target.closest('[data-row-type="header"]');
    if (headerRow) {
        const gIdx = parseInt(headerRow.getAttribute('data-group-idx'));
        if (collapsedGroups.has(gIdx))
            collapsedGroups.delete(gIdx);
        else
            collapsedGroups.add(gIdx);
        buildFlattenedDuplicateRows();
        renderVirtualRows();
        return;
    }
    const fileRow = event.target.closest('[data-row-type="file"]');
    if (fileRow && !event.target.classList.contains('file-check')) {
        selectGroupIndex(parseInt(fileRow.getAttribute('data-group-idx')), fileRow.getAttribute('data-path'));
        return;
    }
    const simpleRow = event.target.closest('[data-row-type="simple-file"]');
    if (simpleRow) {
        const path = simpleRow.getAttribute('data-path');
        selectedGroupIndex = null;
        activeOriginalPath = path;
        activeDuplicatePath = path;
        viewerListContainer.style.display = 'none';
        btnCompareTrigger.style.display = 'none';
        viewerCanvas.style.display = 'flex';
        viewerCompareControls.style.display = 'flex';
        updateComparisonDisplay();
    }
});

resultsViewport.addEventListener('change', (event) => {
    if (event.target.classList.contains('file-check')) {
        const path = event.target.getAttribute('data-path');
        if (event.target.checked) {
            if (!checkedFiles.includes(path))
                checkedFiles.push(path);
        } else {
            checkedFiles = checkedFiles.filter(p => p !== path);
        }
        updateActionButtonsState();
    }
});

viewerListContainer.addEventListener('click', (event) => {
    const item = event.target.closest('.viewer-item');
    if (item && selectedGroupIndex !== null)
        selectGroupIndex(selectedGroupIndex, item.getAttribute('data-path'));
});

function updateActionButtonsState() {
    const hasSelection = checkedFiles.length > 0;
    btnTrash.disabled = !hasSelection;
    btnReflink.disabled = !hasSelection;
    btnHardlink.disabled = !hasSelection;
    btnGenerateReport.disabled = currentScanResults.length === 0;

    if (hasSelection) {
        btnTrash.textContent = `Move to Trash (${checkedFiles.length})`;
        btnReflink.textContent = `Replace with Reflink (${checkedFiles.length})`;
        btnHardlink.textContent = `Replace with Hardlink (${checkedFiles.length})`;
    } else {
        btnTrash.textContent = "Move to Trash";
        btnReflink.textContent = "Replace with Reflink";
        btnHardlink.textContent = "Replace with Hardlink";
    }
}

btnListView.addEventListener('click', () => {
    isGridMode = false;
    btnListView.classList.add('active-toggle');
    btnGridView.classList.remove('active-toggle');
    resultsSliderGroup.style.display = 'none'; // Hide results size slider on list mode
    renderDuplicateResults(currentScanResults);
});
btnGridView.addEventListener('click', () => {
    isGridMode = true;
    btnGridView.classList.add('active-toggle');
    btnListView.classList.remove('active-toggle');
    resultsSliderGroup.style.display = 'flex'; // Show results size slider on grid mode
    renderDuplicateResults(currentScanResults);
});

// Action buttons hooks
btnTrash.addEventListener('click', async() => {
    if (checkedFiles.length === 0)
        return;
    const confirmMsg = `Are you sure you want to move ${checkedFiles.length} files to the Trash?`;
    if (!confirm(confirmMsg))
        return;

    try {
        await invoke('delete_files', {
            paths: checkedFiles
        });
        log(`Moved ${checkedFiles.length} duplicate files to system Trash bin.`, "success");
        checkedFiles = [];
        updateActionButtonsState();
        scanBtn.click();
    } catch (error) {
        log(`Deletions failed: ${error}`, "error");
    }
});

btnReflink.addEventListener('click', async() => {
    if (checkedFiles.length === 0)
        return;
    const confirmMsg = `Replace ${checkedFiles.length} duplicates with space-saving reflinks?`;
    if (!confirm(confirmMsg))
        return;

    const pairs = [];
    currentScanResults.forEach(group => {
        const bestFile = group.files[0].path;
        group.files.forEach((file, index) => {
            if (index > 0 && checkedFiles.includes(file.path)) {
                pairs.push([bestFile, file.path]);
            }
        });
    });

    try {
        await invoke('create_reflinks', {
            pairs
        });
        log(`Established ${pairs.length} CoW hardlinks on the filesystem. Storage optimized.`, "success");
        checkedFiles = [];
        updateActionButtonsState();
        scanBtn.click();
    } catch (error) {
        log(`Linking failed: ${error}`, "error");
    }
});

btnHardlink.addEventListener('click', async() => {
    if (checkedFiles.length === 0)
        return;
    const confirmMsg = `Replace ${checkedFiles.length} duplicates with hardlinks?`;
    if (!confirm(confirmMsg))
        return;

    const pairs = [];
    currentScanResults.forEach(group => {
        const bestFile = group.files[0].path;
        group.files.forEach((file, index) => {
            if (index > 0 && checkedFiles.includes(file.path)) {
                pairs.push([bestFile, file.path]);
            }
        });
    });

    try {
        await invoke('create_hardlinks', {
            pairs
        });
        log(`Established ${pairs.length} hardlinks on the filesystem. Storage optimized.`, "success");
        checkedFiles = [];
        updateActionButtonsState();
        scanBtn.click();
    } catch (error) {
        log(`Hardlinking failed: ${error}`, "error");
    }
});

btnGenerateReport.addEventListener('click', async() => {
    if (currentScanResults.length === 0)
        return;
    btnGenerateReport.disabled = true;
    btnGenerateReport.textContent = "Generating...";
    try {
        const reportPath = await invoke('generate_visualization_report', {
            groups: currentScanResults
        });
        log(`Reports generated in folder: ${reportPath}`, "success");
        alert(`Visual reports saved to: ${reportPath}`);
    } catch (e) {
        log(`Failed to generate report: ${e}`, "error");
    } finally {
        btnGenerateReport.textContent = "Generate Visual Report";
        btnGenerateReport.disabled = false;
    }
});

// Interactive dynamic mouse hover over the thumbnails' corner-zones
let hoverTimeout = null;
let activeHoverImg = null;
let activeHoverPath = null;

document.addEventListener('mouseover', (e) => {
    const zone = e.target.closest('.corner-zone');
    if (zone) {
        const path = zone.getAttribute('data-path');
        const channel = zone.getAttribute('data-channel');
        const img = zone.parentElement.querySelector('img');

        if (img && path) {
            if (hoverTimeout) {
                clearTimeout(hoverTimeout);
            }

            activeHoverImg = img;
            activeHoverPath = path;

            // Debouncer delay of 50ms protects backend from redundant IPC calls
            hoverTimeout = setTimeout(async() => {
                try {
                    const previewSrc = await invoke('get_channel_preview', {
                        path,
                        channel
                    });
                    if (activeHoverImg === img && activeHoverPath === path) {
                        img.src = previewSrc;
                    }
                } catch (err) {
                    console.error(err);
                }
            }, 50);
        }
    }
});

document.addEventListener('mouseout', (e) => {
    const zone = e.target.closest('.corner-zone');
    if (zone) {
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        const path = zone.getAttribute('data-path');
        const img = zone.parentElement.querySelector('img');
        if (img && path) {
            img.src = convertFileSrc(path);
        }
        activeHoverImg = null;
        activeHoverPath = null;
    }
});

// Smooth Wipe handle dragging
let isDraggingWipe = false;

compareWipeViewport.addEventListener('mousedown', (e) => {
    if (compareModeSelect.value === 'wipe') {
        isDraggingWipe = true;
        updateWipeFromMouse(e);
    }
});

window.addEventListener('mousemove', (e) => {
    if (isDraggingWipe && compareModeSelect.value === 'wipe') {
        updateWipeFromMouse(e);
    }
});

window.addEventListener('mouseup', () => {
    isDraggingWipe = false;
});

function updateWipeFromMouse(e) {
    const rect = compareWipeViewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    wipeSlider.value = Math.round(percentage);

    wipeLayerFg.style.clipPath = `polygon(0 0, ${wipeSlider.value}% 0, ${wipeSlider.value}% 100%, 0 100%)`;
    wipeLine.style.left = `${wipeSlider.value}%`;
}

// Custom Grid size scaling slider integrations
resultsGridSizeSlider.addEventListener('input', () => {
    const size = resultsGridSizeSlider.value;
    document.documentElement.style.setProperty('--results-grid-size', `${size}px`);
});

viewerGridSizeSlider.addEventListener('input', () => {
    const size = viewerGridSizeSlider.value;
    document.documentElement.style.setProperty('--viewer-grid-size', `${size}px`);
});

// Sync initial slider view displays on startup
resultsSliderGroup.style.display = 'none';
viewerSizeSliderGroup.style.display = 'none';

initializeModels();
