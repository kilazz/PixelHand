import { For, Show, createSignal, createMemo, onMount } from "solid-js";
import ThumbImage from "./ThumbImage";

function size_formatter(bytes) {
  if (!bytes) return "0 MB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function renderRichMeta(file) {
  const parts = [];
  parts.push(`${file.width}x${file.height}`);

  const format = file.compression_format || file.format_str;
  if (format && format.toLowerCase() !== "unknown") {
    parts.push(format);
  }

  if (file.bit_depth) {
    parts.push(`${file.bit_depth}-bit`);
  }

  if (file.color_space && file.color_space.toLowerCase() !== "unknown") {
    parts.push(file.color_space);
  }

  if (file.mipmap_count && file.mipmap_count > 1) {
    parts.push(`Mips: ${file.mipmap_count}`);
  }

  if (file.has_alpha) {
    parts.push("Alpha");
  }

  return (
    <div style="display: flex; flex-direction: column; justify-content: center; line-height: 1.3; overflow: hidden; height: 100%;">
      <span
        style="font-size: 8pt; font-weight: 500; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
        title={parts.join(" • ")}
      >
        {parts.join(" • ")}
      </span>
      <span style="font-size: 7.5pt; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
        {size_formatter(file.size)}
      </span>
    </div>
  );
}

export default function ResultsPanel(props) {
  const [collapsedGroups, setCollapsedGroups] = createSignal(new Set());

  const flattenedRows = createMemo(() => {
    let rows = [];
    const results = props.results;
    if (!results || results.length === 0) return rows;

    const collapsed = collapsedGroups();

    results.forEach((group, gIdx) => {
      if (group.type === "duplicate") {
        rows.push({ type: "header", groupIndex: gIdx, group: group });
        if (!collapsed.has(gIdx)) {
          group.files.forEach((file, fIdx) => {
            rows.push({
              type: "file",
              file,
              groupIndex: gIdx,
              isBest: fIdx === 0,
            });
          });
        }
      } else if (group.type === "qc") {
        rows.push({ type: "qc", issue: group.issue });
      } else if (group.type === "ai") {
        rows.push({ type: "ai", match: group.match });
      } else if (group.type === "perceptual") {
        rows.push({
          type: "header-perceptual",
          groupIndex: gIdx,
          group: group,
        });
        if (!collapsed.has(gIdx)) {
          group.files.forEach((p) => {
            rows.push({ type: "perceptual", path: p });
          });
        }
      }
    });
    return rows;
  });

  const RowItem = ({ row }) => {
    if (row.type === "header") {
      const isCollapsed = collapsedGroups().has(row.groupIndex);
      return (
        <div
          class="group-card-header-row"
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            "justify-content": "space-between",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
          }}
          onClick={(e) => {
            const curr = new Set(collapsedGroups());
            if (curr.has(row.groupIndex)) curr.delete(row.groupIndex);
            else curr.add(row.groupIndex);
            setCollapsedGroups(curr);
          }}
        >
          <span>
            {isCollapsed ? "▸" : "▾"} Group #{row.groupIndex + 1} (Hash:{" "}
            {row.group.hash?.substring(0, 8)})
          </span>
          <div style="display: flex; gap: 10px; align-items: center;">
            <button
              onClick={(e) => {
                e.stopPropagation();
                props.onSelectGroup(row.groupIndex, row.group.files[0].path);
              }}
              style="padding: 2px 6px; font-size: 7.5pt; height: 20px; background: var(--brand-blue);"
            >
              Compare
            </button>
            <span>{size_formatter(row.group.files[0].size)}</span>
          </div>
        </div>
      );
    }
    if (row.type === "file") {
      const isChecked = props.checkedFiles.includes(row.file.path);
      const fileName = row.file.path.split(/[\\/]/).pop();
      return (
        <div
          class={`duplicate-item ${row.isBest ? "best-match" : ""}`}
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
            cursor: "pointer",
          }}
          onClick={(e) => {
            if (e.target.tagName !== "INPUT")
              props.onSelectGroup(row.groupIndex, row.file.path);
          }}
        >
          <span
            class="col-file"
            style={{ display: "flex", "align-items": "center", gap: "8px" }}
          >
            <input
              type="checkbox"
              class="file-check"
              checked={isChecked}
              disabled={row.isBest}
              onChange={(e) => {
                const curr = [...props.checkedFiles];
                if (e.target.checked) curr.push(row.file.path);
                else curr.splice(curr.indexOf(row.file.path), 1);
                props.setCheckedFiles(curr);
              }}
              style={{ margin: 0 }}
            />
            <ThumbImage path={row.file.path} showCorners={true} />
            <span
              style={{
                overflow: "hidden",
                "text-overflow": "ellipsis",
                "white-space": "nowrap",
              }}
              title={fileName}
            >
              {fileName}
            </span>
          </span>
          <span
            class="col-score"
            style={{
              color: row.isBest
                ? "var(--warning-yellow)"
                : "var(--text-primary)",
            }}
          >
            {row.isBest
              ? "[Best]"
              : row.file.similarity !== undefined && row.file.similarity < 100
              ? `${row.file.similarity.toFixed(1)}%`
              : "Dup"}
          </span>
          <span class="col-path" title={row.file.path}>
            {row.file.path}
          </span>
          <span
            class="col-meta"
            style="display: flex; flex-direction: column; justify-content: center; height: 100%;"
          >
            {renderRichMeta(row.file)}
          </span>
        </div>
      );
    }
    if (row.type === "qc") {
      const fileName = row.issue.path.split(/[\\/]/).pop();
      return (
        <div
          class="duplicate-item"
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
            "border-left": "3px solid var(--danger-red)",
          }}
        >
          <span
            class="col-file"
            style="width:50%; display:flex; align-items:center; gap:8px;"
          >
            <ThumbImage path={row.issue.path} showCorners={true} />
            <span>{fileName}</span>
          </span>
          <span
            class="col-meta"
            style="color: var(--danger-red); font-weight:bold; width:50%; text-align:right;"
          >
            {row.issue.issue}{" "}
            {row.issue.details ? `(${row.issue.details})` : ""}
          </span>
        </div>
      );
    }
    if (row.type === "header-perceptual") {
      const isCollapsed = collapsedGroups().has(row.groupIndex);
      return (
        <div
          class="group-card-header-row"
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
            background: "var(--dark)",
          }}
          onClick={(e) => {
            const curr = new Set(collapsedGroups());
            if (curr.has(row.groupIndex)) curr.delete(row.groupIndex);
            else curr.add(row.groupIndex);
            setCollapsedGroups(curr);
          }}
        >
          <span>
            {isCollapsed ? "▸" : "▾"} Visual Similarity Group #
            {row.group.group_id}
          </span>
        </div>
      );
    }
    if (row.type === "perceptual") {
      return (
        <div
          class="duplicate-item"
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
            cursor: "pointer",
          }}
        >
          <span
            class="col-file"
            style="width:100%; display:flex; align-items:center; gap:8px;"
          >
            <ThumbImage path={row.path} showCorners={true} />
            <span>{row.path.split(/[\\/]/).pop()}</span>
          </span>
        </div>
      );
    }
    if (row.type === "ai") {
      return (
        <div
          class="duplicate-item"
          style={{
            height: "44px",
            display: "flex",
            "align-items": "center",
            padding: "0 10px",
            "border-bottom": "1px solid var(--border-gray)",
            cursor: "pointer",
          }}
        >
          <span
            class="col-file"
            style="width:50%; display:flex; align-items:center; gap:8px;"
          >
            <ThumbImage path={row.match.path} showCorners={true} />
            <span>{row.match.path.split(/[\\/]/).pop()}</span>
          </span>
          <span
            class="col-score"
            style="color: var(--success-green); font-weight:bold; width:50%; text-align:right;"
          >
            {(row.match.similarity || 0).toFixed(1)}%
          </span>
        </div>
      );
    }
    return <div>?</div>;
  };

  return (
    <div class="results-panel">
      <div class="results-header">
        <div class="section-title" style="margin: 0;">
          Results ({props.results.length} items)
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <Show when={props.isGridMode}>
            <div style="display: flex; align-items: center; gap: 6px;">
              <label style="color: var(--text-secondary); font-size: 8pt;">
                Size:
              </label>
              <input
                type="range"
                min="80"
                max="200"
                value={props.gridSize}
                onInput={(e) => props.setGridSize(e.target.value)}
                style="width: 70px;"
              />
            </div>
          </Show>
          <div class="view-toggles">
            <button
              class={!props.isGridMode ? "active-toggle" : ""}
              onClick={() => props.setIsGridMode(false)}
            >
              ☰
            </button>
            <button
              class={props.isGridMode ? "active-toggle" : ""}
              onClick={() => props.setIsGridMode(true)}
            >
              ⊞
            </button>
          </div>
          <button onClick={() => setCollapsedGroups(new Set())}>
            Expand All
          </button>
          <button
            onClick={() => {
              const allIndices = props.results
                .map((g, i) =>
                  g.type === "duplicate" || g.type === "perceptual" ? i : -1
                )
                .filter((i) => i !== -1);
              setCollapsedGroups(new Set(allIndices));
            }}
          >
            Collapse All
          </button>
        </div>
      </div>

      <div
        class={`results-list ${props.isGridMode ? "grid-mode" : ""}`}
        style={
          props.isGridMode
            ? { "--results-grid-size": `${props.gridSize}px` }
            : {}
        }
      >
        <Show when={!props.isGridMode}>
          <div class="results-columns-header">
            <span class="col-file">File</span>
            <span class="col-score">Score</span>
            <span class="col-path">Path</span>
            <span class="col-meta">Metadata</span>
          </div>

          <Show
            when={props.results.length === 0}
            fallback={
              <div style={{ "overflow-y": "auto", flex: 1, "min-height": 0 }}>
                <For each={flattenedRows()}>
                  {(row) => <RowItem row={row} />}
                </For>
              </div>
            }
          >
            <div style="text-align: center; color: #949ba4; margin-top: 100px;">
              Enter directory and click scan to explore assets.
            </div>
          </Show>
        </Show>

        <Show when={props.isGridMode && props.results.length > 0}>
          <For each={props.results}>
            {(group, index) => {
              if (group.type !== "duplicate") return null;
              const sizeMb = (group.files[0].size / (1024 * 1024)).toFixed(2);
              return (
                <div class="group-card">
                  <ThumbImage
                    path={group.files[0].path}
                    wrapperStyle={{
                      width: "var(--results-grid-size)",
                      height: "var(--results-grid-size)",
                    }}
                    imgClass="grid-cover"
                    showCorners={true}
                    onClick={() => props.onSelectGroup(index)}
                  />
                  <div class="group-card-header">
                    <span>Group #{index() + 1}</span>
                    <br />
                    <span style="color: var(--text-secondary);">
                      {sizeMb} MB
                    </span>
                  </div>
                </div>
              );
            }}
          </For>
        </Show>
      </div>

      <div class="results-selection-box">
        <label style="font-weight: bold; color: var(--text-secondary); font-size:7.5pt; text-transform: uppercase;">
          Selection
        </label>
        <div class="selection-buttons-grid">
          <button>Select All</button>
          <button onClick={() => props.setCheckedFiles([])}>
            Deselect All
          </button>
          <button>Select All Except Best</button>
          <button>Invert Selection</button>
        </div>
      </div>
      <div class="results-footer-actions">
        <button class="primary-btn" disabled>
          Replace with Hardlink
        </button>
        <button class="primary-btn" disabled>
          Replace with Reflink
        </button>
        <button class="danger-btn" disabled>
          Move to Trash
        </button>
      </div>
    </div>
  );
}
