import { createSignal, createEffect, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";

const asyncThumbCache = new Map();

export default function ThumbImage(props) {
  const [src, setSrc] = createSignal("");

  createEffect(() => {
    const path = props.path;
    if (!path) {
      setSrc("");
      return;
    }
    const ext = path.split(".").pop().toLowerCase();
    const unsupported = ["hdr", "dds", "exr", "tga", "tif", "tiff"];

    if (unsupported.includes(ext)) {
      if (asyncThumbCache.has(path)) {
        setSrc(asyncThumbCache.get(path));
      } else {
        const placeholder = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="transparent"/></svg>`;
        setSrc(placeholder);
        asyncThumbCache.set(path, placeholder);

        invoke("get_channel_preview", { path, channel: "Composite" })
          .then((base64) => {
            asyncThumbCache.set(path, base64);
            setSrc(base64);
          })
          .catch((e) => {
            console.error("Failed to load generic thumb", path, e);
          });
      }
    } else {
      setSrc(convertFileSrc(path));
    }
  });

  return (
    <div
      class={props.wrapperClass || "thumb-wrapper"}
      style={props.wrapperStyle}
    >
      <img
        class={props.imgClass || "row-thumb"}
        src={src()}
        data-path={props.path}
        onClick={props.onClick}
      />
      <Show when={props.showCorners}>
        <div
          class="corner-zone tl"
          data-channel="R"
          data-path={props.path}
          onClick={(e) => {
            e.stopPropagation();
            if (props.onCornerClick) props.onCornerClick("R", props.path);
          }}
        ></div>
        <div
          class="corner-zone tr"
          data-channel="G"
          data-path={props.path}
          onClick={(e) => {
            e.stopPropagation();
            if (props.onCornerClick) props.onCornerClick("G", props.path);
          }}
        ></div>
        <div
          class="corner-zone bl"
          data-channel="B"
          data-path={props.path}
          onClick={(e) => {
            e.stopPropagation();
            if (props.onCornerClick) props.onCornerClick("B", props.path);
          }}
        ></div>
        <div
          class="corner-zone br"
          data-channel="A"
          data-path={props.path}
          onClick={(e) => {
            e.stopPropagation();
            if (props.onCornerClick) props.onCornerClick("A", props.path);
          }}
        ></div>
      </Show>
    </div>
  );
}
