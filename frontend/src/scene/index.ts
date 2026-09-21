/**
 * The scene's public API — what every host calls, and all it needs to know.
 *
 * A host does three things and nothing else: give us a container, give us a
 * `CellSceneSpec` (from `GET /cells/{id}/scene`, from the Streamlit page's
 * in-process build, from a committed sample file — we do not care), and tell us
 * when something changed. There is deliberately no host detection anywhere in
 * this directory: the Streamlit page, the React SPA, the static harness and a
 * third party's own page all take the same three calls.
 *
 * `mountCellScene` returns null only when the browser cannot give us a WebGL
 * context. `mount` (the friendlier wrapper below) turns that into a visible
 * explanation, because a blank rectangle is not an empty state.
 */

import { mountCellScene } from "./engine.ts";
import type { CellSceneHandle, MountOptions } from "./engine.ts";
import { SCENE_SCHEMA_VERSION } from "./types.ts";
import type { CellSceneSpec, SceneTheme } from "./types.ts";

export { SCENE_SCHEMA_VERSION };
export type { CellSceneSpec, SceneTheme } from "./types.ts";
export type { CellSceneHandle, FrameState, MountOptions } from "./engine.ts";
export { buildScene, buildTimeline, DEFAULT_PEEL, partReadings, peelSweepDeg, readingAt, todayCursor } from "./geometry.ts";
export type { PartReading } from "./geometry.ts";
export { mountCellScene } from "./engine.ts";

/** The version handshake. A host and a renderer that disagree say so. */
export interface VersionCheck {
  ok: boolean;
  expected: number;
  received: number | null;
  message?: string;
}

export function checkSchemaVersion(spec: unknown): VersionCheck {
  const received =
    spec && typeof spec === "object" && "schemaVersion" in spec
      ? Number((spec as { schemaVersion: unknown }).schemaVersion)
      : null;
  if (received === SCENE_SCHEMA_VERSION) return { ok: true, expected: SCENE_SCHEMA_VERSION, received };
  return {
    ok: false,
    expected: SCENE_SCHEMA_VERSION,
    received,
    message:
      received === null
        ? "This scene document carries no schema version, so the renderer cannot know what it describes."
        : `This scene document was produced by schema version ${received}, but this renderer speaks ` +
          `version ${SCENE_SCHEMA_VERSION}. Rebuild the host or regenerate the scene rather than ` +
          "reading a scene the renderer was not written for.",
  };
}

export interface MountResult {
  handle: CellSceneHandle | null;
  version: VersionCheck;
  error: string | null;
}

function emptyState(container: HTMLElement, message: string): void {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.style.cssText =
    "display:flex;align-items:center;justify-content:center;height:100%;min-height:180px;" +
    "padding:18px;text-align:center;font:400 13px/1.5 ui-sans-serif,system-ui,sans-serif;color:#94a3b8;" +
    "border:1px dashed #1f2937;border-radius:10px;";
  box.textContent = message;
  container.appendChild(box);
}

/**
 * Mount a scene, refusing politely anything this renderer cannot draw.
 *
 * The refusals are the point: a spec from a newer producer, a document with no
 * version at all, or a browser with WebGL switched off each get a sentence
 * instead of a blank canvas or an exception in a page nobody is watching.
 */
export function mount(
  container: HTMLElement | string,
  spec: CellSceneSpec,
  options: MountOptions = {},
): MountResult {
  const element = typeof container === "string" ? document.querySelector<HTMLElement>(container) : container;
  if (!element) {
    return {
      handle: null,
      version: { ok: false, expected: SCENE_SCHEMA_VERSION, received: null },
      error: "No container element matched.",
    };
  }
  const version = checkSchemaVersion(spec);
  if (!version.ok) {
    emptyState(element, version.message ?? "Unsupported scene version.");
    return { handle: null, version, error: version.message ?? "Unsupported scene version." };
  }
  const handle = mountCellScene(element, spec, options);
  if (!handle) {
    emptyState(
      element,
      "This browser did not give the page a WebGL context, so the 3D view cannot draw. " +
        "Every number it would have shown is on the cards beside it.",
    );
    return { handle: null, version, error: "WebGL unavailable." };
  }
  return { handle, version, error: null };
}

/**
 * The band and provenance key, in the DOM.
 *
 * Reads the same theme the meshes do, so the legend cannot describe different
 * colours than the ones painted. Hosts that draw their own panel (the Streamlit
 * page does) simply do not call this.
 */
export function renderLegend(container: HTMLElement, theme: SceneTheme): void {
  const rows = theme.sohBands
    .map(
      (band) =>
        `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px">` +
        `<i style="width:10px;height:10px;border-radius:2px;background:${band.color};display:inline-block"></i>` +
        `${band.label}${band.min ? ` ≥ ${band.min}%` : ""}</span>`,
    )
    .join("");
  const provenance = Object.entries(theme.provenanceColors)
    .filter(([key]) => key !== "")
    .map(
      ([key, color]) =>
        `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px">` +
        `<i style="width:10px;height:2px;background:${color};display:inline-block"></i>${key}</span>`,
    )
    .join("");
  container.innerHTML =
    `<div style="font:400 11px/1.6 ui-sans-serif,system-ui,sans-serif;color:${theme.muted}">` +
    `<div style="margin-bottom:2px">${rows}</div><div>${provenance}</div></div>`;
}

/**
 * Mount every `[data-cell-scene]` element in a document.
 *
 * The declarative path exists for hosts that are just HTML (the static harness,
 * a report someone pastes a scene into). The spec comes from a sibling
 * `<script type="application/json" data-cell-scene-spec>` or from the element's
 * `data-src` URL, so a page can embed a scene without writing JavaScript.
 */
export function autoMount(root: ParentNode = document): MountResult[] {
  const results: MountResult[] = [];
  const targets = Array.from(root.querySelectorAll<HTMLElement>("[data-cell-scene]"));
  for (const target of targets) {
    const inline = target.querySelector<HTMLScriptElement>("script[data-cell-scene-spec]");
    const src = target.dataset.src;
    const options: MountOptions = {
      exploded: Number(target.dataset.exploded ?? 0) || 0,
      dataScaled: target.dataset.dataScaled === "true",
      casing: target.dataset.casing === "hidden" ? "hidden" : "translucent",
    };
    if (inline?.textContent) {
      try {
        const spec = JSON.parse(inline.textContent) as CellSceneSpec;
        options.cursor = spec.series.cycles.length - 1;
        results.push(mount(target, spec, options));
      } catch (error) {
        results.push({
          handle: null,
          version: { ok: false, expected: SCENE_SCHEMA_VERSION, received: null },
          error: `Could not parse the inline scene: ${String(error)}`,
        });
      }
    } else if (src) {
      void fetch(src)
        .then((response) => response.json() as Promise<CellSceneSpec>)
        .then((spec) => {
          options.cursor = (spec.series.cycles.length ?? 1) - 1;
          mount(target, spec, options);
        })
        .catch((error: unknown) => {
          emptyState(target, `Could not load ${src}: ${String(error)}`);
        });
    }
  }
  return results;
}
