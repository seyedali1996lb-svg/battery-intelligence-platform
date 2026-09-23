/**
 * The standalone entry: the whole scene engine as one global script.
 *
 * This is what gets built into `app/static/cell_scene/cell_scene.js` and served
 * to hosts that are not this repository's frontend — the Streamlit page's
 * embedded canvas, the static harness, and anyone else's plain HTML page. It is
 * the same code the React SPA imports as a module; only the delivery differs,
 * which is the whole reason `src/scene/` has no framework in it.
 *
 * On load it mounts any declaratively-marked elements in the page, so a host
 * that wants a scene needs a `<div data-cell-scene>` and a JSON script tag and
 * nothing else — while hosts that drive the scene imperatively (Streamlit, the
 * SPA) call `window.CellScene.mount` / `update` themselves.
 */

import {
  SCENE_SCHEMA_VERSION,
  autoMount,
  checkSchemaVersion,
  decodeViewState,
  encodeViewState,
  legendHtml,
  mergeViewState,
  mount,
  mountCellScene,
  renderLegend,
} from "./index.ts";

export interface CellSceneGlobal {
  /** Bumped with the bundle; hosts may print it. */
  version: string;
  schemaVersion: number;
  mount: typeof mount;
  mountCellScene: typeof mountCellScene;
  autoMount: typeof autoMount;
  checkSchemaVersion: typeof checkSchemaVersion;
  renderLegend: typeof renderLegend;
  legendHtml: typeof legendHtml;
  /**
   * The shareable-view codec: hosts that keep their own URL (a router, a
   * report's query string) round-trip a view through these instead of
   * re-inventing one — the engine's own storage uses the same three.
   */
  encodeViewState: typeof encodeViewState;
  decodeViewState: typeof decodeViewState;
  mergeViewState: typeof mergeViewState;
}

const api: CellSceneGlobal = {
  version: "1.0.0",
  schemaVersion: SCENE_SCHEMA_VERSION,
  mount,
  mountCellScene,
  autoMount,
  checkSchemaVersion,
  renderLegend,
  legendHtml,
  encodeViewState,
  decodeViewState,
  mergeViewState,
};

declare global {
  interface Window {
    CellScene?: CellSceneGlobal;
  }
}

window.CellScene = api;

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => autoMount());
} else {
  autoMount();
}

export default api;
