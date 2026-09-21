import { defineConfig } from "vite";
import { resolve } from "node:path";

/**
 * The standalone build of the 3D cell scene (see src/scene/README.md).
 *
 * Separate from `vite.config.ts` on purpose. The SPA build bundles the scene as
 * a module and serves it from the API at `/app`; this build emits ONE
 * self-contained script at `app/static/cell_scene/cell_scene.js`, which is what
 * the Streamlit page embeds and what the static harness next to it loads. Both
 * come from the same `src/scene/` sources, so there is one implementation and
 * two deliveries.
 *
 * The output is committed, because a browser cannot be asked to build it: the
 * Streamlit app serves `app/static/` directly (`server.enableStaticServing`)
 * and must work from a fresh checkout with no Node toolchain. That is also why
 * `tests/test_cell_scene_bundle.py` fails on a stale or hand-edited bundle —
 * a committed artifact nobody can regenerate is a liability, and this one can.
 *
 * `emptyOutDir: false` keeps the harness page, the sample scene and the manifest
 * that live in the same directory.
 */
export default defineConfig({
  // The SPA's `public/` (favicon, icon sheet) belongs to the SPA; the scene
  // bundle lives in a directory shared with its harness page and sample scene,
  // and copying app furniture into it would put files there that no one can
  // explain the origin of.
  publicDir: false,
  build: {
    outDir: resolve(__dirname, "../app/static/cell_scene"),
    emptyOutDir: false,
    sourcemap: false,
    target: "es2020",
    lib: {
      entry: resolve(__dirname, "src/scene/standalone.ts"),
      name: "CellScene",
      formats: ["iife"],
      fileName: () => "cell_scene.js",
    },
    rollupOptions: {
      output: {
        // A stable filename: the artifact is referenced by path from HTML that
        // cannot know a hash, and the freshness guard hashes the file itself.
        entryFileNames: "cell_scene.js",
        assetFileNames: "cell_scene.[ext]",
      },
    },
  },
});
