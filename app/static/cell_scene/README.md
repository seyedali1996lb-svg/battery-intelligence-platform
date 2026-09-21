# The 3D cell scene — build artifacts

Four files, all committed on purpose. This directory is what a **browser** loads:
Streamlit serves `app/static/` straight from disk (`server.enableStaticServing`,
`.streamlit/config.toml`), and the API mounts this same directory at `/scene`.
Neither host can run Node, so the renderer has to already exist here.

| File | What it is |
| --- | --- |
| `cell_scene.js` | The whole scene engine, minified, three.js included (~578 kB, ~146 kB gzipped). Built from `frontend/src/scene/` — one implementation, three hosts. |
| `index.html` | The standalone host: no framework, no build step. Loads the bundle above and a scene document, and does everything the Streamlit page and the React SPA do. |
| `sample_scene.json` | A real `CellSceneSpec` for a real cell, so the page works in a fresh checkout with no data loaded and no API running. Also read by `frontend/src/scene/spec.test.ts`, which is how the Python producer and the JavaScript renderer are held to the same document. |
| `manifest.json` | The hashes that make a stale or hand-edited bundle impossible to commit. |

## Regenerating (after any change under `frontend/src/scene/`)

```bash
cd frontend
npm ci
npm run test:scene      # node --test src/scene/*.test.ts — the geometry and the
                        # contract with src/cell_scene.py
npm run build:scene     # → ../app/static/cell_scene/cell_scene.js

cd ..
python scripts/export_scene_sample.py --cell B0005   # refreshes the sample + manifest
python -m pytest tests/test_cell_scene_bundle.py     # or: export_scene_sample.py --check
```

The build is deterministic in the ways that matter (fixed output filename, no
hash in the name) and the manifest records the digest of the bundle **and of
every source file it was built from** — so "I changed the geometry but forgot to
rebuild" fails a test rather than shipping a scene that disagrees with its own
source. `npm run build` (the SPA) is a separate build of the same sources; it
serves the scene from `frontend/dist` and does not touch this directory.

## Opening the standalone page

* From the Streamlit app: `http://localhost:8501/app/static/cell_scene/index.html`
* From the API: `http://localhost:8000/scene/index.html`
* Any static server: `python -m http.server` in this directory

Query parameters:

| Parameter | Effect |
| --- | --- |
| `?cell=B0005` | Fetch that cell's scene from the REST endpoint (`GET /cells/{id}/scene`) |
| `?api=https://host` | API base URL for `?cell=` (default: same origin) |
| `?token=…` | Bearer token for the REST endpoint — it is authenticated |
| `?spec=./file.json` | Load any URL that serves a `CellSceneSpec` |
| *(nothing)* | Falls back to `sample_scene.json` |

A `file://` double-click cannot `fetch` a sibling file, so the page has no data
path in that mode — serve the directory over HTTP. That is a browser rule, not a
restriction of this page.

## Verifying the render with no compositor

A WebGL canvas whose drawing buffer is not readable cannot be screenshotted — and
in a headless or offscreen webview it may produce no frames at all. The frame is
still readable from inside the page: mount the scene, wait two
`requestAnimationFrame` callbacks (the engine draws in its own), then
`gl.readPixels(...)` on the canvas' `webgl2`/`webgl` context **in the same task**.
The browser clears the buffer at composite time, so reading after a frame in a
later task returns a blank image, which is how a working scene reads as an empty
one. That is how the layout, the explode separation and the per-state differences
were checked during development — the pixels are the evidence, and it is worth
re-running before trusting a change that only the eye is supposed to catch.

## Manual visual checklist

There is no headless browser in this repository's CI, so the numbers, the
geometry and the serialized document are what the automated gates verify. The
things only an eye can catch are below; run through it after any change to
`frontend/src/scene/` or to the theme in `src/cell_scene.py`. Open the harness
page with `?cell=B0005` (or another real cell) and check:

1. **The cell is closed.** Orbit under and behind it: no hole where the casing
   should be, no face that vanishes as you cross behind it. (Casing is drawn as a
   cut-away arc — the *shape* is open, but nothing should flicker or disappear.)
2. **The roll reads as a roll.** Three nested coils, anode inside separator
   inside cathode, none poking through another or through the casing at any
   explode position.
3. **Drag "Exploded view" to 100%.** The cap, vent and positive terminal lift in
   order, the coils separate radially, and everything stays inside the can.
4. **Scrub the life cursor.** The state gauge on the casing fills as SOH falls;
   the part values and their colours change; particles do **not** teleport. The
   **Anatomy list on the right must move with the scene** — those rows come from
   `handle.parts()`, which resolves each part's reading at the scene's own
   cursor, so a row that keeps its value while the slider moves is the bug that
   API exists to prevent.
5. **Press "Play life".** The gauge drains smoothly and stops at the end of the
   timeline — including at the end of the projection, when there is one — and the
   button returns to "▶ Play life" when it does (the engine reports the end via
   `onPlaybackEnd`; it cannot relabel a button that belongs to the host).
   Pressing it at the very end replays from the first cycle rather than doing
   nothing.
6. **Past "today".** On a cell whose forecast routing allows a projection the
   gauge keeps moving and the cycle label says "projected"; on a cell it refuses,
   the timeline simply ends and the note in the disclosures explains why.
7. **Hover a part.** It lights up and its label stays legible over the meshes.
   Clicking a row in the "Anatomy" list turns the camera to that part's side and
   dims the rest.
8. **Toggle "Data-scaled geometry".** The caption under the canvas prints the
   mapping in use — and the SEI film only grows when the split is identified
   (an unidentified cell prints the *total fade* mapping instead and the film
   stays architecture).
9. **Toggle "Strip the casing".** The shell goes, the cell stays readable, and
   nothing else changes.
10. **Colours agree with the rest of the app.** A cell the Streamlit UI calls
    "Degrading" is amber on the casing gauge and in the legend here too.
