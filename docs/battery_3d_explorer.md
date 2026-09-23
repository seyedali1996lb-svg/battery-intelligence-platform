# The Battery 3D Explorer — design, implementation, and every detail

This page is the dedicated record for the **Battery 3D Explorer**: what it is, why it is built
the way it is, and how each part works. It covers the whole system (producer → document →
renderer → hosts) and then walks through, in order, the ten ranked improvements that took it
from "anatomy view that works" to a view that is shareable, accessible, fast, and honest about
its own numbers.

It is deliberately deep. If you want the *standing prompt* the scene is built against (what
"accurate" means, the three contracts), read [`scene_prompt.md`](scene_prompt.md). If you want
the geometry derivations (turn counts, film thickness, what the drawn millimetres mean), read
[METHODOLOGY §28](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/METHODOLOGY.md).
This page is about the *system and this hardening pass*.

---

## 1. What the Explorer is

The Battery 3D Explorer draws one cell's construction — can, wound roll, electrode layers, cap,
vent, tabs, film — part by part, and binds every drawn part to a number the platform already
computes. It adds **no new model**. A reader who has never seen inside a cell should look at it
for thirty seconds and answer: *what is this part, what is it for, and what do we know about it?*

The hard problem is not the 3D. It is that **four different hosts** must render the *same* scene
from the *same* document, and that the picture must never disagree with the numbers printed
beside it. That constraint drives every design decision below.

### The four contracts

| Contract | What it means | Enforced by |
| --- | --- | --- |
| **Data** | Every drawn quantity comes from the document; hosts print nothing the scene did not give them. | `partReadings()` is DOM-free; hosts build tables from the renderer's own snapshot *at the current cursor*, not from `parts[].value` (a producer-time snapshot). |
| **Geometry** | Drawn sizes come from the document's `physical` block in millimetres, not from taste. Nothing leaves the cell's declared envelope at any explode position. | `geometry.ts` derivations recomputed and compared against the producer by `spec.test.ts`. |
| **Honesty** | Provenance (measured / derived / fitted / projected) is tagged per field; a routing refusal produces *no* future and says why; a shared link reopens the same picture. | Schema (`docs/cell_scene.schema.json`) + Python and JS validators + tests. |
| **Hosts agree** | One implementation, four hosts; controls, legend, and view state cannot drift between them. | Pure modules mirrored on both sides (`legend.ts` ↔ `_scene_view.py`) and pinned by tests on each side. |

### The pipeline

```
src/cell_scene.py            frontend/src/scene/            app/ + frontend/
─────────────────────        ────────────────────           ─────────────────────────────
build_cell_scene_from_sources  geometry.ts (DOM-free)        app/_pages/battery3d.py  (Streamlit)
        │                     engine.ts (only file that      app/_scene_view.py       (iframe host)
        ▼                       knows three.js exists)        app/static/cell_scene/index.html (harness)
  CellSceneSpec (JSON)   →    legend.ts / viewstate.ts  →    frontend/src/components/CellSceneView.tsx (SPA)
        │                     annotation.ts / dossier.ts     GET /cells/{id}/scene   (REST)
        ▼                     hud.ts / theme.ts
 docs/cell_scene.schema.json   npm run build:scene →
 (versioned, validated both    app/static/cell_scene/cell_scene.js (committed bundle)
  Python and JavaScript sides)
```

**Why one document and not a per-host widget:** a Streamlit widget cannot be reused by the SPA,
a static page, or a third party's site. A JSON document with a version and a schema can. The
cost is that the renderer must be framework-free (it is — `engine.ts` is the only file that
imports three.js, geometry is pure) and that the bundle must be committed, because neither
Streamlit nor a static server can run Node.

**Why the bundle is committed:** Streamlit serves `app/static/` from disk
(`server.enableStaticServing`, `.streamlit/config.toml`) and the API mounts the same directory
at `/scene`. Neither host can run Node at request time, so the built renderer has to already
exist in the repository. `manifest.json` carries the hashes that make a stale or hand-edited
bundle impossible to commit (and, as improvement #1 below shows, doubles as the cache key).

---

## 2. The renderer's module map

| File | Responsibility | Key design choice |
| --- | --- | --- |
| `types.ts` | `CellSceneSpec`, `SceneTheme`, `PartDossier` types. | The document's shape, typed once. |
| `geometry.ts` | `buildScene`, `buildTimeline`, `partReadings`, `readingAt`, `todayCursor`. | **Pure.** No DOM, no three.js — unit-testable by Node's own runner, and the single source both the dossier and the hosts' tables read from. |
| `engine.ts` | three.js scene graph, camera, raycast, badges, `FrameState`, `mountCellScene`. | Only file that knows three.js exists. Emits `FrameState` every frame; hosts **mirror** it, never duplicate it. |
| `annotation.ts` | `chromeSvg`, `layoutFlank`, **`groupUnmeasured`** (new). | The flank layout maths and the "no reading" grouping are pure, so the badge-collapse behaviour is testable without a browser. |
| `dossier.ts` | `composeDossier` — the part dossier panel. | Pure; reads the *live* reading, not the producer snapshot. |
| `legend.ts` **(new)** | `bandRange`, `legendHtml` — the band/provenance key. | Pure, and **mirrored line-for-line** by `legend_html`/`band_range` in `app/_scene_view.py`, because the Streamlit host draws its legend server-side and a divergence would mean two different keys for one scene. |
| `viewstate.ts` **(new)** | `ViewState`, `encodeViewState`, `decodeViewState`, `mergeViewState`, `readStoredView`, `writeStoredView`. | Pure; no DOM globals except through injected storage functions, so every field round-trips in a Node test. |
| `theme.ts` | Palettes, `readPref`/`writePref`, **`withAlpha` (new)**. | Badge chrome is *derived* from the active palette, not hardcoded. |
| `hud.ts` | The cockpit HUD rail: telemetry, camera presets, hotkeys. | Callbacks up, state down; `paintChrome`-style one-way sync. |
| `standalone.ts` | The static harness's glue (`window.CellScene`, `renderLegend`, URL write-back). | No framework, no build step. |
| `index.ts` | Public surface: `mount`, `autoMount`, `renderLegend`, `checkSchemaVersion`, exports. | The API a third party would import. |

---

## 3. The hosts and how they stay in sync

| Host | Entry | Legend | View-state URL |
| --- | --- | --- | --- |
| **Streamlit iframe** | `app/_pages/battery3d.py` → `app/_scene_view.py` | Server-side `legend_html(theme)` | Slider/checkboxes mirrored from `FrameState`; cursor not pinned at mount |
| **Static harness** | `app/static/cell_scene/index.html` | `window.CellScene.renderLegend(…)` | `syncUrl` → coalesced `history.replaceState` |
| **React SPA** | `frontend/src/components/CellSceneView.tsx` | `sceneModule.renderLegend(…)` | `writeUrl` read-modify-write; `?cell=` written on change |
| **REST / third party** | `GET /cells/{id}/scene` | — (document only) | — |

**The sync rule (improvement #7):** the engine owns state and emits it in `FrameState` every
frame; each host *derives* its controls from that frame. A host never keeps a second copy of
explode/peel/cursor to drift. The Streamlit mount previously pinned `cursor` at mount (so a
restored view's timeline jump was overwritten); the pin was removed and the slider now syncs
via `el("cursor").value = String(state.cursor)` in the frame callback. The SPA replaced its
local duplicated control state with `FrameState`-derived controls. `App.tsx` gained `?tab=` URL
sync (validated, not cast — a hand-edited `?tab=banana` is dropped on the next click).

---

## 4. The ten improvements — why, and how

Each improvement lists the **defect** (or opportunity), the **fix**, and the **evidence** (tests
that would catch a regression).

### #1 — Cache-bust `cell_scene.js` via the manifest's `bundleSha256`

* **Defect.** The bundle was referenced by a static path. A deployed browser caches
  `cell_scene.js` indefinitely, so a new bundle shipped to `app/static/` would never reach users
  who had visited before (observed: a vignette fix was in the bundle but stale renderers kept
  showing the old one).
* **Fix.** `app/_scene_view.py` reads `manifest.json`'s `bundleSha256` at import and sets
  `BUNDLE_URL = BUNDLE_PATH + "?v=" + sha[:12]` (falls back to the plain path when the manifest
  is absent — missing manifest must degrade to "no version", not to a broken page).
  `scripts/export_scene_sample.py` stamps the harness's `<script>` tag with the same `?v=` value
  and validates it in `check()`.
* **How the stamp works.** The export script matches the *whole* script tag with
  `re.subn(f'<script src="cell_scene.js?v={sha[:12]}">', …)` against
  `r'<script src="cell_scene\.js(?:\v=[0-9a-f]*)?">'` and asserts exactly one replacement.
  (An earlier group-based pattern `(">")` demanded a `">"` — quote-gt-quote — that never exists
  in `js"><`, so it always failed with "found 0"; matching the whole tag and substituting it
  outright is what actually works.)
* **Evidence.** `tests/test_cell_scene_page.py` asserts the iframe's script URL ends in `?v=` +
  the digest; `tests/test_cell_scene_bundle.py` asserts the harness tag is stamped and the stamp
  equals `manifest.bundleSha256[:12]`; the digest itself is `sha256(cell_scene.js)`.

### #2 — The legend: `band.min` falsy bug, temperature bands, headings

* **Defect.** The band renderer used `band.min ? …`, so a band starting at **0** (the healthy
  SOH band, `min=0`) read as falsy and lost its `0–90%` label. Temperature bands had **no**
  semantic key at all, and the key had no section headings.
* **Fix.** New pure `frontend/src/scene/legend.ts`:
  * `bandRange(min, max, unit)` — uses `!== null/undefined` checks (not truthiness), so
    `band_range(0, 80, "%")` emits `&lt; 80%` correctly;
  * `legendHtml(theme)` renders three headed sections — **Health** (SOH bands), **Origin**
    (provenance line styles), **Casing temperature** (temperature bands in °C) — omitting any
    section that is empty;
  * an `esc()` helper escapes `& < > "` so a label can never break the markup.
  Python `legend_html` / `band_range` / `_esc` in `app/_scene_view.py` mirror this exactly
  (same four chars escaped, same headings, same omission rule), because the Streamlit host
  renders its legend server-side.
* **Evidence.** `frontend/src/scene/legend.test.ts` (6 tests: zero-min band, escaping, headings,
  empty-section omission); four pins in `tests/test_cell_scene_page.py` (zero-min `band_range`,
  the three headings, empty-section omission, provenance rows).

### #3 — Mobile and touch

* **Defect.** The host layout was a fixed two-column grid; on a phone the stage collapsed and
  controls were unreachable. Touch input only produced `click` — no hover — so a tap could not
  select a part.
* **Fix.**
  * `app/_scene_view.py` adds `@media (max-width: 760px)` (single-column, stage re-heighted) and
    `@media (pointer: coarse)` (larger `button` padding, taller `range` inputs).
  * `engine.ts` gained `pointerdown` raycast — the press coordinates are raycast directly, so
    touch and mouse both hit-test without a preceding hover (`onPointerDown` →
    `setHovered(raycastAtPointer())`).
* **Evidence.** The raycast path is exercised by the pointer tests; the CSS is validated by the
  page-level `AppTest` runs.

### #4 — Accessibility: reduced-motion, ARIA, live region

* **Defect.** The scene animated continuously (pulses, springs) regardless of
  `prefers-reduced-motion`; badges and the canvas were silent to screen readers; a selection
  change was not announced.
* **Fix.**
  * `reducedMotion` is read once per mount via `matchMedia("(prefers-reduced-motion: reduce)")`.
    When set: springs **snap** instead of chasing, camera poses **jump** instead of tweening, the
    target reticle **holds** its rest radius instead of pulsing, and the pulse phase is never
    advanced. *Geometry and interaction are untouched — reduced motion is not reduced function.*
  * The canvas gets an `aria-label`; badges get `role="button"` / `tabindex` / `aria-expanded` /
    `aria-label` (the collapse toggle announces *which* action it performs);
  * a visually-hidden **`aria-live="polite"`** region announces the current selection, cycle,
    and SOH — updated only when the message actually changes, so a screen reader is not spammed
    on every frame.
* **Evidence.** The ARIA attributes are asserted by the render tests; `npx oxlint` (with
  `jsx-a11y`, run in CI) passes with 0 warnings.

### #5 — Collapse the "no reading" badge clutter

* **Defect.** At cursors where many parts have no reading (projected-only parts before their
  cycle, for example), the flanks filled with a dozen identical grey `no reading` badges,
  drowning the badges that did carry a number.
* **Fix.** `groupUnmeasured()` in `annotation.ts` collapses all no-reading badges into a single
  count badge (`"N unmeasured"`), grouped by flank. One flag — `showUnmeasured` — toggles the
  reader's expansion choice; the count badge itself is a button with an explanatory
  `aria-label` (e.g. *"The document carries no reading for N part(s) at this cursor"*). A part
  that is *active* (hovered/selected) escapes the group and always draws its own badge.
* **Evidence.** `frontend/src/scene/annotation.test.ts` gained grouping tests (flank partitioning,
  active-part escape, count badge id `__unmeasured__`, expansion toggle).

### #6 — Performance: raycast cache, overlay gating, idle render, spec memo

* **Defect.** Three per-frame costs ran unconditionally:
  1. the raycast intersected the full mesh list on **every** pointer event (a 240 Hz mouse costs
     240 raycasts/s);
  2. the badge/leader-line overlay rebuilt even when nothing moved;
  3. the render loop drew every frame whether or not anything changed.
  On the Python side, `_spec_for` rebuilt the whole `CellSceneSpec` on every Streamlit rerun.
* **Fix.**
  * **Raycast cache.** `raycastTargets` (the intersectable meshes) is rebuilt only when part
    *visibility* changes. `onPointerMove` merely writes `pointer` and sets `pointerDirty` — one
    coordinate write per event. The actual `raycaster.intersectObjects` runs once per tick, and
    only when `pointerDirty` (or the camera) says it is needed.
  * **Overlay gating.** The leader-line/badge overlay recomputes its layout only when a
    `signature` (part ids, positions, annotations on/off) changes — not every frame.
  * **Idle render.** The engine tracks `needsRender`; when the scene, camera, cursor, and
    selection are all unchanged, the rAF loop skips the `renderer.render` call entirely. An
    untouched scene costs one boolean comparison per tick instead of a full draw.
  * **Spec memo.** `battery3d.py` gained `_SPEC_MEMO` — an identity-keyed single-slot memo
    (graphed by `is`, because `st.cache_data` cannot hash the inputs). Re-navigating the same
    cell reuses the document instead of rebuilding it.
* **Evidence.** The build stays at one `cell_scene.js`; the spec memo is covered by the
  `battery3d` `AppTest` runs (no behavioural change, only fewer rebuilds).

### #7 — Unify the four control surfaces

*Covered in §3 above.* The engine is the single source; hosts mirror `FrameState`. `?tab=` sync
in `App.tsx`, cursor-slider sync in `_scene_view.py`, `FrameState`-derived controls in the SPA.

### #8 — Shareable, persistent view state

* **Defect.** There was no way to say *"look at this cell, peeled 60%, at cycle 150, with the
  anode pinned."* A reload lost the reader's place entirely.
* **Fix.** New pure `frontend/src/scene/viewstate.ts`:
  * `ViewState` — seven fields: `cursor`, `exploded`, `peel`, `layout`, `part`, `annotations`,
    `theme`. Small on purpose: only what changes **how you look**, never a number read from the
    data (a shared link must reopen the *same picture*, not re-derive a different cell).
  * `encodeViewState` / `decodeViewState` — tolerant encode/decode over `URLSearchParams`
    (bad values fall back to default rather than throwing).
  * `mergeViewState` — one precedence for every host: **URL/explicit-host > stored > host
    default**.
  * `readStoredView` / `writeStoredView` — persistence keyed `view:<cellId>`, using `theme.ts`'s
    `readPref`/`writePref` so the whole app has one preference store.
  * The engine gained `FrameState.pinned` (a click) separate from `inspected`
    (`pinned ?? hovered`) — **hover never enters a shareable URL**, because a transient hover is
    not a view. `persistView()` writes on change; the mount epilogue merges stored + explicit
    view so a host that passes no view still lands on the reader's last one.
  * Default cursor when nothing is set = **last measured cycle** (`todayCursor`).
* **Host wiring.** The harness writes the view back to the query string on a 400 ms coalesced
  `history.replaceState` (browsers rate-limit `replaceState`); the SPA's `writeUrl`
  read-modify-writes the current params and writes `?cell=` on change; the Streamlit host no
  longer pins `cursor` at mount.
* **Evidence.** `frontend/src/scene/viewstate.test.ts` (6 tests: round-trip every field,
  `decode` tolerance, `merge` precedence, stored-view read/write).

### #9 — Behavioral and visual tests

* **Defect.** The improvements above (legend, view state, grouping, cache-bust) were *new
  behaviour* with no regression net; a future edit could silently reintroduce the `band.min`
  bug or break view-state precedence.
* **Fix.** Three new pure modules each shipped with a test file, and the Python side gained
  explicit pins:
  * `legend.test.ts` (6) — mirrored by 4 legend pins in `test_cell_scene_page.py`;
  * `viewstate.test.ts` (6);
  * `annotation.test.ts` gained 5 grouping tests;
  * `tests/test_cell_scene.py` gained 2 **r² gate pins** (`mechanism.physics.gate` == `R2_FLOOR`,
    `physics.refitEveryCycles` == `REFIT_EVERY_CYCLES`);
  * `tests/test_cell_scene_bundle.py` / `test_cell_scene_page.py` assert the `?v=` digest and
    the harness stamp;
  * `tests/test_pack_layout_3d.py` / `test_degradation_space_3d.py` pin the SOH hex tokens.
* **Totals.** `npm run test:scene` → **129/129** (was 112). Python: 178 across the five scene
  suites (+56 in physics-calibration/import-validator), 228 when the API and docs-nav tests are
  included. A Playwright smoke test was considered and **deliberately skipped** — there is no
  existing Playwright usage in `tests/`, and the behavioral coverage above is delivered by the
  129 TS tests plus the Python pins.

### #10 — Theme-driven badge chrome + de-duplicate constants

* **Defect.**
  1. Badge chrome colours were hardcoded literals scattered through `engine.ts`, so switching
     palettes left badges in the old palette's colours.
  2. The **`0.3` r² floor** for trusting a physics fit existed as a literal in `src/cell_scene.py`
     *and* as `MIN_FIT_R2_FOR_DOMINANT_MODE` in `physics_calibration.py` — two places that could
     drift.
  3. The **SOH bands** (healthy ≥90%, degrading ≥80%, EOL <80%) and their three colours were
     restated in `_design_tokens.py`, `_ui_helpers.soh_status`, `_pack_layout_3d.SOH_BANDS`,
     `_explore_3d` (`EOL_PCT`, `SOH_COLORSCALE`), and `battery3d._soh_color` — five call sites.
  4. Doc pointers had gone stale (a `vite.scene.config.ts` reference, a `cell_scene.py` doc
     pointer, and pack-layout comments pointing at moved code).
* **Fix.**
  * **Badge chrome from the theme.** `withAlpha(hex, alpha)` in `theme.ts`; badges now derive
    `background`, `border`, `box-shadow`, and value colour from the active palette via
    `withAlpha(theme.panel, …)` / `withAlpha(theme.accent, …)` / `withAlpha(theme.muted, …)`.
  * **One r² gate.** `src/cell_scene.py` defines module-level `R2_FLOOR` and
    `REFIT_EVERY_CYCLES`, imported from `physics_calibration` when available and falling back to
    `0.3` / `25`. Every call site (the fit gate, the spec's `mechanism.physics.gate`, the
    `refitEveryCycles` field) reads these — one source, no drift.
  * **One SOH band table.** Canonical tokens in `app/_design_tokens.py`:
    `SOH_HEALTHY_MIN = 90.0`, `SOH_EOL_MIN = 80.0`, `SOH_HEALTHY_COLOR = "#48bb78"`,
    `SOH_DEGRADING_COLOR = "#f6e05e"`, `SOH_EOL_COLOR = "#fc8181"`. Wired into
    `_ui_helpers.soh_status`, `_pack_layout_3d.SOH_BANDS`, `_explore_3d`
    (`EOL_PCT = SOH_EOL_MIN`, `SOH_COLORSCALE` built from the tokens), and
    `battery3d._soh_color` (which also gained an identity-keyed single-slot `_SPEC_MEMO`).
  * **Stale pointers fixed** in `vite.scene.config.ts`, `src/cell_scene.py`, and the pack-layout
    comments.
* **Evidence.** Hex/token pins in `test_pack_layout_3d.py` and `test_degradation_space_3d.py`
  assert the bands equal the tokens; r² pins in `test_cell_scene.py` assert the spec exposes the
  shared constants.

---

## 5. Regeneration and verification (the exact order)

Rebuild order matters: `build:scene` first (writes the bundle), then the export script (reads the
bundle, writes the manifest, stamps the harness), then tests.

```bash
cd frontend
npm ci
npm run test:scene      # node --test src/scene/*.test.ts — 129 tests
npx tsc --noEmit        # both tsconfigs, 0 errors
npx oxlint              # 0 warnings/errors (jsx-a11y included)
npm run build:scene     # → ../app/static/cell_scene/cell_scene.js

cd ..
python scripts/export_scene_sample.py          # writes manifest.json, stamps index.html ?v=
python scripts/export_scene_sample.py --check  # "cell scene artifacts are current"

pytest tests/test_cell_scene.py tests/test_cell_scene_page.py \
       tests/test_cell_scene_bundle.py tests/test_cell_scene_api.py \
       tests/test_pack_layout_3d.py tests/test_degradation_space_3d.py
npm run build            # full SPA
```

The three strings `npm run build:scene`, `npm run test:scene`, and `export_scene_sample.py` are
**asserted to exist** in `app/static/cell_scene/README.md` by `test_the_docs_that_explain_this_artifact_exist`
in `tests/test_cell_scene_bundle.py` — keep them when editing that file.

---

## 6. Where every piece lives

| Concern | File(s) |
| --- | --- |
| Producer / spec builder | `src/cell_scene.py` (`R2_FLOOR`, `REFIT_EVERY_CYCLES`) |
| Schema | `docs/cell_scene.schema.json` |
| Renderer bundle | `app/static/cell_scene/cell_scene.js` (+ `manifest.json`, `index.html`, `sample_scene.json`, `README.md`) |
| Renderer source | `frontend/src/scene/` (`engine.ts`, `geometry.ts`, `legend.ts`, `viewstate.ts`, `annotation.ts`, `dossier.ts`, `hud.ts`, `theme.ts`, `standalone.ts`, `index.ts`) |
| Streamlit host | `app/_pages/battery3d.py`, `app/_scene_view.py` |
| React SPA host | `frontend/src/components/CellSceneView.tsx`, `frontend/src/App.tsx` |
| Export / cache-bust | `scripts/export_scene_sample.py` |
| Design tokens (SOH bands) | `app/_design_tokens.py` |
| SOH consumers | `app/_ui_helpers.py`, `app/_pack_layout_3d.py`, `app/_pages/_explore_3d.py`, `app/_pages/battery3d.py` |
| TS tests | `frontend/src/scene/*.test.ts` (129) |
| Python tests | `tests/test_cell_scene{,_page,_bundle,_api}.py`, `tests/test_pack_layout_3d.py`, `tests/test_degradation_space_3d.py` |
| Standing prompt | `docs/scene_prompt.md` |
| Geometry derivations | `METHODOLOGY.md` §28 |

---

## 7. Honest limits

* **The Playwright smoke test was skipped.** No existing Playwright usage in `tests/`; the
  behavioral net is the 129 TS tests + Python pins. A visual/interaction smoke test remains an
  open, explicitly-optional item.
* **The bundle is committed**, so any change under `frontend/src/scene/` requires the rebuild →
  export → test order above. `manifest.json`'s `sourceSha256` (which now includes `legend.ts`
  and `viewstate.ts`) makes a skipped rebuild fail fast rather than ship stale.
* **The scene is a schematic, not CAD.** Layer counts, turn counts, and particle counts are
  drawn for legibility; see METHODOLOGY §28 for what the drawn millimetres do and do not mean.
