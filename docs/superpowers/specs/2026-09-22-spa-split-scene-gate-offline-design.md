# Design: SPA code-split, browser test gate, and single-file offline scene

**Status: approved, 2026-09-22.** Three improvements from the Top-10 survey,
scoped together because they share one execution window and land as three
independent commits. Each section stands alone; none depends on another except
that the gate (B) lands last and covers both artifacts.

## Context and measurements

- The SPA build emits `index-*.js` at **623.00 kB / 184.13 kB gzip** because
  `frontend/src/App.tsx` statically imports all seven views; Vite warns that the
  chunk exceeds 500 kB. The scene chunk (615.68 kB / 156.09 kB gzip) is already
  split out by a dynamic import in `CellSceneView.tsx` — the pattern works and
  is the precedent for splitting the rest.
- The standalone harness (`app/static/cell_scene/index.html`) is self-contained
  when served (committed `cell_scene.js` + `sample_scene.json`), but its header
  comment promises a double-click `file://` fallback the code does not
  implement — `fetch()` of a sibling file is blocked on `file://`.
- Backlog sources: `docs/scene_prompt.md` items 2 (screenshot gate) and 4
  (portability). Python Playwright 1.63.0 and Pillow 12.3.0 are already
  installed; Node Playwright is not, and is not needed.

## Non-goals

- No URL/deep-link state (Top-10 item 5), no theming work, no table makeovers.
- No pixel goldens for mid-playback frames (time-dependent by nature).
- The two deliberately deferred product concepts remain deferred per
  `docs/deferred_product_concepts.md`.
- SPA smoke asserts *mount and render cleanliness*, not data correctness —
  that remains the API test suite's job with a real backend.

## A. SPA route-level code splitting

`App.tsx` converts the seven static view imports (`ElectrochemicalWorkbench`,
`CellSceneView`, `FleetSummaryView`, `ActionCenterView`,
`PassportCircularityView`, `UniversalIngestionView`, `LiveMonitorView`) to
`React.lazy` and wraps the view container in one `<Suspense>` whose fallback is
a small "Loading view…" state reusing existing CSS. Login, header, and tab bar
stay eager — they are what first paint needs.

Chosen over vendor-chunk splitting (`manualChunks`) because lazy views remove
whole features from the entry chunk instead of repackaging them; `three` is
already excluded, so the entry is react + recharts (the default tab needs
recharts) + eager shell + whichever views stay out. Success criteria, measured
by `npm run build` before and after and recorded in the CHANGELOG entry:

1. every non-default view module is absent from `index-*.js`;
2. the >500 kB Vite warning is gone, or the entry shrank ≥ 25 % if recharts
   alone keeps it above the threshold (whichever is the honest measured
   outcome);
3. `tsc -b`, `oxlint`, `npm run build`, and the 112 node tests stay green;
4. no behavior change beyond an async tab load — same DOM, same handlers.

## B. Browser gate: scene goldens + mocked-API SPA smoke

New package `tests/e2e/` (pytest, Python Playwright, Pillow — no new Node
dependencies). Collection is unconditional; each test **skips with an
actionable message** when its prerequisite is missing (Chromium binary not
installed → `playwright install chromium`; `frontend/dist/` absent →
`npm run build`). CI installs the browser, so the gate is enforced there.

**Launch and determinism.** Chromium launches with the project's proven
SwiftShader flags (`--use-gl=angle --use-angle=swiftshader
--enable-unsafe-swiftshader`), a fixed viewport and `deviceScaleFactor=1`.
Requests to `fonts.googleapis.com` / `fonts.gstatic.com` are aborted so text
renders from system faces on every machine. Sliders, checkboxes, and buttons in
the harness are driven by in-page event dispatch via `page.evaluate` (the
project already learned that synthetic `.click()` can hang under this driver);
SPA tab buttons use normal locator clicks with short timeouts, falling back to
in-page dispatch if a hang is observed.

**`test_scene_goldens.py`** — four deterministic states of the harness page:

| golden | state | how it is reached |
|---|---|---|
| `rest` | assembled, last measured cycle | default mount, wait for `#soh` non-`—` |
| `exploded` | 100 % explode | dispatch `input` on `#explode` = 100 |
| `stripped` | casing hidden | toggle `#strip-casing` |
| `inspected` | one part focused, others dimmed | hover the first part `<li>`, wait for the camera tween to settle |

Comparison: Pillow decode, a pixel counts as *different* when any channel
differs by more than 10; the test fails when different pixels exceed **0.5 %**
of the viewport. Goldens are **platform-keyed** (`tests/e2e/goldens/win32/`,
`tests/e2e/goldens/linux/`) because Windows and Linux rasterize system fonts
differently. `UPDATE_GOLDENS=1` regenerates the current platform's set; a
missing golden fails with that instruction. Linux goldens are produced once by
a CI run with `UPDATE_GOLDENS=1`, downloaded from the artifact, and committed —
documented in the plan.

**`test_spa_smoke.py`** — serves the built `frontend/dist` over a local static
server, intercepts every API call with Playwright routes returning committed
fixtures (`tests/e2e/fixtures/`, shaped from `frontend/src/api.ts` and
`types.ts`; the scene fixture is the real `sample_scene.json`). Flow: login
form → mocked token → click each of the seven tabs; each view must mount its
`data-testid="view-<tab>"` wrapper (one-line addition per view) and the test
collects `console.error` / `pageerror` throughout — any entry fails the test.

**`test_offline_artifact.py`** — opens the generated `offline.html` via
`file://` and asserts the scene mounts and `#soh` shows a reading with zero
network capability. This is the literal "opens anywhere" proof.

## C. Single-file offline scene (`offline.html`)

`index.html` gains one marker line and a preference in `loadSpec()`:

```js
const EMBEDDED_SPEC = /*__EMBEDDED_SPEC__*/null;
// loadSpec(): if EMBEDDED_SPEC, return it (source "embedded sample");
//             else fetch as today.
```

The harness therefore behaves byte-for-byte as today (marker resolves to
`null`), and its header comment is corrected: the double-click fallback this
page never had now has a name — `offline.html` — instead of a promise it
cannot keep.

`scripts/export_scene_sample.py`, which already owns this directory's
artifacts, additionally writes `offline.html`: `index.html` with the marker
replaced by `sample_scene.json` content and `<script src="cell_scene.js">`
replaced by the bundle inlined. Generation is pure string-marker replacement —
no HTML parsing — and fails loudly if either marker is absent. The manifest
gains `offlineHtml` (bytes) and `offlineHtmlSha256`.

Enforcement joins the existing guards:

- `tests/test_cell_scene_bundle.py` regenerates offline.html in-memory and
  asserts byte-identity with the committed file, plus manifest fields;
- `tests/test_cell_scene_api.py` adds `offline.html` to the served-files
  assertion (reached as `/scene/offline.html`);
- the Battery 3D page's JSON download stays (it is the document itself, the
  same bytes `GET /cells/{id}/scene` serves) and gains a **sibling** button,
  "Download portable view (single HTML)", which inlines the *currently
  viewed* cell's spec into the same template at click time — a passport for
  this cell, not only for the sample. The composition lives in one pure
  helper, `compose_offline_html(spec, index_html, bundle)`, next to the
  producer in `src/cell_scene.py`; `export_scene_sample.py` calls the same
  helper to write the committed `offline.html` with the sample spec, so
  generator and page cannot drift. The helper is unit-tested (marker-missing
  raises, output contains no `__EMBEDDED_SPEC__` marker and no
  `script src="cell_scene.js"`, round-trips the given spec).

## D. Gates and docs

Full chain green before each commit: `npm run lint`, `npm run build`,
`npm run build:scene`, `npm run test:scene`, sample export, full `pytest`
(now including `tests/e2e/`), `python -m pyright`, `python -m mkdocs build
--strict`. CI (`ci.yml`) gains a `playwright install chromium` step ahead of
pytest.

Docs carried to shipped state with their features: `docs/scene_prompt.md`
backlog item 4 → mapping row (item 2 likewise once the gate ships),
`app/static/cell_scene/README.md` checklist/sizes, `CHANGELOG.md` with the
measured entry-chunk numbers and the new gate, this spec.

## E. Commit sequence

1. `docs: spec for spa split, scene gate and offline html` — this document
   (committed before implementation, per the brainstorming workflow).
2. `perf(frontend): lazy-load spa tabs out of the first paint` — Section A.
3. `feat(scene): single-file offline html a passport can open anywhere` —
   Section C + its docs hunks.
4. `test(e2e): scene goldens, spa smoke and the offline file gate` —
   Section B + CI step + remaining docs hunks.

Execution order A → C → B so the gate lands last and covers both new
artifacts. Subject-only conventional messages; `data/relog_*.log` and
`data/rerun_*.log` are never staged.

## Risks and tunables

- **Golden flakiness** (driver updates, GPU variance): the 0.5 % / Δ>10
  tolerances are deliberately stated as *tunable constants* — widen them only
  with a recorded reason in the test file's docstring, never silently.
- **SPA mock drift**: fixtures track `api.ts` by construction (they are
  written from it), and drift shows up as a smoke failure — which is the
  signal to update fixtures, not to weaken the assertion.
- **Entry chunk may stay >500 kB** if recharts alone exceeds it: criterion 2
  allows the honest measured outcome; the CHANGELOG records the real number.
- **offline.html staleness**: covered by byte-identity regeneration in the
  bundle test — the same liability rule the committed bundle already follows.
