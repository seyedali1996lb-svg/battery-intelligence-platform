# Design — Museum-Grade 3D Battery Explorer (Codex / CAD Redesign)

**Date:** 2026-09-21 · **Status:** approved design, pre-implementation
**Scope decision (user):** one design, phased build — five coupled subsystems, each phase green before the next.

## 1. Context

The 3D cell scene is a portable capability: one framework-free renderer
(`frontend/src/scene/`), one versioned document (`docs/cell_scene.schema.json`,
produced by `src/cell_scene.py`), four hosts (Streamlit page, React SPA,
standalone page, `GET /cells/{id}/scene`). Today: 16 anatomy parts, single
theme, SVG leader lines with margin-pinned badges, synchronous explode rebuilds,
host-side controls, React-only dossiers. Verified at 76 Node + 101 Python tests,
bundle 595,359 B with manifest freshness.

**Decisions taken during brainstorming:**

| # | Question | Decision |
|---|---|---|
| 1 | Sequencing | One design, six phased builds, each green |
| 2 | Dossier data | Two-tier + refusals: format-typical specs tagged as such; only platform-computed values shown live; explicit refusal rows otherwise |
| 3 | Missing components | Three new parts → **19** (`gasket`, `cid_ptc`, `bottom_insulator`); exhaust ports are cap detail geometry, not a part |
| 4 | Theme architecture | Document-carried `palettes` map; renderer `setTheme()`; reskin covers stage + HUD + dossier frame, not surrounding app chrome |
| 5 | Overall architecture | **Approach A — engine-centric**: all five systems live in `frontend/src/scene/`; hosts stay thin |

### Invariants that must hold after every phase

1. **Bijection**: `MESH_PART_IDS` ↔ `spec.parts` ↔ schema enum ↔ fixture ↔ sample (19).
2. **Envelope**: nothing drawn leaves the cell's radial envelope at any explode/peel position; ribbons tile one turn's advance exactly; assembled state (`exploded = 0`) is exactly the document's declared winding.
3. **Honesty**: every number carries provenance; typical-for-format values are tagged; absent measurements render refusals, never fabricated values.
4. **Artifacts**: `npm run build:scene` + `python scripts/export_scene_sample.py --cell B0005` keep bundle/sample/manifest hashes fresh; `test_cell_scene_bundle.py` guards staleness.
5. **Compatibility**: additive-optional schema changes only — **no `SCENE_SCHEMA_VERSION` bump** (it is checked with exact equality; a bump would refuse every existing document). Old documents (no `palettes`, `dossier`, `category`) render exactly as today.

### Non-goals

- Reskinning the hosts' surrounding app chrome (only the artifact frame).
- Self-hosted font binaries (CDN link + system fallbacks instead).
- Per-part impedance/overpotential modeling (does not exist; refusal rows instead).
- Vertex-morph unrolling (build-time layout switch with crossfade).
- Slimming host-owned controls (hosts keep their widgets; backlog).
- Screenshot goldens (pre-existing backlog item, unchanged).

## 2. Module architecture (Approach A)

```
frontend/src/scene/
  types.ts        + ScenePart.category?, ScenePart.dossier?, SceneTheme presentation
                  tokens, palettes map, Peel/Layout option types
  geometry.ts     + peel sweep math, layout="unrolled" strips, 3 new part
                  placements, explode keys (gasket/cidPtc/bottomInsulator),
                  pure helpers: peelSweepDeg(), springStep(), mmReadout()
  annotation.ts   (NEW, mostly pure) layoutFlank solver, card model, reticle/pulse
                  math, desaturate/dim colour helpers
  hud.ts          (NEW, DOM) control rail, playback bar, presets, hotkeys,
                  telemetry, theme switcher, drafting chrome overlay
  dossier.ts      (NEW, pure) compose a part's dossier from document + cursor
                  state (two-tier rows, live diagnostics, refusals)
  tween.ts        (NEW, pure) critically-damped spring + camera spherical tween
  theme.ts        + palette lookup, variant helpers
  materials.ts    + PALETTE_MATERIAL_ADJUST {dark, light}
  engine.ts       orchestration only: tick loop, tween application, selection
                  treatment (dim/rim), setTheme, HUD/annotation mounting,
                  renderer.info telemetry
  index.ts        public API: setTheme, camera presets, telemetry getters
```

**Data flow:** document (only source of truth) → `buildScene()` (geometry,
readings, colours) → engine paint → per-frame: tween step → (throttled) rebuild →
annotation layout (pure) → SVG/DOM overlay → `onFrame`/`onInspect` → hosts.
The engine owns all stage-internal UI; hosts render panels around it.

**Error handling:** unknown palette name → document default; document without
`palettes` → theme switcher hidden; document without `dossier` → dossier card
falls back to label + reading (today's panel); WebGL missing / schema mismatch →
existing refusals unchanged; localStorage access failure → preferences simply
not persisted.

## 3. Geometry & exploded view

### 3.1 BuildOptions additions

```ts
interface BuildOptions {
  cursor: number;          // existing
  exploded: number;        // existing, 0..1
  casing: "translucent" | "hidden";  // existing
  dataScaled: boolean;     // existing
  peel: number;            // NEW, 0..1
  layout: "wound" | "unrolled";      // NEW
}
```

- **Peel**: `drawnSweepDeg = clamp(360 × (1 − peel), 90, 360)`.
  Default `peel = 75/360` → **285°, bit-identical to today's `canSweep`** (test
  pins this). `peel = 0.25` → exactly a quarter of the circumference removed
  ("¼ Peel"). Wrap and crimp sweeps follow the can, as now. A quarter-shell
  minimum (90°) always remains; removing the casing entirely stays
  `casing: "hidden"`'s job. `CELL_GEOMETRY.canSweep` is replaced by
  `peelSweepDeg(peel)` (the constant becomes the default's derivation).
- **`breathe` is engine state, not a build option** — sine pulsation is applied
  as group transforms in the tick loop (zero rebuilds): each part group scales
  `1 + 0.012·sin(ωt + φᵢ)` about its own anchor with a small radial nudge,
  ω ≈ 2π/3.2 s.

### 3.2 Spring / tween interpolation

- `springStep(current, target, dtMs, {stiffness, damping})` — critically damped,
  settles ≈350 ms; pure, Node-tested.
- The engine loop advances `build.exploded` / `build.peel` toward their targets
  and calls `rebuild()` throttled: at most every **8 ms** and only when
  `|Δ| ≥ 1e-3`. Preset buttons therefore animate instead of snapping; a slider
  drag simply retargets.
- **Documented fallback** (risk register §9): if profiling shows rebuilds cannot
  sustain ≥30 fps, axial displacements (cap/vent/terminals/gasket/cid/bottom
  disc) move to engine-side group transforms — placements return base mesh +
  offset — leaving only the roll to rebuild. Not in v1.

### 3.3 New anatomy parts (16 → 19)

| id | Geometry | Axial explode key | Drawn (prismatic) |
|---|---|---|---|
| `gasket` | PP flat gasket: lathed ring under the cap plate, inside the crimp groove | `gasket` (up, slightly below cap) | yes |
| `cid_ptc` | CID scored disc + PTC bead drawn inside the boss under the terminal (grouped as one part, per spec) | `cidPtc` (between cap and terminal) | **no** — `drawn: false` + form-factor reason |
| `bottom_insulator` | hdf/PE disc on the can floor under the roll | `bottomInsulator` (down/out) | yes |

- New axial keys in `CELL_GEOMETRY.explode`, **all zero at rest** (assembled = declared winding).
- **Exhaust ports**: scored slots drawn as detail geometry on the cap mesh — not a part (no specs to card); described in the cap's dossier insight.
- Each part: honest card (no per-part measurement → refusal with reason), material entry (`PART_MATERIALS`), dossier block, anchor, `category`.
- Exhaustive churn list: `ANATOMY_PART_IDS`, schema part-id enum + `minItems/maxItems: 19`, `_build_parts` + `_PLAIN_TITLE` + `_unavailable`, fixture, materials, `geometry.test.ts` counts, `tests/test_cell_scene.py` counts, sample + manifest regen, prose: README ("19 anatomy parts"), CHANGELOG, history row, `battery3d.py` docstring, scene_prompt mapping table.

### 3.4 Unrolled layout

- `layout: "unrolled"` rebuilds the three ribbons as a **flat sandwich strip**:
  cathode / separator / anode stacked at their declared thicknesses ×
  `EXPLODE_STACK_GAIN`-style magnification, running horizontally.
- The 1.27 m electrode is compressed to stage width: the **compression factor is
  printed on screen** (same discipline as the film's magnification) and the
  strip's thicknesses ride the declared stack.
- Transition: crossfade opacity dip between builds (~180 ms) + camera tween — no
  vertex morph. Tabs stay attached at the strip ends; particles/film follow the
  strip bounds.
- Tests: strip within stage bounds, thicknesses = declared × gain, compression
  factor string present in `scaleNote`, tab attachment, envelope analog for the
  unrolled build.

### 3.5 HUD-facing readouts

- Explode slider millimetre indicator: at value `v`, show
  `axial: v × (terminalPos lift × unitMm) mm · radial: v × max(explode.radial) mm`
  — both derived from the document/`CELL_GEOMETRY`, labeled a view displacement.
- Quick actions: `[ASSEMBLED]` → exploded 0 · `[EXPLODED 100%]` → exploded 1 ·
  `[CUTAWAY · ¼ PEEL]` → peel 0.25 (toggle back to default) · `[BREATHE]` → toggle.

## 4. Annotation system (`annotation.ts`)

### 4.1 Layout solver (pure, Node-tested)

`layoutFlank(items, height) → { targetY, singleLine }`:
sort by projected `sy`; assign with hard `minGap` (two-row card height + 6);
one relaxation pass pulling each card toward its anchor **without breaking
order or gap**. Invariants (asserted):
(a) no two cards on a flank overlap vertically;
(b) `targetY` monotone in `sy` → **no crossings within a flank**;
(c) flanks partitioned by anchor x → no crossings between flanks;
(d) if `n × minGap > height − margins`, the flank degrades to single-line cards
    (row 2 hidden) rather than overlapping.
Anchor occlusion (`z > 1`, off-screen ±30 px) filtering stays as today.

### 4.2 Micro-callout cards (two rows)

- Row 1: **part name** (600 weight) + **key value + unit** (tabular).
- Row 2: **category** chip + **provenance dot** (colour = `theme.provenanceColors`,
  hover tooltip = provenance word + reason when unavailable).
- Escaped text everywhere (`escapeHtml`), `title` attribute carries full
  label — provenance — unit, as today.
- New optional `ScenePart.category` — producer taxonomy (all 19):

| Category | Parts |
|---|---|
| SHELL | can |
| INSULATION | wrap, gasket, bottom_insulator |
| SEAL | crimp |
| SAFETY | vent, cid_ptc |
| TERMINAL | cap, terminal_pos, terminal_neg |
| WINDING | mandrel |
| ELECTRODE | cathode_sheet, anode_sheet, tab_pos, tab_neg |
| SEPARATOR | separator |
| ELECTROLYTE | electrolyte |
| DEGRADATION | particles, sei_film |

### 4.3 Reticle & pulse

- Concentric target ring + tick marks at the projected anchor (SVG), replacing
  the two plain circles. **Pulse only on the active part**: `r = base +
  1.4·sin(2πt/1.6s)` modulated in the tick loop; inactive reticles static.

### 4.4 Selection treatment (engine)

- Unselected parts: `opacity → min(baseOpacity, 0.15)` and **desaturation**
  toward the palette's neutral (70% mix, pure helper in `annotation.ts`,
  testable with `mixHex`). Base colour/opacity stored per part (extension of the
  existing `baseEmissive` pattern) so clearing restores exactly.
- Selected part: emissive lift (existing `+0.6`) **plus a Fresnel rim** via
  `material.onBeforeCompile` injecting a rim term in the palette accent —
  confined to engine.ts ("the only file that knows three.js").
- `DIM_OPACITY` for badges/lines stays; mesh dimming is the new layer.

### 4.5 Click → pin + smooth frame

- Badge/mesh click pins inspection (existing semantics) and starts a **~500 ms
  eased camera tween**: spherical lerp of azimuth/elevation toward the part's
  anchor, distance preserved and clamped to current fit limits.
- `Frame Camera` button: full fit-to-part-bounds tween (~600 ms).
- Any `OrbitControls` user `start` event cancels an active camera tween.

## 5. Technical dossier

### 5.1 Document-carried (schema addition, optional per part)

```json
"dossier": {
  "latinTitle": "STRATUM ANODICUM",
  "subsystem":  "Graphite / Silicon Intercalation Host & Current Collector",
  "material":   "MCMB graphite / Si blend on 10 µm copper foil",
  "degradation":"Lithium plating under low temp/fast charge, …",
  "insight":    "SEI growth consumes cyclable lithium under high C-rate cycling. …",
  "specs": [
    { "label": "Anode coating", "value": "70", "unit": "µm", "tag": "typical" },
    { "label": "Cu foil",       "value": "10", "unit": "µm", "tag": "typical" },
    { "label": "Loading",       "value": "3.5–4.0", "unit": "mg/cm²", "tag": "typical" },
    { "label": "Porosity",      "value": "25–30", "unit": "%", "tag": "typical" },
    { "label": "Tortuosity",    "value": "2–3", "unit": "τ", "tag": "typical" },
    { "label": "Impedance contribution", "value": "not measured per part",
      "unit": "", "tag": "refusal" }
  ]
}
```

- `tag ∈ { typical, measured, derived, fitted, refusal }` — the two-tier policy
  the user approved. Producer table `_DOSSIER_TABLE` in `src/cell_scene.py`
  holds all 19 (content migrated from React's `PART_DOSSIERS`, enriched).
- Schema: `dossier` + `category` optional on part items; a Python test asserts
  dossier presence, spec-row tags, and one refusal row where the platform has
  nothing. One new disclosure sentence: typical rows are format-level, not this
  cell's datasheet.

### 5.2 Live diagnostics (composed at render time — `dossier.ts`)

Sources, in order of honesty: cell temperature (`series.temperatureC`, measured,
at cursor) · resistance growth (derived) · SOH + band label · SEI thickness nm
(`series.seiThicknessNm`, derived; **refusal when the √n split is unidentified**)
· LLI/LAM attribution (`physics.contribution*`, fitted; **only when the gate
passed**) · the part's own reading (existing, cursor-resolved). Rows that would
need per-part electrochemistry render the `refusal` rows from the document.

### 5.3 Rendering

- Engine HUD renders a **floating dossier card** in the stage overlay (all three
  hosts get it): header `latinTitle // LABEL`, spec table with tag chips, live
  block, insight paragraph, Frame Camera button.
- React keeps its richer right panel but **deletes `PART_DOSSIERS`** and renders
  from `spec.parts[].dossier` + readings — one source of truth.
- Streamlit page: unchanged cards (dossier arrives via the iframe's overlay).

## 6. Dual theme engine

### 6.1 Document shape

```jsonc
"theme":    { …today's SceneTheme… },          // default palette (back-compat)
"palettes": { "codex": {…SceneTheme…}, "obsidian": {…SceneTheme…} }  // optional
```

`SceneTheme` gains **optional** tokens: `name`, `variant: "dark"|"light"`,
`fonts: { display, mono }`, `accent2`. `setTheme(name)` →
`palettes[name] ?? theme`; unknown name → default; `palettes` absent → theme
switcher hidden. **Python test pins identical SOH/temperature band thresholds
across all palettes** — only colours may differ.

### 6.2 Palette values

| Token | codex (parchment) | obsidian (void) |
|---|---|---|
| background | `#F4EEDA` | `#070A10` |
| panel | `#E8DFCA` | `#0C101A` |
| text | `#2A2118` | `#E2E8F0` |
| muted | `#6B5B45` | `#94A3B8` |
| accent | `#2F4F6F` (ink blue) | `#38BDF8` (phosphor cyan) |
| accent2 | `#8A6A2F` (brass) | `#EAB308` (imperial gold) |
| grid | `#C9BCA0` | `#1B2436` |
| metal | `#7D7466` (aged iron) | `#CBD5E0` (titanium) |
| anodeColor | `#A8763F` | `#C89B6A` |
| cathodeColor | `#5F4B8B` | `#7B6BA8` |
| separatorColor | `#CFC5AD` | `#D9E2EC` |
| electrolyteColor | `#4A7FA5` | `#38BDF8` |
| seiColor | `#7A4F9E` | `#A78BFA` |
| sohBands | `#2F855A` / `#B7791F` / `#C53030` (same thresholds) | today's `#48bb78` / `#f6e05e` / `#fc8181` |
| temperatureBands | `#2B6CB0` / `#B7791F` / `#C53030` | `#4299e1` / `#ecc94b` / `#e53e3e` (today's) |
| provenanceColors | measured `#2F855A`, derived `#2F4F6F`, fitted `#6B46C1`, projected `#B7791F`, "" `#8A8172` | today's |
| fonts.display | `'Cinzel','EB Garamond',Georgia,serif` | `'Inter',system-ui,sans-serif` |
| fonts.mono | `'JetBrains Mono',ui-monospace,Menlo,monospace` | same |

### 6.3 Engine-side art direction (per `variant`)

- **Lighting**: codex — warm key `#FFF6E0`, exposure ≈1.15, env intensity ≈0.40,
  warm hemisphere fill; obsidian — today's cool rig (`#ffffff` key, cyan rim,
  exposure 1.05, env 0.55), ACES both.
- **Materials**: `PALETTE_MATERIAL_ADJUST = { dark: today, light: { metalness
  −0.1, roughness +0.12, envMapIntensity ×0.6 } }` in materials.ts (both rows
  test-readable); applied additively over `materialFor()`.
- **Chrome**: SVG drafting frame from palette tokens — margin rules, border
  ticks, scale ruler along the bottom, compass rose (ornate in codex, minimal
  reticle cross in obsidian). Chrome visibility follows the annotations toggle
  (`A` / `setAnnotations`); collapsing the HUD rail (`H`) hides only the rail.
- **Backdrop**: codex — procedural paper grain (runtime canvas noise tile,
  generated once, ~2 KB logic, no asset); obsidian — radial vignette (CSS
  gradient overlay). Both palette-driven, no hard-coded colours.
- **Fonts**: `<link rel=preconnect>` + Google Fonts CSS (`Cinzel:400,600`,
  `EB+Garamond`, `JetBrains+Mono:400,500,600`, `Inter`) with
  `display=swap` added to `frontend/index.html`, `app/static/cell_scene/index.html`,
  and the Streamlit iframe HTML (`app/_scene_view.py`). Offline falls back to
  the stacks above (Georgia / ui-monospace) — correct, just less ornate.

### 6.4 Switching & persistence

`CellSceneHandle.setTheme(name)` + mount option `theme?: string`. Choice and
HUD-collapsed state persist to `localStorage["cell-scene.pref"]` per origin
(try/catch; failure = no persistence, never an error). Hosts may pass an
explicit theme to override.

## 7. Cockpit HUD (`hud.ts`)

Mounted inside the stage container (absolute overlay, `pointer-events` scoped):

| Group | Contents |
|---|---|
| Explode | slider 0–100%, mm readout (§3.5), `[ASSEMBLED]` `[EXPLODED 100%]` |
| Peel | slider 0–100% with a `¼` detent tick, `[CUTAWAY · ¼ PEEL]` |
| Breathe | `[BREATHE]` toggle (§3.1) |
| Lifecycle | playback bar: scrub 0…N, `[▶/⏸]` (Space), cycle · SOH · projected flag, `Today` |
| Layout | `[WOUND]` `[UNROLLED]` (§3.4) |
| Camera | `[ISO]` `[PLAN]` `[SECTION]` `[UNROLLED*]` — *UNROLLED doubles as layout+camera |
| Theme | `[CODEX]` `[OBSIDIAN]` (hidden without `palettes`) |
| Telemetry | FPS (EMA over rAF deltas), draw calls + triangles (`renderer.info.render`, 4 Hz), vertex sum (from painted geometry) |
| Hotkeys | `<kbd>` micro-capsules on the owning control |

**Camera presets** (all eased ~600 ms, cancelled by user drag):

| Preset | Camera | Also sets |
|---|---|---|
| ISO | azimuth 45°, elevation 22°, fit distance | layout wound |
| PLAN | elevation 88°, target raised to cell mid-height | — |
| SECTION | side-on into the cut-away opening, elevation 8° | peel 0.25, exploded 0.35 |
| UNROLLED | front-on to the strip, fit to strip bounds | layout unrolled |

**Hotkeys — scoped to the stage**, never page-global: a key fires only when the
stage container contains `document.activeElement` (it has `tabIndex=0` and takes
focus on pointer-down) **or** the pointer is currently over the stage; the
handler `preventDefault`s only the keys it owns (`Space` would otherwise scroll
the page):

| Key | Action |
|---|---|
| `Space` | play / pause |
| `E` | explode toggle 0 ↔ 100% (spring) |
| `C` | ¼-cutaway toggle |
| `A` | annotations (+ chrome) toggle |
| `H` | collapse / restore HUD rail |
| `Esc` | clear pinned inspection |

Host controls remain functional (same handle API) — HUD is additive in v1;
slimming host widgets is backlog.

## 8. Testing & verification

**Node (76 → ~110):** layout solver invariants (a–d) · `peelSweepDeg` incl.
*default = exactly 285°* · `springStep` convergence/settling · desaturate/dim
helpers · palette lookup + unknown-name fallback · 3 new placements + envelope
at explode 0/0.5/1 + radial datum zeros at rest · unrolled strip bounds,
thicknesses, compression-factor string, tab attachment · category taxonomy
completeness (19/19) · dossier block presence in fixture (JS side) ·
existing ribbon-tiling/honesty/philosophy tests untouched.

**Python (101 → ~125):** schema enum 19, `minItems/maxItems: 19` · `category`
on all parts · dossier shape: required fields, spec-row tag enum, ≥1 refusal row
where applicable, typical-tagged rows present · palettes: band-threshold
equality across `theme`/`codex`/`obsidian`, hex validity, optional-token shape ·
fixture + sample regeneration consistency · `battery3d` page tests unaffected
(its widgets remain).

**Gates per phase:** `npm run test:scene` · `npm run lint` · `npm run build` ·
4 Python scene suites · (phase ⑥) `npm run build:scene` + sample export +
bundle test · pyright · `mkdocs build --strict`.

**Manual (standalone checklist additions):** default peel renders identically to
the previous build · preset buttons spring rather than snap · hotkeys work with
stage focus and never leak to the page · theme switch reskins stage+HUD+dossier
and the legend agrees · unrolled prints its compression factor · dossier refusal
rows read as refusals.

## 9. Risks & fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Spring rebuild cost (19 parts, ~30–45 fps worst case) | Throttle (8 ms, Δ≥1e-3); **fallback**: axial moves to group transforms (§3.2), only roll rebuilds |
| `onBeforeCompile` rim vs three.js minors | Kept in engine.ts; `render.test.ts` still pins the three allowed console warnings; rim degrades to emissive lift if injection fails (try/catch) |
| CDN fonts offline | System fallback stacks declared in palette; page remains correct |
| 19 cards crowding small stages | Solver single-line degradation (§4.1 d) |
| Schema growth | Additive-optional only; no version bump; old docs render as today |
| Streamlit duplication (page widgets + HUD) | Additive v1; documented; slimming is backlog |

## 10. Phases (each ends green, committed)

1. **Geometry** — 3 parts, peel, unrolled, explode keys, schema 19, producer, fixture, counts, sample/manifest. **Green: `7128d56`, `2761453`, `79d4a05`.**
2. **Annotation** — `annotation.ts`, solver + invariants, two-row cards, category, pulse, smooth camera frame. **Green: `2e4290a`, `c2be1e1`, `c35519b`.**
3. **Dossier** — document block + producer table + schema + engine floating card + React `PART_DOSSIERS` deletion + refusal rows. **Green: `7a1e143`, `f430528`, `44d4443`.**
4. **Palettes** — schema tokens, two palettes producer-side, threshold-equality test, `setTheme`, lighting/material adjust, chrome, grain/vignette, host font links. **Green: `8b204ba`, `38a4612`.**
5. **HUD** — `hud.ts` rail, playback, presets, hotkeys, telemetry, breathe, persistence. **Green: `096f8ad`.**
6. **Artifacts & docs** — bundle/sample/manifest, all suites + pyright + lint + mkdocs strict; README, CHANGELOG, docs/history row, scene_prompt mapping + contract, static README checklist, `battery3d.py` prose ("nineteen"), METHODOLOGY test counts.

**Bundle budget:** 595 kB → ≈640 kB raw (hud/annotation/dossier/palettes ≈ +30–45 kB over today's engine additions); manifest re-hashed at phase 6 and any engine-touching phase thereafter.
