# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the
public API surface defined in [`docs/api_stability.md`](docs/api_stability.md).

**Not in this file, on purpose:** changed *numbers*. Retraining the same model on
the same data can legitimately move a metric (a feature-version bump, a loader
fix, a corrected evaluation rule), and a movement like that is recorded in the
experiment registry and in `FEATURE_VERSION`, with the accuracy gate
(`batlab.validation.metric_gate`) deciding whether it is explained. Rubrics for
those changes live in the docs they affect; this file records interface changes.

## [Unreleased]

- **Bloom on the emissive gauge, via `EffectComposer`.** The stage now renders
  `RenderPass → UnrealBloomPass → OutputPass`: linear HDR into MSAA half-float targets, the
  highlights lifted once, ACES applied once on the way out (the chain is pinned in
  `render.test.ts`), with per-palette tuning — Obsidian 0.35/0.45/0.5 so the gauge band and
  terminal speculars glow at rest, Codex 0.16/0.3/0.9 so the lit parchment itself never
  blooms. Shipping it surfaced three latent defects, each now fixed: `extrudeOpen` drew its
  two-sided shell from shared vertices, cancelling every face against its mirror in
  `finalize()` so the shader saw `normalize(vec3(0))` = NaN fragments that poisoned the
  half-float frame (each shell now owns its rings; new test: every vertex a visible triangle
  touches carries a unit normal); the stage vignette and paper grain were assigned to
  `innerHTML` as raw CSS — literal stray text on the stage — and now paint through
  `style.cssText`; and `scene.background` rode through the composer's tone mapping, landing
  the palette ~1 EV too dark, so the void is now the canvas element's own CSS surface and
  renders the palette byte-exact on both palettes. Bundle re-hashed at 639.1 kB (~164 kB
  gzipped); node tests 109 → 112.
- **The scene becomes a codex: nineteen parts, aimed peel, document-carried dossiers, two palettes, a cockpit.** Three
  new anatomy parts — `gasket`, `cid_ptc` (the CID/PTC safety pair) and `bottom_insulator` — take the
  mesh/card/schema-enum bijection to **19** on every side, each with its plain-language card, material entry and axial
  explode lift. The cut-away peel became a build option (`peel`): the default still draws the historic 285° sweep
  byte-for-byte, the `CUTAWAY · ¼ PEEL` action lands on the quarter detent and back, and `layout: "unrolled"` lays the
  three ribbons out as one flat sandwich that **prints what the compression cost** — wound output stays byte-identical
  under any view options (test-pinned). Callouts became two rows — value over provenance dot and unit — placed by a
  pure flank solver that carries no state between frames, and every part's **dossier moved into the document**
  (`dossier` block): Latin/engineering name, subsystem, material, failure mode, camera focus, at least four spec rows
  each tagged `typical`/`measured`/`derived`/`fitted`, and refusal rows carrying a reason instead of a number, shown as
  a card floating over the stage. The document also carries **two palettes** (`palettes.codex`, `palettes.obsidian`)
  with the SOH band thresholds pinned identical across them (Python-tested): `setTheme()` repaints stage, lights,
  materials, chrome, fonts, card and HUD together and persists the reader's choice, while the bands never move. A
  **cockpit HUD** rides the stage's bottom edge — spring-damped explode/peel with a millimetre readout, cutaway,
  breathe, annotations, lifecycle transport, layout switches, camera presets (`iso`/`plan`/`section`/`unrolled`), 4 Hz
  renderer telemetry and a collapse pill — with `Space`/`E`/`C`/`A`/`H`/`Esc` scoped to the stage: they act only while
  it has focus or hover. Node tests 76 → 108 (peel/unroll shapes, flank solver, dossier composer, palette contract, HUD
  mount); Python 108 across the four scene suites (dossier block, palette contract); bundle re-hashed at 618.4 kB
  (~159 kB gzipped).
- **Museum-grade dynamic SVG leader-line annotation overlay and technical component dossier.** Replaces cluttered
  on-mesh floating tags with an architectural drafting annotation system. Each drawn component's anchor is projected
  into viewport space with a precision concentric target reticle (`<circle>`), connecting via orthogonal/elbow dashed
  leader lines (`<polyline>`) to neatly stacked callout badges placed along the left and right viewport flanks with
  vertical collision avoidance. Hovering or clicking a badge or mesh highlights the active leader line and target
  reticle with accent glow, dims non-selected callouts, and opens an engineering **Inspected Component Technical Dossier**
  in `CellSceneView.tsx` featuring Latin/engineering dual naming (e.g. *STRATUM CATHODICUM*, *VALVULA SALUTIS*,
  *MEMBRANA SEI*), subsystem classifications, material specifications, failure modes, telemetry values with provenance
  tags, and camera focus controls. The badges are pinned at the stage's own left/right margins (`BADGE_MARGIN` +
  `BADGE_WIDTH`), so every leader line terminates at the same fixed column edge whatever the camera does, and the
  dossier in the right panel is now the *only* place a part is read about — the SPA's bottom row of part cards is
  gone with the labels it used to sit next to. Internal winding anchors now reside on their true physical layer
  surfaces.
- **The winding is concentric, and it has a core.** `CELL_GEOMETRY.explode.radial` now declares an independent
  radial offset per drawn member (`casing 0, mandrel 0, anode 0.25, separator 0.70, cathode 0.95` mm at full
  explode), so the exploded view separates the stack as concentric shells instead of one pushed bundle: the can and
  mandrel hold their datum while the foils peel outward, each lane's gap is the difference between cumulative
  offsets with the air folded into one turn's advance (so the three ribbons still tile a lap exactly), and the
  drawn outer radius stays inside the 18650 envelope by construction at every explode position — 38 → 3 laps,
  ≈8.55 mm against the 8.75 mm envelope — while the assembled state (`exploded = 0`) is exactly the document's
  declared winding. The core is anatomy now: a sixteenth part, `mandrel`, drawn at the declared 4.0 mm diameter
  over the roll's own height, with its mesh, its plain-language card, its material entry and its dossier — the
  mesh/card/schema-enum bijection held at 16 on every side. Node tests 74 → 76 (five concentric members, offsets
  ordered at explode and zero at rest, air between mandrel and anode, the mandrel's declared diameter); Python 101
  across the four scene suites (67 scene / 13 API / 9 bundle / 12 page) after the schema's `parts` array moved to
  `minItems`/`maxItems` 16; bundle 593.4 → 595.4 kB, sample and manifest re-hashed.
- **A 3D cell scene, as a portable capability rather than a widget.** One framework-free renderer plus one versioned
  JSON document (`docs/cell_scene.schema.json`, built by `src/cell_scene.py`), consumed unchanged by four hosts: the
  Streamlit page *Analyse → Battery 3D*, the React SPA's *Cell 3D* tab, a standalone HTML page served at
  `/scene/index.html` (Streamlit's static route and the API's new `/scene` mount), and a new authenticated REST
  endpoint `GET /cells/{id}/scene` (`?horizon_cycles=` bounds the projection; `0` asks for the measured record only).
  Every drawn part is bound to a measurement the platform already computes and tagged measured / derived / fitted /
  projected; a part no source can speak to is drawn *with a reason* instead of a zero. The film is reported as a derived
  **thickness** in nanometres and particle loss as the linear term's share of fade, and both are drawn only when the two
  fitted fade channels are separable; the timeline's future half is the platform's own hierarchical forecast gated by its
  existing per-cell routing — a refused route yields no future at all. The renderer's geometry is DOM-free and
  unit-tested by Node's own runner (`npm run test:scene`, 76 tests); the renderer bundle is
  built by `npm run build:scene` into `app/static/cell_scene/` and committed, because Streamlit serves it to a browser
  that has no Node — so a manifest records the digest of the bundle and of every source file it was built from, and
  `tests/test_cell_scene_bundle.py` fails on a stale or hand-edited artifact instead of shipping a scene that
  disagrees with its own source.
- **The top of the cell is formed, not stacked.** A new declared `topAssembly` block (cap plate and boss, vent disc,
  terminal button and stud, crimp bead, heat-shrink wrap — every figure millimetres with stated provenance, typical for
  the format rather than this cell's datasheet and declared so) drives the drawn top: the cap is one lathed profile
  (plate, boss and the groove the crimping die rolls the wall into), the vent sits *in* the boss, the button stands on
  it, the crimp bead is a lip rolled proud of the can wall, and the jacket wraps to within 1.2 mm of each end — opened
  by the same cut-away that opens the can, so it cannot hide the winding it protects. The tabs are leads now: each
  leaves the coil's own outermost turn and rises, bent over the cap's thickness, to the terminal it feeds, instead of a
  box floating beside the roll. Two new anatomy parts (`wrap`, `crimp`, taking the scene to 15, with plain-language
  titles on every card), material entries for both, and shape tests pinning that the drawn terminal and vent *equal* the declared
  millimetres, that bead/jacket/can stack outward in that order, that nothing drawn leaves the cell's envelope at any
  explode position, and that a prismatic scene draws its own flat wrap and welded-rim crimp. Node tests 69 → 74;
  bundle 590.5 → 593.4 kB (149.7 → 150.6 kB gzipped), manifest re-hashed.
- **The SEI film is a derived thickness in nanometres, with its display magnification disclosed separately.** The film
  used to be the one drawn *size* that was not a size — a band of the roll's clearance scaled for legibility and
  disclosed as a metaphor. The quantity behind it is fitted, so it is now converted: the lithium the fitted $\sqrt{n}$
  term puts in the film, at the modelled phase's molar volume (Li₂CO₃, 35.02 cm³/mol) and spread over the coated anode
  area the declared winding implies (1475 cm²), is **~80 nm per 1% of initial capacity lost to lithium inventory** —
  1647 nm at 20.8% LLI, 2284 nm at 27.1%. Every constant is tagged (`physical.film.assumptions` + `provenance`), the
  chain reduces to the single factor `nmPerPctLli` a reader can recompute by hand, and the refusal is explicit: a cell
  whose $\sqrt{n}$ channel is not identified reports the factor and withholds the thickness (no number on the card),
  and a stacked prismatic cell — which has no wound electrode length to take an area from — is refused with its reason.
  The drawing is kept as a separate, separately stated claim: `physical.film.display` gives the drawn band (fixed by
  the roll's radial clearance and by what is visible at a 65 mm scale) and the magnification it implies (~189× at the
  top of B0018's scale, ~131× on the film's growth), the scale's top is a fixed 30% of initial capacity rather than
  each record's own maximum so two cells' films stay comparable, and the new `series.seiThicknessNm` is what the
  geometry follows — the fitted share can no longer move the drawn film, which a test now pins both ways.
- **A host contract for the 3D scene, so a host's panels cannot disagree with the scene beside them.**
  `CellSceneHandle.parts()` returns every part's reading **at the scene's own cursor** (id, label, value, unit,
  provenance, availability, a `carried` flag for a cycle with no measurement, and the document's own sentence about
  the part) — the four hosts' part tables are reading it, so scrubbing the cursor moves the cards with the cylinders
  instead of leaving today's numbers next to a cylinder drawn at cycle 15. `MountOptions` are applied **at mount**, not
  only on a later `update()` (a scene opens on the cursor its host asked for), `play()` returns `false` when there is
  no life left to play and replays from the first frame rather than doing nothing, and the new `onPlaybackEnd`
  callback tells a host that playback reached the end of the timeline — the button is the host's, so only the host can
  put its label back.

### Changed

- **The 3D cell view is dimensioned rather than decorated.** A new `physical` block in the scene document declares the cell's
  own millimetres — an 18.4 × 65.0 mm envelope, a 0.25 mm can wall, 0.2 mm of roll clearance, and a 0.175 mm foil-to-foil
  stack (10 µm copper, 70 µm anode, 20 µm separator, 60 µm cathode, 15 µm aluminium) — with the provenance of **every**
  field (`format-standard`, `typical`, `assumed`, `derived`) and a `schematic` list naming what is *not* to a datasheet.
  The renderer derives the winding from it instead of drawing its own idea of one: the pitch is the stack, the turn count is
  what that pitch needs to fill the envelope (38 turns, 17.3 mm of 17.5 mm), and the electrode length those two imply
  (1.27 m) is reported for checking. The three rolled sheets tile one turn's advance exactly, so the drawn roll is a
  *filled* roll — the previous schematic filled 52% of the cross-section and read as a spring. The exploded view now
  un-winds the rolling (×8 stack magnification, 38 laps down to 4) and keeps the roll inside its envelope by construction
  at every explode position; the SEI film, the last legibility-scaled *size*, is drawn as a sheath on the anode's own
  surface where an SEI forms. Ribbons are memoised, so a cursor-only rebuild costs 7 ms rather than 13 ms. A document
  without the block still renders, on the same fallback figures, and reports that they are not this cell's.
- **The 3D view is shaded as an object.** Filmic tone mapping, a procedural room environment for the metal surfaces, a
  material per anatomy part (steel, nickel, aluminium, copper, graphite, membrane, electrolyte) replacing the single
  metalness string test that made a can, a membrane and electrolyte look alike, a shadow-casting key light with a fitted
  shadow camera, and a camera framed from the part's own bounds so a prismatic cell is framed as deliberately as a
  cylinder. `PCFSoftShadowMap` is deliberately *not* used: three r186 still exports it with its implementation removed, so
  naming it silently renders a different filter — a test now asserts the renderer never does.
- **Physics calibration is part of the library, not of the demo application.**
  `batlab.features.physics_calibration` (with `src/physics_calibration.py` kept as a thin
  re-export shim that registers the app's mechanism classifier). `build_features()` used to import
  that module opportunistically from the app's `src/`, which made a feature column — and therefore
  the model's numbers — depend on the caller's `sys.path`: the same four NASA cells scored SOH R²
  **0.9580** in a pip-installed process and **0.9471** with the app importable. Eligibility now
  reads the frame's own declared `source`/`chemistry` attrs (`ANCHOR_PARAM_SETS`), so the
  population is a property of the data, and `register_anchor_param_set()` lets a caller extend it.
  PyBaMM stays an optional extra (`battery-lab[physics]`) and now affects only the display-only
  `physics_spm_capacity_ah` column.
- **`FEATURE_VERSION` is `v13-features-owned-physics`.** A deliberate, value-changing bump: `v12`
  numbers may have come from either population. It invalidates the fold cache and every cached
  bundle, and supersedes the earlier baselines — the measured deltas are recorded in
  [`docs/performance.md`](docs/performance.md). There is now one number per environment.
- **The demo application's loader declares provenance** (`src/data_loader.build_battery()` sets
  `df.attrs["source"]` / `["chemistry"]` from the app's `ChemistryProfile` classifier, via the new
  profile field `dataset_source`). The batlab schema always required these attrs; the omission was
  invisible until the physics block began gating on them.

### Fixed

- **A fleet's trivial baseline can no longer vanish into `None`.** A blank measurement row in a
  per-cycle summary (Severson's `S-b1c0` cycle 11 and `S-b1c18` cycle 39 carry an empty capacity
  column, so `soh_pct` is blank with it) made `baseline_lco_r2()`'s least-squares fit raise —
  sklearn's `LinearRegression` rejects a NaN target while `r2_score` returns NaN for one without
  raising — and the app's defensive `except` recorded `None`. On the 46-cell Severson fleet that
  deleted the denominator of every "+X over the trivial baseline" claim, silently. Both trivial
  baselines (`baseline_lco_r2`, `rul_formula_baseline_lco`) now select the rows they can score,
  report what they set aside (`n_nonfinite_target_rows` / `n_nonfinite_rows_excluded`) and why
  (per-fold `note`), and never return a non-finite headline (`_safe_r2` rejects non-finite input).
  They also accept the `{"cycles": df}` cell shape `run_lco()` accepts, so one dict can be handed
  to both — which is what the metric gate now does. Severson's restored floor is **−0.328**, all
  46 folds scored; no published baseline moved (NASA's `0.6030539120231699` and the CI fixture
  fleet's value are both bit-identical — see [`docs/performance.md`](docs/performance.md)).
- **A failed baseline is disclosed, not swallowed.** `app/_data.py` and the upload path in
  `app/_pages/import_page.py` record the exception text in `metrics["baseline_soh_r2_error"]` /
  `["rul_formula_baseline_r2_error"]` (on the bundle *and* in the registry's `lco_metrics`), and
  convert a non-finite baseline to `None` at the seam rather than storing a NaN the UI would
  render as a number.
- **`check_metric()` no longer passes on a NaN observation.** NaN compares False against every
  floor, ceiling and tolerance, so one reached the gate as a silent pass; it is now treated
  exactly like `None` (not evaluable), which is what rule 3 already said it meant.
- **The harness reports its trend baseline per cell.** `lco_trend_per_cell` read a key the trivial
  baseline does not produce (`r2`), so it was `None` for every cell of every fleet. The counts
  behind the floor (`n_folds_scored` / `n_folds_skipped` / `n_nonfinite_target_rows`) now travel
  with it.

## [0.2.0] — 2026-09-19

The first release shaped as a distributable library rather than a repository you
clone: a typed public API, real entry points, a command line, and a wheel that is
tested before it is published.

### Added

- **The package is installable and named for what is actually free on PyPI.**
  The distribution is `battery-lab` (`pip install battery-lab`); PyPI's `batlab`
  is Lexcelon's unrelated Batlab V1.0 hardware library. The import package is
  unchanged: `import batlab`.
- **Three documented entry points**, so the common path does not require knowing
  the module layout:
  - `batlab.load("nasa")` → `{cell_id: DataFrame}` in the standardized schema.
  - `batlab.benchmark(cells)` → leave-cell-out metrics (alias of
    `batlab.validation.run_lco`).
  - `batlab.validate(cells, model=...)` → grade *any* forecaster with the six
    harness checks (alias of `batlab.harness.validate_forecaster`).
- **A command line**: `python -m batlab` / the `batlab` console script, with
  `version`, `cite`, `datasets` and `benchmark` (`--dataset nasa --out report.json`,
  or an explicit `--loader module:function`). See [`docs/cli.md`](docs/cli.md).
- **Typed result schemas** (`batlab.results`): `LcoResult`, `QuantileLcoResult`,
  `FoldCacheSummary`, `ConfidenceIntervals`, `CalibrationReplayResult`, plus the
  `as_lco()` / `as_quantile_lco()` typed-view bridges. A test asserts every
  declared field is produced by a real run and names any field a real run
  produces that the schema omits.
- **PEP 561 marker** (`batlab/py.typed`), so consumers' type checkers use the
  library's inline annotations.
- **A written, tested deprecation policy** ([`docs/api_stability.md`](docs/api_stability.md))
  with its mechanism in `batlab._deprecation`; a deprecated name keeps working,
  warns with `since`/`removed_in`/`alternative`, and is removable in one commit
  on the announced version.
- **`docs/quickstart.md`** — ten minutes from `pip install battery-lab` to a
  number, starting from the wheel rather than a clone.
- **`docs/performance.md`** — the boot-layer split and the leave-cell-out fold
  cache, with measured numbers and the profiling scripts.
- **A wheel-install CI matrix** (`wheel` job, Linux/macOS/Windows × 3.10–3.13)
  that builds the sdist and wheel, installs the *wheel* into a clean venv, runs
  the quickstart and the CLI, and imports the installed package from outside the
  checkout. **A release workflow** (`.github/workflows/release.yml`) builds on a
  `v*` tag, runs `twine check`, publishes to PyPI when `PYPI_API_TOKEN` is set,
  and attaches the artifacts to the GitHub release otherwise.
- **Leave-cell-out fold cache** (`batlab.validation.fold_cache`): a completed
  fold is keyed on per-cell content digests, cell id, `FEATURE_VERSION`,
  `GBRT_PARAMS`, seed, numeric-stack versions and a code version, then replayed
  instead of refitted. Measured on the 46-cell Severson fleet: a repeated run
  replays 46/46 folds in 2.7 s instead of ~710 s of fits, and a run killed after
  fold 44 resumes with 44 reused. Opt out with `BATLAB_LCO_CACHE=off`; the
  replication recompute and the metric gate opt out by design. The key gained a
  `feature_inputs` component (so folds fitted with the optional physics block
  cannot be replayed without it), which means an existing cache is recomputed
  once — correct, and deliberately not silent.
- **Deferrable boot layers** (`BATLAB_BOOT_LAYERS=background|eager|off`): the
  application serves a core bundle (features + GBRT + predictions) and completes
  the validation, forecasting and calibration layers on a daemon thread,
  disclosing them as *pending* rather than as a failed measurement.
- **Censored-data survival readout** (`batlab.validation.survival`) and the
  **hierarchical partial-pooling forecaster** (`batlab.models.hierarchical`,
  shared with the validation harness so the graded and served estimators cannot
  drift), now routed per cell by `regime_reliability()` verdicts.
- **Severson fleet expanded to all 46 batch-1 records** (from 12), with the five
  cells that never reach 80% kept as right-censored observations instead of being
  discarded. LCO/prospective/hierarchical/ensemble numbers were re-measured on
  the 46-cell population; pre-expansion rows remain in the registry.
- **Docs enforcement**: a test asserting every MkDocs nav entry exists, every
  top-level docs page is reachable from the nav, and every public function/class
  carries a docstring. The docs job already ran `mkdocs build --strict`; that
  gate could not see an orphaned page, which is what the new tests cover.

### Changed

- **`batlab.__version__` is `0.2.0`**, and the package metadata is no longer
  `Development Status :: 3 - Alpha` — it is `4 - Beta`, with the 3.10–3.13
  classifiers and an explicit supported-platform claim.
- **`import batlab` no longer pulls in pandas, scikit-learn or a
  network-touching loader**; submodules resolve lazily on attribute access.
- **`run_lco()` / `run_lco_quantiles()` return plain `dict`s with a documented
  schema** rather than an opaque one. Their return *type* is `dict[str, Any]`,
  not a `TypedDict` — deliberately: callers legitimately extend an LCO result
  (the manifest exporter adds its own keys), which a closed-key TypedDict would
  have made a type error in library and app code alike. Use `as_lco()` at a call
  site to get the typing without changing the object.
- **The React SPA is built, linted (oxlint + `jsx-a11y`) and a11y-checked in CI,
  and served by the API at `/app`** (`mount_spa()`, conditional so an unbuilt
  checkout still boots), instead of sitting untested beside the Streamlit app.
- **One onboarding gate instead of three** in the demo app, and a first-run
  landing fleet whose RUL is actually populated (Zhu 2022 → Severson → NASA →
  synthetic), with the calibrated Q10–Q90 band drawn on the hero and the
  validation prose behind a "Why this number?" disclosure.

### Fixed

- **A number could silently depend on the Python path.** `build_features()`
  opportunistically imports `physics_calibration` — a module of the demo app's
  `src/`, not of the library — so the identical call returns SOH R² **0.9580** in
  a fresh process and **0.9471** when `src/` is importable (four NASA cells). The
  block is still optional, but it is no longer invisible: `build_features()`
  records `attrs["physics_features"]`, `run_lco()` reports `physics_features`
  (true only when every cell carried it), the fold cache keys on it so the two
  populations can never replay each other's folds, and `METHODOLOGY.md` §3 states
  the measured difference. See `docs/api_stability.md`.

- **Loaders no longer resolve raw data relative to the module file.** In a clone
  that is `data/raw`; in a wheel it was `site-packages/…/data/raw` — not the
  user's directory, possibly unwritable, and polluted by every download. All five
  loaders now resolve through `batlab.datasets._paths`:
  `BATLAB_DATA_DIR` → a checkout's `data/raw` → a per-user cache
  (`%LOCALAPPDATA%/batlab/data/raw`, `$XDG_CACHE_HOME/batlab/data/raw`, or
  `~/.cache/batlab/data/raw`).

- **`run_lco` no longer rebuilds every cell's features per fold on the boot
  path** — the validation layer is passed the frames it already has
  (`featured=`), which took NASA's validation layer from 7.4 s to 1.6 s.
- **Quantile calibration no longer spends ~13 minutes proving it cannot measure
  anything.** Every interval metric is computed on observed-label rows only, so
  on a fleet where no cell reaches end-of-life in-window the fitted quantile folds
  could only return not-evaluable placeholders. That result is now returned
  without fitting (2.0 s instead of ~13 min on 46 cells), and a test asserts the
  short-circuited dict is identical to the fitted one on the same fleet.
- **A deployment banner claimed "No auth · session-scoped uploads · data not
  persisted" on every authenticated screen**, which was false on every page that
  could render it. It now states what is actually true.
- **Two accessibility defects in the SPA** (a label/select association and a
  stray `autoFocus`), found by the new `jsx-a11y` lint gate.

### Removed

- Nothing. No public name was removed or renamed in this release, which is why
  there is no `Deprecated` section yet.

[Unreleased]: https://github.com/seyedali1996lb-svg/battery-intelligence-platform/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/seyedali1996lb-svg/battery-intelligence-platform/releases/tag/v0.2.0
