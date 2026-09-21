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

### Added

- **A 3D cell scene, as a portable capability rather than a widget.** One framework-free renderer plus one versioned
  JSON document (`docs/cell_scene.schema.json`, built by `src/cell_scene.py`), consumed unchanged by four hosts: the
  Streamlit page *Analyse → Battery 3D*, the React SPA's *Cell 3D* tab, a standalone HTML page served at
  `/scene/index.html` (Streamlit's static route and the API's new `/scene` mount), and a new authenticated REST
  endpoint `GET /cells/{id}/scene` (`?horizon_cycles=` bounds the projection; `0` asks for the measured record only).
  Every drawn part is bound to a measurement the platform already computes and tagged measured / derived / fitted /
  projected; a part no source can speak to is drawn *with a reason* instead of a zero. The SEI film and particle loss
  are drawn only when the two fitted fade channels are separable, and the timeline's future half is the platform's own
  hierarchical forecast gated by its existing per-cell routing — a refused route yields no future at all. The renderer's
  geometry is DOM-free and unit-tested by Node's own runner (`npm run test:scene`, 53 tests); the renderer bundle is
  built by `npm run build:scene` into `app/static/cell_scene/` and committed, because Streamlit serves it to a browser
  that has no Node — so a manifest records the digest of the bundle and of every source file it was built from, and
  `tests/test_cell_scene_bundle.py` fails on a stale or hand-edited artifact instead of shipping a scene that
  disagrees with its own source.
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
