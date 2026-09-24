# Boot layers and fold caching — what a load actually computes

Two mechanisms keep this platform's startup and its repeat validation inexpensive without changing a single reported number:

- **Boot layers** — a fleet's work is split so the first paint waits only for what it needs.
- **The leave-cell-out fold cache** — a completed validation fold is replayed instead of refitted, and can never return a different answer than the refit would have.

Both are opt-out-able, both disclose what they did, and neither is allowed to turn "not measured yet" or "not measurable" into a number.

---

## Boot layers

`train_and_predict(..., defer_layers=True)` — what the demo app calls through `load_everything()` — separates a fleet's training into a **core layer** and three named layers:

| Layer | What it computes | Why it is deferrable |
|---|---|---|
| **core** | engineered features, the GBRT (SOH + RUL point/quantile heads), served predictions | the first paint needs exactly this |
| **validation** | leave-cell-out GBRT, its trivial baselines, the domain-validity envelope | it *measures* the model; it does not change what the model serves right now |
| **forecast** | hierarchical partial-pooling, the survival readout, per-cell regime routing | it adds answers alongside the core model, it does not replace it |
| **calibration** | quantile LCO coverage and the pooled conformal correction E* | its output *widens* intervals that are already served, once it completes |

Under `BATLAB_BOOT_LAYERS=background` (the default) the core bundle is returned immediately and the three layers finish on a daemon thread (`boot-layers-<fleet>`). One completion step then re-serves the frames (so the conformal widening and the hierarchical columns reach the served rows), logs the run to the experiment registry, and rewrites the bundle cache — so a process killed mid-layer leaves **no half-validated bundle cached and no registry row missing its evidence**.

- `eager` — inline, the pre-2026-09 behaviour, for scripts and tests that need a complete bundle, and therefore a complete registry row, when the call returns.
- `off` — compute nothing; the bundle keeps its placeholders and the app says the layers are off.

### Pending is disclosed, never rendered as a failure

While the layers run, every layer metric is an explicit *pending* placeholder — scalars `None`, container metrics `{}`/`[]` — and `metrics["boot_layers"]` reads `"pending"`. The app shows "Validation layers still computing for: …", and the hero says *Validation pending / awaiting leave-cell-out validation / coverage measurement still running* instead of "Not calibrated". A missing measurement and a bad one are different facts, and they are rendered differently.

A cached core-only bundle is retried on the next boot from the features cache; where that cache is absent the layer records `skipped` and the app keeps serving the pending state rather than silently upgrading itself to a claim it did not compute.

### One duplicate fit, removed on the way

`run_lco()` is called with `featured=raw_fdfs` on the boot path. Without it the harness rebuilds every cell's features — including the PyBaMM-backed physics calibration — a second time per fold, which is pure duplicate work on frames that were built moments earlier. On the 4-cell NASA fleet, that one argument took the validation layer from **7.4 s to 1.6 s**.

---

## The leave-cell-out fold cache

A leave-cell-out fold is a **pure function** of (the pool's cells, their feature frames, the model configuration, the seed) — the same property `batlab._parallel` already relies on to run folds concurrently ("each fold is seeded independently, the fold function is pure"). Nothing about a fold's result depends on when, or how often, it is computed. So `batlab.validation.fold_cache` writes each completed fold to disk as it finishes and replays it when the same configuration comes back.

Measured on the real **46-cell Severson fleet** (35,822 feature rows, 92 exact-splitter GBRT fits at ~53 s each):

| Run | Folds reused | Folds fitted | Wall clock |
|---|---|---|---|
| cold | 0 / 46 | 46 | ~710 s (clean, 8 workers) |
| repeated boot | **46 / 46** | **0** | **2.7 s** |
| killed after fold 44 | 44 / 46 | 2 | 106.7 s |

Identical numbers every way — `soh_r2 = 0.9920`, `rul_r2 = nan`, the same per-cell fold dicts.

### What a fold's result is keyed on

`fold_key()` hashes: the per-cell **content digests** (`fingerprints.dataset_fingerprint`, so a loader fix, a sentinel cleanup or one extra row changes it), the held-out cell id, `FEATURE_VERSION`, `GBRT_PARAMS`, the seed, the tracked **numeric-stack versions** (a different scikit-learn fits a different model), whether out-of-fold predictions were requested, and `FOLD_CACHE_CODE_VERSION` — which a human bumps when the fold's recipe or metrics change, the gap `bundle_cache.MODEL_VERSION` documents for trained bundles.

Deliberately **not** keyed on a dataset *name*: `"Severson"` is a string, and the Tier-4 incident is what comparing by name costs. The registry's per-cell digests exist for the same reason.

One assumption is load-bearing and stated: when a caller passes pre-built feature frames (`featured=`), the key assumes they were derived from the same cells by the same feature version — the invariant the app's own feature cache already relies on.

### Opting out, and failing safely

- `batlab.validation.replication`'s independent recompute and the metric regression gate both pass `use_fold_cache=False`. A check that replays a stored answer is not a check.
- Supplying `forecaster=` disables the cache outright: a caller's model has no identity this module can key on, and a stale hit there would silently grade the wrong estimator.
- Every filesystem error is swallowed and treated as a **miss** — a cache is an optimization and must never be able to fail a training run. Writes go to a temp file and are renamed into place, so a process killed mid-write leaves either the previous result or nothing, never a truncated one: the interrupted-run case the cache exists for is exactly the case where a half-written fold is possible.

### Knobs

| Variable | Default | Meaning |
|---|---|---|
| `BATLAB_LCO_CACHE` | `on` | `on` replays a completed fold; `off` neither reads nor writes; `refresh` ignores existing entries and rewrites them. |
| `BATLAB_LCO_CACHE_DIR` | `<checkout>/.cache/lco` (or the user cache dir for a pip-installed `batlab`) | Cache root override. Tests point this at a tmp path, and the test suite pins `BATLAB_LCO_CACHE=off` so a developer's warm cache can't let a fold-counting or timing test pass on last week's answer. |

`run_lco()` returns `fold_cache: {mode, enabled, key, dir, hits, fitted}` (`dir` is the cache root, `key` the key directory inside it, `hits` the folds replayed) so a repeated boot reports `46/46 reused` rather than leaving it to be inferred from a layer that finished in 0.4 s instead of 12 minutes. The nested schema is declared in `batlab.results.FoldCacheSummary` and checked against a real payload by `tests/test_public_api.py`.

---

## 0.2.0: the physics block moved into the library (feature set `v13`)

`build_features()`'s SEI/LAM block used to be imported from the demo application's `src/`, which made a *feature column* depend on the caller's `sys.path`: on the four NASA cells the same call reported **SOH R² 0.9580** in a pip-installed process and **0.9471** with the app's `src/` importable. It now lives in `batlab.features.physics_calibration` and is gated on the frame's own declared `source`/`chemistry` attrs, so the population is a property of the data:

| four NASA cells, `batlab.load` + `benchmark` (library only) | before | after |
|---|---|---|
| SOH R² | 0.9580 | **0.9471** |
| RUL R² (observed-EOL pool) | 0.7614 | **0.4216** |
| SOH R², the same call with the app's `src/` on `sys.path` | 0.9471 | **0.9471** |
| feature version | `v12-rul-label-provenance` | `v13-features-owned-physics` |

Both after-values are the same floats bit for bit (`soh_r2 = 0.9470975121947384`) on either path — measured, not assumed, and pinned by `tests/test_feature_environment_inputs.py`, which also runs the library in a subprocess with no `src/` anywhere on the path.

The RUL move is the larger one and is worth stating plainly: on this 4-cell fleet the physics block *costs* ~0.34 RUL R² (0.7614 → 0.4216). The production app path — which always had the block, and which preprocesses these four cells differently — reports **0.7449 / 0.4121**, so v13 makes the library agree with the app rather than the other way round, and both then lose to the RUL formula baseline (**0.6768** on the app path). A negative result on n=4 cells with wide bootstrap brackets, published as-is rather than reverted to the population that looked better.

The app path itself is unchanged, verified after the move: `src/data_loader.build_battery` → `compute_features_only` → `run_lco` gives `soh_r2 = 0.7448917744577574`, `rul_r2 = 0.41212031190361514`, `baseline_soh_r2 = 0.6030539120231699`, `rul_formula_baseline_r2 = 0.6767915168720835` — the same numbers the README publishes for it.

Severson's headline is unchanged by the move: 46 cells, `soh_r2 = 0.9920`, `rul_r2 = nan` (not evaluable, 0% observed labels) — re-measured on the real fleet, 46/46 folds fitted cold under the new key.

Consequences worth knowing:

- every v12 fold-cache entry and every v12 bundle is invalidated by the version bump; a first boot after upgrading pays the cold cost once, then caches as usual;
- registry rows logged under v12 remain valid for what they measured, but a v13 row is a different population — `src/metric_history.py` reports movement across a feature-version boundary separately from drift, which is exactly this case;
- the app's own loader (`src/data_loader.build_battery`, which reads the NASA and synthetic CSVs under `data/raw/` directly) had never declared provenance, although the batlab schema requires it. It does now, from the same `ChemistryProfile` classifier the rest of the app uses, so the app's frames keep the physics features they always had;
- PyBaMM remains an optional extra (`pip install "battery-lab[physics]"`), and it can only affect `physics_spm_capacity_ah`, which is not in `FEATURE_COLUMNS` — so its absence changes no reported number, only whether that display column is populated.

### Did any of this cost the demo application anything?

No — and that is a measured claim, not an assurance. The app's numbers on its own
path are unchanged (0.7448917744577574 / 0.41212031190361514 / 0.6030539120231699 /
0.6767915168720835 on the four NASA cells), and each seam was exercised directly
rather than inferred from "the app still boots":

| what the app does | pre-move rule | after |
|---|---|---|
| eligibility, synthetic fleet (`Cell1`… — LiCoO₂, but the synthetic profile) | not eligible | **not eligible** |
| eligibility, NASA `B0005` | `NCA_Kim2011` | **`NCA_Kim2011`** |
| eligibility, Severson `S-b1c0` | `Chen2020` | **`Chen2020`** |
| eligibility, CALCE `CS2_33` | not eligible | **not eligible** |
| `from physics_calibration import …` (3 call sites) | the app's own module | **re-export of the library** — `calibrate_cell`, `physics_ml_agreement`, `physics_gbrt_divergence_report` are the *same objects* |
| `physics_ml_agreement("B0005", df)` | physics verdict + ML verdict | **physics verdict + ML verdict** (`agree=False`, note populated — the app's `recommendations.diagnose_mechanism` is what the shim registers) |
| `physics_gbrt_divergence_report({cid: raw_df})` | per-cell report | **4 cells**, `B0005` first, `closer_model="physics"` |
| physics features in the app's own frames | NASA populated, synthetic all-NaN | **NASA 91.7% finite, synthetic all-NaN** — unchanged |

> **Postscript (2026-09-24).** The `NCA_Kim2011` values in the two eligibility rows above are
> what was true *when that refactor was measured*, and are kept because rewriting them would
> falsify the record. Both anchors have since been found cathode-mismatched — NASA was declared
> LiCoO₂ everywhere but anchored on `nca_ocp_Kim2011`, and Severson (LFP, 2.0 V discharge cutoff
> per Severson et al., *Nature Energy* 4, 383–391) was anchored on `Chen2020`
> (`nmc_LGM50_ocp_Chen2020`, NMC811, 2.5 V cutoff). They are now `Ramadass2004` and `Prada2013`
> respectively. See `METHODOLOGY.md` and the enforcing test
> `test_every_anchor_is_cathode_matched`. The change moves only `param_set`, `chem_label` and
> the display-only `physics_spm_capacity_ah` — never a `FEATURE_COLUMNS` input. Measured by
> running `run_lco()` twice over the real four NASA cells (`use_fold_cache=False`), old anchors
> vs new, diffing every reported field: **no field differs** (SOH R² 0.9471, RUL R² 0.4216 both
> times, `physics_features=True` both times, so the parameter set was genuinely consulted).
> Note that the CI metric gate fleet is synthetic and so is structurally blind to this change —
> its passing is not evidence for it.

Two things worth stating plainly. First, the synthetic fleet was never
calibration-eligible: the pre-move rule keyed on the *profile class*
(`LiCoO2NASAProfile` → NCA_Kim2011, `LFPSeversonProfile` → Chen2020, everything else
None), and synthetic cells resolve to `LiCoO2SyntheticProfile`. The allow-list
narrowed *how* eligibility is decided (data attrs instead of a profile import), not
*who* is eligible — which is the only reason no app number moved.

Second, the app's physics call sites wrap these calls in `try/except`
(`app/_pages/health.py`, `app/_pages/benchmark.py`), so a broken shim would have
degraded silently into "physics unavailable" rather than failing loudly. That is
exactly why the table above calls the functions directly, and why the shim's
registration step is pinned by `tests/test_physics_calibration.py` rather than left
to a boot smoke test.

What *did* have to change app-side: the shim itself, `build_battery()`'s provenance
declaration (a latent defect — the app's own loader had never declared
`source`/`chemistry` for the CSVs it reads directly), and one added profile field.
Three files, no page rewritten.

---

## A blank measurement row, and why a baseline can stop being a number

A per-cycle summary can carry a row with nothing in it. Severson's batch-1
records S-b1c0 (cycle 11) and S-b1c18 (cycle 39) have an empty capacity column
in the source CSV, so `soh_pct = capacity/q0*100` is blank with it.

`run_lco` never shows such a row to a model: `get_model_matrix()` drops rows
whose *features* are non-finite, and the blank row's `dod_proxy` is non-finite.
A least-squares line cannot be handed a blank *target*, though — sklearn's
`LinearRegression` raises (`Input y contains NaN`) and `r2_score` returns NaN for
one without raising at all — so the trivial baselines used to take the whole
fleet's floor down with one blank row: the fit raised inside `app/_data.py`'s
defensive `except`, which recorded `None`, and the "+X over the trivial baseline"
claim lost its denominator with no trace of why.

Both baselines now score only the rows they can score, count what they set
aside, report an unscorable fold as unscorable *with a reason* instead of
averaging a NaN into the mean, and reject a non-finite headline at `_safe_r2`.
They also accept the same `{"cycles": df}` cell shape `run_lco` accepts, so the
CI metric gate hands one dict to both.

Measured by running the pre-fix implementation side by side with the new one
(same process, same frames):

| fleet | pre-fix `baseline_soh_r2` | after | folds | blank rows set aside |
|---|---|---|---|---|
| CI fixture fleet (5 cells) | −1.7676911798184485 | **−1.7676911798184485** (bit-identical) | 5/5 | 0 |
| NASA, production app path (4 cells) | 0.6030539120231699 | **0.6030539120231699** (bit-identical) | 4/4 | 0 |
| Severson (46 cells) | **`ValueError: Input y contains NaN`** | **−0.3278149205944300** | 46/46 | 2 |

The move is therefore additive: one fleet's floor is restored, and no published
baseline changes — which is why this needed no `FEATURE_VERSION` bump and no
fold-cache invalidation (features and folds are untouched; only the baseline's
row selection changed).

Severson's restored floor is **negative**, and that is the honest reading rather
than a bug: 46 LFP cells fade at very different rates, so a single global
straight line is worse than predicting each cell's mean. Against it, the GBRT's
`soh_r2 = 0.9920` is **+1.32** — a much larger advantage than NASA's +0.142,
because a 46-cell fleet is exactly where a per-cell model should beat one line.
The gate now pins this number (`tests/metric_gate_expectations.json`,
declared as a *ceiling* — the risk is the trivial floor getting smarter and
quietly shrinking the claim), and both ways it used to disappear fail the gate:
`None` under rule 3, and a NaN under the new non-finite guard in `check_metric`
(NaN compares False against every floor, ceiling and tolerance, so an unguarded
one would have passed).

App-side, a failed baseline is no longer silent: `metrics["baseline_soh_r2_error"]`
carries the exception text into the bundle and the registry row, `None` there
means "computed".

---

## Measuring it on your own machine

Both profilers call the same functions the app calls — they are not a parallel implementation of the pipeline.

```bash
python scripts/profile_boot.py nasa zhu2022 severson   # core / validation / forecast / calibration per fleet
python scripts/profile_lco.py severson                 # per-phase, per-fold LCO cost, plus cache reuse
python scripts/profile_lco.py severson --no-cache      # force the cold path
```

`profile_boot.py` prints one row per phase per fleet, so the question "what is a boot actually waiting for?" is answered with a stopwatch rather than an argument.

---

## Honest limits

- **A genuinely cold first boot is not instant.** The residual floor is the core fit itself — Severson's GBRT is 43.5 s on a 12-vCPU box. What the split *bounds* is what the first paint waits for, not the total work.
- **A cold validation layer still costs what it costs.** Severson's 46 folds are ~12 minutes of real computation; the cache removes the *repeats* and the interruption penalty, not the first run. The same code is roughly 2 minutes on a 32-core machine, because the 8-worker pool is the binding constraint on a laptop.
- **The interface/quantile layers are not cached.** The fold cache covers `run_lco`'s GBRT folds. The quantile-calibration folds are cheaper and, on a fully-extrapolated fleet, short-circuited entirely (`METHODOLOGY.md` §18).
- **A fold cache is disk, not magic.** Each entry is one small joblib file under `.cache/lco/<key>/`; a fleet's 46 folds are a few hundred kilobytes. Deleting the directory is always safe.
