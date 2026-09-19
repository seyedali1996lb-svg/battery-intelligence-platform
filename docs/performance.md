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
