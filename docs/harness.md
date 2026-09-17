# Validation harness

**Grade any battery-degradation forecaster by the same rules this platform
grades its own.** One call, six checks, and an explicit list of what the run
does *not* support claiming.

```python
from batlab.harness import validate_forecaster

report = validate_forecaster(cells, model=my_model)

print(report["verdict"]["summary"])
for claim in report["verdict"]["claims"]:
    print("  supported:", claim)
for gap in report["verdict"]["withheld"]:
    print("  withheld: ", gap)
```

Or from the command line, with no Python written at all:

```bash
python -m batlab.harness --loader batlab.datasets.nasa:load_nasa_cells
```

## Why this exists

Every accuracy number this platform reports was produced by one model: its own
gradient-boosted regressor. That coupling was backwards. Leave-cell-out
validation, per-row label provenance, conformal interval calibration, declared
metric floors and sealed replication bundles are statements about a
*methodology* — they were just never separable from the model they were welded
to, which made them unusable for anyone else's model and made "our validation
is honest" a claim about a dataset rather than about a method.

The harness is that seam removed. It is also the honest answer to a question
you should ask of any battery-ML result, including this platform's: *how do I
know the number was produced by a test that could have failed?* Every section
below is a way for the answer to be no.

## What it runs

| # | Check | The question it answers | Where it came from |
|---|---|---|---|
| 1 | **Leakage lint** | Can any model feature be read by the RUL label's own generating expression? | The defect behind this platform's former RUL R² = 0.9994 |
| 2 | **Label provenance** | How many RUL rows carry a *measured* end-of-life label, and how many a closed-form extrapolation? | Scored on extrapolated labels, a model is graded on formula recovery |
| 3 | **Leave-cell-out** | Does the model generalize to a cell it has never seen? | Folds hold out whole cells, never rows |
| 4 | **Interval** | Does the claimed 80% interval actually cover 80%? | Conformal quantile recalibration (Romano et al. 2019) fitted per fold on the other folds only |

Every section except the interval one asks a question about *your* model. The
interval section is a separate question with a separate answer: an
interval-capable model (one with `predict_interval`) is calibrated on its own
intervals; a point-only model gets a distribution-free interval built from its
own out-of-fold residuals, and the report says which of the two happened.
With no model supplied at all, the interval section uses this platform's own
published configuration (point + Q10/Q90 quantile regressors, conformally
recalibrated), because inventing a second, unpublished interval number for the
platform's own model would be exactly the kind of silent substitution this
harness exists to prevent. Its point metrics are unchanged either way — the
interval triple uses the same `GBRT_PARAMS` for its point regressor.
| 5 | **Prospective** | Does it *forecast*, or only interpolate a curve it has already seen the end of? | Train on each cell's first half, score the second |
| 6 | **Metric gate** | Did a headline number move without anyone saying so? | Declared floor/ceiling + approved-change log |

Each runs beside the baselines that make its number readable: a per-cell
straight line through cycle number (the "you don't even need a model" floor)
and the closed form that generates extrapolated RUL labels (the "you have not
out-forecast a fade formula" floor). A model that loses to either is reported
as losing.

## Bring your own model

Anything that can be fitted on a feature matrix and predict on it. The
harness builds the features from raw cycles itself, so what gets graded is the
model, not a matrix you pre-shaped.

### An sklearn estimator

```python
from sklearn.ensemble import RandomForestRegressor

validate_forecaster(cells, model=RandomForestRegressor(n_estimators=200))
```

The harness adds **no preprocessing** for a model you supply — pass a
`Pipeline` if your model needs scaling:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

### A factory (recommended)

The harness calls the factory **once per fold and once per target**, so a fold
can never be fitted on another fold's state — the most common way a
hand-rolled cross-validation leaks. A fresh model per call is the contract:

```python
def make_model():
    return GradientBoostingRegressor(n_estimators=300, max_depth=3)
```

Passing a *pre-fitted* estimator is refused rather than deep-copied: a model
that already saw the held-out cell produces a beautiful, meaningless R².

### A PyTorch model

```python
import torch
from batlab.harness import torch_forecaster

def module_factory(in_features):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )

validate_forecaster(cells, model=torch_forecaster(module_factory, epochs=300, lr=5e-3))
```

`torch_forecaster` returns a point model, so it comes with no uncertainty
claim — the harness builds one for it from out-of-fold residuals (check 4).
PyTorch is imported lazily and is not a batlab dependency.

### Anything else

`CallableForecaster(fit_fn, predict_fn)` wraps any pair of callables — an ODE
fit, a Keras model, a subprocess call to MATLAB. Supplying
`interval_fn(state, X)` additionally makes it interval-capable, at which point
check 4 calibrates its own intervals instead of building one for it.

## Reading the report

```text
batlab-forecaster-harness: nasa (4 cells, sha256 0b356f6012c2)
model:   platform default (GBRT, scaled) (GradientBoostingRegressor) [interval-capable]
lint:    PASS — 2 invariant(s)
labels:  580 measured-EOL / 0 extrapolated RUL rows (100% measured)
leave-cell-out (new-cell generalization)
  SOH  R² 0.745  MAE 1.830  CI [0.560, 0.930]
  RUL  R² 0.412  reliable
baselines (same folds)
  trend  R² 0.603 (LCO)  0.241 (prospective)
  formula R² 0.677 (LCO)  0.510 (prospective)
interval calibration (nominal 80.0%)
  raw 72.9% → calibrated 80.3% (E* = 6.300 cycles, mean width 84.100)
prospective (same-cell forecasting)
  SOH  R² 0.492  MAE 1.204
  RUL  R² 0.101  (not reliable/not evaluable)
metric gate: PASS
verdict: PASS — every declared floor/ceiling held
  supported: SOH on unseen cells: R² = 0.745 [0.560, 0.930] (beats the 0.603 trend baseline)
  withheld:  interval coverage: no measured-EOL rows …
```

The `verdict` block is the part to read first. `supported` lists claims the
evidence carries; `withheld` lists the ones the harness declines to make and
names the reason. On a fleet whose cells never reach end-of-life in-window,
RUL is *not reported at all* — not reported with a caveat, not reported as a
lower bound, withheld — because every available label would be a closed-form
extrapolation of the cell's own fade history.

A tie is reported as a tie: a model that matches the per-cell trend baseline
to within 0.005 R² reads "matches", never "beats".

## The metric gate

Measurement without enforcement is a report, not a gate. Declare what the
numbers are allowed to be:

```json
{
  "schema_version": 1,
  "metrics": {
    "soh_r2": {"floor": 0.60, "baseline": 0.745, "tolerance": 0.05},
    "prospective_soh_r2": {"floor": 0.0},
    "rul_r2": {"floor": 0.30, "allow_not_evaluable": true}
  }
}
```

```python
validate_forecaster(cells, model=my_model, gate_path="floors.json")
```

Three rules, all enforced:

- **floor / ceiling** — outside the declared range fails, period.
- **baseline** — movement beyond tolerance fails as an *unexplained change*
  until it is recorded with a reason.
- **not evaluable** — a metric that stopped being computable fails unless the
  expectation explicitly allows it. A metric the harness never measured
  arrives as `null`, not as a number, so it cannot silently clear its floor.

No expectations supplied means the gate reports **NOT CHECKED** and the exit
code stays 0. That is deliberately not a pass: nothing was enforced.

## Sealing a result

```python
from batlab.harness import seal_bundle

seal_bundle(report, cells, "out/nasa-bundle", loader="batlab.datasets.nasa:load_nasa_cells")
```

Writing `benchmark.json` (fold structure + metrics), `harness_report.json`
(the full report), and `replication.json` — the root of trust, carrying the
per-cell content digests, the environment, the model identity, and the SHA-256
of every other file.

A third party then runs:

```bash
python -m batlab.validation.replication out/nasa-bundle \
    --loader batlab.datasets.nasa:load_nasa_cells --recompute
```

and gets pass/fail per check: **seal** (files unmodified), **data-identity**
(their copy of the data digests byte-identically), **environment** (differences
listed), **recompute** (the number re-derives). Add  `--model my_pkg.models:make_forecaster` when the bundle was published for a
model other than this platform's default — a non-default bundle re-run against
the default one is reported as a missing input, not as a failed number.

The bundle also records the loader's *arguments*, and the verifier supplies
them, so a loader whose signature requires one (for example
`experiment_registry:reload_reference_cell_data(dataset)`) verifies without
the reviewer re-typing anything — the command in the bundle's `notes` is the
whole command. A loader that declares a `bundle_dir` parameter is handed the
*verifier's* path to the unpacked bundle, so a bundle verifies wherever it was
unzipped rather than only from the publisher's working directory.

### A bundle that carries its own data

`loader=` is the right design for a public reference fleet: the data is
downloadable, so the bundle stays small and the verifier supplies the dataset.
It is the wrong design for data a third party cannot obtain — a tenant's own
upload, an internal fleet — where it degrades the seal into "checkable if you
have the publisher's filesystem".

```python
seal_bundle(report, cells, "out/my-bundle", embed_data=True, dataset="uploaded")
```

```bash
unzip out/my-bundle.zip -d bundle
python -m batlab.validation.replication bundle \
    --loader batlab.validation.bundle_data:load_bundle_cells --recompute
```

`embed_data=True` writes each cell's cycle table into `cells/` and points the
bundle at the bundle-relative loader, so the number is re-derived with no data
path, no deployment and no network — and the seal covers the data files as
well as the report, because a seal that certifies the report but not the
evidence inside it would certify the wrong thing.

The trade is explicit: **the data is then inside the artifact you share.**
That is the correct default for data nobody else can get and the wrong one for
a public fleet, which is why it is a parameter rather than a policy.

Two serializations exist in `bundle_data.py`, deliberately:
`cell_digest()`'s fingerprint form (`%.10g`, short and diffable, and lossy as
storage) and the stored table (`%.17g`, where every double survives
write-then-read). Storing the fingerprint form would leave a recompute ~1e-5
from the published number — forty times the replication tolerance, on a bundle
that is byte-identical — and the reader pins
`float_precision="round_trip"` for the same reason (pandas' default converter
moves values by 1 ULP). `cells/index.json` records both digests per file, so a
reviewer can walk from the file's bytes to the sealed fingerprint with
`sha256sum` and no tooling in between.

When the cycles are deliberately *not* embedded, the bundle keeps the digests
and the numbers and records the deployment's own store as the data path. If a
verifier cannot load that path, **data-identity FAILS** — it is not degraded to
"unchecked". A named failure the reviewer can act on is worth far more than an
artifact nobody can check reporting PASS.

### A bundle that carries its own model

The recompute check is the one that re-derives the headline number, and it
needs the model. For a bundle published for anything but this platform's own
GBRT that model used to be unobtainable: the bundle recorded only what the
model *was*, so the check could not run and the artifact certified less than it
appeared to. The model now travels the same way the data does.

```python
seal_bundle(report, cells, "out/ridge-bundle",
            loader="experiment_registry:reload_reference_cell_data",
            loader_kwargs={"dataset": "synth"},
            model_source=open("my_model.py").read(),
            model_entry_point="make_model")
```

```bash
unzip out/ridge-bundle.zip -d bundle
python -m batlab.validation.replication bundle \
    --loader experiment_registry:reload_reference_cell_data \
    --model batlab.harness.model_source:load_bundle_model --recompute
```

`model_source` writes the module's text to `model/module.py`, seals it with the
rest of the bundle, and records the *bundle-relative* model loader with its
arguments — so `--model <that loader>` alone is the whole command, and the
number re-derives from the archive plus the public data.

Two properties are not negotiable here, because both are about executing
someone else's code at verification time:

- **Digest first, import second.** `load_bundle_model()` checks the file
  against the digest its own index records and refuses a mismatch *before*
  importing it, so a model edited after sealing is never executed. The index
  sits beside the file, so a reviewer can confirm the bytes with `sha256sum`
  and no tooling in between.
- **Never loaded implicitly.** Importing a Python module runs it. The verifier
  has to name `load_bundle_model` on the command line — a deliberate act —
  and the file is plain, readable source precisely so it can be read first.
  (A `.pkl` could not be: loading it *is* executing it.)

When a non-default model is deliberately **not** carried — the app offers that
choice only for an uploaded module, whose source may be someone's property —
the recompute check fails with a message naming the file it would have loaded
and the exact flag that would let it, rather than silently re-running the
default model and reporting a number mismatch.

## In the app

`Analyse → Bring your own model` is this harness with a UI in front of it. Pick
a fleet, pick a model — the platform's own GBRT, a scaled Ridge, a random
forest, the training-mean floor, or a `.py` module you upload — and read the
verdict: the claims the run supports and the claims it withholds, in two
columns, with every section (fold-level leave-cell-out, baselines, calibration,
prospective, provenance, gate) rendered underneath.

Four things about that page are deliberate, and each is stated on the page
rather than left for you to infer:

- **Your own uploads are on the same list.** The raw cycles are persisted at
  import time (`src/uploaded_store.py`, content-addressed by the upload's own
  hash), which is what makes an uploaded fleet fingerprintable — so it is
  offered as a fleet, first in the picker when you have one, with its cell
  count and upload date. Selecting it grades your model on *your* cells, and
  the sealed bundle covers your data rather than a reference fleet's.
- **You choose whether your data travels in the bundle.** The checkbox (default
  ON for an upload) decides between embedding your raw cycles — verifiable by
  anyone you hand the bundle to, and they receive the data — and recording this
  deployment's store as the data path, which keeps the cycles home and requires
  that store to verify. The page states whichever is currently selected.
- **`.py` modules only, and you read it first.** A serialized model (`.pkl`,
  `.pt`, `.joblib`) executes as part of loading, so there is no moment at which
  it can be inspected — it is refused, with that reason on screen. An uploaded
  module is **not sandboxed**: it runs with the app process's privileges, and
  the page says so directly above the acknowledgement checkbox that gates the
  run. Nothing runs until you tick it.
- **The recorded loader is the platform's own reloader, not a raw dataset
  loader.** The two return different tables — the reloader's frames carry the
  enrichment step's derived columns — and `cell_digest` hashes every column,
  so recording the raw loader would fail a bundle that is in fact correct.

You also choose — for an upload — whether **your model travels inside the
bundle**. With it, `--model batlab.harness.model_source:load_bundle_model`
re-runs the exact configuration that was graded; without it, the bundle records
what the model *was* and the recompute check says so. The catalogue baselines
carry themselves (a 20-line ridge is not a secret), and the platform's own GBRT
needs nothing carried because a verifier re-creates it from the repo.

The verification command printed under a result is the command that can
actually run, decided in one pure function (`harness_models.verify_command`)
and printed with a reason when it cannot: `--recompute` appears when the model
is obtainable (from the repo, or from the bundle) and is left out — with the
sentence explaining what the remaining three checks still hold — when it is
not. On the page's copy the model source also carries its own disclosure: read
`model/module.py` before running the command, because importing it executes it.

Runtime is disclosed per fleet and measured, not estimated: the four-cell NASA
fleet runs in about 15 s under the page's default configuration, the synthetic
and Zhu 2022 fleets in about a minute, and the 46-cell Severson fleet in
several minutes. Nothing runs on page load.

## What it does not do

- **No hyperparameter search, no leaderboard.** It grades one model on one
  fleet and says what that grade supports.
- **Marginal, not conditional, coverage.** The interval numbers are guarantees
  over the calibration population, not per-cell promises; the residual
  interval for a point-only model has *constant width*, so a model whose error
  grows with the horizon is under-covered at long range even when the pooled
  number looks nominal. The report says this next to the number.
- **Public-data validation is not industrial validation.** A leave-cell-out
  number describes generalization to a cell that looks like the ones in the
  fleet you passed in — same chemistry, similar duty cycle, similar
  temperature range. The dataset fingerprint travels in every report and every
  bundle so two runs are comparable only when their data actually matches.
- **Garbage in, disclosed garbage out.** The lint catches label-formula
  leakage mechanically; it cannot prove your features are causal, your cells
  are representative, or your lab data is clean.
- **It does not sandbox model code.** An uploaded `.py` module — or any code
  you hand the in-app uploader — runs in the host process. That is disclosed on
  the page and is the reason the uploader accepts readable text only.
- **A bundle carrying your data is only as private as where you send it.** The
  seal proves the numbers came from the tables inside the artifact; it does not
  encrypt them. Share one and you have shared the cycles.

## See also

- [`notebooks/02_data_leakage.ipynb`](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/notebooks/02_data_leakage.ipynb)
  — the leave-cell-out vs row-split demonstration the lint exists to prevent.
- [`METHODOLOGY.md`](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/METHODOLOGY.md)
  — the formula and provenance class behind every feature and target the
  harness feeds your model.
- `batlab.validation.replication` — the verifier a third party runs.
- `batlab.validation.metric_gate` — the gate's own rules, unit-tested.
