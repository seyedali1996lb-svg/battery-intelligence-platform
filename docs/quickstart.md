# Ten minutes from `pip install` to a number

This page starts where a user starts: from the wheel, in an empty environment,
with no clone and no repository layout to understand. Ten minutes end to end,
including the download of a public dataset.

!!! note "Why the package name differs from the import name"
    The distribution is **`battery-lab`**; the import package is **`batlab`**.
    PyPI's `batlab` is Lexcelon's unrelated Batlab V1.0 hardware library, so the
    distribution had to be named for what is actually free. `import batlab` is
    what the docs, notebooks and examples say.

## 1. Install (1 minute)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install battery-lab
batlab version
```

Core `batlab` depends only on `pandas`, `numpy`, `scikit-learn`, `scipy` and
`requests`. Each dataset's heavier parser is an optional extra so installing the
library does not pull in every loader's dependencies:

```bash
pip install "battery-lab[severson]"     # h5py, for the Severson MATLAB source
pip install "battery-lab[oxford]"       # mat-io
pip install "battery-lab[calce]"        # openpyxl
```

## 2. Load a fleet (30 seconds, plus a download the first time)

```python
import batlab

cells = batlab.load("nasa")            # {cell_id: DataFrame}
print(len(cells), sorted(cells))       # 4 ['B0005', 'B0006', 'B0007', 'B0018']
print(cells["B0005"].columns.tolist())
```

Five datasets are available through one standardized schema:
`nasa`, `severson`, `zhu2022`, `oxford`, `calce` (`batlab.AVAILABLE_DATASETS`).
Downloads are checksum-verified, and `batlab.datasets.validate_schema()` raises a
`SchemaError` with a specific message on any violation:

```python
from batlab.datasets import validate_schema

validate_schema(cells["B0005"], kind="cycle")
```

## 3. Get an honest number (2 minutes)

```python
lco = batlab.benchmark(cells)          # leave-cell-out, not a row-level split
print(f"SOH R² = {lco['soh_r2']:.4f}")
print(f"RUL R² = {lco['rul_r2']}")
print(lco["fold_cache"])               # what was replayed vs refitted
```

Leave-cell-out holds out **entire cells** — the model is scored only on cells it
never saw, which is the question a deployment actually asks. The same data and
model under a naive row-level split reports `R² ≈ 1.00`; the honest number here
is ≈ `0.96` through this loader. That gap is reproduced live in
`notebooks/02_data_leakage.ipynb`.

One output key is worth looking at before comparing your number to anyone
else's: `physics_features`. `build_features()` uses a physics-calibration feature
block when `physics_calibration` is importable, and that module ships with the
demo application rather than with this library — so the same call gives SOH R²
**0.9580** installed on its own and **0.9471** when the demo app's `src/` is on
the path. Both are honest; they are simply different populations, and the flag is
how you know which one you are reading. See
[API stability](api_stability.md#environment-dependent-inputs).

Two things in that output are worth reading carefully, because they are the
library's whole point:

- `rul_reliable`. RUL can only be *measured* for a cell whose data reaches the
  80% end-of-life threshold. Rows from cells that don't carry a closed-form
  **extrapolated** label instead, and scoring a model against those measures
  formula recovery, not forecasting. So RUL is scored on observed labels only,
  a fleet with too few of them reports *not evaluable* rather than a number, and
  a formula baseline is reported alongside the model — a model must beat the
  closed form before its RUL claim means anything.
- `fold_cache`. A leave-cell-out fold is a pure function of its inputs, so a
  completed fold is replayed rather than refitted on the next identical call
  (measured: 46/46 folds in ~2.7 s instead of ~710 s of fits). The numbers are
  identical either way — `hits` tells you which path produced them.

Same thing from a shell:

```bash
batlab benchmark --dataset nasa --out report.json
```

## 4. Grade *your* model by the same rules (3 minutes)

The part most people come for. Point the harness at any forecaster — a
scikit-learn estimator, a PyTorch module, your own numerical fit — and it runs the
same six checks it runs on this platform's own model:

```python
from sklearn.linear_model import Ridge
from batlab.harness import validate_forecaster

report = validate_forecaster(cells, model=Ridge())
print(report["verdict"]["summary"])
for claim in report["verdict"]["claims"]:
    print("  supported:", claim)
for gap in report["verdict"]["withheld"]:
    print("  withheld: ", gap)
```

The checks, each of which is a way for "is this number honest?" to come back *no*:
leakage lint, label provenance, leave-cell-out, interval calibration, the
prospective split, and the metric gate. `model=None` grades this platform's GBRT
and reproduces the published numbers exactly.

Two guarantees make it safe to point at someone else's model: the factory is
called once per fold and once per target (so a fold can never be fitted on
another fold's state), and a **pre-fitted estimator is refused** rather than
deep-copied, because a model that already saw the held-out cell produces a
beautiful, meaningless R². A model you don't trust can be graded in a separate
process: `batlab.harness.sandbox_forecaster(...)`. See
[the harness guide](harness.md).

## 5. Make the result portable (1 minute)

```python
from batlab.harness import seal_bundle

seal_bundle(report, cells, "out/nasa-ridge")     # + embed_data=True to include the cycles
```

```bash
python -m batlab.validation.replication out/nasa-ridge \
    --loader batlab.datasets.nasa:load_nasa_cells --recompute
```

A sealed bundle carries per-cell content digests, the environment, and the
SHA-256 of every file in it, so a reviewer re-derives the number within 1e-6
instead of trusting yours.

## Where to go next

- [Validation harness](harness.md) — the adapters, the gate file format, and the honest limits behind each number.
- [Boot layers and fold caching](performance.md) — why the second run is seconds, and what a cold one really costs.
- [CLI](cli.md) — every verb and flag.
- [API stability](api_stability.md) — what you may depend on, and how breaking changes are announced.
- [Datasets](datasets/index.md) — schema, citation and license per dataset.
- `notebooks/01_quickstart.ipynb` — the same tour with plots.
