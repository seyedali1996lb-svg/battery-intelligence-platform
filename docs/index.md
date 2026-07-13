# batlab

A citable, honest research library for battery degradation analysis.

- Standardized loaders for four public li-ion cycling datasets (NASA, Severson, Oxford, CALCE) — one documented schema, not four ad hoc shapes.
- Literature-cited feature engineering (fade rate, dQ/dV, knee detection, Coulombic Efficiency, stress index).
- Gradient-boosted SOH/RUL models validated with **leave-cell-out** cross-validation, not a row-level split that quietly leaks near-neighbor cycles into the test set.
- A reproducible benchmark manifest format that records the exact seed, fold assignments, and feature-engineering version behind a reported number.

## Install

```bash
pip install -e ".[severson,oxford,calce]"   # extras only needed for those loaders' parsers
```

`batlab` itself only depends on `pandas`, `numpy`, `scikit-learn`, `scipy`, and `requests`. Each dataset loader's heavier parser (`h5py`, `mat-io`, `openpyxl`) is an optional extra so installing `batlab` doesn't pull in every loader's dependencies.

## Quickstart

```python
import batlab
from batlab.datasets import load_nasa_cells, validate_schema
from batlab.features import build_features, get_model_matrix
from batlab.models import train_models, predict
from batlab.validation import run_lco

cells = load_nasa_cells()                       # {cell_id: DataFrame}, one standardized schema
validate_schema(cells["B0005"], kind="cycle")    # raises SchemaError with a specific message on any violation

lco = run_lco(cells)                             # leave-cell-out, not a row-level split
print(lco["rul_reliable"], lco["rul_r2"])

print(batlab.cite())                             # BibTeX for the library
print(batlab.cite(dataset="nasa"))               # + license, for whichever dataset you used
```

The full ten-minute tour, with plots, is [`notebooks/01_quickstart.ipynb`](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/notebooks/01_quickstart.ipynb).

## Why leave-cell-out?

A row-level train/test split on cycles pooled across multiple cells looks fine — until you notice the held-out rows are the near-neighbors of rows the model already trained on. `notebooks/02_data_leakage.ipynb` reproduces this on real NASA data: a naive split reports **R² = 0.998**; leave-cell-out on the identical data and model reports **R² = 0.806**. Neither number is fake — they're answering different questions, and only one of them is the question "does this generalize to a cell the model has never seen?" This is the credibility argument for the whole library: every model in `batlab.models` is validated the second way by default.

## What this is not

`batlab` doesn't simulate cell electrochemistry (see [PyBaMM](https://pybamm.org) for that) and doesn't attempt raw multi-vendor cycler-file structuring at the per-datapoint level (see [beep](https://github.com/TRI-AMDD/beep) for that — `batlab`'s schema is deliberately narrower, standardizing only the per-cycle summary level). It's a validation and baseline-modeling layer for the specific SOH/RUL prediction problem, built around one strong opinion: a result is only useful if it's reported with the validation method and sample size that actually back it up.

## Where to go next

- [Datasets](datasets/index.md) — schema, citations, licenses, and how to add a fifth loader.
- [API reference](api/datasets.md) — every public function, generated from docstrings.
- [Notebooks](notebooks/01_quickstart.ipynb) — the four worked examples, rendered.
- [Project history](history.md) — the Streamlit demo application this library was extracted from, and everything built in it.
