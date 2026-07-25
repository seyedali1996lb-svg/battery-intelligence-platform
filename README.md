# batlab

A citable, honest research library for battery degradation analysis.

`batlab` standardizes the unglamorous part of battery-ML research — loading public cycling
datasets in one consistent schema, engineering literature-cited features, and validating a
SOH/RUL model with leave-cell-out cross-validation instead of a row-level split that quietly
leaks near-neighbor cycles into the test set. It also ships a reproducible benchmark manifest
format, so a reported R² can be checked, not just cited.

[![CI](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml)

**[Full documentation →](https://seyedali1996lb-svg.github.io/battery-intelligence-platform/)** (or build locally: `mkdocs serve` after `pip install -e ".[docs]"`)

**[Methodology →](METHODOLOGY.md)** — the actual formula behind every index and calculation in the platform (SOH, RUL, stress index, dQ/dV, knee detection, GBRT/quantile models, LCO validation, trajectory memory, PyBaMM physics projection, anomaly thresholds, second-life economics), with what's measured vs. derived vs. machine-learned vs. simulated vs. an illustrative assumption stated explicitly for each one.

---

## Install

```bash
pip install -e ".[severson,oxford,calce]"   # extras are only needed for those loaders' parsers
```

Core `batlab` depends only on `pandas`, `numpy`, `scikit-learn`, `scipy`, and `requests`.

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

![NASA B0005 measured vs. predicted SOH](docs/assets/quickstart_soh.png)

*Real output from `notebooks/01_quickstart.ipynb`, not a mockup — GBRT trained on the 4 committed NASA PCoE cells.*

The full ten-minute tour is [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb).

## The honesty argument, in one number

[`notebooks/02_data_leakage.ipynb`](notebooks/02_data_leakage.ipynb) reproduces, on real data, why this library validates the way it does: a naive row-level train/test split on the same 4 NASA cells reports **SOH R² = 0.998**. Leave-cell-out — holding out an entire cell, never seen during training — reports **R² = 0.806** on the identical data and model. Both numbers are real; only one of them answers "does this generalize to a cell the model has never seen?" Every model `batlab.models` trains is validated the second way by default.

That honest 0.806 still doesn't say how much of it the model is *earning* — SOH-vs-cycle curves are smooth, so a trivial linear fit (cycle_number → SOH, no feature engineering) might already explain most of the variance. Same notebook checks it under the identical leave-cell-out fold structure: **R² = 0.603**. GBRT's engineered features are worth a real +0.203 over that baseline, not the full 0.806.

## Datasets

| Dataset | Chemistry | Cells | Auto-download |
|---|---|---|---|
| NASA PCoE (Saha & Goebel 2007) | LiCoO2 | 4 | Yes |
| Severson 2019 (*Nature Energy*) | LFP | 12 | Yes (needs `h5py`) |
| Oxford Path-Dependent 2020 | NCA | 12 (sparse checkpoints) | Yes (needs `mat-io`) |
| CALCE CS2 (Univ. of Maryland) | LiCoO2 | any, manually placed | No — instructive error tells you where |

Every auto-downloaded archive is SHA-256 checksum-verified against a pinned hash before parsing (`batlab.datasets._integrity`) — a corrupted transfer or a substituted file is rejected, not silently parsed into a schema-valid-looking DataFrame.

See [`docs/datasets/`](docs/datasets/index.md) for schema, citations, and licenses, and [`batlab/datasets/CONTRIBUTING.md`](batlab/datasets/CONTRIBUTING.md) for adding a fifth loader.

## What this is not

`batlab` doesn't simulate cell electrochemistry — see [PyBaMM](https://pybamm.org) for physics-based simulation — and doesn't attempt raw multi-vendor cycler-file structuring at the per-datapoint level, the way TRI's [beep](https://github.com/TRI-AMDD/beep) does; this library's schema is deliberately narrower, standardizing only the per-cycle summary level. It's a validation and baseline-modeling layer for the SOH/RUL prediction problem specifically.

## The demo application

This repository also contains a Streamlit application (`app/`) that consumes `batlab` — a fleet-monitoring dashboard, EU Battery Passport generator, and decision-support tool built on top of the library, kept working through this restructuring but not the primary deliverable of this repository going forward. It documents a large amount of real, working engineering — e.g. a second-life battery + solar sizing calculator that runs a real hour-by-hour (8760 hours/year) dispatch simulation against live PVGIS solar data (temperature-aware battery derating cited to real cell documentation, not a monthly approximation or a guessed curve), where testing against a non-European site live caught a real regional-coverage bug before it could ship silently broken; see [`docs/history.md`](docs/history.md) for its full build history and architecture, and run it with:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Development

```bash
pip install -e ".[dev]"      # pytest + every loader/notebook/docs extra
pytest tests/ -v
```

`pyproject.toml`'s `dev` extra pulls in everything needed to run the full test suite, execute the notebooks, and build the docs site.

## Citing

```python
import batlab
print(batlab.cite())
```

Or see [`CITATION.cff`](CITATION.cff). A JOSS paper draft is at [`paper/paper.md`](paper/paper.md) (not yet submitted — see its TODOs).

## License

MIT for `batlab` itself. Each dataset loader carries its own source dataset's license — see [`docs/datasets/`](docs/datasets/index.md) or `batlab.cite(dataset=...)` before redistributing any dataset's data.
