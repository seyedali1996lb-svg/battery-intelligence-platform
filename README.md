# Battery Intelligence Platform

**An open-source battery analytics platform for degradation analysis, state-of-health estimation, remaining useful life prediction, and lifecycle intelligence using public battery datasets.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://seyedali1996lb-svg.github.io/battery-intelligence-platform/)

---

## Why this project exists

A cycling battery — in a lab, a vehicle, or a stationary storage installation — produces a large amount of voltage, current, capacity, and temperature data over its life. The engineering problem is not collecting that data; it's converting it into decisions: is this cell degrading faster than expected, how many cycles does it have left, is it safe to keep in first-life service, and what is a second-life or replacement plan worth in real currency.

This is harder than it looks, for reasons this platform is built specifically to confront:

- **Battery aging is nonlinear and cell-specific** — degradation doesn't follow one universal curve across chemistries or usage profiles.
- **Capacity degradation is the physical symptom, but it isn't self-explanatory** — a fade curve alone doesn't tell you whether it's on-track or accelerating without a baseline to compare it against.
- **SOH estimation needs to be trustworthy at the point it's produced** — it's the number every downstream decision (dispatch, second-life, replacement) is built on.
- **RUL prediction is where the stakes are highest and the failure modes are quietest** — a model can report an excellent R² and still fail on the next cell it sees, if validated the wrong way.
- **Data leakage in ML validation is the single most common way that happens** — see [Validation](#validation) below for a real, reproduced example.
- **There is no standard schema across public battery datasets** — NASA PCoE, the Severson et al. *Nature Energy* dataset, Oxford's path-dependent dataset, and CALCE's CS2 cells each ship in a different file format with different column conventions, which means every research group re-writes the same brittle parsing code.

`batlab`, the research framework underneath this platform, exists to fix the last two problems directly: one standardized schema across all four datasets, and one validation methodology (leave-cell-out) applied by default to every model it trains.

## Core philosophy

**Reliable battery intelligence requires honest validation.**

A model's reported accuracy is only meaningful if the test setup answers the question you actually care about. For SOH/RUL prediction, that question is "how will this model perform on a battery it has never seen?" — not "how well does it interpolate between cycles of a battery it was partly trained on."

- **Random row-level train/test splits are misleading** for cycling data, because consecutive cycles from the same cell are nearly identical to each other. A model can memorize a cell's own trajectory and still score well on a held-out *row* from that same cell, without having learned anything that transfers to a new cell.
- **Leave-cell-out (LCO) validation** holds out entire cells, never seen in training, and reports accuracy only on those. This is the harder, more honest question, and it's the platform's default — not an opt-in mode.
- **Reproducibility matters** because a claimed R² is worthless if it can't be checked. Every number this platform reports traces back to a runnable notebook or a benchmark manifest, not a number typed into a table.

## Platform capabilities

The platform is organized into five engineering modules:

**Battery Data Standardization**
- Unified cycle/summary schema across all four supported datasets
- Auto-downloading, checksum-verified dataset loaders (`batlab.datasets`)
- Schema validation that raises a specific `SchemaError` on any violation, rather than silently passing malformed data downstream

**Battery Health Analytics**
- State-of-health (SOH) estimation from measured capacity
- Capacity fade analysis and fade-rate characterization
- Internal-resistance growth tracking
- Degradation trajectory analysis across a cell's full cycle life

**Machine Learning**
- Literature-cited feature engineering (`batlab.features`)
- A trivial cycle-count baseline model, kept as an honesty check, not just a headline model
- Gradient-boosted regression tree (GBRT) models for point estimates
- Remaining useful life (RUL) prediction
- Quantile regression for uncertainty estimation around each RUL prediction

**Battery Diagnostics**
- dQ/dV differential-capacity analysis
- Knee-point and other aging-indicator detection
- Anomaly-threshold flagging against expected degradation behavior

**Lifecycle Intelligence**
- Second-life viability evaluation, with cited and illustrative-assumption economics kept explicitly separate
- An EU Battery Passport (Regulation 2023/1542 field structure) generator
- Fleet-level lifecycle tracking across multiple cells

## Architecture

```
Data Layer                    batlab.datasets
    │                         standardized schema, checksum-verified downloads
    ▼
Feature Engineering           batlab.features
    │                         literature-cited stress/aging features, dQ/dV, knee detection
    ▼
ML Models                     batlab.models
    │                         GBRT + quantile regressors, leave-cell-out validated by default
    ▼
Analytics Engine              batlab.validation, src/*
    │                         SOH/RUL, anomaly thresholds, deployment sizing, second-life economics
    ▼
Visualization / Application   app/
                               Streamlit dashboard, battery passport, fleet view, live monitor
```

Each layer is independently usable — `batlab` runs standalone as a Python library (see [Installation](#installation)), and the Streamlit application in `app/` is a consumer of it, not the other way around.

## Methodology

Every result this platform produces falls into exactly one of five categories, and the intent is that it's never ambiguous which one you're looking at:

| Category | Examples here | How it's produced |
|---|---|---|
| **Measured** | Voltage, current, capacity, temperature, cycle count | Read directly from the four public datasets' raw files into the standardized schema — never modified |
| **Derived** | SOH, capacity fade rate, resistance growth, dQ/dV peaks, knee point | Deterministic engineering formulas applied to measured data — the exact formula behind each one is documented in [`METHODOLOGY.md`](METHODOLOGY.md) |
| **Predicted (ML)** | RUL point estimate, RUL quantile interval | GBRT / quantile models trained on engineered features, always leave-cell-out validated |
| **Simulated** | PVGIS-driven solar yield in the Solar + Storage Sizing calculator, PyBaMM-based physics capacity projection | Physics-based or third-party-API-driven simulation of a process, not a direct sensor reading |
| **Illustrative assumption** | Second-life resale value, install-cost presets used in payback/NPV calculations | Values with no citation are labeled `"Illustrative — not sourced"` in the UI, explicitly distinct from `"Cited estimate"` values that do have one |

## Validation

[`notebooks/02_data_leakage.ipynb`](notebooks/02_data_leakage.ipynb) reproduces this on real data, using the same 4 NASA PCoE cells and the same GBRT model:

| Validation method | SOH R² | What it actually measures |
|---|---|---|
| Naive random row-level split | **0.998** | How well the model interpolates within cells it has already partly seen |
| Leave-cell-out (this platform's default) | **0.806** | How well the model generalizes to a cell it has never seen |
| Trivial baseline (cycle number → SOH, no engineered features), same LCO folds | 0.603 | The floor — how much of the 0.806 is the smooth shape of aging curves, not the model |

The 0.998 is real and reproducible — and it's also the wrong number to report, because it doesn't answer the question that matters for a deployed model. The honest 0.806 is what `batlab.validation.run_lco` reports by default. Comparing it against the 0.603 trivial-baseline floor under the identical fold structure shows the engineered features are worth a real **+0.203** — not the full 0.806.

This is public-data validation, not industrial validation — the 0.806 describes generalization across 4 NASA cells, not across a manufacturer's fleet. Treat it as evidence the methodology is sound, not as a number that transfers directly to a different chemistry or duty cycle.

## Demo Application

`app/` is a Streamlit application, built on `batlab` — an engineering prototype demonstrating what the platform's analytics look like assembled into an engineer-facing tool, not a production deployment:

- A per-cell **battery health dashboard** — SOH/RUL, degradation trajectory, dQ/dV, anomaly flags
- A **fleet monitoring** concept view across multiple cells at once, plus a simulated live-telemetry Monitor page
- An **EU Battery Passport** generator
- A **second-life / Solar + Storage Sizing calculator** — a real hour-by-hour (8760 hours/year) dispatch simulation against PVGIS solar data, with temperature-aware battery derating cited to real cell documentation (BU-410), not a monthly approximation

Currently demonstrated using public datasets and simulated telemetry. Real BMS integration requires real hardware data.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

See [`docs/history.md`](docs/history.md) for its full build history and architecture.

## Roadmap preview

**Phase 1 — Research foundation** *(current)*
The `batlab` library itself: standardized dataset loaders for all four datasets, literature-cited feature engineering, leave-cell-out-validated GBRT/quantile models, and reproducible benchmark manifests. This phase is what's installable and tested today.

**Phase 2 — Industrial analytics** *(partially delivered)*
Turning per-cell diagnostics into fleet- and deployment-level decision support — the demo app's fleet view, EU Battery Passport, and Solar + Storage Sizing / second-life economics calculator are working examples of this phase, built on public data and clearly labeled assumptions rather than a live industrial dataset.

**Phase 3 — Digital twin architecture** *(not started)*
A defined architecture connecting a cell's measured history, its derived health indicators, and a physics-based degradation model (an early PyBaMM-based capacity projection exists as one candidate building block) into a single continuously-updated representation. No such architecture exists yet — a physics projection module is not a digital twin.

**Phase 4 — Real-time battery integration** *(not started)*
Validating the existing "real API shape, never proven against a live account" Victron VRM and Orion Jr2 adapters (`src/bms_connectors.py`) against an actual account, and pointing the Live Monitor page's MQTT stream (`src/mqtt_stream.py`, `src/live_feed.py`) at real telemetry instead of its current simulated replay feed. This is the honest prerequisite for any future live-BMS or live-ESS capability — not a rebrand of the existing simulation.

## Limitations

Being explicit about what this platform is not, as of today:

**Currently:**
- No proprietary factory or manufacturer data — only the four public datasets listed above
- No real vehicle or stationary-storage fleet — fleet views operate on the same public-dataset cells or an honestly-labelled synthetic fleet
- No validated live BMS connection — the Victron/Orion adapters exist in code but have never been run against a live account, and Live Monitor's telemetry stream is a simulated replay, not a real one

**Future, contingent on real access:**
- Industrial or research partnerships providing real operational data
- Real telemetry replacing the current simulated Live Monitor feed
- Hardware validation of the existing BMS connector adapters against actual devices

See [`docs/history.md`](docs/history.md) for the fuller production-readiness roadmap this summary is drawn from.

## Installation

```bash
pip install -e ".[severson,oxford,calce]"   # extras are only needed for those loaders' parsers
```

Core `batlab` depends only on `pandas`, `numpy`, `scikit-learn`, `scipy`, and `requests`.

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

The full ten-minute tour is [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb). See [`docs/datasets/`](docs/datasets/index.md) for each dataset's schema, citation, and license, and [`batlab/datasets/CONTRIBUTING.md`](batlab/datasets/CONTRIBUTING.md) for adding a fifth loader.

For development (`pytest` plus every loader/notebook/docs extra):

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

```python
import batlab
print(batlab.cite())
```

Or see [`CITATION.cff`](CITATION.cff) for the full academic citation metadata, including a registered ORCID and a Zenodo-archived DOI for the current release. A JOSS paper draft is at [`paper/paper.md`](paper/paper.md) (not yet submitted — see its TODOs).

## License

MIT for this repository's code — see [`LICENSE`](LICENSE). Each dataset loader interoperates with a third-party public dataset that carries its own separate license; see [`docs/datasets/`](docs/datasets/index.md) or `batlab.cite(dataset=...)` before redistributing any dataset's data.
