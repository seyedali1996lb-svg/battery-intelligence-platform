# Datasets

Every loader in `batlab.datasets` returns `{cell_id: DataFrame}` in one standardized schema — see [`batlab.datasets.schema`](../api/datasets.md) for the exact column contract. Call `batlab.datasets.validate_schema(df, kind=...)` on any DataFrame to check it satisfies that contract, whether it came from a built-in loader or your own data (see the [bring-your-own-data notebook](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/notebooks/04_bring_your_own_data.ipynb)).

| Dataset | Loader | Chemistry | Cells | Kind | Auto-download | Citation key |
|---|---|---|---|---|---|---|
| NASA PCoE | `batlab.datasets.nasa` | LiCoO2 | 4 | cycle | Yes | `nasa` |
| Severson 2019 | `batlab.datasets.severson` | LFP | 12 | cycle | Yes | `severson2019` |
| Oxford Path-Dependent 2020 | `batlab.datasets.oxford` | NCA | 12 | checkpoint (~8-14/cell, sparse) | Yes | `oxford2020` |
| CALCE CS2 | `batlab.datasets.calce` | LiCoO2 | any (manual placement) | cycle | No — manual download | `calce` |
| Zhu 2022 voltage relaxation | `batlab.datasets.zhu2022` | NCM+NCA | 9 | cycle (~900-1000/cell, dense) | Yes | `zhu2022` |

Get the full citation (BibTeX + license) for any dataset with `batlab.cite(dataset=<citation key>)`.

## NASA PCoE

Saha & Goebel (2007), NASA Ames Prognostics Center of Excellence. 4 cells (B0005, B0006, B0007, B0018), 18650-format LiCoO2, ~2 Ah nominal, cycled to failure at 24°C. Auto-downloads a ~200 MB ZIP on first call (SHA-256 checksum-verified — see `batlab.datasets._integrity`), caches extracted per-cycle CSVs to `data/raw/`. License: public domain / US government work.

```python
from batlab.datasets import load_nasa_cells
cells = load_nasa_cells()
```

## Severson 2019

Severson et al., *Nature Energy* 2019 — the original fast-charging cycle-life dataset. This loader uses 12 cells from Batch 1 spanning 4 cycle-life bands (short/medium/long/extra). Auto-downloads a ~2.9 GB MATLAB v7.3 (HDF5) file on first call (SHA-256 checksum-verified); requires the `severson` extra (`h5py`). License: research use per data.matr.io terms.

```python
from batlab.datasets import load_severson_cells
cells = load_severson_cells()
```

## Oxford Path-Dependent 2020

Raj et al., *Batteries & Supercaps* 2020 — 12 NCA cells across 4 groups, path-dependence study. **Sparse, not dense**: only ~8-14 Reference Performance Test checkpoints per cell, not per-cycle data — returned in `kind="checkpoint"` schema (`checkpoint_index`, not `cycle_number`) and deliberately not run through the GBRT+LCO pipeline (too few points to honestly validate). Each group's zip (~800 MB) is SHA-256 checksum-verified on download. Requires the `oxford` extra (`mat-io`). License: ODC-ODbL v1.0.

```python
from batlab.datasets import load_oxford_cells
cells = load_oxford_cells()
```

## CALCE CS2

University of Maryland CALCE — 1.1 Ah prismatic LiCoO2 cells, one of the most-cited public li-ion cycling datasets. Verified against a real downloaded file (CS2_35): parses to physically plausible values (~1.0-1.1 Ah capacity, ~0.99 coulombic efficiency) and passes `validate_schema()` — see `batlab/datasets/calce.py`'s module docstring for exactly what was and wasn't checked (one cell verified, not all of them). **No auto-download**: direct per-cell URLs do exist and are reachable, but unlike Severson's single shared batch file, CALCE is one download per cell with no obviously-right default for "download everything," so `load_calce_cells()` raises `CalceDataNotFoundError` with exact instructions when nothing is found locally, rather than guessing a default. Requires the `calce` extra (`openpyxl`).

```python
from batlab.datasets import load_calce_cells, CalceDataNotFoundError
try:
    cells = load_calce_cells()
except CalceDataNotFoundError as exc:
    print(exc)  # tells you exactly where to download from and where to place files
```

## Zhu 2022 voltage relaxation

Zhu et al., *Nature Communications* 2022 — 9 commercial 18650 cells with a blended NCM+NCA cathode, cycled at 25 °C (0.5C charge / ~1C discharge) as part of a study on capacity estimation from voltage relaxation. Dense per-cycle data (~910-1030 cycles/cell, first-cycle capacity ~2.47-2.50 Ah fading to ~62-67% SOH), so it runs through the GBRT + LCO pipeline as a genuinely new chemistry for this library. Auto-downloads a ~356 MB ZIP from Zenodo on first call (SHA-256 checksum-verified; matches the archive's own published md5), then derives per-cycle discharge capacity from the raw `Q discharge/mA.h` column — taking the largest single discharge-run delta per cycle rather than the per-cycle maximum, because a handful of characterization cycles contain multiple discharge segments (see `batlab/datasets/zhu2022.py`'s docstring). Per-cell summaries are cached as small CSVs. License: CC BY 4.0.

```python
from batlab.datasets import load_zhu2022_cells
cells = load_zhu2022_cells()
```

## Adding a 6th loader

See [`batlab/datasets/CONTRIBUTING.md`](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/blob/master/batlab/datasets/CONTRIBUTING.md) for the mechanical checklist: schema contract, citation/license, download policy, fixture-based tests.
