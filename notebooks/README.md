# Worked notebooks

Four notebooks, each runnable top-to-bottom on a fresh clone after `pip install -e ".[notebooks]"`:

| Notebook | What it shows |
|---|---|
| `01_quickstart.ipynb` | Load NASA cells, engineer features, leave-cell-out validate, train, predict SOH/RUL, plot — the ten-minute evaluation. |
| `02_data_leakage.ipynb` | Reproduces the row-level-split R² illusion (~0.998) vs. the honest leave-cell-out number (~0.81) on the same 4 NASA cells, and explains why — then checks how much of that honest number a trivial linear-fit baseline (cycle_number → SOH, no feature engineering) already explains on its own. |
| `03_knee_detection.ipynb` | Knee detection on a real Severson curve, plus an honest negative result on Oxford's sparse checkpoint data (too few points to trust a knee call). |
| `04_bring_your_own_data.ipynb` | Maps a CSV with realistic, non-matching column names into batlab's schema, validates it, runs the full pipeline. |

All four are executed in CI on every push (see `.github/workflows/ci.yml`'s `notebooks` job, driven by `_run_ci.py` in this directory) — a broken notebook fails the build the same way a broken test would.

No network access is required: NASA, Severson, and Oxford's small extracted CSVs are committed to `data/raw/`, and notebook 4 uses a synthetic fixture CSV committed at `sample_data/my_lab_cell.csv`.

## Skipping a cell in CI

If a future notebook needs something CI can't or shouldn't do — a large one-time download this repo doesn't commit, for example — tag that cell `skip-ci` (cell metadata → tags, in JupyterLab's cell toolbar, or `cell.metadata["tags"] = ["skip-ci"]` if editing the `.ipynb` JSON directly). `_run_ci.py` removes tagged cells before executing the notebook, not just marks them as skipped — so don't rely on a later cell reading a variable a skipped cell would have defined.

## Regenerating outputs

Notebooks are committed with their outputs (so they render meaningfully on GitHub without execution). After editing a notebook, re-run it before committing:

```
python notebooks/_run_ci.py
```
