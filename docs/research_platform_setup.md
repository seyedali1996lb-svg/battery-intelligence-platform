# Research Platform setup path

Confirms, concretely, the claim behind the [product direction](product_direction.md)
decision: an external researcher can clone this repository, bring their own
public-dataset-shaped CSV, and get a working analysis — without a Streamlit
account, an API key, a BMS connection, or anything proprietary.

There are two valid entry points, depending on whether you want the library
or the dashboard. Both were exercised, not just read, before writing this
note.

## Path A — `batlab` as a Python library

```bash
git clone https://github.com/seyedali1996lb-svg/battery-intelligence-platform
cd battery-intelligence-platform
pip install -e ".[dev]"
```

Then open [`notebooks/04_bring_your_own_data.ipynb`](notebooks/04_bring_your_own_data.ipynb).
It takes a CSV with realistic, non-`batlab`-native column names
(`notebooks/sample_data/my_lab_cell.csv` — a small synthetic fixture, not
real cell data), renames the columns to the schema contract, validates it,
and runs it through the same `build_features` → `train_models` → `predict`
pipeline every built-in dataset loader's output goes through.

**Verified while writing this note** (not assumed from reading the
notebook): running that exact sequence against the sample fixture through
`batlab.datasets.schema.validate_schema`, `batlab.features.build_features`,
and `batlab.models.train_models` completes end-to-end with no error and
produces real metrics. The metrics themselves are poor (R² well below zero)
because the fixture is a single cell — the notebook says so explicitly in
its "One important honesty note" section: leave-cell-out validation needs
2+ cells, so a single cell only gets an in-sample chronological holdout, not
a validated generalization number. That's the honest, documented behavior,
not a bug.

## Path B — the Streamlit demo app

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Then use the **Import Data** tab (Configure page). It reads
[`docs/import_format_guide.md`](import_format_guide.md) directly and
displays it in-app (`app/_pages/import_page.py`), covering: the required/
optional column schema, minimum cell-count and cycle-count guidance tied to
what leave-cell-out validation actually needs, how to map common BMS/cycler
export formats (Neware, Arbin, Maccor, BatteryArchive) into per-cycle rows,
the DCIR-vs-EIS resistance-measurement-type pitfall (documented from a real
bug this platform hit mixing the two), and a downloadable template
(`data/import_template.csv`). Uploading a valid CSV trains a fresh model and
populates every page (Overview, Health, Fleet, Copilot, Consequences,
Recommendations, Passport) from it.

## What this does not require

Neither path touches: an Anthropic API key (Copilot falls back to template
narration without one), a BMS/CMMS/Circunomics credential, a database
beyond the local gitignored SQLite file, or any data this repository
doesn't already ship or the researcher doesn't bring themselves.

## What's still manual

There's no single "upload straight from your cycler" integration — a
researcher maps their own export format into the documented schema by hand
(or with the notebook as a template) before either path works. Automating
that mapping for arbitrary proprietary lab formats is exactly the gap that
keeps the [testing-lab analytics platform concept deferred](deferred_product_concepts.md),
not something this setup path claims to solve.
