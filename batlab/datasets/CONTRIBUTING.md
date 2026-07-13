# Adding a new dataset loader

This library ships loaders for NASA PCoE, Severson 2019, Oxford Path-Dependent 2020, and CALCE CS2. Adding a fifth follows the same mechanical checklist every time — this doc is that checklist, written so a new loader is a small, reviewable PR rather than a design decision.

## 1. Decide dense-cycle vs. sparse-checkpoint

Look at what the source actually publishes:

- **Dense per-cycle data** (hundreds+ rows per cell, one per real charge/discharge cycle) → `kind="cycle"`. This is what NASA, Severson, and CALCE are.
- **Sparse reference-test checkpoints** (a handful of Reference Performance Test points per cell, not real per-cycle data) → `kind="checkpoint"`. This is what Oxford is — see `batlab/datasets/oxford.py`'s module docstring for the reasoning, and don't force sparse data into the dense shape just to reuse more downstream code; it misrepresents how much data actually exists per cell.

## 2. Match the schema contract

Read `batlab/datasets/schema.py` before writing anything. Your loader's public function must return `{cell_id: DataFrame}` where every DataFrame:

- Has the required columns for its `kind` (`REQUIRED_CYCLE_COLUMNS` or `REQUIRED_CHECKPOINT_COLUMNS`).
- Uses `compute_soh_pct()` for `soh_pct` — don't reimplement the formula; that's the exact bug class this project has hit before (a formula reimplemented slightly differently in each new loader).
- Sets every key in `REQUIRED_ATTRS` on `df.attrs` (`cell_id`, `source`, `chemistry`, `citation`, `license`).
- Passes `validate_schema(df, kind=...)` — call it in your own tests, don't just assume.

Only add optional columns (`OPTIONAL_COLUMNS` in `schema.py`) when the source genuinely supports them. Never fabricate a value to fill a column — omit it instead.

## 3. Citation and license — required, not optional

Add an entry to `batlab/cite.py`'s `_DATASETS` dict: a real BibTeX block (verify author names, year, DOI/URL — don't guess) and the dataset's actual license/redistribution terms, verbatim or accurately summarized. Your loader's `df.attrs["citation"]` must be the exact key you added there, so `cite(dataset=df.attrs["citation"])` resolves for every cell your loader returns.

If you're not fully confident of a citation detail (an exact page range, an ambiguous license clause), say so in the BibTeX comment or docstring rather than presenting false precision — see `nasa`/`severson2019`/`oxford2020`'s entries in `cite.py` for the pattern this library follows.

## 4. Download policy

- If the source has a stable, scriptable, ungated download URL (NASA, Severson, Oxford): auto-download on first call, cache to `data/raw/{source}/`, and never commit the raw download to the repo (only small extracted per-cell CSVs may be committed, same as the existing loaders).
- If the source requires manual registration, an account, or has no stable per-file URL you can verify (CALCE): don't fake a downloader. Raise a specific, instructive exception (see `CalceDataNotFoundError` in `batlab/datasets/calce.py`) telling the user exactly where to get the files and exactly what local directory structure to place them in. A loader that silently does nothing, or one that breaks the moment the source website is restructured, is worse than one that's honest about needing a manual step.

## 5. Tests — fixture-based, no real data committed

Never commit real downloaded data as a test fixture, regardless of the dataset's license. Instead:

- Build a small synthetic DataFrame (or, for a file-format loader, a tiny synthetic file — see `tests/fixtures/`) that mimics the *shape* of the real raw format: same column names, a handful of rows, enough to exercise the parsing logic.
- Test the pure-logic pieces (capacity extraction, column renaming, cycle renumbering) independently of any file I/O where possible — see `batlab/datasets/oxford.py`'s `_capacity_ah_from_table()` / `tests/test_oxford_loader.py` for the pattern.
- Test `validate_schema()` passes on your fixture's output.
- If your loader can raise an instructive not-found error (manual-download case), test that it actually does, with a useful message — not just that it raises `Exception`.

## 6. Wire it in

- Export your public `load_X_cells()` function from `batlab/datasets/__init__.py`.
- If it needs a dependency beyond `batlab`'s core (`pandas`/`numpy`/`scikit-learn`/`scipy`/`requests`), add it as an optional extra in `pyproject.toml`'s `[project.optional-dependencies]` (see `severson`, `oxford`, `calce`) — never as a hard dependency of `batlab` itself, since most users of any one loader don't need every other loader's parser.
- Add your dataset to this project's `README.md` / docs dataset table (see `docs/datasets/` once the docs site exists) so it's actually discoverable.

That's the whole checklist. If a step above doesn't apply to your source (e.g. no license file exists at all), say so explicitly in your loader's docstring rather than leaving it silently unaddressed.
