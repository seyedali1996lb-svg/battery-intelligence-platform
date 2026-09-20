# Command line

The library ships a small CLI so a shell, a CI job or a Makefile can get a
validated number without writing Python. Every verb is a thin wrapper over the
public API — the same functions, called the same way — so a CLI result and a
scripted result cannot disagree (a test asserts exactly that).

```bash
pip install battery-lab
batlab --help          # or: python -m batlab --help
```

## `batlab version`

Prints the library version. Include it in a bug report.

```bash
$ batlab version
batlab 0.2.0
```

## `batlab cite`

BibTeX for the library, or for one dataset including its license:

```bash
batlab cite
batlab cite --dataset nasa
```

```python
# equivalent:
import batlab; print(batlab.cite(dataset="nasa"))
```

## `batlab datasets`

Lists the five supported datasets with the loader behind each one. Nothing is
downloaded — the fleet is only loaded (and may be downloaded, checksum-verified)
with `--load`, which then reports a cell count per fleet:

```bash
$ batlab datasets
Five public datasets, one standardized schema

  nasa      batlab.datasets.nasa:load_nasa_cells
  severson  batlab.datasets.severson:load_severson_cells
  ...

$ batlab datasets --load
  nasa      4 cells
  severson  46 cells
```

A loader whose raw source files are absent reports `unavailable: …` rather than
aborting the listing — `oxford` and `calce` in particular require files placed
locally, and a listing that dies on the first absent dataset is useless.

## `batlab benchmark`

Leave-cell-out validation of batlab's own GBRT on one fleet.

```bash
$ batlab benchmark --dataset nasa
loading batlab.datasets.nasa:load_nasa_cells ...
  4 cells: B0005, B0006, B0007, B0018
leave-cell-out validation (seed=42) ...

leave-cell-out (new-cell generalization)
  SOH R²   0.9471   MAE 0.9875
  RUL R²   0.4216   MAE 8.6698
  folds    0 reused / 4 fitted (cache on)
```

| Flag | Meaning |
|---|---|
| `--dataset {nasa,severson,zhu2022,oxford,calce}` | Friendly alias; default `nasa`. |
| `--loader module:function` | An explicit loader instead of an alias — the same spelling `python -m batlab.harness` uses. |
| `--seed N` | Fold seed (default 42). Folds are seeded individually, so the numbers are reproducible at any worker count. |
| `--out FILE` | Write the full result as JSON (numpy scalars/arrays encoded). |
| `--predictions` | Also compute each fold's out-of-fold rows: slower, larger report. |

`RUL R²` is printed only when the run's `rul_reliable` flag is true. When it is
not, the CLI says **not evaluable** and names the observed-label count, because
"we cannot measure this on this fleet" and "we measured it and it was bad" are
different facts — see `METHODOLOGY.md` §2a (repository root).

## Exit codes

`0` on success; `1` on a loader that fails to import, a fleet with no cells, or a
malformed `--loader` spec (each with a message naming the problem). Argument
errors exit `2` via `argparse`.

## Grading your own model

That is a bigger tool with its own flags and its own report format:

```bash
python -m batlab.harness --loader batlab.datasets.nasa:load_nasa_cells \
    --gate floors.json --seal out/bundle
```

See the [validation harness guide](harness.md). This CLI deliberately does not
duplicate it — `batlab benchmark` grades *batlab's* model, the harness grades
*yours*, and keeping them separate keeps both honest.
