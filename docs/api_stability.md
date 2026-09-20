# API stability

What `batlab` promises about its interface, and what it explicitly does not — so
that "is it safe to depend on this?" has a written answer instead of a guess.

## What is public

| Surface | Stability |
|---|---|
| Names in `batlab.__all__` and in each public submodule's `__all__` (`batlab.datasets`, `batlab.features`, `batlab.models`, `batlab.validation`, `batlab.harness`, `batlab.results`, `batlab.cli`) | **Public.** Covered by this policy. |
| The `TypedDict` schemas in `batlab.results` | **Public**, and *optional by construction* — see below. |
| The CLI verbs and flags in [`docs/cli.md`](cli.md) | **Public.** Flags are added freely; existing flags keep their meaning. |
| Anything prefixed with `_` (`batlab._parallel`, `batlab._deprecation`, `run_lco`'s private helpers) | **Private.** May change in any release, including a patch. |
| The demo application (`app/`), the REST layer (`src/`), the Streamlit pages, the React SPA | **Not part of the library API.** They are consumers of `batlab`, shipped in this repository for the demo, and their internals move with the app. |
| The on-disk cache formats (`.cache/lco/`, `.cache/bundles/`) and their keys | **Not API.** Delete them at any time; the cache is an optimization, and every entry is re-derivable. |

A useful rule of thumb: if it is not in an `__all__`, you may import it, but you
are on your own when it changes.

## Versioning

Semantic versioning, applied to the public surface above:

- **Patch** — bug fixes; no new public names; no behaviour change a caller could
  observe other than the bug being fixed.
- **Minor** — new public names, new optional parameters, new CLI flags, new
  dataset loaders, new schema fields. Existing code keeps working.
- **Major** — a removal, a rename, or a change to what an existing name returns.

`0.x` caveat, stated plainly: before 1.0 a *minor* bump may contain a breaking
change if one is genuinely necessary. The licence for that is bounded by the
deprecation policy below — a break still gets a deprecation cycle, it just may
arrive at 0.4.0 instead of 1.0.0. `1.0` is reserved for the point at which the
public surface above is one we are prepared to freeze.

## Deprecation policy

A deprecated public name:

1. **keeps working**, and returns exactly what it returned before;
2. emits a `DeprecationWarning` naming what to use instead, the version that
   deprecated it, and the version that removes it;
3. is listed in `CHANGELOG.md` (repository root) with the same three facts;
4. survives **at least one minor release** before removal, and is removable in a
   single commit on the announced version — no behavioural shim hiding behind an
   expired warning.

The mechanism is `batlab._deprecation` (`warn_deprecated()` / `deprecated()`),
and `tests/test_public_api.py` asserts its behaviour, so the policy cannot rot
into documentation that nothing enforces.

```python
from batlab import _deprecation

@_deprecation.deprecated(
    "run_lco_legacy", since="0.3.0", removed_in="0.5.0", alternative="run_lco",
)
def run_lco_legacy(cells): ...
```

!!! note "How to read a result"
    Results are ordinary `dict`s. Every schema in `batlab.results` is declared
    `total=False`, because a result a fleet cannot compute *omits* the key rather
    than inventing a value — RUL on a fleet with no observed end-of-life labels is
    the canonical case. Consumers must treat every key as possibly absent and
    every scalar as possibly `nan`/`None`. That is the shape honesty takes here,
    not an oversight.

    For static typing at a call site, use the identity bridge:

    ```python
    from batlab.results import as_lco

    lco = as_lco(batlab.benchmark(cells))
    reveal_type(lco["soh_r2"])   # float
    ```

## Environment-dependent inputs

One class of input is not in the data and not in the code: **what happens to be importable**.
Until 0.2.0 that reached into a feature column — `build_features()` imported the
physics-calibration block from the demo application's `src/`, so a pip-installed `batlab`
produced different numbers from a checkout of the demo app: on four NASA cells, SOH R²
**0.9580** without the block and **0.9471** with it. A library whose numbers move with the
caller's import path is not a library.

The block now ships **inside the library** (`batlab.features.physics_calibration`, feature set
`v13-features-owned-physics`), so there is one number in every environment — measured
bit-identical (`soh_r2 = 0.9470975121947384`) in a bare process and in an app-importable one,
with `tests/test_feature_environment_inputs.py` pinning it, including a subprocess run with no
`src/` anywhere on the path.

What remains environment-dependent in that block is a **declared optional extra**: PyBaMM
(`pip install "battery-lab[physics]"`) supplies only `physics_spm_capacity_ah`, a display-only
column that is not in `FEATURE_COLUMNS` and therefore cannot move a model number.

That is handled rather than hidden:

- eligibility reads the frame's own schema attrs (`df.attrs["source"]` / `["chemistry"]`, both
  `REQUIRED_ATTRS`) against `physics_calibration.ANCHOR_PARAM_SETS`, so the population is a
  property of the DATA — a loader fix or one different row can change it, a `sys.path` cannot;
- every result carries `physics_features: bool` (`batlab.results.LcoResult`), true only when
  **every** cell in the run was calibrated; a mixed fleet is never rounded up;
- two runs are comparable only when they agree on that flag — the same rule as
  `FEATURE_VERSION`, applied to the one input a feature version cannot see;
- the leave-cell-out fold cache keys on it, so a fold fitted with the block is never replayed
  for a run that did not have it;
- `METHODOLOGY.md` §3 carries the measured before/after and the loader-side fix that keeps the
  demo app on the library's path (`src/data_loader.build_battery` now declares provenance).

## What this policy does not cover: the numbers

**Reported metrics are not API.** A retrained model on the same data can move a
metric, and that is not a semantic-versioning event — it is a scientific one:

- A change to how features are computed bumps `FEATURE_VERSION`
  (`batlab.validation.manifest`), which invalidates the fold cache and flags the
  registry row as a different population.
- A movement beyond tolerance in a headline metric recorded in
  `tests/metric_gate_expectations.json` **fails CI** until it is recorded with a
  reason via `scripts/update_metric_baselines.py` — an append-only change log
  that shows up in review.
- A metric that *stops being computable* also fails the gate, unless the
  expectation explicitly allows it.

So the promise is not "these numbers never change". It is "these numbers cannot
change silently, and if they do, the diff says why".

## Checking your version

```bash
batlab version                       # or: python -m batlab version
pip show battery-lab                 # the installed distribution
```

If you report a bug against a release, include that output: the distribution name
(`battery-lab`) and the import name (`batlab`) differ on purpose, and PyPI's
`batlab` is an unrelated hardware library.
