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
3. is listed in [`CHANGELOG.md`](../CHANGELOG.md) with the same three facts;
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
`build_features()` opportunistically uses the physics-calibration feature block, whose module
lives in the demo application's `src/`, not in this library. A pip-installed `batlab` therefore
produces different numbers from a checkout of the demo app — measured on four NASA cells,
SOH R² **0.9580** without the block and **0.9471** with it.

That is handled rather than hidden:

- every result carries `physics_features: bool` (`batlab.results.LcoResult`), true only when
  **every** cell in the run carried the block;
- two runs are comparable only when they agree on that flag — the same rule as
  `FEATURE_VERSION`, applied to the one input a feature version cannot see;
- the leave-cell-out fold cache keys on it, so a fold fitted with the block is never replayed
  for a run that did not have it;
- `METHODOLOGY.md` §3 states the measured difference and the plan to move the module into the
  library (which would be a `FEATURE_VERSION` bump, not a quiet change).

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
