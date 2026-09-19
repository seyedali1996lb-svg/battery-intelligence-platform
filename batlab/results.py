"""Typed schemas for batlab's structured results.

Every public batlab function that returns a bundle of numbers returns it as a
plain ``dict`` — JSON-serializable, printable, diff-able, extendable, and
readable without importing this module. What this module adds is the *schema* of
those dicts, so a consumer can type their own code and get completion and static
checks:

    from batlab.results import LcoResult, as_lco

    def summarise(lco: LcoResult) -> str:
        return f"LCO SOH R² = {lco['soh_r2']:.3f}"

    summarise(as_lco(batlab.benchmark(cells)))

**Why the functions still return plain dicts.** An earlier draft annotated
``run_lco()``'s return type as ``LcoResult`` directly. That is wrong for this
codebase, and the type checker said so: a ``TypedDict`` has a *closed* key set
and cannot be mutated, but callers legitimately do both — ``manifest.py``
enriches an LCO result with manifest metadata (``manifest_seed``,
``environment_diff``, …), and ``run_lco``'s own result is passed to helpers
typed as ``dict``. A ``TypedDict`` return type would have made every one of
those a type error, in library *and* app code. So the runtime type stays
``dict[str, Any]``, exactly as before: this is an annotation and documentation
release, not a breaking one, and it cannot change a number.

**How to get the typing anyway.** The ``as_*()`` bridges below are identity
functions with a precise return type: they hand a checker the schema without
touching the object. ``as_lco(anything_dict_shaped)`` returns the same object,
typed as :class:`LcoResult`.

**Everything is optional, on purpose.** Every ``TypedDict`` here is declared
``total=False``: a result a given fleet cannot compute omits or nulls the key
rather than inventing a value — RUL on a fleet with no observed end-of-life
labels is the canonical case. Consumers must treat each key as possibly absent
and each scalar as possibly ``nan``/``None``. That is deliberate: it is the
shape honesty takes in this library.

**The schemas are tested, not aspirational.** ``tests/test_public_api.py``
asserts every declared field of :class:`LcoResult` and
:class:`QuantileLcoResult` is actually produced by a real run, and names any
field a real run produces that the schema does not declare. A schema that drifts
from the payload is a schema that lies.

See ``docs/api_stability.md`` for the compatibility contract these schemas are
part of.
"""

from __future__ import annotations

from typing import Any, TypedDict

__all__ = [
    "CalibrationReplayResult",
    "ConfidenceIntervals",
    "FoldCacheSummary",
    "LcoResult",
    "QuantileLcoResult",
    "as_lco",
    "as_quantile_lco",
]


class FoldCacheSummary(TypedDict, total=False):
    """What the leave-cell-out fold cache did for one call.

    Provenance about the *computation*, not about the model: ``hits`` folds were
    replayed from disk, ``fitted`` ones were trained in this call. Both paths
    produce identical numbers by construction (see
    ``batlab.validation.fold_cache``), so this is how a caller reports
    "46/46 reused" instead of inferring it from a layer that finished in 0.4 s.

    ``dir`` is the cache ROOT — the directory a human inspects — while ``key``
    names the key directory *inside* it holding this run's folds.
    """

    mode: str
    enabled: bool
    key: str | None
    dir: str
    hits: int
    fitted: int


class ConfidenceIntervals(TypedDict, total=False):
    """Fold-level percentile bootstrap intervals, resampling the CELL.

    Rows inside a cell are strongly correlated, so a row-level bootstrap would
    fake precision. A fleet with fewer than two evaluable folds carries ``None``
    here rather than a fabricated bracket.
    """

    soh_r2: tuple[float, float] | None
    soh_mae: tuple[float, float] | None
    rul_r2: tuple[float, float] | None
    rul_mae: tuple[float, float] | None


class LcoResult(TypedDict, total=False):
    """Schema of :func:`batlab.validation.run_lco` / ``batlab.benchmark``.

    ``rul_r2`` / ``rul_mae`` are scored on **observed-end-of-life rows only** —
    the only population a RUL skill claim can be built from. Rows whose cell
    never reached the EOL threshold in-window carry a closed-form
    *extrapolated* label, and those are reported separately as
    ``rul_extrapolated_*``: a formula-recovery diagnostic, never mixed into the
    headline, never allowed to set ``rul_reliable``.
    """

    fingerprint: Any
    fold_cache: FoldCacheSummary

    # True only when EVERY cell carried the optional physics-calibration feature
    # block (`PHYSICS_FEATURE_COLUMNS`), which lives in the demo app's src/ and is
    # therefore present or absent depending on the ENVIRONMENT rather than the
    # data. The two populations are measurably different (SOH R² 0.9580 vs 0.9471
    # on the four NASA cells), so a number is only comparable to another that
    # reports the same value here — and the fold cache keys on it so the two can
    # never replay each other's folds.
    physics_features: bool

    soh_r2: float
    soh_mae: float

    # None/nan when no fold has observed-EOL rows to score.
    rul_r2: float | None
    rul_mae: float
    rul_reliable: bool
    rul_label_coverage: float
    n_rul_observed_rows: int
    n_rul_extrapolated_rows: int
    rul_extrapolated_r2: float | None
    rul_extrapolated_mae: float | None

    confidence_intervals: ConfidenceIntervals

    # {cell_id: {...}} — the per-fold metrics behind the aggregate. Typed as Any
    # on purpose: it is an open diagnostic bag, and narrowing it would freeze a
    # structure that exists to be read by humans and by the app.
    per_cell: dict[str, Any]


class QuantileLcoResult(TypedDict, total=False):
    """Schema of :func:`batlab.validation.run_lco_quantiles`.

    The measured coverage of the served 80% Q10/Q90 interval: ``rul_interval_coverage``
    is what the quantile models actually delivered on held-out cells, and
    ``global_e_star`` is the pooled conformal widening (clamped at zero — the
    platform widens an interval, it never narrows one). A fleet whose RUL labels
    are all extrapolated carries ``not_evaluable`` and no fitted numbers.
    """

    rul_interval_coverage: float
    rul_interval_width_mean: float
    pooled_conformity_scores: Any
    global_e_star: float | None
    recalibrated_coverage: float
    recalibrated_width_mean: float
    rul_r2: float
    rul_mae: float
    rul_reliable: bool
    rul_label_coverage: float
    per_cell: dict[str, Any]

    # Present only on the short-circuit path (no observed labels anywhere). The
    # string names WHY, so a caller renders a reason rather than a blank.
    not_evaluable: str


class CalibrationReplayResult(TypedDict, total=False):
    """Schema of :func:`batlab.validation.recalibrate_lco_intervals`."""

    nominal: float
    raw: dict[str, Any]
    recalibrated: dict[str, Any]
    skipped: dict[str, Any]


def as_lco(result: dict[str, Any]) -> LcoResult:
    """Typed view of a ``run_lco()`` result. Identity function; mutates nothing."""
    return result  # type: ignore[return-value]


def as_quantile_lco(result: dict[str, Any]) -> QuantileLcoResult:
    """Typed view of a ``run_lco_quantiles()`` result. Identity; mutates nothing."""
    return result  # type: ignore[return-value]
