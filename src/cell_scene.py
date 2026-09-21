"""
``CellSceneSpec`` — the host-agnostic data contract behind every 3D cell view.

Why this exists
---------------
The platform can explain a cell's degradation in numbers, in 2D charts and in
prose, but it has never been able to *show* it: which part of a cell is wearing
out, how far, and how that changes over the cell's life. This module produces
the one artifact every surface needs to do that — a pure, JSON-serializable
description of a single cell, its anatomy, and the measurements attached to
each part.

It is deliberately **framework-free**. No Streamlit, no React, no plotting
library, no three.js. The Streamlit page, the REST endpoint and the tests all
call the same builder, so two surfaces cannot disagree about a cell — the
failure this project has hit repeatedly when independent code paths compute the
same verdict.

Three consumers, one spec
------------------------
``app/_pages/battery3d.py``      builds it in-process and hands it to the renderer
``src/api.py``                   serves it as ``GET /cells/{id}/scene``
``scripts/export_scene_sample.py`` writes a committed sample for the static page
``frontend/src/scene/``          renders it; knows nothing else about this repo

Nothing new is modelled
-----------------------
Every number here already exists somewhere in the platform and is *read*, never
re-derived: the cell's own per-cycle frame, the fitted SEI/LAM physics, the ML
mechanism classifier's verdict, the knee detector, the hierarchical forecast
(gated by the per-cell forecast-routing verdict, so a refused route yields no
projection instead of a fabricated one), and the chemistry profile's declared
form factor. A part the source cannot speak to is reported ``available: false``
with a reason — never a zero, never an estimate.

The physics is read, not invented
---------------------------------
The two data-driven layers of the anatomy are the platform's own fitted
two-term fade law (``batlab.features.physics_calibration``)::

    SOH(n) / SOH₀ = 1 − β_sei·√n − β_lam·n        n = cycles since first cycle + 1

so the SEI film's thickness series is ``β_sei·√n`` and the lost active-material
series is ``β_lam·n``, both as percentages of the cell's initial capacity.

Those two β come from **one full-history fit** per cell
(``physics_calibration.fit_two_term_fade`` over the cell's whole recorded
record), not from the frame's per-cycle physics columns. That choice is
deliberate and was forced by the data: those columns are refit every
``REFIT_EVERY_CYCLES`` window, and on NASA's B0005 the last window's β_sei comes
out at ~1e-19 while β_lam absorbs the fade — so a per-window film thickness
would visibly *shrink* between windows, which no physical SEI film does. A
single full-history fit makes both terms monotone in n by construction, and it
is the same fit the mechanism verdict reads.

That fit has a documented weakness this spec must carry rather than hide: over a
narrow cycle range √n and n are strongly correlated, so the β_sei/β_lam split is
*weakly identified* from capacity data alone, and
``physics_calibration.dominant_mode()`` refuses to name a dominant mechanism
below ``MIN_FIT_R2_FOR_DOMINANT_MODE``. This module uses that function — not its
own comparison — for the physics verdict, publishes ``fit_r2`` beside it, and
reports ``insufficient_data`` exactly where the library does. The ML mechanism
classifier's verdict travels separately, and the two are compared here rather
than blended, because they are independent evidence.

The PyBaMM-backed ``calibrate_cell()`` path is deliberately *not* used here: its
SPM run costs seconds per cell, this spec is rebuilt on every page render and
served over HTTP, and every number this scene needs (the two betas and their r²)
comes from the pure-scipy fit that ``calibrate_cell()`` itself calls.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Bumped whenever the shape of the spec changes incompatibly. The renderer
#: asserts it (Python tests and the TS side both check this same number), so a
#: producer and a renderer cannot drift apart silently.
SCENE_SCHEMA_VERSION = 1

#: The canonical anatomy part ids. The renderer draws exactly one mesh per id
#: and the spec carries exactly one card per id — a bijection asserted by tests
#: on both sides, so geometry can never quietly stop matching the data.
MESH_PART_IDS: tuple[str, ...] = (
    "can", "cap", "vent",
    "terminal_pos", "terminal_neg",
    "tab_pos", "tab_neg",
    "cathode_sheet", "anode_sheet", "separator",
    "electrolyte", "particles", "sei_film",
)

#: Provenance vocabulary for a part's value. Deliberately the same words the
#: rest of the app uses for its provenance banners.
PROVENANCE_MEASURED = "measured"    # read straight off the cell's own record
PROVENANCE_DERIVED = "derived"      # computed from measured quantities
PROVENANCE_FITTED = "fitted"        # produced by a curve fit (weakly identified)
PROVENANCE_PROJECTED = "projected"  # beyond the last measured cycle

DEFAULT_HORIZON_CYCLES = 300
MAX_SERIES_POINTS = 240

#: Form factors the renderer knows how to draw. Anything else is drawn as the
#: generic cylindrical cell with the mismatch disclosed on the scene.
FORM_FACTOR_CYLINDRICAL = "cylindrical"
FORM_FACTOR_PRISMATIC = "prismatic"
FORM_FACTOR_UNKNOWN = "unknown"

#: The host-independent default design tokens. The renderer reads colours from
#: here and hard-codes none of its own, so a host can restyle the scene without
#: the scene knowing anything about the host — and so the SOH bands a cell is
#: painted with are the SAME bands (90/80, and these three colours) the rest of
#: the platform labels it with. ``tests/test_cell_scene_theme.py`` asserts that
#: equivalence against app/_ui_helpers.soh_status() and app/static/theme.css,
#: which is what stops this view from calling a cell green while the app calls
#: it amber.
_DEFAULT_THEME = {
    "background": "#0b1120",
    "panel": "#111827",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "accent": "#63b3ed",
    "grid": "#1f2937",
    "metal": "#cbd5e0",
    # Part colours the renderer paints with. Here rather than in the renderer
    # because they are a host's palette, and because a test can then hold them
    # beside the app's own tokens instead of trusting a JS literal.
    "anodeColor": "#b08d57",
    "cathodeColor": "#7b6ba8",
    "separatorColor": "#d9e2ec",
    "electrolyteColor": "#63b3ed",
    "seiColor": "#b794f4",
    "sohBands": [
        {"min": 90.0, "max": None, "color": "#48bb78", "label": "Healthy"},
        {"min": 80.0, "max": 90.0, "color": "#f6e05e", "label": "Degrading"},
        {"min": 0.0, "max": 80.0, "color": "#fc8181", "label": "End of Life"},
    ],
    "temperatureBands": [
        {"min": 0.0, "max": 30.0, "color": "#4299e1"},
        {"min": 30.0, "max": 45.0, "color": "#ecc94b"},
        {"min": 45.0, "max": None, "color": "#e53e3e"},
    ],
    "provenanceColors": {
        "measured": "#48bb78", "derived": "#63b3ed",
        "fitted": "#b794f4", "projected": "#f6ad55", "": "#718096",
    },
    "sopFloorPct": 70.0,
    "powerGaugeMinPct": 40.0,
    "powerGaugeMaxPct": 110.0,
}


def default_theme() -> dict:
    """A fresh copy of the renderer's default design tokens.

    A copy, not the dict itself: a caller that mutates its theme (the Streamlit
    page does, to add host tokens) must not be able to restyle every other
    caller's scene in the same process, and the app's cached specs would
    otherwise be a shared mutable object across sessions.
    """
    import copy
    return copy.deepcopy(_DEFAULT_THEME)


_TWO_TERM_LAW = "SOH(n)/SOH₀ = 1 − β_sei·√n − β_lam·n   (n = cycles since first cycle + 1)"
_SEI_TERM = "β_sei·√n"
_LAM_TERM = "β_lam·n"


# ---------------------------------------------------------------------------
# Small helpers (kept local: this module must not grow an app dependency)
# ---------------------------------------------------------------------------

def _finite(value, default: float = float("nan")) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if num == num else default


def _is_finite(value) -> bool:
    num = _finite(value)
    return num == num


def _opt(value) -> "float | None":
    """``value`` as a float for the spec, or ``None`` when it isn't a number.

    The renderer distinguishes ``null`` ("no measurement here") from ``0``, so
    this never substitutes a zero for a missing reading.
    """
    num = _finite(value)
    return None if num != num else float(num)


def _column(df, name: str) -> list:
    """One column of a cell frame as a plain list, or ``[]`` when absent."""
    if df is None:
        return []
    try:
        if name not in df.columns:
            return []
        return list(df[name])
    except Exception:
        return []


def _at(values: list, i: int):
    """``values[i]`` or None — a frame whose columns are not all the same length.

    Frames in this codebase are not guaranteed rectangular (a feature block can
    be shorter than the cycle table it was joined onto), and an IndexError on a
    missing column would take a whole page down rather than reporting one part
    as unmeasured.
    """
    return values[i] if 0 <= i < len(values) else None


def _sampled_indices(n: int, max_points: int) -> list[int]:
    """Uniform stride over ``n`` rows, always keeping the first and last.

    A cell's whole recorded life has to survive the trip into the browser, but
    a 1,150-cycle Severson record at one point per cycle would bloat every
    render for no visible gain. Both endpoints are always kept: the first
    measured state and today are the two the user actually looks at.
    """
    if n <= 0:
        return []
    if n <= max_points:
        return list(range(n))
    stride = (n - 1) / (max_points - 1)
    idx = sorted({int(round(i * stride)) for i in range(max_points)})
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


#: Marker for "this value has no JSON form" inside `_json_safe`. A sentinel rather
#: than `None`, because `None` is a meaningful value in this document ("no
#: measurement here") and silently turning a model object into one would invent a
#: reading.
_UNSERIALIZABLE = object()


def _json_safe(value: Any, dropped: "list[str] | None" = None, _path: str = "") -> Any:
    """A JSON-representable copy of `value`, reporting anything it had to drop.

    Needed because the spec is a *contract*, not a Python object graph: the
    forecast router's own result dict carries live estimator objects alongside
    its decision fields, and a document that cannot be serialized is a document
    no renderer — and no `GET /cells/{id}/scene` — can actually receive. Dropping
    is reported rather than silent, so a caller can see that something was left
    out instead of discovering it when a key is missing.
    """
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            safe = _json_safe(item, dropped, f"{_path}.{key}")
            if safe is not _UNSERIALIZABLE:
                out[str(key)] = safe
            elif dropped is not None:
                dropped.append(f"{_path}.{key}".lstrip("."))
        return out
    if isinstance(value, (list, tuple)):
        items = []
        for index, item in enumerate(value):
            safe = _json_safe(item, dropped, f"{_path}[{index}]")
            if safe is not _UNSERIALIZABLE:
                items.append(safe)
            elif dropped is not None:
                dropped.append(f"{_path}[{index}]")
        return items
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        num = float(value)
        return None if num != num else num
    return _UNSERIALIZABLE


def _jsonify(value: Any) -> Any:
    """numpy scalars/arrays → plain Python, so the spec round-trips as JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        num = float(value)
        return None if num != num else num
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, float):
        return None if value != value else value
    return value


# ---------------------------------------------------------------------------
# Physics: the platform's own fitted two-term fade law, evaluated per cycle
# ---------------------------------------------------------------------------

def physics_series(cycles: list[float], beta_sei: "float | None", beta_lam: "float | None") -> dict:
    """Per-cycle SEI (√n) and LAM (linear) contributions, as % of initial capacity.

    ``n`` is cycles-since-the-first-recorded-cycle + 1, matching exactly how
    ``physics_calibration.fit_two_term_fade`` shifts its axis — so these two
    series account for the same fade the fit was scored on, and their sum is
    comparable with the cell's measured capacity loss rather than being an
    independent re-derivation. Both are monotone in n, which is what lets the
    scene animate them as growth rather than as a fit artefact.
    """
    n = len(cycles)
    sei_pct: list["float | None"] = []
    lam_pct: list["float | None"] = []
    sei_share: list["float | None"] = []
    if n == 0:
        return {"seiPct": [], "lamPct": [], "seiSharePct": [], "n": []}

    # ``_is_finite``, not ``x == x``: a refused fit passes ``None`` for both
    # betas, and ``None == None`` is True — which is how "no fit" once reached
    # the arithmetic below as "two equal floats".
    have = _is_finite(beta_sei) and _is_finite(beta_lam)
    b_sei = _finite(beta_sei)
    b_lam = _finite(beta_lam)
    c0 = _finite(cycles[0], 0.0)
    n_axis: list[float] = []
    for i in range(n):
        nn = max(1.0, _finite(cycles[i], c0) - c0 + 1.0)
        n_axis.append(nn)
        if not have:
            sei_pct.append(None)
            lam_pct.append(None)
            sei_share.append(None)
            continue
        s = 100.0 * b_sei * math.sqrt(nn)
        l = 100.0 * b_lam * nn
        sei_pct.append(s)
        lam_pct.append(l)
        total = s + l
        sei_share.append((s / total * 100.0) if total > 0 else None)

    return {"seiPct": sei_pct, "lamPct": lam_pct, "seiSharePct": sei_share, "n": n_axis}


def fit_physics(df) -> dict:
    """The full-history two-term fade fit this scene's geometry is built on.

    Calls the library's own ``fit_two_term_fade`` (the function
    ``calibrate_cell()`` uses internally) rather than reimplementing the fit, and
    reports the fit's own ``r2`` and parameter sigmas so the scene can disclose
    when the β_sei/β_lam apportionment is not trustworthy — the library's
    docstring is explicit that a wide sigma relative to the fitted value means
    exactly that.
    """
    out = {
        "available": False, "splitIdentified": False,
        "betaSei": None, "betaLam": None, "fitR2": None,
        "betaSeiSigma": None, "betaLamSigma": None, "nCyclesUsed": None,
        "seiTStat": None, "lamTStat": None,
        "modelTotalFadePct": None, "measuredFadePct": None,
        "reason": None, "splitReason": None, "kR": None, "kRSigma": None,
    }
    try:
        cycles = df["cycle_number"].to_numpy(dtype=float)
        soh = df["soh_pct"].to_numpy(dtype=float)
    except Exception:
        out["reason"] = "this cell's frame has no cycle/SOH columns to fit"
        return out

    try:
        from physics_calibration import MIN_FIT_R2_FOR_DOMINANT_MODE, fit_two_term_fade
        r2_floor = float(MIN_FIT_R2_FOR_DOMINANT_MODE)
    except Exception:
        r2_floor = 0.3
        fit_two_term_fade = None  # type: ignore[assignment]
    try:
        if fit_two_term_fade is None:
            raise ImportError("the physics calibration module is unavailable")
        fit = fit_two_term_fade(cycles, soh) or {}
    except Exception:
        fit = {}
    if fit.get("beta_sei") is None or fit.get("beta_lam") is None:
        out["reason"] = (
            "the two-term fade fit did not converge for this cell, so the SEI and active-material "
            "layers carry no number"
        )
        return out

    # A fit that converged but explains nothing is not a fade law. This is not a
    # hypothetical: a cell whose SOH never moves — a flat synthetic cell, or a
    # frame whose capacity column is a placeholder — fits to betas of ~1e-7 with
    # r2 = 0.0 (the library's sentinel for "there was no variance to explain"),
    # and drawing geometry from it would render an invented fade curve as this
    # cell's physics. The floor is the library's OWN gate for trusting this
    # fit's apportionment (physics_calibration.MIN_FIT_R2_FOR_DOMINANT_MODE),
    # reused rather than invented, so the scene and the mechanism vocabulary
    # refuse on the same evidence.
    _r2 = _finite(fit.get("r2"))
    if _r2 != _r2 or _r2 < r2_floor:
        out["measuredFadePct"] = _opt(100.0 - _finite(soh[-1]))
        out["nCyclesUsed"] = int(fit.get("n_cycles_used") or len(cycles))
        out["fitR2"] = _opt(fit.get("r2"))
        reason = (
            f"the two-term fade fit explains none of this cell's capacity history "
            f"(r² = {'unavailable' if _r2 != _r2 else '%.3f' % _r2}, below the platform's "
            f"{r2_floor} floor for trusting this fit)"
            if soh.size and float(np.nanmax(soh)) - float(np.nanmin(soh)) < 1e-9 else
            f"the two-term fade fit is too poor to draw from "
            f"(r² = {'unavailable' if _r2 != _r2 else '%.3f' % _r2}, below the platform's "
            f"{r2_floor} floor for trusting this fit)"
        )
        out["reason"] = reason
        out["splitReason"] = (
            "This cell's capacity history gives the two-term fit nothing to separate — there is no "
            "measured fade to apportion between a √n film and a linear active-material loss.")
        return out

    b_sei = _finite(fit.get("beta_sei"))
    b_lam = _finite(fit.get("beta_lam"))
    s_sei = _finite(fit.get("beta_sei_sigma"))
    s_lam = _finite(fit.get("beta_lam_sigma"))

    # Is each channel distinguishable from zero? The library's own docstring
    # says to judge this by whether sigma is wide relative to the fitted value;
    # |β|/σ is that judgement made numeric. It matters here more than anywhere
    # else in the app: √n and n are so collinear over a few hundred cycles that
    # a fit can place essentially ALL fade in either term, and on every cell
    # this platform ships, β_sei comes out far below its own sigma. Drawing a
    # film thickness from that coefficient would be drawing noise.
    sei_t = (abs(b_sei) / s_sei) if (b_sei == b_sei and s_sei == s_sei and s_sei > 0) else None
    lam_t = (abs(b_lam) / s_lam) if (b_lam == b_lam and s_lam == s_lam and s_lam > 0) else None
    identified = bool(sei_t is not None and lam_t is not None and sei_t >= 1.0 and lam_t >= 1.0)

    t_end = max(1.0, _finite(cycles[-1], 1.0) - _finite(cycles[0], 1.0) + 1.0) if len(cycles) else 1.0
    total_model = 100.0 * (b_sei * math.sqrt(t_end) + b_lam * t_end) if (b_sei == b_sei and b_lam == b_lam) else float("nan")
    try:
        measured = 100.0 - float(soh[-1])
    except Exception:
        measured = float("nan")

    out.update({
        "available": True,
        "splitIdentified": identified,
        "betaSei": _opt(b_sei),
        "betaLam": _opt(b_lam),
        "fitR2": _opt(fit.get("r2")),
        "betaSeiSigma": _opt(s_sei),
        "betaLamSigma": _opt(s_lam),
        "seiTStat": _opt(sei_t),
        "lamTStat": _opt(lam_t),
        "modelTotalFadePct": _opt(total_model),
        "measuredFadePct": _opt(measured),
        "nCyclesUsed": int(fit.get("n_cycles_used") or len(cycles)),
    })
    if not identified:
        _weak = "SEI (√n)" if (sei_t is not None and sei_t < 1.0) else "active-material (linear)"
        out["splitReason"] = (
            "This cell's data cannot separate the √n (SEI, lithium-inventory) fade from the linear "
            f"(active-material) fade: the {_weak} coefficient fits below its own standard error "
            f"({'|β_sei|/σ = %.2g' % sei_t if sei_t is not None else 'β_sei/σ unavailable'}, "
            f"{'|β_lam|/σ = %.2g' % lam_t if lam_t is not None else 'β_lam/σ unavailable'}), and over a "
            "narrow cycle range the two regressors are strongly correlated. The combined fitted fade is "
            "still identified and is shown; the split between the two layers is not, so the scene does "
            "not draw one."
        )

    # The resistances channel is the library's own note on which evidence is the
    # *more* robust standalone signal for an active SEI/LLI channel, so it rides
    # along whenever the frame carries resistance.
    try:
        if "resistance_ohm" in df.columns:
            from physics_calibration import fit_resistance_growth
            kr = fit_resistance_growth(cycles, df["resistance_ohm"].to_numpy(dtype=float)) or {}
            out["kR"] = _opt(kr.get("k_r"))
            out["kRSigma"] = _opt(kr.get("k_r_sigma"))
    except Exception:
        pass
    return out


def form_factor_for(profile) -> "tuple[str, str]":
    """``(form_factor, note)`` for a chemistry profile, from its own declaration.

    Reads the profile's passport chemistry string rather than guessing from the
    cell id — the platform moved this exact class of decision to
    ``ChemistryProfile`` precisely because hand-rolled id checks kept missing
    newly added datasets. An unrecognised or absent profile declares itself
    unknown and the scene says so instead of drawing a confident 18650.
    """
    passport = ""
    try:
        passport = str(getattr(profile, "passport_chemistry", "") or "")
    except Exception:
        passport = ""
    lowered = passport.lower()
    if "prismatic" in lowered:
        return FORM_FACTOR_PRISMATIC, passport
    if "cylindrical" in lowered or "18650" in lowered or "21700" in lowered or "4680" in lowered:
        return FORM_FACTOR_CYLINDRICAL, passport
    return FORM_FACTOR_UNKNOWN, passport


# ---------------------------------------------------------------------------
# The anatomy: one card per part, each pointing at a real number
# ---------------------------------------------------------------------------

def _part(part_id: str, label: str, *, value=None, unit: str = "", provenance: str = "",
          meaning: str = "", law: str = "", series: "list | None" = None,
          available: bool = True, reason: "str | None" = None) -> dict:
    return {
        "id": part_id,
        "label": label,
        "value": _opt(value) if value is not None else None,
        "unit": unit,
        "provenance": provenance,
        "meaning": meaning,
        "law": law,
        "series": series,
        "available": bool(available),
        "unavailableReason": reason,
    }


def _unavailable(part_id: str, label: str, reason: str, meaning: str) -> dict:
    """A part this source cannot speak to.

    Reported as unavailable *with a reason* rather than given a zero: on a
    scene where size means something, a silently-zero part reads as "this part
    is fine", which is a claim the platform has not earned.
    """
    return _part(part_id, label, available=False, reason=reason, meaning=meaning)


def _build_parts(df, last_row, profile, phys, has_dqdv: bool, fit: dict) -> list[dict]:
    """The 13 anatomy cards, each wired to the most specific signal that exists."""
    def _last(col: str):
        return last_row.get(col) if last_row is not None else None

    temperature = _last("temperature_c")
    resistance = _last("resistance_ohm")
    r_norm = _last("resistance_normalized")
    sop = _last("sop_pct")
    power_limited = _last("is_power_limited")
    peak_value = _last("dqdv_sim_peak_value")
    peak_area = _last("dqdv_sim_area")

    sei_series = phys["seiPct"]
    lam_series = phys["lamPct"]
    sei_last = next((v for v in reversed(sei_series) if v is not None), None)
    lam_last = next((v for v in reversed(lam_series) if v is not None), None)
    _fit_reason = fit.get("reason") or "the two-term fade fit is unavailable for this cell"

    parts: list[dict] = []

    parts.append(_part(
        "can", "Cell casing", value=temperature, unit="°C", provenance=PROVENANCE_MEASURED,
        meaning=(
            "The casing's surface temperature is the one thing about the enclosure this platform "
            "actually measures, and it is how the cell's thermal duty enters every other calculation. "
            "The scene tints the casing with it."
        ),
        law="direct measurement (cycler thermocouple)",
        series=[_opt(v) for v in _column(df, "temperature_c")],
        available=_is_finite(temperature),
        reason=None if _is_finite(temperature) else "no temperature recorded for this cell",
    ))

    parts.append(_unavailable(
        "cap", "Top cap assembly",
        "no measurement in the cycle-summary data separates the cap from the rest of the cell",
        "The cap carries the vent and the positive terminal; it is drawn so the cell reads as a real "
        "enclosure rather than a cylinder. Nothing in this dataset measures it.",
    ))
    parts.append(_unavailable(
        "vent", "Vent disc",
        "no measurement in the cycle-summary data",
        "The vent is a safety device. This platform has no pressure, vent or abuse-test channel, so it "
        "is drawn as architecture and deliberately carries no number.",
    ))

    parts.append(_part(
        "terminal_pos", "Positive terminal", value=resistance, unit="Ω",
        provenance=PROVENANCE_MEASURED,
        meaning=(
            "The DC internal resistance the cycler measured. It is the terminal-to-terminal figure — "
            "welds, tabs, foils and the electrochemical impedance all folded into one number — and it is "
            "what throttles the cell's power as it ages."
        ),
        law="direct measurement (DCIR)",
        series=[_opt(v) for v in _column(df, "resistance_ohm")],
        available=_is_finite(resistance),
        reason=None if _is_finite(resistance) else "no resistance reading for this cell",
    ))
    parts.append(_part(
        "terminal_neg", "Negative terminal", value=resistance, unit="Ω",
        provenance=PROVENANCE_MEASURED,
        meaning=(
            "The same measured DC resistance, shown at the other terminal because the current path is "
            "the point: whatever this number does, the whole cell's power delivery does with it."
        ),
        law="direct measurement (DCIR)",
        series=[_opt(v) for v in _column(df, "resistance_ohm")],
        available=_is_finite(resistance),
        reason=None if _is_finite(resistance) else "no resistance reading for this cell",
    ))

    parts.append(_part(
        "tab_pos", "Positive tab", value=r_norm, unit="× initial",
        provenance=PROVENANCE_DERIVED,
        meaning=(
            "Resistance growth against the cell's own first reading — the trend, separated from the "
            "chemistry-dependent absolute value, which is what makes it comparable between a NASA "
            "18650 and a CALCE prismatic cell."
        ),
        law="R(n) / R(first positive reading)",
        series=[_opt(v) for v in _column(df, "resistance_normalized")],
        available=_is_finite(r_norm),
        reason=None if _is_finite(r_norm) else "no usable resistance history for this cell",
    ))
    parts.append(_part(
        "tab_neg", "Negative tab", value=sop, unit="% of initial power",
        provenance=PROVENANCE_DERIVED,
        meaning=(
            "State of Power — peak-power capability relative to the cell's own fresh state, from "
            "P ∝ 1/R. A cell can hold healthy capacity and still be unfit for a pulse-power duty, "
            "which is exactly what this part tells you"
            + (" — and this cell has crossed the 70% power-fade floor." if power_limited is True else ".")
        ),
        law="SoP = R(initial) / R(now) × 100; floor at 70%",
        series=[_opt(v) for v in _column(df, "sop_pct")],
        available=_is_finite(sop),
        reason=None if _is_finite(sop) else "no resistance history to derive power capability from",
    ))

    if has_dqdv:
        parts.append(_part(
            "cathode_sheet", "Cathode coating", value=peak_value, unit="dQ/dV peak amplitude (proxy)",
            provenance=PROVENANCE_DERIVED,
            meaning=(
                "The height of the simulated dQ/dV peak. Read it as a *re-expression*, not as electrode "
                "evidence: the platform derives its dQ/dV features by running the cycle's own capacity "
                "and resistance through a parametric LiCoO₂ OCV model, so the peak tracks capacity "
                "rather than adding an independent measurement of the coating. It is shown because "
                "seeing capacity loss appear as a shrinking differential-capacity peak is how an "
                "electrochemist reads a cell — not because this column is a fresh instrument."
            ),
            law="simulated dQ/dV: V(Q) from a parametric LiCoO₂ OCV model, peak taken from the curve",
            series=[_opt(v) for v in _column(df, "dqdv_sim_peak_value")],
            available=_is_finite(peak_value),
            reason=None if _is_finite(peak_value) else "no dQ/dV features for this cell",
        ))
        parts.append(_part(
            "anode_sheet", "Anode coating", value=peak_area, unit="dQ/dV curve area (proxy)",
            provenance=PROVENANCE_DERIVED,
            meaning=(
                "The integral of the simulated dQ/dV curve — the same proxy as the cathode's peak "
                "height, seen as area rather than height. The platform's two other dQ/dV columns are "
                "deliberately *not* drawn here: the peak's state-of-charge position is a constant of the "
                "OCV model (identical for every cell in the fleet, so it carries no per-cell "
                "information) and the peak width comes out degenerate at zero. Neither is this cell's "
                "state, and a layer that silently showed a model constant would be a lie told in "
                "geometry."
            ),
            law="simulated dQ/dV: trapz of the modelled curve over the cycle's capacity",
            series=[_opt(v) for v in _column(df, "dqdv_sim_area")],
            available=_is_finite(peak_area),
            reason=None if _is_finite(peak_area) else "no dQ/dV features for this cell",
        ))
    else:
        _dqdv_reason = (
            "this cell's source declares dQ/dV inapplicable — the plateau of an LFP cathode makes the "
            "LiCoO₂-shaped OCV model the platform simulates with invalid here"
            if profile is not None and getattr(profile, "dqdv_applicable", False) is False
            else "no dQ/dV features recorded for this cell"
        )
        parts.append(_unavailable(
            "cathode_sheet", "Cathode coating", _dqdv_reason,
            "The cathode is drawn as the architecture it is. No electrode-level measurement is claimed "
            "for this cell, so the geometry carries none — the same refusal the rest of the platform "
            "applies to dQ/dV on inapplicable chemistries.",
        ))
        parts.append(_unavailable(
            "anode_sheet", "Anode coating", _dqdv_reason,
            "The anode is drawn as the architecture it is; no electrode-level signal is available from "
            "this source, and an estimate here would be invented rather than measured.",
        ))

    parts.append(_unavailable(
        "separator", "Separator",
        "no measurement in the cycle-summary data",
        "The separator is drawn between the two coatings because the layer order is what makes a "
        "jelly roll a jelly roll. Nothing measured here speaks to it.",
    ))
    parts.append(_unavailable(
        "electrolyte", "Electrolyte",
        "no measurement in the cycle-summary data",
        "Electrolyte consumption is, physically, the SEI reaction — so the honest place to read it is "
        "the SEI film card, which is fitted. The volume itself is architecture.",
    ))

    split_ok = bool(fit.get("splitIdentified"))
    split_reason = fit.get("splitReason") or (
        "this cell's data does not separate the two fade channels, so neither layer is given a number"
    )

    parts.append(_part(
        "particles", "Active material particles", value=lam_last if split_ok else None,
        unit="% of initial capacity (fitted)",
        provenance=PROVENANCE_FITTED,
        meaning=(
            "Loss of active material: the linear term of the platform's two-term fade fit, evaluated at "
            "this cycle. It is the half of capacity fade caused by particles cracking, disconnecting or "
            "otherwise leaving the circuit — as opposed to lithium being consumed."
            if split_ok else
            "The layer is drawn because particles are what a cell's coating is made of, but this cell's "
            "data cannot tell active-material loss apart from lithium-inventory loss, so no number is "
            "claimed for it. The combined fitted fade is on the capacity readout instead."
        ),
        law=f"{_LAM_TERM}  (full-history fitted, see the fit-quality caveat below)",
        series=lam_series if split_ok else None,
        available=split_ok and lam_last is not None,
        reason=None if (split_ok and lam_last is not None) else (split_reason if not split_ok else _fit_reason),
    ))
    parts.append(_part(
        "sei_film", "SEI film on the anode", value=sei_last if split_ok else None,
        unit="% of initial capacity (fitted)",
        provenance=PROVENANCE_FITTED,
        meaning=(
            "Loss of lithium inventory: growth of the passivating film on the anode, the diffusion-"
            "limited √n term of the same fit. It is the single clearest visual statement of why a "
            "battery ages at all — the film thickens, permanently, and the lithium it traps never cycles "
            "again. On this scene the film's drawn thickness is this term, so the geometry *is* the "
            "physics rather than an illustration of it."
            if split_ok else
            "An SEI film exists on every aged cell, but this cell's capacity history cannot say how much "
            "of its fade it accounts for — √n and n are too alike over this many cycles. The scene draws "
            "the film as architecture and puts no number on it, rather than drawing a thickness the fit "
            "did not earn."
        ),
        law=f"{_SEI_TERM}  (full-history fitted, see the fit-quality caveat below)",
        series=sei_series if split_ok else None,
        available=split_ok and sei_last is not None,
        reason=None if (split_ok and sei_last is not None) else (split_reason if not split_ok else _fit_reason),
    ))

    return parts


# ---------------------------------------------------------------------------
# Mechanism: two independent verdicts, compared rather than blended
# ---------------------------------------------------------------------------

def _mechanism(df, graph, cell_id: str, fit: dict) -> dict:
    """Physics verdict (from the fitted βs) and ML verdict, side by side.

    The physics side uses ``physics_calibration.dominant_mode()`` — the library
    function that owns this decision, including its own r² gate and its 1.5×
    contribution ratio — so the scene reports ``insufficient_data`` in exactly
    the cases the rest of the platform does. The ML side goes through ``graph``
    when the caller has one, mirroring the Health page so the two surfaces
    cannot disagree on the same cell.
    """
    out = {
        "physics": None, "ml": None, "agree": None, "agreementIsEvidence": False,
        "note": "", "emphasis": "neutral", "emphasisSource": "none",
    }

    b_sei = _finite(fit.get("betaSei"))
    b_lam = _finite(fit.get("betaLam"))
    fit_r2 = _finite(fit.get("fitR2"))
    cycles = _column(df, "cycle_number")
    at_cycle = max(1.0, (_finite(cycles[-1], 1.0) - _finite(cycles[0], 1.0) + 1.0)) if cycles else 1.0

    if b_sei == b_sei and b_lam == b_lam and fit_r2 == fit_r2:
        try:
            from physics_calibration import MIN_FIT_R2_FOR_DOMINANT_MODE, dominant_mode
            key, label = dominant_mode(b_sei, b_lam, at_cycle, fit_r2)
        except Exception:
            key, label = "insufficient_data", "Insufficient data"
            MIN_FIT_R2_FOR_DOMINANT_MODE = 0.3
        out["physics"] = {
            "key": key, "label": label,
            "betaSei": _opt(b_sei), "betaLam": _opt(b_lam),
            "fitR2": _opt(fit_r2),
            "atCycle": _opt(at_cycle),
            "gate": float(MIN_FIT_R2_FOR_DOMINANT_MODE),
            "gatePassed": bool(key != "insufficient_data"),
            "contributionSeiPct": _opt(100.0 * b_sei * math.sqrt(at_cycle)),
            "contributionLamPct": _opt(100.0 * b_lam * at_cycle),
        }
    elif fit.get("reason"):
        out["physics"] = {
            "key": "insufficient_data", "label": "Insufficient data",
            "betaSei": None, "betaLam": None, "fitR2": None, "atCycle": _opt(at_cycle),
            "gate": None, "gatePassed": False, "contributionSeiPct": None,
            "contributionLamPct": None, "reason": fit.get("reason"),
        }

    try:
        if graph is not None:
            from knowledge_graph import get_or_compute_mechanism
            edge = get_or_compute_mechanism(graph, cell_id, df) or {}
            verdict = str(edge.get("verdict") or edge.get("mechanism") or "")
            out["ml"] = {
                "verdict": verdict,
                "confidenceLabel": edge.get("confidence_label"),
                "body": edge.get("verdict_body") or edge.get("note"),
                "source": "knowledge_graph (shared verdict)",
            }
        else:
            from recommendations import diagnose_mechanism
            ml = diagnose_mechanism(df) or {}
            out["ml"] = {
                "verdict": str(ml.get("verdict", "")),
                "confidenceLabel": ml.get("confidence_label"),
                "body": ml.get("verdict_body"),
                "source": "recommendations.diagnose_mechanism (direct)",
            }
    except Exception:
        out["ml"] = None

    physics_key = (out.get("physics") or {}).get("key")
    ml_verdict = ((out.get("ml") or {}).get("verdict") or "").upper()
    ml_key = None
    if ml_verdict:
        ml_key = "lam" if ("LAM" in ml_verdict and "LLI" not in ml_verdict) else (
            "lli" if ("LLI" in ml_verdict and "LAM" not in ml_verdict) else "mixed"
        )

    # Agreement is only evidence if the two sides are genuinely independent. When
    # the fitted channels are not separable, the physics verdict is downstream of
    # whichever ridge the optimiser landed on, so "they agree" would be a claim
    # with no content — this reports no agreement instead of a cosy one.
    if not fit.get("splitIdentified"):
        out["agree"] = None
        out["agreementIsEvidence"] = False
    elif physics_key and ml_key:
        if physics_key == "insufficient_data" or "INSUFFICIENT" in ml_verdict:
            out["agree"] = None
            out["agreementIsEvidence"] = False
        else:
            out["agree"] = bool(physics_key == ml_key)
            out["agreementIsEvidence"] = True
    else:
        out["agreementIsEvidence"] = False

    if out["agree"] is True:
        out["note"] = (
            "The fitted physics and the ML classifier independently name the same dominant "
            "mechanism, which is the strongest statement this platform can make about why a cell is "
            "fading."
        )
    elif out["agree"] is False:
        out["note"] = (
            "The fitted physics and the ML classifier disagree about which mechanism dominates. Both "
            "are shown rather than arbitrated: the physics split is weakly identified from capacity "
            "data alone (√n and n are strongly correlated over a narrow cycle range), so a "
            "disagreement is information about the evidence, not a verdict."
        )
    elif not fit.get("splitIdentified"):
        out["note"] = (
            "No agreement is claimed here, because there is nothing independent to agree with: this "
            "cell's capacity history cannot separate the √n and linear fade channels, so the physics "
            "verdict is a consequence of which ridge the fit settled on rather than separate evidence. "
            "The ML classifier's verdict stands on its own signals — coulombic-efficiency trend, fade "
            "curvature and resistance rise rate — and the scene emphasises that instead."
        )
    else:
        out["note"] = (
            "One of the two mechanism verdicts is not available for this cell (insufficient data, or "
            "the fit fell below its r² gate), so no agreement is claimed either way."
        )

    physics_key = physics_key or ""
    if fit.get("splitIdentified"):
        out["emphasis"] = (
            "sei" if physics_key == "lli" else
            "particles" if physics_key == "lam" else "neutral"
        )
        out["emphasisSource"] = "physics" if out["emphasis"] != "neutral" else "none"
    else:
        # Physics cannot apportion the fade, so the emphasis follows the one
        # verdict that is still independent evidence, and says that it did.
        out["emphasis"] = (
            "sei" if ml_key == "lli" else
            "particles" if ml_key == "lam" else "neutral"
        )
        out["emphasisSource"] = "ml" if out["emphasis"] != "neutral" else "none"
    return out


# ---------------------------------------------------------------------------
# Projection: the platform's existing forecast, gated by its existing routing
# ---------------------------------------------------------------------------

def _projection(cell_id: str, bundle, bundles, horizon_cycles: int, featured_dfs) -> "dict | None":
    """The forward curve, or ``None`` with a reason why not.

    Uses ``batlab.models.hierarchical.project_future_soh`` — the very curve the
    Health page's 12-month forecast draws, posterior band included — and is
    gated by ``forecast_routing.route_forecast_for_cell``, so a cell whose
    regime routing says ``refuse`` produces no future at all. A 3D scene that
    drew a confident-looking line for a cell the platform refuses to forecast
    would undo the platform's own honesty rule in the most persuasive possible
    medium.
    """
    if int(horizon_cycles) <= 0:
        return {
            "available": False,
            "reason": (
                "This scene was asked for the measured record only (a horizon of zero cycles), so it "
                "stops at the last measured cycle."
            ),
        }

    fit = (bundle or {}).get("hierarchical_fit") if isinstance(bundle, dict) else None
    if not fit:
        return {
            "available": False,
            "reason": (
                "No hierarchical forecast fit is available for this cell, so the scene stops at the "
                "last measured cycle."
            ),
        }

    route = None
    if bundles:
        try:
            from forecast_routing import route_forecast_for_cell
            route = route_forecast_for_cell(cell_id, bundles, featured_dfs=featured_dfs)
        except Exception:
            route = None
    if route and route.get("served") == "refuse":
        return {
            "available": False,
            "reason": (
                "The platform's per-cell forecast routing refuses to forecast this cell — "
                f"{route.get('reason') or 'neither model clears its validity gate in this regime'}"
            ),
            "route": _route_summary(route),
        }

    try:
        from batlab.models.hierarchical import project_future_soh
        proj = project_future_soh(fit, cell_id, int(horizon_cycles))
    except Exception:
        proj = None
    if not proj:
        return {
            "available": False,
            "reason": (
                "The hierarchical model has no pooled prior for this cell's chemistry, so it cannot "
                "project beyond the measured record."
            ),
        }

    cycles = [_opt(v) for v in proj.get("cycles", [])]
    soh = [_opt(v) for v in proj.get("soh_pct", [])]
    q10 = [_opt(v) for v in proj.get("soh_q10_pct", [])]
    q90 = [_opt(v) for v in proj.get("soh_q90_pct", [])]
    if not cycles or not any(v is not None for v in soh):
        return {"available": False, "reason": "The projection returned no usable cycles."}

    shrink = proj.get("shrinkage_weight_prior")
    model_label = (route or {}).get("model_label") or "Hierarchical partial-pooling (shrunk log-fade line)"
    return {
        "available": True,
        "cycles": cycles,
        "sohPct": soh,
        "sohQ10Pct": q10,
        "sohQ90Pct": q90,
        "kind": "projected",
        "modelLabel": model_label,
        "shrinkageFromPrior": _opt(shrink),
        "route": _route_summary(route) if route else None,
        "reason": None,
    }


def _route_summary(route) -> "dict | None":
    """The forecast router's verdict, minus the estimator objects it carries.

    `route_forecast_for_cell` returns its decision *and* the fitted models behind
    it. The models are not serializable and not part of this contract, so they
    are dropped and named — a renderer that wanted them would be re-implementing
    the forecast, and a reader of the document can see exactly what was left out
    (`unserializedKeys`) instead of guessing why a field is missing.
    """
    if not isinstance(route, dict):
        return None
    dropped: list[str] = []
    safe = _json_safe(route, dropped)
    if isinstance(safe, dict) and dropped:
        safe["unserializedKeys"] = sorted({name for name in dropped})
    return safe if isinstance(safe, dict) else None


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

#: ``ChemistryProfile.source_kind`` → the bundle dict key it corresponds to.
#: These are the keys ``app/main.py``'s ``load_everything()`` publishes and
#: ``app/_design_tokens.PACK_BUNDLE_KEY`` maps to; kept here so this module is
#: usable without importing the app.
_BUNDLE_KEY_ALIASES = {
    "synthetic": "synth", "synth": "synth",
    "uploaded": "upload", "upload": "upload",
    "nasa": "nasa", "severson": "severson",
    "zhu2022": "zhu2022", "calce": "calce", "oxford": "oxford",
}


def resolve_bundle_key(cell_id: str) -> str:
    """Which bundle in ``{key: bundle}`` describes this cell's fleet."""
    try:
        from chemistry_profiles import ChemistryProfile
        profile = ChemistryProfile.for_cell(cell_id)
        kind = str(getattr(profile, "source_kind", "") or "")
    except Exception:
        kind = ""
    return _BUNDLE_KEY_ALIASES.get(kind, kind or "synth")


def build_cell_scene(
    cell_id: str,
    df,
    *,
    bundle: "dict | None" = None,
    bundles: "dict | None" = None,
    graph=None,
    horizon_cycles: int = DEFAULT_HORIZON_CYCLES,
    max_points: int = MAX_SERIES_POINTS,
    theme: "dict | None" = None,
) -> dict:
    """Build the full ``CellSceneSpec`` for one cell.

    ``df`` is the cell's featured per-cycle frame — the same object every other
    page charts. ``bundle`` is the model bundle for the cell's fleet (used for
    the forecast); ``bundles`` is the ``{key: bundle}`` map the forecast router
    needs to decide *whether* a forecast may be drawn. ``graph`` is the shared
    knowledge graph, optional: without it the mechanism verdict comes straight
    from the classifier rather than the graph's cached edge.

    Everything is read; nothing is re-derived. Missing inputs become
    ``available: false`` with a reason, and a refused forecast becomes
    ``projection.available: false`` with the router's own words.
    """
    try:
        from chemistry_profiles import ChemistryProfile
        profile = ChemistryProfile.for_cell(cell_id)
    except Exception:
        profile = None

    form_factor, form_factor_note = form_factor_for(profile)
    source_label = str(getattr(profile, "source_label", "") or "")
    provenance = str(getattr(profile, "provenance", "") or "")
    chemistry = str(getattr(profile, "passport_chemistry", "") or getattr(profile, "short_name", "") or "")
    dqdv_applicable = bool(getattr(profile, "dqdv_applicable", False))

    cycles_raw = [_finite(v) for v in _column(df, "cycle_number")]
    n_rows = min(len(cycles_raw), len(_column(df, "soh_pct"))) if cycles_raw else 0
    idx = _sampled_indices(n_rows, max_points)

    col_soh = _column(df, "soh_pct")
    col_capacity = _column(df, "capacity_ah")
    col_resistance = _column(df, "resistance_ohm")
    col_resistance_norm = _column(df, "resistance_normalized")
    col_temperature = _column(df, "temperature_c")
    col_sop = _column(df, "sop_pct")
    col_fade_30 = _column(df, "fade_rate_30cy")

    cycles = [cycles_raw[i] for i in idx]
    soh = [_opt(_at(col_soh, i)) for i in idx]
    capacity = [_opt(_at(col_capacity, i)) for i in idx]
    resistance = [_opt(_at(col_resistance, i)) for i in idx]
    resistance_norm = [_opt(_at(col_resistance_norm, i)) for i in idx]
    temperature = [_opt(_at(col_temperature, i)) for i in idx]
    sop = [_opt(_at(col_sop, i)) for i in idx]
    fade_30 = [_opt(_at(col_fade_30, i)) for i in idx]
    beta_sei_all = _column(df, "physics_beta_sei")
    beta_lam_all = _column(df, "physics_beta_lam")

    fit = fit_physics(df)
    phys_full = physics_series(cycles_raw[:n_rows], fit.get("betaSei"), fit.get("betaLam"))
    phys = {
        "seiPct": [phys_full["seiPct"][i] for i in idx],
        "lamPct": [phys_full["lamPct"][i] for i in idx],
        "seiSharePct": [phys_full["seiSharePct"][i] for i in idx],
    }
    # The identified quantity: total modelled fade, which is drawn whenever the
    # channel split is not, so the scene still animates the fitted physics even
    # on a cell where the two channels cannot be told apart.
    fade_model = [
        (s + l) if (s is not None and l is not None) else None
        for s, l in zip(phys_full["seiPct"], phys_full["lamPct"])
    ]
    phys["fadeModelPct"] = [fade_model[i] for i in idx]

    last_row = None
    try:
        if df is not None and len(df) > 0:
            last_row = df.iloc[-1]
    except Exception:
        last_row = None

    # Both conditions, because they answer different questions: the profile's
    # declaration says whether the LiCoO₂-shaped OCV model is *valid* for this
    # chemistry (false for LFP's flat plateau, NCM, CALCE), and the data check
    # says whether the columns were ever computed. A cell whose frame happens to
    # carry the columns while its chemistry profile refuses the model is exactly
    # the case the profile flag exists to catch.
    has_dqdv = dqdv_applicable and any(_is_finite(v) for v in _column(df, "dqdv_sim_peak_value"))

    parts = _build_parts(df, last_row, profile, phys, has_dqdv, fit)

    # Part series are sampled on the same index set as the top-level series, so
    # a renderer can index every array with the same cursor — and a test can
    # assert that alignment rather than trusting it.
    _series_by_part = {
        "can": "temperature_c", "terminal_pos": "resistance_ohm",
        "terminal_neg": "resistance_ohm", "tab_pos": "resistance_normalized",
        "tab_neg": "sop_pct", "cathode_sheet": "dqdv_sim_peak_value",
        "anode_sheet": "dqdv_sim_area",
    }
    for part in parts:
        col = _series_by_part.get(part["id"])
        if col:
            values = _column(df, col)
            part["series"] = [_opt(_at(values, i)) for i in idx]
            if all(v is None for v in part["series"]):
                part["series"] = None
        elif part["id"] in ("sei_film", "particles"):
            # Only when the split is identified. The anatomy cards refuse these
            # two layers on an unidentifiable split, and a per-cycle array is a
            # number every renderer can draw — assigning it here would quietly
            # turn that refusal back into a drawn film.
            if fit.get("splitIdentified"):
                part["series"] = phys["seiPct" if part["id"] == "sei_film" else "lamPct"]

    mechanism = _mechanism(df, graph, cell_id, fit)

    try:
        from batlab.features.knee_detection import detect_knee
        soh_series = df["soh_pct"] if (df is not None and "soh_pct" in df.columns) else None
        cycle_series = df["cycle_number"] if (df is not None and "cycle_number" in df.columns) else None
        # This module deliberately does not import pandas (it is the one file a
        # JS host's document is built from), so the two Series are typed as the
        # library declares them rather than re-declared here.
        knee_raw = (
            detect_knee(soh_series, cycle_series)  # pyright: ignore[reportArgumentType]
            if soh_series is not None else {}
        )
    except Exception:
        knee_raw = {}
    knee = {
        "detected": bool(knee_raw.get("detected", False)),
        "cycle": _opt(knee_raw.get("cycle")),
        "sohAtKnee": _opt(knee_raw.get("soh_at_knee")),
        "confidence": _opt(knee_raw.get("confidence")),
        "phase": knee_raw.get("phase"),
    }

    projection = _projection(
        cell_id, bundle, bundles, horizon_cycles,
        featured_dfs={cell_id: df} if df is not None else None,
    )

    fit_r2_series = [_opt(v) for v in _column(df, "physics_fit_r2")]
    fit_r2_last = fit.get("fitR2")
    # The frame's own per-cycle physics columns are refit every REFIT_EVERY_CYCLES
    # window. Recorded so the scene can say why it does not draw them as geometry.
    n_window_betas = len({round(v, 12) for v in beta_sei_all if _is_finite(v)})

    def _last_finite(values):
        for v in reversed(values):
            if v is not None:
                return v
        return None

    spec = {
        "schemaVersion": SCENE_SCHEMA_VERSION,
        "cell": {
            "id": str(cell_id),
            "source": source_label,
            "chemistry": chemistry,
            "formFactor": form_factor,
            "formFactorNote": form_factor_note,
            "provenance": provenance,
            "dqdvApplicable": dqdv_applicable,
        },
        "series": {
            "cycles": cycles,
            "sohPct": soh,
            "capacityAh": capacity,
            "resistanceOhm": resistance,
            "resistanceNormalized": resistance_norm,
            "temperatureC": temperature,
            "sopPct": sop,
            "fadeRate30cy": fade_30,
            "seiPct": phys["seiPct"],
            "lamPct": phys["lamPct"],
            "seiSharePct": phys["seiSharePct"],
            "fadeModelPct": phys["fadeModelPct"],
        },
        "record": {
            "firstCycle": _opt(cycles[0]) if cycles else None,
            "lastCycle": _opt(cycles[-1]) if cycles else None,
            "nCycles": len(cycles),
            "nRecordedRows": n_rows,
            "sampled": len(cycles) < n_rows,
            "lastSohPct": _last_finite(soh),
        },
        "physics": {
            "law": _TWO_TERM_LAW,
            "seiTerm": _SEI_TERM,
            "lamTerm": _LAM_TERM,
            "available": bool(fit.get("available")),
            "splitIdentified": bool(fit.get("splitIdentified")),
            "splitReason": fit.get("splitReason"),
            "betaSei": fit.get("betaSei"),
            "betaLam": fit.get("betaLam"),
            "fitR2": fit_r2_last,
            "betaSeiSigma": fit.get("betaSeiSigma"),
            "betaLamSigma": fit.get("betaLamSigma"),
            "seiTStat": fit.get("seiTStat"),
            "lamTStat": fit.get("lamTStat"),
            "modelTotalFadePct": fit.get("modelTotalFadePct"),
            "measuredFadePct": fit.get("measuredFadePct"),
            "kR": fit.get("kR"),
            "kRSigma": fit.get("kRSigma"),
            "nCyclesUsed": fit.get("nCyclesUsed"),
            "reason": fit.get("reason"),
            "fitR2Series": fit_r2_series,
            "refitEveryCycles": 25,
            "perWindowBetasInFrame": n_window_betas,
            "spmCapacityAh": _opt(_column(df, "physics_spm_capacity_ah")[-1]) if _column(df, "physics_spm_capacity_ah") else None,
            "source": (
                "batlab.features.physics_calibration.fit_two_term_fade over this cell's full recorded "
                "history (the same fit calibrate_cell() calls; the PyBaMM SPM path is not used here)"
            ),
        },
        "mechanism": mechanism,
        "knee": knee,
        "projection": projection,
        "parts": parts,
        "geometryScales": _geometry_scales(),
        "disclosures": _disclosures(
            form_factor, form_factor_note, mechanism, projection, dqdv_applicable, provenance,
            bool(fit.get("splitIdentified")), str(fit.get("splitReason") or ""),
        ),
        "theme": {**default_theme(), **(theme or {})},
    }
    return _jsonify(spec)


def _geometry_scales() -> dict:
    """How the renderer turns each data series into drawn size — display, not physics.

    Kept explicit and in the spec so the renderer never invents a mapping, and
    so the part card can print the one it used. ``note`` is the sentence the
    scene shows when a user switches to data-scaled geometry.
    """
    return {
        "fade_model": {
            "target": "capacityLoss",
            "renderAs": "stateGauge",
            "from": "series.fadeModelPct", "unit": "% of initial capacity (fitted)",
            "displayMin": 0.0, "displayMax": None,
            "note": (
                "The drawn capacity loss is the fitted two-term model's TOTAL fade. It is what the scene "
                "uses when the √n and linear channels cannot be separated for this cell, so it shows "
                "what the data identifies rather than an apportionment it does not. It is drawn as the "
                "state gauge on the casing — how much of a fresh cell's usable capacity this cell still "
                "holds — and never as the size of a physical layer."
            ),
        },
        "sei_film": {
            "target": "shellThickness",
            "from": "series.seiPct", "unit": "% of initial capacity (fitted)",
            "displayMin": 0.0, "displayMax": 20.0,
            "note": (
                "The film's drawn thickness and its glow both track the fitted √n lithium-inventory "
                "loss, from the anatomical minimum at zero to full thickness at 20% of initial capacity "
                "lost to lithium inventory. It is a legibility-scaled metaphor, not a film thickness in "
                "micrometres — no source here measures one — and it is drawn only when this cell's two "
                "fade channels are separable. The combined fitted fade is on the state gauge instead."
            ),
        },
        "particles": {
            "target": "lostFraction",
            "from": "series.lamPct", "unit": "% of initial capacity (fitted)",
            "displayMin": 0.0, "displayMax": None,
            "note": (
                "The share of drawn particles rendered as no longer cycling is the share of the fitted "
                "total fade attributed to active-material loss (100 minus the SEI share) — a share of "
                "this cell's fitted fade, not a count of real particles. Real particle loss is "
                "heterogeneous, and the drawn cloud is a legibility-limited sample of the coating."
            ),
        },
        "can": {
            "target": "colour",
            "from": "series.temperatureC", "unit": "°C",
            "displayMin": 0.0, "displayMax": 60.0,
            "note": "Casing tint maps the measured surface temperature onto a 0–60 °C display range.",
        },
        "tab_neg": {
            "target": "colour",
            "from": "series.sopPct", "unit": "% of initial power",
            "displayMin": 40.0, "displayMax": 110.0,
            "note": "Tab tint maps power capability; the 70% power-fade floor is drawn as a band edge.",
        },
        "terminal_pos": {
            "target": "crossSection",
            "from": "series.resistanceOhm", "unit": "Ω",
            "displayMin": None, "displayMax": None,
            "note": (
                "Drawn terminal/joint cross-section narrows as measured DC resistance rises. It is a "
                "schematic of the impedance, not a measurement of the weld."
            ),
        },
    }


def _disclosures(form_factor: str, form_factor_note: str, mechanism: dict,
                 projection: "dict | None", dqdv_applicable: bool,
                 provenance: str, split_identified: bool, split_reason: str) -> list[str]:
    """The honesty strip, assembled per cell rather than written once for all cells."""
    out = [
        "This scene is a schematic of a cell's construction, not a metrology model or a CAD drawing. "
        "Layer counts, turn counts and particle counts are drawn for legibility; the numbers on the "
        "part cards are the ones that are real.",
        "Anatomical proportions are shown by default. Data-scaled geometry is opt-in, and when it is "
        "on the scene prints the mapping it used.",
        "Every part card is tagged measured, derived, fitted or projected. A part this source cannot "
        "speak to says so instead of showing a zero.",
    ]
    if form_factor == FORM_FACTOR_PRISMATIC:
        out.append(
            f"Drawn as a prismatic (stacked-layer) cell because this source's cells are: "
            f"{form_factor_note or 'declared prismatic by its chemistry profile'}."
        )
    elif form_factor == FORM_FACTOR_UNKNOWN:
        out.append(
            "This cell's form factor is not declared, so it is drawn as a generic cylindrical cell. "
            "Do not read the internal geometry as this cell's construction."
        )
    else:
        out.append(
            "Drawn as a cylindrical 18650-class cell: "
            f"{form_factor_note or 'declared cylindrical by this source'}."
            " Internal geometry is schematic; the real jelly roll has far more turns than are drawn."
        )
    if mechanism.get("physics") and mechanism["physics"].get("gatePassed") is False:
        out.append(
            f"The SEI-vs-LAM split comes from a two-term fit whose r² ({mechanism['physics'].get('fitR2')}) "
            f"falls below the platform's {mechanism['physics'].get('gate')} gate for naming a dominant "
            "mechanism, so no dominant mechanism is named from the physics side."
        )
    if split_identified:
        out.append(
            "The SEI and LAM layers come from a joint fit of √n and n to capacity data alone, and for "
            "this cell both channels fit above their own standard error — but those two regressors are "
            "still strongly correlated over a narrow cycle range, so read the split as evidence rather "
            "than as a measurement of a film. The independently-fitted resistance-growth rate is the "
            "more robust standalone signal for an active SEI channel and is drawn on the tabs."
        )
    else:
        out.append(
            f"The SEI film and the active-material particles are drawn but carry no number. {split_reason}"
        )
    if not dqdv_applicable:
        out.append(
            "The electrode coatings carry no number for this cell: its source declares dQ/dV "
            "inapplicable (the platform's dQ/dV features simulate a LiCoO₂-shaped OCV curve, which the "
            "flat plateau of an LFP cell invalidates)."
        )
    else:
        out.append(
            "The electrode coatings carry a *proxy*, not an electrode measurement: the platform's dQ/dV "
            "features are simulated from each cycle's own capacity and resistance, so they re-express "
            "the capacity trend through an OCV model rather than adding independent evidence about the "
            "coating. The peak's state-of-charge position and width columns are constants of that model "
            "and are deliberately not drawn."
        )
    if projection is None or not projection.get("available"):
        reason = (projection or {}).get("reason") or "no forecast is available for this cell"
        out.append(f"The scene stops at the last measured cycle. {reason}")
    else:
        route = (projection or {}).get("route") or {}
        if route.get("reason"):
            out.append(f"Forecast routing: {route['reason']}")
        out.append(
            "Beyond the last measured cycle the cell is drawn from the platform's hierarchical forecast "
            "with its posterior band — a projection, not a prediction, and only where this cell's "
            "regime routing allows a forecast at all."
        )
    if provenance:
        out.append(f"Data provenance for this cell: {provenance}.")
    out.append(
        "Nothing in this view feeds Pack RUL, any published accuracy number, or a decision surface. It "
        "is an explanation of measurements taken elsewhere in the platform."
    )
    return out


def build_cell_scene_from_sources(
    cell_id: str,
    featured_dfs: dict,
    bundles: "dict | None" = None,
    *,
    graph=None,
    horizon_cycles: int = DEFAULT_HORIZON_CYCLES,
    max_points: int = MAX_SERIES_POINTS,
    theme: "dict | None" = None,
) -> "dict | None":
    """Convenience wrapper: find the cell's frame and bundle, then build the spec.

    Returns ``None`` when the cell has no frame loaded, so a caller can render
    its own empty state rather than a scene built from nothing.
    """
    df = None
    try:
        df = featured_dfs.get(cell_id)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        return None
    bundle = None
    if bundles:
        bundle = bundles.get(resolve_bundle_key(cell_id))
    return build_cell_scene(
        cell_id, df, bundle=bundle, bundles=bundles, graph=graph,
        horizon_cycles=horizon_cycles, max_points=max_points, theme=theme,
    )
