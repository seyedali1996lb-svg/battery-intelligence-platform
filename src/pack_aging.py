"""
Load-driven aging feedback for the Virtual Pack Builder.

``src/pack_builder.py`` reports *how much* more current the hardest-worked cell
carries than its fair share (the load ratio). That is a number with no time in
it. This module turns it into a consequence: if a cell carries ``k×`` its fair
share, how much faster does it fade, and how far does the pack's SOH spread
diverge as a result?

Two functions, both pure and independently testable:

``measure_cell_aging_inputs(frame, cell_id)``
    Reads the two *measured* inputs the projection needs out of a cell's own
    recorded history — neither is assumed from any source:

    - **the cell's own fade slope** (SOH percentage points lost per cycle), an
      ordinary least-squares fit of ``soh_pct`` against ``cycle_number`` over
      the cycles the cell actually recorded. The rolling ``fade_rate_30cy``
      column is a 30-cycle mean of |Δcapacity| and is dominated by per-cycle
      measurement noise (on B0005 it reads ~2× the cell's true long-run slope),
      so it is reported for context but not used as the projection rate.
    - **the cell's own resistance growth** (ohms per SOH percentage point), an
      OLS fit of ``resistance_ohm`` against ``soh_pct`` over points with a
      strictly positive resistance. This is what makes the load *move*: as the
      cells age at different rates their resistances separate, so the shares
      computed at pack assembly are not the shares the pack runs at forever.

    Both fits refuse rather than guess: a fade slope that is not positive, a
    resistance trend with fewer than ``MIN_RESISTANCE_POINTS`` usable points or
    a correlation weaker than ``MIN_RESISTANCE_CORRELATION``, and a history
    shorter than ``MIN_FADE_CYCLES`` all come back ``None`` with a reason, never
    as a substituted value. A 0 Ω reading is treated as the sentinel it is (see
    ``pack_builder._usable_resistance``), not as a short circuit.

``simulate_load_aging(cells, groups, ...)``
    Steps the pack forward cycle by cycle, twice — once with the loading
    feedback and once without — and reports the difference between the two.

    Per cycle, for each parallel group::

        share_c   = (1/R_c) / Σ_j (1/R_j)              [resistance-only, see
        load_c    = share_c × X                         pack_builder]
        fade_c    = fade_c⁰ × load_c ** α
        SOH_c    -= fade_c · Δcycle
        R_c      += gain_c · fade_c · Δcycle            [gain_c is measured]

    ``fade_c⁰`` is the cell's own measured fade slope, ``α`` is the assumed
    current→fade exponent, and ``gain_c`` is the cell's own measured dR/dSOH.

The direction of the feedback is *computed, not assumed*
--------------------------------------------------------
An easy story to tell is "the loaded cell heats itself into impedance and
sheds its share" — self-limiting. The loop does not encode that, because it is
not generally true: what decides the drift is each cell's **absolute**
resistance growth per cycle (``gain_c × fade_c``), and a cell with a large
starting resistance typically gains more ohms per cycle than a low-resistance
one even when its relative growth is the same. On NASA's fleet in a 2p build
the hardest-worked cell's load ratio *rises* (1.060 → 1.071 over 100 cycles)
rather than falling. ``load_relief_pct`` reports the measured direction per
cell; both signs occur in one pack, and neither is assumed here.

The same caution applies to divergence. Because resistance falls with age, the
cells that carry the most current are usually the *strongest* in their group —
so the feedback speeds their fade and the group's states of health **converge**,
narrowing the pack's spread, while a loaded cell that is already the weaker one
diverges. Current sharing redistributes aging rather than simply adding it, and
``spread_delta_pct`` carries the sign of whichever effect dominates the
selection actually modelled.

The exponent, stated rather than buried
---------------------------------------
The projection needs one number the data cannot supply. A reference fleet is
cycled at one duty cycle, so it contains no variation in load to identify a
rate exponent from — the honest move is to name the assumption and let it be
swept, which is what the UI does.

``DEFAULT_CURRENT_AGING_EXPONENT = 0.7`` is deliberately **not** a new number:
it is the same ``(C / 1C) ** 0.7`` sub-linear C-rate term the platform's own
``stress_index`` feature already applies (``batlab/features/engineering.py``,
with its stated rationale — current distribution in a porous electrode is not
uniform, so fade is sub-linear in C-rate, Doyle-Fuller-Newman). One convention
across the app beats a second invented one. It is an assumption borrowed for
consistency, not a parameter fitted to any fleet here, and α = 0.7 is the
default the UI shows, not a hidden constant.

What this is, and what it is not
--------------------------------
It is a **scenario comparison**, not a lifetime prediction:

- The two runs share the same fade model, so the *difference* between them is
  attributable to loading rather than to the fade model's imperfections. The
  absolute SOH at the horizon inherits every approximation; the divergence that
  loading adds does not.
- Fade is held constant per cycle — no knee acceleration is extrapolated. Real
  fade accelerates with age, and the loaded cell's extra loss compounds with
  it, so the extra loss reported here is a **lower bound** on the divergence
  the loading will produce.
- The fair-share assumption is explicit: a load ratio of 1.0 means the pack is
  loaded so that an evenly-split group reproduces the cell's *historical*
  per-cell current. A pack driven harder than the cells' measured duty has an
  additional, uniform acceleration that this comparison does not show (it would
  shift both runs together and by the same factor).
- Sharing is the first-order, resistance-only, steady-state model of
  ``pack_builder.compute_group_current_sharing()``: no SOC-dependent OCV
  differences, no thermal feedback, no contact resistance.
- A cell whose resistance growth could not be measured is projected with a
  **constant** resistance, so its load never decays — the upper bound of the
  loading effect, disclosed per cell rather than averaged away.
- Nothing here is fed back into ``pack_rul`` or any published accuracy number.
"""

from __future__ import annotations

import math

from pack_builder import _finite, _usable_resistance

# ---------------------------------------------------------------------------
# Assumptions (stated, sweepable)
# ---------------------------------------------------------------------------

#: The platform's existing sub-linear C-rate exponent (see ``stress_index`` in
#: ``batlab/features/engineering.py``). Used here as the current→fade exponent.
DEFAULT_CURRENT_AGING_EXPONENT = 0.7

#: Default scenario horizon, in cycles. Arbitrary and disclosed as such.
DEFAULT_HORIZON_CYCLES = 100.0
DEFAULT_STEP_CYCLES = 1.0

#: Points recorded per projected trajectory (including both endpoints).
MAX_RECORDED_POINTS = 41

#: Minimum evidence before a measured input is reported at all.
MIN_FADE_CYCLES = 20
MIN_RESISTANCE_POINTS = 10
MIN_RESISTANCE_CORRELATION = 0.5
MIN_RECENT_CYCLES = 5
RECENT_WINDOW_CYCLES = 30

#: The platform's own end-of-life convention (``eol_threshold_pct = 80``).
EOL_SOH_PCT = 80.0

#: Hard floor for the projection's arithmetic, applied to *both* runs so the
#: two stay comparable: a negative SOH is meaningless to draw, and a cell that
#: burns through its whole capacity inside the horizon is the case where the
#: scenario has stopped being informative anyway. Such a cell reports the floor,
#: and the cycle it crossed EOL is reported separately.
SOH_FLOOR_PCT = 0.0


def _finite_or_none(value) -> "float | None":
    """``value`` as a real number, or None when missing/non-numeric/NaN."""
    num = _finite(value, float("nan"))
    return None if num != num else num


def _linear_fit(xs: list[float], ys: list[float]) -> tuple["float | None", "float | None"]:
    """OLS slope and Pearson correlation of ``ys`` against ``xs``.

    Returns ``(None, None)`` when the fit is undetermined (fewer than two
    points, or no spread in either variable) rather than a slope of 0, which
    would silently read as "measured: no trend".
    """
    n = len(xs)
    if n < 2:
        return None, None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None, None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def _column(frame, name: str) -> list:
    """One column of a cell frame as a plain list, or ``[]`` when absent."""
    if frame is None:
        return []
    try:
        if name not in frame.columns:
            return []
        return list(frame[name])
    except AttributeError:
        return []


def measure_cell_aging_inputs(cell_frame, cell_id: "str | None" = None) -> dict:
    """The measured inputs the load-aging projection needs, from one cell frame.

    Everything returned is either a number read out of this cell's own record or
    an explicit refusal with a reason (``unavailable``). Nothing is imputed and
    no cross-fleet default is substituted — a cell that hasn't recorded enough
    history says so.

    Keys:
      cycle_min/cycle_max   the recorded cycle window the fits were taken over
      soh_now               the cell's latest finite SOH (%)
      resistance_ohm        the cell's latest strictly-positive resistance (Ω)
      fade_pct_per_cycle    measured long-run fade slope (%SOH/cycle), or None
      fade_r2               fit quality of that slope (None when unavailable)
      fade_cycles           points behind the slope
      fade_recent_pct_per_cycle / fade_recent_r2
                            the same fit over the last RECENT_WINDOW_CYCLES,
                            reported for context only — the projection uses the
                            long-run slope, and the UI shows both when they
                            disagree materially
      resistance_gain_ohm_per_soh_pct
                            measured dR/dSOH (Ω per SOH point, positive = grows
                            as the cell fades), or None
      resistance_gain_corr / resistance_points
                            fit quality and evidence behind that gain
      resistance_gain_measured
                            True only when the gain is a real measurement; False
                            means the projection must hold resistance constant
                            (the constant-load upper bound)
      unavailable           list of (what, why) refusals, for disclosure
    """
    out = {
        "cell_id": str(cell_id or ""),
        "cycle_min": None, "cycle_max": None,
        "soh_now": float("nan"), "resistance_ohm": float("nan"),
        "fade_pct_per_cycle": None, "fade_r2": None, "fade_cycles": 0,
        "fade_recent_pct_per_cycle": None, "fade_recent_r2": None,
        "resistance_gain_ohm_per_soh_pct": None, "resistance_gain_corr": None,
        "resistance_points": 0, "resistance_gain_measured": False,
        "unavailable": [],
    }

    cycles = _column(cell_frame, "cycle_number")
    sohs = _column(cell_frame, "soh_pct")
    resistances = _column(cell_frame, "resistance_ohm")
    n_rows = min(len(cycles), len(sohs)) if cycles else 0

    if n_rows == 0:
        out["unavailable"].append(("cell history", "no recorded cycles with SOH"))
        return out

    finite_cycles = [float(cycles[i]) for i in range(n_rows) if _finite_or_none(sohs[i]) is not None]
    out["cycle_min"] = min(finite_cycles) if finite_cycles else None
    out["cycle_max"] = max(finite_cycles) if finite_cycles else None
    _latest_soh = _finite_or_none(sohs[n_rows - 1])
    out["soh_now"] = _latest_soh if _latest_soh is not None else float("nan")

    latest_resistance = float("nan")
    for value in reversed(resistances):
        usable = _usable_resistance(value)
        if usable is not None:
            latest_resistance = usable
            break
    out["resistance_ohm"] = latest_resistance

    # ── Long-run fade slope: SOH% lost per cycle, over the whole record ────
    fit_cycles: list[float] = []
    fit_sohs: list[float] = []
    for i in range(n_rows):
        soh = _finite_or_none(sohs[i])
        if soh is None:
            continue
        fit_cycles.append(float(cycles[i]))
        fit_sohs.append(soh)
    out["fade_cycles"] = len(fit_cycles)
    if len(fit_cycles) >= MIN_FADE_CYCLES:
        slope, corr = _linear_fit(fit_cycles, fit_sohs)
        if slope is not None and slope < 0.0:
            out["fade_pct_per_cycle"] = -slope
            out["fade_r2"] = (corr * corr) if corr is not None else None
        else:
            out["unavailable"].append((
                "fade rate",
                f"no measurable fade across {len(fit_cycles)} recorded cycles "
                f"(SOH trend is flat or rising)",
            ))
    else:
        out["unavailable"].append((
            "fade rate",
            f"only {len(fit_cycles)} cycles with SOH recorded (need {MIN_FADE_CYCLES})",
        ))

    # ── Recent fade slope, context only ───────────────────────────────────
    if fit_cycles:
        cutoff = fit_cycles[-1] - RECENT_WINDOW_CYCLES
        recent_cycles = [c for c in fit_cycles if c >= cutoff]
        recent_sohs = [s for c, s in zip(fit_cycles, fit_sohs) if c >= cutoff]
        if len(recent_cycles) >= MIN_RECENT_CYCLES:
            slope, corr = _linear_fit(recent_cycles, recent_sohs)
            if slope is not None and slope < 0.0:
                out["fade_recent_pct_per_cycle"] = -slope
                out["fade_recent_r2"] = (corr * corr) if corr is not None else None

    # ── Resistance growth per SOH point: the self-limiting term ────────────
    gain_sohs: list[float] = []
    gain_res: list[float] = []
    for i in range(min(len(sohs), len(resistances))):
        soh = _finite_or_none(sohs[i])
        resistance = _usable_resistance(resistances[i])
        if soh is None or resistance is None:
            continue
        gain_sohs.append(soh)
        gain_res.append(resistance)
    out["resistance_points"] = len(gain_sohs)
    if len(gain_sohs) < MIN_RESISTANCE_POINTS:
        out["unavailable"].append((
            "resistance growth",
            f"only {len(gain_sohs)} cycles with a positive resistance "
            f"(need {MIN_RESISTANCE_POINTS})",
        ))
    else:
        slope, corr = _linear_fit(gain_sohs, gain_res)
        if slope is None:
            out["unavailable"].append(("resistance growth", "resistance never varies with SOH in this record"))
        elif slope >= 0.0:
            out["unavailable"].append((
                "resistance growth",
                "measured resistance falls as SOH falls — refusing to project a "
                "self-limiting term from that",
            ))
        elif corr is None or corr > -MIN_RESISTANCE_CORRELATION:
            out["unavailable"].append((
                "resistance growth",
                f"resistance/SOH correlation too weak to fit "
                f"(r={corr if corr is None else round(corr, 2)}, need ≤ -{MIN_RESISTANCE_CORRELATION})",
            ))
        else:
            out["resistance_gain_ohm_per_soh_pct"] = -slope
            out["resistance_gain_corr"] = corr
            out["resistance_gain_measured"] = True

    return out


def _record_grid(n_steps: int) -> list[int]:
    """The step indices to record a trajectory at (always both endpoints)."""
    if n_steps <= 0:
        return [0]
    points = min(MAX_RECORDED_POINTS, n_steps + 1)
    grid = sorted({int(round(i * n_steps / (points - 1))) for i in range(points)})
    return grid


def _eol_crossing(cycles: list[float], sohs: list[float], eol: float) -> "float | None":
    """First cycle at which SOH reaches ``eol``, linearly interpolated.

    Returns None when the trajectory never gets there inside the horizon, so a
    cell that stays above EOL is reported as such rather than given a
    projected end-of-life it hasn't reached.
    """
    for i in range(1, len(sohs)):
        if sohs[i] <= eol < sohs[i - 1]:
            span = sohs[i - 1] - sohs[i]
            if span <= 0.0:
                return cycles[i]
            frac = (sohs[i - 1] - eol) / span
            return cycles[i - 1] + frac * (cycles[i] - cycles[i - 1])
    if sohs and sohs[0] <= eol:
        return cycles[0]
    return None


def _project_group(members: list[dict], *, step: float, exponent: float,
                   sharing: bool, grid: list[int]) -> None:
    """Run one parallel group's cells together, updating each member in place.

    Only cells that share a node can affect each other, so the loop is per
    group: the group's shares are recomputed from every member's resistance at
    each step, then each member's own fade and resistance growth are applied.
    That recomputation is the whole point — the shares drift as the cells' own
    measured resistances separate, in whichever direction their measured gains
    dictate (see this module's note: the drift is not assumed to be
    self-limiting).

    ``sharing=False`` (a group of one, or a group whose cells can't all be
    projected) runs each member at its own baseline load, so the two scenarios
    coincide rather than producing a fabricated feedback number.
    """
    n_steps = grid[-1] if grid else 0

    # Seed each cell's resistance and its share of the first step.
    for m in members:
        m["_resistance"] = m["resistance0"]
        m["_soh"] = m["soh0"]
    if sharing:
        cond = [1.0 / m["resistance0"] for m in members]
        total = sum(cond)
        for m, c in zip(members, cond):
            m["_current_ratio"] = (c / total) * len(members) if total > 0 else 1.0
    else:
        for m in members:
            m["_current_ratio"] = 1.0

    cycles: list[float] = []
    soh_with: list[list[float]] = [[] for _ in members]
    soh_without: list[list[float]] = [[] for _ in members]
    ratios: list[list[float]] = [[] for _ in members]
    grid_set = set(grid)

    for k in range(n_steps + 1):
        if k in grid_set:
            cycles.append(k * step)
            for j, m in enumerate(members):
                soh_with[j].append(m["_soh"])
                soh_without[j].append(max(SOH_FLOOR_PCT, m["soh0"] - m["fade"] * k * step))
                ratios[j].append(m["_current_ratio"] if sharing and exponent > 0 else 1.0)
        if k == n_steps:
            break

        for m in members:
            ratio = m["_current_ratio"] if (sharing and exponent > 0.0) else 1.0
            fade = m["fade"] * (ratio ** exponent if (sharing and exponent > 0.0) else 1.0)
            delta = fade * step
            m["_soh"] = max(SOH_FLOOR_PCT, m["_soh"] - delta)
            if m.get("gain"):
                m["_resistance"] = m["_resistance"] + float(m["gain"]) * delta

        if sharing and exponent > 0.0:
            cond = [1.0 / m["_resistance"] if m["_resistance"] > 0 else 0.0 for m in members]
            total = sum(cond)
            for m, c in zip(members, cond):
                m["_current_ratio"] = (c / total) * len(members) if total > 0 else 1.0

    for j, m in enumerate(members):
        # A cell already past EOL at the latest measured cycle has no "cycles to
        # EOL" to compare — reporting 0 for both runs would read as "the
        # loading changes nothing", which is not what it means.
        below_eol = bool(soh_with[j]) and soh_with[j][0] <= EOL_SOH_PCT
        record = {
            "cycles": cycles,
            "soh_with": soh_with[j],
            "soh_without": soh_without[j],
            "load_ratio": ratios[j],
            "soh_with_end": soh_with[j][-1] if soh_with[j] else float("nan"),
            "soh_without_end": soh_without[j][-1] if soh_without[j] else float("nan"),
            "extra_loss_pct": (soh_without[j][-1] - soh_with[j][-1]) if soh_with[j] else float("nan"),
            "load_ratio_start": ratios[j][0] if ratios[j] else 1.0,
            "load_ratio_end": ratios[j][-1] if ratios[j] else 1.0,
            "below_eol_at_start": below_eol,
            "eol_with": None if below_eol else _eol_crossing(cycles, soh_with[j], EOL_SOH_PCT),
            "eol_without": None if below_eol else _eol_crossing(cycles, soh_without[j], EOL_SOH_PCT),
        }
        m["result"] = record


def _stdev(values: list[float]) -> float:
    """Population-free sample stdev, NaN when fewer than two values."""
    finite = [v for v in values if v == v]
    if len(finite) < 2:
        return float("nan")
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))


def simulate_load_aging(cell_stats: list, groups: list[list[int]], *,
                        horizon_cycles: float = DEFAULT_HORIZON_CYCLES,
                        exponent: float = DEFAULT_CURRENT_AGING_EXPONENT,
                        step_cycles: float = DEFAULT_STEP_CYCLES) -> dict:
    """Estimate what the current-sharing load ratio does to the pack over time.

    ``cell_stats`` entries carry ``cell_id``, ``soh_pct``, ``resistance_ohm``
    plus the measured inputs from :func:`measure_cell_aging_inputs`:
    ``fade_pct_per_cycle`` and, when available,
    ``resistance_gain_ohm_per_soh_pct``. ``groups`` is the same list of index
    groups ``pack_builder.build_parallel_groups()`` returns.

    Runs the pack forward twice — loading feedback on and off — over
    ``horizon_cycles`` and returns both trajectories plus the difference
    between them. The load ratio is recomputed from the resistances at every
    step, so the self-limiting effect is present rather than assumed away.

    Returns ``available=False`` with a reason when no parallel group can be
    modelled at all (no two cells sharing a node with usable inputs); every
    refusal and every cell projected with a constant resistance is named in the
    result rather than averaged into a headline.
    """
    horizon = _finite(horizon_cycles, DEFAULT_HORIZON_CYCLES)
    horizon = horizon if horizon > 0 else DEFAULT_HORIZON_CYCLES
    step = _finite(step_cycles, DEFAULT_STEP_CYCLES)
    step = step if step > 0 else DEFAULT_STEP_CYCLES
    alpha = _finite(exponent, DEFAULT_CURRENT_AGING_EXPONENT)

    result = {
        "available": False, "unavailable_reason": None,
        "exponent": alpha, "horizon_cycles": horizon, "step_cycles": step,
        "eol_soh_pct": EOL_SOH_PCT,
        "cells": [], "groups": [], "excluded": [],
        "loaded_cell_id": None, "loaded_group": None,
        "loaded_load_ratio": float("nan"),
        "loaded_fade_acceleration_pct": float("nan"),
        "loaded_load_relief_pct": float("nan"),
        "loaded_extra_loss_pct": float("nan"),
        "loaded_equivalent_cycles": float("nan"),
        "loaded_eol_shift_cycles": None,
        "spread_without_pct": float("nan"), "spread_with_pct": float("nan"),
        "spread_delta_pct": float("nan"),
        "bottleneck_with": None, "bottleneck_without": None,
        "pack_soh_with_pct": float("nan"), "pack_soh_without_pct": float("nan"),
        "n_projected": 0, "n_constant_load": 0,
        "spread_curve_cycles": [], "spread_curve_with": [], "spread_curve_without": [],
    }
    if not cell_stats or not groups:
        result["unavailable_reason"] = "no cells selected"
        return result

    # ── Which cells can be projected at all ───────────────────────────────
    stats: list[dict | None] = []
    for cell in cell_stats:
        soh0 = _finite(cell.get("soh_pct"), float("nan"))
        fade = _finite(cell.get("fade_pct_per_cycle"), float("nan"))
        resistance0 = _usable_resistance(cell.get("resistance_ohm"))
        unavailable = []
        if soh0 != soh0:
            unavailable.append("SOH not available")
        if fade != fade or fade <= 0.0:
            unavailable.append("no measured fade slope")
        if resistance0 is None:
            unavailable.append("no usable resistance")
        if unavailable:
            stats.append(None)
            result["excluded"].append({
                "cell_id": str(cell.get("cell_id", "?")),
                "reason": "; ".join(unavailable),
            })
            continue
        gain = _finite(cell.get("resistance_gain_ohm_per_soh_pct"), float("nan"))
        stats.append({
            "cell_id": str(cell.get("cell_id", "?")),
            "soh0": soh0, "fade": fade, "resistance0": resistance0,
            "gain": gain if gain == gain and gain > 0 else None,
            "gain_measured": bool(cell.get("resistance_gain_measured")),
            "_soh": soh0, "_resistance": resistance0, "_current_ratio": 1.0,
        })

    if not any(s is not None for s in stats):
        result["unavailable_reason"] = (
            "no selected cell has a measurable fade slope and resistance to project from "
            "(needs a recorded SOH and resistance history)"
        )
        return result

    n_steps = int(math.ceil(horizon / step))
    grid = _record_grid(n_steps)

    # ── Per group: sharing is modelled only when the whole group is usable ─
    projected: list[dict] = []
    for group_index, members in enumerate(groups):
        member_ids = [str(cell_stats[i].get("cell_id", "?")) for i in members if 0 <= i < len(cell_stats)]
        group_stats = [stats[i] for i in members if 0 <= i < len(stats)]
        usable = [s for s in group_stats if s is not None]
        sharing = len(usable) >= 2 and len(usable) == len(group_stats)
        reason = None
        if len(group_stats) < 2:
            reason = "only one cell shares this node — a series string loads every cell identically"
        elif len(usable) != len(group_stats):
            missing = [str(cell_stats[i].get("cell_id", "?")) for i in members
                       if 0 <= i < len(stats) and stats[i] is None]
            reason = (
                "current sharing not modelled for this group because "
                f"{len(missing)} of its cells cannot be projected ({', '.join(missing)})"
            )

        if usable:
            _project_group(
                usable, step=step, exponent=alpha, sharing=sharing, grid=grid,
            )
        result["groups"].append({
            "group": group_index + 1,
            "cell_ids": member_ids,
            "sharing_modelled": sharing,
            "reason": reason,
            "load_ratio_start": (
                usable[0]["result"]["load_ratio_start"] if sharing and usable else 1.0
            ),
        })

        for member in usable:
            outcome = member["result"]
            ratio_start = outcome["load_ratio_start"]
            record = {
                "cell_id": member["cell_id"],
                "group": group_index + 1,
                "soh_start": member["soh0"],
                "fade_pct_per_cycle": member["fade"],
                "resistance_start_ohm": member["resistance0"],
                "resistance_gain_ohm_per_soh_pct": member["gain"],
                "resistance_gain_measured": bool(member["gain_measured"]),
                "sharing_modelled": sharing,
                "sharing_reason": reason,
                "load_ratio_start": ratio_start,
                "load_ratio_end": outcome["load_ratio_end"],
                "load_relief_pct": (
                    (outcome["load_ratio_end"] / ratio_start - 1.0) * 100.0
                    if sharing and ratio_start > 0 else 0.0
                ),
                "fade_acceleration_pct": (
                    (ratio_start ** alpha - 1.0) * 100.0 if sharing and alpha > 0 else 0.0
                ),
                "soh_with_end": outcome["soh_with_end"],
                "soh_without_end": outcome["soh_without_end"],
                "extra_loss_pct": outcome["extra_loss_pct"],
                # The extra loss expressed in the cell's own baseline cycles:
                # "this imbalance spent X cycles' worth of its measured fade".
                # Reads on a cell already past EOL, where a cycle-to-EOL shift
                # does not.
                "equivalent_cycles_lost": (
                    outcome["extra_loss_pct"] / member["fade"] if member["fade"] > 0 else float("nan")
                ),
                "below_eol_at_start": outcome["below_eol_at_start"],
                "eol_with": outcome["eol_with"],
                "eol_without": outcome["eol_without"],
                "cycles": outcome["cycles"],
                "soh_with": outcome["soh_with"],
                "soh_without": outcome["soh_without"],
                "load_ratio_curve": outcome["load_ratio"],
            }
            projected.append(record)

    if not projected:
        result["unavailable_reason"] = "no cell could be projected"
        return result

    result["available"] = True
    result["n_projected"] = len(projected)
    result["n_constant_load"] = sum(1 for r in projected if r["resistance_gain_ohm_per_soh_pct"] is None)
    result["cells"] = projected

    # ── The loaded cell: the one carrying the most current at t=0 ──────────
    loaded = max(projected, key=lambda r: (r["load_ratio_start"], r["cell_id"]))
    result["loaded_cell_id"] = loaded["cell_id"]
    result["loaded_group"] = loaded["group"]
    result["loaded_load_ratio"] = loaded["load_ratio_start"]
    result["loaded_fade_acceleration_pct"] = loaded["fade_acceleration_pct"]
    result["loaded_load_relief_pct"] = loaded["load_relief_pct"]
    result["loaded_extra_loss_pct"] = loaded["extra_loss_pct"]
    result["loaded_equivalent_cycles"] = loaded["equivalent_cycles_lost"]
    if loaded["eol_with"] is not None and loaded["eol_without"] is not None:
        result["loaded_eol_shift_cycles"] = loaded["eol_without"] - loaded["eol_with"]

    # ── Divergence: the pack's spread, with and without the feedback ───────
    result["spread_without_pct"] = _stdev([r["soh_without_end"] for r in projected])
    result["spread_with_pct"] = _stdev([r["soh_with_end"] for r in projected])
    if (result["spread_with_pct"] == result["spread_with_pct"]
            and result["spread_without_pct"] == result["spread_without_pct"]):
        result["spread_delta_pct"] = result["spread_with_pct"] - result["spread_without_pct"]

    with_end = [r for r in projected if r["soh_with_end"] == r["soh_with_end"]]
    without_end = [r for r in projected if r["soh_without_end"] == r["soh_without_end"]]
    result["bottleneck_with"] = min(with_end, key=lambda r: r["soh_with_end"])["cell_id"] if with_end else None
    result["bottleneck_without"] = min(without_end, key=lambda r: r["soh_without_end"])["cell_id"] if without_end else None
    result["pack_soh_with_pct"] = min((r["soh_with_end"] for r in with_end), default=float("nan"))
    result["pack_soh_without_pct"] = min((r["soh_without_end"] for r in without_end), default=float("nan"))

    # Spread-vs-cycle curve — divergence as a *rate*, not just an endpoint.
    reference = projected[0]["cycles"]
    result["spread_curve_cycles"] = list(reference)
    for i, cycle in enumerate(reference):
        slice_cells = [r for r in projected if len(r["cycles"]) > i]
        result["spread_curve_without"].append(_stdev([r["soh_without"][i] for r in slice_cells]))
        result["spread_curve_with"].append(_stdev([r["soh_with"][i] for r in slice_cells]))

    return result
