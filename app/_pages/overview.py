"""Page: Overview."""

import sys
import os
import html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    _md_html, _action_bar, base_layout, soh_status, _soh_sparkline_svg,
    _cell_provenance, _resample_df, LEGEND_H,
)
from data_loader import CELL_STRESS_PROFILES, _stress_factor
from batlab.validation.lco import RUL_RELIABLE_FLOOR
from trajectory_memory import reconcile_rul_estimates
from chemistry_profiles import ChemistryProfile


# ---------------------------------------------------------------------------
# Trajectory match card
# ---------------------------------------------------------------------------

def _render_trajectory_match_card(
    cell_id: str,
    match: "TrajectoryMatch | None",
) -> None:
    """
    Render the trajectory-match warning card for an already-resolved match.

    The match is computed once by the caller (page_overview), alongside
    reconcile_rul_estimates() — not recomputed here — so this card is only
    ever shown when it does NOT materially disagree with the primary RUL
    model's estimate; a disagreeing match is routed to
    _render_reconciliation_card() instead, never both.
    """
    if match is None:
        return

    # Colour scheme by severity
    _colours = {
        "critical": ("#7f1d1d", "#fca5a5", "#ef4444"),
        "high":     ("#451a03", "#fcd34d", "#f59e0b"),
        "watch":    ("#0c1a3d", "#93c5fd", "#3b82f6"),
    }
    _bg, _text, _border = _colours.get(match.warning_level, _colours["watch"])
    _icon = "⚠" if match.warning_level == "critical" else ("▲" if match.warning_level == "high" else "●")
    _level_label = match.warning_level.upper()

    _sim_pct = int(match.best_similarity * 100)
    _cells_str = ", ".join(match.matched_cells[:3])
    if match.n_matches > 3:
        _cells_str += f" +{match.n_matches - 3} more"

    # Recommendation sentence
    _action_cycle = match.predicted_eol_min - 5
    _rec = (
        f"Schedule replacement before cycle {_action_cycle:,}."
        if _action_cycle > match.current_cycle
        else "Immediate inspection recommended — estimated EOL is imminent."
    )

    _md_html(
        f"""<div style="
            background:{_bg};
            border:1px solid {_border};
            border-left:4px solid {_border};
            border-radius:8px;
            padding:14px 18px;
            margin-bottom:16px;
        ">
          <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
                      color:{_text};margin-bottom:6px">
            {_icon} TRAJECTORY MATCH — {_level_label}
          </div>
          <div style="font-size:13px;color:#e2e8f0;line-height:1.6;margin-bottom:8px">
            This cell's degradation trajectory closely matches
            <strong style="color:{_text}">{match.n_matches}
            {"cell" if match.n_matches == 1 else "cells"}</strong>
            that reached end-of-life in the training data
            (best match: <strong style="color:{_text}">{match.best_cell_id}</strong>,
            {_sim_pct}% similarity · <em>{match.failure_mode}</em>).
          </div>
          <div style="display:flex;gap:24px;margin-bottom:10px">
            <div>
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.07em">You are at</div>
              <div style="font-size:20px;font-weight:700;color:#e2e8f0">
                Cycle {match.current_cycle:,}
              </div>
              <div style="font-size:11px;color:#94a3b8">SOH {match.current_soh:.1f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.07em">Est. cycles remaining</div>
              <div style="font-size:20px;font-weight:700;color:{_text}">
                {match.cycles_remaining_min}–{match.cycles_remaining_max}
              </div>
              <div style="font-size:11px;color:#94a3b8">
                before EOL · based on {match.n_matches} matched {"trajectory" if match.n_matches == 1 else "trajectories"}
              </div>
            </div>
            <div>
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.07em">Predicted EOL window</div>
              <div style="font-size:20px;font-weight:700;color:{_text}">
                Cycle {match.predicted_eol_min:,}–{match.predicted_eol_max:,}
              </div>
              <div style="font-size:11px;color:#94a3b8">
                Matched cells: {_cells_str}
              </div>
            </div>
          </div>
          <div style="font-size:12px;color:{_text};font-weight:600">
            → {_rec}
          </div>
        </div>"""
    )
    if st.button("Go to Decide & Ask →", key=f"traj_to_dec_{cell_id}", use_container_width=True):
        st.session_state.page = "decision"
        st.rerun()


def _render_reconciliation_card(cell_id: str, reconcile: dict) -> None:
    """
    Render ONE unified card when the primary RUL model and the trajectory-
    match model disagree beyond reconcile_rul_estimates()'s threshold,
    instead of showing both as independently confident widgets.

    Live-reproduced bug this replaces: the hero card said "657 cycles
    remaining, reliable" while the Trajectory Match card said "HIGH
    confidence, 35-65 cycles remaining" for the same cell, an 8x+
    disagreement with no reconciliation, hierarchy, or guidance on which
    number to act on.
    """
    primary = reconcile["primary_cycles"]
    match_mid = reconcile["match_cycles_mid"]
    ratio = reconcile["ratio"]
    favor = reconcile["favor"]
    favored_value = primary if favor == "primary" else match_mid
    favored_label = "the primary model's" if favor == "primary" else "the trajectory-match model's"

    _md_html(
        f"""<div style="
            background:#451a03;
            border:1px solid #f59e0b;
            border-left:4px solid #f59e0b;
            border-radius:8px;
            padding:14px 18px;
            margin-bottom:16px;
        ">
          <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
                      color:#fcd34d;margin-bottom:6px">
            ⚠ MODELS DISAGREE — TREAT WITH ELEVATED CAUTION
          </div>
          <div style="font-size:13px;color:#e2e8f0;line-height:1.6;margin-bottom:10px">
            The primary RUL model and the trajectory-match model give
            substantially different remaining-life estimates for this cell
            ({ratio:.0%} relative difference) — showing both as independent
            "reliable" numbers would be misleading, so here they are reconciled.
          </div>
          <div style="display:flex;gap:24px;margin-bottom:10px">
            <div>
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.07em">Primary model says</div>
              <div style="font-size:20px;font-weight:700;color:#e2e8f0">
                {primary:.0f} cycles
              </div>
              <div style="font-size:11px;color:#94a3b8">
                integrates this cell's full history
              </div>
            </div>
            <div>
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.07em">Trajectory match says</div>
              <div style="font-size:20px;font-weight:700;color:#e2e8f0">
                {match_mid:.0f} cycles
              </div>
              <div style="font-size:11px;color:#94a3b8">
                based on similarity to a short-term failure pattern
              </div>
            </div>
          </div>
          <div style="font-size:12px;color:#fcd34d;font-weight:600">
            → Recommended: plan around {favored_value:.0f} cycles remaining
            ({favored_label} more conservative estimate) until the two
            estimates converge with more data.
          </div>
        </div>"""
    )
    if st.button("Go to Decide & Ask →", key=f"reconcile_to_dec_{cell_id}", use_container_width=True):
        st.session_state.page = "decision"
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, split_cycle: int, cell_id: str,
                  rul_reliable: bool = True, bundle: dict | None = None,
                  trajectory_memory: "TrajectoryMemory | None" = None):
    _action_bar("overview")
    st.markdown(f"# Is {cell_id} healthy right now?")

    _rdf = _resample_df(df)   # downsampled for charts; df still used for latest/computations
    latest         = df.iloc[-1]
    current_soh    = latest["soh_pct"]
    current_rul    = latest["rul_pred"]
    current_cycle  = int(latest["cycle_number"])
    total_fade     = latest["capacity_fade_ah"]
    confidence     = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    # Quantile interval (populated by predict() for calibrated cells)
    rul_q10 = float(latest["rul_q10"]) if "rul_q10" in latest.index else None
    rul_q90 = float(latest["rul_q90"]) if "rul_q90" in latest.index else None

    # SoP from features (computed by build_features)
    sop_pct = float(latest["sop_pct"]) if "sop_pct" in latest.index else None

    # Per-cell fold R² — drives confidence inline reason
    fold_r2 = None
    _n_lco_cells = None
    if bundle is not None:
        _lco_per_cell = bundle["metrics"].get("lco_per_cell", {})
        cell_fold = _lco_per_cell.get(cell_id, {})
        fold_r2   = cell_fold.get("rul_r2", None)
        # Sample-size honesty (Battery Engineering Accuracy review finding):
        # leave-cell-out validation on n=4 (NASA) or n=12 (Severson) cells is
        # a thin population for any fleet-level reliability claim -- this
        # used to only appear in a settings-page footnote. Now shown
        # directly on the badge next to every RUL number, not just on request.
        if cell_id in _lco_per_cell:
            _n_lco_cells = len(_lco_per_cell)

    # RUL display: suppress when model doesn't generalise (LCO R² < floor)
    # or when early-cycle features haven't stabilised yet.
    rul_calibrating = (not rul_reliable) or (confidence == "Calibrating")
    rul_display     = "—" if rul_calibrating else f"{current_rul:.0f}"
    rul_sub         = "not calibrated" if not rul_reliable else "cycles to 80% SOH"

    # Application EOL threshold — adjust displayed RUL after rul_calibrating is known
    app_eol = float(st.session_state.get("eol_threshold_pct", 80.0))
    adj_rul = current_rul
    if not rul_calibrating and current_rul is not None and app_eol != 80.0:
        fade_50 = float(latest.get("fade_rate_50cy", 0)) * 100  # SOH %/cy
        if fade_50 > 1e-6:
            adj_rul = max(0, (current_soh - app_eol) / fade_50)
        rul_display = f"{adj_rul:.0f}"
        rul_sub     = f"cycles to {app_eol:.0f}% SOH (app threshold)"

    if rul_calibrating:
        if fold_r2 is not None and fold_r2 > 0:
            _pct_to_thresh = min(99, int(fold_r2 / RUL_RELIABLE_FLOOR * 100))
            conf_html = (
                f"<span class='tag-calibrating'>"
                f"Calibrating — {_pct_to_thresh}% to reliability</span>"
            )
        elif current_cycle < 40:
            _cycles_needed = max(1, 40 - current_cycle)
            conf_html = (
                f"<span class='tag-calibrating'>"
                f"Calibrating — needs {_cycles_needed} more cycles</span>"
            )
        else:
            conf_html = "<span class='tag-calibrating'>Calibrating</span>"
    else:
        conf_html = (
            f"<span class='tag-model' title='Leave-cell-out cross-validation on {_n_lco_cells} cells "
            f"— a small population; treat as directional, not a fleet-scale reliability guarantee.'>MODEL"
            + (f" · n={_n_lco_cells}" if _n_lco_cells else "")
            + (f" · R²={fold_r2:.2f}" if fold_r2 is not None else "")
            + "</span>"
        )

    # Cross-dataset transfer-error honesty badge (companion to the sample-
    # size badge above): if this cell's own dataset was ever used as the
    # training side of a logged cross-chemistry generalization study (see
    # src/experiment_registry.py's run_cross_chemistry_transfer()), show
    # the real transfer error next to the LCO badge — same "disclose the
    # real number, even a bad one" principle, extended from "does this
    # model generalize to a held-out cell of the same chemistry" to "does
    # it generalize to a different chemistry at all".
    _transfer_html = ""
    # Plain-text twin of _transfer_html's title= tooltip -- rendered visibly
    # below the hero card (see conf_reason's existing render site) since a
    # hover-only title= has no keyboard/screen-reader path. See the
    # accessibility audit that added this.
    _transfer_reason = ""
    try:
        import html as _html_mod
        import experiment_registry as _reg

        # cross_chemistry_runs_for_train_dataset() only knows "nasa"/"severson"/
        # "synth" (the one real transfer study logged) — Oxford/uploaded map to
        # None (no logged run) rather than a hand-rolled is_nasa/startswith chain.
        _kind         = ChemistryProfile.for_cell(cell_id).source_kind
        _cell_dataset = _kind if _kind in ("nasa", "severson", "synth") else None

        if _cell_dataset:
            _transfer_runs = _reg.cross_chemistry_runs_for_train_dataset(_cell_dataset)
            _evaluated = [r for r in _transfer_runs if r["rul_mae"] is not None]
            if _evaluated:
                _t = _evaluated[0]
                _eval_ds = _t["dataset"].split("_to_", 1)[1]
                _tooltip = (
                    f"Trained on {_cell_dataset}, zero-shot evaluated on {_eval_ds} — "
                    f"RUL MAE {_t['rul_mae']:.0f} cycles, R²={_t['rul_r2']:.2f}. "
                    "Cross-chemistry transfer is expected to be weak; see Benchmark for the full record."
                )
                _transfer_html = (
                    f"<span class='tag-calibrating' title=\"{_html_mod.escape(_tooltip)}\">"
                    f"CROSS-CHEM TRANSFER: WEAK ({_eval_ds})</span>"
                )
                _transfer_reason = _tooltip
            elif _transfer_runs:
                _t = _transfer_runs[0]
                _eval_ds = _t["dataset"].split("_to_", 1)[1]
                _transfer_html = (
                    f"<span class='tag-calibrating' title=\"{_html_mod.escape(_t['notes'] or '')}\">"
                    f"CROSS-CHEM TRANSFER: NOT EVALUATED ({_eval_ds})</span>"
                )
                _transfer_reason = _t["notes"] or ""
    except Exception:
        _transfer_html = ""  # honesty badge is best-effort -- never block the page over it
    rul_hero = "Not calibrated" if not rul_reliable else f"Est. {current_rul:.0f} cycles remaining"

    # ── Reconcile the primary RUL model against the trajectory-match model ──
    # Computed once here (not inside _render_trajectory_match_card) so the
    # hero card's plain-English summary and the trajectory panel below it
    # can never show two independently "confident" but contradictory
    # cycles-remaining numbers for the same cell.
    _traj_match = None
    if trajectory_memory is not None:
        try:
            _traj_match = trajectory_memory.match(cell_id, df)
        except Exception:
            _traj_match = None
    _primary_cycles_remaining = None if rul_calibrating else adj_rul
    _reconcile = reconcile_rul_estimates(_primary_cycles_remaining, _traj_match)

    # Confidence inline reason (two non-engineer templates per CTO spec)
    if rul_calibrating and not rul_reliable:
        if fold_r2 is not None:
            conf_reason = (
                "RUL not calibrated — model accuracy insufficient for this cell's "
                f"degradation profile (held-out fold R²={fold_r2:.2f}, below 0.30 reliability floor)"
            )
        else:
            conf_reason = (
                "RUL not calibrated — model accuracy insufficient for this cell's degradation profile"
            )
    elif not rul_calibrating and fold_r2 is not None:
        conf_reason = (
            f"RUL predictions were tested against data this cell never trained on "
            f"(leave-cell-out validation on {_n_lco_cells} cells) — fold R²={fold_r2:.2f} "
            f"(reliable above 0.30 floor). {_n_lco_cells} cells is a thin population — "
            f"treat as directional, not a fleet-scale reliability guarantee."
        )
    else:
        conf_reason = None

    # Time estimate: cycles/day from test_date if present, else assume 1 cycle/day
    if "test_date" in df.columns and df["test_date"].notna().any():
        dates = pd.to_datetime(df["test_date"].dropna())
        span_days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
        cycles_per_day = len(df) / span_days
        rate_note = f"estimated at {cycles_per_day:.1f} cycle/day from data"
    else:
        cycles_per_day = 1.0
        rate_note = "estimated at 1 cycle/day assumption"
    months_remaining = None
    if not rul_calibrating and current_rul is not None and cycles_per_day > 0:
        months_remaining = current_rul / cycles_per_day / 30.44

    # Source tag resolved via ChemistryProfile.for_cell() (not an is_nasa boolean)
    # so Severson/Oxford/uploaded cells report as real measured data instead of
    # falling into a "Synthetic" else-branch — same fix pattern as src/passport.py.
    _profile = ChemistryProfile.for_cell(cell_id)
    if _profile.provenance == "synthetic":
        p  = CELL_STRESS_PROFILES.get(cell_id, {})
        sf = _stress_factor(p.get("temp_mean",25), p.get("c_rate",1.0), p.get("dod",1.0))
        source_tag = f"Synthetic · Stress {sf:.2f}x baseline"
    elif _profile.source_kind == "nasa":
        source_tag = "NASA real · 24°C · 2A discharge"
    else:
        source_tag = f"{_profile.display_name} · real measured data"

    sparkline_svg = _soh_sparkline_svg(df["soh_pct"])
    interval_html = ""
    # D5: epistemic vs aleatoric uncertainty explanation
    _unc_explanation = ""
    if not rul_calibrating and rul_q10 is not None and rul_q90 is not None and rul_q90 > rul_q10:
        _band_width = rul_q90 - rul_q10
        _band_pct   = _band_width / max(float(current_rul), 1) * 100 if current_rul else 0
        if _band_pct > 40:
            # Wide band — diagnose source
            if current_cycle < 60:
                _unc_explanation = (
                    f"Wide uncertainty ({_band_width:.0f}-cycle spread) because this cell has only "
                    f"{current_cycle} cycles — the model hasn't seen enough of its degradation curve "
                    f"(epistemic). Uncertainty will narrow as cycles accumulate."
                )
            elif fold_r2 is not None and fold_r2 < 0.55:
                _unc_explanation = (
                    f"Wide uncertainty ({_band_width:.0f}-cycle spread) reflects genuine cell-to-cell "
                    f"variability in this chemistry — the model has enough data but the degradation "
                    f"pathway is inherently variable (aleatoric, R²={fold_r2:.2f})."
                )
            else:
                _unc_explanation = (
                    f"Uncertainty band spans {_band_width:.0f} cycles ({_band_pct:.0f}% of median RUL). "
                    f"Mix of data sparsity and chemistry variability — "
                    f"interval will tighten with more cycles."
                )
        elif _band_pct > 15:
            _unc_explanation = (
                f"Moderate uncertainty ({_band_width:.0f}-cycle band). "
                f"Model is well-calibrated; remaining spread reflects natural cycle-to-cycle variation."
            )
        interval_html = (
            f"<div style='font-size:11px;color:#8896a8;margin-top:4px'>"
            f"80% interval: <strong style='color:#a0aec0'>{rul_q10:.0f}–{rul_q90:.0f} cycles</strong>"
            + (f"&nbsp;·&nbsp;<span style='color:#a0aec0'>{_unc_explanation}</span>" if _unc_explanation else "")
            + f"</div>"
        )

    _md_html(
        f"""
        <div class="hero-card" role="region" aria-label="Battery status for cell {cell_id}">
            <div class="hero-label" aria-hidden="true">Battery Status · {cell_id}</div>
            <div class="hero-value {status_colour}" role="status" aria-label="Health status: {status_label}">{status_label}</div>
            <div class="hero-sub">
                SOH: <strong style="color:#e2e8f0">{current_soh:.1f}%</strong>
                &nbsp;<span>{sparkline_svg}</span>&nbsp;
                &nbsp;·&nbsp; {rul_hero}
                &nbsp;·&nbsp; {source_tag}
                &nbsp;·&nbsp; {conf_html}
                {("&nbsp;·&nbsp; " + _transfer_html) if _transfer_html else ""}
            </div>
            {interval_html}
        </div>
        """
    )

    if conf_reason:
        conf_c = "#fc8181" if (rul_calibrating and not rul_reliable) else "#4a5568"
        st.markdown(
            f"<div style='font-size:12px;color:{conf_c};margin:-8px 0 12px;"
            f"padding:6px 12px;background:#1a202c;border-radius:6px;"
            f"border-left:3px solid {conf_c}'>{conf_reason}</div>",
            unsafe_allow_html=True,
        )

    if _transfer_reason:
        st.markdown(
            f"<div style='font-size:12px;color:#a0aec0;margin:-8px 0 12px;"
            f"padding:6px 12px;background:#1a202c;border-radius:6px;"
            f"border-left:3px solid #a0aec0'>{html.escape(_transfer_reason)}</div>",
            unsafe_allow_html=True,
        )

    # ── Plain-English summary sentence ───────────────────────────────────────
    if rul_calibrating:
        _plain_sentence = (
            f"This cell has completed {current_cycle:,} cycles at {current_soh:.1f}% SOH — "
            f"RUL cannot be estimated reliably for this cell's degradation profile."
        )
    elif months_remaining is not None and months_remaining > 0:
        _action = (
            "no immediate action required" if current_soh >= 85
            else "consider inspection or load reduction" if current_soh >= 80
            else "replacement or second-life evaluation recommended"
        )
        if _reconcile["disagree"] and _reconcile["favor"] == "match":
            # The trajectory-match model's shorter, more urgent estimate is
            # the one being favored below — the plain-English sentence must
            # not simultaneously say "no immediate action required" purely
            # off this cell's current SOH while a "models disagree" card
            # recommends planning around a much shorter remaining life.
            _action = "see the models-disagree note below before assuming no action is needed"
        _plain_sentence = (
            f"This cell is estimated to need replacement in approximately "
            f"{months_remaining:.0f} months ({current_rul:.0f} cycles) at current usage — {_action}."
        )
    else:
        _plain_sentence = (
            f"This cell has completed {current_cycle:,} cycles at {current_soh:.1f}% SOH."
        )
    st.markdown(
        f"<div style='font-size:14px;color:#a0aec0;margin:-4px 0 18px;"
        f"padding:10px 16px;background:#1a202c;border-radius:8px;"
        f"border-left:3px solid #2d3748'>{_plain_sentence}</div>",
        unsafe_allow_html=True,
    )

    # ── Trajectory match warning ──────────────────────────────────────────────
    # A disagreeing match is reconciled into one unified card instead of
    # being shown alongside the hero card as a second, independently
    # "confident" number.
    if _reconcile["disagree"]:
        _render_reconciliation_card(cell_id, _reconcile)
    else:
        _render_trajectory_match_card(cell_id, _traj_match)

    # ── 📊 Fleet context: benchmark comparison + pin baseline (U1 density reduction) ──
    with st.expander("📊 Fleet context", expanded=False):
        # ── Benchmark comparison (M4) ────────────────────────────────────────
        try:
            _bm_fdfs = bundle.get("featured_dfs") or {}
            if len(_bm_fdfs) >= 3:
                _peer_sohs = []
                for _bm_cid, _bm_df in _bm_fdfs.items():
                    if _bm_cid != cell_id and "soh_pct" in _bm_df.columns:
                        _peer_last = _bm_df["soh_pct"].dropna()
                        if len(_peer_last):
                            _peer_sohs.append(float(_peer_last.iloc[-1]))
                if _peer_sohs:
                    import numpy as _np_bm
                    _pct_rank = int((_np_bm.array(_peer_sohs) < current_soh).sum() / len(_peer_sohs) * 100)
                    _bm_med   = float(_np_bm.median(_peer_sohs))
                    _bm_col   = "#48bb78" if _pct_rank >= 60 else "#d69e2e" if _pct_rank >= 30 else "#e53e3e"
                    _bm_word  = "better than" if _pct_rank >= 60 else "in line with" if _pct_rank >= 30 else "below"
                    st.markdown(
                        f"<div style='font-size:12px;color:#8896a8;margin:0 0 16px;"
                        f"padding:7px 14px;background:#111827;border-radius:6px;"
                        f"border-left:3px solid {_bm_col}'>"
                        f"Fleet benchmark: this cell's SOH is "
                        f"<strong style='color:{_bm_col}'>{_bm_word} {_pct_rank}% of fleet peers</strong> "
                        f"(fleet median {_bm_med:.1f}%, n={len(_peer_sohs)} cells)."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

        # ── C4: Pin baseline ──────────────────────────────────────────────────
        _pinned = st.session_state.get("pinned_cell")
        _pin_col1, _pin_col2 = st.columns([8, 2])
        with _pin_col2:
            if _pinned == cell_id:
                if st.button("📌 Unpin baseline", key=f"unpin_{cell_id}", use_container_width=True):
                    import db as _db
                    st.session_state["pinned_cell"] = None
                    _db.set_setting(st.session_state["auth_org_id"], "pinned_cell", None)
                    st.rerun()
            else:
                if st.button("📌 Pin as baseline", key=f"pin_{cell_id}", use_container_width=True,
                             help="Pin this cell as a comparison baseline — all other cells show a delta rail."):
                    import db as _db
                    st.session_state["pinned_cell"] = cell_id
                    _db.set_setting(st.session_state["auth_org_id"], "pinned_cell", cell_id)
                    st.rerun()
        if _pinned and _pinned != cell_id:
            st.markdown(
                f"<div style='font-size:11px;color:#8896a8;margin:-6px 0 0;padding:5px 12px;"
                f"background:#111827;border-radius:6px;border-left:3px solid #4a5568'>"
                f"Baseline: <strong style='color:#e2e8f0'>{_pinned}</strong> — navigate to that cell to compare."
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── D6: Model confidence history ─────────────────────────────────────────
    if "soh_pct" in df.columns and "soh_pred" in df.columns:
        _conf_df = df[["cycle_number", "soh_pct", "soh_pred"]].dropna()
        if len(_conf_df) >= 20:
            with st.expander("Prediction confidence history", expanded=False):
                # Rolling absolute error as proxy for model confidence at each cycle window
                import numpy as _np_d6
                _err = (_conf_df["soh_pct"] - _conf_df["soh_pred"]).abs()
                _win = min(20, max(5, len(_conf_df) // 8))
                _roll_err = _err.rolling(_win, min_periods=3).mean()
                _cycles_d6 = _conf_df["cycle_number"].tolist()
                _err_vals  = _roll_err.tolist()
                _min_err   = float(_np_d6.nanmin(_err_vals)) if _err_vals else 0
                _max_err   = float(_np_d6.nanmax(_err_vals)) if _err_vals else 1
                _trend_dir = "improving" if _err_vals[-1] < _err_vals[0] else (
                    "stable" if abs(_err_vals[-1] - _err_vals[0]) < 0.5 else "diverging")
                _converged = _err_vals[-1] < 1.5 if _err_vals else False
                _interp = (
                    "Model has converged — error is low and stable. Confidence is high."
                    if _converged and _trend_dir in ("stable", "improving") else
                    "Model is still calibrating — wait for more cycles for tighter estimates."
                    if _trend_dir == "diverging" else
                    "Model confidence is improving as more data accumulates."
                )
                _fig_d6 = go.Figure()
                _fig_d6.add_trace(go.Scatter(
                    x=_cycles_d6, y=_err_vals,
                    mode="lines",
                    line=dict(color="#63b3ed", width=2),
                    name="Rolling MAE (SOH %)",
                    fill="tozeroy", fillcolor="rgba(99,179,237,0.08)",
                    hovertemplate="Cycle %{x}<br>Error: %{y:.2f}%<extra></extra>",
                ))
                _fig_d6.add_hline(y=1.5, line_dash="dot", line_color="#48bb78", line_width=1,
                                  annotation_text="Reliable", annotation_position="top right",
                                  annotation_font_color="#48bb78", annotation_font_size=10)
                _fig_d6.update_layout(
                    **base_layout(
                        height=180,
                        showlegend=False,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="Abs error (SOH %)",
                                   zeroline=False, range=[0, max(_max_err * 1.2, 3)]),
                    ),
                )
                st.plotly_chart(_fig_d6, use_container_width=True)
                st.caption(
                    f"Rolling {_win}-cycle mean absolute error between predicted and measured SOH. "
                    f"Trend: {_trend_dir}. {_interp}"
                )

    # ── 3 primary secondary metrics: RUL · Fade Rate · Resistance ────────────
    _fade_30 = float(latest["fade_rate_30cy"]) if "fade_rate_30cy" in latest.index else None
    _res_ohm = float(latest["resistance_ohm"]) if "resistance_ohm" in latest.index else None
    fade_display = f"{_fade_30 * 1000:.2f} mAh/cy" if _fade_30 is not None else "—"
    fade_sub     = "30-cycle rolling window" if _fade_30 is not None else "not available"
    res_display  = f"{_res_ohm * 1000:.1f} mΩ" if _res_ohm is not None else "—"
    res_sub      = "DC internal resistance" if _res_ohm is not None else "not available"
    months_chip_html = (
        f"<div class='metric-chip-sub' style='margin-top:2px'>"
        f"~{months_remaining:.0f} months · {rate_note}</div>"
        if months_remaining is not None else ""
    )

    _md_html(
        f"""<div class="metric-row" role="list" aria-label="Key performance metrics">
          <div class="metric-chip" role="listitem">
            <div class="metric-chip-label">Est. Remaining Life</div>
            <div class="metric-chip-value" aria-label="Estimated remaining life: {rul_display} {rul_sub}">{rul_display}</div>
            <div class="metric-chip-sub">{rul_sub}</div>
            {months_chip_html}
          </div>
          <div class="metric-chip" role="listitem">
            <div class="metric-chip-label">Fade Rate</div>
            <div class="metric-chip-value" style="font-size:20px" aria-label="Fade rate: {fade_display}">{fade_display}</div>
            <div class="metric-chip-sub">{fade_sub}</div>
          </div>
          <div class="metric-chip" role="listitem">
            <div class="metric-chip-label">Internal Resistance</div>
            <div class="metric-chip-value" style="font-size:20px" aria-label="Internal resistance: {res_display}">{res_display}</div>
            <div class="metric-chip-sub">{res_sub}</div>
          </div>
        </div>"""
    )

    # ── Cell Details expander: secondary metrics ──────────────────────────────
    sop_display = f"{sop_pct:.0f}%" if sop_pct is not None else "—"
    _cum_kwh    = float(latest["cumulative_kwh"]) if "cumulative_kwh" in latest.index else None
    kwh_display = f"{_cum_kwh:.2f} kWh" if _cum_kwh is not None else "—"
    _eq_cy      = float(latest["equivalent_cycles"]) if "equivalent_cycles" in latest.index and not pd.isna(latest["equivalent_cycles"]) else None
    eq_cy_display = f"{_eq_cy:,.0f}" if _eq_cy is not None else "—"
    eq_cy_sub   = "stress-normalized (CATL/BYD metric)" if _eq_cy is not None else "not available"
    _ce         = float(latest["coulombic_efficiency"]) if "coulombic_efficiency" in latest.index else None
    ce_display  = f"{_ce*100:.3f}%" if _ce is not None else "—"

    with st.expander("Cell Details", expanded=False):
        _md_html(
            f"""<div class="metric-row" role="list" aria-label="Cell detail metrics">
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">State of Health</div>
                <div class="metric-chip-value">{current_soh:.1f}%</div>
                <div class="metric-chip-sub">vs 100% at cycle 1</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Cycles Completed</div>
                <div class="metric-chip-value">{current_cycle:,}</div>
                <div class="metric-chip-sub">charge / discharge</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Capacity Lost</div>
                <div class="metric-chip-value">{total_fade*1000:.0f} mAh</div>
                <div class="metric-chip-sub">since commissioning</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Peak Power (SoP)</div>
                <div class="metric-chip-value" style="font-size:20px">{sop_display}</div>
                <div class="metric-chip-sub">of initial capability</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Energy Delivered</div>
                <div class="metric-chip-value" style="font-size:18px">{kwh_display}</div>
                <div class="metric-chip-sub">cumulative throughput</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Equiv. Cycles</div>
                <div class="metric-chip-value" style="font-size:18px">{eq_cy_display}</div>
                <div class="metric-chip-sub">{eq_cy_sub}</div>
              </div>
              <div class="metric-chip" role="listitem">
                <div class="metric-chip-label">Coulombic Eff.</div>
                <div class="metric-chip-value" style="font-size:18px">{ce_display}</div>
                <div class="metric-chip-sub">last cycle Q_d / Q_c</div>
              </div>
            </div>"""
        )

    # ── Calendar Age Analysis ─────────────────────────────────────────────────
    if "cumulative_days" in df.columns:
        with st.expander("📅 Calendar Age Analysis", expanded=False):
            try:
                cal_last = float(df["cumulative_days"].iloc[-1])
                avg_rest = float(df["days_between_cycles"].mean()) if "days_between_cycles" in df.columns else 1.0
                total_fade_ah = float(df["capacity_ah"].iloc[0]) - float(df["capacity_ah"].iloc[-1])
                calendar_fade_pct = min(35, max(5, 100 * (cal_last * 0.00003 * 0.5) / max(total_fade_ah, 1e-6)))

                cal_c1, cal_c2, cal_c3 = st.columns(3)
                with cal_c1:
                    st.metric("Total Calendar Time", f"{cal_last:.0f} days", help="since first cycle")
                    st.markdown("<div style='font-size:11px;color:#8896a8;margin-top:-8px'>since first cycle</div>", unsafe_allow_html=True)
                with cal_c2:
                    st.metric("Avg Rest Between Cycles", f"{avg_rest:.1f} days", help="mean rest period")
                    st.markdown("<div style='font-size:11px;color:#8896a8;margin-top:-8px'>mean rest period</div>", unsafe_allow_html=True)
                with cal_c3:
                    st.metric("Calendar vs Cycle Fade", f"~{calendar_fade_pct:.0f}% calendar", help="remainder from cycling")
                    st.markdown("<div style='font-size:11px;color:#8896a8;margin-top:-8px'>remainder from cycling</div>", unsafe_allow_html=True)

                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=_rdf["cycle_number"], y=_rdf["cumulative_days"],
                    name="Actual calendar time", line=dict(color="#63b3ed", width=2),
                    hovertemplate="Cycle %{x}: %{y:.1f} days<extra>Calendar</extra>",
                ))
                max_cy = int(df["cycle_number"].max())
                fig_cal.add_trace(go.Scatter(
                    x=[0, max_cy], y=[0, max_cy],
                    name="1 cycle/day baseline", line=dict(color="#4a5568", width=1.5, dash="dash"),
                    hoverinfo="skip",
                ))
                fig_cal.update_layout(
                    **base_layout(
                        height=260, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="Days elapsed", zeroline=False),
                    ),
                )
                fig_cal.update_layout(title=dict(text="Calendar Time vs Cycle Count", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(fig_cal, use_container_width=True)
            except Exception as _e:
                st.info(f"Calendar age analysis unavailable: {_e}")

    # ── 📈 Full trend & projections (U1 density reduction) ───────────────────
    with st.expander("📈 Full trend & projections", expanded=False):
        df_train = _rdf[_rdf["cycle_number"] <= split_cycle]
        df_test  = _rdf[_rdf["cycle_number"] >  split_cycle]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_rdf["cycle_number"], y=_rdf["soh_pct"],
            name="Actual SOH", line=dict(color="#3a4a5e", width=1), mode="lines",
            hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Actual</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=_rdf["cycle_number"], y=_rdf["soh_rolling_avg"],
            name="10-cycle avg", line=dict(color="#63b3ed", width=2), mode="lines",
            hovertemplate="Cycle %{x}: %{y:.1f}%<extra>10-cy avg</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_test["cycle_number"], y=df_test["soh_pred"],
            name="Model (test)", line=dict(color="#48bb78", width=2, dash="dot"), mode="lines",
            hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Model</extra>",
        ))
        # F2: ±σ confidence band derived from leave-cell-out residuals
        if len(df_test) >= 5 and "soh_pred" in df_test.columns:
            import numpy as _np_f2
            _f2_valid = df_test[["cycle_number", "soh_pct", "soh_pred"]].dropna()
            if len(_f2_valid) >= 3:
                _sigma_f2 = float((_f2_valid["soh_pct"] - _f2_valid["soh_pred"]).std())
                _sigma_f2 = max(min(_sigma_f2, 8.0), 0.3)
                _cx_f2  = _f2_valid["cycle_number"].tolist()
                _cu_f2  = (_f2_valid["soh_pred"] + _sigma_f2).clip(upper=102.0).tolist()
                _cl_f2  = (_f2_valid["soh_pred"] - _sigma_f2).clip(lower=60.0).tolist()
                fig.add_trace(go.Scatter(
                    x=_cx_f2 + _cx_f2[::-1],
                    y=_cu_f2 + _cl_f2[::-1],
                    fill="toself",
                    fillcolor="rgba(72,187,120,0.10)",
                    line=dict(width=0),
                    name=f"±{_sigma_f2:.1f}% model uncertainty (σ)",
                    hoverinfo="skip",
                ))
        fig.add_vline(
            x=split_cycle, line_dash="dot", line_color="#4a5568", line_width=1,
            annotation_text=f"Train → Test (cy {split_cycle})",
            annotation_position="top left",
            annotation_font_color="#4a5568", annotation_font_size=11,
        )
        # ── Scenario projections ──
        import numpy as np
        last_cycle = df["cycle_number"].iloc[-1]
        last_soh   = df["soh_pct"].iloc[-1]
        eol_line   = float(st.session_state.get("eol_threshold_pct", 80.0))
        nominal_rate    = float(df["fade_rate_50cy"].iloc[-1]) * 100   # % SOH / cycle
        optimistic_rate = nominal_rate * 0.6
        pessimistic_rate = nominal_rate * 1.5

        def _proj(rate):
            if rate > 1e-9:
                n_steps = min(500, int((last_soh - eol_line) / rate) + 10)
            else:
                n_steps = 200
            n_steps = max(n_steps, 2)
            proj_cycles = np.arange(last_cycle, last_cycle + n_steps)
            proj_soh    = last_soh - rate * np.arange(n_steps)
            proj_soh    = np.clip(proj_soh, 60.0, None)
            return proj_cycles.tolist(), proj_soh.tolist()

        for rate, name, color in [
            (nominal_rate,     "Nominal projection",          "#63b3ed"),
            (optimistic_rate,  "Optimistic (−40% stress)",    "#48bb78"),
            (pessimistic_rate, "Pessimistic (+50% stress)",   "#fc8181"),
        ]:
            px, py = _proj(rate)
            fig.add_trace(go.Scatter(
                x=px, y=py, name=name, mode="lines",
                line=dict(dash="dash", width=1.5, color=color),
                opacity=0.7,
                hovertemplate="Cycle %{x}: %{y:.1f}%<extra>" + name + "</extra>",
            ))

        fig.add_hline(
            y=eol_line, line_dash="dot", line_color="#e53e3e", line_width=1,
            annotation_text=f"EOL threshold ({eol_line:.0f}%)",
            annotation_position="bottom right",
        )
        y_min = max(df["soh_pct"].min() - 2, 60)
        fig.update_layout(
            **base_layout(
                height=340, legend=LEGEND_H,
                xaxis=dict(title="Cycle", zeroline=False),
                yaxis=dict(title="SOH %",
                           zeroline=False, range=[y_min, 101]),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        _ov_prov = _cell_provenance(cell_id)
        _ov_kind = ChemistryProfile.for_cell(cell_id).source_kind
        if _ov_kind == "severson":
            _ov_caption = "Capacity data: ● MEASURED (Severson 2019, Nature Energy). Projections: ◐ SIMULATED (linear extrapolation)."
        elif _ov_kind == "oxford":
            _ov_caption = "Capacity data: ● MEASURED (Raj et al. 2020, Oxford dataset). Projections: ◐ SIMULATED (linear extrapolation)."
        elif _ov_prov == "measured":
            _ov_caption = "Capacity data: ● MEASURED (NASA PCoE). Projections: ◐ SIMULATED (linear extrapolation)."
        else:
            _ov_caption = "All data: ○ SYNTHETIC — no physical measurements underlie any value."
        st.caption(
            "Projections assume constant fade rate. Optimistic = 40% stress reduction. Pessimistic = 50% stress increase. "
            + _ov_caption
        )
