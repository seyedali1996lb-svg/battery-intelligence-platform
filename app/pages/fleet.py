import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from utils import base_layout, LEGEND_H, PLOTLY_CONFIG, _md_html, soh_status, NASA_CELL_IDS
from data_loader import CELL_STRESS_PROFILES
from knee_detection import detect_knee


def page_fleet(featured_dfs: dict, bundles: dict):
    st.markdown("# Fleet")

    # ── Build fleet summary row per cell ──
    # Bundle and per-cell reliability lookup is source-aware; uploaded cells use
    # the "upload" bundle, NASA cells the "nasa" bundle, synthetic the "synth" bundle.
    rows = []
    _synth_ids = set(CELL_STRESS_PROFILES.keys())

    def _bundle_for_cell(cid: str) -> dict | None:
        if cid in NASA_CELL_IDS:
            return bundles.get("nasa")
        if cid in _synth_ids:
            return bundles.get("synth")
        return bundles.get("upload")      # uploaded cell

    for cell_id, df in featured_dfs.items():
        bndl      = _bundle_for_cell(cell_id)
        if bndl is None:
            continue
        per_cell  = bndl["metrics"].get("per_cell_rul_reliable", {})
        rul_ok    = per_cell.get(cell_id, bndl["metrics"].get("rul_reliable", False))
        is_nasa   = cell_id in NASA_CELL_IDS
        latest    = df.iloc[-1]
        soh       = latest["soh_pct"]
        cycle     = int(latest["cycle_number"])
        fade_30   = latest.get("fade_rate_30cy", float("nan")) * 1000  # mSOH/cy
        rul       = latest["rul_pred"] if rul_ok else None
        eol_row   = df[df["is_eol"]]
        eol_at    = int(eol_row["cycle_number"].iloc[0]) if len(eol_row) else None
        cycles_to_eol = max(0, eol_at - cycle) if eol_at else None

        is_upload = not is_nasa and cell_id not in _synth_ids
        status_label, _ = soh_status(soh)

        # Knee-point detection per cell
        knee_result = detect_knee(df["soh_pct"], df["cycle_number"])

        # Degradation trend: compare current 30-cy fade rate vs 30 cycles earlier
        trend = "Stable"
        if "fade_rate_30cy" in df.columns and len(df) >= 31:
            fade_now  = df["fade_rate_30cy"].iloc[-1]
            fade_prev = df["fade_rate_30cy"].iloc[-31]
            delta_pct = (fade_now - fade_prev) / (abs(fade_prev) + 1e-9) * 100
            if delta_pct > 20:
                trend = "Accelerating"
            elif delta_pct < -20:
                trend = "Decelerating"

        rows.append({
            "cell_id":      cell_id,
            "source":       "NASA" if is_nasa else ("Uploaded" if is_upload else "Synthetic"),
            "soh":          soh,
            "status":       status_label,
            "cycle":        cycle,
            "fade_30":      fade_30,
            "rul":          rul,
            "rul_ok":       rul_ok,
            "eol_at":       eol_at,
            "cycles_to_eol": cycles_to_eol,
            "trend":        trend,
            "knee":         knee_result,
        })

    # Sort: worst SOH first (most urgent)
    rows.sort(key=lambda r: r["soh"])

    # ── Header metrics ──
    n_eol       = sum(1 for r in rows if r["status"] == "End of Life")
    n_degrading = sum(1 for r in rows if r["status"] == "Degrading")
    n_healthy   = sum(1 for r in rows if r["status"] == "Healthy")
    worst_soh   = rows[0]["soh"]
    best_soh    = rows[-1]["soh"]
    n_nasa      = sum(1 for r in rows if r["source"] == "NASA")
    n_synth     = sum(1 for r in rows if r["source"] == "Synthetic")
    n_upload    = sum(1 for r in rows if r["source"] == "Uploaded")
    src_parts   = []
    if n_synth:  src_parts.append(f"{n_synth} synthetic")
    if n_nasa:   src_parts.append(f"{n_nasa} NASA real")
    if n_upload: src_parts.append(f"{n_upload} uploaded")
    src_sub = " · ".join(src_parts) or "—"

    _md_html(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-label">Total Cells</div>
                <div class="metric-chip-value">{len(rows)}</div>
                <div class="metric-chip-sub">{src_sub}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">End of Life</div>
                <div class="metric-chip-value" style="color:#fc8181">{n_eol}</div>
                <div class="metric-chip-sub">below 80% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Degrading</div>
                <div class="metric-chip-value" style="color:#f6e05e">{n_degrading}</div>
                <div class="metric-chip-sub">80–90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Healthy</div>
                <div class="metric-chip-value" style="color:#48bb78">{n_healthy}</div>
                <div class="metric-chip-sub">above 90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Fleet SOH Range</div>
                <div class="metric-chip-value" style="font-size:20px">{worst_soh:.0f}–{best_soh:.0f}%</div>
                <div class="metric-chip-sub">worst to best</div>
            </div>
        </div>
        """
    )

    # ── Ranking table ──
    st.markdown("<div class='section-header'>Health Ranking — Worst First</div>", unsafe_allow_html=True)

    STATUS_COLOUR = {"Healthy": "#48bb78", "Degrading": "#f6e05e", "End of Life": "#fc8181"}
    SOURCE_STYLE  = {
        "NASA":      "background:rgba(104,211,145,0.12);color:#48bb78;border:1px solid rgba(104,211,145,0.25)",
        "Synthetic": "background:rgba(74,85,104,0.3);color:#718096;border:1px solid #2d3748",
        "Uploaded":  "background:rgba(99,179,237,0.12);color:#63b3ed;border:1px solid rgba(99,179,237,0.25)",
    }

    table_rows_html = ""
    for rank, r in enumerate(rows, 1):
        sc = STATUS_COLOUR[r["status"]]
        ss = SOURCE_STYLE[r["source"]]
        soh_bar = int(max(0, min(100, r["soh"])))
        bar_colour = sc

        rul_cell = (
            f"{r['rul']:.0f} cy" if (r["rul"] is not None and r["rul_ok"])
            else "<span style='color:#4a5568'>—</span>"
        )
        eol_cell = (
            f"<span style='color:#fc8181'>Reached at {r['eol_at']}</span>"
            if r["eol_at"] and r["cycles_to_eol"] == 0
            else (f"{r['cycles_to_eol']} cy" if r["cycles_to_eol"] is not None else "—")
        )

        TREND_STYLE = {
            "Accelerating": ("⚡", "#fc8181"),
            "Stable":       ("→",  "#a0aec0"),
            "Decelerating": ("↘",  "#48bb78"),
        }
        trend_icon, trend_colour = TREND_STYLE.get(r["trend"], ("→", "#a0aec0"))

        table_rows_html += f"""
        <tr style="border-bottom:1px solid #1a202c">
            <td style="padding:14px 12px;color:#4a5568;font-size:13px">{rank}</td>
            <td style="padding:14px 12px">
                <span style="font-weight:600;color:#e2e8f0;font-size:14px">{r['cell_id']}</span>
                <span style="margin-left:8px;font-size:10px;padding:2px 6px;border-radius:8px;{ss}">{r['source']}</span>
            </td>
            <td style="padding:14px 12px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="color:{sc};font-weight:700;font-size:15px;min-width:46px">{r['soh']:.1f}%</span>
                    <div style="flex:1;background:#1a202c;border-radius:3px;height:6px;min-width:80px">
                        <div style="background:{bar_colour};width:{soh_bar}%;height:6px;border-radius:3px"></div>
                    </div>
                </div>
            </td>
            <td style="padding:14px 12px">
                <span style="font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px;
                             background:{sc}22;color:{sc};border:1px solid {sc}44">{r['status']}</span>
            </td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['cycle']:,}</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['fade_30']:.2f} mSOH/cy</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{rul_cell}</td>
            <td style="padding:14px 12px;font-size:13px">{eol_cell}</td>
            <td style="padding:14px 12px;font-size:12px;color:{trend_colour}">{trend_icon} {r['trend']}</td>
            <td style="padding:14px 12px;font-size:12px;color:{'#9f7aea' if r['knee']['detected'] else '#2d3748'}">
                {'⬡ cy ' + str(r['knee']['cycle']) if r['knee']['detected'] else '—'}
            </td>
        </tr>
        """

    _md_html(
        f"""
        <table style="width:100%;border-collapse:collapse;font-family:sans-serif">
            <thead>
                <tr style="border-bottom:2px solid #2d3748">
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">#</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cell</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">SOH</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Status</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cycles</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Fade Rate</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Est. RUL</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">EOL Proximity</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Trend</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Knee</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
        """
    )

    # ── CSV Export ──────────────────────────────────────────────────────────
    _csv_rows = []
    for r in rows:
        _csv_rows.append({
            "cell_id": r["cell_id"], "source": r["source"],
            "soh_pct": round(r["soh"], 2), "status": r["status"],
            "cycle": r["cycle"], "fade_rate_mSOH_cy": round(r["fade_30"], 3),
            "rul_cycles": round(r["rul"], 0) if r["rul"] is not None else "",
            "eol_at_cycle": r["eol_at"] or "",
            "cycles_to_eol": r["cycles_to_eol"] or "",
            "trend": r["trend"],
            "knee_cycle": r["knee"]["cycle"] if r["knee"]["detected"] else "",
        })
    _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export fleet table as CSV",
        data=_csv_bytes,
        file_name="fleet_health_summary.csv",
        mime="text/csv",
    )

    # ── SOH distribution chart ──
    st.markdown("<div class='section-header'>SOH Distribution Across Fleet</div>", unsafe_allow_html=True)

    sorted_ids  = [r["cell_id"] for r in rows]
    sorted_sohs = [r["soh"] for r in rows]
    bar_colours = [STATUS_COLOUR[r["status"]] for r in rows]

    fig = go.Figure(go.Bar(
        x=sorted_ids, y=sorted_sohs,
        marker_color=bar_colours,
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#fc8181", line_width=1,
                  annotation_text="EOL (80%)", annotation_position="top right",
                  annotation_font_color="#fc8181", annotation_font_size=11)
    fig.add_hline(y=90, line_dash="dot", line_color="#f6e05e", line_width=1,
                  annotation_text="Degrading (90%)", annotation_position="top right",
                  annotation_font_color="#f6e05e", annotation_font_size=11)
    fig.update_layout(
        height=280,
        **base_layout(
            xaxis=dict(title="Cell", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[50, 102]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Risk matrix: SOH vs RUL ──
    st.markdown("<div class='section-header'>Risk Matrix — SOH vs Remaining Life</div>", unsafe_allow_html=True)

    cal_rows   = [r for r in rows if r["rul_ok"] and r["rul"] is not None]
    uncal_rows = [r for r in rows if not (r["rul_ok"] and r["rul"] is not None)]

    if cal_rows:
        import numpy as _np
        rul_vals    = [r["rul"] for r in cal_rows]
        rul_med     = float(_np.median(rul_vals))
        soh_thresh  = 80.0   # EOL threshold (consistent with rest of platform)

        def _quadrant(soh_v, rul_v):
            h_soh = soh_v >= soh_thresh
            h_rul = rul_v >= rul_med
            if h_soh and h_rul:     return "Continue", "#48bb78"
            if h_soh and not h_rul: return "Watch",    "#d69e2e"
            if not h_soh and h_rul: return "Act",      "#f6ad55"
            return "Critical", "#fc8181"

        fig_risk = go.Figure()

        # Calibrated cells — colored by quadrant
        fig_risk.add_trace(go.Scatter(
            x=[r["soh"] for r in cal_rows],
            y=[r["rul"] for r in cal_rows],
            mode="markers+text",
            text=[r["cell_id"] for r in cal_rows],
            textposition="top center",
            textfont=dict(size=11, color="#a0aec0"),
            marker=dict(
                size=16,
                color=[_quadrant(r["soh"], r["rul"])[1] for r in cal_rows],
                line=dict(color="#1a202c", width=1),
            ),
            customdata=[[r["cell_id"], r["soh"], r["rul"], _quadrant(r["soh"], r["rul"])[0]]
                        for r in cal_rows],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "SOH: %{customdata[1]:.1f}%<br>"
                "RUL: %{customdata[2]:.0f} cycles<br>"
                "Quadrant: %{customdata[3]}<extra></extra>"
            ),
            name="Calibrated RUL",
            showlegend=False,
        ))

        # Uncalibrated cells — plotted at y=0 with X markers
        if uncal_rows:
            fig_risk.add_trace(go.Scatter(
                x=[r["soh"] for r in uncal_rows],
                y=[0] * len(uncal_rows),
                mode="markers+text",
                text=[r["cell_id"] for r in uncal_rows],
                textposition="top center",
                textfont=dict(size=11, color="#4a5568"),
                marker=dict(size=14, color="#4a5568", symbol="x", line=dict(color="#4a5568", width=2)),
                customdata=[[r["cell_id"], r["soh"]] for r in uncal_rows],
                hovertemplate="<b>%{customdata[0]}</b><br>SOH: %{customdata[1]:.1f}%<br>RUL: not calibrated<extra></extra>",
                name="RUL not calibrated",
                showlegend=bool(uncal_rows),
            ))

        # Quadrant dividers
        fig_risk.add_vline(x=soh_thresh, line_dash="dash", line_color="#fc8181", line_width=1)
        fig_risk.add_hline(y=rul_med,    line_dash="dash", line_color="#4a5568",  line_width=1)

        # Quadrant labels
        x_lo, x_hi = 55, max(r["soh"] for r in cal_rows) + 5
        y_hi        = max(rul_vals) * 1.05
        for (sx, sy, label, c) in [
            (soh_thresh - 1, rul_med + y_hi * 0.02, "ACT",      "#f6ad55"),
            (soh_thresh + 1, rul_med + y_hi * 0.02, "CONTINUE", "#48bb78"),
            (soh_thresh - 1, y_hi * 0.04,           "CRITICAL", "#fc8181"),
            (soh_thresh + 1, y_hi * 0.04,           "WATCH",    "#d69e2e"),
        ]:
            fig_risk.add_annotation(
                x=sx, y=sy, text=label, showarrow=False,
                font=dict(size=9, color=c, family="monospace"),
                xanchor="right" if sx < soh_thresh else "left",
            )

        fig_risk.update_layout(
            height=320,
            **base_layout(
                xaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                           zeroline=False, range=[x_lo, x_hi]),
                yaxis=dict(title="Est. RUL (cycles)", gridcolor="#232d3b", linecolor="#2d3748",
                           zeroline=False, range=[-y_hi * 0.08, y_hi]),
            ),
        )
        fig_risk.update_layout(legend=dict(font=dict(size=11, color="#718096")))
        _risk_event = st.plotly_chart(
            fig_risk, use_container_width=True,
            on_select="rerun", key="fleet_risk_chart",
            selection_mode="points",
        )
        # Navigate to Health page for the clicked cell
        _risk_sel = (_risk_event or {}).get("selection", {})
        _risk_pts  = _risk_sel.get("points", []) if _risk_sel else []
        if _risk_pts:
            _clicked_cell = _risk_pts[0].get("customdata", [None])[0]
            if _clicked_cell:
                st.session_state["selected_cell"] = _clicked_cell
                st.session_state["page"] = "health"
                st.rerun()

        if uncal_rows:
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;margin-top:-8px'>"
                f"✕ = RUL not calibrated (LCO fold R² below 0.30) — plotted at y=0. "
                f"Quadrant split at SOH 80% and median RUL of calibrated cells ({rul_med:.0f} cy).</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Risk matrix requires at least one cell with a calibrated RUL estimate.")

    st.caption("Click any cell in the risk matrix to open its Health page.")

    # ── Cell-to-Cell Spread Trending ────────────────────────────────────────
    st.markdown("<div class='section-header'>📉 Fleet Spread Over Time — σ(SOH)</div>", unsafe_allow_html=True)
    _md_html("""<div style="font-size:13px;color:#718096;margin-bottom:14px;line-height:1.6">A <strong style="color:#e2e8f0">rising σ(SOH)</strong> means one cell is falling behind the fleet — the earliest warning of a cell that will force a pack-level service event. When spread exceeds ~3%, investigation is warranted.</div>""")
    try:
        import numpy as _np_sp
        _cy_min = int(max(df.iloc[0]["cycle_number"] if len(df) > 0 else 1 for df in featured_dfs.values()))
        _cy_max = int(min(df.iloc[-1]["cycle_number"] for df in featured_dfs.values()))
        if _cy_max > _cy_min + 20 and len(featured_dfs) >= 2:
            _check_cycles = list(range(_cy_min, _cy_max + 1, max(1, (_cy_max - _cy_min) // 80)))
            _sigma_data = []
            for _cy in _check_cycles:
                _sohs = [
                    float(_np_sp.interp(_cy, fdf["cycle_number"].values, fdf["soh_pct"].values))
                    for fdf in featured_dfs.values()
                    if _cy >= fdf["cycle_number"].min() and _cy <= fdf["cycle_number"].max()
                ]
                if len(_sohs) >= 2:
                    _sigma_data.append({"cycle": _cy, "sigma": float(_np_sp.std(_sohs)), "n": len(_sohs)})
            if _sigma_data:
                _sd_df = pd.DataFrame(_sigma_data)
                _sigma_smoothed = pd.Series(_sd_df["sigma"]).rolling(5, min_periods=1).mean()
                _fig_spread = go.Figure()
                _fig_spread.add_trace(go.Scatter(
                    x=_sd_df["cycle"].tolist(), y=_sd_df["sigma"].tolist(),
                    name="σ(SOH) raw", mode="lines",
                    line=dict(color="#4a5568", width=1),
                    hovertemplate="Cycle %{x}: σ=%{y:.2f}%<extra>Raw spread</extra>",
                ))
                _fig_spread.add_trace(go.Scatter(
                    x=_sd_df["cycle"].tolist(), y=_sigma_smoothed.tolist(),
                    name="σ(SOH) smoothed", mode="lines",
                    line=dict(color="#63b3ed", width=2.5),
                    hovertemplate="Cycle %{x}: σ=%{y:.2f}% (smoothed)<extra></extra>",
                ))
                _fig_spread.add_hline(y=3.0, line=dict(color="#f6ad55", width=1, dash="dot"),
                                      annotation_text="3% — investigation threshold",
                                      annotation=dict(font=dict(size=9, color="#f6ad55")))
                _fig_spread.add_hline(y=5.0, line=dict(color="#fc8181", width=1, dash="dot"),
                                      annotation_text="5% — service threshold",
                                      annotation=dict(font=dict(size=9, color="#fc8181")))
                _fig_spread.update_layout(
                    **base_layout(
                        height=280, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                        yaxis=dict(title="σ(SOH) %", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                    ),
                )
                _fig_spread.update_layout(title=dict(text=f"Fleet SOH Spread ({len(featured_dfs)} cells)", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(_fig_spread, use_container_width=True)
                _peak_sigma = float(_sd_df["sigma"].max())
                _trend_dir = "rising" if _sigma_smoothed.iloc[-1] > _sigma_smoothed.iloc[max(0, len(_sigma_smoothed)//2)] else "falling or stable"
                st.caption(f"Peak spread: {_peak_sigma:.2f}% SOH — current trend: {_trend_dir}.")
        else:
            st.info("Fleet spread analysis requires ≥ 2 cells with overlapping cycle ranges.")
    except Exception as _sp_e:
        st.info(f"Spread trend unavailable: {_sp_e}")

    # ── Second-life screening ──
    st.markdown("<div class='section-header'>Second-Life Readiness Screening</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#718096;margin-bottom:20px;line-height:1.6'>"
        "Conventional second-life assessment window: <strong style='color:#e2e8f0'>SOH 70–85%</strong>. "
        "Cells above 85% are still in primary life. Below 70% is below most application floors. "
        "Click a cell in the sidebar to open the Consequences page for detailed economics.</div>",
        unsafe_allow_html=True,
    )

    SL_BUCKETS = {
        "primary":    ("Primary Life",          "SOH > 85%",    "#4a5568", "#1a202c"),
        "candidate":  ("Second-Life Candidate", "SOH 70–85%",   "#48bb78", "#1a2e22"),
        "below_floor":("Below Floor",           "SOH < 70%",    "#fc8181", "#2d0f0f"),
    }

    def _sl_bucket(r_soh):
        if r_soh > 85.0:  return "primary"
        if r_soh >= 70.0: return "candidate"
        return "below_floor"

    bucketed = {"primary": [], "candidate": [], "below_floor": []}
    for r in rows:
        bucketed[_sl_bucket(r["soh"])].append(r)

    sl_cols = st.columns(3)
    for col, (bkey, (blabel, brange, bfg, bbg)) in zip(sl_cols, SL_BUCKETS.items()):
        cells_in_bucket = bucketed[bkey]
        count = len(cells_in_bucket)
        pills = "".join(
            f"<div style='display:inline-block;margin:4px;padding:5px 12px;"
            f"background:{bfg}22;border:1px solid {bfg}44;border-radius:20px;"
            f"font-size:12px;font-weight:600;color:{bfg}'>"
            f"{r['cell_id']} <span style='font-weight:400;color:{bfg}88'>{r['soh']:.0f}%</span>"
            f"</div>"
            for r in cells_in_bucket
        ) if cells_in_bucket else (
            f"<div style='font-size:12px;color:#4a5568;font-style:italic;padding:8px 0'>None</div>"
        )
        with col:
            _md_html(
                f"""
                <div style="background:{bbg};border:1px solid {bfg}33;border-radius:10px;padding:18px;min-height:120px">
                    <div style="font-size:10px;font-weight:700;color:{bfg};text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px">{blabel}</div>
                    <div style="font-size:12px;color:{bfg}88;margin-bottom:12px">{brange} · {count} cell{'s' if count != 1 else ''}</div>
                    <div style="line-height:2">{pills}</div>
                </div>
                """
            )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Virtual Pack Builder ──
    st.markdown("<div class='section-header'>Virtual Pack Builder</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#718096;margin-bottom:14px;line-height:1.6'>"
        "Select cells to model as a series/parallel pack. Pack health is constrained by "
        "the weakest cell — the cell with lowest SOH determines effective pack capacity, "
        "and the cell with fewest remaining cycles determines when the pack needs service."
        "</div>",
        unsafe_allow_html=True,
    )
    all_cell_ids = [r["cell_id"] for r in rows]
    pack_selection = st.multiselect(
        "Select cells for virtual pack",
        options=all_cell_ids,
        default=all_cell_ids[:min(4, len(all_cell_ids))],
        key="fleet_pack_cells",
        label_visibility="collapsed",
    )
    if pack_selection:
        import statistics as _stats
        pack_rows = [r for r in rows if r["cell_id"] in pack_selection]
        weakest = min(pack_rows, key=lambda r: r["soh"])
        strongest = max(pack_rows, key=lambda r: r["soh"])
        pack_soh = weakest["soh"]
        soh_vals = [r["soh"] for r in pack_rows]
        pack_spread = _stats.stdev(soh_vals) if len(soh_vals) > 1 else 0.0
        soh_range = strongest["soh"] - weakest["soh"]
        pack_rul_rows = [r for r in pack_rows if r["rul"] is not None and r["rul_ok"]]
        pack_rul = min(r["rul"] for r in pack_rul_rows) if pack_rul_rows else None
        pack_status_label, pack_status_c = soh_status(pack_soh)
        rul_display = f"{pack_rul:.0f} cy" if pack_rul is not None else "—"
        rul_note = "" if pack_rul_rows else "RUL not calibrated for any selected cell"
        n_uncal = len(pack_rows) - len(pack_rul_rows)
        # Spread health: <2% is excellent, 2-5% is watch, >5% is high imbalance
        spread_c = "#48bb78" if pack_spread < 2 else ("#f6ad55" if pack_spread < 5 else "#fc8181")
        spread_label = "Balanced" if pack_spread < 2 else ("Watch" if pack_spread < 5 else "Imbalanced")

        _md_html(
            f"""
            <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:10px;
                        padding:18px 22px;margin-bottom:12px">
                <div style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;
                            letter-spacing:0.07em;margin-bottom:10px">
                    Virtual Pack · {len(pack_selection)} cells selected</div>
                <div style="display:flex;gap:32px;flex-wrap:wrap">
                    <div>
                        <div style="font-size:11px;color:#4a5568">Pack SOH (weakest cell)</div>
                        <div style="font-size:28px;font-weight:700;color:{STATUS_COLOUR[pack_status_label]}">{pack_soh:.1f}%</div>
                        <div style="font-size:11px;color:#4a5568">{weakest['cell_id']} · {pack_status_label}</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#4a5568">Cell SOH Spread (σ)</div>
                        <div style="font-size:28px;font-weight:700;color:{spread_c}">{pack_spread:.1f}%</div>
                        <div style="font-size:11px;color:{spread_c}">{spread_label} · range {soh_range:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#4a5568">Pack RUL (shortest)</div>
                        <div style="font-size:28px;font-weight:700;color:#a0aec0">{rul_display}</div>
                        <div style="font-size:11px;color:#4a5568">{rul_note if rul_note else (f'{n_uncal} cell(s) excluded from RUL' if n_uncal else 'all cells calibrated')}</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#4a5568">Cells in pack</div>
                        <div style="font-size:28px;font-weight:700;color:#e2e8f0">{len(pack_selection)}</div>
                        <div style="font-size:11px;color:#4a5568">{', '.join(pack_selection)}</div>
                    </div>
                </div>
                <div style="font-size:11px;color:#4a5568;margin-top:10px;padding-top:10px;border-top:1px solid #2d3748">
                    High cell spread (σ&gt;5%) forces the BMS to restrict charge/discharge to protect the weakest cell,
                    reducing effective pack capacity below the weakest cell's SOH alone.
                </div>
            </div>
            """
        )
    else:
        st.markdown(
            "<div style='font-size:13px;color:#4a5568;padding:12px'>Select at least one cell above to build a virtual pack.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Enhanced Virtual Pack Builder ──
    st.subheader("🔋 Virtual Pack Builder")
    st.caption("Model series or parallel configurations and assess cell matching quality for pack assembly.")
    _pb_col1, _pb_col2 = st.columns(2)
    with _pb_col1:
        selected_pack_cells = st.multiselect(
            "Select cells for pack",
            options=list(featured_dfs.keys()),
            default=list(featured_dfs.keys())[:min(4, len(featured_dfs))],
            key="pack_cells",
        )
    with _pb_col2:
        pack_topology = st.radio("Configuration", ["Series", "Parallel"], horizontal=True, key="pack_topology")

    if len(selected_pack_cells) < 2:
        st.warning("Select at least 2 cells.")
    else:
        _fdfs = featured_dfs
        _soh_values = [float(_fdfs[c]["soh_pct"].iloc[-1]) for c in selected_pack_cells]
        _rul_values = [float(_fdfs[c]["rul_pred"].iloc[-1]) if "rul_pred" in _fdfs[c].columns else None for c in selected_pack_cells]
        _cap_values = [float(_fdfs[c]["capacity_ah"].iloc[-1]) for c in selected_pack_cells]
        _res_values = [float(_fdfs[c]["resistance_ohm"].iloc[-1]) for c in selected_pack_cells]

        if pack_topology == "Series":
            _pack_soh = min(_soh_values)
            _pack_rul = min(r for r in _rul_values if r is not None) if any(r is not None for r in _rul_values) else None
            _pack_capacity = min(_cap_values)
            _pack_resistance = sum(_res_values)
            _bottleneck_cell = selected_pack_cells[_soh_values.index(min(_soh_values))]
            _cap_label = f"{_pack_capacity * 1000:.0f} mAh"
        else:
            _cap_total = sum(_cap_values)
            _pack_soh = sum(s * c for s, c in zip(_soh_values, _cap_values)) / _cap_total
            _pack_rul = min(r for r in _rul_values if r is not None) if any(r is not None for r in _rul_values) else None
            _pack_capacity = _cap_total
            _res_safe = [r for r in _res_values if r > 0]
            _pack_resistance = 1 / sum(1 / r for r in _res_safe) if _res_safe else float("nan")
            _bottleneck_cell = selected_pack_cells[_soh_values.index(min(_soh_values))]
            _cap_label = f"{_pack_capacity * 1000:.0f} mAh total"

        _pm1, _pm2, _pm3, _pm4 = st.columns(4)
        _pm1.metric("Pack SOH", f"{_pack_soh:.1f}%")
        _pm2.metric("Pack RUL", f"{_pack_rul:.0f} cy" if _pack_rul is not None else "—")
        _pm3.metric("Pack Capacity", _cap_label)
        _pm4.metric("Pack Resistance", f"{_pack_resistance * 1000:.1f} mΩ" if not (_pack_resistance != _pack_resistance) else "—")

        _soh_spread = max(_soh_values) - min(_soh_values)
        if _soh_spread > 5:
            st.error(f"⚠️ **{_bottleneck_cell}** is the pack bottleneck (SOH spread: {_soh_spread:.1f}%). Consider replacing or rebalancing.")
        elif _soh_spread > 2:
            st.warning(f"⚡ SOH spread is {_soh_spread:.1f}%. Monitor {_bottleneck_cell} closely.")
        else:
            st.success(f"✅ Pack is well-balanced (SOH spread: {_soh_spread:.1f}%)")

        st.subheader("Cell Matching Score")
        st.caption("Cells with similar degradation trajectories are better matched for pack assembly (minimises balancing losses).")
        _match_rows = []
        for _i in range(len(selected_pack_cells)):
            for _j in range(_i + 1, len(selected_pack_cells)):
                _ca, _cb = selected_pack_cells[_i], selected_pack_cells[_j]
                _soh_diff = abs(_soh_values[_i] - _soh_values[_j])
                _cap_diff = abs(_cap_values[_i] - _cap_values[_j]) / (_cap_values[_i] + 1e-9) * 100
                _res_diff = abs(_res_values[_i] - _res_values[_j]) / (_res_values[_i] + 1e-9) * 100
                _score = max(0, min(100, 100 - (_soh_diff * 2 + _cap_diff * 1.5 + _res_diff * 0.5)))
                _rec = ("Excellent match" if _score > 80 else
                        "Good match" if _score > 60 else
                        "Acceptable" if _score > 40 else
                        "Poor — avoid pairing")
                _match_rows.append({"Cell A": _ca, "Cell B": _cb, "Match Score": f"{_score:.0f}", "Recommendation": _rec})
        if _match_rows:
            _match_df = pd.DataFrame(_match_rows)
            st.dataframe(_match_df.head(200), use_container_width=True, hide_index=True)
            if len(_match_df) > 200:
                st.caption(f"Showing up to 200 rows — {len(_match_df)} total.")

        _bar_colors = []
        for _sv in _soh_values:
            _dist = abs(_sv - _pack_soh)
            if _dist <= 2:
                _bar_colors.append("#48bb78")
            elif _dist <= 5:
                _bar_colors.append("#f6ad55")
            else:
                _bar_colors.append("#fc8181")
        _fig_pack = go.Figure(go.Bar(
            x=selected_pack_cells, y=_soh_values,
            marker_color=_bar_colors,
            hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
        ))
        _fig_pack.add_hline(y=_pack_soh, line_dash="dash", line_color="#63b3ed", line_width=1,
                            annotation_text=f"Pack SOH {_pack_soh:.1f}%", annotation_font_color="#63b3ed")
        _fig_pack.update_layout(
            height=250,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=36, b=10),
            xaxis=dict(gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False, range=[50, 102]),
        )
        st.plotly_chart(_fig_pack, use_container_width=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Honest copy on cross-type RUL ──
    st.markdown("<div class='section-header'>About This Ranking</div>", unsafe_allow_html=True)
    st.markdown(
        """
        **Ranking method:** Cells are ranked by current SOH %, fade rate (30-cycle rolling average),
        and EOL proximity. SOH % is directly comparable across all cells — it's always relative
        to each cell's own initial capacity, so 75% SOH means the same thing for a NASA cell
        as for a synthetic cell.

        **Why RUL is not ranked across cell types:**
        Remaining Useful Life (cycles) is predicted by separate models for synthetic vs NASA cells,
        because the resistance features used for prediction are on incompatible measurement scales
        (synthesised internal resistance vs EIS electrolyte resistance from impedance spectroscopy).
        Comparing "300 cycles remaining" from one model to "40 cycles remaining" from another
        would be a meaningless number — the models aren't measuring the same thing.

        Individual RUL estimates are shown per cell on the Overview and Health pages (drill in
        via the cell selector in the sidebar).
        """,
    )

    with st.expander("Roadmap: Unified Ranking — what would unlock cross-type RUL comparison?"):
        st.markdown(
            """
            **Gate: 8+ real cells with diverse usage histories.**

            The current 4 NASA cells were all tested at identical lab conditions (24°C, 2A discharge).
            That means the only variation between them is cell-to-cell manufacturing spread —
            not the operating temperature, C-rate, and DoD variation that would make a combined
            resistance signal meaningful.

            Once 8 or more real cells are available with varied operating conditions, two changes
            become worthwhile:

            1. **Replace `resistance_ohm` with `resistance_normalized`** (ratio to each cell's
               own initial resistance) as the only resistance feature. Both synthetic and real
               cells start at 1.0 and rise — this is comparable across measurement methods.

            2. **Train one unified model** on the combined dataset. Validate with leave-cell-out
               to confirm it generalises across both real and synthetic cells.

            Until then, ranking by SOH and fade rate is the honest choice.
            """
        )

    # ── 🚨 Anomaly Alert History Log ─────────────────────────────────────────
    st.markdown("<div class='section-header'>🚨 Anomaly Alert History</div>", unsafe_allow_html=True)
    _anom_log = []
    for _cid, _fdf in featured_dfs.items():
        if "capacity_anomaly" in _fdf.columns:
            _cap_anom = _fdf[_fdf["capacity_anomaly"] == True]
            if len(_cap_anom) > 0:
                _last10 = _fdf.iloc[-min(10, len(_fdf)):]
                _recent = int(_last10["capacity_anomaly"].sum())
                _anom_log.append({
                    "cell_id": _cid,
                    "type": "Capacity",
                    "total_flags": len(_cap_anom),
                    "last_10_cycles": _recent,
                    "last_flagged_cycle": int(_cap_anom["cycle_number"].iloc[-1]),
                    "severity": "High" if _recent >= 3 else ("Moderate" if _recent >= 1 else "Low"),
                })
        if "resistance_anomaly" in _fdf.columns:
            _res_anom = _fdf[_fdf["resistance_anomaly"] == True]
            if len(_res_anom) > 0:
                _last10r = _fdf.iloc[-min(10, len(_fdf)):]
                _recent_r = int(_last10r["resistance_anomaly"].sum())
                _anom_log.append({
                    "cell_id": _cid,
                    "type": "Resistance",
                    "total_flags": len(_res_anom),
                    "last_10_cycles": _recent_r,
                    "last_flagged_cycle": int(_res_anom["cycle_number"].iloc[-1]),
                    "severity": "High" if _recent_r >= 3 else ("Moderate" if _recent_r >= 1 else "Low"),
                })
    if _anom_log:
        _anom_df = pd.DataFrame(_anom_log).sort_values(["severity", "last_10_cycles"], ascending=[True, False])
        _SEV_COLOUR = {"High": "#fc8181", "Moderate": "#f6ad55", "Low": "#718096"}
        for _, _al in _anom_df.iterrows():
            _sc = _SEV_COLOUR[_al["severity"]]
            _md_html(
                f"<div style='display:flex;align-items:center;gap:16px;padding:10px 14px;"
                f"margin-bottom:6px;background:#1e2a38;border-radius:8px;"
                f"border-left:3px solid {_sc}'>"
                f"<span style='font-size:13px;font-weight:700;color:#e2e8f0;min-width:80px'>{_al['cell_id']}</span>"
                f"<span style='font-size:11px;padding:2px 8px;border-radius:4px;"
                f"background:{_sc}22;color:{_sc};border:1px solid {_sc}44'>{_al['type']}</span>"
                f"<span style='font-size:12px;color:#a0aec0'>Total: <strong style='color:#e2e8f0'>{_al['total_flags']}</strong> flags</span>"
                f"<span style='font-size:12px;color:#a0aec0'>Last 10 cycles: <strong style='color:{_sc}'>{_al['last_10_cycles']}</strong></span>"
                f"<span style='font-size:12px;color:#4a5568'>Last at cycle {_al['last_flagged_cycle']}</span>"
                f"<span style='margin-left:auto;font-size:11px;font-weight:700;color:{_sc}'>{_al['severity']}</span>"
                f"</div>"
            )
    else:
        st.success("No anomaly flags detected across the fleet.")
