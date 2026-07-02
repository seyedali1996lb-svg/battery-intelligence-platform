"""Page: Compare"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _md_html


def page_compare(cell_ids: list, active_fdfs: dict, bundles: dict):
    st.markdown("# ⚖️ Cell Comparison")

    if len(cell_ids) < 2:
        st.warning("Comparison requires at least 2 cells in the active fleet.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cell_a = st.selectbox("Cell A", options=cell_ids, index=0, key="compare_cell_a")
    with col2:
        cell_b = st.selectbox("Cell B", options=cell_ids, index=min(1, len(cell_ids) - 1), key="compare_cell_b")

    if cell_a == cell_b:
        st.info("Select two different cells to compare.")
        return

    df_a = active_fdfs[cell_a]
    df_b = active_fdfs[cell_b]

    # ── Summary metrics ──
    st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _soh_a  = float(df_a["soh_pct"].iloc[-1])
    _soh_b  = float(df_b["soh_pct"].iloc[-1])
    _cyc_a  = int(df_a["cycle_number"].iloc[-1])
    _cyc_b  = int(df_b["cycle_number"].iloc[-1])
    _res_a  = float(df_a["resistance_ohm"].iloc[-1])
    _res_b  = float(df_b["resistance_ohm"].iloc[-1])
    _fade_a = float(df_a["fade_rate_50cy"].iloc[-1])
    _fade_b = float(df_b["fade_rate_50cy"].iloc[-1])

    _mc1.metric(f"SOH — {cell_a}", f"{_soh_a:.1f}%", delta=f"{_soh_a - _soh_b:+.1f}%")
    _mc2.metric(f"SOH — {cell_b}", f"{_soh_b:.1f}%")
    _mc3.metric(f"Cycles — {cell_a}", f"{_cyc_a:,}", delta=f"{_cyc_a - _cyc_b:+,}")
    _mc4.metric(f"Cycles — {cell_b}", f"{_cyc_b:,}")

    _mc5, _mc6, _mc7, _mc8 = st.columns(4)
    _mc5.metric(f"Resistance — {cell_a}", f"{_res_a*1000:.1f} mΩ", delta=f"{(_res_a - _res_b)*1000:+.1f} mΩ")
    _mc6.metric(f"Resistance — {cell_b}", f"{_res_b*1000:.1f} mΩ")
    _mc7.metric(f"Fade rate — {cell_a}", f"{_fade_a*1000:.2f} mSOH/cy", delta=f"{(_fade_a - _fade_b)*1000:+.2f}")
    _mc8.metric(f"Fade rate — {cell_b}", f"{_fade_b*1000:.2f} mSOH/cy")

    # ── SOH trajectory ──
    st.markdown("<div class='section-header'>SOH Trajectory Comparison (10-cycle rolling avg)</div>", unsafe_allow_html=True)
    _soh_col_a = "soh_rolling_avg" if "soh_rolling_avg" in df_a.columns else "soh_pct"
    _soh_col_b = "soh_rolling_avg" if "soh_rolling_avg" in df_b.columns else "soh_pct"

    _fig_soh = go.Figure()
    _fig_soh.add_trace(go.Scatter(
        x=df_a["cycle_number"], y=df_a[_soh_col_a], name=cell_a,
        line=dict(color="#63b3ed", width=2),
        hovertemplate=f"<b>{cell_a}</b> cy %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    _fig_soh.add_trace(go.Scatter(
        x=df_b["cycle_number"], y=df_b[_soh_col_b], name=cell_b,
        line=dict(color="#fc8181", width=2),
        hovertemplate=f"<b>{cell_b}</b> cy %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    _fig_soh.update_layout(
        height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"), margin=dict(l=10, r=10, t=36, b=10), hovermode="x unified",
        xaxis=dict(title="Cycle", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        yaxis=dict(title="SOH %", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        legend=dict(font=dict(size=11, color="#718096")),
    )
    st.plotly_chart(_fig_soh, use_container_width=True)

    # ── Resistance trajectory ──
    st.markdown("<div class='section-header'>Resistance Trajectory Comparison</div>", unsafe_allow_html=True)
    _fig_res = go.Figure()
    _fig_res.add_trace(go.Scatter(
        x=df_a["cycle_number"], y=df_a["resistance_ohm"] * 1000, name=cell_a,
        line=dict(color="#63b3ed", width=2),
        hovertemplate=f"<b>{cell_a}</b> cy %{{x}}: %{{y:.1f}} mΩ<extra></extra>",
    ))
    _fig_res.add_trace(go.Scatter(
        x=df_b["cycle_number"], y=df_b["resistance_ohm"] * 1000, name=cell_b,
        line=dict(color="#fc8181", width=2),
        hovertemplate=f"<b>{cell_b}</b> cy %{{x}}: %{{y:.1f}} mΩ<extra></extra>",
    ))
    _fig_res.update_layout(
        height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"), margin=dict(l=10, r=10, t=36, b=10), hovermode="x unified",
        xaxis=dict(title="Cycle", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        yaxis=dict(title="Resistance (mΩ)", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        legend=dict(font=dict(size=11, color="#718096")),
    )
    st.plotly_chart(_fig_res, use_container_width=True)

    # ── Engineering Radar Chart ──
    st.markdown("<div class='section-header'>Engineering Radar — Multi-Metric Health Profile</div>", unsafe_allow_html=True)
    _md_html("""<div style="font-size:12px;color:#718096;margin-bottom:10px">Normalized 0–1 scale. <strong style="color:#e2e8f0">Outer = better</strong> for SOH, CE, dQ/dV; <strong style="color:#e2e8f0">Inner = better</strong> for resistance and fade rate (inverted). Standard CATL / A123 engineering review format.</div>""")
    try:
        def _radar_val(df, col, invert=False, scale=1.0, floor=0.0):
            if col not in df.columns:
                return None
            v = float(df[col].iloc[-1])
            if pd.isna(v):
                return None
            v = (v - floor) / (scale - floor + 1e-9)
            v = max(0.0, min(1.0, v))
            return (1.0 - v) if invert else v

        _radar_axes = [
            ("SOH %",          "soh_pct",              False, 100.0, 60.0),
            ("CE",             "ce_rolling_30cy",       False, 1.0,   0.97),
            ("dQ/dV Peak",     "dqdv_peak_value",       False, None,  0.0),
            ("Fade Rate",      "fade_rate_30cy",        True,  0.005, 0.0),
            ("Resistance",     "resistance_normalized", True,  1.8,   1.0),
            ("Throughput kWh", "cumulative_kwh",        False, None,  0.0),
        ]

        _raw_a, _raw_b, _labels = [], [], []
        for _lbl, _col, _inv, _scale, _floor in _radar_axes:
            _va = _radar_val(df_a, _col, _inv, _scale or 1.0, _floor) if _scale else None
            _vb = _radar_val(df_b, _col, _inv, _scale or 1.0, _floor) if _scale else None
            if _va is None and _vb is None:
                continue
            if _scale is None:
                _rv_a  = float(df_a[_col].iloc[-1]) if _col in df_a.columns and not pd.isna(df_a[_col].iloc[-1]) else 0.0
                _rv_b  = float(df_b[_col].iloc[-1]) if _col in df_b.columns and not pd.isna(df_b[_col].iloc[-1]) else 0.0
                _dyn_max = max(abs(_rv_a), abs(_rv_b), 1e-9)
                _va = max(0.0, min(1.0, _rv_a / _dyn_max)) if not _inv else 1.0 - max(0.0, min(1.0, _rv_a / _dyn_max))
                _vb = max(0.0, min(1.0, _rv_b / _dyn_max)) if not _inv else 1.0 - max(0.0, min(1.0, _rv_b / _dyn_max))
            _raw_a.append(_va if _va is not None else 0.0)
            _raw_b.append(_vb if _vb is not None else 0.0)
            _labels.append(_lbl)

        if len(_labels) >= 3:
            _labels_c = _labels + [_labels[0]]
            _vals_a_c = _raw_a  + [_raw_a[0]]
            _vals_b_c = _raw_b  + [_raw_b[0]]
            _fig_radar = go.Figure()
            _fig_radar.add_trace(go.Scatterpolar(
                r=_vals_a_c, theta=_labels_c, name=cell_a,
                line=dict(color="#63b3ed", width=2),
                fill="toself", fillcolor="rgba(99,179,237,0.08)",
            ))
            _fig_radar.add_trace(go.Scatterpolar(
                r=_vals_b_c, theta=_labels_c, name=cell_b,
                line=dict(color="#fc8181", width=2),
                fill="toself", fillcolor="rgba(252,129,129,0.08)",
            ))
            _fig_radar.update_layout(
                height=380, paper_bgcolor="#0e1117", font=dict(color="#e2e8f0", size=12),
                polar=dict(
                    bgcolor="#0e1117",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e2a38", linecolor="#2d3748", tickfont=dict(size=9, color="#4a5568")),
                    angularaxis=dict(gridcolor="#1e2a38", linecolor="#2d3748", tickfont=dict(size=11, color="#a0aec0")),
                ),
                legend=dict(font=dict(size=11, color="#718096")),
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(_fig_radar, use_container_width=True)
        else:
            st.info("Insufficient features available for radar chart.")
    except Exception as _re:
        st.info(f"Radar chart unavailable: {_re}")

    # ── Summary verdict ──
    _soh_winner  = cell_a if _soh_a > _soh_b else cell_b
    _fade_winner = cell_a if _fade_a < _fade_b else cell_b
    st.markdown(
        f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
        f"padding:18px 22px;margin-top:8px'>"
        f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:8px'>Summary Verdict</div>"
        f"<div style='font-size:13px;color:#a0aec0;line-height:1.8'>"
        f"<strong style='color:#63b3ed'>{_soh_winner}</strong> has higher current SOH. "
        f"<strong style='color:#63b3ed'>{_fade_winner}</strong> is degrading more slowly."
        f"</div></div>",
        unsafe_allow_html=True,
    )
