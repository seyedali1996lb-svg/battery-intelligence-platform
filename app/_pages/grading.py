import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # app/ dir for utils, sidebar
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import (
    base_layout, LEGEND_H, PLOTLY_CONFIG, _md_html, soh_status,
    friendly, _soh_sparkline_svg, _cell_provenance, _analysis_provenance,
    NASA_CELL_IDS,
)
from design_system import (
    make_badge, make_state_badge, provenance_banner,
    BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST, BADGE_UNAVAIL,
    BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC,
    ACTION_META, CONF_META,
)


def page_grading(cell_ids: list, active_fdfs: dict, bundles: dict, selected: str):
    import numpy as _np_grade

    _md_html("""<div style="padding-top:32px;margin-bottom:8px"><div style="font-size:22px;font-weight:700;color:#e2e8f0">⏳ Cell Grading</div><div style="font-size:13px;color:#718096;margin-top:2px">Early-cycle lifetime prediction · Severson et al. (2019, Nature Energy)</div></div>""")
    st.caption(
        "Grades A–C are derived from the first 100 cycles only: fade rate, capacity variance, "
        "and resistance slope. A high score predicts long life; a low score flags early replacement."
    )

    # ── Compute grades ──────────────────────────────────────────────────────
    _GRADE_CFG = {
        "A": {"color": "#48bb78", "bg": "#1a2e22", "border": "#2f6846",
              "label": "Grade A", "lifetime": "> 800 cycles", "icon": "🟢"},
        "B": {"color": "#ed8936", "bg": "#2d2012", "border": "#7b4a10",
              "label": "Grade B", "lifetime": "400 – 800 cycles", "icon": "🟡"},
        "C": {"color": "#fc8181", "bg": "#2d1515", "border": "#7b2020",
              "label": "Grade C", "lifetime": "< 400 cycles", "icon": "🔴"},
        "—": {"color": "#718096", "bg": "#1a202c", "border": "#2d3748",
              "label": "No data", "lifetime": "Insufficient data", "icon": "⚪"},
    }

    _grade_results = []
    for _cid in cell_ids:
        _gdf = active_fdfs.get(_cid)
        if _gdf is None:
            continue
        _early = _gdf[_gdf["cycle_number"] <= 100]
        if len(_early) < 20:
            _grade_results.append({"cid": _cid, "grade": "—", "score": None,
                                   "fade": None, "var": None, "slope": None})
            continue
        _cap0  = float(_early["capacity_ah"].iloc[0])
        _res0  = max(float(_early["resistance_ohm"].iloc[0]), 1e-6)
        _fade  = (float(_early["capacity_ah"].iloc[0]) - float(_early["capacity_ah"].iloc[-1])) / len(_early)
        _var   = float(_early["capacity_ah"].var())
        _slope = float(_np_grade.polyfit(_early["cycle_number"], _early["resistance_ohm"], 1)[0])
        # Normalise metrics relative to cell's own initial values so the score
        # is scale-independent (works for both 0.74 Ah synthetic and 1.8 Ah NASA cells)
        _fade_pct_per_cy  = _fade / _cap0 * 100            # % capacity lost per cycle
        _cv2              = _var / (_cap0 ** 2) * 1e4       # dimensionless variance × 1e4
        _r_slope_pct      = abs(_slope) / _res0 * 100       # % resistance growth per cycle
        _score = float(_np_grade.clip(
            100 - _fade_pct_per_cy * 400 - _cv2 * 8 - _r_slope_pct * 150,
            0, 100,
        ))
        _grade = "A" if _score >= 75 else ("B" if _score >= 50 else "C")
        _grade_results.append({"cid": _cid, "grade": _grade, "score": _score,
                                "fade": _fade, "var": _var, "slope": _slope})

    if not _grade_results:
        st.info("No cells available.")
        return

    # Sort: A first, then by score descending
    _grade_results.sort(key=lambda r: ({"A": 0, "B": 1, "C": 2, "—": 3}[r["grade"]],
                                        -(r["score"] or 0)))

    # ── Grade cards ─────────────────────────────────────────────────────────
    _cols_per_row = 3
    for _i in range(0, len(_grade_results), _cols_per_row):
        _row_items = _grade_results[_i:_i + _cols_per_row]
        _cols = st.columns(len(_row_items))
        for _col, _r in zip(_cols, _row_items):
            _cfg = _GRADE_CFG[_r["grade"]]
            _score_bar = ""
            if _r["score"] is not None:
                _pct = _r["score"]
                _score_bar = (
                    f"<div style='margin:10px 0 4px;background:#0e1117;border-radius:4px;height:6px'>"
                    f"<div style='width:{_pct:.0f}%;background:{_cfg['color']};height:6px;border-radius:4px'></div>"
                    f"</div>"
                    f"<div style='font-size:11px;color:#718096'>{_pct:.1f} / 100</div>"
                )
                _details = (
                    f"<div style='margin-top:10px;font-size:11px;color:#718096;line-height:1.8'>"
                    f"Fade rate: {_r['fade']*1000:.4f} mAh/cy<br>"
                    f"Cap. variance: {_r['var']*1e6:.2f} µAh²<br>"
                    f"R slope: {_r['slope']*1000:.4f} mΩ/cy"
                    f"</div>"
                )
            else:
                _details = "<div style='margin-top:8px;font-size:11px;color:#718096'>Need ≥ 20 early cycles</div>"
            with _col:
                _md_html(
                    f"<div style='background:{_cfg['bg']};border:1px solid {_cfg['border']};"
                    f"border-radius:10px;padding:16px 18px;margin-bottom:12px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div style='font-size:15px;font-weight:700;color:#e2e8f0'>{_r['cid']}</div>"
                    f"<div style='font-size:22px;font-weight:800;color:{_cfg['color']}'>{_r['grade']}</div>"
                    f"</div>"
                    f"<div style='font-size:11px;color:{_cfg['color']};margin-top:2px'>"
                    f"{_cfg['icon']} Predicted lifetime: {_cfg['lifetime']}</div>"
                    f"{_score_bar}"
                    f"{_details}"
                    f"</div>"
                )

    # ── Score bar chart ──────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _valid = [r for r in _grade_results if r["score"] is not None]
    if _valid:
        _color_map = {"A": "#48bb78", "B": "#ed8936", "C": "#fc8181"}
        _fig_grade = go.Figure(go.Bar(
            x=[r["cid"] for r in _valid],
            y=[r["score"] for r in _valid],
            marker=dict(color=[_color_map[r["grade"]] for r in _valid]),
            text=[f"{r['score']:.0f}" for r in _valid],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
        ))
        _fig_grade.update_layout(
            **base_layout(
                height=300,
                xaxis=dict(title="Cell", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                yaxis=dict(title="Grade Score (0–100)", gridcolor="#232d3b",
                           linecolor="#2d3748", zeroline=False, range=[0, 115]),
            ),
        )
        _fig_grade.add_hline(y=75, line_dash="dot", line_color="#48bb78", line_width=1,
                              annotation_text="A threshold", annotation_position="right")
        _fig_grade.add_hline(y=50, line_dash="dot", line_color="#ed8936", line_width=1,
                              annotation_text="B threshold", annotation_position="right")
        _fig_grade.update_layout(title=dict(
            text="Grade Score by Cell — sorted best to worst",
            font=dict(size=12, color="#a0aec0"), x=0))
        st.plotly_chart(_fig_grade, use_container_width=True)

    st.caption(
        "Methodology: simplified adaptation of Severson et al. (2019, Nature Energy). "
        "Score = 100 − f(fade rate) − f(capacity variance) − f(resistance slope) over first 100 cycles. "
        "Not a validated commercial grading system."
    )

    st.caption(
        "Methodology: simplified adaptation of Severson et al. (2019). Score derived from early-cycle "
        "fade rate, capacity variance, and resistance slope. Not a validated commercial grading system."
    )
