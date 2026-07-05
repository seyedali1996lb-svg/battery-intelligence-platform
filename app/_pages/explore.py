"""Page: Explore (Compare / Cluster / Cohort)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _md_html, _empty_state, base_layout, _action_bar, NASA_CELL_IDS, SEVERSON_CELL_PREFIX


def page_compare(cell_ids: list, active_fdfs: dict, bundles: dict):
    _action_bar("compare")
    st.markdown("# Explore")

    _exp_view = st.radio(
        "view", ["Compare", "Cluster", "Cohort", "Pack Builder"],
        horizontal=True, key="explore_view_radio", label_visibility="collapsed",
    )

    # ── Cluster tab ──────────────────────────────────────────────────────────
    if _exp_view == "Cluster":
        _page_explore_cluster(cell_ids, active_fdfs, bundles)
        return

    # ── Cohort tab ───────────────────────────────────────────────────────────
    if _exp_view == "Cohort":
        _page_explore_cohort(active_fdfs)
        return

    # ── Pack Builder tab ─────────────────────────────────────────────────────
    if _exp_view == "Pack Builder":
        _page_pack_builder(cell_ids, active_fdfs)
        return

    # ── Compare tab (default) ─────────────────────────────────────────────
    if len(cell_ids) < 2:
        st.warning("Comparison requires at least 2 cells in the active fleet.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cell_a = st.selectbox("Cell A", options=cell_ids, index=0, key="compare_cell_a")
    with col2:
        cell_b = st.selectbox("Cell B", options=cell_ids, index=min(1, len(cell_ids) - 1), key="compare_cell_b")

    if cell_a == cell_b:
        _empty_state(
            "Select two different cells",
            "Cell A and Cell B must be different to generate a comparison.",
            icon="↔",
        )
        return

    df_a = active_fdfs[cell_a]
    df_b = active_fdfs[cell_b]

    # ── Summary metrics comparison ──
    st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _soh_a = float(df_a["soh_pct"].iloc[-1])
    _soh_b = float(df_b["soh_pct"].iloc[-1])
    _cyc_a = int(df_a["cycle_number"].iloc[-1])
    _cyc_b = int(df_b["cycle_number"].iloc[-1])
    _res_a = float(df_a["resistance_ohm"].iloc[-1]) if "resistance_ohm" in df_a.columns else float("nan")
    _res_b = float(df_b["resistance_ohm"].iloc[-1]) if "resistance_ohm" in df_b.columns else float("nan")
    _fade_a = float(df_a["fade_rate_50cy"].iloc[-1])
    _fade_b = float(df_b["fade_rate_50cy"].iloc[-1])

    _mc1.metric(f"SOH — {cell_a}", f"{_soh_a:.1f}%", delta=f"{_soh_a - _soh_b:+.1f}%")
    _mc2.metric(f"SOH — {cell_b}", f"{_soh_b:.1f}%")
    _mc3.metric(f"Cycles — {cell_a}", f"{_cyc_a:,}", delta=f"{_cyc_a - _cyc_b:+,}")
    _mc4.metric(f"Cycles — {cell_b}", f"{_cyc_b:,}")

    _mc5, _mc6, _mc7, _mc8 = st.columns(4)
    _res_a_str = f"{_res_a*1000:.1f} mΩ" if not (isinstance(_res_a, float) and _res_a != _res_a) else "N/A"
    _res_b_str = f"{_res_b*1000:.1f} mΩ" if not (isinstance(_res_b, float) and _res_b != _res_b) else "N/A"
    _res_delta_str = f"{(_res_a - _res_b)*1000:+.1f} mΩ" if _res_a == _res_a and _res_b == _res_b else None
    _mc5.metric(f"Resistance — {cell_a}", _res_a_str, delta=_res_delta_str)
    _mc6.metric(f"Resistance — {cell_b}", _res_b_str)
    _mc7.metric(f"Fade rate — {cell_a}", f"{_fade_a*1000:.2f} mSOH/cy", delta=f"{(_fade_a - _fade_b)*1000:+.2f}")
    _mc8.metric(f"Fade rate — {cell_b}", f"{_fade_b*1000:.2f} mSOH/cy")

    # ── SOH trajectory comparison ──
    st.markdown("<div class='section-header'>SOH Trajectory Comparison (10-cycle rolling avg)</div>", unsafe_allow_html=True)
    _soh_col_a = "soh_rolling_avg" if "soh_rolling_avg" in df_a.columns else "soh_pct"
    _soh_col_b = "soh_rolling_avg" if "soh_rolling_avg" in df_b.columns else "soh_pct"

    _fig_soh = go.Figure()
    _fig_soh.add_trace(go.Scatter(
        x=df_a["cycle_number"], y=df_a[_soh_col_a],
        name=cell_a, line=dict(color="#63b3ed", width=2),
        hovertemplate=f"<b>{cell_a}</b> cy %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    _fig_soh.add_trace(go.Scatter(
        x=df_b["cycle_number"], y=df_b[_soh_col_b],
        name=cell_b, line=dict(color="#fc8181", width=2),
        hovertemplate=f"<b>{cell_b}</b> cy %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    _fig_soh.update_layout(
        height=300,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
        xaxis=dict(title="Cycle", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        yaxis=dict(title="SOH %", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
        legend=dict(font=dict(size=11, color="#718096")),
    )
    st.plotly_chart(_fig_soh, use_container_width=True)

    # ── Resistance trajectory comparison ──
    st.markdown("<div class='section-header'>Resistance Trajectory Comparison</div>", unsafe_allow_html=True)
    if "resistance_ohm" in df_a.columns and "resistance_ohm" in df_b.columns:
        _fig_res = go.Figure()
        _fig_res.add_trace(go.Scatter(
            x=df_a["cycle_number"], y=df_a["resistance_ohm"] * 1000,
            name=cell_a, line=dict(color="#63b3ed", width=2),
            hovertemplate=f"<b>{cell_a}</b> cy %{{x}}: %{{y:.1f}} mΩ<extra></extra>",
        ))
        _fig_res.add_trace(go.Scatter(
            x=df_b["cycle_number"], y=df_b["resistance_ohm"] * 1000,
            name=cell_b, line=dict(color="#fc8181", width=2),
            hovertemplate=f"<b>{cell_b}</b> cy %{{x}}: %{{y:.1f}} mΩ<extra></extra>",
        ))
        _fig_res.update_layout(
            height=280,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=36, b=10),
            hovermode="x unified",
            xaxis=dict(title="Cycle", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="Resistance (mΩ)", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
            legend=dict(font=dict(size=11, color="#718096")),
        )
        st.plotly_chart(_fig_res, use_container_width=True)
    else:
        _empty_state("Resistance data unavailable", "One or both cells are missing resistance measurements.", icon="Ω")

    # ── Engineering Radar Chart (CATL / A123 format) ─────────────────────────
    st.markdown("<div class='section-header'>Engineering Radar — Multi-Metric Health Profile</div>", unsafe_allow_html=True)
    _md_html("""<div style="font-size:12px;color:#8896a8;margin-bottom:10px">Normalized 0–1 scale. <strong style="color:#e2e8f0">Outer = better</strong> for SOH, CE, dQ/dV; <strong style="color:#e2e8f0">Inner = better</strong> for resistance and fade rate (inverted). Standard CATL / A123 engineering review format.</div>""")
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
            ("SOH_Q",        "soh_pct",              False, 100.0, 60.0),
            ("CE (η_CE)",    "ce_rolling_30cy",       False, 1.0,   0.97),
            ("ICA Peak",     "dqdv_peak_value",       False, None,  0.0),
            ("Q̇_fade",      "fade_rate_30cy",        True,  0.005, 0.0),
            ("R_internal",   "resistance_normalized", True,  1.8,   1.0),
            ("E_throughput", "cumulative_kwh",        False, None,  0.0),
        ]

        # For axes without a fixed scale, normalise across both cells
        _raw_a, _raw_b, _labels = [], [], []
        for _lbl, _col, _inv, _scale, _floor in _radar_axes:
            _va = _radar_val(df_a, _col, _inv, _scale or 1.0, _floor) if _scale else None
            _vb = _radar_val(df_b, _col, _inv, _scale or 1.0, _floor) if _scale else None
            if _va is None and _vb is None:
                # skip this axis
                continue
            if _scale is None:
                # dynamic scale: max of both raw values
                _rv_a = float(df_a[_col].iloc[-1]) if _col in df_a.columns and not pd.isna(df_a[_col].iloc[-1]) else 0.0
                _rv_b = float(df_b[_col].iloc[-1]) if _col in df_b.columns and not pd.isna(df_b[_col].iloc[-1]) else 0.0
                _dyn_max = max(abs(_rv_a), abs(_rv_b), 1e-9)
                _va = max(0.0, min(1.0, _rv_a / _dyn_max)) if not _inv else 1.0 - max(0.0, min(1.0, _rv_a / _dyn_max))
                _vb = max(0.0, min(1.0, _rv_b / _dyn_max)) if not _inv else 1.0 - max(0.0, min(1.0, _rv_b / _dyn_max))
            _raw_a.append(_va if _va is not None else 0.0)
            _raw_b.append(_vb if _vb is not None else 0.0)
            _labels.append(_lbl)

        if len(_labels) >= 3:
            _labels_c  = _labels + [_labels[0]]
            _vals_a_c  = _raw_a  + [_raw_a[0]]
            _vals_b_c  = _raw_b  + [_raw_b[0]]
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
                height=380,
                paper_bgcolor="#0e1117",
                font=dict(color="#e2e8f0", size=12),
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
    _soh_winner = cell_a if _soh_a > _soh_b else cell_b
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

    # end of Compare view — Cluster and Cohort are handled via radio+return above


# ---------------------------------------------------------------------------
# Explore sub-pages (Cluster / Cohort) — called from page_compare radio switch
# ---------------------------------------------------------------------------

def _page_explore_cluster(cell_ids: list, active_fdfs: dict, bundles: dict):
    """Cross-fleet degradation clustering view."""
    st.markdown(
        "<div class='section-header'>Cross-Fleet Degradation Clustering</div>",
        unsafe_allow_html=True,
    )
    _md_html(
        "<div style='font-size:12px;color:#8896a8;margin-bottom:10px'>"
        "K-means on last-cycle feature vectors (SOH, fade rate, resistance, CE). "
        "Cells in the same cluster share a degradation signature — their historical "
        "trajectories can inform RUL estimates for each other."
        "</div>"
    )
    # Cells picked in the Compare tab's selectboxes this session, if any — used
    # to highlight matching points on the cluster scatter below. Read directly
    # from the widget keys rather than threading params, since Compare's
    # selectboxes aren't rendered when the user lands here via Cluster first.
    cell_a = st.session_state.get("compare_cell_a")
    cell_b = st.session_state.get("compare_cell_b")
    try:
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.cluster import KMeans as _KM
        from sklearn.metrics import silhouette_score as _sil

        # ── Build last-cycle feature matrix across all fleet cells ──
        _FEAT_COLS = ["soh_pct", "fade_rate_30cy", "resistance_ohm", "ce_rolling_30cy"]
        _clust_rows = []
        for _cid, _cdf in active_fdfs.items():
            _row = {"cell_id": _cid}
            for _fc in _FEAT_COLS:
                if _fc in _cdf.columns:
                    _v = _cdf[_fc].dropna()
                    _row[_fc] = float(_v.iloc[-1]) if len(_v) else float("nan")
                else:
                    _row[_fc] = float("nan")
            _clust_rows.append(_row)

        import pandas as _pd_cl
        _clust_df = _pd_cl.DataFrame(_clust_rows).set_index("cell_id")

        # Drop columns that are all-NaN, then drop any remaining NaN rows
        _clust_df = _clust_df.dropna(axis=1, how="all")
        _clust_df_clean = _clust_df.dropna()

        if len(_clust_df_clean) < 3:
            _empty_state(
                "Fleet too small for clustering",
                "Cross-fleet clustering requires at least 3 cells with complete feature data "
                "(SOH, fade rate, resistance, CE). Some cells may have been excluded due to missing columns.",
                "→ Import additional cell data or switch to a data source with more cells.",
                "◎",
            )
        else:
            _X_raw = _clust_df_clean.values
            _X_sc  = _SS().fit_transform(_X_raw)

            # Auto-select k via silhouette (2–4), capped at n_cells-1
            _k_max = min(4, len(_clust_df_clean) - 1)
            if _k_max < 2:
                _best_k = 2
            else:
                _best_k, _best_sil = 2, -1.0
                for _k in range(2, _k_max + 1):
                    _km_try = _KM(n_clusters=_k, random_state=42, n_init=10)
                    _lab_try = _km_try.fit_predict(_X_sc)
                    if len(set(_lab_try)) < 2:
                        continue
                    _s = _sil(_X_sc, _lab_try)
                    if _s > _best_sil:
                        _best_sil, _best_k = _s, _k

            _km   = _KM(n_clusters=_best_k, random_state=42, n_init=10)
            _labs = _km.fit_predict(_X_sc)
            _clust_df_clean = _clust_df_clean.copy()
            _clust_df_clean["cluster"] = _labs

            # Cluster colour palette
            _CL_COLOURS = ["#63b3ed", "#68d391", "#f6ad55", "#fc8181"]
            _cl_col_map  = {i: _CL_COLOURS[i % len(_CL_COLOURS)] for i in range(_best_k)}

            # ── Scatter: SOH vs fade rate, coloured by cluster ──
            _fig_cl = go.Figure()
            for _ki in range(_best_k):
                _mask = _clust_df_clean["cluster"] == _ki
                _sub  = _clust_df_clean[_mask]
                _pts_x = _sub["soh_pct"].tolist()
                _pts_y = (_sub["fade_rate_30cy"] * 1000).tolist() if "fade_rate_30cy" in _sub.columns else [0] * len(_sub)
                _c     = _cl_col_map[_ki]
                _fig_cl.add_trace(go.Scatter(
                    x=_pts_x, y=_pts_y,
                    mode="markers+text",
                    name=f"Cluster {_ki + 1}",
                    text=_sub.index.tolist(),
                    textposition="top center",
                    textfont=dict(size=9, color=_c),
                    marker=dict(size=10, color=_c, line=dict(width=1, color="#0e1117")),
                    hovertemplate="<b>%{text}</b><br>SOH: %{x:.1f}%<br>Fade: %{y:.3f} mSOH/cy<extra></extra>",
                ))

            # Highlight selected cells A and B with rings (only if set this session)
            for _hcid, _hcol in [(cell_a, "#ffffff"), (cell_b, "#fbd38d")]:
                if _hcid and _hcid in _clust_df_clean.index:
                    _hr = _clust_df_clean.loc[_hcid]
                    _fig_cl.add_trace(go.Scatter(
                        x=[_hr["soh_pct"]],
                        y=[_hr["fade_rate_30cy"] * 1000] if "fade_rate_30cy" in _hr.index else [0],
                        mode="markers",
                        name=f"{_hcid} (selected)",
                        marker=dict(size=18, color="rgba(0,0,0,0)",
                                    line=dict(width=2, color=_hcol)),
                        hovertemplate=f"<b>{_hcid}</b> (selected)<extra></extra>",
                        showlegend=True,
                    ))

            _fig_cl.update_layout(
                height=360,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"),
                margin=dict(l=10, r=10, t=36, b=40),
                hovermode="closest",
                xaxis=dict(title="SOH %", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
                yaxis=dict(title="30-cy Fade Rate (mSOH/cy)", gridcolor="#1e2a38", linecolor="#2d3748", zeroline=False),
                legend=dict(font=dict(size=10, color="#718096")),
                title=dict(text=f"Fleet degradation clusters (k={_best_k}, silhouette-optimised)",
                           font=dict(size=12, color="#a0aec0"), x=0),
            )
            st.plotly_chart(_fig_cl, use_container_width=True)

            # ── Similar-cell lookup (only shown if Compare selections exist) ──
            if cell_a or cell_b:
                st.markdown(
                    "<div style='font-size:12px;font-weight:600;color:#e2e8f0;margin:12px 0 6px'>"
                    "Cells with similar degradation signature</div>",
                    unsafe_allow_html=True,
                )
                _sim_c1, _sim_c2 = st.columns(2)
                for _scol, _scid in [(_sim_c1, cell_a), (_sim_c2, cell_b)]:
                    if _scid and _scid in _clust_df_clean.index:
                        _sk = int(_clust_df_clean.loc[_scid, "cluster"])
                        _peers = [c for c in _clust_df_clean[_clust_df_clean["cluster"] == _sk].index if c != _scid]
                        _ck    = _cl_col_map[_sk]
                        _peer_html = " &nbsp;".join(
                            f"<span style='background:{_ck}22;border:1px solid {_ck}55;color:{_ck};"
                            f"font-size:11px;font-weight:600;padding:2px 9px;border-radius:8px'>{p}</span>"
                            for p in _peers
                        ) if _peers else "<span style='color:#4a5568;font-size:12px'>No peers — unique cluster</span>"
                        _scol.markdown(
                            f"<div style='background:#1e2a38;border:1px solid {_ck}44;border-radius:8px;"
                            f"padding:12px 14px'>"
                            f"<div style='font-size:11px;font-weight:700;color:{_ck};margin-bottom:6px'>"
                            f"{_scid} — Cluster {_sk + 1}</div>"
                            f"<div style='font-size:11px;color:#a0aec0;margin-bottom:8px'>Similar cells:</div>"
                            f"{_peer_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Cluster summary table ──
            with st.expander("Cluster statistics"):
                _cl_summary = []
                for _ki in range(_best_k):
                    _sub = _clust_df_clean[_clust_df_clean["cluster"] == _ki]
                    _row_s = {"Cluster": f"Cluster {_ki + 1}", "Cells": len(_sub)}
                    for _fc in _clust_df_clean.columns:
                        if _fc == "cluster":
                            continue
                        _row_s[_fc] = f"{_sub[_fc].mean():.3f}" if _fc in _sub else "—"
                    _cl_summary.append(_row_s)
                st.dataframe(_pd_cl.DataFrame(_cl_summary), use_container_width=True, hide_index=True)

    except ImportError:
        st.info("scikit-learn required for clustering (already in requirements.txt).")
    except Exception as _ce:
        st.info(f"Clustering unavailable: {_ce}")


def _page_explore_cohort(active_fdfs: dict):
    """Cohort analysis view — cells grouped by user-set tags."""
    # ── Inline cohort tagging ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Tag cells by batch, supplier, or site</div>", unsafe_allow_html=True)
    st.caption("Tags are applied immediately. Use cohort names like 'Batch-A', 'Site-London', 'Supplier-X'.")
    import db
    if "cell_cohort_tags" not in st.session_state:
        st.session_state["cell_cohort_tags"] = db.load_cohort_tags()
    _tag_cols = st.columns(min(4, len(active_fdfs)))
    for _ti, _tcid in enumerate(active_fdfs.keys()):
        with _tag_cols[_ti % 4]:
            _cur_tag = st.session_state["cell_cohort_tags"].get(_tcid, "")
            _new_tag = st.text_input(_tcid, value=_cur_tag, key=f"cohort_tag_explore_{_tcid}",
                                     placeholder="e.g. Batch-A")
            if _new_tag != _cur_tag:
                st.session_state["cell_cohort_tags"][_tcid] = _new_tag
                db.save_cohort_tag(_tcid, _new_tag)
                st.rerun()

    # ── Cohort analysis ────────────────────────────────────────────────────────
    _e5_tags = st.session_state.get("cell_cohort_tags", {})
    _tagged  = {cid: tag for cid, tag in _e5_tags.items() if tag.strip() and cid in active_fdfs}
    if _tagged:
        st.markdown("<div class='section-header'>Cohort Analysis</div>", unsafe_allow_html=True)
        st.caption("Cells grouped by tag (set on Import page). Comparing avg SOH and fade rate across cohorts.")
        _cohorts: dict[str, list] = {}
        for cid, tag in _tagged.items():
            _cohorts.setdefault(tag, []).append(cid)
        for cid in active_fdfs:
            if cid not in _tagged:
                _cohorts.setdefault("(untagged)", []).append(cid)
        _cohort_stats = []
        for _ctag, _ccids in _cohorts.items():
            _sohs  = [float(active_fdfs[c]["soh_pct"].iloc[-1]) for c in _ccids if c in active_fdfs]
            _fades = [float(active_fdfs[c]["fade_rate_30cy"].iloc[-1]) * 1000
                      for c in _ccids if c in active_fdfs and "fade_rate_30cy" in active_fdfs[c].columns]
            if not _sohs:
                continue
            _cohort_stats.append({
                "cohort": _ctag, "n": len(_sohs),
                "avg_soh": sum(_sohs) / len(_sohs),
                "avg_fade": sum(_fades) / len(_fades) if _fades else None,
                "cells": ", ".join(_ccids),
            })
        if len(_cohort_stats) >= 2:
            _coh_colours = ["#63b3ed", "#48bb78", "#f6ad55", "#fc8181", "#9f7aea"]

            _fig_coh = go.Figure()
            for _ci, _cs in enumerate(_cohort_stats):
                _cc = _coh_colours[_ci % len(_coh_colours)]
                _fig_coh.add_trace(go.Bar(
                    name=_cs["cohort"], x=[_cs["cohort"]], y=[_cs["avg_soh"]],
                    marker_color=_cc, width=0.5,
                    text=[f"{_cs['avg_soh']:.1f}%"], textposition="outside",
                    hovertemplate=f"<b>{_cs['cohort']}</b><br>Avg SOH: %{{y:.1f}}%<br>Cells: {_cs['cells']}<extra></extra>",
                ))
            _fig_coh.add_hline(y=80, line_dash="dot", line_color="#fc8181", line_width=1)
            _fig_coh.update_layout(
                height=260, showlegend=False,
                **base_layout(
                    xaxis=dict(title="Cohort", gridcolor="#232d3b", linecolor="#2d3748"),
                    yaxis=dict(title="Avg SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                               zeroline=False, range=[50, 105]),
                ),
            )
            st.plotly_chart(_fig_coh, use_container_width=True)

            # ── Cohort SOH trajectories (temporal view) ──
            _cohort_cids = {_cs["cohort"]: [c for c in _cohorts[_cs["cohort"]] if c in active_fdfs]
                            for _cs in _cohort_stats}
            _has_temporal_data = any(
                "cycle_number" in active_fdfs[c].columns and "soh_pct" in active_fdfs[c].columns
                for _cids in _cohort_cids.values() for c in _cids
            )
            if _has_temporal_data:
                st.markdown("<div class='section-header'>Cohort SOH Trajectories</div>", unsafe_allow_html=True)
                st.caption("Bold line = cohort average SOH by cycle number. Thin lines = individual member cells.")
                _fig_traj = go.Figure()
                for _ci, _cs in enumerate(_cohort_stats):
                    _cc = _coh_colours[_ci % len(_coh_colours)]
                    _ctag = _cs["cohort"]
                    _member_dfs = [
                        active_fdfs[c] for c in _cohort_cids[_ctag]
                        if "cycle_number" in active_fdfs[c].columns and "soh_pct" in active_fdfs[c].columns
                    ]
                    if not _member_dfs:
                        continue

                    if len(_member_dfs) == 1:
                        _df = _member_dfs[0]
                        _fig_traj.add_trace(go.Scatter(
                            x=_df["cycle_number"], y=_df["soh_pct"],
                            mode="lines", name=_ctag,
                            line=dict(color=_cc, width=2.5),
                            hovertemplate=f"<b>{_ctag}</b> cy %{{x}}: %{{y:.1f}}%<extra></extra>",
                        ))
                        continue

                    # Multi-cell cohort: bin cycle_number and average SOH per bin,
                    # plus thin overlay lines per member for visual spread.
                    import numpy as _np_coh
                    for _df in _member_dfs:
                        _fig_traj.add_trace(go.Scatter(
                            x=_df["cycle_number"], y=_df["soh_pct"],
                            mode="lines", line=dict(color=_cc, width=1),
                            opacity=0.25, showlegend=False, hoverinfo="skip",
                        ))

                    _cmin = min(_df["cycle_number"].min() for _df in _member_dfs)
                    _cmax = max(_df["cycle_number"].max() for _df in _member_dfs)
                    if _cmax > _cmin:
                        _edges = _np_coh.linspace(_cmin, _cmax, 31)  # 30 bins
                        _binned_rows = []
                        for _df in _member_dfs:
                            _tmp = _df[["cycle_number", "soh_pct"]].copy()
                            _tmp["bin"] = pd.cut(_tmp["cycle_number"], bins=_edges, include_lowest=True)
                            _binned_rows.append(_tmp)
                        _all_binned = pd.concat(_binned_rows)
                        _avg_by_bin = _all_binned.groupby("bin", observed=True)["soh_pct"].mean().dropna()
                        _bin_x = [iv.mid for iv in _avg_by_bin.index]
                        _bin_y = _avg_by_bin.values
                    else:
                        # All members share one cycle value — nothing to bin.
                        _bin_x = [_cmin]
                        _bin_y = [sum(_df["soh_pct"].mean() for _df in _member_dfs) / len(_member_dfs)]

                    _fig_traj.add_trace(go.Scatter(
                        x=_bin_x, y=_bin_y,
                        mode="lines", name=_ctag,
                        line=dict(color=_cc, width=2.5),
                        hovertemplate=f"<b>{_ctag}</b> (avg) cy %{{x:.0f}}: %{{y:.1f}}%<extra></extra>",
                    ))

                _fig_traj.update_layout(
                    height=300,
                    **base_layout(
                        xaxis=dict(title="Cycle Number", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                        yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                                   zeroline=False, range=[50, 105]),
                    ),
                )
                st.plotly_chart(_fig_traj, use_container_width=True)

            # Cohort comparison narrative
            _best  = max(_cohort_stats, key=lambda c: c["avg_soh"])
            _worst = min(_cohort_stats, key=lambda c: c["avg_soh"])
            if _best["cohort"] != _worst["cohort"]:
                _gap = _best["avg_soh"] - _worst["avg_soh"]
                _md_html(
                    f"<div style='font-size:12px;color:#a0aec0;padding:4px 0 12px'>"
                    f"<strong style='color:#e2e8f0'>{_worst['cohort']}</strong> is degrading "
                    f"faster than <strong style='color:#e2e8f0'>{_best['cohort']}</strong> — "
                    f"{_gap:.1f}% average SOH gap. Possible batch, supplier, or operating condition difference.</div>"
                )
    else:
        _empty_state(
            "No cohort tags set yet",
            "Tag cells above by batch, supplier, or site to compare cohort health.",
            icon="🏷",
        )


def _cell_source(cell_id: str) -> str:
    """Coarse data-source tag used to block physically-meaningless mixed-source packs."""
    if cell_id in NASA_CELL_IDS:
        return "nasa"
    if cell_id.startswith(SEVERSON_CELL_PREFIX):
        return "severson"
    return "synthetic"


def _page_pack_builder(cell_ids: list, active_fdfs: dict):
    """Virtual N-cell series/parallel pack simulation."""
    st.markdown("<div class='section-header'>Virtual Pack Builder</div>", unsafe_allow_html=True)
    _md_html(
        "<div style='font-size:12px;color:#8896a8;margin-bottom:10px'>"
        "Simulate a series or parallel pack from selected cells, using each cell's latest-cycle "
        "capacity, resistance, and SOH. Capacity and resistance scale differently across chemistries "
        "and sources — selections must come from a single data source (NASA, Severson, or synthetic)."
        "</div>"
    )

    selected = st.multiselect(
        "Cells for this pack", options=cell_ids, key="pack_builder_cells",
        help="Pick 2 or more cells from the same data source.",
    )
    if len(selected) < 2:
        _empty_state(
            "Select at least 2 cells",
            "Choose 2 or more cells from the same data source to build a virtual pack.",
            icon="🔋",
        )
        return

    sources = {_cell_source(c) for c in selected}
    if len(sources) > 1:
        _empty_state(
            "Mixed data sources selected",
            f"Selected cells span {', '.join(sorted(sources))} — capacity and resistance scales are "
            "not comparable across chemistries/sources. Choose cells from a single source.",
            "→ Narrow your selection to one source and try again.",
            "⚠",
        )
        return

    topology = st.radio(
        "Pack topology", ["Series", "Parallel"],
        horizontal=True, key="pack_builder_topology",
    )

    rows = []
    for cid in selected:
        df = active_fdfs.get(cid)
        if df is None or len(df) == 0:
            continue
        cap = float(df["capacity_ah"].iloc[-1]) if "capacity_ah" in df.columns else float("nan")
        res = float(df["resistance_ohm"].iloc[-1]) if "resistance_ohm" in df.columns else float("nan")
        soh = float(df["soh_pct"].iloc[-1]) if "soh_pct" in df.columns else float("nan")
        rows.append({"cell_id": cid, "capacity_ah": cap, "resistance_ohm": res, "soh_pct": soh})

    pdf = pd.DataFrame(rows).dropna(subset=["capacity_ah", "soh_pct"])
    if len(pdf) < 2:
        _empty_state(
            "Insufficient data",
            "Selected cells are missing capacity or SOH data at the latest cycle.",
            icon="⚠",
        )
        return

    _bottleneck_idx = pdf["capacity_ah"].idxmin()
    _bottleneck_cid = pdf.loc[_bottleneck_idx, "cell_id"]
    _bottleneck_soh = float(pdf.loc[_bottleneck_idx, "soh_pct"])
    _avg_soh_weighted = float((pdf["soh_pct"] * pdf["capacity_ah"]).sum() / pdf["capacity_ah"].sum())
    _spread = float(pdf["soh_pct"].max() - pdf["soh_pct"].min())

    if topology == "Series":
        pack_capacity = float(pdf["capacity_ah"].min())
    else:
        pack_capacity = float(pdf["capacity_ah"].sum())

    _has_resistance = pdf["resistance_ohm"].notna().all()
    if _has_resistance:
        if topology == "Series":
            pack_resistance = float(pdf["resistance_ohm"].sum())
        else:
            pack_resistance = float(1.0 / (1.0 / pdf["resistance_ohm"]).sum())
    else:
        pack_resistance = float("nan")

    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("Pack capacity", f"{pack_capacity:.2f} Ah")
    _mc2.metric("Pack resistance", f"{pack_resistance*1000:.1f} mΩ" if _has_resistance else "N/A")
    _mc3.metric(
        "Bottleneck-cell SOH" if topology == "Series" else "Capacity-weighted avg SOH",
        f"{_bottleneck_soh:.1f}%" if topology == "Series" else f"{_avg_soh_weighted:.1f}%",
    )
    _mc4.metric("SOH spread (member cells)", f"{_spread:.1f}%")

    st.caption(
        f"Pack SOH is reported two ways since which framing is \"correct\" is a modelling choice: "
        f"bottleneck-cell SOH ({_bottleneck_soh:.1f}%) — meaningful for series packs, where usable "
        f"capacity is gated by the weakest cell — and capacity-weighted average SOH "
        f"({_avg_soh_weighted:.1f}%) — meaningful for parallel packs, where capacity sums across cells."
    )

    _md_html(
        f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
        f"padding:14px 18px;margin-top:8px'>"
        f"<strong style='color:#fc8181'>Weakest cell: {_bottleneck_cid}</strong> — "
        f"{'this cell caps the pack usable capacity in series topology.' if topology == 'Series' else 'lowest individual capacity contribution in parallel topology.'}"
        f"</div>"
    )

    with st.expander("Per-cell breakdown"):
        st.dataframe(pdf.set_index("cell_id"), use_container_width=True)
