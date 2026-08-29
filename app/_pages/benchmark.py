"""Page: Benchmark — leaderboard across every logged experiment run.

Fleet-wide, not cell-specific (same shape as page_fleet()) -- this reads
the experiment registry (src/experiment_registry.py), not the currently
selected cell's bundle. Every real GBRT training run (built-in NASA/
synthetic/Severson reference fleets, plus this org's own uploaded-data
runs) is logged automatically at fit time -- see experiment_registry.py's
module docstring for why there is no manual "log this run" step.
"""

import json

import _paths  # noqa: F401
import streamlit as st
import pandas as pd

from utils import _action_bar, _empty_state, render_card

_SORT_OPTIONS = {
    "RUL MAE (cycles)": "rul_mae",
    "RUL R2":           "rul_r2",
    "SOH MAE (%)":      "soh_mae",
    "SOH R2":           "soh_r2",
    "Logged":           "timestamp",
}


def _fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "—"


def page_benchmark(org_id: int) -> None:
    import experiment_registry as reg

    _action_bar("benchmark")
    st.markdown("# Model Benchmark")
    st.markdown(
        "#### Every logged GBRT training run — filterable, sortable, "
        "with fold-level drill-down"
    )

    all_runs = reg.leaderboard(tenant_org_id=org_id)
    if not all_runs:
        _empty_state(
            "No experiment runs logged yet",
            "Runs are logged automatically the first time each dataset "
            "trains (or retrains) on this deployment — nothing to show "
            "until that first fit happens.",
        )
        return

    datasets    = sorted({r["dataset"] for r in all_runs})
    chemistries = sorted({r["chemistry"] for r in all_runs if r["chemistry"]})

    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
    with col_f1:
        dataset_filter = st.selectbox("Dataset", ["All"] + datasets, key="bench_dataset_filter")
    with col_f2:
        chem_filter = st.selectbox("Chemistry", ["All"] + chemistries, key="bench_chem_filter")
    with col_f3:
        sort_label = st.selectbox("Sort by", list(_SORT_OPTIONS.keys()), key="bench_sort_col")
    with col_f4:
        ascending = st.selectbox(
            "Order", ["Ascending (best/lowest first)", "Descending (highest first)"],
            key="bench_sort_dir",
        ) == "Ascending (best/lowest first)"

    runs = reg.leaderboard(
        tenant_org_id=org_id,
        dataset=None if dataset_filter == "All" else dataset_filter,
        chemistry=None if chem_filter == "All" else chem_filter,
        sort_by=_SORT_OPTIONS[sort_label],
        ascending=ascending,
    )

    if not runs:
        _empty_state("No runs match these filters", "Try a different dataset or chemistry.")
        return

    table = pd.DataFrame([
        {
            "Run ID":     r["run_id"],
            "Dataset":    r["dataset"],
            "Chemistry":  r["chemistry"] or "—",
            "Cells":      r["n_cells"],
            "SOH MAE":    _fmt(r["soh_mae"]),
            "SOH R2":     _fmt(r["soh_r2"]),
            "RUL MAE":    _fmt(r["rul_mae"], 1),
            "RUL R2":     _fmt(r["rul_r2"]),
            "Reliable":   "✓" if r["rul_reliable"] else "—",
            "Logged":     (r["timestamp"] or "")[:19],
            "Commit":     r["git_commit"] or "—",
        }
        for r in runs
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        f"{len(runs)} run(s) · scope: this org's uploaded-data runs + the "
        "shared platform reference-dataset runs (NASA/synthetic/Severson)"
    )

    # ── Fold-level drill-down ───────────────────────────────────────────────
    st.markdown("#### Fold-level drill-down")
    run_labels = {f"{r['run_id']}  ·  {r['dataset']} ({r['chemistry'] or '—'})": r["run_id"] for r in runs}
    picked_label = st.selectbox("Select a run", list(run_labels.keys()), key="bench_drilldown_run")
    run = next(r for r in runs if r["run_id"] == run_labels[picked_label])

    render_card(
        f"<div style='font-size:12px;color:#8896a8;line-height:1.9'>"
        f"<b>Dataset:</b> {run['dataset']} &nbsp;·&nbsp; "
        f"<b>Chemistry:</b> {run['chemistry'] or '—'} &nbsp;·&nbsp; "
        f"<b>Seed:</b> {run['seed']} &nbsp;·&nbsp; "
        f"<b>Feature version:</b> {run['feature_version']}<br>"
        f"<b>Cells ({run['n_cells']}):</b> {', '.join(run['cell_ids']) or '—'}<br>"
        f"<b>Features used ({len(run['feature_set'])}):</b> {', '.join(run['feature_set']) or '—'}"
        + (f"<br><b>Notes:</b> {run['notes']}" if run.get("notes") else "")
        + "</div>",
        padding="14px 18px",
    )

    fold_metrics = run.get("fold_metrics") or {}
    if fold_metrics:
        fold_table = pd.DataFrame([
            {
                "Cell":    cell_id,
                "SOH MAE": _fmt(fold.get("soh_mae")),
                "SOH R2":  _fmt(fold.get("soh_r2")),
                "RUL MAE": _fmt(fold.get("rul_mae"), 1),
                "RUL R2":  _fmt(fold.get("rul_r2")),
            }
            for cell_id, fold in fold_metrics.items()
        ])
        st.dataframe(fold_table, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No per-cell fold breakdown for this run — expected for a "
            "cross-dataset transfer run (a single train/eval split, not "
            "leave-cell-out)."
        )

    # ── Auto-generated model card (P2) ──────────────────────────────────────
    with st.expander("Model card (auto-generated per run)", expanded=False):
        try:
            import model_cards as mc
            card = mc.build_model_card(run)
        except Exception as exc:
            st.caption(f"Model card unavailable for this run: {exc}")
        else:
            st.markdown(mc.model_card_markdown(card))
            st.download_button(
                "Download model card (JSON)",
                data=json.dumps(card, indent=2, default=str),
                file_name=f"model_card_{run['run_id']}.json",
                mime="application/json",
                key=f"bench_model_card_{run['run_id']}",
            )

    # ── Physics vs GBRT held-out-cell divergence (Phase 6) ──────────────────
    _render_physics_divergence_section(run["dataset"])


def _render_physics_divergence_section(selected_dataset: str) -> None:
    """
    physics_calibration.physics_gbrt_divergence_report() compares, per
    calibration-eligible cell, a real leave-cell-out GBRT fold against a
    physics fit calibrated on that cell's own history. Gated behind a
    button (not auto-run on page load) since it retrains one GBRT model
    per eligible cell -- real cost, not free like the rest of this page's
    reads from the already-logged registry.

    Only offered for NASA/Severson (physics_calibration.py's own
    calibration-eligibility gate — see its module docstring for why Oxford/
    synthetic/uploaded cells are out of scope) and only when raw cell data
    is reloadable for the run's dataset (experiment_registry.
    reload_reference_cell_data() — the same reload path "Regenerate this
    report" already uses elsewhere in this app).
    """
    import experiment_registry as reg

    base_dataset = selected_dataset.split("_to_")[0]  # strip cross-chemistry "X_to_Y" suffixes
    if base_dataset not in ("nasa", "severson"):
        return

    st.markdown("#### Physics vs GBRT — held-out-cell divergence")
    st.caption(
        "Compares a genuine leave-cell-out GBRT fold against a physics fit "
        "calibrated directly on that same cell's own history — surfaced "
        "honestly, including any disagreement, per this project's standing "
        "practice of never suppressing model disagreement (see the "
        "Overview page's RUL reconciliation)."
    )
    if st.button("Run held-out-cell divergence check", key=f"bench_divergence_{base_dataset}"):
        with st.spinner("Running leave-cell-out folds + physics calibration per cell…"):
            try:
                cell_data = reg.reload_reference_cell_data(base_dataset)
                from physics_calibration import physics_gbrt_divergence_report
                report = physics_gbrt_divergence_report(cell_data)
            except Exception as exc:
                st.info(f"Divergence check unavailable: {exc}")
                return
        if not report:
            st.caption("No calibration-eligible cells with enough data in this dataset.")
            return
        div_table = pd.DataFrame([
            {
                "Cell":              r["cell_id"],
                "Physics dominant mode": r["physics_dominant_mode_label"].split("(")[0].strip(),
                "Physics fit R2":    _fmt(r["physics_fit_r2"]),
                "GBRT SOH MAE (%)":  _fmt(r["gbrt_soh_mae"]),
                "Physics SOH MAE (%)": _fmt(r["physics_soh_mae"]),
                "Closer to actual":  r["closer_model"],
                "Divergence":        f"{r['divergence_pct']:.0f}%",
            }
            for r in report
        ])
        st.dataframe(div_table, use_container_width=True, hide_index=True)
        st.caption(report[0]["note"])
