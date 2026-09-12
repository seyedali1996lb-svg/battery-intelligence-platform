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
    # v12: RUL metrics can legitimately be NaN in-memory (no observed-EOL
    # rows to score) or None after a SQLite roundtrip (SQLite stores NaN as
    # NULL). Both render as "—", never as "nan".
    if v is None:
        return "—"
    try:
        if v != v:  # NaN != NaN
            return "—"
    except TypeError:
        pass
    return f"{v:.{decimals}f}"


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
            "Model":      ("PINN" if (r.get("model_kind") == "pinn") else "GBRT"),
            "Cells":      r["n_cells"],
            "SOH MAE":    _fmt(r["soh_mae"]),
            "SOH R2":     _fmt(r["soh_r2"]),
            "Baseline R2": _fmt(r.get("baseline_soh_r2")),
            "Model +vs base": _fmt((r["soh_r2"] - r["baseline_soh_r2"]) if (r.get("soh_r2") is not None and r.get("baseline_soh_r2") is not None) else None, 3),
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
        "shared platform reference-dataset runs (NASA/synthetic/Severson). "
        "Where a run has a baseline R², the 'Model +vs base' column shows the "
        "model's real advantage over a trivial cycle-number->SOH linear fit under "
        "the same leave-cell-out folds — most of a raw R² is smooth aging curves, "
        "not model skill. Runs are labelled by **Model**: GBRT is the production "
        "model; PINN is the physics-regularized estimator benchmarked through "
        "the same folds (see the model-comparison table below)."
    )

    # ── Accuracy by source: model R² vs the trivial baseline ────────────────
    # Grouped by (dataset, chemistry) rather than chemistry alone — the
    # synthetic fleet and the NASA PCoE cells share the chemistry "LiCoO2",
    # so a chemistry-only grouping would conflate a simulated fleet with real
    # measured cells. Only LCO runs carrying a baseline are shown (a
    # cross-chemistry transfer run has no baseline by construction).
    st.markdown("#### Accuracy by chemistry / source")
    acc_rows = reg.accuracy_by_source(tenant_org_id=org_id)
    # Which model SELECTION would pick for each chemistry — the row with the
    # largest genuine advantage, not the largest raw R². Reported from the same
    # registry this table is built on, so the rule that answers a cell is
    # auditable instead of implicit (src/model_selection.py).
    from model_selection import per_chemistry_accuracy as _per_chem
    _selected_by_chem = {
        r["chemistry"]: r["selected_key"] for r in _per_chem(tenant_org_id=org_id)
    }
    if not acc_rows:
        st.caption(
            "No LCO run with a baseline logged yet — this table fills in once "
            "a reference fleet or an uploaded dataset has trained on this "
            "deployment."
        )
    else:
        from batlab.validation.bootstrap import format_ci as _fmt_ci
        acc_table = pd.DataFrame([
            {
                "Source":          a["dataset"],
                "Chemistry":       a["chemistry"],
                "Selected":        ("✓ best skill" if _selected_by_chem.get(a["chemistry"]) == a["dataset"] else ""),
                "Cells":           a["n_cells"],
                "Model R2":        _fmt(a["soh_r2"]),
                "SOH 95% CI":      _fmt_ci((a.get("ci_intervals") or {}).get("soh_r2")),
                "Baseline R2":     _fmt(a["baseline_soh_r2"]),
                "Real advantage":  f"{a['advantage']:+.3f}",
                "RUL R2":          _fmt(a["rul_r2"]),
                "RUL 95% CI":      _fmt_ci((a.get("ci_intervals") or {}).get("rul_r2")),
                "RUL labels obs.": (
                    f"{a['rul_label_coverage'] * 100:.0f}%"
                    if a.get("rul_label_coverage") is not None else "—"
                ),
                "RUL reliable":    "✓" if a["rul_reliable"] else "—",
            }
            for a in acc_rows
        ])
        st.dataframe(acc_table, use_container_width=True, hide_index=True)
        st.caption(
            "**Selected** marks the model this platform would use to answer a new "
            "cell of that chemistry: the one with the largest **Real advantage**, not "
            "the largest raw R² (a chemistry whose aging curves are smooth would "
            "otherwise always win on R² alone). Where two sources share a chemistry "
            "— NASA and the synthetic fleet are both LiCoO2 — only one carries the "
            "mark, and a cell whose own source has a model always uses that one "
            "regardless of this ranking."
        )
        st.caption(
            "**Model R2** is the GBRT's leave-cell-out R² on cells it never saw. "
            "The **95% CI** brackets are fold-level bootstrap intervals: the "
            "aggregate is a mean over leave-one-cell-out folds, and the bracket's "
            "width IS part of the accuracy claim — a wide interval on a 4-cell "
            "fleet means thin evidence, stated plainly (resampling unit is the "
            "cell, not the row; RUL intervals describe only the folds with "
            "measured labels). **Baseline R2** is a trivial linear fit of "
            "cycle_number → SOH under the *identical* folds — what a dumb "
            "straight line already explains. **Real advantage** is the difference: "
            "how much the model and its engineered features actually earn. "
            "**RUL R2** is scored only on rows whose cell actually reached "
            "end-of-life inside its recorded window (measured labels); **RUL "
            "labels obs.** shows what fraction of each population's RUL rows "
            "that is — a RUL R² on a population with 0% observed labels does "
            "not exist and is shown as —, never extrapolated to look complete."
        )
        st.caption(
            "Reading the baseline honestly: a **positive** baseline means aging "
            "curves are smooth enough that a straight line captures much of the "
            "variance, so part of the model's raw R² is the shape of aging, not "
            "model skill (NASA is the clearest case). A **negative** baseline means "
            "a single global line is worse than just predicting the mean — the "
            "cells have genuinely different fade rates — so the model's advantage "
            "there reflects real cross-cell prediction. Either way, the advantage "
            "column, not the raw R², is the honest measure of learned skill. "
            "Sample sizes (the Cells column) are small — NASA's LCO rests on 4 "
            "held-out cells — so treat these as directional, not fleet-scale "
            "guarantees."
        )

    # ── Model kind: GBRT vs PINN on identical folds ─────────────────────────
    # The GBRT had an honest published number; the physics-regularized PINN
    # did not, so "which model should we use?" was answered by assertion.
    # Both are now fitted and evaluated by the same leave-cell-out harness and
    # the same trivial baseline, so the only difference is the model itself —
    # and the table reports the result even when the PINN loses.
    st.markdown("#### Model comparison — production GBRT vs physics-regularized PINN")
    mk_rows = reg.model_kind_comparison(tenant_org_id=org_id)
    if not mk_rows:
        st.caption(
            "No model-kind comparison logged yet — the PINN is benchmarked "
            "through the same leave-cell-out harness as the GBRT, and this "
            "table fills in automatically once the reference fleets have trained."
        )
    else:
        gbrt_rows = [r for r in mk_rows if r["model_kind"] == "gbrt"]
        pinned = [r for r in gbrt_rows if r.get("has_counterpart")]
        mk_table = pd.DataFrame([
            {
                "Source":       r["dataset"],
                "Chemistry":    r["chemistry"],
                "Model":        r["model_label"],
                "Cells":        r["n_cells"],
                "SOH R2":       _fmt(r["soh_r2"]),
                "SOH MAE (%)":  _fmt(r["soh_mae"]),
                "Baseline R2":  _fmt(r.get("baseline_soh_r2")),
                "Real adv.":    _fmt(r.get("advantage")),
                "RUL R2":       _fmt(r["rul_r2"]),
                "RUL MAE (cy)": _fmt(r["rul_mae"], 1),
                "vs GBRT":      (_fmt(r["pinn_minus_gbrt_soh_r2"])
                                 if r.get("pinn_minus_gbrt_soh_r2") is not None else "—"),
            }
            for r in mk_rows
        ])
        st.dataframe(mk_table, use_container_width=True, hide_index=True)
        st.caption(
            "**Baseline R2** and the leave-cell-out folds are IDENTICAL for both "
            "rows — the trivial `cycle_number → SOH` baseline is model-independent, "
            "so it is reused rather than recomputed. The only difference between "
            "the two numbers is the model. **vs GBRT** is the PINN's SOH R² minus "
            "the GBRT's; a negative value means the physics estimator lost on "
            "held-out cells, and is reported as-is."
        )
        st.caption(
            "Scope of the PINN row: its degradation law is shared across cells "
            "and fitted on the *training* cells only, with each cell anchored at "
            "its own first observed SOH (`s0 - β_sei·√n - β_lam·n^γ`). Its RUL is a "
            "projection of that fitted curve to the 80% EOL threshold, which can "
            "fall outside the observed window — an extrapolation, not a measured "
            "miss. So treat a weak PINN number as 'the physics shape fits the "
            "population poorly', not as a defect in the harness. The GBRT is the "
            "production model; this table exists so the choice is measured, and it "
            "currently reports the GBRT ahead."
        )
        if not pinned:
            st.caption(
                "No dataset has both model kinds logged yet, so no head-to-head "
                "delta is available — a single-model row carries no comparison."
            )

    # ── Cross-chemistry transfer: the counterexample ────────────────────────
    # Deliberately sits immediately after the per-chemistry LCO table, with
    # no divider: that table says "on a held-out cell of the SAME chemistry,
    # the model is this good"; this one says what happens on a cell of a
    # DIFFERENT chemistry the model has never seen a single row of. The gap
    # between the two tables is the real cost of "a cell I haven't seen",
    # and it is the single most informative accuracy number on the platform
    # — so it must be impossible to miss, not tucked into a one-off study.
    st.markdown("#### Cross-chemistry transfer — will the model work on an unseen cell?")
    xfer_rows = reg.cross_chemistry_benchmark(tenant_org_id=org_id)
    if not xfer_rows:
        st.caption(
            "No cross-chemistry transfer study logged yet — this table fills "
            "in automatically once the reference fleets have trained."
        )
    else:
        xfer_table = pd.DataFrame([
            {
                "Train → Eval":  f"{x['train_dataset']} → {x['eval_dataset']}",
                "Chemistry":     x["chemistry"],
                "SOH R2":        _fmt(x["soh_r2"]),
                "SOH MAE (%)":   _fmt(x["soh_mae"]),
                "RUL MAE (cy)":  _fmt(x["rul_mae"], 1),
                "RUL R2":        _fmt(x["rul_r2"]),
                "Shared feats":  str(x["n_features"]) if x["evaluated"] else "—",
                "Status":        "evaluated" if x["evaluated"] else "not evaluated",
            }
            for x in xfer_rows
        ])
        st.dataframe(xfer_table, use_container_width=True, hide_index=True)
        st.caption(
            "**Method (not the same thing as the table above):** one GBRT model "
            "is trained on *all* cells of the training domain — no cell is held "
            "out, because there is no same-chemistry held-out cell to spare — "
            "then evaluated, unmodified, on *all* cells of a different "
            "chemistry it has never seen a single row of. It may only use the "
            "feature columns both domains actually have (the *Shared feats* "
            "column), so a smaller number there is a weaker model, not a "
            "cleaner comparison. Rows are sorted worst-first, and a negative R² "
            "means the transferred model is **worse than predicting the "
            "training domain's mean**."
        )
        st.caption(
            "**Why this matters more than any number above it:** a laboratory "
            "leave-cell-out R² answers \"does this generalize to another cell "
            "of the same chemistry?\" This table answers the question a buyer "
            "actually asks — \"will this work on *my* cells?\" — and the honest "
            "answer for a chemistry the model wasn't trained on is: not yet. "
            "The NASA → Severson result (LiCoO2 → LFP) is the clearest "
            "counterexample on the platform and the reason per-chemistry "
            "reporting above is not interchangeable with a general accuracy "
            "claim. See `src/battery_knowledge.py`'s why-resistance-scales-differ "
            "entry for the mechanism behind it."
        )
        for _x in xfer_rows:
            if not _x["evaluated"] and _x.get("notes"):
                st.caption(
                    f"**{_x['train_dataset']} → {_x['eval_dataset']} — not evaluated:** {_x['notes']}"
                )

    # ── Prospective split: forecasting vs curve-fitting ──────────────────
    # The evaluation the LCO tables cannot perform: withhold the FUTURE. A
    # model trained on the first half of each cell's life is scored only on
    # the second half — the deployment question. The gap between the LCO
    # number and this one is the interpolation the LCO setup was silently
    # earning, so it sits here as its own section rather than a footnote.
    st.markdown("#### Prospective split — does the model forecast, or only curve-fit?")
    prosp_rows = reg.prospective_benchmark(tenant_org_id=org_id)
    if not prosp_rows:
        st.caption(
            "No prospective-split study logged yet — this table fills in "
            "automatically once the reference fleets have trained."
        )
    else:
        from batlab.validation.bootstrap import format_ci as _fmt_ci
        prosp_table = pd.DataFrame([
            {
                "Dataset":       p["dataset"],
                "Chemistry":     p["chemistry"],
                "Train fraction": (
                    f"{p['train_fraction']:.0%}" if p.get("train_fraction") is not None else "—"
                ),
                "Cells":         p["n_cells"],
                "SOH R2":        _fmt(p["soh_r2"]),
                "SOH 95% CI":    _fmt_ci((p.get("ci_intervals") or {}).get("soh_r2")),
                "LCO SOH R2":    _fmt(p.get("lco_soh_r2")),
                "Forecasting gap": (
                    f"{p['forecasting_gap']:+.3f}" if p.get("forecasting_gap") is not None else "—"
                ),
                "Trend baseline": _fmt(p.get("baseline_soh_r2")),
                "RUL R2":        _fmt(p["rul_r2"]),
                "Formula base.": _fmt(p.get("rul_formula_baseline_r2")),
                "RUL labels obs.": (
                    f"{p['rul_label_coverage'] * 100:.0f}%"
                    if p.get("rul_label_coverage") is not None else "—"
                ),
            }
            for p in prosp_rows
        ])
        st.dataframe(prosp_table, use_container_width=True, hide_index=True)
        st.caption(
            "**Method:** the model trains ONLY on the first half of each cell's "
            "recorded cycles and is scored ONLY on the remainder — it never "
            "sees a cycle from the window it is evaluated on. This is NOT "
            "leave-cell-out (the same cell supplies its early cycles to "
            "training and its late cycles to testing); it is the deployment "
            "question — \"what happens NEXT for a cell we have history for?\" — "
            "which no held-out-cell evaluation can answer. **Forecasting gap** "
            "is prospective minus LCO: negative means the model loses that much "
            "skill when denied the future, i.e. how much of the LCO number was "
            "interpolation rather than forecasting. **Trend baseline** is a "
            "per-cell straight line fit on the train window and extrapolated "
            "forward — the null hypothesis here; a model below it has learned "
            "nothing beyond extending the early trend. **Formula base.** is the "
            "RUL closed form under the identical split. A negative SOH R² means "
            "the model is worse than predicting the training window's mean on "
            "the future — reported as-is, because that IS the forecasting skill."
        )

    # ── Fold-level drill-down ─────────────────────────────────────────────
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
