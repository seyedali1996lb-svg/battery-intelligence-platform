"""
Page: Bring your own model — grade any forecaster by this platform's rules.

This page is the validation harness (batlab.harness) with a UI in front of it.
You pick a reference fleet and a model — the platform's own GBRT, a cheap
linear baseline, a random forest, a training-mean floor, or a .py module you
upload — and it runs the SAME six checks every number on the Benchmark page was
measured under: feature-leakage lint, RUL label provenance, leave-cell-out
generalization, interval calibration, the prospective (same-cell forecasting)
split, and a declared-floors metric gate. What comes back is a verdict plus two
lists: the claims the run supports, and the claims it WITHHELD.

The withheld list is the point of the page. A model that cannot be scored on
this fleet's end-of-life evidence gets no RUL number at all — not a caveated
one, not a formula-extrapolated one. \"Not evaluable\" is an output here.

Why the fleet is a reference fleet and not your upload
------------------------------------------------------
Grading a model means re-deriving features from RAW cycles, and the app's
uploaded-data path retains only the engineered frames (src/bundle_cache.py),
not the raw cycles the harness needs. Rather than grade a model on frames it
cannot fingerprint, the page grades against fleets whose raw data is
reproducibly reloadable, and says so. Uploaded-data grading needs the raw
upload persisted first; that is a data-model change, not a page change.

Why an uploaded model can be read before it runs
------------------------------------------------
The uploader accepts exactly one thing: a .py module defining make_model().
Its source is shown to you and nothing executes until you tick an
acknowledgement. Serialized models (.pkl/.pt/.joblib/...) are REFUSED, with
the reason printed on the page: loading one runs code from the file as part of
loading it, so there is no moment at which it can be inspected first. An
uploaded module is also NOT sandboxed — it runs with this process's
privileges, which is stated on the page rather than glossed over.
"""

from __future__ import annotations

import json

import _paths  # noqa: F401
import streamlit as st
import pandas as pd

from utils import _action_bar, _empty_state, render_card
from harness_models import (
    BUILTIN_MODELS,
    ModelModuleError,
    STARTER_MODULE_SOURCE,
    available_fleets,
    builtin_model,
    example_gate_expectations,
    fleet_loader_record,
    load_model_module,
    seal_zip_bytes,
    summarise_report,
    unsupported_upload_reason,
)

_RESULT_KEY = "byom_result"

# Reference runtimes are stated so a user can decide BEFORE pressing Run, and
# they are MEASURED, not estimated: a progress-free few-minute button reads as
# a hang, and a made-up number is just a different kind of dishonesty. Measured
# for the page's default configuration (leave-cell-out + prospective +
# intervals + gate) on 2026-09-16. Fleets with no measurement are omitted
# rather than guessed at.
_RUNTIME_HINT = {
    "nasa": "≈15 s measured (4 real cells, measured end-of-life)",
    "synth": "≈1 min measured (8 deterministic cells)",
    "zhu2022": "≈1 min measured (9 real cells)",
    "severson": "several minutes (46 cells — the platform's largest fleet)",
}


def _fmt(value, decimals: int = 3) -> str:
    """None/NaN render as an em dash — never as \"nan\" (v12 convention)."""
    if value is None:
        return "—"
    try:
        if value != value:  # NaN != NaN
            return "—"
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _pct(value, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _read_upload_source(uploaded) -> tuple[str, "str | None"]:
    """Decode an uploaded file to text, or say why it cannot be shown."""
    try:
        return uploaded.getvalue().decode("utf-8"), None
    except UnicodeDecodeError:
        return "", (
            "This file is not UTF-8 text, so it is not a readable Python "
            "module. Only .py modules written as plain text are accepted."
        )


def _render_config_panel(fleets: list[dict]) -> dict:
    """Draw the setup widgets and return the chosen configuration."""
    usable = [f for f in fleets if f["available"]]
    unavailable = [f for f in fleets if not f["available"]]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1 · Fleet to grade on")
        if not usable:
            _empty_state(
                "No reference fleet is reloadable on this deployment",
                "Every fleet below needs its raw data cached locally. "
                "`python -m batlab.datasets.nasa` populates NASA; the synthetic "
                "fleet needs no download.",
            )
            return {}
        fleet_key = st.selectbox(
            "Fleet",
            options=[f["key"] for f in usable],
            format_func=lambda k: next(
                f["label"] + (f" — {f['n_cells']} cells" if f["n_cells"] else "")
                for f in usable if f["key"] == k
            ),
            key="byom_fleet",
            label_visibility="collapsed",
        )
        fleet = next(f for f in usable if f["key"] == fleet_key)
        st.caption(fleet["detail"])
        if _RUNTIME_HINT.get(fleet_key):
            st.caption(f"Expected runtime: {_RUNTIME_HINT[fleet_key]}")
        if unavailable:
            st.caption(
                "**Not reloadable here:** "
                + " · ".join(f"{f['label'].split(' — ')[0]} ({f['unavailable_reason']})"
                             for f in unavailable)
            )

    with c2:
        st.markdown("##### 2 · Model to grade")
        model_key = st.selectbox(
            "Model",
            options=list(BUILTIN_MODELS.keys()),
            format_func=lambda k: BUILTIN_MODELS[k]["label"],
            key="byom_model",
            label_visibility="collapsed",
        )
        st.caption(BUILTIN_MODELS[model_key]["detail"])

    fab = {"fleet_key": fleet_key if usable else None, "model_key": model_key}

    if model_key == "uploaded_module":
        # Deliberately NOT its own numbered step: an upload only exists in one
        # branch of step 2, and numbering it would leave steps 2 → 4 looking
        # like a missing section on every other model choice.
        uploaded = st.file_uploader(
            "Model module (.py)", type=["py"], key="byom_upload",
            help="A single Python file defining make_model().",
        )
        if uploaded is not None:
            refusal = unsupported_upload_reason(uploaded.name)
            if refusal:
                st.error(refusal)
                fab["upload_error"] = refusal
            else:
                source, decode_error = _read_upload_source(uploaded)
                if decode_error:
                    st.error(decode_error)
                    fab["upload_error"] = decode_error
                else:
                    fab["upload_source"] = source
                    with st.expander("Read the file before it runs (source)", expanded=True):
                        st.code(source, language="python")
                        st.caption(
                            "This is the exact text that will be executed inside "
                            "this app's process. **There is no sandbox** — an "
                            "uploaded module can read files, open sockets, and use "
                            "everything this app can reach. It is not limited to "
                            "model code, and nothing here inspects what it imports."
                        )
                    fab["upload_ack"] = st.checkbox(
                        "I have read this file and accept that it runs in this "
                        "app's process",
                        key="byom_upload_ack",
                    )
        st.download_button(
            "Download a starter module",
            data=STARTER_MODULE_SOURCE,
            file_name="my_model.py",
            mime="text/x-python",
            key="byom_starter_download",
            help="A documented template showing the three accepted model shapes.",
        )

    st.markdown("##### 3 · What to run")
    s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
    with s1:
        fab["splits_lco"] = st.checkbox(
            "Leave-cell-out (unseen cells)", value=True, key="byom_splits_lco",
            help="The new-cell generalization question. Needed to seal a bundle.",
        )
    with s2:
        fab["splits_prospective"] = st.checkbox(
            "Prospective (forecast the future)", value=True, key="byom_splits_prospective",
            help="Train on each cell's first half, score the second half. The "
                 "deployment question. Without it, interpolation can be mistaken "
                 "for forecasting.",
        )
    with s3:
        fab["intervals"] = st.checkbox(
            "Interval calibration", value=True, key="byom_intervals",
            help="Measure real coverage of a nominal 80% interval.",
        )
    with s4:
        fab["seed"] = int(st.number_input(
            "Seed", value=42, min_value=0, step=1, key="byom_seed",
        ))

    with st.expander("Metric gate — declare the floors this run must clear", expanded=False):
        st.caption(
            "The gate is the only thing that can turn a measured number into a "
            "FAIL. With no floors declared the verdict is **NOT CHECKED**, which "
            "is deliberately not a pass. The defaults below are floors the "
            "platform's own model clears on the NASA fleet — and which the "
            "training-mean baseline fails, so you can watch a gate fail on "
            "purpose."
        )
        fab["enforce_gate"] = st.checkbox(
            "Enforce these expectations", value=True, key="byom_enforce_gate",
        )
        gate_text = st.text_area(
            "Expectations (JSON)",
            value=json.dumps(example_gate_expectations(), indent=2),
            height=190,
            key="byom_gate_json",
            disabled=not fab["enforce_gate"],
        )
        fab["gate_text"] = gate_text

    return fab


def _run(cfg: dict) -> dict:
    """Reload the fleet, grade the model, seal the bundle. Returns a result."""
    import experiment_registry as reg
    from batlab.harness import validate_forecaster, format_report

    fleet_key = cfg["fleet_key"]
    splits = tuple(
        s for s, on in (("lco", cfg["splits_lco"]), ("prospective", cfg["splits_prospective"]))
        if on
    )
    if not splits:
        raise ValueError(
            "Select at least one split. With neither leave-cell-out nor "
            "prospective selected there is nothing to score the model on."
        )

    gate = None
    if cfg["enforce_gate"]:
        try:
            gate = json.loads(cfg["gate_text"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"The expectations are not valid JSON — line {exc.lineno}: {exc.msg}"
            ) from exc
        if not isinstance(gate, dict) or not gate.get("metrics"):
            raise ValueError(
                "Expectations must be a JSON object with a non-empty 'metrics' "
                "mapping, e.g. {\"metrics\": {\"soh_r2\": {\"floor\": 0.5}}}"
            )

    model = None
    if cfg["model_key"] == "uploaded_module":
        module = load_model_module(cfg["upload_source"])
        model = module["factory"]
        uploaded_probe = module
    else:
        entry = builtin_model(cfg["model_key"], seed=cfg["seed"])
        model = entry["model"]
        uploaded_probe = None

    cell_data = reg.reload_reference_cell_data(fleet_key)
    if len(cell_data) < 2:
        raise ValueError(
            f"This deployment reloaded only {len(cell_data)} cell(s) for "
            f"{fleet_key!r}. A leave-cell-out structure needs at least 2."
        )

    report = validate_forecaster(
        cell_data,
        model=model,
        seed=cfg["seed"],
        splits=splits,
        intervals=cfg["intervals"],
        gate=gate,
        dataset=fleet_key,
    )

    record = fleet_loader_record(fleet_key)
    zip_bytes = None
    seal_error = None
    if "lco" in splits:
        try:
            zip_bytes = seal_zip_bytes(
                report, cell_data, dataset=fleet_key,
                loader=record["loader"], loader_kwargs=record["loader_kwargs"],
            )
        except Exception as exc:  # sealing failing must not lose the report
            seal_error = f"{type(exc).__name__}: {exc}"

    return {
        "report": report,
        "summary": summarise_report(report),
        "report_text": format_report(report),
        "report_json": json.dumps(report, indent=2, default=str),
        "zip_bytes": zip_bytes,
        "seal_error": seal_error,
        "loader": record,
        "uploaded_module": uploaded_probe,
        "fleet_label": next(
            (f["label"] for f in available_fleets() if f["key"] == fleet_key), fleet_key
        ),
        "model_label": (
            BUILTIN_MODELS[cfg["model_key"]]["label"]
            if cfg["model_key"] != "uploaded_module"
            else f"uploaded module ({cfg.get('upload_name', 'module')})"
        ),
    }


def _render_verdict(summary: dict) -> None:
    status = summary["status"]
    if status == "pass":
        st.success(f"**{summary['status_label']}**")
    elif status == "fail":
        st.error(f"**{summary['status_label']}**")
    else:
        st.warning(f"**{summary['status_label']}**")

    render_card(
        f"<div style='font-size:12px;color:#a0aec0;line-height:2'>"
        f"<b style='color:#e2e8f0'>Model:</b> {summary['model_label']}<br>"
        f"<b style='color:#e2e8f0'>Fleet:</b> {summary['dataset_name']} "
        f"({summary['n_cells']} cells) · sha256 "
        f"<code>{(summary['dataset_sha256'] or '')[:16]}</code><br>"
        f"<b style='color:#e2e8f0'>Interval:</b> "
        f"{'interval-capable model' if summary['interval_capable'] else 'point-only model'} "
        f"— {summary['interval_source'] or 'n/a'}<br>"
        f"<b style='color:#e2e8f0'>Leakage lint:</b> "
        f"{'clean' if summary['lint_ok'] else 'violations found — every number below is suspect'}"
        f"</div>",
        padding="14px 18px",
    )


def _render_claims(summary: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Supported claims")
        if summary["claims"]:
            for claim in summary["claims"]:
                st.markdown(f"- {claim}")
            st.caption(
                "Each claim names the evaluation it came from. Compare them "
                "against the baselines section below before reading any single "
                "R² as skill — most of a raw R² is the shape of aging, not the "
                "model."
            )
        else:
            st.markdown("_None._ This model supports no accuracy claim on this fleet.")
    with c2:
        st.markdown("#### Withheld claims")
        if summary["withheld"]:
            for withheld in summary["withheld"]:
                st.markdown(f"- {withheld}")
            st.caption(
                "A withheld claim is a claim the harness refuses to make — not "
                "a computation that failed. The usual causes: no measured "
                "end-of-life rows in the fleet, a split that was not requested, "
                "or a label population too small for a conformal quantile."
            )
        else:
            st.markdown("_None._ Nothing the harness normally reports was withheld here.")
            st.caption(
                "That is only a statement about coverage of the checks, not "
                "about accuracy: check the calibration and prospective sections "
                "below for what the numbers actually mean."
            )


def _render_lco(summary: dict) -> None:
    lco = summary["lco"]
    if not isinstance(lco, dict):
        st.caption("Leave-cell-out was not part of this run.")
        return
    st.markdown("#### Leave-cell-out — unseen cells")
    headline = pd.DataFrame([{
        "SOH R²": _fmt(lco.get("soh_r2")),
        "SOH MAE (%)": _fmt(lco.get("soh_mae")),
        "RUL R²": _fmt(lco.get("rul_r2")),
        "RUL MAE (cy)": _fmt(lco.get("rul_mae"), 1),
        "RUL observed rows": lco.get("n_rul_observed_rows"),
        "RUL label coverage": _pct(lco.get("rul_label_coverage")),
        "RUL reliable": "yes" if lco.get("rul_reliable") else "no",
    }])
    st.dataframe(headline, use_container_width=True, hide_index=True)

    folds = lco.get("per_cell") or {}
    if folds:
        fold_table = pd.DataFrame([
            {
                "Held-out cell": cell,
                "SOH R²": _fmt(fold.get("soh_r2")),
                "SOH MAE (%)": _fmt(fold.get("soh_mae")),
                "RUL R² (observed)": _fmt(fold.get("rul_r2")),
                "RUL R² (extrapolated)": _fmt(fold.get("rul_r2_extrapolated")),
                "Observed labels": (fold.get("rul_label_kinds") or {}).get("observed"),
                "Extrapolated labels": (fold.get("rul_label_kinds") or {}).get("extrapolated"),
            }
            for cell, fold in folds.items()
        ])
        st.dataframe(fold_table, use_container_width=True, hide_index=True)
    st.caption(
        "Each row is one fold: a model fitted on the other cells and scored on "
        "this one. **RUL R² (extrapolated)** is reported for inspection and "
        "never enters the headline — those labels come from a closed-form fade "
        "formula, so scoring against them measures formula recovery, not "
        "prediction."
    )


def _render_baselines(summary: dict) -> None:
    base = summary["baselines"] or {}
    st.markdown("#### Baselines — what a dumb model already gets")
    if base.get("status") != "computed":
        st.caption(base.get("reason") or "Baselines were not computed for this run.")
        return
    rows = []
    for label, key in (
        ("Straight line through cycle number (LCO)", "lco_trend_r2"),
        ("Closed-form fade formula, RUL (LCO)", "lco_rul_formula_r2"),
        ("Straight line, prospective", "prospective_trend_r2"),
        ("Closed-form fade formula, RUL (prospective)", "prospective_rul_formula_r2"),
    ):
        if key in base:
            rows.append({"Baseline": label, "R²": _fmt(base.get(key))})
    for label, key in (
        ("Straight line (LCO)", "lco_trend_error"),
        ("Fade formula (LCO)", "lco_rul_formula_error"),
        ("Straight line (prospective)", "prospective_trend_error"),
        ("Fade formula (prospective)", "prospective_rul_formula_error"),
    ):
        if base.get(key):
            rows.append({"Baseline": f"{label} — unavailable", "R²": base[key]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(base.get("note", ""))


def _render_calibration(summary: dict) -> None:
    cal = summary["calibration"] or {}
    st.markdown("#### Interval calibration — is the 80% interval really 80%?")
    status = cal.get("status")
    if status in ("quantile_interval", "conformal_residual"):
        table = pd.DataFrame([{
            "Method": cal.get("label") or cal.get("method"),
            "Nominal": _pct(cal.get("nominal")),
            "Raw coverage": _pct(cal.get("raw_coverage")),
            "Conformal correction E*": _fmt(cal.get("e_star_cycles"), 2),
            "Measured coverage": _pct(cal.get("calibrated_coverage")),
            "Mean width (cycles)": _fmt(cal.get("mean_width_cycles"), 1),
            "Calibration folds": cal.get("n_calibration_folds", len(cal.get("per_cell") or {})),
        }])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            f"**Method:** {cal.get('method')} "
            "**Raw coverage** is what the interval achieved before correction "
            "(— when the model had no interval head at all); **Measured "
            "coverage** is what the corrected interval actually covers on cells "
            "the model never trained on."
        )
        st.caption(f"**Scope of this number:** {cal.get('scope')}")
        if cal.get("skipped"):
            st.caption(
                "Folds skipped rather than padded: "
                + "; ".join(f"{c} ({why})" for c, why in (cal["skipped"] or {}).items())
            )
    else:
        st.caption(f"Not measured: {cal.get('reason', 'interval calibration did not run')}")


def _render_prospective(summary: dict) -> None:
    pro = summary["prospective"] or {}
    st.markdown("#### Prospective — does the model forecast, or only curve-fit?")
    if pro.get("status") != "computed":
        st.caption(pro.get("reason", "The prospective split was not part of this run."))
        return
    table = pd.DataFrame([{
        "Train fraction": _pct(pro.get("train_fraction"), 0),
        "Cells evaluated": pro.get("n_cells_evaluated"),
        "Cells skipped": pro.get("n_cells_skipped"),
        "SOH R² (future window)": _fmt(pro.get("soh_r2")),
        "SOH MAE (%)": _fmt(pro.get("soh_mae")),
        "RUL R² (observed)": _fmt(pro.get("rul_r2")),
        "RUL label coverage": _pct(pro.get("rul_label_coverage")),
        "Train rows": pro.get("n_train_rows"),
        "Test rows": pro.get("n_test_rows"),
    }])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "The model trains only on each cell's early cycles and is scored only "
        "on the remainder — a window it never saw a row of. A negative SOH R² "
        "here means the model is worse than predicting the training window's "
        "mean on the future, which is what the number says."
    )


def _render_provenance(summary: dict) -> None:
    prov = summary["provenance"] or {}
    lint = summary["lint_violations"]
    st.markdown("#### Provenance and leakage checks")
    st.dataframe(pd.DataFrame([
        {"Check": "Feature-leakage lint", "Result": "clean" if summary["lint_ok"] else "VIOLATIONS"},
        {"Check": "RUL rows with measured labels", "Result": str(prov.get("observed_rows", "—"))},
        {"Check": "RUL rows with formula-extrapolated labels",
         "Result": str(prov.get("extrapolated_rows", "—"))},
        {"Check": "Observed label fraction", "Result": _pct(prov.get("observed_fraction"))},
        {"Check": "RUL headline evaluable", "Result": "yes" if prov.get("rul_reliable") else "no"},
    ]), use_container_width=True, hide_index=True)
    if lint:
        for violation in lint:
            st.error(str(violation))
    elif summary["lint_ok"]:
        st.caption(
            "The lint passed: no quantity read by the RUL label's generating "
            "expression is also a model feature."
        )
    if prov.get("note"):
        st.caption(prov["note"])


def _render_gate(summary: dict) -> None:
    gate = summary["gate"] or {}
    st.markdown("#### Metric gate — declared floors, enforced")
    verdict = gate.get("verdict")
    if verdict == "not_checked":
        st.warning(gate.get("reason", "No expectations were declared."))
        observed = gate.get("observed") or {}
        if observed:
            st.dataframe(pd.DataFrame([
                {"Measured metric": name, "Value": _fmt(value),
                 "Floor": "none declared", "Verdict": "UNTRACKED"}
                for name, value in observed.items()
            ]), use_container_width=True, hide_index=True)
        return
    rows = [
        {
            "Metric": r.get("name"),
            "Observed": _fmt(r.get("observed")),
            "Floor": _fmt(r.get("floor")),
            "Ceiling": _fmt(r.get("ceiling")),
            "Verdict": str(r.get("verdict", "")).upper(),
            "Detail": r.get("detail"),
        }
        for r in (gate.get("results") or []) + (gate.get("untracked") or [])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if verdict == "fail":
        st.error(
            "A declared expectation was violated. The number did not move, the "
            "expectation did not move, and this run says so instead of "
            "reporting a metric it did not meet."
        )
    st.caption(
        "UNTRACKED means a measured metric had no declared floor — reported "
        "rather than passed, because a headline number nobody gave a floor to "
        "is a number nobody is checking."
    )


def _render_downloads(result: dict) -> None:
    st.markdown("#### Take the evidence with you")
    summary = result["summary"]
    record = result["loader"]
    if result["zip_bytes"]:
        st.download_button(
            "Download the sealed bundle (.zip)",
            data=result["zip_bytes"],
            file_name=f"sealed_bundle_{summary['dataset_name']}.zip",
            mime="application/zip",
            key="byom_download_zip",
            help="benchmark.json + harness_report.json + replication.json, with "
                 "per-cell digests and the model identity sealed in.",
        )
    elif result["seal_error"]:
        st.info(
            "The run completed but could not be sealed: "
            f"{result['seal_error']} The report below is unaffected."
        )
    else:
        st.caption(
            "No sealed bundle for this run: sealing needs the leave-cell-out "
            "section, whose number the bundle's recompute check re-derives."
        )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download the report (JSON)",
            data=result["report_json"],
            file_name=f"harness_report_{summary['dataset_name']}.json",
            mime="application/json",
            key="byom_download_json",
        )
    with d2:
        st.download_button(
            "Download the report (text)",
            data=result["report_text"],
            file_name=f"harness_report_{summary['dataset_name']}.txt",
            mime="text/plain",
            key="byom_download_text",
        )

    st.markdown("##### Verify it independently")
    st.code(
        f"python -m batlab.validation.replication <unzipped-bundle> \\\n"
        f"    --loader {record['loader']} --recompute",
        language="bash",
    )
    st.caption(
        "The bundle records the exact loader and its arguments, so the check "
        "re-derives the leave-cell-out number from the same raw cycles rather "
        "than trusting this page. The loader is the platform's own reference "
        "reloader, which returns the cycle tables *after* the enrichment step "
        "the feature builder reads — recording a raw dataset loader instead "
        "would digest a different table and fail a bundle that is correct."
    )
    with st.expander("Full harness report (text)"):
        st.code(result["report_text"], language="text")


def _render_result(result: dict) -> None:
    summary = result["summary"]
    _render_verdict(summary)
    _render_claims(summary)
    _render_lco(summary)
    _render_baselines(summary)
    _render_calibration(summary)
    _render_prospective(summary)
    _render_provenance(summary)
    _render_gate(summary)
    _render_downloads(result)


def page_model_validation() -> None:
    _action_bar("model_validation")
    st.markdown("# Bring Your Own Model")
    st.markdown(
        "#### Grade any forecaster by this platform's validation rules — and "
        "read the claims it does *not* support"
    )
    st.caption(
        "Every number on the Benchmark page was produced by the harness behind "
        "this page: leave-cell-out folds, RUL scored only on measured "
        "end-of-life rows, a prospective split for forecasting, conformal "
        "interval calibration, and a metric gate that can only fail on floors "
        "someone declared. Point it at your own model and it applies the same "
        "rules — including the ones that produce no number at all."
    )

    fleets = available_fleets()
    cfg = _render_config_panel(fleets)
    if not cfg or not cfg.get("fleet_key"):
        return

    gate = None
    if cfg.get("enforce_gate"):
        try:
            gate = json.loads(cfg.get("gate_text") or "{}")
        except json.JSONDecodeError:
            gate = "invalid"

    # What the widgets say right now. Compared against the stored run so a
    # stale result is never shown beside a changed configuration — Streamlit
    # re-runs this whole script on every interaction, including the ones that
    # only serve a download button.
    config_key = {
        "fleet": cfg["fleet_key"],
        "model": cfg["model_key"],
        "splits": (cfg["splits_lco"], cfg["splits_prospective"]),
        "intervals": cfg["intervals"],
        "seed": cfg["seed"],
        "gate": gate,
        "upload_name": (st.session_state.get("byom_upload").name
                        if st.session_state.get("byom_upload") is not None else None),
    }

    st.markdown("##### 4 · Run")
    ready, blocking = True, []
    if cfg["model_key"] == "uploaded_module":
        if cfg.get("upload_error"):
            ready, blocking = False, ["the uploaded module was refused (see the error above)"]
        elif not cfg.get("upload_source"):
            ready, blocking = False, ["no model module uploaded yet"]
        elif not cfg.get("upload_ack"):
            ready, blocking = False, ["tick the acknowledgement after reading the file"]
    if not cfg["splits_lco"] and not cfg["splits_prospective"]:
        ready, blocking = False, ["select at least one split"]

    if blocking:
        st.info("Before this can run: " + "; ".join(blocking) + ".")

    stored = st.session_state.get(_RESULT_KEY)
    if stored and stored.get("config_key") != config_key:
        # Keep the old result visible? No: showing one fleet's numbers under
        # another fleet's selector is exactly the mislabelling this page exists
        # to prevent. Say it changed instead.
        st.caption(
            "The configuration changed since the last run — the results below "
            "are from that earlier configuration and are labelled as such. "
            "Press Run to grade the current setup."
        )

    if st.button("Run the validation harness", type="primary", key="byom_run",
                 disabled=not ready):
        with st.spinner("Reloading raw cycles, then running every check…"):
            try:
                result = _run(cfg | {"upload_name": config_key["upload_name"]})
            except (ModelModuleError, ValueError, KeyError) as exc:
                st.error(f"The run could not start: {exc}")
                return
            except Exception as exc:  # noqa: BLE001 — a failed run must not kill the page
                st.error(
                    f"The run failed: {type(exc).__name__}: {exc}. Nothing was "
                    "recorded and no partial result is shown — a run that "
                    "stopped half way has no verdict."
                )
                return
        result["config_key"] = config_key
        st.session_state[_RESULT_KEY] = result
        stored = result

    if not stored:
        st.caption(
            "Nothing has been graded yet. Pressing Run reloads this fleet's raw "
            "cycles and fits the model once per fold and per target — real "
            "compute, which is why nothing runs until you ask."
        )
        return

    st.markdown("---")
    _render_result(stored)
