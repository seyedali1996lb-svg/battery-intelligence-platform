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

Two kinds of fleet, and what changes when the fleet is yours
------------------------------------------------------------
Grading a model means re-deriving features from RAW cycles, so the fleet
picker offers two things: the deployment's reference datasets (public,
reproducibly reloadable), and — on the same list, not a separate feature — the
org's own uploads, whose raw cycles are persisted at import time
(src/uploaded_store.py) and fingerprinted from there.

That second entry changes what a sealed bundle can promise, and the page makes
the choice explicit rather than deciding for you:

- **Embed the raw cycles in the bundle** (default for an upload). The artifact
  verifies anywhere, by anyone holding it, because the data is inside it — and
  handing someone the bundle hands them your data.
- **Point at this deployment's store.** The bundle keeps the digests and the
  numbers, the cycles stay home, and verifying needs this deployment's
  data/uploaded_fleets/ as well as the artifact.

A reference fleet is neither case (its data is public), so it records the
dataset's own loader. fleet_plan() in src/harness_models.py is what turns the
selection into the recorded data path, so the command printed on screen and the
bundle that was actually sealed cannot disagree.

Why an uploaded model can be read before it runs
------------------------------------------------
The uploader accepts exactly one thing: a .py module defining make_model().
Its source is shown to you and nothing executes until you tick an
acknowledgement. Serialized models (.pkl/.pt/.joblib/...) are REFUSED, with
the reason printed on the page: loading one runs code from the file as part of
loading it, so there is no moment at which it can be inspected first.

An uploaded module is also kept out of this process entirely. It is imported
and fitted in a sandbox child (batlab.harness.sandbox): a policy on imports and
on audited operations (no network, no subprocesses, no ctypes, writes confined
to its own scratch directory) plus wall-clock, memory and CPU caps. The page
prints which of those controls this deployment actually enforces, because the
half that cannot be enforced on a given platform is the half a reader needs to
know about — the full reasoning is in the sandbox module's docstring.
"""

from __future__ import annotations

import json
from typing import Any

import _paths  # noqa: F401
import streamlit as st
import pandas as pd

from utils import _action_bar, _empty_state, render_card
from harness_models import (
    BUILTIN_MODELS,
    ModelModuleError,
    STARTER_MODULE_SOURCE,
    available_fleets,
    UPLOADED_LOADER_SPEC,
    builtin_model,
    example_gate_expectations,
    fleet_plan,
    describe_module,
    load_model_module,
    model_bundle_source,
    describe_enforcement,
    seal_zip_bytes,
    summarise_report,
    verify_command,
    unsupported_upload_reason,
    uploaded_fleet_entries,
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


def _seed_now() -> int:
    """The current seed, readable before the widget is drawn this run.

    Streamlit repopulates session_state with each widget's value at the start
    of a rerun, so the model-source caption can be built in step 2 (above the
    seed input) without guessing at 42 and disagreeing with the run.
    """
    try:
        return int(st.session_state.get("byom_seed", 42))
    except (TypeError, ValueError):  # pragma: no cover - a number_input is int
        return 42


def _render_config_panel(fleets: "list[dict[str, Any]]", org_id: int) -> dict:
    """Draw the setup widgets and return the chosen configuration."""
    # Typed dicts, not bare dict: with unknown value types the checker resolves
    # the selectbox options to Sequence[Never] and reads the return as None.
    usable: "list[dict[str, Any]]" = [f for f in fleets if f["available"]]
    unavailable = [f for f in fleets if not f["available"]]
    fab: dict = {}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1 · Fleet to grade on")
        if not usable:
            _empty_state(
                "No fleet is reloadable on this deployment",
                "Reference fleets need their raw data cached locally "
                "(`python -m batlab.datasets.nasa` populates NASA; the synthetic "
                "fleet needs no download), and your own fleet appears here once "
                "you have run an upload through the Import page.",
            )
            return {}
        # The keys are strings by construction (a reference dataset's name, an
        # upload's content hash). Stating that to the checker matters: with
        # unknown key types it resolves the options to Sequence[Never] and
        # types this widget's return as None, which poisons every use of it.
        fleet_keys = [str(f["key"]) for f in usable]
        fleet_key = st.selectbox(
            "Fleet",
            options=fleet_keys,
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
        elif fleet.get("kind") == "uploaded":
            st.caption(
                "Expected runtime: scales with your upload "
                f"({fleet.get('n_rows') or '?'} cycle rows) — the fits are the "
                "same ones the reference fleets run."
            )

        if fleet.get("kind") == "uploaded":
            fab["embed_data"] = st.checkbox(
                "Put my raw cycles inside the sealed bundle",
                value=True,
                key="byom_embed_data",
                help=(
                    "On: the bundle can be verified by anyone you give it to, "
                    "because the data is inside it. Off: the bundle records "
                    "this deployment's store instead — your data stays home and "
                    "verifying needs that store. Either way the per-cell digests "
                    "are sealed in the bundle."
                ),
            )
            st.caption(
                fleet_plan(fleet, org_id=org_id, embed_data=fab["embed_data"])["privacy_note"]
            )

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

    fab["fleet_key"] = fleet_key if usable else None
    fab["fleet"] = fleet
    fab.setdefault("embed_data", False)
    fab.setdefault("include_model_source", True)
    fab["model_key"] = model_key

    if model_key == "uploaded_module":
        # What the sandbox enforces here, BEFORE a file is chosen: it is a
        # property of this deployment, not of the module about to be uploaded,
        # and someone deciding whether to upload at all needs it first.
        _render_enforcement()
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
                            "This is the exact text that will be run — in a "
                            "**separate sandbox process**, not in this app's "
                            "process. Read it anyway: the code still runs, and "
                            "inside the sandbox it can still read what this "
                            "deployment can read."
                        )
                        _render_enforcement()
                    fab["upload_ack"] = st.checkbox(
                        "I have read this file and accept that it runs in the "
                        "model sandbox",
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

    # What the sealed bundle will be able to re-derive. Stated next to the model
    # picker because it is a property of the MODEL, not of the download: the
    # platform's own model needs nothing carried (it is reproducible from the
    # repo), a catalogue baseline carries itself, and an uploaded module is the
    # one case where the source may be somebody's property — so it is the only
    # one offered as a choice rather than decided for you.
    if model_key == "uploaded_module":
        if fab.get("upload_source") and not fab.get("upload_error"):
            fab["include_model_source"] = st.checkbox(
                "Put my model module inside the sealed bundle",
                value=True,
                key="byom_embed_model",
                help=(
                    "On: the bundle can be recomputed by whoever you hand it to, "
                    "because your model's source is inside it. Off: the bundle "
                    "records only what model was graded, and recompute needs the "
                    "module from you."
                ),
            )
            st.caption(model_bundle_source(
                model_key, seed=_seed_now(), upload_source=fab["upload_source"],
                include_upload=fab["include_model_source"],
            )["note"])
    elif model_key != "platform_default":
        st.caption(model_bundle_source(model_key, seed=_seed_now())["note"])

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
    from batlab.harness import validate_forecaster, format_report

    fleet_key = cfg["fleet_key"]
    plan = fleet_plan(
        cfg["fleet"], org_id=cfg.get("org_id") or 0, embed_data=cfg.get("embed_data", False)
    )
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
    uploaded_probe = None
    close_sandbox = None
    if cfg["model_key"] == "uploaded_module":
        # Starts the sandbox child and loads the module in it. Everything the
        # model does from here on happens there; this object is a proxy.
        module = load_model_module(cfg["upload_source"])
        model = module["factory"]
        close_sandbox = module.get("close")
        # What the result carries: never the live sandbox (it would be kept
        # alive by a dictionary sitting in the session).
        uploaded_probe = describe_module(module)
    else:
        entry = builtin_model(cfg["model_key"], seed=cfg["seed"])
        model = entry["model"]

    cell_data = plan["reload"]()
    if len(cell_data) < 2:
        raise ValueError(
            f"This deployment reloaded only {len(cell_data)} cell(s) for "
            f"{fleet_key!r}. A leave-cell-out structure needs at least 2."
        )

    try:
        report = validate_forecaster(
            cell_data,
            model=model,
            seed=cfg["seed"],
            splits=splits,
            intervals=cfg["intervals"],
            gate=gate,
            dataset=plan["dataset"],
        )
    finally:
        # A sandbox that outlived its run would be a child process per rerun.
        if callable(close_sandbox):
            close_sandbox()

    # What travels with the number. The platform's own model needs nothing
    # carried; a catalogue baseline carries itself; an uploaded module carries
    # its source only if the user left that on. Decided here, from the SAME
    # cfg the run used, so the command printed below describes the artifact
    # that was actually sealed.
    source_plan = model_bundle_source(
        cfg["model_key"],
        seed=cfg["seed"],
        upload_source=cfg.get("upload_source") if cfg["model_key"] == "uploaded_module" else None,
        include_upload=cfg.get("include_model_source", True),
    )

    zip_bytes = None
    seal_error = None
    if "lco" in splits:
        try:
            zip_bytes = seal_zip_bytes(
                report, cell_data, dataset=plan["dataset"],
                loader=plan["loader"], loader_kwargs=plan["loader_kwargs"],
                embed_data=plan["embed_data"],
                model_source=source_plan["source"],
                model_entry_point=source_plan["entry_point"],
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
        "loader": {"loader": plan["loader"], "loader_kwargs": plan["loader_kwargs"]},
        "embed_data": plan["embed_data"],
        "privacy_note": plan["privacy_note"],
        "uploaded_module": uploaded_probe,
        # Whether the recorded command can also re-derive the number. The
        # recompute check needs the model itself: the platform's own
        # configuration is reproducible from the repo (batlab's default GBRT),
        # and anything else has to be handed over as code — which
        # source_plan.source is, when it travels, and which is why a nil source
        # here means the printed command must NOT claim --recompute.
        "default_model": cfg["model_key"] == "platform_default",
        "model_key": cfg["model_key"],
        "model_source_embedded": source_plan["source"] is not None,
        "model_source_label": source_plan["label"],
        "model_source_note": source_plan["note"],
        "model_source_is_user_code": source_plan["is_user_code"],
        "fleet_label": plan["label"],
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
    embedded = bool(result.get("embed_data"))
    bundle_name = f"sealed_bundle_{summary['dataset_name']}.zip"
    # Whether the command can re-derive the number is decided in one pure,
    # tested place (harness_models.verify_command): --recompute is printable
    # only when the verifier can obtain the model that was graded, and a
    # command that fails on the number printed next to it is worse than a
    # command with a stated limitation.
    verify = verify_command(result, bundle_file_name=bundle_name)
    carries_model = verify["carries_model"]
    if result["zip_bytes"]:
        st.download_button(
            "Download the sealed bundle (.zip)",
            data=result["zip_bytes"],
            file_name=bundle_name,
            mime="application/zip",
            key="byom_download_zip",
            help=(
                "benchmark.json + harness_report.json + replication.json, with "
                "per-cell digests and the model identity sealed in"
                + (", plus cells/ — your raw cycle tables, one CSV per cell."
                   if embedded else ".")
            ),
        )
        st.caption(result.get("privacy_note") or "")
        if carries_model and result.get("model_source_is_user_code"):
            st.caption(result.get("model_source_note") or "")
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
    if carries_model:
        st.caption(
            "The bundle carries the model it was published for "
            f"(`{result.get('model_source_label')}`), as source at "
            "`model/module.py`, so this command re-runs the exact configuration "
            "that was graded."
        )
    if embedded:
        st.caption(
            "This bundle carries the raw cycles, so it verifies on its own — no "
            "dataset download, no access to this deployment, and no working "
            "directory to be in."
        )
        st.code(verify["command"], language="bash")
        st.caption(
            "`cells/index.json` records, for every cell, the file's own "
            "sha256 **and** the `cell_digest` it reloads to — the same value "
            "as the sealed `cell_digests`, so a reviewer can walk from the "
            "file's bytes to the fingerprint with `sha256sum` and no tooling "
            "in between. The seal covers the cell files as well as the report, "
            "so a file edited after sealing fails the seal check, and a bundle "
            "whose index points outside `cells/` is refused rather than read."
        )
    else:
        st.code(verify["command"], language="bash")
        if record["loader"] == UPLOADED_LOADER_SPEC:
            st.caption(
                "This bundle does not carry your data: the recorded loader "
                "reads this deployment's `data/uploaded_fleets/` store, so "
                "running the check needs that store as well as the bundle. The "
                "seal still covers the report, the digests, and the model "
                "identity — and those digests are what the number was measured "
                "against."
            )
        else:
            st.caption(
                "The bundle records the exact loader and its arguments, so the "
                "check re-derives the leave-cell-out number from the same raw "
                "cycles rather than trusting this page. The loader is the "
                "platform's own reference reloader, which returns the cycle "
                "tables *after* the enrichment step the feature builder reads — "
                "recording a raw dataset loader instead would digest a "
                "different table and fail a bundle that is correct. That "
                "module lives in this checkout's `src/`, which the verifier "
                "puts on its import path itself, so the command needs no "
                "`PYTHONPATH` when it is run from a clone."
            )
    if carries_model and result.get("model_source_is_user_code"):
        st.caption(
            "**Read `model/module.py` before you run that command.** Importing "
            "it executes it, in your process — the same reason this app accepts "
            "`.py` rather than a pickle is the reason the bundle ships readable "
            "source and never loads it for you. The file's bytes are checked "
            "against the digest the bundle records before the import happens."
        )
    elif verify["reason"]:
        st.caption(verify["reason"])

    with st.expander("Full harness report (text)"):
        st.code(result["report_text"], language="text")


# The sandbox's own report, in the page's words. Order matters: what runs where
# first, then the caps, then the two things a reader most needs to know are NOT
# guaranteed. Rendering this by NAME (not by pasting the sandbox's dict) means an
# enforcement key that disappears fails visibly here instead of silently
# shrinking the disclosure.
_ENFORCEMENT_ROWS = (
    ("Where the model runs", "process"),
    ("Wall clock", "wall_clock"),
    ("Memory", "memory"),
    ("CPU", "cpu"),
    ("Imports it is refused", "imports"),
    ("Operations it is refused", "operations"),
    ("File writes", "file_writes"),
    ("**Not enforced**", "not_enforced"),
)


def _enforcement_table(enforced: dict) -> None:
    rows = [
        {"Control": label, "What happens": str(enforced.get(key) or "unreported")}
        for label, key in _ENFORCEMENT_ROWS
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_enforcement() -> None:
    """What this deployment actually enforces, BEFORE anything is uploaded.

    The summary sentence is not decoration and not a duplicate: it states the
    one thing a reader must not have to infer from a table — that the boundary
    is containment, not a container — and it is what the page's tests can
    assert on, which keeps the disclosure from silently shrinking when a
    platform changes.
    """
    enforced = describe_enforcement()
    st.markdown("###### The sandbox this deployment runs your module in")
    _enforcement_table(enforced)
    st.caption(
        "Your module runs in a separate process under the caps above — it does "
        "not get this app's privileges. What is NOT enforced: "
        f"{enforced['not_enforced']} The module's source is therefore still "
        "shown, and read, before it runs."
    )
    st.caption(
        "**This costs time, and here is the measured amount.** The sandbox answers "
        "one fit or predict at a time (that is what keeps answers matched to "
        "questions), so an uploaded model's folds are fitted one after another "
        "instead of in parallel, and the child loads the platform's own library "
        "stack before your model runs: on the NASA fleet's default configuration "
        "that was ≈4 s sandboxed versus ≈2.5 s in-process (linear model, "
        "2026-09-17). A heavier model pays more."
    )


def _render_sandbox(result: dict) -> None:
    """The same report, as sealed with the run — after the fact, not promised."""
    module = result.get("uploaded_module") or {}
    enforced = module.get("sandbox")
    if not enforced:
        return
    st.markdown("#### The sandbox this model ran in")
    st.caption(
        f"The module was copied to a scratch file (`{module.get('path')}`), "
        f"imported and fitted there in a separate process (class "
        f"`{module.get('probe_class')}`) — never in this app's process — and "
        "the scratch file was removed when the run ended. The controls below "
        "are what this deployment enforced while it ran."
    )
    _enforcement_table(enforced)
    if module.get("output"):
        with st.expander("What the module itself printed (captured in the sandbox)"):
            st.code(str(module["output"]), language="text")


def _render_result(result: dict) -> None:
    summary = result["summary"]
    _render_verdict(summary)
    _render_claims(summary)
    _render_sandbox(result)
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
        "rules — including the ones that produce no number at all — on the "
        "reference fleets, or on the cells you uploaded yourself."
    )

    org_id = st.session_state.get("auth_org_id")
    # The org's own uploads come FIRST: if you have just uploaded your own
    # cells, grading your model on them is the question you came with. With no
    # upload persisted the list is exactly the reference fleets, so the
    # default selection is unchanged (NASA, when cached).
    fleets = uploaded_fleet_entries(org_id or 0) + available_fleets()
    cfg = _render_config_panel(fleets, org_id or 0)
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
        # A different answer to "does my data (or my model) travel in the
        # bundle?" is a different run, and the sealed download is the thing
        # that changes.
        "embed_data": bool(cfg.get("embed_data")),
        "include_model_source": bool(cfg.get("include_model_source")),
        # getattr, not .name: session_state.get() is typed as Any|None, and a
        # repeated call in a guard does not narrow the expression it guards.
        "upload_name": getattr(st.session_state.get("byom_upload"), "name", None),
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
                result = _run(cfg | {"upload_name": config_key["upload_name"],
                                     "org_id": org_id or 0})
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
