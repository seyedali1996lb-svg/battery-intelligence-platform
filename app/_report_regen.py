"""
Report regeneration widget — replays the recorded training pipeline behind a bundle.

Extracted from _ui_helpers.py as a self-contained widget used by both the Cell
Workbench and the EU Passport Reports page.

Hyperparameter divergence is a *gate*, not a footnote
----------------------------------------------------
replay_run() always re-trains with the CURRENT batlab.models.gbrt.GBRT_PARAMS —
run_lco() has no hyperparams argument at all — so when a run's own logged
snapshot differs from today's constants, whatever this button produces is not
that run's number. The registry already detected this (hyperparams_match /
hyperparams_diff); the change here is to check it BEFORE the button rather than
warn after it, require an explicit acknowledgment to proceed, and label the
reproduced column as not-a-reproduction when the run has diverged. A reader can
no longer click straight through and mistake the result for a faithful replay.
"""

from __future__ import annotations

import streamlit as st

import _paths  # noqa: F401


def _render_divergence_error(diff: dict) -> None:
    """Loud, specific pre-flight error for a run that can't be faithfully
    regenerated: names every diverging parameter with recorded-vs-current.

    Markdown, not raw HTML — st.error() renders its body as Markdown (HTML is
    escaped by default), so an <ul>/<li> block would show up literally."""
    bullets = "\n".join(
        f"- `{k}`: recorded **{recorded}** → current **{current}**"
        for k, (recorded, current) in diff.items()
    )
    st.error(
        "**This run cannot be faithfully regenerated.** The GBRT hyperparameters "
        "have changed since this run was trained, and replay always re-trains with "
        "the *current* settings (`run_lco()` takes no hyperparameters argument at "
        "all). Any number produced below is a fresh run under today's settings, "
        "not this run's recorded metric:\n\n" + bullets,
        icon="⚠️",
    )


def render_regenerate_report_button(bundle: dict, org_id: "int | None", key_suffix: str) -> None:
    """\"Regenerate this report\" — replays the recorded pipeline (cell_ids +
    feature_version + seed) behind a currently-displayed bundle, via the
    experiment registry (src/experiment_registry.py).

    bundle["metrics"]["experiment_run_id"] is set by the training call
    sites themselves — absent only for a bundle loaded from a disk cache
    written before this feature existed, handled below as "nothing to replay".
    """
    import experiment_registry as reg

    run_id = (bundle.get("metrics") or {}).get("experiment_run_id")
    with st.expander("🔁 Regenerate this report", expanded=False):
        if not run_id:
            st.caption(
                "This result predates experiment-run logging (or was served "
                "from an older disk cache) — nothing to replay."
            )
            return

        run = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
        if run is None and org_id is not None:
            run = reg.get_run(org_id, run_id)
        if run is None:
            st.caption(f"Logged run `{run_id}` not found — the registry may have been reset.")
            return

        st.caption(
            f"Recorded run `{run_id}` — dataset **{run['dataset']}**, "
            f"{run['n_cells']} cells, seed {run['seed']}, "
            f"git `{run['git_commit']}`, logged {run['timestamp'][:19]}."
        )

        if run["dataset"] not in reg.REFERENCE_DATASETS:
            st.caption(
                "⚠ Replay isn't available for this run — only the trained "
                "result is persisted for uploaded data, not the original "
                "uploaded cycle data, so there is nothing to re-fit against."
            )
            return

        # Pre-flight, not post-hoc: see this module's docstring.
        divergence = reg.hyperparams_divergence(run)
        acknowledged = True
        if divergence:
            _render_divergence_error(divergence)
            acknowledged = st.checkbox(
                "I understand the regenerated numbers will **not** be this run's "
                "recorded numbers.",
                key=f"regen_ack_{key_suffix}",
            )

        if st.button("Regenerate", key=f"regen_{key_suffix}", disabled=not acknowledged):
            with st.spinner("Reloading source data and re-running the recorded pipeline…"):
                try:
                    cell_data = reg.reload_reference_cell_data(run["dataset"], cell_ids=run["cell_ids"])
                    result = reg.replay_run(run["org_id"], run_id, cell_data)
                except ValueError as exc:
                    st.error(f"Replay failed: {exc}")
                    return

            _faithful = result["hyperparams_match"]
            if _faithful:
                st.success("Replay complete.")
            else:
                # Repeated *before* the numbers, not after them, so the
                # disclosure is read before the values are.
                st.error(
                    "**Replay complete — but this is NOT a reproduction of the "
                    "recorded run.** It re-trained with the current GBRT "
                    "hyperparameters, so the two columns below are two different "
                    "models. Do not read the difference as environment or data drift.",
                    icon="⚠️",
                )

            col_rec, col_repro = st.columns(2)

            def _num(v, decimals=3, suffix=""):
                # v12: recorded/reproduced RUL can be None (no observed-EOL
                # rows in any fold) — render as "not evaluable", never crash.
                if v is None:
                    return "not evaluable"
                try:
                    if v != v:  # NaN
                        return "not evaluable"
                except TypeError:
                    pass
                return f"{v:.{decimals}f}{suffix}"

            with col_rec:
                st.markdown("**Recorded** (as logged)")
                st.write(f"SOH R²: {_num(result['recorded']['soh_r2'])}")
                st.write(f"RUL MAE: {_num(result['recorded']['rul_mae'], 1, ' cycles')}")
            with col_repro:
                st.markdown(
                    "**Reproduced now**" if _faithful
                    else "**Reproduced now** — current hyperparameters, not this run's"
                )
                st.write(f"SOH R²: {_num(result['soh_r2'])}")
                st.write(f"RUL MAE: {_num(result['rul_mae'], 1, ' cycles')}")

            if not result["environment_match"]:
                st.warning(
                    "Library versions differ from the original run "
                    f"(numpy/pandas/scikit-learn): {result['environment_diff']} — "
                    "small numeric drift is expected, treat this as "
                    "reproduced-in-spirit rather than byte-for-byte confirmed."
                )
            elif _faithful:
                st.caption("Same library versions as the original run — numbers match exactly.")
