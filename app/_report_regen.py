"""
Report regeneration widget — replays the recorded training pipeline behind a bundle.

Extracted from _ui_helpers.py as a self-contained widget used by both the Cell
Workbench and the EU Passport Reports page.
"""

from __future__ import annotations

import streamlit as st

import _paths  # noqa: F401


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

        if st.button("Regenerate", key=f"regen_{key_suffix}"):
            with st.spinner("Reloading source data and re-running the recorded pipeline…"):
                try:
                    cell_data = reg.reload_reference_cell_data(run["dataset"], cell_ids=run["cell_ids"])
                    result = reg.replay_run(run["org_id"], run_id, cell_data)
                except ValueError as exc:
                    st.error(f"Replay failed: {exc}")
                    return

            st.success("Replay complete.")
            col_rec, col_repro = st.columns(2)
            with col_rec:
                st.markdown("**Recorded**")
                st.write(f"SOH R²: {result['recorded']['soh_r2']:.3f}")
                st.write(f"RUL MAE: {result['recorded']['rul_mae']:.1f} cycles")
            with col_repro:
                st.markdown("**Reproduced now**")
                st.write(f"SOH R²: {result['soh_r2']:.3f}")
                st.write(f"RUL MAE: {result['rul_mae']:.1f} cycles")

            if not result["environment_match"]:
                st.warning(
                    "Library versions differ from the original run "
                    f"(numpy/pandas/scikit-learn): {result['environment_diff']} — "
                    "small numeric drift is expected, treat this as "
                    "reproduced-in-spirit rather than byte-for-byte confirmed."
                )
            else:
                st.caption("Same library versions as the original run — numbers match exactly.")

            if not result["hyperparams_match"]:
                st.warning(
                    "The GBRT hyperparameters this platform currently trains with "
                    f"differ from what this run's own logged record shows: "
                    f"{result['hyperparams_diff']} (format: {{param: (recorded, current)}}). "
                    "This replay used the CURRENT hyperparameters, not the "
                    "recorded ones — any difference above is a real reason the "
                    "reproduced numbers could diverge from the recorded ones, "
                    "separate from environment or data drift."
                )
