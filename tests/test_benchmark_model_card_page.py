"""
AppTest-based page-level verification for the Benchmark page's
auto-generated model-card expander (P2). Follows the same pattern as
tests/test_deployment_sizing_page.py: bypass login via session_state, run
the real app script, and assert no exception. The expander only renders
when at least one run is logged, so this test seeds a run into the
isolated test DB first (via experiment_registry.log_run, the same call the
real training sites make).
"""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
import pytest
import db as db_module
from streamlit.testing.v1 import AppTest

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _logged_in_app(role: str, page: str, data_mode: str, **extra_state) -> AppTest:
    at = AppTest.from_file(_MAIN_PY, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = role
    at.session_state["page"] = page
    at.session_state["data_mode"] = data_mode
    for k, v in extra_state.items():
        at.session_state[k] = v
    return at


def test_benchmark_model_card_expander_renders_without_exception(isolated_db):
    """A logged run must render the auto-generated model card (markdown +
    JSON download button) without crashing the Benchmark page."""
    import experiment_registry as reg

    reg.log_run(
        org_id=1,
        dataset="nasa",
        chemistry="LiCoO2",
        feature_set=["fade_rate_30cy", "sop_pct"],
        feature_version="v11-usage-profile",
        hyperparams={"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                     "subsample": 0.8, "random_state": 42},
        seed=42,
        cell_ids=["B0005", "B0006"],
        n_rows=336,
        lco_metrics={
            "soh_mae": 1.2, "soh_r2": 0.806, "rul_mae": 50.0, "rul_r2": 0.75,
            "rul_reliable": True,
            "per_cell": {"B0005": {"soh_r2": 0.9, "rul_r2": 0.7}},
        },
    )

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception, f"Benchmark page crashed: {at.exception}"

    expander_labels = [e.label for e in at.expander]
    assert any("Model card" in label for label in expander_labels), expander_labels

    # The card body is st.markdown inside the expander — its markdown should
    # include the model-card header and a limitation line.
    text = "\n".join(m.value for m in at.markdown)
    assert "Model card" in text
    assert "Limitations" in text
    assert "public laboratory datasets" in text


def test_benchmark_accuracy_by_source_table_renders(isolated_db):
    """The per-chemistry/source accuracy table must render with model R²,
    baseline R² and real advantage side by side, and must keep NASA (real
    LiCoO2 cells) and the synthetic fleet (simulated LiCoO2) as SEPARATE rows
    rather than collapsing them by chemistry."""
    import experiment_registry as reg

    def _log(dataset, chemistry, soh_r2, baseline, n_cells):
        reg.log_run(
            org_id=1,
            dataset=dataset,
            chemistry=chemistry,
            feature_set=["cycle_number"],
            feature_version="v11-usage-profile",
            hyperparams={"random_state": 42},
            seed=42,
            cell_ids=[f"C{i}" for i in range(n_cells)],
            n_rows=n_cells * 100,
            lco_metrics={
                "soh_mae": 1.0, "soh_r2": soh_r2,
                "baseline_soh_r2": baseline,
                "rul_mae": 20.0, "rul_r2": 0.7, "rul_reliable": True,
                "per_cell": {},
            },
        )

    _log("nasa", "LiCoO2", 0.759, 0.603, 4)
    _log("severson", "LFP", 0.986, -0.758, 12)
    _log("synth", "LiCoO2", 0.998, -1.967, 8)

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception, f"Benchmark page crashed: {at.exception}"

    text = "\n".join(m.value for m in at.markdown)
    assert "Accuracy by chemistry" in text

    # Find the accuracy table by its columns (the leaderboard table is separate).
    acc = [df.value for df in at.dataframe if "Real advantage" in getattr(df.value, "columns", [])]
    assert acc, [list(getattr(df.value, "columns", [])) for df in at.dataframe]
    table = acc[0]

    # Grouped by (dataset, chemistry): NASA and synth are both LiCoO2 yet must
    # remain distinct rows — a simulated fleet must not stand in for real cells.
    assert len(table) == 3
    assert set(table["Source"]) == {"nasa", "severson", "synth"}

    nasa = table[table["Source"] == "nasa"].iloc[0]
    assert nasa["Model R2"] == "0.759"
    assert nasa["Baseline R2"] == "0.603"
    assert nasa["Real advantage"] == "+0.156"

    # Sorted by real advantage descending — synth's largest genuine skill first.
    assert list(table["Source"]) == ["synth", "severson", "nasa"]


def test_benchmark_marks_the_best_skill_model_per_chemistry(isolated_db):
    """Model selection is per CHEMISTRY and ranked on genuine advantage, not raw
    R²: NASA and the synthetic fleet are both LiCoO2, and the synthetic fleet's
    0.998 beats NASA's 0.759 on R² — but the point of the mark is that exactly
    one model per chemistry is chosen, by Real advantage."""
    import experiment_registry as reg

    def _log(dataset, chemistry, soh_r2, baseline, n_cells):
        reg.log_run(
            org_id=1, dataset=dataset, chemistry=chemistry,
            feature_set=["cycle_number"], feature_version="v11-usage-profile",
            hyperparams={"random_state": 42}, seed=42,
            cell_ids=[f"C{i}" for i in range(n_cells)], n_rows=n_cells * 100,
            lco_metrics={
                "soh_mae": 1.0, "soh_r2": soh_r2, "baseline_soh_r2": baseline,
                "rul_mae": 20.0, "rul_r2": 0.7, "rul_reliable": True, "per_cell": {},
            },
        )

    # LiCoO2: synth has the larger advantage (+2.965 vs NASA's +0.156).
    _log("nasa", "LiCoO2", 0.759, 0.603, 4)
    _log("synth", "LiCoO2", 0.998, -1.967, 8)
    _log("severson", "LFP", 0.986, -0.758, 12)

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception, f"Benchmark page crashed: {at.exception}"

    acc = [df.value for df in at.dataframe if "Real advantage" in getattr(df.value, "columns", [])]
    assert acc, [list(getattr(df.value, "columns", [])) for df in at.dataframe]
    table = acc[0]
    assert "Selected" in table.columns

    marked = set(table[table["Selected"] == "✓ best skill"]["Source"])
    assert marked == {"synth", "severson"}, table[["Source", "Selected"]].to_dict()
    assert "nasa" not in marked

    # The rule that picked them is stated on the page, not left implicit.
    captions = "\n".join(c.value for c in at.caption)
    assert "largest **Real advantage**" in captions


def test_benchmark_accuracy_table_absent_without_baselines(isolated_db):
    """With no baselined runs, the section shows an honest empty-state caption
    rather than an empty or fabricated table."""
    import experiment_registry as reg

    reg.log_run(
        org_id=1, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v11-usage-profile",
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=["B0005", "B0006"], n_rows=300,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.8, "rul_mae": 20.0, "rul_r2": 0.7,
                     "rul_reliable": True, "per_cell": {}},
    )

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception

    text = "\n".join(m.value for m in at.markdown)
    assert "Accuracy by chemistry" in text

    # The empty state is an honest caption (not an empty/fabricated table).
    captions = "\n".join(c.value for c in at.caption)
    assert "No LCO run with a baseline logged yet" in captions
    # And no baselined-table columns were emitted.
    assert not [df for df in at.dataframe
                if "Real advantage" in getattr(df.value, "columns", [])]


def test_benchmark_renders_permanent_cross_chemistry_counterexample(isolated_db):
    """The NASA -> Severson transfer failure must be a permanent, visible
    part of the Benchmark page (next to the per-chemistry LCO table, sorted
    worst-first), not a one-off study output someone has to go re-derive."""
    import experiment_registry as reg

    def _transfer(dataset, chemistry, soh_r2, rul_mae, rul_r2):
        reg.log_run(
            org_id=reg.PLATFORM_ORG_ID, dataset=dataset, chemistry=chemistry,
            feature_set=["cycle_number", "fade_rate_30cy"],
            feature_version="v11-usage-profile",
            hyperparams={"random_state": 42}, seed=42,
            cell_ids=["B0005", "S-b1c2"], n_rows=100,
            lco_metrics={
                "soh_mae": 22.0, "soh_r2": soh_r2,
                "rul_mae": rul_mae, "rul_r2": rul_r2,
                "rul_reliable": False, "per_cell": {},
            },
            notes="Cross-chemistry generalization study.",
        )

    _transfer("nasa_to_severson", "LiCoO2 -> LFP", -34.592, 1350.9, -1.138)
    reg.log_cross_chemistry_unavailable("nasa", "oxford", "Oxford schema incompatible.")

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception, f"Benchmark page crashed: {at.exception}"

    text = "\n".join(m.value for m in at.markdown)
    assert "Cross-chemistry transfer" in text
    assert "unseen cell" in text

    xfer = [df.value for df in at.dataframe
            if "Train → Eval" in getattr(df.value, "columns", [])]
    assert xfer, [list(getattr(df.value, "columns", [])) for df in at.dataframe]
    table = xfer[0]

    # Note: load_everything() runs the study for real at platform init, so on
    # top of this test's seeded pair the table may also carry genuinely-computed
    # reference pairings. The assertions below are about the INVARIANTS, not a
    # fixed row count.
    assert "nasa → severson" in set(table["Train → Eval"])

    # Worst transfer first: the seeded NASA -> Severson counterexample leads,
    # and evaluated SOH R² values are non-increasing down the table.
    worst = table.iloc[0]
    assert worst["Train → Eval"] == "nasa → severson"
    assert worst["SOH R2"] == "-34.592"
    assert worst["RUL R2"] == "-1.138"
    assert worst["RUL MAE (cy)"] == "1350.9"
    assert worst["Chemistry"] == "LiCoO2 -> LFP"
    evaluated_r2 = [float(v) for v in table["SOH R2"] if v != "—"]
    assert evaluated_r2 == sorted(evaluated_r2)

    # The not-evaluable pairing is disclosed, not silently dropped.
    oxford = table[table["Train → Eval"] == "nasa → oxford"].iloc[0]
    assert oxford["Status"] == "not evaluated"
    assert oxford["SOH R2"] == "—"
    # Not-evaluated rows come after the numbers.
    assert table.iloc[-1]["Status"] == "not evaluated"
    captions = "\n".join(c.value for c in at.caption)
    assert "not evaluated" in captions.lower()
    assert "Oxford schema incompatible" in captions

    # The method must be spelled out as NOT leave-cell-out, so it can't be read
    # as a like-for-like comparison with the LCO table above it. (st.caption is
    # its own AppTest element type, not markdown.)
    assert "no cell is held out" in captions
    assert "not the same thing as the table above" in captions


def test_benchmark_model_kind_comparison_renders_gbrt_vs_pinn(isolated_db):
    """The GBRT-vs-PINN table must render both model kinds on the same
    population with a signed delta, and must report a PINN LOSS rather than
    omitting the losing row."""
    import experiment_registry as reg

    def _log_kind(kind, soh_r2, rul_r2, rul_mae):
        reg.log_run(
            org_id=1, dataset="nasa", chemistry="LiCoO2",
            feature_set=["cycle_number"], feature_version="v11-usage-profile",
            hyperparams={"model_kind": kind}, seed=42,
            cell_ids=["B0005", "B0006"], n_rows=200,
            lco_metrics={
                "soh_mae": 1.0, "soh_r2": soh_r2, "baseline_soh_r2": 0.603,
                "rul_mae": rul_mae, "rul_r2": rul_r2, "rul_reliable": True,
                "per_cell": {},
            },
            model_kind=kind,
        )

    _log_kind("gbrt", 0.759, 0.63, 50.0)
    _log_kind("pinn", 0.301, -0.40, 260.0)  # deliberately loses

    at = _logged_in_app(role="Engineer", page="benchmark", data_mode="nasa")
    at.run()
    assert not at.exception, f"Benchmark page crashed: {at.exception}"

    text = "\n".join(m.value for m in at.markdown)
    assert "Model comparison" in text

    tables = [df.value for df in at.dataframe
              if "Model" in getattr(df.value, "columns", [])
              and "vs GBRT" in getattr(df.value, "columns", [])]
    assert tables, [[list(getattr(df.value, "columns", [])) for df in at.dataframe]]
    table = tables[0]

    kinds = set(table["Model"])
    assert any("GBRT" in k for k in kinds)
    assert any("PINN" in k for k in kinds)

    # The losing PINN row is present, with its negative delta shown as-is.
    pinn_row = table[table["Model"].str.contains("PINN")].iloc[0]
    assert pinn_row["SOH R2"] == "0.301"
    assert pinn_row["vs GBRT"] == "-0.458"

    # Identical baseline on both rows — the comparison is model-only.
    gbrt_row = table[table["Model"].str.contains("GBRT")].iloc[0]
    assert gbrt_row["Baseline R2"] == pinn_row["Baseline R2"]
    assert gbrt_row["Cells"] == pinn_row["Cells"]

    captions = "\n".join(c.value for c in at.caption)
    assert "IDENTICAL" in captions
    assert "production model" in captions
