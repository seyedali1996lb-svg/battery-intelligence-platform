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
