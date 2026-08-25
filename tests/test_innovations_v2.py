"""
Comprehensive test suite for the 10 platform improvements and innovations.
"""

import numpy as np
import pandas as pd
import pytest

from batlab.features.vectorized import extract_features_vectorized, benchmark_vectorized_speedup
from batlab.features.partial_cycles import rainflow_counting, reconstruct_ocv_relaxation, partial_ica_analysis
from batlab.datasets.cycler_mapper import detect_cycler_format, ingest_cycler_data
from src.pinn_model import BatteryPINNEstimator
from src.dynamic_circularity import calculate_dynamic_lca, generate_verifiable_credential_passport, match_second_life_bids
from src.action_center import action_center
from src.task_queue import task_queue
from src.streaming_analytics import streaming_engine


def test_vectorized_features():
    df = pd.DataFrame({
        "cycle_number": np.arange(1, 101),
        "capacity_ah": 2.0 - 0.002 * np.arange(1, 101),
        "resistance_ohm": 0.02 + 0.0001 * np.arange(1, 101),
        "temperature_mean_c": np.full(100, 25.0),
        "c_rate": np.full(100, 1.0),
    })
    
    feats = extract_features_vectorized(df)
    assert not feats.empty
    assert "fade_rate_10cy" in feats.columns
    assert "soh_velocity" in feats.columns
    assert "stress_index" in feats.columns
    assert "resistance_trend_10cy" in feats.columns
    
    bench = benchmark_vectorized_speedup(df, iterations=5)
    assert bench["n_rows"] == 100
    assert bench["throughput_rows_per_sec"] > 0


def test_rainflow_cycle_counting():
    # Simulated driving profile: 100% -> 80% -> 85% -> 50% -> 60% -> 20% -> 100%
    soc_profile = np.array([100, 80, 85, 50, 60, 20, 100])
    res = rainflow_counting(soc_profile)
    
    assert res["total_cycles"] > 0
    assert res["equivalent_full_cycles"] > 0
    assert "dod_histogram" in res
    assert isinstance(res["cycles"], list)


def test_ocv_relaxation_reconstruction():
    time_sec = np.arange(0, 300, 1.0)
    # Voltage relaxation during rest (current drops to 0 at t=60)
    current_a = np.where(time_sec < 60, -2.0, 0.0)
    voltage_v = np.where(time_sec < 60, 3.6, 3.7 + 0.1 * (1.0 - np.exp(-(time_sec - 60) / 40.0)))
    
    ocv_res = reconstruct_ocv_relaxation(time_sec, voltage_v, current_a)
    assert ocv_res["detected_rests"] == 1
    assert ocv_res["estimated_ocv_v"] is not None
    assert ocv_res["estimated_ocv_v"] >= 3.7
    assert ocv_res["instant_r0_ohm"] is not None


def test_partial_ica_analysis():
    v = np.linspace(3.0, 4.2, 200)
    # Synthetic capacity charging curve with a peak in dQ/dV around 3.8V
    q = 1.0 / (1.0 + np.exp(-(v - 3.8) / 0.1))
    
    ica_res = partial_ica_analysis(v, q, v_min=3.4, v_max=4.1)
    assert ica_res["peak_v"] is not None
    assert 3.6 <= ica_res["peak_v"] <= 4.0
    assert ica_res["peak_dqdv"] > 0


def test_cycler_format_detection_and_ingestion():
    # Arbin-style DataFrame
    arbin_df = pd.DataFrame({
        "Cycle_Index": [1, 1, 1, 2, 2, 2],
        "Test_Time(s)": [10, 20, 30, 40, 50, 60],
        "Voltage(V)": [3.6, 3.7, 3.8, 3.6, 3.7, 3.8],
        "Current(A)": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Discharge_Capacity(Ah)": [0.0, 0.5, 1.8, 0.0, 0.5, 1.78],
        "Temperature(C)": [24.0, 24.2, 24.5, 24.1, 24.3, 24.6],
    })
    
    detection = detect_cycler_format(arbin_df)
    assert detection["hardware"] == "Arbin"
    assert "voltage_v" in detection["mapped_columns"]
    
    std_df, report = ingest_cycler_data(arbin_df)
    assert not std_df.empty
    assert "cycle_number" in std_df.columns
    assert "capacity_ah" in std_df.columns
    assert report["total_cycles_ingested"] >= 1


def test_pinn_estimator():
    cycles = np.arange(1, 50)
    # LLI sqrt(n) fade
    soh = 100.0 - 0.5 * np.sqrt(cycles) - 0.01 * cycles
    
    estimator = BatteryPINNEstimator(nominal_capacity_ah=2.0, chemistry="LFP")
    estimator.fit(cycles, soh)
    assert estimator.fitted
    
    preds = estimator.predict(np.arange(1, 100))
    assert len(preds["soh_pct"]) == 99
    assert preds["soh_pct"][0] >= preds["soh_pct"][-1]  # Monotonicity
    assert "knee_risk_score" in preds
    
    rul = estimator.estimate_rul(current_cycle=50, eol_soh_pct=80.0)
    assert rul["rul_cycles"] > 0
    assert "dominant_mechanism" in rul


def test_dynamic_lca_and_verifiable_passport():
    lca = calculate_dynamic_lca(
        cell_id="B0005",
        chemistry="LFP",
        nominal_kwh=0.0066,
        cumulative_throughput_kwh=18.5,
        region="GERMANY",
    )
    assert lca["net_lifecycle_co2_kg"] > 0
    assert "mfg_co2_kg" in lca
    assert "use_phase_co2_kg" in lca
    
    vc = generate_verifiable_credential_passport(
        cell_id="B0005",
        org_id="1",
        chemistry="LFP",
        soh_pct=74.8,
        rul_cycles=120,
        resistance_ohm=0.032,
        carbon_data=lca,
    )
    assert vc["@context"]
    assert vc["credentialSubject"]["cellId"] == "B0005"
    assert "proof" in vc
    assert "jws" in vc["proof"]


def test_second_life_bids():
    bids = match_second_life_bids(
        cell_id="B0005",
        chemistry="LFP",
        soh_pct=78.5,
        resistance_growth_pct=130.0,
        nominal_kwh=0.0066,
    )
    assert len(bids) > 0
    assert any(b["status"] == "QUALIFIED" for b in bids)


def test_action_center_triage_and_dispatch():
    actions = action_center.list_actions(org_id=1)
    assert len(actions) > 0
    
    # Create ticket
    act = action_center.create_action(
        cell_id="TEST-01",
        title="Test Anomaly",
        category="SAFETY",
        severity="CRITICAL",
        description="Voltage drop",
        recommended_action="INSPECT",
        soh_pct=82.0,
        org_id=1,
    )
    assert act["id"].startswith("act-")
    
    # Triage
    triaged = action_center.triage_action(act["id"], new_status="IN_PROGRESS", assigned_to="Engineer Alice")
    assert triaged["status"] == "IN_PROGRESS"
    assert triaged["assigned_to"] == "Engineer Alice"
    
    # Dispatch
    receipt = action_center.dispatch_workflow(act["id"], target_system="CMMS")
    assert receipt["status"] == "SUCCESS"
    assert receipt["target_system"] == "CMMS"


def test_task_queue():
    def _dummy_job(task, x, y):
        task.update(50, "Calculating sum...")
        return x + y
        
    task_id = task_queue.submit_task("Sum Job", _dummy_job, 10, 20, org_id=1)
    assert task_id.startswith("task-")
    
    time_limit = 5.0
    import time
    t0 = time.time()
    task_data = task_queue.get_task(task_id)
    while task_data["status"] not in ("COMPLETED", "FAILED") and (time.time() - t0) < time_limit:
        time.sleep(0.1)
        task_data = task_queue.get_task(task_id)
        
    assert task_data["status"] == "COMPLETED"
    assert task_data["result"] == 30


def test_streaming_anomaly_engine():
    engine = streaming_engine
    
    # Normal readings
    res_norm = engine.process_reading(cell_id="STR-01", voltage_v=3.8, current_a=-1.0, temperature_c=25.0)
    assert res_norm["severity"] in ("NORMAL", "MEDIUM")
    
    # Thermal runaway precursor
    res_trp = engine.process_reading(cell_id="STR-01", voltage_v=3.2, current_a=-1.0, temperature_c=58.0)
    assert res_trp["severity"] == "CRITICAL"
    assert any(a["code"] == "IEC_62619_TRP" for a in res_trp["anomalies"])
