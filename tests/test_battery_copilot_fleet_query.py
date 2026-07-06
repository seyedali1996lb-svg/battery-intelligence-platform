"""Unit tests for battery_copilot.answer_fleet_query() — the "Ask the fleet"
natural-language front door on the Fleet page (U5). Uses a hand-built
fleet_stats dict matching the shape build_fleet_stats() produces, rather
than running the full feature pipeline."""

from battery_copilot import answer_fleet_query


def _make_fleet_stats(unreliable=None):
    rows = [
        {"cell_id": "B0005", "source": "nasa", "soh": 92.0, "cycle": 40,
         "fade_30": 0.5, "resistance": 0.1, "rul_reliable": True, "rul_pred": 300},
        {"cell_id": "B0006", "source": "nasa", "soh": 78.0, "cycle": 140,
         "fade_30": 1.8, "resistance": 0.15, "rul_reliable": True, "rul_pred": 20},
        {"cell_id": "B0007", "source": "nasa", "soh": 85.0, "cycle": 100,
         "fade_30": 1.0, "resistance": 0.12, "rul_reliable": False, "rul_pred": None},
    ]
    return {
        "rows": rows,
        "n_cells": len(rows),
        "soh_mean": 85.0, "soh_median": 85.0, "soh_min": 78.0, "soh_max": 92.0,
        "fade_mean": 1.1,
        "eol_cells": ["B0006"],
        "degrading_cells": ["B0007"],
        "sorted_by_soh": sorted(rows, key=lambda r: r["soh"]),
        "sorted_by_fade": sorted(rows, key=lambda r: r["fade_30"], reverse=True),
        "unreliable_rul": unreliable if unreliable is not None else ["B0007"],
    }


def test_budget_question_routes_to_replacement_budget():
    stats = _make_fleet_stats()
    answer = answer_fleet_query("What will replacement cost over the next 12 months?", stats)
    assert "Replacement Budget" in answer


def test_risk_question_routes_to_fleet_risk():
    stats = _make_fleet_stats()
    answer = answer_fleet_query("What is the business risk in my fleet?", stats)
    assert "Risk Assessment" in answer


def test_alerts_question_routes_to_alerts():
    stats = _make_fleet_stats()
    answer = answer_fleet_query("What are the current fleet alerts?", stats)
    assert "B0006" in answer  # the one EOL cell should be named


def test_routed_answer_appends_reliability_caveat_when_cells_unreliable():
    stats = _make_fleet_stats(unreliable=["B0007"])
    answer = answer_fleet_query("fleet alerts", stats)
    assert "Reliability note" in answer
    assert "B0007" in answer.split("Reliability note")[1]


def test_no_caveat_appended_when_all_cells_reliable():
    stats = _make_fleet_stats(unreliable=[])
    answer = answer_fleet_query("fleet alerts", stats)
    assert "Reliability note" not in answer


def test_unmatched_question_returns_honest_fallback_not_fabricated_answer():
    stats = _make_fleet_stats()
    answer = answer_fleet_query("why is B0018 losing capacity so fast", stats)
    assert "I can answer fleet-level questions about" in answer
    assert "per-cell Copilot" in answer
