"""
ML-based (unsupervised) anomaly detection over cycle-level features.

The platform already has two anomaly surfaces, both rule-based: the
named-threshold AnomalyDetector (src/mqtt_stream.py, IEC 62619:2022 rules)
and the streaming CUSUM / Mahalanobis engine (src/streaming_analytics.py).
Both fire on named rules — they can only catch what a rule anticipated.
This module adds the complementary unsupervised signal: an Isolation Forest
(sklearn) learns the *normal region of a cell's own feature space* from
its historical cycles, then flags cycles whose state is novel relative to
that history — including patterns no named rule anticipated (a slow
joint drift in capacity+resistance that no single-channel threshold
crosses, an unusual operating temperature for THIS cell, etc.).

Honest scope, stated plainly:
  1. Unsupervised novelty detection, not fault classification. A flagged
     cycle is "unusual for this cell's own history", not "this specific
     fault". Flags are review signals to feed the rule-based engines /
     operator triage, not diagnoses.
  2. The threshold is contamination-based (default 5% of a cell's cycles
     are assumed anomalous) — a modeling assumption, not a physical limit.
  3. Isolation Forest scores are relative: the same absolute score means
     different things across cells, so the report returns each cell's own
     score distribution rather than a global cutoff.
  4. Warmup handling: the fade-rate feature needs a 30-cycle rolling
     window, so a cell's first 30 cycles carry no full rolling history.
     They are honestly reported as unscored warmup (not scored on a
     fabricated feature, and not allowed to teach the forest an artificial
     early-cycle cluster).
  5. The feature matrix is deliberately small and deterministic (capacity,
     fade rate, resistance growth, temperature) so the module works on the
     standard cycle schema without pulling the full ML feature pipeline;
     it is a health-signal detector, not a second training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Fewer scored cycles than this, and a per-cell IsolationForest fit is noise.
MIN_CYCLES_FOR_FIT = 30

# Default assumed fraction of anomalous cycles per cell (contamination).
DEFAULT_CONTAMINATION = 0.05

# Rolling window for the fade-rate feature (matches the platform's
# fade_rate_30cy idea; the first `_FADE_WINDOW` cycles are warmup).
_FADE_WINDOW = 30

_REQUIRED_COLUMNS = {"cycle_number", "capacity_ah"}


def _score_or_none(v) -> "float | None":
    """Normalize a score value for JSON emission: NaN/None -> None (JSON
    null), numbers -> rounded float. A DataFrame round-trip turns Python
    None into NaN, which json.dumps would serialize as invalid JSON
    ('NaN') — the API must emit real nulls for unscored warmup rows."""
    if v is None:
        return None
    f = float(v)
    if np.isnan(f):
        return None
    return round(f, 4)


def _note_or_none(v) -> "str | None":
    """Same DataFrame-round-trip normalization for the per-cycle note
    column: None becomes NaN in a str-typed column, which must become a
    real JSON null again on emission."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


class MLAnomalyDetector:
    """Per-cell Isolation Forest novelty detector over cycle-level features.

    fit(df) learns the cell's normal region; score(df) (or fit_predict(df))
    returns per-cycle scores. Scores follow sklearn's convention: lower =
    more anomalous, negative = outside the learned normal region."""

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        if not (0.0 < contamination < 0.5):
            raise ValueError("contamination must be in (0, 0.5).")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: "IsolationForest | None" = None
        self.feature_columns: list = []

    # ------------------------------------------------------------------
    # Feature matrix — deterministic, from the standard cycle schema
    # ------------------------------------------------------------------

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Cycle data is missing required columns {sorted(missing)} — "
                f"expected at least {sorted(_REQUIRED_COLUMNS)}."
            )
        if len(df) < MIN_CYCLES_FOR_FIT:
            raise ValueError(
                f"Need at least {MIN_CYCLES_FOR_FIT} cycles for a meaningful "
                f"per-cell IsolationForest fit; this cell has {len(df)}."
            )

        out = pd.DataFrame(index=df.index)
        out["capacity_ah"] = df["capacity_ah"].astype(float)

        # Rolling fade rate over a 30-cycle window — NaN during warmup
        # (the first _FADE_WINDOW cycles have no full window). The NaN is
        # deliberate: warmup rows are excluded from fit and scored as
        # unscored warmup rather than fitted on a fabricated value.
        cap = out["capacity_ah"]
        out["fade_rate_30cy"] = (cap - cap.shift(_FADE_WINDOW)).abs()

        if "soh_pct" in df.columns and df["soh_pct"].notna().any():
            out["soh_pct"] = df["soh_pct"].astype(float)
        else:
            first = float(cap.iloc[0])
            out["soh_pct"] = cap / first * 100.0

        if "resistance_ohm" in df.columns and df["resistance_ohm"].notna().any():
            res = df["resistance_ohm"].astype(float)
            out["resistance_ohm"] = res
            first_res = float(res.iloc[0])
            out["resistance_growth_pct"] = res / first_res * 100.0

        if "temperature_c" in df.columns and df["temperature_c"].notna().any():
            out["temperature_c"] = df["temperature_c"].astype(float)

        self.feature_columns = list(out.columns)
        return out

    # ------------------------------------------------------------------
    # Fit / score
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MLAnomalyDetector":
        """Learn the cell's normal region from its own cycle history.
        Warmup rows (no full fade-rate rolling history) are excluded from
        the fit — fitting on them would teach the forest a distinct
        early-cycle cluster. Raises ValueError when fewer than
        MIN_CYCLES_FOR_FIT scored cycles remain."""
        X = self._feature_matrix(df)
        X_valid = X.dropna()
        if len(X_valid) < MIN_CYCLES_FOR_FIT:
            raise ValueError(
                f"Need at least {MIN_CYCLES_FOR_FIT} cycles with full rolling "
                f"history for a meaningful per-cell IsolationForest fit; this "
                f"cell has {len(df)} cycles, of which only {len(X_valid)} are "
                f"scorable (the first {_FADE_WINDOW} are warmup for the "
                f"fade-rate feature)."
            )
        self.model = IsolationForest(
            contamination=self.contamination,  # pyright: ignore[reportArgumentType]
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(X_valid)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-cycle anomaly scores (sklearn convention: lower = more
        anomalous, negative = outlier) and is_anomaly flags. Must be called
        after fit() (or use fit_predict() for the one-shot path). Warmup
        rows are NOT scored — they get anomaly_score=None and
        is_anomaly=False with an honest warmup note."""
        if self.model is None:
            raise RuntimeError(
                "MLAnomalyDetector.score() requires fit() first — "
                "use fit_predict() for a one-shot call."
            )
        X = self._feature_matrix(df)
        X_valid = X.dropna()
        scores = self.model.score_samples(X_valid)  # higher = more normal
        # Flip to anomaly-score convention: higher = more anomalous.
        anomaly_scores = -scores
        threshold = float(np.quantile(anomaly_scores, 1.0 - self.contamination))

        result = []
        for idx, row in X.iterrows():
            cycle_number = int(df.loc[idx, "cycle_number"])
            if row.isna().any():
                result.append({
                    "cycle_number": cycle_number,
                    "anomaly_score": None,
                    "is_anomaly": False,
                    "note": f"warmup — first {_FADE_WINDOW} cycles, no full fade-rate rolling history; not scored.",
                })
                continue
            pos = X_valid.index.get_loc(idx)
            result.append({
                "cycle_number": cycle_number,
                "anomaly_score": round(float(anomaly_scores[pos]), 4),
                "is_anomaly": bool(anomaly_scores[pos] >= threshold),
                "note": None,
            })
        return pd.DataFrame(result)

    def fit_predict(self, df: pd.DataFrame) -> dict:
        """One-shot: fit on a cell's cycles, score every cycle, return a
        report dict (see detect_anomalous_cycles())."""
        self.fit(df)
        scored = self.score(df)
        scored_valid = scored[scored["anomaly_score"].notna()]
        flagged = scored_valid[scored_valid["is_anomaly"]]
        normal = scored_valid[~scored_valid["is_anomaly"]]
        n_warmup = int((~scored["anomaly_score"].notna()).sum())
        return {
            "n_cycles": len(scored),
            "n_scored": int(len(scored_valid)),
            "n_warmup_unscored": n_warmup,
            "n_flagged": int(flagged.shape[0]),
            "contamination_assumed": self.contamination,
            "feature_columns": list(self.feature_columns),
            "flagged_cycles": [int(c) for c in flagged["cycle_number"]],
            "anomaly_score_min": round(float(scored_valid["anomaly_score"].min()), 4),
            "anomaly_score_max": round(float(scored_valid["anomaly_score"].max()), 4),
            "normal_score_max": round(float(normal["anomaly_score"].max()), 4) if len(normal) else None,
            "per_cycle": [
                {"cycle_number": int(r["cycle_number"]), "anomaly_score": _score_or_none(r["anomaly_score"]),  # pyright: ignore[reportArgumentType]
                 "is_anomaly": bool(r["is_anomaly"]),
                 "note": _note_or_none(r.get("note"))}
                for _, r in scored.iterrows()
            ],
            "caveats": [
                "Unsupervised novelty detection, not fault classification — a flagged cycle is unusual for this cell's own history, not a diagnosed fault.",
                "The flag threshold is contamination-based (assumed fraction of anomalous cycles), a modeling assumption, not a physical limit.",
                "IsolationForest scores are relative per cell — compare a cell against its own history, not against other cells' scores.",
                "Review flagged cycles against the rule-based engines (IEC 62619 rules in src/mqtt_stream.py, CUSUM/Mahalanobis in src/streaming_analytics.py) before acting.",
            ],
        }


def detect_anomalous_cycles(df: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION) -> dict:
    """One-shot unsupervised ML anomaly scan of one cell's cycle history.

    df must carry the standard cycle schema columns (cycle_number,
    capacity_ah, optionally resistance_ohm / temperature_c / soh_pct).
    Raises ValueError with an honest message when the cell has too few
    scored cycles (a per-cell fit that small is noise)."""
    return MLAnomalyDetector(contamination=contamination).fit_predict(df)


def detect_fleet_anomalies(
    cell_data: dict,
    contamination: float = DEFAULT_CONTAMINATION,
) -> dict:
    """Run detect_anomalous_cycles() per cell and aggregate into a fleet
    report: {cell_id: report} plus a summary of how many cells flagged any
    cycles. Cells too small for a fit are listed in `skipped` with the
    honest reason rather than silently dropped."""
    summary = {"n_cells": len(cell_data), "n_flagged_cells": 0, "n_flagged_cycles": 0, "skipped": {}}
    per_cell = {}
    for cell_id, df in cell_data.items():
        try:
            report = detect_anomalous_cycles(df, contamination=contamination)
        except ValueError as exc:
            summary["skipped"][cell_id] = str(exc)
            continue
        per_cell[cell_id] = report
        if report["n_flagged"] > 0:
            summary["n_flagged_cells"] += 1
            summary["n_flagged_cycles"] += report["n_flagged"]
    return {"summary": summary, "per_cell": per_cell}
