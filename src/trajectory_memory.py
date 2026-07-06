"""
Failure trajectory memory.

Builds a library of failure signatures from every cell that reached EOL
in the training data. When a live cell's recent trajectory closely matches
a stored failure signature, raises a TrajectoryMatch warning with a
time-bounded prediction of remaining cycles.

Algorithm
---------
1. For each cell that has crossed the 80% SOH threshold:
   - Extract the WINDOW_CYCLES cycles immediately before EOL.
   - Build a TREND VECTOR: for each feature, compute the linear slope over
     the window, normalized by the feature's mean absolute value.
     This gives a scale-invariant "rate of change" signature that is
     comparable across cells operating at different absolute degradation
     levels (e.g. fast vs slow degraders have the same direction of change
     even if the magnitudes differ).
   - Store along with EOL metadata.

2. For the current cell:
   - Build the same trend vector from its most recent WINDOW_CYCLES.
   - Cosine similarity against every stored signature.
   - If best similarity >= SIMILARITY_THRESHOLD → TrajectoryMatch.

3. EOL estimate:
   - The matching reference cell failed at eol_cycle.
   - The current cell's SOH position within the 100→80% decline
     estimates how far through the failure window it is.
   - remaining cycles ≈ (1 − fraction_through) × window_length

Why trend vectors, not raw curve comparison
-------------------------------------------
Z-scoring a 50-cycle window within that window removes inter-cell
variance (a fast-degrader and a slow-degrader both look like "rising
trend" after per-window z-score). Normalized slopes preserve the
direction AND relative magnitude of each feature's rate of change,
making two cells with the same failure mechanism genuinely similar
even if their absolute values differ.

Dependencies: numpy, scipy (already in requirements).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_CYCLES       = 50    # cycles of pre-EOL trajectory used as signature
EOL_SOH_PCT         = 80.0  # SOH floor defining end-of-life
MIN_WINDOW_LEN      = 15    # minimum usable points in a window
SIMILARITY_THRESHOLD = 0.65 # cosine similarity gate (lower than raw-curve
                             # approach because trend vectors are compact)
MIN_COMMON_FEATURES = 3     # require at least this many shared features

# Features used for trajectory matching.
# Only features present in BOTH the signature and the target cell are used.
MATCH_FEATURES = [
    "fade_rate_30cy",       # primary degradation rate
    "fade_acceleration",    # second derivative — non-linear fade onset
    "soh_velocity_50cy",    # health-loss speed per cycle
    "resistance_normalized",# normalized resistance (scale-invariant)
    "ce_rolling_30cy",      # CE slope → LLI signal (absent for NASA cells)
    "stress_index",         # composite aging driver (absent for Severson)
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FailureSignature:
    cell_id: str
    source: str               # "nasa" | "synthetic" | "severson" | "uploaded"
    eol_cycle: int            # cycle where SOH first crossed EOL_SOH_PCT
    soh_at_window_start: float
    failure_mode: str         # heuristic from CE + resistance signals
    feature_names: list[str]  # features actually in this signature
    trend_vector: np.ndarray  # shape (len(feature_names),) — normalized slopes


@dataclass
class TrajectoryMatch:
    matched_cells: list[str]      # all cell_ids with similarity >= threshold
    similarities: list[float]     # parallel to matched_cells
    best_cell_id: str
    best_similarity: float
    current_cycle: int
    current_soh: float
    predicted_eol_min: int
    predicted_eol_max: int
    cycles_remaining_min: int
    cycles_remaining_max: int
    failure_mode: str
    n_matches: int
    warning_level: str            # "critical" | "high" | "watch"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_from_id(cell_id: str) -> str:
    if cell_id.startswith("B"):
        return "nasa"
    if cell_id.startswith("S-"):
        return "severson"
    return "synthetic"


def _classify_failure_mode(window: pd.DataFrame) -> str:
    """
    Heuristic on the pre-EOL window: classify as LLI, LAM, or Mixed.
    CE slope (negative) → LLI; resistance rise (positive) → LAM.
    """
    lli, lam = False, False
    if "ce_rolling_30cy" in window.columns:
        ce = window["ce_rolling_30cy"].dropna()
        if len(ce) >= 5:
            slope = float(np.polyfit(np.arange(len(ce)), ce.values, 1)[0])
            lli = slope < -4e-5   # CE falling > 4e-5 / cycle
    if "resistance_normalized" in window.columns:
        rn = window["resistance_normalized"].dropna()
        if len(rn) >= 5:
            slope = float(np.polyfit(np.arange(len(rn)), rn.values, 1)[0])
            lam = slope > 0.0015  # resistance rising > 0.15% / cycle
    if lli and lam:
        return "Mixed (LLI + LAM)"
    if lli:
        return "LLI (Loss of Lithium Inventory)"
    if lam:
        return "LAM (Loss of Active Material)"
    return "Mixed / Undetermined"


def _trend_vector(df_window: pd.DataFrame, features: list[str]) -> np.ndarray:
    """
    Build a scale-invariant trend vector for a trajectory window.

    For each feature:
      slope = linear regression coefficient across the window (cycles as x)
      normalized_slope = slope / max(|mean|, epsilon)

    This represents "how fast each signal is changing, relative to its
    magnitude" — directly comparable across cells of different sizes.
    """
    trends = []
    for f in features:
        vals = df_window[f].ffill().bfill().fillna(0.0).values.astype(float)
        if len(vals) < 5:
            trends.append(0.0)
            continue
        x = np.arange(len(vals), dtype=float)
        slope = float(np.polyfit(x, vals, 1)[0])
        mean_abs = max(float(np.abs(vals).mean()), 1e-9)
        trends.append(slope / mean_abs)
    return np.array(trends, dtype=float)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two equal-length 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TrajectoryMemory:
    """
    Library of failure trajectory signatures built from all cells that
    reached EOL. Query with .match() for any currently-monitored cell.
    """

    def __init__(self) -> None:
        self._signatures: list[FailureSignature] = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, all_featured_dfs: dict[str, pd.DataFrame]) -> None:
        """
        Extract failure signatures from every cell that reached EOL.

        all_featured_dfs: {cell_id: df_with_features}
            Pass the union of all sources so the library is as large as
            possible regardless of which source is currently active.
        """
        self._signatures = []
        for cell_id, df in all_featured_dfs.items():
            if df is None or len(df) == 0:
                continue
            if "soh_pct" not in df.columns:
                continue

            eol_mask = df["soh_pct"] < EOL_SOH_PCT
            if not eol_mask.any():
                continue

            eol_cycle = int(df.loc[eol_mask, "cycle_number"].iloc[0])
            pre_eol = df[df["cycle_number"] <= eol_cycle].tail(WINDOW_CYCLES)
            if len(pre_eol) < MIN_WINDOW_LEN:
                continue

            available = [
                f for f in MATCH_FEATURES
                if f in pre_eol.columns and pre_eol[f].notna().sum() >= 5
            ]
            if len(available) < MIN_COMMON_FEATURES:
                continue

            tv = _trend_vector(pre_eol, available)
            soh_start = float(pre_eol["soh_pct"].iloc[0]) if "soh_pct" in pre_eol.columns else 90.0

            self._signatures.append(FailureSignature(
                cell_id            = cell_id,
                source             = _source_from_id(cell_id),
                eol_cycle          = eol_cycle,
                soh_at_window_start= soh_start,
                failure_mode       = _classify_failure_mode(pre_eol),
                feature_names      = available,
                trend_vector       = tv,
            ))

    @property
    def n_signatures(self) -> int:
        return len(self._signatures)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, org_id: int) -> None:
        """Persist the current signature library to SQLite (src/db.py), scoped to org_id."""
        import db
        db.save_failure_signatures(org_id, self._signatures)

    def load(self, org_id: int) -> None:
        """Replace the current signature library with what's persisted for org_id in SQLite."""
        import db
        self._signatures = db.load_failure_signatures(org_id)

    def merge_dedupe_by_cell_id(self, other: list[FailureSignature]) -> None:
        """
        Union self._signatures with `other`, keyed by cell_id. When a cell_id
        appears in both, the version already in self._signatures wins (it was
        just freshly built from the currently-loaded cells, so it's the most
        up-to-date signature for that cell).
        """
        by_id = {sig.cell_id: sig for sig in other}
        by_id.update({sig.cell_id: sig for sig in self._signatures})
        self._signatures = list(by_id.values())

    # ── Query ──────────────────────────────────────────────────────────────

    def match(
        self,
        cell_id: str,
        df: pd.DataFrame,
    ) -> Optional[TrajectoryMatch]:
        """
        Compare the current cell's recent trajectory against all signatures.
        Returns TrajectoryMatch if any signature scores >= SIMILARITY_THRESHOLD,
        else None.

        Cells already at EOL (SOH < 80%) are not checked — the warning is
        only actionable while cycles remain.
        """
        if not self._signatures:
            return None
        if "soh_pct" not in df.columns or len(df) < 15:
            return None

        current_soh = float(df["soh_pct"].iloc[-1])
        if current_soh < EOL_SOH_PCT:
            return None   # already at EOL

        current_cycle = int(df["cycle_number"].iloc[-1])
        recent = df.tail(WINDOW_CYCLES)

        matches: list[tuple[float, FailureSignature]] = []

        current_source = _source_from_id(cell_id)

        for sig in self._signatures:
            if sig.cell_id == cell_id:
                continue
            # Cosine similarity on normalized slope vectors is only physically
            # meaningful within the same chemistry — a real Severson LFP
            # cell's degradation curve shape has no principled relationship
            # to a synthetic LiCoO2 cell's, so cross-source "matches" were
            # methodologically bogus (this is what was driving a majority of
            # a real Severson fleet being flagged against synthetic failure
            # signatures). Restrict candidates to the querying cell's own
            # source; if that source has no EOL signatures of its own yet,
            # the honest answer is "no match available", not a borrowed one.
            if sig.source != current_source:
                continue

            # Use only features present in both signature and current window
            common = [
                f for f in sig.feature_names
                if f in recent.columns and recent[f].notna().sum() >= 3
            ]
            if len(common) < MIN_COMMON_FEATURES:
                continue

            # Build comparable trend vectors using only common features
            sig_idx  = [sig.feature_names.index(f) for f in common]
            sig_tv   = sig.trend_vector[sig_idx]
            cur_tv   = _trend_vector(recent, common)
            sim      = _cosine_sim(sig_tv, cur_tv)

            if sim >= SIMILARITY_THRESHOLD:
                matches.append((sim, sig))

        if not matches:
            return None

        matches.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_sig = matches[0]

        # ── EOL estimate ──────────────────────────────────────────────────
        # How far through the 100%→80% SOH decline is the current cell?
        soh_range = max(1.0, best_sig.soh_at_window_start - EOL_SOH_PCT)
        fraction  = max(0.0, min(1.0, (best_sig.soh_at_window_start - current_soh) / soh_range))
        cycles_elapsed      = fraction * WINDOW_CYCLES
        cycles_remaining_c  = max(5, int(WINDOW_CYCLES - cycles_elapsed))
        spread              = max(10, int(cycles_remaining_c * 0.30))
        cycles_remaining_min = max(1, cycles_remaining_c - spread)
        cycles_remaining_max = cycles_remaining_c + spread
        predicted_eol_min   = current_cycle + cycles_remaining_min
        predicted_eol_max   = current_cycle + cycles_remaining_max

        warning_level = (
            "critical" if cycles_remaining_min <= 20 else
            "high"     if cycles_remaining_min <= 40 else
            "watch"
        )

        return TrajectoryMatch(
            matched_cells        = [s.cell_id for _, s in matches],
            similarities         = [sim for sim, _ in matches],
            best_cell_id         = best_sig.cell_id,
            best_similarity      = best_sim,
            current_cycle        = current_cycle,
            current_soh          = current_soh,
            predicted_eol_min    = predicted_eol_min,
            predicted_eol_max    = predicted_eol_max,
            cycles_remaining_min = cycles_remaining_min,
            cycles_remaining_max = cycles_remaining_max,
            failure_mode         = best_sig.failure_mode,
            n_matches            = len(matches),
            warning_level        = warning_level,
        )

    def match_fleet(
        self,
        all_featured_dfs: dict[str, pd.DataFrame],
    ) -> dict[str, TrajectoryMatch]:
        """
        Run match() across all cells. Returns {cell_id: TrajectoryMatch}
        for cells with matches only.
        """
        results = {}
        for cell_id, df in all_featured_dfs.items():
            m = self.match(cell_id, df)
            if m is not None:
                results[cell_id] = m
        return results


# ---------------------------------------------------------------------------
# Reconciliation — primary RUL model vs. trajectory-match model
# ---------------------------------------------------------------------------

DISAGREEMENT_RATIO_THRESHOLD = 0.40  # relative difference above which the two
                                      # estimates are treated as disagreeing,
                                      # not just imprecise


def reconcile_rul_estimates(
    primary_cycles_remaining: "float | None",
    match: "TrajectoryMatch | None",
) -> dict:
    """
    Compare the primary RUL model's cycles-remaining estimate against the
    trajectory-match model's estimate for the same cell, and decide whether
    they should be shown as two independent, both-confident widgets, or as
    one reconciled "models disagree" message.

    Two models for the same cell used to be rendered as fully independent
    call-outs (Overview's hero RUL card + the separate Trajectory Match
    card) with no arbitration — e.g. "657 cycles remaining, reliable" next
    to "HIGH confidence, 35-65 cycles remaining" for the same cell, an
    8x+ disagreement presented as if both were simply facts. This function
    is the arbitration step: when the two diverge beyond
    DISAGREEMENT_RATIO_THRESHOLD, the caller should replace both
    independent call-outs with a single card built from this function's
    output, biased toward the more conservative (shorter) estimate.

    Returns a dict:
        {
            "disagree": bool,
            "primary_cycles": float | None,
            "match_cycles_mid": float | None,
            "ratio": float | None,          # relative difference, 0-1+
            "favor": "primary" | "match" | None,  # the more conservative one
        }
    """
    if match is None or primary_cycles_remaining is None:
        return {
            "disagree": False,
            "primary_cycles": primary_cycles_remaining,
            "match_cycles_mid": None,
            "ratio": None,
            "favor": None,
        }

    match_mid = (match.cycles_remaining_min + match.cycles_remaining_max) / 2.0
    denom = max(primary_cycles_remaining, match_mid, 1.0)
    ratio = abs(primary_cycles_remaining - match_mid) / denom
    disagree = ratio > DISAGREEMENT_RATIO_THRESHOLD
    favor = "primary" if primary_cycles_remaining <= match_mid else "match"

    return {
        "disagree": disagree,
        "primary_cycles": primary_cycles_remaining,
        "match_cycles_mid": match_mid,
        "ratio": ratio,
        "favor": favor if disagree else None,
    }
