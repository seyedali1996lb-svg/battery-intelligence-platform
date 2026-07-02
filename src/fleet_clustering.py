"""
Cross-fleet degradation clustering.

Clusters cells by their degradation fingerprint so that a cell's cluster
history (how similar cells have degraded) can supplement individual RUL
predictions — especially for cells whose fold R² is below the reliability floor.

Fingerprint vector per cell:
  [fade_rate_30cy, resistance_normalized, temp_rolling_30cy,
   fade_acceleration, soh_pct (latest), cycle_number (normalized)]

Clustering: KMeans (k=3 by default — Healthy / Degrading / Critical archetypes).
Falls back gracefully if sklearn is unavailable or if fewer than 3 cells exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# Archetype labels assigned by centroid SOH ranking (highest SOH → Healthy)
_ARCHETYPE_LABELS = {0: "Healthy", 1: "Degrading", 2: "Critical"}
_ARCHETYPE_COLORS = {
    "Healthy":   "#48bb78",
    "Degrading": "#d69e2e",
    "Critical":  "#e53e3e",
}


@dataclass
class CellCluster:
    cell_id: str
    cluster_id: int
    archetype: str
    color: str
    fingerprint: dict[str, float]   # raw feature values
    peer_cells: list[str]           # other cells in the same cluster
    cluster_soh_median: float
    cluster_rul_median: float | None


@dataclass
class ClusterResult:
    assignments: dict[str, CellCluster]   # cell_id → CellCluster
    n_clusters: int
    feature_names: list[str]
    inertia: float | None


def build_fingerprint(df: pd.DataFrame) -> dict[str, float]:
    """Extract the degradation fingerprint from a cell's featured DataFrame."""
    latest = df.iloc[-1]
    return {
        "fade_rate_30cy":       float(latest.get("fade_rate_30cy", 0.0)) * 1000,
        "resistance_normalized":float(latest.get("resistance_normalized", 1.0)),
        "temp_rolling_30cy":    float(latest.get("temp_rolling_30cy", 25.0)),
        "fade_acceleration":    float(latest.get("fade_acceleration", 0.0)) * 1000,
        "soh_pct":              float(latest.get("soh_pct", 100.0)),
        "cycle_normalized":     float(latest.get("cycle_number", 0.0)) / 500.0,
    }


def cluster_fleet(
    featured_dfs: dict[str, pd.DataFrame],
    n_clusters: int = 3,
    random_state: int = 42,
) -> ClusterResult | None:
    """
    Cluster all cells in featured_dfs by degradation fingerprint.

    Returns None if fewer than n_clusters cells are available or if
    sklearn is not installed.
    """
    if len(featured_dfs) < max(n_clusters, 2):
        return None

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    cell_ids   = list(featured_dfs.keys())
    prints     = {cid: build_fingerprint(df) for cid, df in featured_dfs.items()}
    feat_names = list(next(iter(prints.values())).keys())

    X = np.array([[fp[f] for f in feat_names] for fp in prints.values()])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_sc)

    # Rank clusters by their centroid SOH (descending) so label 0 = Healthy
    centroid_soh = [
        X[labels == k, feat_names.index("soh_pct")].mean()
        for k in range(n_clusters)
    ]
    rank_map = {
        old: new
        for new, old in enumerate(np.argsort(centroid_soh)[::-1])
    }
    labels_ranked = np.array([rank_map[l] for l in labels])

    # Build cluster-level statistics
    cluster_soh_medians  = {}
    cluster_rul_medians  = {}
    for k in range(n_clusters):
        members = [cell_ids[i] for i, l in enumerate(labels_ranked) if l == k]
        sohs    = [float(featured_dfs[m].iloc[-1]["soh_pct"]) for m in members]
        cluster_soh_medians[k] = float(np.median(sohs)) if sohs else 0.0
        ruls = []
        for m in members:
            r = featured_dfs[m].iloc[-1].get("rul_pred")
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                ruls.append(float(r))
        cluster_rul_medians[k] = float(np.median(ruls)) if ruls else None

    assignments = {}
    for i, cid in enumerate(cell_ids):
        k         = int(labels_ranked[i])
        arch      = _ARCHETYPE_LABELS.get(k, f"Cluster {k}")
        peers     = [cell_ids[j] for j, l in enumerate(labels_ranked) if l == k and j != i]
        assignments[cid] = CellCluster(
            cell_id=cid,
            cluster_id=k,
            archetype=arch,
            color=_ARCHETYPE_COLORS.get(arch, "#718096"),
            fingerprint=prints[cid],
            peer_cells=peers,
            cluster_soh_median=cluster_soh_medians[k],
            cluster_rul_median=cluster_rul_medians[k],
        )

    return ClusterResult(
        assignments=assignments,
        n_clusters=n_clusters,
        feature_names=feat_names,
        inertia=float(km.inertia_),
    )


def archetype_color(archetype: str) -> str:
    return _ARCHETYPE_COLORS.get(archetype, "#718096")
