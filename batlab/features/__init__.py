"""Literature-cited feature engineering: fade rate, dQ/dV, knee detection, CE/resistance trends, stress index."""

from batlab.features.dqdv import add_dqdv_features, extract_dqdv_features, simulate_vq_curve
from batlab.features.engineering import (
    FEATURE_COLUMNS,
    TARGET_RUL,
    TARGET_SOH,
    build_features,
    feature_summary,
    get_model_matrix,
)
from batlab.features.knee_detection import degradation_phases, detect_knee

__all__ = [
    "build_features",
    "get_model_matrix",
    "feature_summary",
    "FEATURE_COLUMNS",
    "TARGET_SOH",
    "TARGET_RUL",
    "detect_knee",
    "degradation_phases",
    "simulate_vq_curve",
    "extract_dqdv_features",
    "add_dqdv_features",
]
