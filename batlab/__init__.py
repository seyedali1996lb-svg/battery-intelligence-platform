"""
batlab — a citable, honest research library for battery degradation analysis.

Standardized dataset loaders (batlab.datasets), literature-cited feature
engineering (batlab.features), leave-cell-out-validated GBRT SOH/RUL models
(batlab.models), and reproducible benchmark manifests (batlab.validation).

    import batlab
    print(batlab.cite())
"""

from batlab.cite import cite

__version__ = "0.1.1"

__all__ = ["cite", "__version__"]
