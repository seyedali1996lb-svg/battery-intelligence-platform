"""Standardized battery-cycling dataset loaders. See batlab.datasets.schema for the contract every loader returns."""

from batlab.datasets.schema import SchemaError, compute_soh_pct, validate_schema
from batlab.datasets.nasa import load_nasa_cells
from batlab.datasets.oxford import load_oxford_cells
from batlab.datasets.severson import load_severson_cells

__all__ = [
    "validate_schema",
    "compute_soh_pct",
    "SchemaError",
    "load_nasa_cells",
    "load_severson_cells",
    "load_oxford_cells",
]
