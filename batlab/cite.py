"""
Citation helper for batlab and the public datasets its loaders wrap.

cite() with no arguments returns BibTeX for the library itself.
cite(dataset="severson2019") returns BibTeX for one dataset, with its
license/redistribution terms appended as a trailing comment.

Every batlab dataset loader sets df.attrs["citation"] to the dataset key
used here (e.g. "nasa", "severson2019", "oxford2020"), so
cite(dataset=df.attrs["citation"]) always resolves without the caller
needing to know the key mapping.
"""

from __future__ import annotations

_BATLAB_BIBTEX = """@software{batlab,
  title  = {batlab: a citable, honest research library for battery degradation analysis},
  author = {Hosseini, Ali},
  year   = {2026},
  url    = {https://github.com/seyedali1996lb-svg/battery-intelligence-platform},
  note   = {Standardized dataset loaders, leave-cell-out-validated SOH/RUL models, and reproducible benchmark manifests.}
}"""

# Each entry: BibTeX block + the dataset's redistribution/use license, verbatim.
# Page/volume numbers are included where the author is confident of them from
# the well-known published record; anything not independently verified here
# is left out rather than fabricated — check the DOI before a formal citation.
_DATASETS: dict[str, dict[str, str]] = {
    "nasa": {
        "bibtex": """@techreport{saha2007nasa,
  title       = {Battery Data Set},
  author      = {Saha, Bhaskar and Goebel, Kai},
  institution = {NASA Ames Prognostics Center of Excellence (PCoE)},
  year        = {2007},
  url         = {https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/}
}""",
        "license": "Public domain / US government work.",
    },
    "severson2019": {
        "bibtex": """@article{severson2019data,
  title   = {Data-driven prediction of battery cycle life before capacity degradation},
  author  = {Severson, Kristen A. and Attia, Peter M. and Jin, Norman and Perkins, Nicholas and Jiang, Benben and Yang, Zi and Chen, Michael H. and Aykol, Muratahan and Herring, Patrick K. and Fraggedakis, Dimitrios and Bazant, Martin Z. and Harris, Stephen J. and Chueh, William C. and Braatz, Richard D.},
  journal = {Nature Energy},
  volume  = {4},
  pages   = {383--391},
  year    = {2019},
  doi     = {10.1038/s41560-019-0356-8}
}""",
        "license": "Research use per data.matr.io terms (https://data.matr.io/1/) — no explicit open-redistribution "
                    "license published by the authors; verify current terms before redistributing beyond research use.",
    },
    "oxford2020": {
        "bibtex": """@article{raj2020pathdependent,
  title   = {Investigation of Path Dependent Degradation in Lithium-Ion Batteries},
  author  = {Raj, Tara and Wang, Alex A. and Monroe, Charles W. and Howey, David A.},
  journal = {Batteries \\& Supercaps},
  volume  = {3},
  pages   = {1189--1201},
  year    = {2020},
  doi     = {10.1002/batt.202000160}
}""",
        "license": "Open Data Commons Open Database License (ODC-ODbL) v1.0 — "
                    "https://opendatacommons.org/licenses/odbl/1-0/. Redistribution and adaptation are "
                    "permitted provided attribution and share-alike terms are honored.",
    },
    "calce": {
        "bibtex": """@misc{calce_battery_data,
  title        = {CALCE Battery Data},
  author       = {{Center for Advanced Life Cycle Engineering (CALCE), University of Maryland}},
  howpublished = {\\url{https://calce.umd.edu/battery-data}},
  doi          = {10.21227/w9rg-7173}
}""",
        "license": "Open access via calce.umd.edu/battery-data. CALCE requests that any publication "
                    "using this data cite the CALCE article(s) describing the experiments that "
                    "generated it, in addition to the dataset itself — see the source page for the "
                    "specific paper per cell series.",
    },
}


def cite(dataset: str | None = None) -> str:
    """
    Return a BibTeX citation block.

    cite() -> the batlab library's own citation.
    cite(dataset="severson2019") -> that dataset's citation, with its
        license terms appended as a trailing comment line.
    """
    if dataset is None:
        return _BATLAB_BIBTEX

    entry = _DATASETS.get(dataset)
    if entry is None:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Known datasets: {sorted(_DATASETS)}. "
            "Pass no argument to cite the batlab library itself."
        )
    return f"{entry['bibtex']}\n% License: {entry['license']}"


def license_text(dataset: str) -> str:
    """Return just the license/redistribution terms for one dataset (no BibTeX)."""
    entry = _DATASETS.get(dataset)
    if entry is None:
        raise KeyError(f"Unknown dataset {dataset!r}. Known datasets: {sorted(_DATASETS)}.")
    return entry["license"]


def known_datasets() -> list[str]:
    """List every dataset key cite()/license_text() can resolve."""
    return sorted(_DATASETS)
