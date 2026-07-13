---
title: 'batlab: a citable, honest research library for battery degradation analysis'
tags:
  - Python
  - battery
  - lithium-ion
  - state-of-health
  - remaining-useful-life
  - machine-learning
  - reproducibility
authors:
  - name: Ali Hosseini
    orcid: 0009-0000-9005-363X
    affiliation: 1
affiliations:
  - name: Politecnico di Torino
    index: 1
date: 13 July 2026
bibliography: paper.bib
---

<!--
Resolved (as of 2026-07-13):
  - ORCID: 0009-0000-9005-363X, verified resolving, published name confirmed
    as "Ali Hosseini" (matches this paper's author name throughout).
  - Affiliation: "Politecnico di Torino".
  - Acknowledgements: "No external funding was received for this work."
  - AI-usage disclosure: applied, with explicit author sign-off.
  - "State of the field" (beep/PyBaMM) claims re-checked against current
    README/homepage — both hold up. Not an exhaustive docs read; worth a
    final skim if either project's scope has visibly changed by actual
    submission time.
  - Archive DOI: 10.5281/zenodo.21346275 (v0.1.1). This is a version-specific
    DOI, not a concept DOI (one that stays constant across future releases) —
    check your Zenodo GitHub settings page for the concept DOI if you'd
    rather cite that instead. JOSS's submission form asks for this as a
    separate "archive DOI" field, not something written into this file's body.

Still open — TODO(human author) before submitting to JOSS:
  - Research impact statement (below) is honestly "no external users yet" —
    replace once there's something real to report (adoption, downloads,
    citations, a dependent package). Don't inflate it, don't delete it
    while the honest answer is still "not yet".
-->

# Summary

Lithium-ion battery degradation — the gradual loss of a cell's usable capacity (State of Health,
SOH) and the cycles remaining before it crosses a usability threshold (Remaining Useful Life,
RUL) — is a widely studied machine-learning problem, but the published literature and available
tooling make it easy to report an inflated, non-reproducible accuracy number without noticing.
`batlab` is a Python library that packages three things researchers in this space need but
usually have to rebuild themselves for every project: (1) standardized loaders for public
cycling datasets (NASA PCoE [@saha2007nasa], Severson 2019 [@severson2019data], Oxford
Path-Dependent 2020 [@raj2020pathdependent], CALCE CS2 [@calce_battery_data]) that all return one
documented DataFrame schema instead of four incompatible ad hoc shapes; (2) literature-cited
feature engineering and gradient-boosted SOH/RUL models validated with leave-cell-out (LCO)
cross-validation rather than a row-level split; and (3) a reproducible-benchmark manifest format
that records the exact fold assignments, random seed, and feature-engineering version behind a
reported number, so it can be checked rather than taken on faith.

# Statement of need

A recurring, quantified failure mode in applied battery-ML work is evaluating a model with a
row-level train/test split on data pooled across multiple cells. Because consecutive cycles of
the same cell are highly autocorrelated, a random row-level split leaks near-neighbor cycles from
the same cell into both the training and test sets — the model is not being asked to generalize
to an unseen cell, only to interpolate between cycles it has effectively already seen. `batlab`'s
own worked example (`notebooks/02_data_leakage.ipynb`) reproduces this on four NASA PCoE cells:
a naive row-level split reports SOH R² = 0.998, while leave-cell-out — training on three cells,
testing on a fourth held out entirely — reports R² = 0.806 on the identical data and model. The
gap is not a minor calibration difference; it is the difference between a number that describes
the model's actual ability to generalize to a new cell and one that does not.

Researchers evaluating a new feature, model architecture, or dataset for battery SOH/RUL
prediction currently have to write dataset-specific parsers, decide how to split their data, and
implement their own reliability gating from scratch — work that is redone, inconsistently, across
many papers and repositories, and rarely open-sourced in a form another group can directly reuse.
`batlab` is aimed at that researcher: install it, load one of four public datasets in one
standardized schema, and get an LCO-validated baseline with an honest, per-cell reliability
number (`RUL_RELIABLE_FLOOR`) in minutes, or extend it with a fifth dataset loader following the
mechanical checklist in `batlab/datasets/CONTRIBUTING.md`.

# State of the field

`batlab` is complementary to, not competing with, the two most relevant existing open-source
projects in this space:

- **PyBaMM** [@sulzer2021pybamm] is a physics-based battery simulator (single-particle,
  DFN, and related electrochemical models). It *simulates* how a cell should behave given a
  parameter set; it does not load or validate cycling datasets, and it is not a machine-learning
  library. `batlab` and PyBaMM answer different questions — this project's own Streamlit
  application (the demo consumer of this library) uses PyBaMM's SEI-fade projection alongside
  `batlab`'s data-driven GBRT model specifically to compare a physics prior against a
  data-driven one, which only makes sense because the two tools do not overlap in scope.
- **beep** [@herring2020beep] (Toyota Research Institute) is the closest prior art for
  standardizing battery cycling data structuring — it defines a per-datapoint and per-cycle
  "summary" schema and a pipeline for structuring raw cycler files (Arbin, Maccor, BioLogic,
  ...) into it. `batlab`'s dataset schema (`batlab/datasets/schema.py`) is deliberately narrower:
  it standardizes only the cycle-summary level (one row per cycle), not beep's raw per-datapoint
  structuring or its multi-vendor file-format parsing, and it adds `soh_pct` as a first-class,
  always-present column rather than leaving SOH computation to downstream consumers. `batlab`
  also does not attempt beep's early-prediction modeling layer; its own modeling code is
  restricted to the LCO-validated GBRT SOH/RUL baseline plus the reproducible-manifest format,
  which beep does not provide. A project already using beep for raw data structuring could adopt
  `batlab`'s LCO validation and manifest format on top of beep's own summary tables with modest
  adaptation, since both use a per-cycle-row convention.

Beyond these two, most public battery-ML work ships as a one-off analysis notebook or repository
tied to a single paper's dataset and model, without a reusable, documented dataset-loader or
validation-manifest layer other researchers can import directly — which is the gap `batlab`
targets.

# Software design

`batlab` is split into four subpackages with a narrow, explicit dependency direction:
`batlab.datasets` (loaders, schema contract, no dependency on the other three), `batlab.features`
(pure functions of a schema-conformant DataFrame), `batlab.models` (GBRT training/prediction, no
dependency on `datasets`), and `batlab.validation` (leave-cell-out evaluation and the
reproducible split-manifest format, depending only on `features`). Dataset-specific heavy
dependencies (`h5py` for Severson's MATLAB v7.3 files, `mat-io` for Oxford's MCOS-serialized
MATLAB tables, `openpyxl` for CALCE's Arbin Excel exports) are optional extras, not core
dependencies, so installing `batlab` to use one loader does not require every loader's parser.
Every loader either auto-downloads from a stable, ungated URL and caches locally (NASA, Severson,
Oxford), or — when no such URL exists, as for CALCE — raises a specific, instructive exception
naming exactly where to obtain the data and where to place it, rather than presenting a fake or
silently-broken automated download.

# Research impact statement

`batlab` was extracted from an existing, actively developed battery fleet-management
application (this repository's `app/`) into a standalone library in this release; it has no
external users, citations, or dependent projects yet. TODO(human author): replace this paragraph
once there is real evidence to report — adoption by a specific research group, a dependent
package, download counts, or citations in other work. A JOSS reviewer will check this section
against what's actually verifiable (e.g. PyPI download stats, GitHub stars/forks/dependents,
citing papers); do not inflate it, and do not delete it if the honest answer is still "not yet."

# AI usage disclosure

This software and paper were produced through an AI-assisted development process: essentially all
code (the `batlab` package, its test suite, the four worked notebooks, and this documentation
site) was implemented by an AI coding assistant (Claude, Anthropic), operating from a detailed
specification and interactive direction from the human author across an extended session. The
human author set the sprint's scope and constraints, made every product decision the assistant
surfaced as a choice (e.g. what platform functionality to defer removing, whether to auto-download
a new dataset by default), and reviewed the assistant's work at each stage rather than writing the
implementation directly. Citation details (every BibTeX entry in `paper.bib` and `batlab/cite.py`)
and several factual claims in this paper (dataset licenses, the beep/PyBaMM comparison, the
Zenodo DOI, the ORCID) were independently verified against primary sources — publisher DOI
records, project documentation, or live resolution checks — during development, specifically
because a citable research library's value proposition depends on that holding up under scrutiny.

# Acknowledgements

No external funding was received for this work.

# References
