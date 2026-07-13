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
    orcid: 0000-0000-0000-0000  # TODO(human author): fill in real ORCID before submission
    affiliation: 1
affiliations:
  - name: TODO(human author) — independent researcher / institutional affiliation
    index: 1
date: 13 July 2026
bibliography: paper.bib
---

<!--
TODO(human author) before submitting to JOSS:
  - Fill in the real ORCID and affiliation above.
  - Confirm the AI-usage disclosure section below accurately describes how this
    paper and the underlying software were produced, and adjust it if it doesn't.
  - Confirm "State of the field" claims about beep/PyBaMM against their latest
    documentation before submission — this was written from the author's/
    assistant's knowledge, not a fresh read of their current source.
  - Add a real research-impact statement once batlab has external usage to cite
    (downloads, dependent papers, or adoption) — the current draft is honest
    that there isn't one yet, which JOSS reviewers should see rather than a
    fabricated one, but check whether that's still true at submission time.
  - Decide whether to keep the AI-authorship acknowledgement in Acknowledgements
    or move it into the disclosure section only — either is defensible, pick one
    and be consistent.
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

# AI usage disclosure

Significant portions of this software's implementation, tests, notebooks, and this paper's prose
were drafted with the assistance of an AI coding assistant (Claude, Anthropic), directed and
reviewed by the human author. Citation details (BibTeX entries in `paper.bib` and
`batlab/cite.py`) were independently verified against publisher/DOI records during development
rather than generated from the assistant's unverified recall, specifically because a citable
research library's core value proposition depends on that accuracy. TODO(human author): confirm
this description remains accurate at submission time, and expand it if any section's authorship
split should be described more specifically per JOSS's current AI-disclosure policy.

# Acknowledgements

TODO(human author): acknowledge any funding, institutional support, or individual contributors
here.

# References
