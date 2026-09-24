# Design: Technical Case Study — Battery Degradation & Physics-Based Modelling

**Date:** 2026-09-24
**Status:** approved by user (cell, model strategy, dQ/dV scope, report format decided in-session)
**Scope:** additive. No existing module is redesigned, removed, or repurposed.

---

## 1. Problem

The repository is a wide battery analytics platform: five dataset loaders, a leave-cell-out
validation harness, an ML layer, a Streamlit/React UI. Its validation and honesty discipline
(`METHODOLOGY.md`, `docs/overclaiming_audit.md`) is genuinely strong.

What it does not have is **one end-to-end battery engineering narrative that a battery engineer
can read top to bottom**. Four specific gaps were confirmed by inspection:

| Gap | Evidence |
|---|---|
| No equivalent circuit model | `grep` for `equivalent circuit\|thevenin\|ECM\|RC` → zero matches. `src/eis_model.py` is a *forward* Nyquist simulator from assumed constants; it never fits data. |
| dQ/dV is simulated, not measured | `batlab/features/dqdv.py` differentiates a LiCoO₂ OCV polynomial, and prefixes every output `dqdv_sim_`. Its docstring states raw time series is unavailable. |
| No characterization plots | Schema is summary-level (one row per cycle). No V-vs-t, I-vs-t, V-vs-Q anywhere. |
| No terminal-voltage model validation | All validation is SOH/RUL R². Nothing asks whether a model reproduces measured voltage, and no physical model has a train/validation split. |

**The unlock:** `data/raw/B0006.mat` already contains full per-cycle time series
(`Voltage_measured`, `Current_measured`, `Temperature_measured`, `Time`, `Capacity`) plus 278
impedance records with measured `Re` and `Rct`. The raw data needed for the entire requested
pipeline is in the repo; only the summary row was ever parsed.

**Goal:** one credible, reproducible case study — raw data → conclusions — that reads as a
junior battery engineer / MSc-level project, not an AI dashboard.

---

## 2. Dataset and cell

**NASA PCoE Battery Data Set #5, cell B0006.** Chosen by the user from the four available cells.

Measured characteristics (probed directly, not assumed):

- 616 cycle records: **170 charge, 168 discharge, 278 impedance**.
- **Discharge:** constant **−2.01 A**, ~9.4 s median sample interval, 179–371 points per record,
  2792–3690 s duration, terminates near 2.44 V (protocol cutoff 2.7 V; undershoot observed).
- **Charge:** CC 1.51 A → CV taper to 4.2 V, ~2.9 s median interval, then a trailing rest
  (median **932 s**) settling to 4.18–4.20 V.
- **Impedance:** `Re` 0.0612 → 0.0736 Ω, `Rct` 0.0785 → 0.1000 Ω, 278 records, zero NaN.
- **Fade:** 2.0353 → 1.1857 Ah reported discharge capacity; crosses 80% SOH in-window.
- Chemistry: LiCoO₂ (18650, ~2 Ah nominal), 24 °C ambient — from the loader's own documented
  protocol constants (`batlab.datasets.nasa`), not invented.

Known artifacts found during inspection, which the quality stage must detect:

- Voltage samples > 4.35 V (max observed 8.083 V) and < 2.0 V (min 2.121 V).
- |I| > 3 A startup transients inside charge records (min −4.481 A).
- `Voltage_load` / `Current_load` zero-padding at record boundaries.
- Coulombic totals over life: **262.49 Ah in / 259.87 Ah out → 0.9900 lifetime ratio**, while
  naive per-cycle CE computes to ≈0.975–0.99 — not credible as a Faradaic efficiency.

---

## 3. Architecture

New subpackage inside the installable, pyright-checked, pytest-covered library:

```
batlab/case_study/
    __init__.py              public API + version constant
    raw.py                   NASA .mat -> per-cycle time-series records (additive raw loader)
    quality.py               data-quality checks -> QualityReport
    preprocess.py            segmentation, charge/discharge isolation, ∫I·dt capacity, outlier policy
    metrics.py               Q(n), CE(n), R0_onset(n), T(n); characterization figure generation
    soh.py                   SOH definition, reference capacity, EOL criterion
    degradation.py           fade, EIS Re/Rct growth, V-curve evolution, measured dQ/dV
    ecm.py                   first-order Thevenin model + simulation
    estimation.py            R0, OCV(z), R1, C1 estimation
    validation.py            chronological split + RMSE / MAE / max-error
    pybamm_crosscheck.py     unfitted PyBaMM SPM physics baseline
    report.py                REPORT.md + figures + results.json
    run.py                   pipeline orchestrator
    __main__.py              `python -m batlab.case_study`
```

Supporting files:

```
tests/test_case_study.py
docs/case_study.md                    full narrative doc, mkdocs nav entry
README.md                             new section (see §9)
reports/case_study/                   generated output (gitignored except README stub)
```

### Reuse

- `batlab.datasets._paths.raw_data_dir()` — data location resolution.
- `batlab.datasets.nasa` — protocol constants, citation, chemistry; the raw loader sits beside it.
- `batlab.features.knee_detection` — knee point on the SOH curve.
- `batlab.datasets.schema.compute_soh_pct` — consistent SOH convention.
- `batlab.cite` — citation strings for the report.

### Non-goals (explicitly out of scope)

- No change to the summary schema, `FEATURE_COLUMNS`, `run_lco`, or the GBRT/hierarchical models.
- No change to `batlab/features/dqdv.py` — its simulated path remains for summary-only datasets
  (user decision: additive only).
- No new UI pages, no new API endpoints, no new dependencies beyond those already installed
  (numpy, pandas, scipy, matplotlib, pybamm).
- No synthetic data anywhere in the case study.

---

## 4. Pipeline stages

### 4.1 `raw.py` — raw data

`load_raw_cell(cell_id) -> RawCell`, parsing `data/raw/{cell_id}.mat` via `scipy.io.loadmat`
(`simplify_cells=True`) into per-record dataclasses carrying `kind` (charge/discharge/impedance),
`time_s`, `voltage_v`, `current_a`, `temperature_c`, `reported_capacity_ah`, and for impedance
`re_ohm`, `rct_ohm`. Current sign convention preserved as measured: **discharge negative,
charge positive**. No filtering at this stage — this is the raw layer.

### 4.2 `quality.py` — data quality check

`run_quality_checks(raw) -> QualityReport`, a dataclass of named checks, each returning
`{n_flagged, examples, threshold, disposition}`. Checks:

1. Missing values (NaN/inf) per channel per record.
2. Non-monotonic or duplicated timestamps.
3. Voltage plausibility: flag V outside [2.0, 4.35] V.
4. Current plausibility: flag |I| > 3 A during charge (startup transients).
5. Zero-padded load channels at record boundaries.
6. Duplicate records (identical time+voltage+current arrays).
7. Sample-interval consistency vs the record's median dt.
8. Capacity outliers across cycles (robust z-score, |z| > 3.5).
9. Reported-vs-integrated capacity agreement (informational, not a rejection).

Every check reports **counts and dispositions**; nothing is silently dropped. The report renders
as a table in `REPORT.md`.

### 4.3 `preprocess.py` — preprocessing

`preprocess(raw, quality) -> PreprocessedCell`, with each decision recorded in
`preprocessing_log: list[PreprocessingStep]` (step, rationale, n_affected) — rendered in the
report so no transformation is invisible.

- **Cycle segmentation:** group records into electrochemical cycles in file order
  (charge → discharge → impedance), preserving the source's own `type` field as ground truth.
- **Charge/discharge separation:** isolate each phase; derive ±2 A discharge windows and
  CC/CV charge windows from measured current.
- **Rest segment detection:** contiguous |I| < 0.05 A runs with ≥ 3 samples; used for OCV anchors.
- **Capacity calculation:** `Q = ∫|I| dt / 3600` by trapezoid over the discharge window, compared
  against the reported `Capacity` field. Measured agreement: mean **+0.94%**, std 0.35%, max
  +1.63% — reported as a transparency metric, not a correction.
- **Outlier handling:** artifacts flagged by `quality.py` are excluded *per-window*, with counts.
  Physical measurements are never clipped or imputed to look plausible.
- **Voltage/current/time consistency:** assert equal lengths and finite values per record.

### 4.4 `metrics.py` — battery characterization

Per-cycle metrics plus publication-quality figures (matplotlib, `Agg` backend, no seaborn):

- `capacity_ah(n)` — integrated and reported, both kept.
- `coulombic_efficiency(n)` — Q_discharge / Q_charge, **with a credibility flag** (§6).
- `r0_onset_ohm(n)` — ΔV/ΔI at discharge onset. Measured: 0.106 → 0.132 Ω over life.
- `temperature_c(n)` — mean over discharge.
- Figures: `fig01_voltage_time`, `fig02_current_time`, `fig03_voltage_capacity`,
  `fig04_capacity_vs_cycle`, `fig05_coulombic_efficiency`, `fig06_temperature_vs_cycle`.

### 4.5 `soh.py` — state of health

```
SOH(n) = Q(n) / Q_ref × 100
```

Defined explicitly in code and report:

- `Q_ref` = **mean of the first 3 valid discharge capacities** (guards against a single
  anomalous formation cycle; the exact cycles used are printed in the report).
- **EOL criterion:** SOH ≤ 80% (standard automotive/ESS convention, consistent with the rest of
  the repository).
- **Assumptions:** same-temperature comparison, discharge-capacity-based, no resistance term.
- Reported alongside the alternative `Q_ref = Q(1)` for comparability with `compute_soh_pct`.

Figure: `fig07_soh_vs_cycle`, with the EOL line and the knee point from
`batlab.features.knee_detection`.

### 4.6 `degradation.py` — degradation analysis

Separates **demonstrated** from **interpretation** in both code (`findings: list[Finding]` with
`kind: "demonstrated" | "hypothesis"`) and prose.

Demonstrated (directly measured):

1. **Capacity fade** — absolute and %/cycle, linear fit over early/late windows, knee location.
2. **Resistance growth** — three independent series: EIS `Re`, EIS `Rct` (278 real points), and
   `R0_onset` from the voltage step. Growth reported for each; their *disagreement* is discussed
   (§5.1).
3. **Voltage curve evolution** — overlay of V-vs-Q at early/mid/late cycles.
4. **Measured dQ/dV** — computed from real V-Q data via `np.gradient` on monotone-resampled
   curves, with peak position/area tracked across cycles. Prefix `dqdv_meas_` to distinguish
   from the existing `dqdv_sim_` columns.
5. **Coulombic-efficiency trend** — trend only (§6).

Hypotheses (explicitly not claimed as proven): LLI vs LAM attribution, SEI growth. Each carries
a `required_measurement` string — e.g. reference/three-electrode cell, EIS distribution of
relaxation times, post-mortem electrode microscopy — stated in `REPORT.md`.

Figures: `fig08_resistance_growth`, `fig09_dqdv_evolution`, `fig10_voltage_curve_evolution`.

### 4.7 `ecm.py` — battery model

First-order Thevenin equivalent circuit:

```
V(t) = OCV(z(t)) − I(t)·R0 − V1(t)
dV1/dt = −V1/(R1·C1) + I/C1
dz/dt  = −I / (3600 · Q_nominal)
```

Sign convention follows the measured data (**discharge current negative ⇒ z decreases**).

`ECMParams` dataclass: `r0_ohm, r1_ohm, c1_f, q_nominal_ah, ocv` (callable z → V).
`simulate(params, current_a, time_s, z0) -> v_terminal, soc` — explicit solution of the linear
RC ODE (exponential exact update, not Euler), so the simulation itself introduces no
discretisation error at 9.4 s sampling.

### 4.8 `estimation.py` — parameter estimation

Staged, because constant-current discharge makes R0 and R1 **partly degenerate** (both appear as
a single IR drop). The degeneracy is disclosed rather than hidden.

1. **R0 — independent measurement, not a fit output.** Median ΔV/ΔI across the discharge onset
   step on calibration cycles. Cross-checked against EIS `Re`; the disagreement (onset ≈ 0.116 Ω
   vs EIS Re ≈ 0.061 Ω) is reported with a physical explanation: within the 10 s sample
   interval the film and charge-transfer contributions have already begun to respond, so the
   onset step measures R0 + partial R1, while EIS `Re` is the high-frequency intercept alone.
2. **OCV(z)** — monotone PCHIP through relaxation anchors (charge trailing rest ⇒ z ≈ 1,
   discharge trailing rest ⇒ z ≈ 0) refined by a bounded polynomial fit.
   *Identifiability caveat:* NASA records have no mid-SOC rest, so interior OCV is inferred from
   the fit under a monotonicity constraint — stated in Limitations.
3. **R1, C1** — `scipy.optimize.least_squares` (Trust Region Reflective, physical bounds
   R>0, C>0) minimising squared voltage residual over calibration-cycle discharge curves with
   OCV and R0 held fixed.

Documented per parameter: physical meaning, estimation method, assumptions, limitations.

### 4.9 `validation.py` — model validation

- **Split: chronological, 60% calibration / 40% validation, never shuffled** — matching the
  repository's own `batlab.validation.prospective` philosophy. Cycles, not rows, are the unit.
- **Voltage:** RMSE, MAE, max absolute error in **mV**, computed globally and per cycle on the
  validation window.
- **Capacity:** simulate each validation discharge to the 2.7 V cutoff → predicted capacity vs
  measured, reported as RMSE/MAE in Ah and %.
- Figures: `fig11_model_validation` (measured vs predicted V, held-out), `fig12_validation_errors`
  (residual trace + histogram), `fig13_ecm_params_vs_cycle`.

### 4.10 `pybamm_crosscheck.py` — physics-based baseline

Unfitted PyBaMM **SPM** with `parameter_sets["Ramadass2004"]` (LiCoO₂ — chemistry-matched),
solved at 2 A discharge. Its voltage-curve RMSE against measured data is reported as the
**physics baseline the fitted ECM must beat**, on the same validation cycles.

Constraints that keep this non-decorative:

- PyBaMM is imported lazily inside this module; if unavailable, the report records
  "PyBaMM cross-check not run" rather than fabricating a number.
- The Ramadass2004 parameter set describes a *different* cell, so absolute capacity will differ
  from B0006. Comparison is made on the SOC-normalised voltage curve, and the parameter-set
  mismatch is stated as a limitation — the number measures *curve shape*, not capacity.

Verified working in this environment: PyBaMM 26.6.2, SPM solve ≈ 0.2 s.

### 4.11 `report.py` + `run.py` — conclusions and reproducibility

`run.py` executes the stages in order and writes `reports/case_study/`:

- `REPORT.md` — Problem → Dataset → Quality → Preprocessing → Characterization → SOH →
  Degradation → Model → Estimation → Validation → Engineering Interpretation → Limitations,
  with images referenced by relative path.
- `results.json` — every headline number, machine-readable and diff-able (mirrors
  `batlab.cli --out` encoding conventions).
- `fig*.png` — all figures.

`__main__.py` exposes the single documented command:

```
python -m batlab.case_study
```

---

## 5. Engineering interpretation (what REPORT.md must answer)

1. What happened to the battery?
2. How quickly did it degrade (rate, in %/cycle and Ah/cycle, with the knee if detected)?
3. Which degradation signatures are visible, each tagged demonstrated vs hypothesis?
4. What does the model capture well?
5. Where does the model fail (worst cycles, end-of-life region, rest transitions)?
6. Engineering implications.
7. What additional measurements would be needed to identify mechanisms confidently?

### 5.1 The two honest findings already known to be reportable

- **R0 disagreement:** onset-step resistance (~0.116 Ω) vs EIS `Re` (~0.061 Ω). Not an error —
  different physical quantities measured by different means. Reported as a finding.
- **Coulombic efficiency:** absolute level ≈0.98 is not physically credible for a healthy cell.
  Only the *relative trend* is interpreted; the lifetime ratio 0.9900 is reported separately as
  the more trustworthy aggregate.

---

## 6. Scientific honesty rules (binding on implementation)

- No fabricated results. Every number in `REPORT.md` and `results.json` is computed at runtime.
- No invented chemistry. Chemistry comes from `batlab.datasets.nasa.CHEMISTRY` / documented
  protocol constants.
- No mechanism claimed because a curve "looks like" it. LLI/LAM/SEI are hypotheses with stated
  required measurements.
- No synthetic data. If any synthetic fixture appears in tests, it is labelled synthetic and
  never reaches report output.
- No accuracy reported that was not computed. Metrics are computed on the held-out window only.
- If the data cannot support an analysis, say so in Limitations and implement the strongest valid
  alternative.
- CE credibility limitation and OCV identifiability limitation must both appear in Limitations.

---

## 7. Testing

`tests/test_case_study.py`, using synthetic fixtures explicitly labelled as such:

| Test | Asserts |
|---|---|
| Capacity integration | ∫I·dt on a known synthetic waveform recovers Q to 1e-6 |
| SOH | Q/Q_ref × 100; Q_ref averaging; EOL threshold crossing detected |
| Cycle segmentation | synthetic charge/discharge/rest sequence segments correctly |
| Quality checks | injected NaN / non-monotonic t / V=8 V / I=−4 A each flagged with correct counts |
| ECM prediction | I=0 ⇒ V→OCV; constant I ⇒ steady state V = OCV − I(R0+R1); τ = R1C1 recovered |
| Parameter estimation | synthetic ECM data with known R0/R1/C1 recovered within tolerance |
| Validation metrics | RMSE/MAE/max match hand-computed values |
| Split integrity | calibration and validation cycles are disjoint, ordered, and cover the cell |
| Raw loader | parses the real B0006.mat if present, else `pytest.skip` |
| dQ/dV | measured peak located on a synthetic V-Q curve with a known peak |

---

## 8. Documentation

- `docs/case_study.md` — full narrative, added to `mkdocs.yml` nav.
- `METHODOLOGY.md` — one new section covering the case study's SOH/ECM/validation formulas,
  keeping its measured/derived/predicted/simulated taxonomy intact.
- `tests/test_docs_nav.py` must still pass (nav entries asserted against committed sources).

---

## 9. README section

New top-level section **"Technical Case Study: Battery Degradation & Physics-Based Modelling"**,
placed after `## Validation`, containing: Problem · Dataset · Methodology · Battery Engineering
Concepts · Results · Model · Validation · Engineering Interpretation · Limitations · How to
Reproduce — with 3–4 representative embedded figures, written so a battery engineer understands
the work without reading the codebase.

---

## 10. Acceptance criteria

1. `python -m batlab.case_study` runs from a clean checkout with `data/raw/B0006.mat` present and
   regenerates every figure and `REPORT.md`.
2. `pytest tests/test_case_study.py` passes.
3. Full suite and `pyright` (0 errors) still pass — no existing behaviour changed.
4. Every headline number in `REPORT.md` traces to `results.json`.
5. Limitations contain, at minimum: CE credibility, OCV identifiability, single-cell scope,
   PyBaMM parameter-set mismatch, CC-discharge R0/R1 degeneracy.
