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
- **§4.12 (the chemistry/parameter-set fix) is the one deliberate exception to "no change to
  existing functionality"** — added on explicit user instruction ("fix the app as a whole") after
  it surfaced as a verified correctness defect. Its blast radius is measured in §4.12: no model
  input and no gated metric changes.

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

- `Q_ref` = **the first valid discharge capacity, `Q(1)`** (primary). This is the definition
  `batlab.datasets.schema.compute_soh_pct` and `METHODOLOGY.md` §1 already use, so the case
  study's SOH numbers are directly cross-checkable against the rest of the platform — a reader
  who compares `fig07` against the app's own SOH view for B0006 gets the same curve. It is also
  the convention in Birkl et al. 2017, which the repository already cites.
- **A noise guard was considered and rejected on the data, not on principle.** The originally
  proposed `Q_ref = mean(first 3 valid cycles)` exists to protect against an anomalous formation
  cycle. Measured on B0006, the first six discharge capacities are
  `2.0353, 2.0251, 2.0133, 2.0133, 2.0005, 2.0139 Ah`: cycle 1 is the **maximum** and the fade is
  smooth and monotone, so the guard defends against a condition this cell does not exhibit — and
  it would cost `SOH(1) = 100.53% > 100%` plus comparability with every other SOH number in the
  repository. `Q(1)` is used instead.
- **`Q_ref` sensitivity is reported, not assumed.** `results.json` and `REPORT.md` carry the EOL
  cycle computed under *both* definitions (`Q(1)` and `mean(first 3)` = 2.0246 Ah) and the
  cycle-count difference between them. This converts a definition choice from an unexamined
  assumption into a demonstrated robustness result: if the EOL cycle barely moves, the conclusion
  is not an artefact of the reference choice.
- **EOL criterion:** SOH ≤ 80% (standard automotive/ESS convention, consistent with the rest of
  the repository).
- **Assumptions:** same-temperature comparison, discharge-capacity-based, no resistance term.

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
- The Ramadass2004 parameter set describes a *different* cell (nominal 1.0 Ah against B0006's
  2.035 Ah measured), so **absolute capacity is reported but explicitly framed as a finding, not
  as a fair score**. Two numbers are produced deliberately, for different reasons:
  1. **Voltage RMSE vs SOC** — the fair comparison. Both models are evaluated on the same
     validation cycles over a common normalised-SOC axis, so curve *shape* is what is scored.
  2. **Absolute capacity error** — reported as-is, with the reason stated in plain language: the
     ECM's `Q_nominal` is estimated from *this* cell, while the PyBaMM set describes *a
     different* cell. That gap is precisely the quantity that measures how much cell-specific
     parameters matter, so it is a result worth reporting rather than a number to hide.
- The report states plainly that the fitted ECM has home-field advantage on the capacity metric,
  so the comparison is not presented as a like-for-like contest it is not.
- `Ramadass2004` is chosen over `NCA_Kim2011` (the repository's existing NASA anchor) because
  `Ramadass2004`'s positive-electrode OCP is `lico2_ocp_Ramadass2004` — genuinely LiCoO₂, matching
  the chemistry this repository declares for NASA — whereas `NCA_Kim2011` loads
  `nca_ocp_Kim2011`, a nickel-cobalt-aluminium cathode. See §4.12.

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

### 4.12 App-wide chemistry/parameter-set consistency fix (added by user decision)

**Scope added after §4.10 surfaced a pre-existing defect: the repository declares NASA cells as
LiCoO₂ in the loader, the README, the docs, the sidebar and the Health page, but anchors them on
an NCA PyBaMM parameter set. User instruction: "Go with your recommendation and fix the app as a
whole."**

The defect is broader than the single row first noticed — *both* shipped anchors are
cathode-mismatched, because nothing in the codebase ever checked that an anchor's parameter set
was chemistry-matched to the chemistry string it was keyed on:

| declared `(source, chemistry)` | before | positive-electrode OCP PyBaMM actually loads | after | window: declared → set |
|---|---|---|---|---|
| `("nasa", "LiCoO2")` | `NCA_Kim2011` | `nca_ocp_Kim2011` ❌ NCA | **`Ramadass2004`** `lico2_ocp_Ramadass2004` ✅ | 2.7 V → 2.8 V (Δ 0.1) |
| `("severson2019", "LFP")` | `Chen2020` | `nmc_LGM50_ocp_Chen2020` ❌ NMC811 | **`Prada2013`** `LFP_ocp_Afshar2017` ✅ | **2.0 V → 2.0 V (exact)** |

Both replacements are verified correct on **two independent axes** — cathode identity *and*
voltage window:

- `Ramadass2004`: `lico2_ocp_Ramadass2004` matches the declared `LiCoO2`; its 2.8–4.2 V window
  matches NASA's 4.2 V charge / 2.7 V discharge protocol (`batlab/datasets/nasa.py`) to 0.1 V.
  `NCA_Kim2011` matched the window exactly but had the wrong cathode — window fit was almost
  certainly what the original selection optimised for. Chemistry wins: the positive-electrode OCP
  curve is the dominant term in an SPM discharge trajectory, so a 0.1 V window offset is
  second-order next to an entirely different cathode.
- `Prada2013`: `LFP_ocp_Afshar2017` matches the declared `LFP`, and its 2.0 V lower cutoff
  matches Severson's **exactly** — `batlab/datasets/severson.py` sets
  `voltage_discharge_cutoff_v = 2.0`, cited to Severson et al., *Nature Energy* 4, 383–391.
  `Chen2020` misses by 0.5 V *and* has an NMC cathode.

**Measured blast radius (this is the reason the change is safe).** Probed by swapping each anchor
in-process and diffing `calibrate_cell()` field by field: only `param_set`, `chem_label` and
`spm_capacity_ah` move. `beta_sei`, `beta_lam`, `k_r`, `fit_r2` and `dominant_mode_key` come back
**bit-identical**, because they are `scipy.optimize.curve_fit` against the cell's own measured
history and PyBaMM never participates in them. `physics_spm_capacity_ah` is explicitly *not* in
`FEATURE_COLUMNS` (`batlab/features/engineering.py`), so it is display-only. Therefore:

> **metric-gate impact: NONE — measured, not argued.**
>
> The measurement that actually supports this is `run_lco()` run twice over the real four NASA
> cells (B0005, B0006, B0007, B0018) with `use_fold_cache=False` — once with the old anchors,
> once with the new — diffing **every** reported field:
>
> ```
> === DIFF (old -> new), every reported field ===
>   (no field differs)
> soh_r2: 0.9470975121947384 -> 0.9470975121947384   UNCHANGED
> rul_r2: 0.421595059300934  -> 0.421595059300934    UNCHANGED
> ```
>
> with `physics_features = True` in both runs, i.e. the parameter set *was* resolved and the
> display column *was* populated — the model still did not move. The recovered SOH R²
> (0.9471) and RUL R² (0.4216) match the values `docs/history.md` records for this fleet.
>
> **An earlier draft of this section cited the CI metric gate as the confirmation. That would have
> been wrong, and the distinction matters.** `GATE_FLEET_CELL_IDS = ["Cell1","Cell3","Cell5",
> "Cell6","Cell8"]` is the *synthetic* fleet, and synthetic cells are deliberately absent from
> `ANCHOR_PARAM_SETS` — so during a gate run the parameter set is never consulted and the physics
> columns are all-NaN. The gate passing after this change is structurally guaranteed and
> therefore carries **zero** evidential weight about it. The gate does pass (5/5 in
> `tests/test_ci_metric_gate.py`), but that is a consistency check, not proof of this claim; the
> `run_lco()` diff above is the proof. Asserting otherwise would be exactly the kind of
> borrowed-confidence assurance this repository's methodology exists to prevent.
>
> Full suite after the change: **1993 passed, 11 skipped**; `pyright`: **0 errors**.

**Full edit inventory:**

1. `batlab/features/physics_calibration.py` — both `ANCHOR_PARAM_SETS` values; `_CHEM_LABEL`
   gains `Ramadass2004` and `Prada2013`; four docstrings/comments that assert the old mapping.
2. `src/pybamm_rul.py` — `_PARAM_MAP["nasa"]` and `["severson"]` (these two must track
   `ANCHOR_PARAM_SETS`; `physics_calibration.py:236` documents that invariant explicitly, so
   changing one without the other would silently break a stated contract); `_CHEM_LABEL` gains
   both; module docstring's parameter-set table.
3. **New regression test** — for *every* entry in `ANCHOR_PARAM_SETS`, assert the PyBaMM
   parameter set's positive-electrode OCP function name contains the cathode marker for the
   declared chemistry. This is the actual cure: the defect survived one prior cleanup pass
   (`docs/history.md:120` records an earlier "NASA mislabeled NCA in 5 places" fix that left the
   anchor behind) precisely because nothing enforced the invariant.
4. `METHODOLOGY.md`, `docs/performance.md`, `docs/history.md` — three stale statements of the old
   mapping.
5. `tests/test_physics_calibration.py` — five assertions that pin the *buggy* value.

**Deliberately left unchanged, with reasons:**

- `ANCHOR_PARAM_SETS` *keys* — eligibility is keyed on `(source, chemistry)`; only values change,
  so no cell gains or loses calibration eligibility.
- `zhu2022 → NCA_Kim2011` — correct. Zhu 2022 is an NCM+NCA blend and the existing comment
  honestly says no parameter set matches it exactly.
- `calce/synthetic/uploaded → Marquis2019` — **chemistry-correct** (`lico2_ocp_Dualfoil1998` is
  LiCoO₂). Its 3.105–4.1 V window is a poor match for a 2.7–4.2 V protocol, which is a genuine
  sub-optimality, but it is *not* the defect being fixed. Changing it would move displayed numbers
  on the application's default fleet without correcting an error, so it is out of scope and
  recorded here as a known non-change rather than silently "improved".
- `docs/history.md` entry 20 — that is a record of what was true when written; rewriting history
  would be its own dishonesty. Only `docs/history.md:46`, a *current-state* dataset table reading
  the garbled `"LiCoO₂ NCA 18650"`, is corrected.

**Honesty note carried into the case study's Limitations:** NASA's own dataset documentation does
not definitively specify the cathode chemistry. "LiCoO₂" is this repository's *declared* claim,
consistent across its loader, docs and UI. The fix makes the physics anchor consistent with that
declaration; it does not independently verify the declaration against the primary source.

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

**Already landed, from §4.12** (not part of `test_case_study.py`; they guard the chemistry fix):

| Test | Asserts |
|---|---|
| `test_every_anchor_is_cathode_matched` | For every `ANCHOR_PARAM_SETS` entry, PyBaMM's positive-electrode OCP function name carries the declared chemistry's cathode marker — the invariant both original anchors violated |
| `test_anchor_param_map_stays_in_sync_with_pybamm_rul` | The library's and the demo app's parameter-set dicts agree, since `_param_set_for_cell` documents that they must |
| `test_every_param_map_target_has_a_chem_label` | `project_rul` indexes `_CHEM_LABEL[param_set]` directly — an unlabelled target is a `KeyError` raised when a user opens a page |
| `test_live_monitor_physics_twin_check_runs_against_streamed_telemetry` | Now asserts `"LiCoO₂" in chemistry model` **and** `"NCA" not in …`; previously asserted the opposite (`"NCA" in`), pinning the very leftover the earlier cleanup missed |

The last row is worth stating plainly, because it is the defect in miniature:
`test_nasa_cells_never_labeled_nca_chemistry` asserted NASA is never labelled NCA, while this test
asserted NASA's chemistry model *is* NCA — two tests in one suite on opposite sides of the same
question, both passing. The earlier cleanup matched the literal display phrase `"NCA chemistry"`,
which `"NCA/Graphite (Kim 2011)"` does not contain. That is why the cure is a check on the
**cathode OCP function PyBaMM actually loads** rather than on any display string: strings can be
partially cleaned; physics either matches or it does not.

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
