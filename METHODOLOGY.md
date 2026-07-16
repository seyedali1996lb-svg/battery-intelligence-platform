# Methodology

Every index, factor, and calculation used in this platform, with the actual formula, where it lives in the code, and — critically — whether it's a **measured value**, a **derived/statistical calculation**, a **machine-learned prediction**, a **physics-based simulation**, or an **illustrative assumption**. That distinction is the single most important thing to get right when explaining this platform to a technical audience, so every section below states it explicitly.

The platform is a Streamlit application (`app/`) built on top of `batlab`, an installable Python library (`batlab/`) that does the actual dataset loading, feature engineering, modeling, and validation. Anything with a `batlab.*` path below is in the library and has its own unit tests in `tests/`; anything with an `app/` or `src/` path is application-layer code.

---

## 1. State of Health (SOH)

**What it is:** the cell's current capacity as a percentage of its own first measured capacity.

**Formula** (`batlab/datasets/schema.py::compute_soh_pct`):

```math
\text{SOH}(n) = \frac{Q(n)}{Q(1)} \times 100
```

where $Q(n)$ is the measured discharge capacity (Ah) at cycle $n$, and $Q(1)$ is the cell's own first measured cycle — not a manufacturer nameplate rating. This is the standard definition used throughout the battery-degradation literature (e.g. Birkl et al. 2017) and is computed identically for every dataset loader (NASA, Severson, Oxford, CALCE, and any uploaded data), so it's directly comparable across sources.

**Type:** measured / derived directly from measured capacity. Not a model output, not an estimate.

---

## 2. Remaining Useful Life (RUL) — three different numbers, not one

The platform reports RUL from **three independent methods**, deliberately kept separate rather than blended into one number, because they answer different questions and have different failure modes:

### 2a. RUL training label (ground truth used to train the ML model)

**Formula** (`batlab/features/engineering.py::build_features`):

```math
\text{RUL}(n) = \text{cycle}_{\text{EOL}} - n
```

where $\text{cycle}_{\text{EOL}}$ is the first cycle at which $Q(n) \le Q(1) \times \text{EOL threshold}$ (default 80%, the standard automotive/ESS end-of-life convention). If the cell hasn't reached that threshold yet in the available data, RUL is instead **extrapolated** from the current 50-cycle fade rate:

```math
\text{RUL}(n) = \frac{Q(n) - Q_{\text{EOL}}}{\text{fade\_rate}_{50\text{cy}}}
```

This is the *label* the GBRT model in §6 is trained to predict — not a prediction itself.

### 2b. GBRT point + interval prediction

See §6 below.

### 2c. Physics-based projection (PyBaMM)

See §9 below.

### 2d. Failure-trajectory-memory estimate

See §8 below.

---

## 3. Feature engineering — every derived signal, with its formula

All of the following are computed in `batlab/features/engineering.py::build_features()`. Every one of them is a **deterministic statistical transform of measured data** — none involve machine learning at this stage; ML only enters at the modeling step (§6).

| Feature | Formula | Notes |
|---|---|---|
| `capacity_fade_ah` | $\max(Q(1) - Q(n),\ 0)$ | Absolute capacity lost since cycle 1 |
| `fade_rate_{10,30,50}cy` | rolling mean of $\lvert Q(n) - Q(n-1) \rvert$ over a 10/30/50-cycle window | Three windows trade off noise (short) vs. lag (long) |
| `fade_acceleration` | rolling mean (10cy) of the discrete second difference $Q(n) - 2Q(n-1) + Q(n-2)$ | Sign indicates whether fade is speeding up or slowing down |
| `soh_velocity_50cy` | rolling mean (50cy) of $\text{SOH}(n) - \text{SOH}(n-1)$ | % health lost per cycle |
| `resistance_normalized` | $R(n) / R(1)$ | $R(1)$ = first non-zero resistance reading (guards against a missing-measurement sentinel of 0 in some sources) |
| `resistance_trend_30cy` | rolling mean (30cy) of $R(n) - R(n-1)$ | |
| `temp_rolling_30cy` | rolling mean (30cy) of measured temperature | Separates operating regime from cycle-to-cycle noise |
| `sop_pct` (State of Power) | $\dfrac{R(1)}{\max(R(n), 10^{-6})} \times 100$ | Constant-voltage approximation: peak power $\propto 1/R$. A rate-capability proxy, not a measured power test |
| `ce_rolling_30cy` | rolling mean (30cy) of Coulombic Efficiency ($Q_{\text{discharge}}/Q_{\text{charge}}$) | Only present where the source dataset reports charge capacity |
| `ce_drop_rate` | rolling mean (10cy) of $\text{CE}(n) - \text{CE}(n-1)$ | |
| `c_rate_rolling_10cy` | rolling mean (10cy) of the discharge C-rate | Protocol-known for NASA/synthetic; absent for Severson/uploaded |
| `dod_proxy` | $\text{clip}\!\left(Q(n)/Q(1),\ 0,\ 1\right)$ | Cells cycled at partial depth-of-discharge show a capacity plateau above the EOL threshold; this captures that |
| `cumulative_ah`, `cumulative_kwh` | $\sum_{i=1}^{n} Q(i)$, and the same $\times\,3.7\text{V}/1000$ | Lifetime throughput. 3.7 V is the LiCoO₂ nominal cell voltage used as a fixed conversion constant |
| `capacity_anomaly`, `resistance_anomaly` | $\lvert x(n) - \bar{x}_{30} \rvert > 2.5\,\sigma_{30}$ | Centered 30-cycle rolling mean/std; flags sudden spikes/dips |

### Stress Index — the one feature with an explicit physical derivation

```math
\text{stress\_index}(n) = \left(\frac{C(n)}{1\text{C}}\right)^{0.7} \times \exp\!\left[\frac{E_a}{R}\left(\frac{1}{298.15} - \frac{1}{T(n) + 273.15}\right)\right]
```

- The **C-rate term** uses a sub-linear exponent (0.7, not 1.0) because current distribution inside a porous electrode is non-uniform at high rate — this follows Doyle-Fuller-Newman porous-electrode theory qualitatively, not a fitted coefficient from a specific published dataset.
- The **temperature term** is a standard Arrhenius relation with $E_a/R \approx 6920\,\text{K}$, a typical activation energy for LiCoO₂ SEI growth, referenced to 25 °C (298.15 K).
- The result is dimensionless and normalized to 1.0 at the reference condition (1C, 25 °C). This is the one feature in the pipeline whose functional form is inspired by a physical model rather than being a general literature-motivated statistical proxy — see the inline derivation comment in `engineering.py` for the full reasoning, and that module's own docstring for exactly what is and isn't claimed about its literature citations (they establish that the *category* of signal is an established diagnostic, not that any formula here is transcribed from a numbered equation in a cited paper).

**Type:** all of §3 is derived/statistical, computed deterministically from measured data. None of it is machine-learned or physics-simulated.

---

## 4. Differential capacity (dQ/dV) analysis — ⚠ simulated, not measured

**What it is:** peak position, amplitude, area, and width of the dQ/dV curve, a classic electrode-diagnostic technique (Dubarry & Liaw 2009) that identifies *which* electrode mechanism (loss of active material vs. loss of lithium inventory) is driving fade.

**Why it's flagged:** none of the four datasets this platform loads (NASA, Severson, Oxford, CALCE) expose raw voltage/current time series — only per-cycle summary values (capacity, resistance). A true dQ/dV curve requires the full discharge voltage curve, which isn't available. Instead, `batlab/features/dqdv.py::simulate_vq_curve()` **simulates** a plausible LiCoO₂ discharge voltage curve from a parametric open-circuit-voltage (OCV) polynomial model:

```math
\text{OCV}(\text{SOC}) = 3.7 + 0.7\,\text{SOC} - 0.5\,\text{SOC}^2 + 0.3\,\text{SOC}^3 - 0.1(1-\text{SOC})^3 + 0.08\,e^{-20\,\text{SOC}} - 0.05\,e^{-20(1-\text{SOC})}
```

then computes $dQ/dV$ from that simulated curve numerically via `np.gradient`, and extracts:

- `dqdv_sim_peak_value` — the peak amplitude
- `dqdv_sim_peak_soc` — the SOC at which the peak occurs
- `dqdv_sim_area` — trapezoidal integral of the curve
- `dqdv_sim_fwhm` — full-width-at-half-maximum of the peak

Every one of these column names carries the `_sim_` marker deliberately, so it's visible in the raw feature list or a model-importance table, not just in a docstring someone might not read.

**Type:** physically-motivated simulation, not a measurement and not a fit to real electrochemical data. Treat these as a *proxy* signal the model can use to distinguish cells, not as a real dQ/dV measurement. (This model also assumes a LiCoO₂-shaped OCV curve, which is inapplicable to LFP's flat plateau — the app UI surfaces this caveat for Severson/LFP cells specifically.)

---

## 5. Knee-point detection

**What it is:** the cycle at which capacity fade transitions from a slow plateau into rapid, non-linear decline — the most actionable early-warning signal in fleet management.

**Algorithm** (`batlab/features/knee_detection.py`, L-method / "Kneedle", Satopää et al. 2011, adapted):

1. Smooth SOH with a centered rolling window (default 15 cycles).
2. Normalize both cycle number and SOH to $[0,1]$.
3. Draw a straight line between the first and last point of the normalized curve.
4. The knee is the point of **maximum perpendicular distance** from that line:
```math
d(n) = \frac{\lvert x_{\text{norm}}(n) + y_{\text{norm}}(n) - 1 \rvert}{\sqrt{2}}
```
5. **Confidence** is that maximum distance scaled against the theoretical maximum for a right-angle step ($0.5$), clipped to $[0,1]$.
6. A knee is only reported "detected" if confidence $> 0.15$ and it falls outside the first quarter of the curve (early-cycle noise is excluded by construction).

Why not the second derivative directly: it's noise-sensitive even after smoothing and produces false positives on cells with multi-phase fade. The L-method is more robust on the amount of data these datasets actually provide.

**Type:** deterministic geometric algorithm on measured (smoothed) data. Not machine-learned.

---

## 6. Predictive models — GBRT for SOH and RUL

**What it is:** two Gradient Boosting Regressors (scikit-learn `GradientBoostingRegressor`) — one predicts SOH%, one predicts RUL (cycles) — trained on the feature table from §3, plus two more GBRT models trained with **quantile loss** (α = 0.10 and α = 0.90) to produce an 80% prediction interval around the RUL point estimate.

```
n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
```
(`batlab/models/gbrt.py::GBRT_PARAMS`) — chosen to balance accuracy against training time on ~500–1500 cycle datasets; not exhaustively hyperparameter-tuned.

**Training split:** chronological (`shuffle=False`), never random. Battery data is a time series — cycle 900 has effectively "seen" cycle 100's information if training data is shuffled and split randomly, which would make evaluation misleadingly optimistic. See §7 for why this matters even more at the cell level.

**Explainability:** both models expose scikit-learn's built-in `feature_importances_` (fraction of total loss reduction attributable to each feature across every tree) — this is what powers the "why this prediction?" breakdown in the app, e.g. `batlab/models/gbrt.py::top_drivers()`.

**Uncertainty:** the Q10/Q90 quantile models bound an 80% interval — the true RUL is expected to fall inside `[Q10, Q90]` about 80% of the time if the model is well-calibrated. `train_models()` reports the *empirical* coverage of this interval on the held-out test set (`rul_interval_coverage`) as a self-check, not just an assumed 80%.

**The "Calibrating" tag:** predictions made before cycle 50 are labeled "Calibrating" rather than given as a hard number — the rolling-window features (§3) haven't accumulated enough history to be stable that early, and showing a confident-looking number from noisy early features would be dishonest.

**Type:** machine-learned point estimate + machine-learned uncertainty interval. This is the only place in the platform where a supervised model, rather than a formula, produces the number.

---

## 7. Validation methodology — Leave-Cell-Out (LCO) cross-validation

**This is the credibility argument for the whole platform**, and the thing most worth walking a professor through directly.

**The wrong way (what a naive first attempt does):** concatenate every cell's cycles into one table, shuffle, hold out 20% of *rows* for testing. This reports a deceptively high R² (0.998 on the 4 NASA cells in this platform's own regression-test notebook) because the model has already seen 80%+ of every cell — including cycles adjacent to the "held-out" ones — during training. It's being asked "can you interpolate a cycle you've almost already seen," not "can you generalize to a battery you've never seen."

**The honest way — leave-cell-out (`batlab/validation/lco.py::run_lco`):** hold out an *entire cell*, train on the other N−1 cells, test only on the held-out one, repeat once per cell, average the metrics. On the same 4 NASA cells this reports **R² = 0.806** — the number this platform reports and validates every model against by default.

**How much of that 0.806 is the model actually earning?** `notebooks/02_data_leakage.ipynb` also checks a trivial linear-fit baseline (cycle number → SOH, no engineered features at all) under the identical LCO fold structure: **R² = 0.603**. SOH-vs-cycle curves are smooth, so a dumb straight line already explains most of the variance — the GBRT model and its engineered features are worth a real **+0.203** over that baseline, not the full 0.806. Both numbers are published together deliberately, so the reported accuracy isn't implicitly overclaimed as "all model skill."

**Per-cell reliability gate:** `RUL_RELIABLE_FLOOR = 0.3` (`batlab/validation/lco.py`) — a cell whose individual LCO RUL-R² falls below this shows "Not calibrated" in the UI instead of a number. This is deliberately *per-cell*, not a single dataset-wide average, because the aggregate honest number can hide real per-cell spread: on the 4 NASA cells (`notebooks/02_data_leakage.ipynb`'s own output), per-cell RUL-R² ranges from 0.464 (B0006) to 0.924 (B0018) — all four currently clear the 0.3 floor in this run, but the spread itself is the reason the gate is checked per cell rather than trusting a single averaged number that would mask it.

**Sample size, stated plainly:** NASA gives only 4 cells for LCO (n=4 held-out folds). This is a real, permanent limitation of a small public dataset, not a bug — it's disclosed in the notebooks and in this document rather than hidden. Severson (12 cells) and CALCE (any number, user-supplied) give more folds when used.

**Reproducibility:** `batlab/validation/manifest.py` can export the exact fold assignments, random seed, feature-engineering code version, and installed numpy/pandas/scikit-learn versions that produced a given R² number to a JSON manifest, so a reported result can be independently re-run and checked rather than taken on faith.

**Type:** the validation *methodology* itself, not a calculation that produces a per-cell number — but it's the thing that makes every number in §6 meaningful rather than misleading.

---

## 8. Failure trajectory memory — cosine-similarity pattern matching

**What it is:** a library of "what did the last 50 cycles look like right before other cells failed," queried against the current cell's recent trajectory to flag an early warning before the model's point-estimate RUL alone would.

**Algorithm** (`src/trajectory_memory.py`):

1. For every cell that has crossed the 80% SOH EOL threshold, take its final 50 cycles and compute a **trend vector**: for each of a fixed set of features (fade rate, fade acceleration, SOH velocity, normalized resistance, CE rolling mean, stress index), fit a linear slope over the window and normalize it by that feature's mean absolute value:
```math
\tilde{s}_f = \frac{\text{slope}_f}{\max(\lvert \bar{x}_f \rvert,\ 10^{-9})}
```
This makes the signature scale-invariant — a fast-degrading and a slow-degrading cell with the *same failure mechanism* produce similar-direction vectors even though their absolute magnitudes differ.

2. For the cell currently being monitored, build the same trend vector from its own most recent 50 cycles.

3. Compute **cosine similarity** against every stored signature:
```math
\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\lVert \vec{a} \rVert \, \lVert \vec{b} \rVert}
```
A match requires similarity ≥ 0.65, at least 3 features in common between the two vectors, and — critically — **the same chemistry/source population**. Comparing a real Severson LFP cell's trend vector against a synthetic LiCoO₂ failure signature is not physically meaningful, so cross-source matches are excluded by construction, not just discouraged.

4. If matched, the remaining-cycles estimate is a simple linear interpolation: how far the current cell's SOH has fallen through the matched signature's 100%→80% decline determines what fraction of that signature's 50-cycle window remains, with a ±30% spread band.

**Type:** unsupervised nearest-neighbor pattern matching on measured/derived features — not a trained model, not a physics simulation.

---

## 9. Physics-based RUL projection (PyBaMM)

**What it is:** a second, independent RUL estimate grounded in electrochemistry rather than statistics, used as a cross-check against the GBRT prediction.

**Method** (`src/pybamm_rul.py`):

1. Run one PyBaMM Single Particle Model (SPM) discharge simulation using a published parameter set matched to the cell's chemistry (`Chen2020` for LFP/Severson, `NCA_Kim2011` for NASA, `Marquis2019` for synthetic/LiCoO₂) — this anchors the projection in real electrochemistry rather than an arbitrary curve shape.
2. Fit the classic **SEI-growth square-root fade law** to the cell's own measured SOH history:
```math
\frac{Q(n)}{Q_0} = 1 - \beta \sqrt{n}
```
via `scipy.optimize.curve_fit`, which also returns the parameter's standard error $\sigma_\beta$ from the fit's covariance matrix.
3. Project forward using $\beta$ (central estimate) and $\beta \pm 2\sigma_\beta$ (a ~95% band) until the projected SOH crosses the EOL threshold, giving `rul_physics` plus an optimistic/pessimistic band.

This is genuinely different from the GBRT extrapolation in §6: the PyBaMM parameter set fixes the *chemistry*, while $\beta$ is fit to *this specific cell's* measured fade rate — so it's neither a pure physics simulation (which wouldn't know this cell's actual degradation rate) nor a pure statistical fit (which wouldn't be anchored in an electrochemical model at all).

**In Live Monitor, this re-runs periodically against streamed telemetry** (every 15 new readings, since a full SPM run takes a few seconds) rather than only once against historical data — but it's explicitly labeled as a "physics-consistency re-check," not a continuously-reparameterized live digital twin, since the PyBaMM parameter set itself stays fixed per chemistry rather than being refit from the stream.

**Type:** physics-based simulation (SPM) combined with a statistical fit ($\beta$) to real measured data. Not machine-learned.

---

## 10. Real-time anomaly detection (Live Monitor / MQTT stream)

**What it is:** rule-based flags on live streamed telemetry, most of them tied to explicit IEC 62619:2022 (lithium battery safety standard for industrial applications) thresholds rather than statistically fitted cutoffs.

All in `src/mqtt_stream.py::AnomalyDetector`:

| Flag | Rule | Source |
|---|---|---|
| `UNDERVOLTAGE` / `OVERVOLTAGE` | outside chemistry-specific voltage bounds (e.g. LFP: 2.50–3.65 V, NCA: 2.50–4.20 V) | cell datasheet convention |
| `UNDERTEMPERATURE` | $T < -20°C$ | IEC 62619 §6.2 |
| `OVERTEMPERATURE` | $T > 45°C$ (charging limit) | IEC 62619 §6.2.3 |
| `THERMAL_RUNAWAY_PRECURSOR` | temperature rate-of-rise $> 5°C$ between consecutive readings | IEC 62619 §8.2 |
| `TEMP_RATE_HIGH` | rate-of-rise $> 2°C$ (warning tier, below the critical threshold) | engineering judgment |
| `CAPACITY_PLUNGE` | SOH drop $> 5\%$ in one reading | IEC 62619 §8.2 sudden-loss threshold, possible lithium-plating signal |
| `ZSCORE_{voltage,current,temperature}` | per-channel rolling Z-score $> 2.5$ over a 20-reading window: $z = \lvert(x - \bar{x})/\sigma\rvert$ | standard statistical outlier threshold |
| `MULTI_SIGNAL_ANOMALY` | ≥2 channels each with $z>1.5$ **and** combined Euclidean-norm $\sqrt{\sum z_i^2} > 3.5$ | correlated multi-channel drift is a stronger fault signal than any one channel alone crossing its own threshold — this exists specifically to catch cases no single-channel check can see |

**Type:** deterministic rule-based thresholds, mostly drawn from a named safety standard (IEC 62619) rather than fitted from this platform's own data. The Z-score components are standard statistics, not domain-specific tuning.

---

## 11. Degradation mechanism diagnosis (LLI vs. LAM)

**What it is:** a rule-based classifier distinguishing **Loss of Lithium Inventory** (LLI — SEI growth consuming cyclable lithium) from **Loss of Active Material** (LAM — physical electrode degradation), the two dominant li-ion fade mechanisms.

**Method** (`src/recommendations.py::diagnose_mechanism`) — three independent signals, each contributing points to an `lli_score` / `lam_score`:

1. **CE trend**: declining Coulombic Efficiency slope → +3 LLI; CE deficit from 100% $>0.05\%$ → +2 LLI.
2. **Fade-curve shape**: fit SOH against cycle with a quadratic; strongly negative curvature (accelerating fade) → +3 LAM; positive curvature (decelerating/stabilizing) → +1 LLI.
3. **Resistance rise rate**: fast rise ($>0.02\,\Omega/1000\text{cy}$, normalized) → +2 LAM (particle-contact loss); moderate rise → +1 LLI (SEI-associated).

**Verdict**: LLI if `lli_score > 1.5 × lam_score`, LAM if the reverse, otherwise "Mixed/Insufficient data." Any of the three signals may be unavailable depending on what the source dataset reports (e.g. NASA has no coulombic_efficiency column) — the verdict's confidence is reduced accordingly, not silently assumed.

**Type:** rule-based scoring on statistical fits (linear/quadratic regression slopes) of measured data — not machine-learned, not a single formula but a documented decision procedure.

---

## 12. Second-life economics & sustainability estimates — ⚠ illustrative, not validated

Everything in this section (`src/consequences.py`) is explicitly labeled in its own source code as **either a cited industry estimate or an unsourced illustrative assumption** — never a validated model output. This distinction matters enough that the module's own docstring states it up front, and this document preserves that same honesty rather than dressing these up as more rigorous than they are.

**Second-life value:**
```math
V_{\text{gross}} = Q(n)_{\text{kWh}} \times P_{\$/\text{kWh}}, \qquad V_{\text{net}} = \max(0,\ V_{\text{gross}} - C_{\text{repack}})
```
where $P_{\$/\text{kWh}}$ (default $90/kWh) is a user-adjustable slider seeded from Harper et al. (2019, *Nature*) and NREL (2019) published ranges, and $C_{\text{repack}}$ (default \$10/cell) is explicitly labeled "illustrative — not sourced" (no public figure exists for this line item).

**Break-even SOH**: the SOH at which projected second-life net value drops below the recycling value — the crossover point where the sustainability recommendation should shift from "reuse" to "recycle."

**CO₂ avoided**: reuse avoids manufacturing a new cell (full credit for `co2_manufacture`, a cited IVL Swedish Environmental Research Institute range); recycling instead gets only a partial (~15%) credit, from Dunn et al. (2015)'s estimate that recycled cathode material reduces manufacturing CO₂ by 10–20%.

**What's explicitly NOT computed**: a full lifecycle carbon audit (mining → manufacture → transport → use → end-of-life) and a verified carbon footprint declaration under EU Battery Regulation Article 7 both show as "Not available in this demonstration" in the Compliance/Passport page — this platform does not have the real supply-chain data a genuine declaration would require, and says so rather than fabricating a number.

**Type:** illustrative financial/environmental modeling using cited industry ranges as inputs — explicitly not validated predictions, and not derived from this platform's own measured data at all (only the SOH-to-kWh conversion is).

---

## 13. Simple threshold classifications

A few UI status labels are plain threshold rules, not calculations in their own right — listed here for completeness since a professor asking "what's behind X" may mean these too:

- **Health status** (`app/utils.py::soh_status`): Healthy (SOH ≥ 90%), Degrading (80–89%), End of Life (< 80%).
- **EOL threshold**: 80% SOH by default throughout the platform (`eol_threshold_pct`), the standard automotive/ESS industry convention — user-configurable in Settings.
- **Confidence tag**: "Calibrating" below cycle 50, "Model" thereafter (§6).

---

## Where this is enforced, not just described

Every formula above that isn't a one-line UI threshold has a corresponding unit test in `tests/` (e.g. `tests/test_features.py`, `tests/test_dqdv.py`, `tests/test_knee_detection.py`, `tests/test_lco_eval.py`, `tests/test_trajectory_memory.py`) that checks the actual computed values against known inputs — this document describes what the tests already verify, not a separate claim. See [`CONTRIBUTING.md`](batlab/datasets/CONTRIBUTING.md) and the `notebooks/` directory for runnable, real-data demonstrations of §3, §6, and §7 specifically.

For dataset-level citations (which published paper each dataset itself comes from, and its license), see [`docs/datasets/`](docs/datasets/index.md) or run `batlab.cite(dataset=...)`. This document is about the *calculations this platform performs*, not the provenance of the raw data those calculations run on.
