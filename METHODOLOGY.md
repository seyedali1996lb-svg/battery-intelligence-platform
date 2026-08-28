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

**Why it's flagged:** none of the four summary-level datasets this platform's dQ/dV module operates on (NASA, Severson, Oxford, CALCE) expose raw voltage/current time series — only per-cycle summary values (capacity, resistance). (The Zhu 2022 loader's raw files *do* contain full voltage/current time series, but its loader deliberately reduces them to the same summary schema, so the simulation remains the applicable path there too.) A true dQ/dV curve requires the full discharge voltage curve, which the summary schema doesn't carry. Instead, `batlab/features/dqdv.py::simulate_vq_curve()` **simulates** a plausible LiCoO₂ discharge voltage curve from a parametric open-circuit-voltage (OCV) polynomial model:

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
---

## 14. Partial-cycle & field telemetry processing (Rainflow & OCV relaxation)

**Rainflow Cycle Counting** (`batlab.features.partial_cycles::rainflow_counting`):
Implements the ASTM E1049-85 four-point standard to decompose irregular field time-series (EV driving / BESS dispatch) into discrete full and half hysteresis cycles:
```math
\text{EFC} = \sum_{k} \frac{\Delta \text{DoD}_k \times c_k}{100\%}
```
where $c_k \in \{0.5, 1.0\}$ is the cycle count weight and $\Delta \text{DoD}_k = |s_{k+1} - s_k|$ is the State of Charge range.

**OCV Relaxation Fitting** (`batlab.features.partial_cycles::reconstruct_ocv_relaxation`):
Extracts asymptotic equilibrium open-circuit voltage $V_{\text{ocv}}$ and ohmic DC internal resistance $R_0$ from resting intervals ($|I| < I_{\text{thresh}}$):
```math
V(t) = V_{\infty} - (V_{\infty} - V_0) e^{-t / \tau}, \qquad R_0 = \frac{|\Delta V_{\text{step}}|}{|\Delta I_{\text{step}}|}
```

---

## 15. Hybrid Physics-Informed ML (PINN) & degradation decomposition

**PINN Regularized Loss** (`src/pinn_model.py::BatteryPINNEstimator`):
Couples data-driven regression with electrochemical ODEs for diffusion-limited SEI growth (LLI) and particle cracking fatigue (LAM):
```math
\text{SOH}(n) = 1 - \beta_{\text{sei}} \sqrt{n} \, e^{-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)} C_{\text{rate}}^{0.5} - \beta_{\text{lam}} n^{\gamma}
```
The loss minimizes:
```math
\mathcal{L} = \frac{1}{N} \sum (\text{SOH}_{\text{pred}} - \text{SOH}_{\text{meas}})^2 + \lambda_{\text{phys}} \sum \max\left(0, \frac{d\text{SOH}}{dn}\right)^2 + \lambda_{\text{reg}} \|\Theta\|^2
```
strictly enforcing physical monotonicity and non-negative degradation rates.

---

## 16. Dynamic LCA carbon accounting & Verifiable Battery Passports

**Dynamic Carbon Footprint** (`src/dynamic_circularity.py::calculate_dynamic_lca`):
Calculates cradle-to-grave emissions using regional grid carbon intensity $I_{\text{grid}}$ (g CO₂e/kWh) during charging:
```math
\text{CO}_{2,\text{net}} = E_{\text{nom}} \times I_{\text{mfg}} + \left(\frac{E_{\text{throughput}}}{\eta} - E_{\text{throughput}}\right) \times I_{\text{grid}} + E_{\text{nom}} \times I_{\text{rec}}
```
**W3C Verifiable Credential** (`generate_verifiable_credential_passport`):
Encodes verified SOH, RUL quantiles, and carbon intensity into a W3C-compliant JSON-LD document with SHA-256 / Ed25519 cryptographic signatures for EU Regulation 2023/1542 DPP compliance.

---

## 17. High-frequency streaming anomaly detection (CUSUM & Mahalanobis)

**Statistical CUSUM** (`src/streaming_analytics.py::StreamingAnomalyEngine`):
Tracks cumulative positive and negative drifts in terminal voltage with slack $k$ and threshold $h$:
```math
S_t^+ = \max(0, S_{t-1}^+ + (V_t - \mu) - k), \qquad S_t^- = \max(0, S_{t-1}^- - (V_t - \mu) - k)
```
**Multivariate Mahalanobis Distance Proxy**:
```math
D_M = \sqrt{\frac{1}{3}\left(z_V^2 + z_I^2 + z_T^2\right)}
```
flagging correlated multi-signal precursors ($D_M > 3.0$) and IEC 62619:2022 thermal runaway triggers ($dT/dt > 2.0\,{}^\circ\text{C/min}$).

---

## 18. Quantile-interval calibration — LCO coverage and conformal recalibration

**What it is:** the Q10/Q90 quantile GBRT models (see §6) claim an 80% prediction interval, but `train_models()` only ever measured that interval's empirical coverage on a chronological holdout of cells the model had already partly seen. `batlab.validation.calibration` asks the honest LCO version of the question and adds a leakage-free correction:

**`run_lco_quantiles()`** trains the RUL point + Q10/Q90 models under the same leave-cell-out folds as `run_lco()` and reports empirical coverage on cells never seen in training:

```math
\text{coverage} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[y_i \in [\hat{q}_{10,i},\, \hat{q}_{90,i}]]
```

**`recalibrate_lco_intervals()`** applies **conformal quantile recalibration** (Romano, Patterson & Candès, NeurIPS 2019): for a held-out fold, compute each calibration point's conformity score $E_i = \max(\hat{q}_{10,i} - y_i,\ y_i - \hat{q}_{90,i})$ — how far the true value falls outside its own raw interval — using only the *other* folds' points; then widen the held-out fold's interval to $[\hat{q}_{10} - E^*,\ \hat{q}_{90} + E^*]$, where $E^*$ is the $(1-\alpha)(n+1)/n$ empirical quantile of those scores ($\alpha = 0.2$ for the default 80% interval). The recalibrator never sees the cell it is applied to, and the method carries a distribution-free marginal coverage guarantee at the nominal level.

**Why not isotonic-on-(predicted-quantile, true) pairs:** fitting isotonic regression of the true value on the predicted quantile approximates a conditional *mean*, not a quantile — it collapses an over-narrow interval instead of correcting it (fitting on $(F_i(y_i), y_i)$ pairs is mathematically a fixed point that recovers the model's own inverse CDF). Conformal scores are the standard, correct alternative.

**Type:** empirical coverage is a deterministic statistic on measured/predicted values; recalibration is a distribution-free statistical correction with a marginal guarantee — still not a per-cell conditional guarantee, and with LCO's small fold populations the corrected coverage is an estimate, not a certificate.

## 19. Per-prediction local attribution (occlusion-based, SHAP-style)

**What it is:** `feature_importances_` (§6) answers "what drives this model in general" but not "why did THIS cell's RUL come out at 512 cycles?". `batlab.models.attribution::occlusion_attribution()` answers the local question: for one prediction $x$ and feature $j$, draw $n$ values of feature $j$ from a reference distribution, substitute each into $x$ one at a time, and record the mean absolute prediction change:

```math
\phi_j(x) = \frac{1}{n}\sum_{k=1}^{n} \left| f(x) - f(x^{(k)})\right|,\qquad x^{(k)}_j \sim P_{X_j}^{\text{ref}}
```

A feature the model leans on for this specific row shows a large mean change; an irrelevant feature leaves the prediction untouched. `mean_attribution()` averages this per-row signal into a local-first feature ranking, and `top_attributions()` returns the per-row top-N for a "why this prediction?" UI.

**Type:** a Monte Carlo sensitivity/perturbation estimate, seeded and deterministic for a fixed seed. It is SHAP-*style* (counterfactual perturbation attribution), not the exact game-theoretic TreeSHAP values, which require the `shap` package's tree-path algorithm — the module deliberately has no `shap` dependency, and the values should be read as a ranking signal rather than a precise Shapley decomposition.

## 20. Market-data adapters (pluggable price/carbon feeds)

**What it is:** the Lifecycle Intelligence layer's external-data boundary — the market-side sibling of the `BMSAdapter` protocol in `src/bms_connectors.py`. `src/market_data.py` defines a `MarketDataAdapter` Protocol (`name`, `is_configured()`, `fetch_hourly_prices()`, `fetch_carbon_intensity()`) and a registry with three built-ins:

- **Synthetic** — deterministic, offline, always available (the tested default). A two-peak daily price shape (night trough 23–06, morning peak 07–10, midday shoulder 11–16, evening peak 17–21, late-evening drop 22) with seed-controlled day-to-day jitter and an optional injected price spike for arbitrage tests.
- **EIA Open Data** (`api.eia.gov/v2/electricity/rto/region-data`) — hourly balancing-authority prices, built against the documented JSON shape, returned in USD/kWh (converted from the API's $/MWh). Requires a free EIA API key.
- **ENTSO-E Transparency** — day-ahead prices for a bidding zone via the documented XML `documentType=A44` endpoint, parsed with the stdlib only, returned in EUR/kWh (converted from €/MWh). Requires a free ENTSO-E REST API key.

**Contracts** (same as every external adapter in this project, see `src/adapter_contract.py`): None when not configured (missing key) — never raises; `{"error": str}` on request failure — never raises; result dict on success. The EIA/ENTSO-E adapters are built against documented API shapes but are **not verified against live accounts** (no API key exists for this project) — the same honest scope as the BMS connectors.

**Currency normalization:** `to_eur_per_kwh()` converts USD feeds to EUR at `USD_TO_EUR = 0.92`, an explicitly labeled illustrative FX assumption (see the ASSUMPTIONS convention in §12), recorded in the result's `fx_assumption` field so the conversion is auditable.

**Carbon intensity:** `resolve_carbon_intensity()` prefers a live per-hour series from a configured adapter (e.g. the synthetic feed's deterministic daily carbon shape) and falls back to the static IEA/EEA regional table (`src/dynamic_circularity.GRID_CARBON_INTENSITY` — the single source of truth, imported, not duplicated). `calculate_dynamic_lca()` accepts the resulting live value additively via a new optional `grid_intensity_g_kwh` parameter (§16), so dynamic-LCA can run on live intensity without changing any existing caller.

**Type:** external-data fetch adapters (synthetic: deterministic model; EIA/ENTSO-E: documented-shape REST/XML clients, untested against live accounts).

## 21. Health-aware dispatch (SoP-limited, RUL/SOH-narrowed arbitrage)

**What it is:** the platform's answer to the battery-storage-software cohort (Capture Energy, Solship, Deepgrid — see the competitive comparison): those optimizers bid dispatchable capacity while *assuming the battery is healthy*. `src/health_aware_dispatch.py` prices the same arbitrage opportunity but constrains the schedule with the platform's own health signals.

**Health constraints** (`health_constrained_band()`):

```math
P_{\text{cap}} = E_{\text{nom}} \times C_{\text{rate}} \times \underbrace{\max\!\left(0.3,\ \min\!\left(1,\ \tfrac{\text{sop\_pct}}{100}\right)\right)}_{\text{SoP power factor}}
```

- **SoP-limited power cap** — a cell at 50% State-of-Power (the resistance-derived rate-capability proxy from §3) delivers at most half its nominal C-rate power, floored at 30%.
- **RUL/SOH-narrowed operating band** — a cell with a *reliable* RUL below 200 cycles, or SOH below 80% without a reliable RUL, is dispatched within a narrowed `[40%, 85%]` SOC band instead of the healthy `[10%, 95%]` band, reducing per-cycle depth-of-discharge (the same stress-reduction rationale the second-life literature and §12's application fit use).

**Dispatch heuristic** (`arbitrage_schedule()`): a threshold heuristic, not an LP/MILP optimizer — the same explicit scope decision `src/deployment_sizing.py` already made for the solar+storage engine. The battery charges when price is below the window's 35th percentile and discharges above the 65th percentile, with the health-constrained power cap and SOC band applied each hour. EFC delivered is computed by running the resulting SOC trajectory through the platform's own rainflow engine (§14), turning-point-preprocessed (the engine's extrema detector misses plateau-heavy square-wave profiles).

**Comparison** (`schedule_comparison()`): dispatches the same price window twice — once under the cohort's implicit "assume healthy" behavior (soh=100%, no SoP limit, no RUL caution), once with the real health signals — and reports the signed delta (revenue, EFC, mean cycle DoD). With the threshold heuristic the sign can flip on degenerate price shapes, so the delta is presented as a signed comparison, not a guaranteed loss.

**Type:** deterministic rule-based dispatch (derived from measured/health-model inputs), with EFC accounting via the rainflow engine. Explicitly *not* a validated optimal-control result.

## 22. Grid-services revenue stack & managed charging

**Grid-services revenue** (`src/grid_services.py`) — per-site revenue potential across the three ways a battery makes money:

```math
V_{\text{total}} = V_{\text{arbitrage}} + P_{\text{MW}} \cdot r_{\text{reg}} \cdot h_{\text{service}} + P_{\text{MW}} \cdot r_{\text{cap}}
```

- **Arbitrage** — from `arbitrage_schedule()` (§21), annualized by repeating a representative price window to 8760 hours (flagged `arbitrage_annualized_from_window=True`) or used directly when the window is the full year.
- **Frequency regulation (ancillary)** — capacity held for aFRR/FCR-style service earns `r_reg` (default 25 €/MW/h) for the hours in service; its energy cycling contributes EFC stress, accounted on the cost/stress ledger, not as revenue.
- **Capacity (reserve)** — availability payments at `r_cap` (default 40,000 €/MW/yr).

Every rate is an entry in `GRID_SERVICES_ASSUMPTIONS` in the same `{value, slider_range, unit, label, source}` shape as §12's `ASSUMPTIONS` — defaults are labeled "Illustrative — not sourced" and the arbitrage-vs-ancillary exclusivity is stated in the result (`exclusivity_note`), not hidden.

**Managed charging** (`src/managed_charging.py`) — cheapest-hour EV charging over a price window: sort hours by price, greedily fill the energy needed to reach `target_soc_pct` at up to `max_charge_kw`, efficiency applied so wall energy ≥ battery energy. The unmanaged baseline (charge immediately at max power) gives the savings delta; the session's flexibility is measured as the rainflow EFC of its SOC trajectory. It is a **recommendation, not a control signal** — the OCPP connector reads completed sessions from a Central System's REST reporting API but does not push commands (see `src/bms_connectors.py`'s honest scope), and this module makes the same no-push scope explicit.

**Type:** illustrative financial/engineering estimates on supplied market prices — every figure labeled, none presented as a validated market outcome.

## 23. Fleet aggregation (dispatchable-capacity offers)

**What it is:** per-cell health/SoC headroom aggregated into one VPP-style offer (`src/fleet_aggregation.py`):

```math
E_{\text{offer}} = \sum_{i} E_{\text{cur},i} \times \frac{\text{band}_{\text{high},i} - \max(\text{SOC}_i,\, \text{band}_{\text{low},i})}{100}, \qquad
P_{\text{offer}} = \sum_{i} E_{\text{cur},i} \times C_{\text{rate}} \times f_{\text{SoP},i}
```

where $E_{\text{cur},i} = E_{\text{nom},i} \times \text{SOH}_i/100$ (SoH-limited current capacity) and the per-cell band/power factor come from the same `health_constrained_band()` as §21 — so a faded or power-limited cell is offered *less*, and a caution-flagged cell offers shallower depth-of-discharge, rather than being bid at nameplate. A cell whose current SOC is already at/above its band high limit is excluded from the window (listed separately with a reason).

**Type:** deterministic aggregation of measured/derived health signals — a capability *offer*, explicitly not a dispatch control signal (stated in the returned `caveats`).

## 24. Model cards, health-as-a-service & live-carbon LCA

**Auto-generated model cards** (`src/model_cards.py`): every logged training run (see §6's reproducibility discussion and `src/experiment_registry.py`) renders as a structured card via `build_model_card(run)` — model identity (framework, feature version, seed, git commit, timestamp), data (chemistry, n_cells/n_rows, feature set, and the dataset's real license resolved through `batlab.cite`'s own license text — the single source of truth; the synthetic fleet is disclosed as internally generated, a tenant upload as license-unknown), validation (LCO method statement, SOH/RUL metrics, per-fold breakdown), hyperparameters, the reproducibility contract (replay via `evaluate_from_manifest`; recorded-vs-current `GBRT_PARAMS` divergence surfaced in `hyperparams_diff_vs_current`, never hidden), and platform-standard limitations (public-lab data, small fold populations, RUL = cycles-to-80%-SOH, dQ/dV-as-simulation — plus NOT-leave-cell-out and synthetic-fleet caveats where applicable). `model_card_markdown()` renders it for the Benchmark page, which also offers a JSON download per run.

**Health-as-a-service** (`GET /cells/{id}/health`): a single machine-readable record composing existing, already-validated plumbing — LCO-validated SOH/RUL (quantiles only when the per-cell RUL reliability floor is met, §7), State-of-Power (the §3 rate-capability proxy), fade rate, an explicit per-metric `confidence` map, EU-passport-facing fragments (chemistry, IEC 62902 R-code via §12's `eol_r_code_recommendation()`, best second-life application via `application_fit()`), and the run's model card. No new computation — the endpoint composes, it does not re-derive.

**Live-carbon LCA**: the market-data adapter's `resolve_carbon_intensity()` (§20) feeds `calculate_dynamic_lca()`'s additive `grid_intensity_g_kwh` override (§16); the REST endpoint opts in via `use_live_carbon=true` and reports `carbon_resolution` (source live/static, window mean, per-hour series) alongside the footprint. Static IEA/EEA table remains the default — live is opt-in, and the response says which source was actually used.

**Type:** structured provenance/reporting layers over already-validated computations — no new model, no new estimate.

## 25. ML-based unsupervised anomaly detection

**What it is:** the complement to the named-rule anomaly engines (§10, §17). Those fire on thresholds a human anticipated; a per-cell Isolation Forest (`src/ml_anomaly.py`) learns the normal region of a cell's *own* feature space from its historical cycles and flags cycles that are novel relative to that history — the signal that catches patterns no rule was written for (e.g. a slow joint capacity+resistance drift no single-channel threshold crosses, or an operating temperature unusual for THIS cell).

**Features** (`MLAnomalyDetector._feature_matrix`): capacity, a 30-cycle rolling fade rate, resistance + resistance-growth, and temperature — derived deterministically from the standard cycle schema; deliberately NOT the full §3 feature pipeline (this is a health-signal detector, not a second training pipeline).

**Warmup handling:** the fade-rate feature needs 30 cycles of history, so a cell's first 30 cycles carry no full rolling window. They are excluded from the fit (fitting on them would teach the forest a spurious early-cycle cluster) and reported in `per_cycle` as unscored warmup with `anomaly_score=null` and an honest note — never scored on a fabricated feature, never silently dropped.

**Scores and threshold:** sklearn's IsolationForest convention is inverted so higher = more anomalous; a cycle is flagged when its score is at or above the (1 − contamination) empirical quantile (default contamination 0.05 — an assumed fraction of anomalous cycles, a modeling assumption, not a physical limit). Scores are relative per cell, so the report returns each cell's own score distribution rather than a global cutoff.

**Honest framing** (in the returned `caveats`, not just this document): unsupervised novelty detection, not fault classification — a flagged cycle is "unusual for this cell's own history", not a diagnosed fault; flags are review signals to cross-check against the named-rule engines before acting. Cells with too few scored cycles are refused with an explicit reason (or listed in `detect_fleet_anomalies()`'s `skipped` map), never given a noisy fit.

**Type:** unsupervised ML novelty detection (IsolationForest) on derived cycle features — a review signal, not a diagnosis.

## 26. Digital twin architecture (Phase 3)

**What it is:** the defined architecture connecting a cell's measured history, its derived health indicators, and a physics-based degradation model into one continuously-updated representation — `src/digital_twin.py`'s `CellTwin`. The twin holds three layers in a single object: (a) measured per-cycle history as it arrives (merged idempotently per cycle number), (b) the platform's standard health indicators (SOH, 30-cycle fade rate, knee detection via §4, EOL flag), and (c) a physics projection — the same SEI sqrt-fade model as §12/§19 (`Q(n) = Q0·(1 − β·√n)`, `β` re-fit by `scipy.optimize.curve_fit` on every update batch, 2σ bands, optional one-time PyBaMM SPM anchor for nominal capacity). Every update re-derives all three layers from the merged history, so a consumer holding a twin always reads one self-consistent state instead of reassembling independently-computed verdicts (the disagreement bug class this platform has fixed repeatedly).

**Honest boundaries** (in the snapshot's `labels`, not just this document): the projection is a *projection*, not a prediction — physics-based forward extrapolation of measured fade; the parameter set is fixed per chemistry, not re-parameterized from telemetry; and without a real BMS feed this is "not a live-synced digital twin" (the same real-BMS-validation trigger that gates the Lifecycle Intelligence layer gates a deeper twin). The API path (`GET /cells/{id}/twin`) deliberately skips the slow SPM anchor and reports `spm_capacity_ah: null`.

**Robustness:** the twin ingests only numeric per-cycle measurements — string/None annotation columns (e.g. `confidence_tag`) that real app frames carry are skipped rather than crashing a float cast, and cycles without a `capacity_ah` are excluded from the fit. Fit failures degrade to a `last_error` field, never an exception to the caller.

**Type:** continuously-updated physics-informed state representation (SEI fade + derived indicators), honestly labeled projection.

## Where this is enforced, not just described

Every formula above that isn't a one-line UI threshold has a corresponding unit test in `tests/` (e.g. `tests/test_features.py`, `tests/test_dqdv.py`, `tests/test_knee_detection.py`, `tests/test_lco_eval.py`, `tests/test_trajectory_memory.py`, `tests/test_innovations_v2.py`, `tests/test_api_v2_endpoints.py`, `tests/test_market_data.py`, `tests/test_health_aware_dispatch.py`, `tests/test_grid_services.py`, `tests/test_managed_charging.py`, `tests/test_fleet_aggregation.py`, `tests/test_api_p1_endpoints.py`, `tests/test_model_cards.py`, `tests/test_ml_anomaly.py`, `tests/test_api_p2_endpoints.py`, `tests/test_benchmark_model_card_page.py`) that checks the actual computed values against known inputs.
