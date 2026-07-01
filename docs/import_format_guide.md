# Battery Data Import Format Guide

> **Disclaimer — read before uploading data.**
> This import format is designed for research and demonstration purposes.
> Results produced by this platform — including SOH estimates, RUL predictions,
> second-life fit scores, and lifecycle carbon figures — **should not be used
> for regulatory submissions or safety-critical decisions without independent
> validation** by qualified engineers using accredited measurement equipment and
> certified analysis methods. The platform shows what the model thinks given the
> data you provide; it does not replace a Battery Management System, a
> third-party capacity audit, or a regulatory-grade test protocol.

---

## Template file

Download: [`data/import_template.csv`](../data/import_template.csv)

Open in any spreadsheet application or text editor. Replace the example rows
with your own data. Do not change the column names or their order.

---

## Column reference

### Required columns

| Column | Type | Example | Description |
|---|---|---|---|
| `cell_id` | string | `MyCell_01` | Unique identifier for the cell. Must be identical across all rows belonging to the same cell. Use a consistent format — the platform uses this to group rows into per-cell histories. |
| `cycle_number` | integer | `1`, `2`, `3` | Discharge cycle index, starting at 1 and incrementing by 1 per complete discharge cycle. Must be sequential within each `cell_id`. |
| `capacity_ah` | float | `2.031` | Measured discharge capacity in amp-hours for that cycle. This is the platform's primary health signal — SOH is computed as `capacity_ah / initial_capacity_ah × 100`. |
| `resistance_ohm` | float | `0.052` | Internal resistance in ohms at that cycle. DC internal resistance (DCIR) is preferred. EIS-derived electrolyte resistance (Re) is also accepted — see the resistance measurement note below. |

### Optional columns

| Column | Type | Example | Description |
|---|---|---|---|
| `temperature_c` | float | `24.3` | Average cell temperature during the cycle in degrees Celsius. If blank, the platform substitutes 25°C and labels every temperature-dependent output **"Assumed 25°C — not measured"** so the assumption is visible, not hidden. |
| `test_date` | string (ISO 8601) | `2023-01-05` | Date the cycle was recorded, in `YYYY-MM-DD` format. Not used in modelling. Improves lifecycle timeline display on the Passport page and in PDF reports. |
| `notes` | free text | `Formation cycle` | Per-cycle annotation. Not used in modelling. Displayed on the Passport page only. |

---

## Minimum data requirements

### Why minimums exist at all

The platform's predictions are produced by a Gradient Boosting model validated
with **leave-cell-out (LCO) cross-validation**: train on N−1 cells, test on the
one held-out cell entirely. LCO is the only honest way to ask "does this model
generalise to a cell it has never seen?" — a row-level train/test split on a
single cell's cycle history just tests whether the model can interpolate within
that cell's own trajectory, not whether it works on new cells.

This structure creates hard floor requirements on cell count.

### Cell count

| Cells uploaded | Behaviour |
|---|---|
| **1 cell** | Platform refuses to show model outputs. With only one cell, there is no held-out cell to validate against — LCO requires at least 2 folds (train on 1, test on 1). Showing predictions without any generalisation evidence would be misleading. |
| **2 cells** | Platform runs LCO with 2 folds, but reliability is low. With only one cell in each training fold, the model has almost no signal to learn from. Results will appear but per-cell reliability scores will likely show **"Calibrating"** on both cells. Treat outputs as directional only. |
| **3+ cells (recommended minimum)** | LCO produces meaningful per-cell reliability scores. Each held-out fold trains on at least 2 cells, which is enough signal for the model to learn cell-to-cell variation. Below this count the RUL reliability gate has little power to distinguish calibrated from uncalibrated cells. |
| **8+ cells (ideal)** | Reliable per-cell fold R² scores. Sufficient variation for the model to learn stress-driven degradation differences. This is the threshold at which the platform's unified fleet RUL model becomes trustworthy — see the Fleet page roadmap. |

### What "Calibrating" means

The RUL reliability gate computes a per-cell fold R² from LCO. If a cell's
held-out fold R² falls below **0.30**, the platform suppresses the RUL
prediction for that cell and shows **"Not calibrated"** everywhere a cycle-count
appears. "Calibrating" appears on Overview for early cycles where prediction
features (rolling fade rate, resistance trend) have not yet stabilised — usually
before cycle 50. This is not a failure; it is the platform being honest that
early-cycle predictions carry high uncertainty.

### Cycle count

| Cycles per cell | Behaviour |
|---|---|
| **< 50 cycles** | RUL features have not stabilised. Expect "Calibrating" on Overview. SOH tracking still works (it only needs capacity readings). |
| **50–100 cycles** | Platform runs. Reliability flags may show "Calibrating" on some cells. |
| **100+ cycles (recommended)** | Sufficient history for fade rate signals and resistance trends to be meaningful. Per-cell RUL reliability scores are more likely to reflect true generalisation ability. |
| **500+ cycles** | Enough history to observe degradation inflection points. RUL predictions become most useful when the cell has traversed at least one distinct fade regime. |

---

## Mapping common BMS export formats

### Timestamp-indexed raw data → per-cycle capacity

Most BMS systems export continuous voltage/current/time traces rather than
per-cycle summaries. You need one row per **complete discharge cycle** in the
upload format.

**Step 1 — identify cycle boundaries.** A cycle boundary is typically a
transition from discharge to charge (current sign flip) or a rest period longer
than a defined gap. Most BMS export tools have a "cycle segmentation" or
"discharge summary" export option — use that if available.

**Step 2 — compute discharge capacity for each cycle.**
Capacity is the integral of current over the discharge period:

```
capacity_ah = Σ (current_A × Δtime_hours)
```

In plain language: for each second (or sample interval) of discharge, multiply
the current in amps by the fraction of an hour that interval represents, then
sum across the whole discharge. This gives amp-hours out per cycle. Most BMS
analysis software (Neware, Arbin, Maccor, BatteryArchive) computes this
automatically in their cycle summary exports.

**Step 3 — assign cycle numbers.** Number cycles sequentially from 1 within
each cell. Do not use global timestamps as cycle identifiers.

### Resistance measurement types — important note

The platform uses resistance as a degradation feature alongside capacity. There
are two common measurement methods, and they are **not interchangeable**:

| Method | Typical range (18650 LiCoO₂) | Notes |
|---|---|---|
| **DC internal resistance (DCIR)** | 0.05–0.15 Ω | Measured from voltage step under a current pulse. Includes all ohmic + polarisation contributions. Preferred for this platform. |
| **EIS electrolyte resistance (Re)** | 0.04–0.07 Ω | Derived from electrochemical impedance spectroscopy. Represents only the electrolyte/contact resistance component — a subset of total resistance. |

**Do not mix measurement types across cells in the same upload.** The platform
learned this the hard way with its existing datasets: the 8 synthetic cells use
bulk internal resistance (0.15–0.40 Ω) while the 4 NASA PCoE cells use
EIS-derived Re (0.04–0.07 Ω). A combined model trained on both produced
R²=−0.49 because the same feature column carried physically incompatible values.
The platform runs two separate models as a result.

If your cells were measured with different resistance methods — for example,
some via DCIR pulse and others via EIS — either exclude the resistance column
entirely (leave it blank for all cells; the platform will fall back to
capacity-only features) or ensure all cells in the same upload use the same
method.

---

## What happens after upload

1. **Format validation** — the platform checks column names, data types, and
   that `cycle_number` is sequential within each `cell_id`. Any row failing
   validation is flagged with a specific error before the model runs.

2. **LCO training** — a Gradient Boosting model is trained on your data using
   leave-cell-out cross-validation. Each cell is held out in turn; the model is
   trained on the remaining cells and evaluated on the held-out cell.

3. **Per-cell reliability scoring** — each cell receives a fold R² score from
   its held-out evaluation. Cells below the 0.30 reliability floor have RUL
   predictions suppressed across all pages.

4. **Dashboard population** — results appear across all pages: Overview (SOH +
   RUL per cell), Health (degradation trajectory), Fleet (ranked by SOH + fade),
   Copilot (narrated summary), Consequences (second-life economics),
   Recommendations (action + confidence), and Passport (regulatory field
   structure).

**Estimated processing time: 60–90 seconds** for datasets up to ~20 cells ×
1,000 cycles. Larger datasets scale approximately linearly with total row count.

---

## Quick checklist before uploading

- [ ] At least 3 cells in the file (different `cell_id` values)
- [ ] At least 100 cycles per cell recommended
- [ ] `cycle_number` starts at 1 and increments by 1 within each cell
- [ ] `capacity_ah` is in amp-hours (not mAh, not Wh)
- [ ] `resistance_ohm` uses the same measurement method across all cells
- [ ] Optional columns left as empty fields (not omitted entirely) if not measured
- [ ] `test_date` in `YYYY-MM-DD` format if included
- [ ] No merged cells, no units in the header row, no multi-row headers
