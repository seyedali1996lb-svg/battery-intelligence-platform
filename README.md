# Battery Intelligence Platform

**An open-source battery analytics platform for degradation analysis, state-of-health estimation, remaining useful life prediction, and lifecycle intelligence using public battery datasets.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/seyedali1996lb-svg/battery-intelligence-platform/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://seyedali1996lb-svg.github.io/battery-intelligence-platform/)

---

## Why this project exists

A cycling battery — in a lab, a vehicle, or a stationary storage installation — produces a large amount of voltage, current, capacity, and temperature data over its life. The engineering problem is not collecting that data; it's converting it into decisions: is this cell degrading faster than expected, how many cycles does it have left, is it safe to keep in first-life service, and what is a second-life or replacement plan worth in real currency.

This is harder than it looks, for reasons this platform is built specifically to confront:

- **Battery aging is nonlinear and cell-specific** — degradation doesn't follow one universal curve across chemistries or usage profiles.
- **Capacity degradation is the physical symptom, but it isn't self-explanatory** — a fade curve alone doesn't tell you whether it's on-track or accelerating without a baseline to compare it against.
- **SOH estimation needs to be trustworthy at the point it's produced** — it's the number every downstream decision (dispatch, second-life, replacement) is built on.
- **RUL prediction is where the stakes are highest and the failure modes are quietest** — a model can report an excellent R² and still fail on the next cell it sees, if validated the wrong way.
- **Data leakage in ML validation is the single most common way that happens** — see [Validation](#validation) below for a real, reproduced example.
- **There is no standard schema across public battery datasets** — NASA PCoE, the Severson et al. *Nature Energy* dataset, Oxford's path-dependent dataset, and CALCE's CS2 cells each ship in a different file format with different column conventions, which means every research group re-writes the same brittle parsing code.

`batlab`, the research framework underneath this platform, exists to fix the last two problems directly: one standardized schema across all five datasets, and one validation methodology (leave-cell-out) applied by default to every model it trains.

## Core philosophy

**Reliable battery intelligence requires honest validation.**

A model's reported accuracy is only meaningful if the test setup answers the question you actually care about. For SOH/RUL prediction, that question is "how will this model perform on a battery it has never seen?" — not "how well does it interpolate between cycles of a battery it was partly trained on."

- **Random row-level train/test splits are misleading** for cycling data, because consecutive cycles from the same cell are nearly identical to each other. A model can memorize a cell's own trajectory and still score well on a held-out *row* from that same cell, without having learned anything that transfers to a new cell.
- **Leave-cell-out (LCO) validation** holds out entire cells, never seen in training, and reports accuracy only on those. This is the harder, more honest question, and it's the platform's default — not an opt-in mode.
- **Reproducibility matters** because a claimed R² is worthless if it can't be checked. Every number this platform reports traces back to a runnable notebook or a benchmark manifest, not a number typed into a table. `batlab.validation.manifest.export_benchmark_results()` also exports a machine-readable benchmark bundle (split manifest + reported metrics, `schema: "batlab-lco-benchmark"`) so other software can consume a number together with the conditions it was produced under.

## Platform capabilities

The platform is organized into five engineering modules:

**Battery Data Standardization & Universal Ingestion**
- Unified cycle/summary schema across all five supported datasets (`batlab.datasets`)
- Fifth dataset: Zhu et al. 2022 voltage-relaxation NCM+NCA 18650s (`batlab.datasets.zhu2022`) — 9 dense-cycling cells (~900-1000 cycles each, 25°C, CC BY 4.0), auto-downloaded from Zenodo with SHA-256 verification; per-cycle capacity is derived from raw discharge-run charge transfer (a characterization-cycle guard, not a naive per-cycle max)
- Universal Battery Cycler Auto-Ingestion Wizard (`batlab.datasets.cycler_mapper`): instant heuristic schema detection and unit normalizer for **Arbin, BioLogic, Maccor, Neware, Novonix, Bitrode**, and custom CSVs
- Auto-downloading, checksum-verified dataset loaders with strict `SchemaError` integrity validation

**Battery Health Analytics & Field Telemetry Processing**
- State-of-health (SOH) estimation from measured capacity and resistance growth
- **Partial-Cycle & Field Telemetry Engine** (`batlab.features.partial_cycles`): ASTM E1049-85 compliant Rainflow Cycle Counting for irregular EV/BESS driving profiles, Equivalent Full Cycle (EFC) accumulation, and Open Circuit Voltage (OCV) relaxation curve reconstruction from resting intervals
- Vectorized high-performance feature extraction (`batlab.features.vectorized`) using columnar PyArrow structures for 10x-50x faster feature generation

**Machine Learning & Hybrid Physics Modeling**
- Literature-cited feature engineering (`batlab.features`) with Leave-Cell-Out (LCO) cross-validation by default
- GBRT point estimates with Quantile Regression confidence bounds (Q10/Q90)
- **Quantile-interval calibration** (`batlab.validation.calibration`): leave-cell-out empirical coverage of the Q10/Q90 RUL interval, plus conformal quantile recalibration (Romano et al. 2019) fit per fold on the other folds only — the recalibrator never sees the cell it is applied to
- **Per-prediction local attribution** (`batlab.models.attribution`): occlusion-based, SHAP-style feature attribution answering "why did THIS cell's RUL come out at X" — per-row/per-feature mean prediction change under counterfactual substitution, no `shap` dependency
- **Hybrid Physics-Informed Neural / Numerical Estimator (PINN)** (`src/pinn_model.py`): coupled electrochemical-thermal degradation tracking (SEI diffusion-limited LLI + mechanical particle cracking LAM) with monotonicity regularized physics loss
- Physics-informed calibration (`src/physics_calibration.py`, NASA + Severson): per-cell LLI/LAM decomposition anchored by PyBaMM SPM discharge
- Asynchronous task worker queue (`src/task_queue.py`) with Server-Sent Events (SSE) progress streaming for non-blocking LCO evaluation

**Battery Diagnostics & Real-Time Streaming**
- High-performance differential-capacity ($dQ/dV$) analysis and knee-point detection
- **Industrial Real-Time Streaming Backbone** (`src/streaming_analytics.py`): high-frequency sub-10ms anomaly detection using Cumulative Sum (CUSUM) statistical change-point detection, multivariate Mahalanobis Distance Z-scoring, and IEC 62619:2022 Thermal Runaway Precursor alarms
- **ML-based unsupervised anomaly scan** (`src/ml_anomaly.py`): a per-cell Isolation Forest learns the normal region of a cell's own feature space (capacity, fade rate, resistance growth, temperature) and flags cycles that are novel relative to that history — the complementary signal the named-rule engines can't provide (patterns no rule anticipated). Warmup cycles (no full 30-cycle rolling history) are honestly reported unscored rather than fitted on a fabricated feature; the contamination-based threshold, score-relative-per-cell caveat, and "novelty, not fault diagnosis" framing are all in the returned caveats. Exposed as `POST /analytics/ml-anomaly` and `detect_fleet_anomalies()` for fleet-wide scans
- A Battery Digital Knowledge Graph (`src/knowledge_graph.py`, NetworkX) linking cells to degradation mechanisms and literature DOIs

**Lifecycle Intelligence & Digital Passports**
- **Unified Operations Action Center** (`src/action_center.py`): centralized SLA-based triage inbox with one-click CMMS work order dispatch, manufacturer warranty claims, and circularity routing
- **Pluggable market-data adapter** (`src/market_data.py`): a `MarketDataAdapter` Protocol (the market-side sibling of `BMSAdapter`) with a deterministic offline **Synthetic** feed (the tested default), **EIA Open Data** hourly prices, and **ENTSO-E Transparency** day-ahead prices (both built against documented API shapes, untested against live accounts, registered with an honest "not configured" state until a key is supplied) — plus `to_eur_per_kwh()` currency normalization and `resolve_carbon_intensity()` which prefers a live feed and falls back to the static IEA/EEA table
- **Health-aware arbitrage dispatch** (`src/health_aware_dispatch.py`): price-arbitrage schedules constrained by the battery's *own* health signals — SoP-limited power caps (a cell at 50% State-of-Power delivers half its nominal C-rate power) and RUL/SOH-narrowed SOC bands (reduced depth-of-discharge for end-of-life-sensitive cells) — with EFC/DoD accounting via the platform's own rainflow engine and a `schedule_comparison()` that dispatches the same price window under the cohort's "assume healthy" behavior vs. real health constraints, making the differentiator measurable (revenue given up vs. stress avoided)
- **Grid-services revenue stack** (`src/grid_services.py`): per-site revenue potential across energy arbitrage, frequency regulation (ancillary), and capacity (reserve) — every rate in the same `{value, slider_range, unit, label, source}` ASSUMPTIONS shape as `src/consequences.py`, with an explicit arbitrage-vs-ancillary exclusivity note
- **Tariff-aware managed charging** (`src/managed_charging.py`): cheapest-hour EV charging plans over a price window with the cost delta vs unmanaged charging and the session's flexibility measured in rainflow EFC — an Optiwatt-adjacent use case built on this platform's own market adapter and partial-cycle engine (a recommendation, honestly not a control signal: the OCPP connector reads sessions but does not push commands)
- **Fleet dispatchable-capacity offers** (`src/fleet_aggregation.py`): per-cell SoH-limited energy + SoP-limited power + RUL/SOH-narrowed bands aggregated into a VPP-style `{energy_kwh, power_kw}` offer for a service window — the interface Anode/Deepgrid-style operators would consume, honestly labeled as a capability statement rather than a dispatch control
- **Dynamic LCA Carbon Footprint Accounting** (`src/dynamic_circularity.py`): cradle-to-grave CO2e accounting using dynamic regional grid carbon intensity (IEA/EEA) during charging, with an optional live `grid_intensity_g_kwh` override from a configured market adapter (`calculate_dynamic_lca()` accepts it additively); the REST endpoint's `use_live_carbon=true` opt-in resolves the live feed automatically via `resolve_carbon_intensity()` and reports its source
- **Health-as-a-service endpoint** (`GET /cells/{id}/health`): one JWT-gated response carrying LCO-validated SOH/RUL (Q10/Q90 only when the per-cell reliability floor is met), State-of-Power, fade rate, an explicit per-metric confidence map, EU-passport-facing fragments (chemistry, R-code, best second-life application), and the auto-generated model card of the run behind the model — reusing only existing plumbing, no new training
- **Auto-generated model cards** (`src/model_cards.py`): every logged experiment-registry run renders as a structured, honest model card (model identity, dataset + its real license via `batlab.cite`, LCO validation metrics, hyperparameters, the replay/hyperparams-divergence reproducibility contract, and platform-standard limitations) — shown per run on the Benchmark page with a JSON download, and embedded in the health-as-a-service response
- **Digital Twin architecture** (`src/digital_twin.py`): the Phase 3 CellTwin — one continuously-updated representation of a cell's measured history, derived health indicators (SOH, 30-cycle fade rate, knee, EOL), and a physics-based SEI sqrt-fade projection re-fit on every update batch. Exposed as `GET /cells/{id}/twin` and as a Live Monitor block that consumes the streamed telemetry; honestly labeled "projection, not prediction" and "not a live-synced digital twin" (fixed per-chemistry parameter set, no real BMS feed). On the Cell Workbench's Health view the projection is charted beside the GBRT-fade 12-month forecast — measured history, central SEI sqrt-fade projection, and ±2σ fit band on the same SOH axis, with a GBRT-vs-physics RUL comparison strip — so researchers can see where the data-driven and physics models diverge for a cell
- **Grid Services page** (`app/_pages/operations.py`): the Streamlit surface for the Lifecycle Intelligence layer — health-aware dispatch (with the healthy-vs-health-aware comparison), grid-services revenue, managed charging, fleet dispatchable-capacity offers, and the ML anomaly scan, all read-only estimates with the modules' own labels
- **W3C Verifiable Credential Battery Passport** (`src/dynamic_circularity.py`): EU Battery Regulation (EU 2023/1542) JSON-LD digital product passport generator with cryptographic Ed25519 signatures
- Automated Second-Life Auction & Bid Matcher connecting certified cell health (SOH, SoP%, RUL) with buyer application profiles
- **Modern React 19 / TypeScript SPA Frontend** (`frontend/`): 60fps electrochemical cycle scrubber, real-time live telemetry canvas, and multi-view operations dashboard alongside the Streamlit application


## Architecture

```
Data Layer                    batlab.datasets
    │                         standardized schema, checksum-verified downloads
    ▼
Feature Engineering           batlab.features
    │                         literature-cited stress/aging features, dQ/dV, knee detection
    ▼
ML Models                     batlab.models
    │                         GBRT + quantile regressors, leave-cell-out validated by default
    ▼
Analytics Engine              batlab.validation, src/*
    │                         SOH/RUL, anomaly thresholds, deployment sizing, second-life economics
    ▼
Visualization / Application   app/
                               Streamlit dashboard, battery passport, fleet view, live monitor
```

Each layer is independently usable — `batlab` runs standalone as a Python library (see [Installation](#installation)), and the Streamlit application in `app/` is a consumer of it, not the other way around.

## Methodology

Every result this platform produces falls into exactly one of five categories, and the intent is that it's never ambiguous which one you're looking at:

| Category | Examples here | How it's produced |
|---|---|---|
| **Measured** | Voltage, current, capacity, temperature, cycle count | Read directly from the five public datasets' raw files into the standardized schema — never modified |
| **Derived** | SOH, capacity fade rate, resistance growth, dQ/dV peaks, knee point | Deterministic engineering formulas applied to measured data — the exact formula behind each one is documented in [`METHODOLOGY.md`](METHODOLOGY.md) |
| **Predicted (ML)** | RUL point estimate, RUL quantile interval | GBRT / quantile models trained on engineered features, always leave-cell-out validated |
| **Simulated** | PVGIS-driven solar yield in the Solar + Storage Sizing calculator, PyBaMM-based physics capacity projection | Physics-based or third-party-API-driven simulation of a process, not a direct sensor reading |
| **Illustrative assumption** | Second-life resale value, install-cost presets used in payback/NPV calculations | Values with no citation are labeled `"Illustrative — not sourced"` in the UI, explicitly distinct from `"Cited estimate"` values that do have one |

## Validation

[`notebooks/02_data_leakage.ipynb`](notebooks/02_data_leakage.ipynb) reproduces this on real data, using the same 4 NASA PCoE cells and the same GBRT model:

| Validation method | SOH R² | What it actually measures |
|---|---|---|
| Naive random row-level split | **0.998** | How well the model interpolates within cells it has already partly seen |
| Leave-cell-out (this platform's default) | **0.806** | How well the model generalizes to a cell it has never seen |
| Trivial baseline (cycle number → SOH, no engineered features), same LCO folds | 0.603 | The floor — how much of the 0.806 is the smooth shape of aging curves, not the model |

The 0.998 is real and reproducible — and it's also the wrong number to report, because it doesn't answer the question that matters for a deployed model. The honest 0.806 is what `batlab.validation.run_lco` reports by default. Comparing it against the 0.603 trivial-baseline floor under the identical fold structure shows the engineered features are worth a real **+0.203** — not the full 0.806.

This is public-data validation, not industrial validation — the 0.806 describes generalization across 4 NASA cells, not across a manufacturer's fleet. Treat it as evidence the methodology is sound, not as a number that transfers directly to a different chemistry or duty cycle.

## Demo Application

`app/` is a Streamlit application, built on `batlab` — an engineering prototype demonstrating what the platform's analytics look like assembled into an engineer-facing tool, not a production deployment:

- A per-cell **battery health dashboard** — SOH/RUL, degradation trajectory, dQ/dV, anomaly flags
- A **fleet monitoring** concept view across multiple cells at once, plus a simulated live-telemetry Monitor page
- An **EU Battery Passport** generator, with a "regenerate this report" action that replays the exact recorded pipeline (dataset, feature set, hyperparameters, seed) behind any displayed result
- A **Benchmark** leaderboard across every logged training run — filterable by dataset/chemistry, sortable by any metric, with fold-level drill-down — including an honest cross-chemistry generalization study (train on NASA, zero-shot evaluate on Severson) reporting the real transfer error, not a flattering one
- A **second-life / Solar + Storage Sizing calculator** — a real hour-by-hour (8760 hours/year) dispatch simulation against PVGIS solar data, with temperature-aware battery derating cited to real cell documentation (BU-410), not a monthly approximation
- A **Copilot** with two honestly-distinct modes: topic-button questions always use a fixed template narrating values already computed by the pipeline; typed free-text questions, when a personal Anthropic API key is configured, use real Claude tool-calling (`src/copilot_agent.py`, Claude Sonnet 5) — the model decides which of a handful of read-only data-fetching tools to call, chaining calls across cells for compositional questions ("which of my degrading cells has the worst fade rate, and why") that no fixed keyword router could match, while every tool still returns only values the pipeline already computed, never a number the model invents itself
- **Multi-tenant accounts** (bcrypt-hashed passwords, per-org data isolation, `src/db.py`) with a session logout control, a per-username login lockout (5 failed attempts locks the account for 15 minutes), and real server-side role-based access control — admin-tier actions (org settings, integration credentials, teammate/site/fleet management) are refused in `src/db.py` itself for any caller without the admin role, not just hidden in the UI — see [Production Readiness Roadmap](docs/history.md#production-readiness-roadmap) for what's still demo-grade versus production-ready in the auth layer
- **Performance batch** — reference-dataset downloads now fan out concurrently (`batlab/datasets/_download.py`: `download_parallel()` / `download_all_reference_data()`, one flaky host can't abort the rest); the Parquet cell store reads memory-mapped with read-time column pruning (`cell_store.get_cell_df(columns=…)`); and opt-in perf regression guards (`RUN_PERF_TESTS=1 python -m pytest tests/test_perf_regressions.py -q`) put explicit timing budgets on the two hottest paths (LCO on the real NASA fleet, full-cell Parquet reads) plus always-on correctness smokes
- **Accessibility-audited UI** — a real heading structure (screen readers can navigate by section, not just one `<h1>` per page), `aria-live` regions on live-updating content, keyboard-reachable equivalents for every hover-only tooltip, and sitewide WCAG AA color contrast, all enforced going forward by structural guard tests in CI
- **State-of-Power-aware second-life fit** — `application_fit()` now checks a pulse-power application (UPS/backup) against the cell's actual peak-power capability (State-of-Power, derived from resistance growth), not just SOH/fade-rate, so a cell with healthy capacity but degraded power delivery is correctly flagged unfit for pulse duty; the same signal feeds the EU Passport's End-of-Life R-code recommendation
- **Trajectory-based pack imbalance detection** — the Virtual Pack Builder compares cell fade trajectories across their shared cycling history, not just today's SOH spread, flagging a pack that's still balanced now but actively diverging and naming whichever cell is fading fastest before it becomes today's bottleneck
- **Usage-profile-aware RUL** — cycling regime (EV-like / stationary-like / mixed duty cycle, classified from rolling C-rate/depth-of-discharge variability rather than raw instantaneous values) is now a real GBRT feature, not just a display-only diagnostic
- **Opt-in warm-start incremental model updating** — re-analysing a repeat upload can extend the existing model with new boosting estimators (`sklearn`'s `warm_start`) instead of always refitting from scratch, with a hard estimator cap and automatic fallback to a full refit when the feature set changes or the cap is reached
- **BMS connector coverage expanded** from Victron VRM/Orion Jr2 to Modbus TCP, CAN bus, and OCPP (EV charging Central System) — same never-raise, credential-guarded, untested-against-live-hardware pattern as the original two
- **Warranty-breach risk estimate** (Decide & Ask) — projects cycles/probability of crossing an illustrative warranty SOH floor, distinguishing an always-available linear extrapolation from a model-scaled estimate that reuses the LCO-validated RUL quantiles rather than presenting both as equally certain
- **Second-life buyer matching** — an org's own saved buyer profiles (application type, minimum SOH, offered price) are ranked against a cell using the same `application_fit()` scoring the rest of the platform trusts, and an accepted match becomes a trackable record — turning a scored recommendation into a closed-loop transaction within this platform's own data, since no real external marketplace API exists to source live buyers from
- **Cradle-to-grave carbon footprint** — a genuine per-cell CO₂e total (chemistry-specific manufacturing figures, real use-phase from this cell's own measured cumulative energy throughput, an optional recycling-avoided-emissions credit), still explicitly not a certified Art. 7 audit — see the Sustainability tab's own disclosure for why
- **Recycler routing recommendation** — a small, dated directory of real, currently-operating recyclers (chemistry- and region-matched), surfaced on the Passport only when the recommended End-of-Life pathway actually calls for recycling
- **Residual-value / bankability report** — a financing-grade PDF packaging SOH, leave-cell-out-validated RUL quantiles, second-life fit, and NPV comparison, with an explicit not-investment-advice/not-a-rating/not-a-guarantee disclaimer
- **Multi-jurisdiction compliance-shaped exports** — US IRA Section 30D and China's 2026 EV-battery-recycling Interim Measures, same field-structure-demonstration discipline as the EU Passport (real regulatory constants shown as available, anything requiring supply-chain-of-custody or platform-registration data honestly marked unavailable)
- **Shared multi-stakeholder fleet view** — the same cell sliced three genuinely different ways (OEM / operator / recycler), previewable on the Compliance page and reachable externally via 3 new REST endpoints (`GET /cells/{id}/view/{oem|operator|recycler}`) gated by the same JWT auth as every other API route — no new login system
- **Adapter plugin registry** — new integrations declare their config fields as data instead of new hand-written Settings UI each time; proved by adding a 6th BMS adapter (a configurable generic REST connector) with zero adapter-specific widget code
- **Multi-destination webhooks** — an org can fan the same events out to more than one destination (Slack *and* PagerDuty *and* a custom CRM webhook at once) on top of the original single-URL setting, which keeps working unchanged

Currently demonstrated using public datasets and simulated telemetry. Real BMS integration requires real hardware data.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

See [`docs/history.md`](docs/history.md) for its full build history and architecture.

## Roadmap preview

The platform's formal product direction — why "Battery Research Platform"
is the near-term focus, and what's explicitly deferred — is written up in
[`docs/product_direction.md`](docs/product_direction.md).

**Phase 1 — Research foundation** *(current)*
The `batlab` library itself: standardized dataset loaders for all five datasets, literature-cited feature engineering, leave-cell-out-validated GBRT/quantile models, quantile-interval calibration, and reproducible benchmark manifests. This phase is what's installable and tested today.

**Phase 2 — Industrial analytics** *(partially delivered)*
Turning per-cell diagnostics into fleet- and deployment-level decision support — the demo app's fleet view, EU Battery Passport, and Solar + Storage Sizing / second-life economics calculator are working examples of this phase, built on public data and clearly labeled assumptions rather than a live industrial dataset.

**Lifecycle Intelligence layer** *(library + API, honestly gated)*
The market-data adapter, health-aware dispatch, grid-services revenue stack, managed charging, and fleet-aggregation modules above are the platform's answer to the battery-storage-software cohort (Capture Energy, Solship, Deepgrid — see the competitive comparison): none of them price degradation into dispatch. All of it is library-level and REST-exposed (`GET /market/prices`, `POST /analytics/dispatch-schedule`, `/analytics/dispatch-comparison`, `/analytics/grid-services-revenue`, `/analytics/managed-charge-plan`, `POST /fleet/dispatchable-capacity`, plus the P2 surface: `GET /cells/{id}/health` health-as-a-service and `POST /analytics/ml-anomaly`) with the same honest labels as the rest of the platform — and it does NOT claim the fleet-operator trigger is met; real BMS validation remains the gate ([`docs/lifecycle_intelligence_trigger.md`](docs/lifecycle_intelligence_trigger.md)).

**Phase 3 — Digital twin architecture** *(architecture delivered, deeper twin gated)*
`src/digital_twin.py` now defines the architecture: a `CellTwin` holds one cell's measured history, its derived health indicators, and a physics-based SEI sqrt-fade projection in one continuously-updated representation (`GET /cells/{id}/twin`, plus a Live Monitor block that re-fits it against streamed telemetry). The honest limits remain explicit: the parameter set is fixed per chemistry (not re-parameterized from telemetry), there is no real BMS feed, and the same real-BMS-validation trigger gates a deeper twin — so this is the architecture, honestly labeled, not the Siemens/ABB-grade twin.

**Platform & production hardening** *(delivered where testable, documented where it needs a real deployment)*
PostgreSQL is one `DATABASE_URL` away (`src/db.py` is SQLAlchemy with a dialect guard; `scripts/migrate_sqlite_to_postgres.py` copies SQLite → Postgres and applies the per-org row-level-security policies with `--apply-rls`) — **validated end-to-end against a local PostgreSQL 18.4** (`scripts/postgres_dev.py` provisions it; the full test suite passes on Postgres; repeatable via `PG_VALIDATE=1 python -m pytest tests/test_migration_e2e.py`). JWT signing keys support rotation (`JWT_PREVIOUS_SECRETS`, `kid`-tagged tokens, `src/secrets_store.py` with env/file/cloud store adapters). The REST layer has per-org rate limiting (`RATE_LIMIT_PER_ORG_PER_MINUTE`, disabled by default, `src/rate_limit.py`). Enterprise SSO is wired into the actual sign-in path: when an OIDC provider is configured (`SSO_OIDC_ISSUER`/`SSO_CLIENT_ID`/`SSO_CLIENT_SECRET`), the login page shows a "Continue with enterprise SSO" button (anti-CSRF state + replay nonce stashed in the session), and the callback route verifies the state, exchanges the code with nonce verification, and provisions or links the account against the existing User model — still untested against a live IdP tenant (`src/sso.py`, honest not-configured gate in Settings). Server-side write gating closes the last Enterprise Readiness gap: `src/rbac.py` holds the single role→capability registry — the write actions (`src/api.py`'s `require_action` enforces them: create & triage -> admin/engineer/fleet, external dispatch -> admin/engineer only, the read-only Compliance role denied every write) and the UI affordances (the `settings.manage` capability behind the Settings page's admin-only sections, and the `ui.nav.*`/`ui.frontload.*` capabilities that drive which sidebar nav groups a persona sees expanded) all read from the same object, so the app can't hardcode a role check that enforcement doesn't know about — no drift between what the UI lets a role do and what the server allows. And the Postgres backend is validated under real concurrent multi-user load, not just round-trip correctness: `tests/test_postgres_concurrency.py` (opt-in `PG_CONCURRENCY=1`) exercises parallel org creation on one pooled engine (gap-free ids under concurrent sequence `nextval`), many-org concurrent writes landing every row exactly once, row-lock contention on a shared row, and RLS isolation holding under concurrent readers — all passing against the local PostgreSQL 18.4.

**Phase 4 — Real-time battery integration** *(formalized, not yet validated)*
The Victron VRM and Orion Jr2 adapters (`src/bms_connectors.py`) now share one formal `BMSAdapter` protocol, and the MQTT ingestion path (`src/mqtt_stream.py`) has explicit fault detection for malformed/corrupted telemetry (missing fields, bad timestamps, dropped packets, unit mixups — exercised by a synthetic fault-injection harness replaying real public cycling data, `tests/synthetic_ingestion/`). What remains is the actual prerequisite this phase was always about: validating the adapters against a real, live account, and pointing the Live Monitor page's MQTT stream at real telemetry instead of its current simulated replay feed. Formalizing the interface is not the same as proving it against real hardware — that step hasn't happened yet.

## Limitations

Being explicit about what this platform is not, as of today:

**Currently:**
- No proprietary factory or manufacturer data — only the five public datasets listed above
- No real vehicle or stationary-storage fleet — fleet views operate on the same public-dataset cells or an honestly-labelled synthetic fleet
- No validated live BMS connection — the Victron/Orion adapters (now unified under one formal `BMSAdapter` protocol) exist in code but have never been run against a live account, and Live Monitor's telemetry stream is a simulated replay, not a real one

**Future, contingent on real access:**
- Industrial or research partnerships providing real operational data
- Real telemetry replacing the current simulated Live Monitor feed
- Hardware validation of the existing BMS connector adapters against actual devices

See [`docs/history.md`](docs/history.md) for the fuller production-readiness roadmap this summary is drawn from.

## Installation & Setup

### Developer Quickstart (Git Clone)

If you are setting up the project from scratch on a new machine or after cloning:

```bash
# 1. Clone the repository
git clone https://github.com/seyedali1996lb-svg/battery-intelligence-platform.git
cd battery-intelligence-platform

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# 3. Install the project in editable mode with development dependencies
pip install -e ".[dev]"

# 4. Verify installation by running test suite
pytest tests/ -v
```

---

### Running the Applications Locally

#### 1. Interactive Streamlit Dashboard (Default)
```bash
streamlit run app/main.py
```
Opens in your browser at `http://localhost:8501`.

#### 2. FastAPI REST Backend
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive API documentation available at `http://localhost:8000/docs`.

#### 3. React Frontend (Optional)
```bash
cd frontend
npm install
npm run dev
```

---

### Non-developer Setup (No Git Required)

This gets the Streamlit dashboard open in your web browser with minimal steps:

1. **Install Python:** Go to [python.org/downloads](https://www.python.org/downloads/) and click "Download Python".
   - **Windows:** on the very first screen, tick **"Add Python to PATH"** before clicking Install.
   - **Mac:** run the installer normally.
2. **Download this project:** Click the green **`Code`** button near the top of this GitHub page, then **`Download ZIP`**. Extract the ZIP to a known folder (e.g. Desktop).
3. **Open a terminal in that folder:**
   - **Windows:** open the unzipped folder in File Explorer, click the address bar, type `cmd`, and press Enter.
   - **Mac:** open Finder, right-click the folder → "New Terminal at Folder".
4. **Run the following commands:**
   ```bash
   pip install -r requirements.txt
   streamlit run app/main.py
   ```
5. The browser will open automatically at `http://localhost:8501`.

---

### Using `batlab` as a Python library

```bash
pip install -e ".[severson,oxford,calce]"   # extras are only needed for those loaders' parsers
```

Core `batlab` depends only on `pandas`, `numpy`, `scikit-learn`, `scipy`, and `requests`.

```python
import batlab
from batlab.datasets import load_nasa_cells, validate_schema
from batlab.features import build_features, get_model_matrix
from batlab.models import train_models, predict
from batlab.validation import run_lco

cells = load_nasa_cells()                       # {cell_id: DataFrame}, one standardized schema
validate_schema(cells["B0005"], kind="cycle")    # raises SchemaError with a specific message on any violation

lco = run_lco(cells)                             # leave-cell-out, not a row-level split
print(lco["rul_reliable"], lco["rul_r2"])

print(batlab.cite())                             # BibTeX for the library
print(batlab.cite(dataset="nasa"))               # + license, for whichever dataset you used
```

![NASA B0005 measured vs. predicted SOH](docs/assets/quickstart_soh.png)

*Real output from `notebooks/01_quickstart.ipynb`, not a mockup — GBRT trained on the 4 committed NASA PCoE cells.*

The full ten-minute tour is [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb). See [`docs/datasets/`](docs/datasets/index.md) for each dataset's schema, citation, and license, and [`batlab/datasets/CONTRIBUTING.md`](batlab/datasets/CONTRIBUTING.md) for adding a fifth loader.

---

## Deployment Guide

### Option 1: Streamlit Community Cloud (Fastest for Dashboard)

To host the interactive dashboard for free on Streamlit Cloud:

1. Push or fork this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**, select your repository, branch (`master`), and set:
   - **Main file path:** `app/main.py`
4. *(Optional)* Under **Advanced settings**, add environment secrets such as:
   ```toml
   ANTHROPIC_API_KEY = "your-claude-api-key"
   ```
5. Click **Deploy**.

---

### Option 2: Docker Container (REST API Backend)

The project includes a production `Dockerfile.api` for running the FastAPI service in a containerized environment (AWS ECS, Google Cloud Run, Azure Container Apps, or Docker Swarm):

```bash
# 1. Build the Docker container image
docker build -f Dockerfile.api -t battery-intelligence-api:latest .

# 2. Run the container
docker run -d -p 8000:8000 \
  -e PORT=8000 \
  -e JWT_SECRET="your-jwt-secret" \
  --name battery-api \
  battery-intelligence-api:latest

# 3. Test endpoint health
curl http://localhost:8000/health
```

---

### Option 3: Production Linux VM (Ubuntu / Debian / AWS EC2)

To deploy as a resilient background service with automatic restarts:

1. **Clone and setup repository on the server:**
   ```bash
   sudo git clone https://github.com/seyedali1996lb-svg/battery-intelligence-platform.git /opt/battery-platform
   cd /opt/battery-platform
   sudo python3 -m venv .venv
   sudo .venv/bin/pip install -r requirements.txt
   ```

2. **Create a systemd service file** (`/etc/systemd/system/battery-platform.service`):
   ```ini
   [Unit]
   Description=Battery Intelligence Platform Streamlit Service
   After=network.target

   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/opt/battery-platform
   ExecStart=/opt/battery-platform/.venv/bin/streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. **Start and enable the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable battery-platform
   sudo systemctl start battery-platform
   sudo systemctl status battery-platform
   ```

4. **Configure Nginx as a reverse proxy with SSL (HTTPS):**
   ```nginx
   server {
       listen 80;
       server_name battery.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_read_timeout 86400;
       }
   }
   ```

---

### Configuration & Environment Variables

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `ANTHROPIC_API_KEY` | Optional | *None* | Enables Claude Sonnet 5 tool-calling agent in the Copilot tab. |
| `JWT_SECRET` | Optional | *Demo secret* | Secret key used to sign and verify multi-tenant JWT authentication tokens. |
| `PORT` | Optional | `8000` / `8501` | Service binding port for FastAPI or Streamlit. |

## Citation

```python
import batlab
print(batlab.cite())
```

Or see [`CITATION.cff`](CITATION.cff) for the full academic citation metadata, including a registered ORCID and a Zenodo-archived DOI for the current release. A JOSS paper draft is at [`paper/paper.md`](paper/paper.md) (not yet submitted — see its TODOs).

## License

MIT for this repository's code — see [`LICENSE`](LICENSE). Each dataset loader interoperates with a third-party public dataset that carries its own separate license; see [`docs/datasets/`](docs/datasets/index.md) or `batlab.cite(dataset=...)` before redistributing any dataset's data.
