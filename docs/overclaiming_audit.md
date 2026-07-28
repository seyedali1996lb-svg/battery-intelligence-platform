# Overclaiming audit (2026-07-28)

Part of formalizing the [product direction](product_direction.md) decision:
a pass over `README.md`, `docs/history.md`, onboarding copy, and the
Settings page's in-product roadmap, checking for language implying
fleet-readiness, live-hardware validation, or production deployment beyond
what's actually proven.

## Method

- Read `README.md` end to end against the current codebase, not memory of
  an earlier version.
- Grepped `app/` for `roadmap`, `production`, `enterprise`, `live fleet`,
  `real fleet`, `industrial deployment`, `production-grade`, and similar
  phrases across every page module and the onboarding/role-card copy in
  `app/main.py`.
- Read the Settings page's full "Production Readiness Roadmap" table
  (`app/_pages/settings.py`) and its `docs/history.md` counterpart line by
  line.
- Read `frontend/README.md` (the React proof-of-concept's own scope
  documentation).

## Finding: "portfolio demo" framing contradicted an already-decided correction

**Not** an overclaim — the opposite problem, but still copy that needed
fixing to be consistent. `app/_pages/settings.py`'s Production Readiness
Roadmap caption, and two spots in `docs/history.md` ("This platform runs as
a portfolio demo..." and "...that a portfolio project cannot provide"),
described the platform as a "portfolio demo" / "portfolio project." Ali
explicitly corrected this framing in an earlier session: this is a real
research assistant with a genuine goal of becoming an actual
energy-industry product, not a portfolio piece. Calling it a "portfolio
demo" in user-facing copy directly contradicted that decision and, more
concretely, no longer matches reality now that the [product
direction](product_direction.md) note names a real, currently-true claim
(Research Platform) instead.

**Fixed** in three places — `app/_pages/settings.py`'s roadmap caption and
both `docs/history.md` occurrences — replacing "portfolio demo"/"portfolio
project" with an explicit pointer to the product-direction decision: the
Streamlit app is a real demo built on a real research library, run with
intentional constraints for the fleet-operator use cases the roadmap table
covers, not a claim of production fleet-operations readiness. The
substance of every roadmap row (what's real, what's simulated, what's
demo-grade) was already accurate and is unchanged — only the framing
sentence around the table changed.

## Findings: no overclaiming beyond the framing issue above

Checked specifically and found already correctly hedged, with no changes
needed:

- **`README.md`** already has an explicit "Limitations" section
  ("Currently: No proprietary factory or manufacturer data... No real
  vehicle or stationary-storage fleet... No validated live BMS
  connection...") and a "Roadmap preview" phase table that marks Phase 4
  (real-time battery integration) as "formalized, not yet validated," not
  "done." The "Demo Application" section explicitly calls `app/` "an
  engineering prototype... not a production deployment" and states "Real
  BMS integration requires real hardware data."
- **Onboarding role cards** (`app/main.py`, the 4-card first-run
  interstitial) describe each role's workflow ("Monitor fleet · Prioritise
  replacements · Alerts" for Fleet Manager, etc.) without claiming live
  data or production readiness — they describe features that exist.
- **`frontend/README.md`** already states plainly "This is not the primary
  UI" and documents the frozen-proof-of-concept decision with its own
  rationale — no changes needed.
- **Settings page roadmap rows** (beyond the framing caption above) already
  distinguish "demo behaviour" from "production path" per gap, correctly
  labeling BMS/Circunomics adapters as "untested against a live account"
  and MQTT as connecting to a public test broker, not a real fleet.

## Not touched

The Settings roadmap's closing line — "Estimated effort to reach
internal-fleet MVP: 3–4 sprints" — was left as-is. It's a scoped technical
estimate (auth + persistence + MQTT production broker + Docker deployment
pipeline) about closing engineering gaps, not a claim that fleet-readiness
itself is 3–4 sprints away; the [Lifecycle Intelligence readiness
trigger](lifecycle_intelligence_trigger.md) note makes clear that a
validated live BMS account, not sprint count, is what actually gates that
claim.
