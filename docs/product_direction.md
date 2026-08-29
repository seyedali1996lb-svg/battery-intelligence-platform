# Product direction

**Status: decided, 2026-07-28.** This is a formal product decision, not a
speculative roadmap entry. It names what this platform is *for*, right now,
so that positioning, messaging, and prioritization stop being implicit.

## The decision

**Primary near-term direction: Battery Research Platform** — a citable,
open-source analytics toolkit for university and lab researchers working on
public battery-cycling datasets.

Concretely, that means: `batlab` (the standardized-schema, leave-cell-out-validated
research library) plus the Streamlit demo app built on it, distributed as an
open-source repository a researcher can clone, install, and run against
public datasets or their own lab's cycling data — with no proprietary
dependency anywhere in the path.

This is not a new build. It is a name for the direction the platform has
already been converging on: `batlab`'s extraction as an installable library
with its own quickstart/data-leakage/knee-detection/bring-your-own-data
notebooks, the MkDocs documentation site, `CITATION.cff` with a registered
ORCID and Zenodo DOI, and a JOSS paper draft, all predate this note. What
was missing was someone saying out loud that this is the direction, not one
open option among several.

## Why this one, and why now

The original roadmap named four possible product concepts for this
platform. Battery Research Platform is the only one of the four this
platform can honestly claim readiness for today, because it needs nothing
beyond what's already built or already in-flight in the other phases:

| Concept | Audience | Needs beyond today | Status |
|---|---|---|---|
| **Battery Research Platform** | Universities, labs | Nothing new | **Primary direction — this note** |
| Battery Lifecycle Intelligence | EV/ESS fleet operators | A real BMS account validated end-to-end | Not ready — see [Lifecycle Intelligence readiness trigger](lifecycle_intelligence_trigger.md) |
| Testing-lab analytics platform | Battery test labs | Hardened ingestion for arbitrary proprietary file formats | Deliberately deferred — see [deferred product concepts](deferred_product_concepts.md) |
| Manufacturer data-infrastructure product | Cell manufacturers | Any manufacturing-data access, even synthetic | Deliberately deferred — see [deferred product concepts](deferred_product_concepts.md) |

The other three each require something the platform does not have and
cannot fabricate: a live BMS account, a real (or even synthetic) proprietary
file format to harden ingestion against, or any manufacturing-data access at
all. Claiming readiness for any of them today would be exactly the kind of
overclaiming this project has repeatedly audited itself against (see
[`docs/history.md`](history.md)'s "Deliberate scope limits" and "Production
Readiness Roadmap" sections). The Research Platform direction requires none
of that — it only requires the platform to be honest about validation
methodology and easy to run, both of which are already true.

## What "ready" looks like, concretely

- `pip install -e ".[severson,oxford,calce]"` installs `batlab` as a
  standalone library with no Streamlit/app dependency required.
- Five public datasets (NASA PCoE, Severson 2019, Oxford Path-Dependent,
  CALCE) load through one standardized schema, with checksum-verified
  downloads and a `SchemaError` on any malformed input.
- Every model is leave-cell-out validated by default, and the gap between
  that and a naive row-level split is reproduced live in
  [`notebooks/02_data_leakage.ipynb`](notebooks/02_data_leakage.ipynb) —
  R²=0.998 (misleading) vs. R²=0.806 (honest), not asserted, run.
- [`notebooks/04_bring_your_own_data.ipynb`](notebooks/04_bring_your_own_data.ipynb)
  and the Streamlit app's Import Data page (backed by
  [`docs/import_format_guide.md`](import_format_guide.md)) both let a
  researcher bring their own lab's CSV through the identical pipeline —
  confirmed working end-to-end as part of writing this note (see
  [Research Platform setup path](research_platform_setup.md)).
- `CITATION.cff` carries a registered ORCID and a Zenodo-archived DOI; a
  JOSS paper draft exists at `paper/paper.md` (not yet submitted).
- MIT-licensed code; each dataset loader clearly separates its own
  third-party license from the repository's.

None of this required new engineering to write this note — it's an
inventory of what already exists, assembled into an explicit claim for the
first time.

## What changes because of this decision

- **Messaging leads with the library, not the demo.** `README.md`'s framing
  (open-source library + honest-validation-by-default + demo app built on
  top) already matches this and needs no further correction beyond the
  overclaiming audit below.
- **Prioritization favors researcher-facing gaps** (a fifth dataset loader,
  clearer schema docs, notebook polish, JOSS submission) over
  fleet-operator-facing gaps (live BMS validation, full RBAC write-gating,
  scalable data layer) when the two compete for the same session's time.
  Note the degree of RBAC progress: server-side write gating for the
  Decision/CMMS-ticket API and the rest of the org write surface landed in
  `src/rbac.py` — ticket create/triage → admin/engineer/fleet, external
  dispatch → admin/engineer, `decision.log` → admin/engineer/fleet,
  `webhooks.manage` and `fleet-assets.manage` (site/fleet/pack CRUD) → admin
  only, Compliance read-only — enforced at `src/api.py`'s REST boundary via
  `require_action` AND in `src/db.py` (the UI's trust boundary) through a
  shared `_require_cap`, and the Streamlit UI now reads the *same* keys (the
  Settings admin-only sections via `settings.manage` plus the granular
  webhook/fleet-asset sections via their specific capability, the sidebar's
  per-persona nav front-loading via `ui.nav.*`/`ui.frontload.*`) — so UI
  affordances and server enforcement can't drift apart. The last
  single-writer helpers were folded onto the registry too — `team.manage`
  (invite teammates) → admin, `cohort.manage` (tag a cell with a
  cohort/batch label) → admin/engineer/fleet, and `set_setting()` on
  `_ADMIN_ONLY_SETTING_KEYS` now gates on `settings.manage` instead of a
  raw `role != "admin"` — and team-members / cohort-tag / settings writes
  are reachable at the REST boundary on the same keys. What remains is
  purely edge breadth, not the split or the surface: a handful of
  single-writer read paths and session-only conveniences never reach API
  clients end to end, and per-user convenience settings are still written
  only through the UI. The fleet-operator-facing work
  already in flight (Phase 4 in `README.md`'s roadmap preview) continues —
  this note doesn't stop it — it just isn't the thing being claimed as
  *done* right now.
- **The demo app stays a demo app, on purpose.** It's real, useful evidence
  that the library's outputs assemble into a usable tool, not a
  soon-to-be-replaced placeholder for a fleet-operator product.

## See also

- [Overclaiming audit](overclaiming_audit.md) — what was checked in
  README/in-product copy against this decision, and what (if anything) was
  corrected.
- [Research Platform setup path](research_platform_setup.md) — the
  concrete clone-to-analysis path this decision claims works, verified.
- [Lifecycle Intelligence readiness trigger](lifecycle_intelligence_trigger.md)
  — the explicit condition for ever claiming the EV/ESS fleet-operator
  product is ready, and what changes in messaging when it's met.
- [Deferred product concepts](deferred_product_concepts.md) — the testing-lab
  analytics platform and manufacturer data-infrastructure product, each
  with its own real reopening trigger.
- [Hosted-researcher pricing](hosted_researcher_pricing.md) — scoped
  explicitly out of this note; see that file for why.
