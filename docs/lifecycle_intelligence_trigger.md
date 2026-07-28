# Lifecycle Intelligence readiness trigger

Part of the [product direction](product_direction.md) decision. "Battery
Lifecycle Intelligence" is the EV/ESS fleet-operator product concept — the
one the platform's `FleetAsset` hierarchy, `BMSAdapter` protocol, MQTT
ingestion fault detection, and EU Battery Passport were all built toward.
None of that is claimed as *ready* today, and this note exists so that
claim has one unambiguous, checkable trigger instead of a vague sense of
"getting close."

## The trigger

**One real BMS account, successfully connected end-to-end via the adapters
in `src/bms_connectors.py`.**

Specifically: a `VictronVRMAdapter` or `OrionBMSAdapter` instance (both
implementing the `BMSAdapter` `Protocol` in that file) fetching real
telemetry from a real, live Victron VRM or Orion Jr2 account — not the
public API shape verified by a 401 response, not a synthetic/replayed
fixture, not `test.mosquitto.org` — flowing into the standard cycle-data
schema and rendering on Live Monitor / feeding a fleet.

This is deliberately a narrow, binary condition. It does not require the
MQTT production broker migration, RBAC write-gating, the PostgreSQL
data-layer migration, or any of the other rows in the Settings page's
Production Readiness Roadmap table — those are real engineering gaps
worth closing regardless, but none of them is what actually separates
"formalized" from "validated" for this specific product claim. A live BMS
account is.

## What changes in messaging once this happens

- `README.md`'s Roadmap preview, Phase 4 line changes from *"formalized,
  not yet validated"* to a dated, specific claim: which adapter, which
  account type (anonymized/permissioned appropriately), what was verified.
- The Limitations section's "No validated live BMS connection" bullet
  moves out of "Currently" and into a dated changelog entry — not deleted,
  since Live Monitor's *default* feed likely stays a simulated replay for
  demo purposes even after one real account is validated.
- `docs/product_direction.md`'s readiness table updates the Lifecycle
  Intelligence row from "Not ready" to a real status, and this file's own
  framing shifts from "the trigger for claiming readiness" to "how
  readiness was established" (kept as a record, not deleted).
- The Settings page roadmap's "Real BMS integration" row's "Demo
  behaviour"/"Production path" split collapses into one row describing
  what's actually validated.
- Nothing about the Battery Research Platform direction changes — the two
  product concepts are independent; validating a BMS account doesn't
  retroactively change what's true about the public-dataset research use
  case, and vice versa.

## Confirmed: current copy does not jump ahead of this

Checked as part of the [overclaiming audit](overclaiming_audit.md):
`README.md` already frames Phase 4 as "formalized, not yet validated" and
explicitly separates "formalizing the interface" from "proving it against
real hardware." The Settings roadmap row and `docs/history.md`'s matching
row both say "untested against a live account" for both adapters. No
in-product copy currently claims or implies this trigger has already been
met — it hasn't.
