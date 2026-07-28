# Deferred product concepts

Part of the [product direction](product_direction.md) decision. Two of the
four original roadmap concepts are not being pursued right now. This note
exists so they're documented as *deliberately deferred*, each with a real
trigger that would reopen them, rather than silently forgotten and
rediscovered as "wait, did we ever decide against this?" in some future
session.

## Testing-lab analytics platform

**Audience:** battery test labs (contract testing houses, R&D labs running
proprietary cycler fleets) who want fleet-level analytics across their own
in-house test data, not just public datasets.

**Why deferred:** blocked on hardened ingestion for arbitrary proprietary
customer file formats. The platform today handles exactly two ingestion
shapes well: the four public datasets' own native formats (`batlab.datasets`,
checksum-verified, one schema each) and one manually-mapped CSV schema for
a researcher bringing their own data (see [Research Platform setup
path](research_platform_setup.md)) — which requires that researcher to do
the column-mapping themselves, by hand or via the bring-your-own-data
notebook as a template. A testing lab has neither: they have their own
cycler vendor's raw export format (Neware/Arbin/Maccor/BatteryArchive, each
with its own quirks), often several different formats across an
equipment-mixed fleet, and an expectation that the platform does the
mapping, not that they do. `docs/import_format_guide.md`'s own "Mapping
common BMS export formats" section documents *how a human maps a format by
hand*; it is not an automated ingestion layer, and building one for
arbitrary vendor formats without a real lab's actual files to test against
would mean guessing at edge cases that only show up in real proprietary
exports.

**Real trigger to reopen:** a real testing lab's actual raw export files
(even one lab, even a small sample) become available to build and test an
automated-mapping ingestion layer against. Guessing at the mapping logic
without real files to validate it against would repeat this project's own
documented mistake pattern (see `docs/history.md`'s "Deliberate scope
limits" — declining to substitute assumptions for real data elsewhere in
this codebase, e.g. the NASA→Oxford cross-chemistry transfer study).

## Manufacturer data-infrastructure product

**Audience:** cell manufacturers wanting to enrich their own product's
lifecycle data (EU Battery Passport, warranty analytics) with real
manufacturing birth-certificate data — batch QC pass rates, factory-measured
initial capacity, production-line provenance — rather than data inferred
from cycling behavior alone.

**Why deferred:** blocked on having any manufacturing-data access at all,
even synthetic. `src/manufacturing_connector.py`
already exists as an interface-only `abc.ABC` contract (`fetch_production_batch()`,
`fetch_cell_birth_certificate()`) — every method body is
`raise NotImplementedError`, with no `requests` import and no network code,
one deliberate step more conservative than `src/cmms_adapter.py`'s "documented
adapter pattern, no real system named" precedent. The reason it stops at an
interface: every real manufacturing execution system (Siemens Opcenter, SAP
ME, Critical Manufacturing, a custom in-house system) exposes a different,
proprietary integration surface, usually disclosed only under NDA to an
actual manufacturing partner — unlike a BMS or a maintenance-ticketing API,
there is no broadly-similar public REST shape to build even a plausible
generic implementation against. There isn't even a synthetic manufacturing
dataset to prototype against the way the synthetic cycling fleet stands in
for real telemetry elsewhere in this platform.

**Real trigger to reopen:** either a real manufacturing partnership
providing actual (even limited, even anonymized) MES data to build a real
connector against, or a credible synthetic manufacturing-data generator
becoming worth building on its own merits (unlike the cycling data case,
no existing physics-informed synthetic model for batch QC / birth-certificate
data exists to adapt).

## What this means in practice

Neither of these is on the near-term roadmap. If either audience's
data becomes available, treat it as a new scoping conversation, not an
assumption that the existing interface-only contracts are "basically done
already" — `manufacturing_connector.py` in particular documents an intended
*shape*, not a validated one.
