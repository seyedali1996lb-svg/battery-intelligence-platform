# Hosted-researcher pricing — out of scope

Part of the [product direction](product_direction.md) decision. The task
that produced this note asked for a lightweight pricing/business-model
sketch for a possible hosted-researcher tier of the Research Platform, but
only if it seemed genuinely useful — with explicit permission to mark it
out of scope instead of guessing at numbers.

**Marked out of scope.** Reasons:

- There is no hosting infrastructure today — the platform runs locally
  (`streamlit run app/main.py`) or on a single free-tier Streamlit Cloud
  deployment for the public demo. A pricing model for a tier that doesn't
  exist yet, with no cost basis (compute, storage, support load) to price
  against, would be numbers invented to fill a template, which is exactly
  what this project's core philosophy (see the main `README.md`'s "Core
  philosophy" section — a number is only meaningful if it answers a real
  question) argues against.
- There's no signal yet that researchers want a *hosted* tier specifically.
  The open-source, self-hosted Research Platform direction this note's
  sibling document commits to is free and requires no hosting decision at
  all; a hosted tier would be a distinct product decision (who operates it,
  what data isolation it needs, what support commitment it implies) that
  hasn't been asked for by anyone using this platform.
- Drafting speculative pricing risks the opposite failure mode this whole
  task exists to prevent: stating a business commitment ("here's what we'd
  charge") that reads as decided when it isn't. The instruction to ask
  before any business/strategic commitment beyond documentation applies
  directly here.

**What would make this worth revisiting:** a real researcher or lab asking
for hosting (not self-hosting), or the maintainer deciding to actually stand
up shared hosting infrastructure for other reasons. Either would supply the
missing cost basis and real demand signal this note doesn't have.
