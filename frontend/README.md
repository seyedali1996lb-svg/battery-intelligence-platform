# Battery Intelligence Platform — API Proof-of-Concept Client

**This is not the primary UI.** The Streamlit app (`streamlit run app/main.py`
from the repo root) is the full, 7-page product — Overview, Health, Explore,
Compliance, Fleet, Diagnose & Decide, Live Monitor. This `frontend/` directory
is a small, deliberately incomplete React + TypeScript client demonstrating
that `src/api.py`'s FastAPI REST layer is a real, usable, JWT-authenticated,
org-scoped integration surface — not just an internal implementation detail.

## Scope decision (design review, 2026-07-06)

A cross-functional review flagged this frontend as a "215-line, 3-component
skeleton next to a full 7-page product" and asked for an explicit decision:
commit to React as the long-term primary surface, or stop letting it grow
as an ambiguous parallel effort. **Decision: Streamlit stays primary.** This
frontend is frozen as a proof-of-concept, not developed toward feature
parity. If a real API-first frontend rebuild is ever undertaken, it would be
a deliberate, resourced initiative — not an organic continuation of these
three views.

## What's here

- `src/api.ts` — typed fetch client for `src/api.py`'s REST endpoints
- `src/components/Login.tsx` — calls `POST /auth/login`, stores the JWT
- `src/components/FleetSummaryView.tsx` — calls `/fleet/summary` + `/fleet/alerts`
- `src/components/CellDetailView.tsx` — cell picker + SOH/RUL history chart

No routing, no state management library, no deployment target configured —
intentionally minimal, matching its actual purpose (prove the API works end
to end), not simulating a bigger app that isn't being built.

## Run locally

```bash
npm install
npm run dev
```

Requires `src/api.py` running separately (`uvicorn src.api:app --reload
--port 8000` from the repo root) — see the main README's "Run locally"
section for full instructions, including demo credentials.
