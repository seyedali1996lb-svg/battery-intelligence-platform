"""
Second-life marketplace buyer-matching.

src/circunomics_adapter.py is purely one-way: it submits a listing and
gets back a response, with no concept of a buyer or a negotiated match.
This module turns a scored second-life recommendation into a trackable,
closed-loop transaction — ranking an org's own saved buyer profiles
(src/db.py's BuyerProfile, a saved-contacts list, not a shared
cross-tenant marketplace, since no real external marketplace API exists
to source one from) against a specific cell, reusing
consequences.application_fit()'s existing SOH/fade/power scoring rather
than inventing a second, parallel fit model.

A "match" here means a proposed/accepted/completed transaction RECORD
within this platform's own data (src/db.py's MarketplaceMatch) — not an
actual API call to a real buyer. No real external marketplace this
project has access to supports that.
"""


def score_buyer_match(
    cell_soh: float,
    cell_fade_30_mah_cy: float,
    fleet_fade_median: "float | None",
    buyer_profile: dict,
    cell_sop_pct: "float | None" = None,
) -> dict:
    """
    Score how well a cell fits one buyer profile.

    Reuses consequences.application_fit()'s existing fit scoring for the
    buyer's stated application_type — this function adds only the
    buyer-specific layer application_fit() has no way to know: the
    buyer's own minimum-SOH requirement (which may be stricter or looser
    than the application archetype's own band) and, at the caller's
    discretion, their offered price for ranking multiple eligible buyers.

    Returns {"eligible": bool, "application_fit": "fit"|"marginal"|
    "not_fit"|None, "meets_buyer_soh_floor": bool, "reasons": list[str]}.
    application_fit is None (and eligible False) if the buyer's
    application_type doesn't match any known consequences.SECOND_LIFE_APPS
    key — a real data-entry problem, not a scoring outcome, surfaced
    distinctly rather than silently treated as "not fit".
    """
    from consequences import application_fit

    app_type = buyer_profile.get("application_type")
    fit_scores = application_fit(cell_soh, cell_fade_30_mah_cy, fleet_fade_median, sop_pct=cell_sop_pct)
    app_result = fit_scores.get(app_type)

    if app_result is None:
        return {
            "eligible": False, "application_fit": None,
            "meets_buyer_soh_floor": False,
            "reasons": [f"Buyer's application type {app_type!r} doesn't match a known second-life application."],
        }

    meets_buyer_soh_floor = cell_soh >= buyer_profile.get("min_soh_pct", 0.0)
    eligible = app_result["fit"] in ("fit", "marginal") and meets_buyer_soh_floor

    reasons = list(app_result["reasons"])
    if not meets_buyer_soh_floor:
        reasons.append(
            f"Cell SOH {cell_soh:.1f}% is below this buyer's stated {buyer_profile.get('min_soh_pct', 0.0):.0f}% floor."
        )

    return {
        "eligible": eligible,
        "application_fit": app_result["fit"],
        "meets_buyer_soh_floor": meets_buyer_soh_floor,
        "reasons": reasons,
    }


def rank_buyers_for_cell(
    cell_soh: float,
    cell_fade_30_mah_cy: float,
    fleet_fade_median: "float | None",
    buyer_profiles: list,
    cell_sop_pct: "float | None" = None,
) -> list:
    """
    Score every buyer profile against one cell and rank them: eligible
    buyers first, then by offered price (highest first) among eligible
    buyers — the best economic outcome for the seller among genuinely
    fitting options. Ineligible buyers are still returned (not filtered
    out), sorted last, so a caller can show "why this buyer wasn't a fit"
    rather than silently hiding them.
    """
    scored = []
    for b in buyer_profiles:
        result = score_buyer_match(cell_soh, cell_fade_30_mah_cy, fleet_fade_median, b, cell_sop_pct=cell_sop_pct)
        scored.append({**b, **result})
    return sorted(scored, key=lambda x: (not x["eligible"], -x.get("price_per_kwh_usd", 0.0)))
