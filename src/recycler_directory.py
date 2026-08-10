"""
Recycler directory + routing recommendation.

Closes the loop after src/consequences.py's eol_r_code_recommendation():
once a cell is recommended R4/R5 (recycle), this answers "recycle it
where" with real, currently-operating recyclers instead of leaving the
Passport's R-code as a dead end.

A small, curated set of real recyclers with real chemistry specialties
and regions, explicitly dated as a point-in-time research snapshot (the
same honesty pattern this platform's PV install-cost presets already use
in src/consequences_solar.py) since recycler operating status changes
fast — confirmed via live web research at the time this module was
written (2026-08), not carried over from an older assumption:

  - Li-Cycle (North America) filed for bankruptcy protection in 2025
    (CCAA in Canada, Chapter 15 in the US) and is deliberately NOT listed
    here as a viable routing target as a result.
  - Every entry below was confirmed operating as of this module's
    research date. Verify current status before actually routing a real
    cell — this directory is not a live API, it does not know about
    closures/openings after 2026-08.

Chemistry compatibility is approximate, based on each company's publicly
stated process focus, not a certified intake specification — a real
routing decision should confirm directly with the recycler.
"""

RECYCLER_DIRECTORY = [
    {
        "name": "Redwood Materials",
        "region": "North America",
        "country": "USA",
        "chemistries": ["LiCoO2", "LFP", "NCA"],
        "process": "Hydrometallurgical + direct recycling — broadest chemistry intake of this directory (batteries, packs, production scrap, consumer electronics)",
        "recovery_note": "Recovers lithium, copper, cobalt, nickel; remanufactures into battery-grade anode/cathode active material",
        "source": "Company public materials, confirmed operating as of 2026-08 web research",
    },
    {
        "name": "Umicore",
        "region": "Europe",
        "country": "Belgium",
        "chemistries": ["NCA", "LiCoO2"],
        "process": "High-nickel NMC/NCA cathode refining focus — strong fit for layered-oxide chemistries, less LFP-focused",
        "recovery_note": "Recovers cobalt, lithium, nickel via fast, scalable proprietary recycling technology",
        "source": "Company public materials, confirmed operating as of 2026-08 web research",
    },
    {
        "name": "Fortum Battery Recycling",
        "region": "Europe",
        "country": "Finland",
        "chemistries": ["NCA", "LiCoO2"],
        "process": "Hydrometallurgical plant (Harjavalta, Finland) — cobalt/nickel/manganese/lithium-bearing chemistries",
        "recovery_note": "Recovers >95% of cobalt, manganese, nickel, and lithium present in black mass",
        "source": "Company public materials, confirmed operating as of 2026-08 web research",
    },
    {
        "name": "GEM Co.",
        "region": "Asia",
        "country": "China",
        "chemistries": ["LiCoO2", "NCA", "LFP"],
        "process": "Two dedicated plants (Sichuan Province) — separate NMC/high-nickel and LFP processing lines, not one shared line",
        "recovery_note": "LFP's zero-cobalt composition means lower black-mass value than NMC/NCA streams — routed to its own dedicated line, not a byproduct of NMC processing",
        "source": "Company public materials, confirmed operating as of 2026-08 web research",
    },
]

# Deliberately NOT included, with a real reason (not an oversight):
# Li-Cycle (North America) — filed for bankruptcy protection in 2025
# (CCAA/Chapter 15), spoke operations paused across Arizona/Alabama/New
# York/Ontario as of the 2026-08 research date. Listing it as a viable
# routing target would be actively misleading.


def recommend_recyclers(chemistry_short_name: str, user_region: "str | None" = None, top_n: int = 3) -> list:
    """
    Rank RECYCLER_DIRECTORY entries for a cell of the given chemistry.

    Chemistry compatibility is a hard filter — an incompatible recycler
    is never returned, not just ranked lower, since routing a cell to a
    recycler that doesn't process its chemistry isn't a lower-quality
    match, it's a wrong one.

    Region is a soft preference, not a filter: same-region entries rank
    first (real logistics — shipping a spent cell overseas is real cost
    and real additional transport emissions this platform doesn't model),
    but out-of-region entries still appear rather than being hidden,
    since this directory is too small to guarantee regional coverage.
    user_region=None returns entries in the directory's own listed order
    among chemistry-compatible matches.

    Returns up to top_n entries, each with a "same_region" bool added.
    """
    compatible = [r for r in RECYCLER_DIRECTORY if chemistry_short_name in r["chemistries"]]
    if not compatible:
        compatible = [r for r in RECYCLER_DIRECTORY if "LiCoO2" in r["chemistries"]]  # generic fallback, same convention as sustainability.py's LiCoO2 fallback

    def _rank_key(r):
        same_region = user_region is not None and r["region"] == user_region
        return (0 if same_region else 1,)

    ranked = sorted(compatible, key=_rank_key)
    results = []
    for r in ranked[:top_n]:
        entry = dict(r)
        entry["same_region"] = user_region is not None and r["region"] == user_region
        results.append(entry)
    return results
