"""
Small, hand-written battery-domain knowledge corpus for the Copilot's
retrieval step (src/copilot_retrieval.py). Each entry is an original
summary in my own words, not copied from any paper or standard —
written to explain concepts the model bundle's numbers don't explain
by themselves (why a metric matters, what a term means).
"""

DOCUMENTS: list[dict] = [
    {
        "id": "iec62619-thermal-runaway",
        "text": (
            "IEC 62619:2022 defines THERMAL_RUNAWAY_PRECURSOR as a temperature rise "
            "exceeding roughly 5C per monitoring step. It is a precursor warning, not "
            "confirmation of runaway itself — it flags that the rate of heating has "
            "crossed a threshold consistent with an early exothermic reaction (e.g. "
            "separator breakdown, internal short). Recommended response: halt active "
            "charge/discharge and inspect before resuming."
        ),
    },
    {
        "id": "iec62619-undertemperature",
        "text": (
            "IEC 62619:2022's undertemperature limit (around -20C) exists because "
            "charging a lithium-ion cell below 0C risks lithium plating on the anode "
            "instead of normal intercalation. Plated lithium doesn't fully reversibly "
            "de-plate, causing permanent capacity loss and, in worse cases, dendrite "
            "growth that can pierce the separator. Discharging in the cold is safer "
            "than charging in the cold."
        ),
    },
    {
        "id": "lli-vs-lam",
        "text": (
            "Loss of Lithium Inventory (LLI) and Loss of Active Material (LAM) are the "
            "two dominant degradation mechanisms tracked by this platform. LLI shows up "
            "as a falling Coulombic Efficiency trend — lithium is being consumed by "
            "side reactions (mainly SEI growth) faster than it cycles reversibly. LAM "
            "shows up as rising internal resistance and nonlinear (accelerating) fade — "
            "active electrode material is physically degrading, cracking, or losing "
            "electrical contact. A cell can show both at once (\"Mixed\"), and the "
            "distinction matters for the recommended action: LLI-dominant cells often "
            "respond to milder charge protocols; LAM-dominant cells usually cannot be "
            "recovered by protocol changes."
        ),
    },
    {
        "id": "chemistry-lfp-vs-nmc-vs-lco",
        "text": (
            "LFP (lithium iron phosphate) has a flat voltage plateau around 3.2V and "
            "very stable cycle life, but lower energy density and no distinct dQ/dV "
            "peak — the OCV-derivative simulation this platform uses for LiCoO2 cells "
            "does not apply to LFP and is explicitly disabled for Severson cells. "
            "LiCoO2 (used by the NASA PCoE cells here) has higher energy density but "
            "is more thermally sensitive and shows a sharper dQ/dV peak that shifts and "
            "shrinks with degradation. NMC/NCA sit between the two: good energy "
            "density, moderate thermal stability, and dQ/dV behavior similar to LiCoO2 "
            "but with different peak positions depending on nickel content."
        ),
    },
    {
        "id": "dqdv-interpretation",
        "text": (
            "Differential capacity analysis (dQ/dV) plots the derivative of capacity "
            "with respect to voltage during a charge or discharge. Peaks correspond to "
            "phase transitions in the electrode material. As a cell degrades, dQ/dV "
            "peaks typically shrink in height (less active material undergoing the "
            "transition), shift position (electrode potential changes), and broaden "
            "(more polarization / less uniform reaction). This platform tracks peak "
            "value, peak SOC position, area under the curve, and full-width-at-half-"
            "max (FWHM) as degradation-tracking features."
        ),
    },
    {
        "id": "calendar-vs-cycle-aging",
        "text": (
            "Calendar aging is degradation that happens purely from elapsed time at a "
            "given state of charge and temperature, independent of how many "
            "charge/discharge cycles occurred. Cycle aging is degradation driven by "
            "the electrochemical stress of actual charge/discharge events (C-rate, "
            "depth of discharge). A cell sitting unused in a warehouse still ages "
            "calendrically (SEI keeps growing slowly even at rest); a cell cycled hard "
            "ages from both mechanisms simultaneously. Separating the two matters for "
            "second-life suitability: a low-cycle-count cell that's been calendar-aged "
            "for years may have less remaining life than its cycle count alone "
            "suggests."
        ),
    },
    {
        "id": "second-life-suitability",
        "text": (
            "A cell is generally considered a second-life candidate once its SOH falls "
            "to roughly 70-85% of original capacity — no longer suitable for demanding "
            "EV duty (where energy density and consistent range matter most) but still "
            "useful for stationary storage applications with lower power/energy "
            "density requirements. Key second-life screening criteria beyond raw SOH: "
            "internal resistance rise (affects usable power), pack-level SOH spread "
            "(mismatched cells in a repurposed pack degrade the weakest cell fastest), "
            "and remaining calendar life at the storage application's typical "
            "operating temperature."
        ),
    },
    {
        "id": "eu-battery-passport-fields",
        "text": (
            "The EU Battery Regulation (EU) 2023/1542 requires a Battery Passport with "
            "identity, state-of-health, lifecycle, and carbon-footprint information. "
            "Fields split into three honesty categories in this platform: 'available' "
            "(a real pipeline output, like SOH from the trained model), 'estimated' "
            "(a cited literature or industry assumption, like CO2 manufacturing "
            "footprint), and 'unavailable' (something the regulation requires that "
            "this demo genuinely cannot provide, like manufacturer supply-chain "
            "records or an accredited third-party carbon audit)."
        ),
    },
    {
        "id": "severson-dataset-methodology",
        "text": (
            "The Severson et al. (2019, Nature Energy) dataset cycles 124 commercial "
            "LFP/graphite cells under different fast-charging protocols specifically "
            "to study whether early-cycle data can predict eventual cycle life — the "
            "same underlying question this platform's own models try to answer. Cells "
            "span a range of fast-charge protocols, producing genuinely different "
            "cycle-life outcomes (from a few hundred to over a thousand cycles) purely "
            "from charging-rate stress, which is why Severson cells are grouped into "
            "cycle-life bands in this platform's fleet views."
        ),
    },
    {
        "id": "nasa-pcoe-methodology",
        "text": (
            "The NASA PCoE Battery Aging Dataset (Saha & Goebel, 2007) cycles 18650 "
            "LiCoO2 cells under a fixed protocol: repeated charge/discharge at 2A "
            "constant current with periodic electrochemical impedance spectroscopy "
            "(EIS) measurements to track internal resistance directly, not just infer "
            "it from voltage sag. Because all NASA cells share the same test "
            "conditions (24C, 2A), cross-cell variation in this dataset reflects "
            "manufacturing spread rather than differences in operating stress — "
            "unlike the synthetic fleet in this platform, which deliberately injects "
            "temperature/C-rate/DoD variation to simulate a real mixed-use fleet."
        ),
    },
    {
        "id": "why-leave-cell-out-validation",
        "text": (
            "Leave-cell-out (LCO) cross-validation trains a model on all cells except "
            "one, then tests it on the held-out cell entirely. This is the correct way "
            "to ask 'does this model generalize to a battery it has never seen?' — a "
            "row-level train/test split on a multi-cell dataset would put some cycles "
            "of every cell into training, so the model would already have partially "
            "seen every cell's behavior, making the test score misleadingly "
            "optimistic. This platform's fold R² per cell (not a dataset average) is "
            "what gates whether RUL is shown or withheld for that specific cell."
        ),
    },
    {
        "id": "why-resistance-scales-differ",
        "text": (
            "Internal resistance is not directly comparable across cell formats and "
            "chemistries. NASA's 18650 LiCoO2 cells (measured via EIS) show electrolyte "
            "resistance around 0.04-0.07 ohm; this platform's synthetic model uses a "
            "bulk resistance range of 0.15-0.40 ohm. Training one combined model "
            "across both sources produced a negative R² (worse than predicting the "
            "mean) because the model couldn't reconcile the same feature name meaning "
            "physically different things. This is why this platform trains one model "
            "per data source and ranks fleet health by SOH (a percentage, scale-"
            "invariant) rather than RUL (model-dependent, not comparable across "
            "sources) when comparing cells from different origins."
        ),
    },
    {
        "id": "state-of-power-vs-state-of-health",
        "text": (
            "State of Health (SOH) measures remaining capacity relative to original — "
            "how much energy the cell can still store. State of Power (SoP) measures "
            "peak power capability relative to original, driven mainly by internal "
            "resistance (P_peak is roughly proportional to 1/R at constant voltage). "
            "A cell can have good SOH but poor SoP if its resistance has risen "
            "disproportionately to its capacity fade — meaning it still holds a lot "
            "of energy but can't deliver it quickly, which matters more for power-"
            "hungry applications (fast acceleration, fast charging) than for steady, "
            "low-power storage."
        ),
    },
    {
        "id": "coulombic-efficiency",
        "text": (
            "Coulombic Efficiency (CE) is the ratio of discharge capacity to the "
            "previous charge capacity for the same cycle — ideally very close to "
            "100%. A CE below 100% means some charge went into an irreversible side "
            "reaction instead of reversible lithium storage, most commonly SEI "
            "(solid-electrolyte interphase) growth on the anode. A CE trending "
            "downward over many cycles (even by a fraction of a percent per cycle) is "
            "one of the earliest, most sensitive signals of accelerating lithium "
            "inventory loss — often visible before capacity fade itself accelerates."
        ),
    },
    {
        "id": "knee-point-degradation",
        "text": (
            "Battery capacity fade is rarely linear over a cell's full life. Most "
            "cells show a long, roughly linear or slowly-accelerating plateau, then a "
            "'knee point' where fade suddenly accelerates sharply — the onset of "
            "compounding failure mechanisms (e.g. lithium plating triggering further "
            "SEI growth, or mechanical particle cracking accelerating LAM). Detecting "
            "the knee early is one of the most actionable signals in fleet "
            "management, because remaining life past the knee drops much faster than "
            "a simple linear extrapolation from before the knee would suggest."
        ),
    },

    # -----------------------------------------------------------------------
    # IEA industry-context entries (added for the Solar + Storage Sizing
    # feature's "Industry context" callout). Same 2-field schema as above —
    # citations embedded inline in the prose, not a separate field. Grounded
    # in real figures pulled from the actual IEA report pages (WebFetch +
    # WebSearch against iea.org while writing these, not from memory) —
    # see src/deployment_sizing.py's module docstring for why citation
    # accuracy matters here as much as anywhere else in this app.
    # -----------------------------------------------------------------------
    {
        "id": "iea-ev-battery-deployment-2026",
        "text": (
            "Global EV battery deployment reached about 1.2 TWh in 2025, roughly "
            "30% higher than 2024, and the IEA projects this climbing toward "
            "3 TWh by 2030 and 4-5 TWh by 2035 (higher still, near 9 TWh, under "
            "its Net Zero pathway). That scale matters for second-life planning: "
            "the much larger fleet of batteries entering service now becomes the "
            "much larger pool of batteries reaching end-of-life a decade or so "
            "later, which is the pipeline second-life and recycling capacity has "
            "to be sized against. (Source: IEA, Global EV Outlook 2026.)"
        ),
    },
    {
        "id": "iea-lfp-chemistry-shift-2026",
        "text": (
            "Lithium iron phosphate (LFP) now accounts for more than 55% of EV "
            "batteries deployed globally, up from around 50% in 2024 — but the "
            "shift is regionally lopsided. LFP powers roughly two-thirds of "
            "electric car sales in China and other emerging markets, while EU "
            "adoption is above 10% and US LFP share actually fell sharply "
            "(nearly halved) in 2025. Since LFP's flatter degradation curve and "
            "lower fire risk are exactly the traits that make a cell attractive "
            "for stationary second-life reuse, this regional split shapes where "
            "second-life-suitable chemistry is actually concentrated. (Source: "
            "IEA, Global EV Outlook 2026.)"
        ),
    },
    {
        "id": "iea-battery-cost-trends-2026",
        "text": (
            "Average EV battery prices fell about 8% in 2025, with LFP packs "
            "running over 40% cheaper per kWh than NMC on average — and "
            "regional manufacturing costs diverged further, with Chinese "
            "battery prices roughly 30% below North America's and 35% below "
            "Europe's. Falling new-cell prices are a double-edged sword for "
            "second-life economics: they lower the cost of a fresh replacement "
            "cell, which is exactly the competing option a second-life reuse "
            "decision has to beat on price. (Source: IEA, Global EV Outlook 2026.)"
        ),
    },
    {
        "id": "iea-stationary-storage-secondlife-2026",
        "text": (
            "Stationary battery storage is now a major share of total battery "
            "deployment in its own right — about one-third of all battery "
            "deployment in the United States in 2025 was stationary storage, "
            "not EVs. At the same time, the IEA describes second-life reuse of "
            "retired EV batteries as still 'challenging' in practice: safety "
            "requalification requirements and the falling price of new cells "
            "both cut into the economic case for repurposing a used pack rather "
            "than buying new. The expanding second-hand EV market is, in the "
            "meantime, extending batteries' first-life service before they "
            "reach that reuse-or-recycle decision at all. (Source: IEA, Global "
            "EV Outlook 2026.)"
        ),
    },
    {
        "id": "iea-battery-patent-dominance-2026",
        "text": (
            "Battery technology reached an unprecedented 40% share of all "
            "energy-technology patents in 2023, a concentration the IEA says no "
            "other single energy technology has ever achieved — reflecting how "
            "central batteries have become to energy security and industrial "
            "policy, not just to clean-energy deployment. China, Korea, and "
            "Japan remain the leading sources of lithium-ion patents, though "
            "their relative shares have shifted substantially: Japan's share of "
            "cathode-material patents fell from about 50% in 2010 to under 10% "
            "by 2022, while China's rose from roughly 4% to nearly 40% over the "
            "same period. (Source: IEA, The State of Energy Innovation 2026.)"
        ),
    },
    {
        "id": "iea-battery-circularity-growth-2026",
        "text": (
            "Patent filings covering battery circularity — recycling, in-vehicle "
            "reuse, and repurposing for new applications such as stationary "
            "storage — grew at roughly 42% per year from 2017 to 2023, far "
            "outpacing both battery manufacturing patents (about 16%/yr) and "
            "patenting generally (about 2%/yr) over the same period. The scale "
            "driving that growth is real: the IEA projects around 1.2 million "
            "EV batteries reaching end-of-life by 2030, rising to roughly 14 "
            "million by 2040. Asian companies held about 63% of these patent "
            "families in 2023, with China's share rising from about 5% (2013) "
            "to 29% (2023); European companies held roughly 20%, concentrated "
            "in collection and material-recovery technology rather than reuse. "
            "(Source: IEA, battery recycling innovation coverage accompanying "
            "The State of Energy Innovation 2026.)"
        ),
    },
]


# ---------------------------------------------------------------------------
# Direct id lookup — for callers that need one specific, guaranteed-topical
# snippet (e.g. the Solar + Storage Sizing "Industry context" callout) rather
# than TF-IDF-ranked retrieval. copilot_retrieval.retrieve()'s min_score
# threshold is tuned for open-ended Copilot Q&A and offers no guarantee a
# short, specific query returns one of these particular documents — a direct
# id lookup has no such ambiguity.
# ---------------------------------------------------------------------------

_DOCUMENTS_BY_ID = {d["id"]: d["text"] for d in DOCUMENTS}

# Shown next to the Solar + Storage Sizing calculator's result — the most
# directly on-topic entry for "sizing a second-life battery for stationary
# storage use".
INDUSTRY_CONTEXT_DOC_IDS = ["iea-stationary-storage-secondlife-2026"]


def get_document(doc_id: str) -> "str | None":
    """Return one corpus entry's text by id, or None if the id doesn't exist."""
    return _DOCUMENTS_BY_ID.get(doc_id)
