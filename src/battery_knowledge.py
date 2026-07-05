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
]
