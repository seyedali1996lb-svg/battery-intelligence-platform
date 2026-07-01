"""
Copilot narrative templates.

All prose strings live here so the team can update wording without touching
the logic in copilot.py. Variables are filled with Python's str.format_map().
Keep placeholders in {braces}. Do not add new placeholders without a
corresponding key in the context dict that answer functions build.
"""

TEMPLATES = {

    # ── Health ────────────────────────────────────────────────────────────────

    "health_intro": (
        "**{cell}** is currently at **{soh:.1f}% state of health** — "
        "{status_note}.\n\n"
        "This reading is from cycle {cycle} of {source} ({operating_note}). "
        "Over the last 30 cycles it has been losing approximately "
        "**{fade:.2f} mAh of capacity per cycle**."
    ),

    "health_eol_reached": (
        "{cell} crossed the end-of-life threshold at cycle {eol_at} "
        "and is now {past_eol} cycles past it."
    ),

    "health_eol_projected": (
        "Based on the current fade trajectory, the end-of-life threshold "
        "is projected at cycle {eol_at} — approximately {remaining} cycles from now."
    ),

    "health_eol_none": (
        "End of life has not been reached and is not projected within "
        "the current data window."
    ),

    "health_top_driver": (
        "The model's strongest signal for this assessment is **{name}**, "
        "which accounts for {pct:.0f}% of the SOH prediction."
    ),

    # ── RUL ───────────────────────────────────────────────────────────────────

    "rul_not_calibrated": (
        "**A reliable remaining-life estimate is not available for {cell}.**\n\n"
        "During leave-cell-out validation — where the model was trained on all "
        "other cells and tested on {cell} alone — the RUL model achieved "
        "**{r2_str}** for this cell. The reliability threshold is R²≥{floor}.\n\n"
        "An R² below this floor means the model's remaining-life predictions for "
        "{cell} were less accurate than simply guessing the average remaining "
        "life across similar cells. Displaying a cycle count would create false "
        "confidence in a number the model cannot support.\n\n"
        "This does not mean {cell}'s data is wrong — it means the model has not "
        "seen enough cells with a similar degradation profile to generalise "
        "reliably to this one. Adding more cells to the training set may change this."
    ),

    "rul_eol_reached": (
        "**{cell} has already reached end of life**, crossing the 80% SOH "
        "threshold at cycle {eol_at}. It is currently at cycle {cycle}.\n\n"
        "The model's remaining-life estimate is {rul:.0f} cycles{conf}. At this "
        "stage that reflects how much further below EOL the cell continues to "
        "operate — not a forward-looking projection."
    ),

    "rul_normal": (
        "**{cell}** has an estimated **{rul:.0f} cycles of useful life remaining** "
        "before reaching the 80% SOH end-of-life threshold{conf}.\n\n"
        "This estimate is based on the cell's current capacity fade rate and "
        "resistance trajectory at cycle {cycle}. It assumes operating conditions "
        "remain similar to those observed so far — significant changes in "
        "temperature, charge rate, or depth of discharge would shift the actual "
        "remaining life."
    ),

    "rul_interval_note": (
        "The 80% prediction interval spans **{q10:.0f}–{q90:.0f} cycles**, "
        "reflecting the model's uncertainty across cells with similar degradation "
        "histories. A wide interval means the model has seen varied outcomes for "
        "cells at this stage; a narrow one means it has seen consistent outcomes."
    ),

    # ── Recent trajectory ─────────────────────────────────────────────────────

    "recent_header": (
        "**Cycles {start}–{end} ({window} cycles): {cell} is {overall}.**"
    ),

    # ── Anomaly ───────────────────────────────────────────────────────────────

    "anomaly_footer": (
        "*Peers: cells from the same data source ({source}). "
        "Cross-source comparison is not meaningful due to incompatible resistance scales.*"
    ),

    # ── Fleet compare ─────────────────────────────────────────────────────────

    "fleet_compare_cross_mode_note": (
        "*Cross-mode comparison stays out of scope — resistance scales are "
        "incompatible across NASA, synthetic, and uploaded data sources.*"
    ),

    "fleet_compare_not_enough_peers": (
        "Not enough same-mode cells for a fleet comparison.\n\n"
        "**{cell}** is in {mode} mode with only {n_peers} other "
        "cell{plural} ({peer_ids}). "
        "Fleet averages computed from fewer than 2 peers reduce to pairwise "
        "comparison — 'faster than the fleet average' would mean 'faster than "
        "one other cell,' which is not a meaningful fleet statistic.\n\n"
        "Upload at least 3 cells in the same mode to enable fleet ranking."
    ),

    # ── Alerts ────────────────────────────────────────────────────────────────

    "alerts_header": "**Fleet alert summary — {n} cells monitored**",

    "alerts_eol": (
        "**End of Life ({count} cell{plural}):** "
        "{cells} — below 80% SOH, past the replacement threshold."
    ),

    "alerts_degrading": (
        "**Degrading ({count} cell{plural}):** "
        "{cells} — between 80% and 90% SOH, monitor closely."
    ),

    "alerts_no_eol": "No cells are at or past end of life.",

    "alerts_unreliable_rul": (
        "**RUL not calibrated:** {cells} — remaining-life estimates are withheld "
        "(model fold R² below {floor} on leave-cell-out validation)."
    ),

    "alerts_all_calibrated": "All cells have calibrated RUL estimates.",

    "alerts_footer": (
        "**Fleet health:** {n_eol} at EOL · {n_deg} degrading · {n_healthy} healthy"
    ),
}
