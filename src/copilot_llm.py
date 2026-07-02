"""
Battery Intelligence Copilot — LLM layer.

Wraps the existing template-based context system with real Claude API calls.
Falls back gracefully to template strings if ANTHROPIC_API_KEY is not set or
the API call fails.

Usage:
    from copilot_llm import llm_answer, is_llm_available

    if is_llm_available():
        response = llm_answer(query_key, ctx, ctx_b=None, fleet_stats=None)
    else:
        response = template_answer(...)   # existing copilot.py functions
"""

from __future__ import annotations

import os
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_llm_available() -> bool:
    """Return True if the Anthropic API key is set in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# System prompt — battery engineering grounding rules
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a battery engineering copilot embedded in the Battery Intelligence Platform.
Your role is to explain battery health data clearly and precisely to engineers.

STRICT RULES — follow all of them, always:
1. Only use facts explicitly provided in the <context> block. Do not invent, guess, or
   infer values not present there.
2. If a value is marked "(NOT CALIBRATED — withheld)" you must say the RUL cannot be
   shown for this cell and briefly explain why (fold R² below reliability floor).
3. Distinguish real NASA measurements from synthetic/physics-model data when the context
   says so. Never conflate them.
4. Use precise engineering language (SOH, RUL, SEI, fold R², dQ/dV, Arrhenius) where
   appropriate, but define any term on first use.
5. State confidence honestly. If uncertainty is high, say so.
6. Response format: clear prose paragraphs. No bullet-list walls. Use bold **for key
   numbers only**. Keep it under 250 words unless detail genuinely requires more.
7. Never make recommendations (Continue/Inspect/Recycle) — that is the Recommendations
   page's job. Your role is explanation, not decision.
"""

# ---------------------------------------------------------------------------
# Context → prompt serialiser
# ---------------------------------------------------------------------------

def _context_to_text(ctx: dict) -> str:
    """Convert a cell context dict to a compact, readable text block."""
    lines = [
        f"Cell ID: {ctx['cell_id']}",
        f"Data source: {ctx.get('data_source', 'unknown')}",
        f"SOH: {ctx.get('soh', '?'):.1f}% ({ctx.get('status', '?')} — {ctx.get('status_note', '')})",
        f"Current cycle: {ctx.get('current_cycle', '?')}",
        f"Fade rate (30-cycle): {ctx.get('fade_rate_30cy', 0)*1000:.2f} mAh/cycle",
        f"Fade rate (50-cycle): {ctx.get('fade_rate_50cy', 0)*1000:.2f} mAh/cycle",
        f"Fade accelerating: {ctx.get('fade_accelerating', False)} ({ctx.get('fade_ratio', 1):.1f}× baseline)",
        f"Resistance: {ctx.get('resistance_ohm', 0)*1000:.1f} mΩ ({ctx.get('resistance_normalized', 1):.2f}× initial)",
        f"Temperature (30-cycle avg): {ctx.get('temp_30cy', 25):.1f}°C",
    ]

    # RUL
    rul_reliable = ctx.get("rul_reliable", False)
    rul_pred     = ctx.get("rul_pred")
    rul_lo       = ctx.get("rul_q10")
    rul_hi       = ctx.get("rul_q90")
    if rul_reliable and rul_pred is not None:
        lines.append(f"RUL: {rul_pred:.0f} cycles (90% interval: {rul_lo:.0f}–{rul_hi:.0f} cycles)")
    else:
        rul_r2 = ctx.get("rul_fold_r2")
        r2_str = f"fold R²={rul_r2:.2f}" if rul_r2 is not None else "fold R² unknown"
        lines.append(f"RUL: NOT CALIBRATED — withheld ({r2_str} < 0.30 reliability floor)")

    # Top drivers
    drivers = ctx.get("top_drivers", [])
    if drivers:
        d_lines = [f"  {i+1}. {d.get('label', d.get('feature','?'))} ({d.get('importance_pct', 0):.0f}% split importance)" for i, d in enumerate(drivers[:5])]
        lines.append("Top SOH model drivers (split importance — see SHAP tab for correlated-feature correction):")
        lines.extend(d_lines)

    # Anomaly
    if ctx.get("is_anomalous"):
        lines.append(f"Anomaly detected: {ctx.get('anomaly_description', 'unusual behaviour flagged')}")

    # CE
    ce = ctx.get("ce_rolling_30cy")
    if ce is not None:
        lines.append(f"Coulombic efficiency (30-cycle avg): {ce:.4f} ({ce*100:.2f}%)")

    return "\n".join(lines)


def _fleet_to_text(fleet_stats: dict) -> str:
    if not fleet_stats:
        return "Fleet stats: not available."
    lines = [
        f"Fleet cells: {fleet_stats.get('n_cells', '?')}",
        f"SOH range: {fleet_stats.get('soh_min', 0):.1f}% – {fleet_stats.get('soh_max', 0):.1f}%",
        f"SOH median: {fleet_stats.get('soh_median', 0):.1f}%",
        f"EOL cells (SOH < 80%): {', '.join(fleet_stats.get('eol_cells', [])) or 'none'}",
        f"Fast-degrading cells: {', '.join(fleet_stats.get('degrading_cells', [])) or 'none'}",
        f"Uncalibrated RUL cells: {', '.join(fleet_stats.get('unreliable_rul', [])) or 'none'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query-specific user prompts
# ---------------------------------------------------------------------------

_QUERY_PROMPTS = {
    "health": (
        "Explain the health status of this cell based on the context below. "
        "Cover: current SOH and what it means, whether fade is stable or accelerating, "
        "resistance trajectory, and what the key model drivers tell us physically. "
        "Be specific about the numbers."
    ),
    "drivers": (
        "Explain why the model predicts this cell's SOH at its current value. "
        "Focus on the top model drivers: what each feature measures physically, "
        "why it is important for this cell specifically, and what the feature's current "
        "value tells us about the degradation mechanism. "
        "Note that split-based importances can mis-credit correlated fade-rate features."
    ),
    "rul": (
        "Explain the remaining useful life estimate for this cell. "
        "If RUL is calibrated, interpret the number and the uncertainty range. "
        "If RUL is not calibrated, explain the reliability gate and what can be "
        "said about remaining life from SOH and fade rate alone. "
        "Be precise about what 'remaining useful life to 80% SOH' means in practice."
    ),
    "recent": (
        "Describe what has happened to this cell in recent cycles. "
        "Focus on: whether the fade rate has changed, resistance trend, "
        "any anomalies detected, and whether the recent trajectory is consistent "
        "with or diverging from the model's prediction."
    ),
    "anomaly": (
        "Assess whether this cell is behaving unusually compared to its own history "
        "and the fleet baseline. Explain what 'anomalous' means for battery cells, "
        "whether this cell shows anomalous signals, and what physical mechanisms "
        "might produce such behaviour if present."
    ),
    "fleet_compare": (
        "Explain how this cell ranks within its fleet on key health metrics. "
        "Compare SOH, fade rate, and resistance against fleet medians/ranges. "
        "Explain what any relative advantage or disadvantage means practically."
    ),
    "alerts": (
        "Summarise the fleet alert status. List any cells at or near end of life, "
        "cells with fast degradation, and cells with uncalibrated RUL. "
        "For each flagged cell, give the key number and a one-sentence explanation."
    ),
    "compare": (
        "Compare these two cells directly. Focus on: which has better health and why, "
        "whether their degradation rates are converging or diverging, "
        "and what the resistance comparison tells us about their degradation mechanisms. "
        "Be concrete about numbers."
    ),
}


# ---------------------------------------------------------------------------
# Main LLM answer function
# ---------------------------------------------------------------------------

def llm_answer(
    query_key: str,
    ctx: Optional[dict],
    ctx_b: Optional[dict] = None,
    fleet_stats: Optional[dict] = None,
) -> str:
    """
    Call Claude to generate a grounded answer for the given query.

    Returns the response string, or raises an exception if the API call fails.
    Caller should catch exceptions and fall back to template responses.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build context block
    context_parts = []
    if ctx:
        context_parts.append(f"<context name='cell_a'>\n{_context_to_text(ctx)}\n</context>")
    if ctx_b:
        context_parts.append(f"<context name='cell_b'>\n{_context_to_text(ctx_b)}\n</context>")
    if fleet_stats:
        context_parts.append(f"<context name='fleet'>\n{_fleet_to_text(fleet_stats)}\n</context>")

    context_block = "\n\n".join(context_parts)
    user_prompt   = f"{_QUERY_PROMPTS.get(query_key, 'Explain the battery data.')}\n\n{context_block}"

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text


def llm_answer_stream(
    query_key: str,
    ctx: Optional[dict],
    ctx_b: Optional[dict] = None,
    fleet_stats: Optional[dict] = None,
):
    """
    Generator that yields text chunks from a streaming Claude response.
    Use with Streamlit's st.write_stream().
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    context_parts = []
    if ctx:
        context_parts.append(f"<context name='cell_a'>\n{_context_to_text(ctx)}\n</context>")
    if ctx_b:
        context_parts.append(f"<context name='cell_b'>\n{_context_to_text(ctx_b)}\n</context>")
    if fleet_stats:
        context_parts.append(f"<context name='fleet'>\n{_fleet_to_text(fleet_stats)}\n</context>")

    context_block = "\n\n".join(context_parts)
    user_prompt   = f"{_QUERY_PROMPTS.get(query_key, 'Explain the battery data.')}\n\n{context_block}"

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
