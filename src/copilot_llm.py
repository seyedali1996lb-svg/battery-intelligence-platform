"""
Battery Intelligence Copilot — LLM layer.

Rephrases an already-computed, grounded template answer (from
battery_copilot.py) using Claude Haiku. Falls back to the template
string unchanged if no API key is passed or the API call fails.

Usage:
    from copilot_llm import llm_answer
    answer = llm_answer(query_label, template_answer, api_key)
"""

from __future__ import annotations


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
# Main LLM answer function
# ---------------------------------------------------------------------------

def llm_answer(
    query_label: str,
    template_answer: str,
    api_key: str,
) -> str:
    """
    Rephrase and clarify an already-computed, grounded template answer using
    Claude Haiku. The template answer (built by battery_copilot.py's
    context_summary()/answer_query()/answer_*() functions, all of which only
    use values already present in the model bundle) is the single source of
    truth passed as context — Claude is instructed to rephrase and clarify
    it, not introduce new facts or numbers.

    Falls back to template_answer unchanged if api_key is empty, or if the
    API call fails for any reason (network error, invalid key, rate limit)
    — never raises, so callers don't need their own try/except.
    """
    if not api_key:
        return template_answer

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_prompt = (
            f"The user's question was: \"{query_label}\"\n\n"
            "Rephrase and clarify the grounded answer below in clear, natural "
            "prose for a battery engineer. Do not add any fact, number, or claim "
            "that isn't already present in it.\n\n"
            f"<template_answer>\n{template_answer}\n</template_answer>"
        )
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    except Exception:
        return template_answer
