"""Page: Reports"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _md_html, NASA_CELL_IDS


def page_reports(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport
    from consequences import ASSUMPTIONS, application_fit, financial_comparison
    from report_pdf import build_report_pdf

    is_nasa = selected in NASA_CELL_IDS
    source  = "nasa" if is_nasa else "synth"
    p       = build_passport(selected, df, bundle, rul_reliable, is_nasa)

    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])

    st.markdown("# Reports")
    st.markdown(f"##### Demonstration report export · {selected}")

    _md_html("""<div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);border-radius:10px;padding:14px 20px;margin-bottom:28px;font-size:13px;color:#718096;line-height:1.7"><strong style="color:#63b3ed">Demonstration report</strong> — not a regulatory document. Exports the current battery's identity, SOH/RUL with reliability flags, second-life recommendation (if applicable), and the assumption register, with the same Available / Estimate / Not-available-in-demo labelling used throughout this platform.</div>""")

    second_life = None
    if soh <= 85.0:
        fade_30 = float(latest.get("fade_rate_30cy", 0.0))
        fit     = application_fit(soh, fade_30, fleet_fade_median=None)
        best_key, best = max(fit.items(), key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]])

        a   = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        fc  = financial_comparison(
            soh=soh, source=source,
            recycling_value=a["recycling_value"], new_cell_cost=a["new_cell_cost"],
            sl_value_per_kwh=a["second_life_value_per_kwh"], repack_cost=a["repack_cost"],
        )
        second_life = {
            "best_app": best["name"],
            "best_fit": best["fit"],
            "financials": {
                "Reuse (second-life)": fc["sl_net"],
                "Recycle now":         fc["recycle_value"],
                "Buy new cell":        -fc["new_cell_cost"],
            },
        }

    st.markdown("##### Preview")
    st.markdown(f"**Cell:** {selected} &nbsp;·&nbsp; **SOH:** {soh:.1f}%")
    if second_life:
        st.markdown(
            f"**Second-life fit:** {second_life['best_app']} ({second_life['best_fit']}) — "
            f"figures are cited estimates, see Consequences page for full sliders."
        )
    else:
        st.markdown("**Second-life fit:** still in primary life — no recommendation yet.")
    summ = p["summary"]
    st.markdown(
        f"**Field coverage:** {summ['n_available']} available · {summ['n_estimated']} estimated · "
        f"{summ['n_unavailable']} not available in demo"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    pdf_bytes = build_report_pdf(p, second_life, ASSUMPTIONS)
    st.download_button(
        label="Download demonstration report (PDF)",
        data=pdf_bytes,
        file_name=f"battery_passport_{selected}.pdf",
        mime="application/pdf",
        type="primary",
    )
