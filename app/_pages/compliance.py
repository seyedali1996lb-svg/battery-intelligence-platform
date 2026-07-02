"""Page: Compliance — tabbed view combining Passport and Reports."""

import streamlit as st


def page_compliance(selected: str, df, bundle, rul_reliable: bool):
    from _pages.passport import page_passport
    from _pages.reports  import page_reports

    st.markdown("# Compliance")
    st.markdown("##### EU Battery Regulation 2023/1542 · Passport fields · PDF report export")

    _tab_passport, _tab_reports = st.tabs(["EU Battery Passport", "Reports & Export"])

    with _tab_passport:
        page_passport(selected, df, bundle, rul_reliable)

    with _tab_reports:
        page_reports(selected, df, bundle, rul_reliable)
