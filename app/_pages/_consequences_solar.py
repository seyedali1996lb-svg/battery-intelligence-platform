"""Consequences page — Solar + Storage Sizing tool.

Extracted from consequences.py's page_consequences() (was 1,163 lines --
the 3rd largest file in app/_pages/). This tool (PV + battery deployment
sizing against a real hour-by-hour dispatch simulation via PVGIS) is
self-contained: its own widget inputs, its own module-level presets/
constants, its own cached PVGIS wrappers and CSV-upload parsers -- nothing
here is referenced by the rest of the EOL Economics page, and nothing it
needs comes from there except the cell's current capacity in kWh (already
computed by financial_comparison() before this is called) and the
cell_id (used only as a session_state cache key).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _empty_state, _md_html, base_layout, render_card, metric_tile_html


# ---------------------------------------------------------------------------
# Solar + Storage Sizing — consumption/tariff presets
# ---------------------------------------------------------------------------
# Simple illustrative monthly-weight curves (sum to 1.0), not literature-cited —
# a user picking "Winter-heavy"/"Summer-heavy" is approximating their own bill's
# shape, not relying on this app for a validated load-profile model. "Custom"
# is handled separately in the UI (a 12-row st.data_editor), not a weight curve.
SEASONAL_SHAPES = {
    "Flat":          [1 / 12] * 12,
    "Winter-heavy":  [0.12, 0.11, 0.09, 0.07, 0.06, 0.05, 0.05, 0.05, 0.06, 0.08, 0.10, 0.16],
    "Summer-heavy":  [0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.13, 0.13, 0.10, 0.07, 0.06, 0.05],
}

# Same illustrative-curve status as SEASONAL_SHAPES, but for hour-of-day
# rather than month-of-year — feeds deployment_sizing.build_hourly_consumption().
DAILY_LOAD_SHAPES = {
    "Flat": [1 / 24] * 24,
    "Residential (morning+evening peaks)": [
        0.020, 0.015, 0.012, 0.010, 0.010, 0.015, 0.035, 0.060,
        0.055, 0.035, 0.030, 0.030, 0.032, 0.030, 0.028, 0.032,
        0.045, 0.065, 0.080, 0.075, 0.065, 0.050, 0.035, 0.026,
    ],
    "Commercial (daytime)": [
        0.010, 0.010, 0.010, 0.010, 0.010, 0.015, 0.025, 0.050,
        0.070, 0.080, 0.085, 0.085, 0.080, 0.085, 0.085, 0.080,
        0.070, 0.055, 0.035, 0.020, 0.015, 0.012, 0.010, 0.010,
    ],
}

_COMPASS_DIRECTIONS = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}

# All cost/tariff presets below are point-in-time research snapshots, not
# live-updated data — surfaced explicitly in the UI so a user doesn't mistake
# a cited figure for a currently-guaranteed rate. Germany's feed-in tariff in
# particular is on a scheduled downward path (EEG degression), so this date
# matters more than it might for a genuinely stable figure.
COST_PRESETS_RESEARCHED_AS_OF = "July 2026"

# Only Germany/France get a feed-in-tariff preset — researched live across
# 6 EU markets (Italy, Spain, Netherlands, Poland, Austria, Portugal), none
# of which have a comparably simple, currently-stable flat per-kWh export
# rate to cite honestly: Italy uses a tax-credit mechanism, Spain and
# Poland compensate at a variable spot/marginal price, the Netherlands runs
# 1:1 net metering (not a separate export tariff, and being phased out by
# 2027), Austria's rate varies with market price for larger systems, and no
# current Portugal figure was found. Deliberately omitted rather than
# forcing a misleading single-number preset — "Custom" covers all of them.
FEED_IN_TARIFF_PRESETS = {
    "None (self-consumption only)": (0.0, None),
    "Germany (~€0.079/kWh, EEG Aug 2025)": (0.079, f"EEG feed-in tariff for surplus-feed systems ≤10kW, effective Aug 2025 (pv-magazine reporting) — researched {COST_PRESETS_RESEARCHED_AS_OF}, a point-in-time snapshot. 20-year guaranteed rate at commissioning, so a new installation's rate may differ; EEG rates are also on a scheduled downward path (degression), so verify the current published rate."),
    "France (~€0.235/kWh, <3kWp)": (0.235, f"French residential feed-in tariff for systems <3kWp, revised quarterly — researched {COST_PRESETS_RESEARCHED_AS_OF}, a snapshot, verify the current published rate before relying on it."),
    "Custom": (None, None),
}

# PV install-cost presets by country, researched live (this session): real
# ranges (Germany ~€1,400-1,600/kWp, France ~€1,300-1,700/kWp, Italy
# ~€1,100-1,500/kWp, Spain ~€1,400-1,800/kWp — IRENA Renewable Power
# Generation Costs + Fraunhofer ISE Photovoltaics Report), midpoints used
# here. BESS install cost is deliberately NOT localized — no comparably
# reliable per-country breakdown was found in this session's research
# (only a Germany-specific example and an EU-wide average), so it stays at
# SIZING_ASSUMPTIONS's single EU-wide default regardless of region chosen,
# rather than fabricating country-specific precision that isn't backed by
# a real source.
SITE_REGION_PV_COST_EUR_PER_KWP = {
    "EU average (default)": None,
    "Germany": 1500.0,
    "France": 1500.0,
    "Italy": 1300.0,
    "Spain": 1600.0,
}


@st.cache_data(show_spinner=False, ttl=86400)
def _cached_pv_yield_hourly(lat: float, lon: float, peakpower_kwp: float, tilt_deg: float, azimuth_deg: float, year: int) -> dict:
    """Cached wrapper around pvgis_client.fetch_pv_yield_hourly() — PVGIS's
    historical-year hourly output doesn't change hour to hour, and this is
    one of only two calculations on this page that hit a network API, so
    results are cached for a day per input combo. Passed to
    deployment_sizing.size_deployment() as its pv_yield_fn — azimuth_deg
    here is already in PVGIS convention (size_deployment converts once from
    the user's compass bearing before calling this for each pv_kwp step)."""
    from pvgis_client import fetch_pv_yield_hourly
    return fetch_pv_yield_hourly(
        lat=lat, lon=lon, peakpower_kwp=peakpower_kwp, tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg, year=year,
    )


@st.cache_data(show_spinner=False, ttl=86400)
def _cached_pv_yield_annual(lat: float, lon: float, peakpower_kwp: float, tilt_deg: float, azimuth_deg: float) -> dict:
    """Cached wrapper around pvgis_client.fetch_pv_yield() (PVcalc, multi-year
    climate average) — passed to size_deployment() as pv_yield_annual_fn,
    used to scale the single-reference-year hourly shape to a long-term
    average total (deployment_sizing.scale_hourly_to_multiyear_average())."""
    from pvgis_client import fetch_pv_yield
    return fetch_pv_yield(
        lat=lat, lon=lon, peakpower_kwp=peakpower_kwp, tilt_deg=tilt_deg, azimuth_deg=azimuth_deg,
    )


@st.cache_data(show_spinner=False, ttl=86400)
def _cached_tmy_ghi(lat: float, lon: float) -> dict:
    """Cached wrapper around pvgis_client.fetch_tmy_ghi() — passed to
    size_deployment() as tmy_ghi_fn when pv_weather_source="typical_year".
    Independent of PV system geometry, so cached/fetched once per site
    regardless of how many PV sizes are explored."""
    from pvgis_client import fetch_tmy_ghi
    return fetch_tmy_ghi(lat=lat, lon=lon)


def _parse_two_column_timestamp_csv(raw: "pd.DataFrame") -> "list | None":
    """Shared logic for _parse_hourly_consumption_csv()'s two-column path —
    parses column 0 as a timestamp, column 1 as a numeric value, sorts
    chronologically, and resamples to hourly (summed, so sub-hourly
    interval reads aggregate correctly). Returns list[8760] only if that
    produces exactly one full year of hourly rows, else None."""
    try:
        ts = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
        vals = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        combined = pd.DataFrame({"ts": ts, "val": vals}).dropna()
        if len(combined) == 0:
            return None
        hourly = combined.set_index("ts").sort_index()["val"].resample("h").sum()
        if len(hourly) == 8760:
            return hourly.tolist()
    except Exception:
        pass
    return None


def _parse_hourly_consumption_csv(uploaded_file) -> "list | None":
    """Parses a user-uploaded consumption CSV in either of two formats:

    (a) two columns (timestamp, kWh) — a real smart-meter-style export.
    Tried first if the file has 2+ columns, with a header row (the common
    convention), then again treating row 0 as data (a genuinely
    header-less export) if the header-row attempt didn't produce exactly
    8760 hourly rows after resampling.

    (b) one numeric column, 8760 rows, an optional header row
    (auto-detected — if the first cell isn't numeric, it's treated as a
    header and dropped) — used as-is, already assumed hourly and in order.
    Tried if (a) isn't applicable or doesn't produce exactly 8760 rows.

    Returns a list[8760] on success, or None if none of these formats
    parse to exactly 8760 numeric values."""
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        raw = pd.read_csv(uploaded_file)
        if raw.shape[1] >= 2:
            result = _parse_two_column_timestamp_csv(raw)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        raw_no_header = pd.read_csv(uploaded_file, header=None)
        if raw_no_header.shape[1] >= 2:
            result = _parse_two_column_timestamp_csv(raw_no_header)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        raw = pd.read_csv(uploaded_file, header=None)
        col = raw.iloc[:, 0]
        try:
            float(col.iloc[0])
        except (TypeError, ValueError):
            col = col.iloc[1:]
        values = pd.to_numeric(col, errors="coerce").dropna().tolist()
    except Exception:
        return None
    if len(values) != 8760:
        return None
    return values


# ---------------------------------------------------------------------------
# Solar + Storage Sizing tool
# ---------------------------------------------------------------------------

def render_solar_storage_sizing(selected: str, current_kwh: float) -> None:
    from deployment_sizing import SIZING_ASSUMPTIONS, size_deployment, night_window_hours

    # mode_landing_ess: set by the "Plan a storage deployment" use-case landing
    # in app/main.py — popped (not just read) here since page_consequences()
    # is the single common function every route to this page funnels through
    # (main.py's router sends "decision"/"consequences"/"recommendations" all
    # through decision.py's wrapper, which already read this same flag for
    # its own outer expander before calling this function), so this is the
    # one safe place to clear it for good.
    _ess_landing = st.session_state.pop("mode_landing_ess", False)
    with st.expander("☀️ Solar + Storage Sizing", expanded=_ess_landing):
        _md_html(
            """
            <div style="font-size:12px;color:#8896a8;margin-bottom:4px;line-height:1.6">
                Size a PV array + this second-life battery against a real consumption
                and tariff profile, using PVGIS (EU Commission public solar data) for
                PV yield — no local irradiance modelling. All figures below are in
                <strong>€</strong>, unlike the rest of this page ($) — this tool is
                grounded in EU tariff/solar data, so the currency switch is deliberate.
            </div>
            <div style="font-size:11px;color:#4a5568;margin-bottom:16px;line-height:1.6">
                Runs a real hour-by-hour (8760 hours/year) dispatch simulation — not a
                monthly approximation. PV magnitude is scaled to PVGIS's multi-year
                climate average (the hour-to-hour shape is still one reference year's
                real weather), and the battery's power cap derates with ambient
                temperature. Still a documented heuristic, not a forecast-aware
                optimizer — a full LP/MILP replacement was considered and explicitly
                declined, see the "Solar + Storage Sizing" row in Settings → Configure's
                roadmap table for the full list of simplifications.
            </div>
            """
        )

        siz_c1, siz_c2, siz_c3 = st.columns(3)
        siz_custom_df = None
        siz_hourly_upload = None
        with siz_c1:
            st.markdown("**Consumption**")
            siz_shape = st.selectbox(
                "Profile", list(SEASONAL_SHAPES.keys()) + ["Custom", "Upload hourly CSV"], key="siz_shape",
            )
            if siz_shape == "Upload hourly CSV":
                st.caption(
                    "One numeric column, 8760 rows (Jan 1 00:00 → Dec 31 23:00, local "
                    "time), optional header — e.g. exported from a smart meter. This is "
                    "the most accurate consumption input this tool supports."
                )
                _uploaded = st.file_uploader("Hourly kWh CSV", type=["csv"], key="siz_hourly_csv")
                if _uploaded is not None:
                    siz_hourly_upload = _parse_hourly_consumption_csv(_uploaded)
                    if siz_hourly_upload is None:
                        st.error("Couldn't parse this file into exactly 8760 numeric hourly values.")
            elif siz_shape == "Custom":
                st.caption("Enter your own monthly kWh (e.g. from utility bills).")
                _default_custom = pd.DataFrame({
                    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    "kWh": [333.3] * 12,
                })
                siz_custom_df = st.data_editor(
                    _default_custom, key="siz_custom_monthly", hide_index=True,
                    num_rows="fixed", use_container_width=True,
                    column_config={"Month": st.column_config.TextColumn(disabled=True)},
                )
            else:
                siz_annual_kwh = st.number_input(
                    "Annual consumption (kWh)", min_value=0.0, value=4000.0, step=100.0,
                    key="siz_annual_kwh",
                )
            siz_weekend_shape_label = "Flat"
            if siz_shape != "Upload hourly CSV":
                siz_daily_shape = st.selectbox(
                    "Daily shape", list(DAILY_LOAD_SHAPES.keys()), key="siz_daily_shape",
                )
                siz_use_weekend_shape = st.checkbox("Different shape on weekends", key="siz_use_weekend_shape")
                if siz_use_weekend_shape:
                    siz_weekend_shape_label = st.selectbox(
                        "Weekend daily shape", list(DAILY_LOAD_SHAPES.keys()),
                        index=0, key="siz_weekend_daily_shape",
                    )
        with siz_c2:
            st.markdown("**Tariff**")
            siz_tariff_model_label = st.selectbox(
                "Tariff model", ["Single rate", "Day/Night", "Custom hours"], key="siz_tariff_model",
            )
            siz_tariff_high = st.number_input(
                "High tariff (€/kWh)", min_value=0.0, value=0.30, step=0.01, key="siz_tariff_high",
            )
            siz_low_tariff_hours = None
            siz_tariff_low = siz_tariff_high
            if siz_tariff_model_label == "Single rate":
                st.caption("No time-of-use distinction — battery only offsets load directly, no grid arbitrage.")
            elif siz_tariff_model_label == "Day/Night":
                siz_tariff_low = st.number_input(
                    "Night tariff (€/kWh)", min_value=0.0, value=0.12, step=0.01, key="siz_tariff_low",
                )
                night_c1, night_c2 = st.columns(2)
                with night_c1:
                    siz_night_start = st.number_input(
                        "Night starts (hour)", min_value=0, max_value=23, value=23, step=1, key="siz_night_start",
                    )
                with night_c2:
                    siz_night_end = st.number_input(
                        "Night ends (hour)", min_value=0, max_value=23, value=7, step=1, key="siz_night_end",
                    )
                siz_low_tariff_hours = night_window_hours(siz_night_start, siz_night_end)
            else:
                siz_tariff_low = st.number_input(
                    "Low tariff (€/kWh)", min_value=0.0, value=0.12, step=0.01, key="siz_tariff_low",
                )
                siz_low_hours_selected = st.multiselect(
                    "Low-tariff hours (0-23)", list(range(24)),
                    default=[23, 0, 1, 2, 3, 4, 5, 6], key="siz_low_hours",
                )
                siz_low_tariff_hours = set(siz_low_hours_selected)
        with siz_c3:
            st.markdown("**PV site**")
            siz_lat = st.number_input(
                "Latitude", min_value=-90.0, max_value=90.0, value=45.80, step=0.01,
                key="siz_lat",
            )
            siz_lon = st.number_input(
                "Longitude", min_value=-180.0, max_value=180.0, value=15.98, step=0.01,
                key="siz_lon",
            )
            siz_tilt = st.slider("Tilt (°)", 0, 90, 30, key="siz_tilt")
            siz_orientation = st.selectbox(
                "Orientation", list(_COMPASS_DIRECTIONS.keys()), index=4,  # default S
                key="siz_orientation",
            )
            siz_area = st.number_input(
                "Available area (m²)", min_value=0.0, value=30.0, step=1.0,
                key="siz_area",
            )
            siz_tz_mode = st.selectbox(
                "Timezone", ["Auto (from longitude)", "Custom UTC offset"], key="siz_tz_mode",
                help="PVGIS's hourly data is UTC-timestamped; the longitude estimate can "
                     "be off by ~1h near timezone boundaries or during DST — override it "
                     "here if you know the site's real civil offset.",
            )
            siz_utc_offset_override = None
            if siz_tz_mode == "Custom UTC offset":
                siz_utc_offset_override = st.number_input(
                    "UTC offset (hours)", min_value=-12, max_value=14, value=1, step=1,
                    key="siz_utc_offset",
                )
            siz_site_region = st.selectbox(
                "Site region (PV cost preset)", list(SITE_REGION_PV_COST_EUR_PER_KWP.keys()),
                key="siz_site_region",
                help="Only PV install cost is localized — no comparably reliable "
                     "per-country BESS cost breakdown was found in research for this "
                     "tool, so battery cost stays at the EU-wide default regardless.",
            )
            if siz_site_region != "EU average (default)":
                st.caption(f"Cost snapshot researched {COST_PRESETS_RESEARCHED_AS_OF} — verify current pricing before relying on it.")
            siz_weather_source_label = st.selectbox(
                "PV weather data", ["Typical year (TMY shape)", "Single reference year (2013)"],
                key="siz_weather_source",
                help="Typical year uses PVGIS's TMY (statistically representative "
                     "months from a multi-year dataset) to shape hour-to-hour PV "
                     "generation, combined with PVGIS's real monthly totals for "
                     "magnitude — falls back to the single reference year if TMY "
                     "data is unavailable.",
            )

        fin_c1, fin_c2 = st.columns(2)
        with fin_c1:
            st.markdown("**Financial constraints**")
            siz_max_payoff = st.number_input(
                "Max payoff period (years)", min_value=1.0, value=15.0, step=1.0,
                key="siz_max_payoff",
            )
            siz_max_invest = st.number_input(
                "Max investment (€)", min_value=0.0, value=25_000.0, step=500.0,
                key="siz_max_invest",
            )
            st.caption(
                "Battery size is searched automatically (1-20 units of this cell) — "
                "not the pack-size input above, which is for the financial comparison cards only."
            )
        with fin_c2:
            st.markdown("**Feed-in tariff**")
            siz_feedin_label = st.selectbox(
                "Export remuneration", list(FEED_IN_TARIFF_PRESETS.keys()), key="siz_feedin",
            )
            _preset_val, _preset_note = FEED_IN_TARIFF_PRESETS[siz_feedin_label]
            if siz_feedin_label == "Custom":
                siz_feedin_value = st.number_input(
                    "Feed-in tariff (€/kWh)", min_value=0.0, value=0.0, step=0.01, key="siz_feedin_custom",
                )
            else:
                siz_feedin_value = _preset_val
                if _preset_note:
                    st.caption(_preset_note)

        if st.button("Calculate", key="siz_calculate_btn"):
            if siz_shape == "Upload hourly CSV" and siz_hourly_upload is None:
                st.error("Upload a valid 8760-row hourly CSV before calculating.")
            else:
                load_hourly_override = None
                if siz_shape == "Upload hourly CSV":
                    load_hourly_override = siz_hourly_upload
                    monthly_consumption = [0.0] * 12  # ignored — load_hourly_kwh_override takes over
                    daily_shape = DAILY_LOAD_SHAPES["Flat"]
                    weekend_shape = None
                else:
                    if siz_shape == "Custom" and siz_custom_df is not None:
                        monthly_consumption = [float(v) for v in siz_custom_df["kWh"]]
                    else:
                        monthly_consumption = [siz_annual_kwh * w for w in SEASONAL_SHAPES[siz_shape]]
                    daily_shape = DAILY_LOAD_SHAPES[siz_daily_shape]
                    weekend_shape = DAILY_LOAD_SHAPES[siz_weekend_shape_label] if siz_use_weekend_shape else None
                azimuth_compass = _COMPASS_DIRECTIONS[siz_orientation]
                tariff_model = {
                    "Single rate": "single_rate", "Day/Night": "day_night", "Custom hours": "custom",
                }[siz_tariff_model_label]
                weather_source = (
                    "typical_year" if siz_weather_source_label == "Typical year (TMY shape)" else "single_year"
                )

                siz_assumptions = {}
                if siz_feedin_value is not None:
                    siz_assumptions["feed_in_tariff_eur"] = siz_feedin_value
                _region_pv_cost = SITE_REGION_PV_COST_EUR_PER_KWP[siz_site_region]
                if _region_pv_cost is not None:
                    siz_assumptions["pv_install_cost_eur_per_kwp"] = _region_pv_cost

                with st.spinner("Querying PVGIS and running the hourly dispatch simulation..."):
                    result = size_deployment(
                        lat=siz_lat, lon=siz_lon, tilt_deg=siz_tilt,
                        azimuth_compass_deg=azimuth_compass,
                        available_area_m2=siz_area,
                        cell_kwh_per_cell=current_kwh,
                        monthly_consumption_kwh=monthly_consumption,
                        daily_load_shape=daily_shape,
                        weekend_daily_shape=weekend_shape,
                        load_hourly_kwh_override=load_hourly_override,
                        tariff_model=tariff_model,
                        tariff_high_eur=siz_tariff_high, tariff_low_eur=siz_tariff_low,
                        low_tariff_hours=siz_low_tariff_hours,
                        utc_offset_override=siz_utc_offset_override,
                        max_payoff_years=siz_max_payoff, max_investment_eur=siz_max_invest,
                        assumptions=siz_assumptions or None,
                        pv_yield_fn=_cached_pv_yield_hourly,
                        pv_yield_annual_fn=_cached_pv_yield_annual,
                        pv_weather_source=weather_source,
                        tmy_ghi_fn=_cached_tmy_ghi,
                    )
                st.session_state[f"siz_result_{selected}"] = result

        result = st.session_state.get(f"siz_result_{selected}")
        if result is None:
            _empty_state(
                "No sizing yet",
                "Set the inputs above and click Calculate to size a PV + battery deployment.",
                icon="☀",
            )
        elif result["winner"] is None:
            _empty_state(
                "Solar yield unavailable",
                result.get("constraint_note") or "PVGIS did not return usable data for any size explored. Try again shortly.",
                icon="☀",
            )
        else:
            winner = result["winner"]
            if result["pv_errors"]:
                st.warning(
                    f"PVGIS was unavailable for {len(result['pv_errors'])} of the PV sizes explored — "
                    "results below reflect only the sizes that succeeded."
                )
            if result.get("scaling_notes"):
                st.caption(
                    f"⚠ Multi-year averaging unavailable for {len(result['scaling_notes'])} PV size(s) — "
                    "those candidates use one reference year's data unscaled."
                )
            if not result["feasible"]:
                st.info(result["constraint_note"])

            w_cols = st.columns(4)
            with w_cols[0]:
                render_card(metric_tile_html("PV size", f"{winner['pv_kwp']:.2f} kWp"))
            with w_cols[1]:
                render_card(metric_tile_html(
                    "Battery size", f"{winner['battery_kwh']*1000:.0f} Wh",
                    sub=f"{winner['n_cells']} cells",
                ))
            with w_cols[2]:
                payback_str = f"{winner['payback_years']:.1f} yr" if winner["payback_years"] is not None else "—"
                render_card(metric_tile_html(
                    "Payback", payback_str,
                    value_color="#48bb78" if result["feasible"] else "#f6ad55",
                ))
            with w_cols[3]:
                render_card(metric_tile_html(
                    "15-yr NPV", f"€{winner['npv_eur']:,.0f}",
                    sub=f"€{winner['investment_eur']:,.0f} investment",
                    value_color="#48bb78" if winner["npv_eur"] > 0 else "#fc8181",
                ))

            monthly = winner["monthly"]
            siz_fig = go.Figure()
            months_lbl = [m["month"] for m in monthly]
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["direct_pv_kwh"] for m in monthly], name="Direct PV", marker_color="#f6ad55"))
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["battery_output_kwh"] for m in monthly], name="Battery discharge", marker_color="#68d391"))
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["grid_import_kwh"] for m in monthly], name="Still grid-import", marker_color="#4a5568"))
            siz_fig.update_layout(base_layout(
                barmode="stack",
                xaxis=dict(title="Month", tickmode="linear", zeroline=False),
                yaxis=dict(title="kWh / month", zeroline=False),
            ))
            st.plotly_chart(siz_fig, use_container_width=True)

            _total_arb = sum(m["arb_charge_kwh"] for m in monthly)
            _total_export = sum(m["export_kwh"] for m in monthly)
            st.caption(
                f"Across the year: {_total_arb:,.0f} kWh bought at the low tariff for arbitrage · "
                f"{_total_export:,.0f} kWh exported at the feed-in tariff."
            )

            with st.expander(f"All {len(result['candidates'])} sizes explored", expanded=False):
                st.caption(
                    "Every PV x battery combination evaluated (coarse + refine passes) — "
                    "the winner above is the highest-NPV row satisfying your payoff/investment limits."
                )

                _plottable = [c for c in result["candidates"] if c["payback_years"] is not None]
                if _plottable:
                    _cand_fig = go.Figure()
                    for _label, _feasible_flag, _color in (
                        ("Feasible", True, "#48bb78"), ("Not feasible", False, "#718096"),
                    ):
                        _pts = [c for c in _plottable if c["feasible"] == _feasible_flag]
                        if _pts:
                            _cand_fig.add_trace(go.Scatter(
                                x=[c["payback_years"] for c in _pts],
                                y=[c["npv_eur"] for c in _pts],
                                mode="markers", name=_label,
                                marker=dict(color=_color, size=9),
                                text=[f"{c['pv_kwp']:.2f} kWp, {c['n_cells']} cells" for c in _pts],
                                hovertemplate="%{text}<br>Payback: %{x:.1f} yr<br>NPV: €%{y:,.0f}<extra></extra>",
                            ))
                    _cand_fig.add_trace(go.Scatter(
                        x=[winner["payback_years"]], y=[winner["npv_eur"]],
                        mode="markers", name="Winner",
                        marker=dict(color="#f6ad55", size=15, symbol="star", line=dict(color="#1e2a38", width=1)),
                        text=[f"{winner['pv_kwp']:.2f} kWp, {winner['n_cells']} cells"],
                        hovertemplate="%{text}<br>Payback: %{x:.1f} yr<br>NPV: €%{y:,.0f}<extra></extra>",
                    ))
                    _cand_fig.update_layout(base_layout(
                        xaxis=dict(title="Payback (years)", zeroline=False),
                        yaxis=dict(title="NPV (€)", zeroline=False),
                    ))
                    st.plotly_chart(_cand_fig, use_container_width=True)

                _cand_df = pd.DataFrame([
                    {
                        "PV (kWp)": round(c["pv_kwp"], 2),
                        "Battery (Wh)": round(c["battery_kwh"] * 1000, 0),
                        "Cells": c["n_cells"],
                        "Investment (€)": round(c["investment_eur"], 0),
                        "Payback (yr)": round(c["payback_years"], 1) if c["payback_years"] is not None else None,
                        "NPV (€)": round(c["npv_eur"], 0),
                        "Feasible": c["feasible"],
                    }
                    for c in result["candidates"]
                ]).sort_values("NPV (€)", ascending=False, ignore_index=True)
                st.dataframe(_cand_df, use_container_width=True, hide_index=True)

            try:
                from battery_knowledge import get_document, industry_context_doc_ids
                _ctx_ids = industry_context_doc_ids(winner["payback_years"], winner["battery_kwh"])
            except Exception:
                _ctx_ids = []
            for _ctx_id in _ctx_ids:
                _ctx_text = get_document(_ctx_id)
                if _ctx_text:
                    render_card(
                        "<div style='font-size:10px;font-weight:700;color:#718096;text-transform:uppercase;"
                        "letter-spacing:0.08em;margin-bottom:8px'>Industry context</div>"
                        f"<div style='font-size:12px;color:#a0aec0;line-height:1.6'>{_ctx_text}</div>",
                        extra_style="margin-top:16px",
                    )
