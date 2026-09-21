"""Page: Battery 3D — a cell you can look at, with every part wired to a measurement.

Why this page exists
--------------------
The platform can tell you a cell is at 84% SOH with an LLI-dominated fade and a
rising resistance, and it can chart all three. What it could not do is show you
*where* any of that is: which part of the cell is wearing out, how the pieces
relate, and how the whole thing moves through its life. That is what this view
adds — and the discipline that comes with it is that nothing on it is
decoration: every drawn part is bound to a number the platform already computes,
and a part nothing measures is drawn as architecture and says so.

What is on the page
-------------------
* **The scene** — the anatomy, drawn by the shared renderer (app/_scene_view.py
  builds the iframe; the renderer itself is the committed bundle in
  app/static/cell_scene/ that the React SPA and the standalone page load too).
* **Vitals** — the five numbers that decide how the cell reads, in the app's own
  tokens, above the canvas.
* **Part cards** — all thirteen parts, each tagged measured / derived / fitted /
  projected, with the value it carries today, what it means, and the law it came
  from. A part with no measurement says why instead of showing a zero.
* **Fit quality** — the reconciliation between the fitted two-term law and the
  measured fade, plus the disclosure strip the renderer also carries.

Nothing here is a prediction, and nothing here feeds Pack RUL or an accuracy
number: the projection drawn past "today" is the platform's own hierarchical
forecast, gated by the same per-cell routing the Health page uses, and a cell
whose routing refuses simply has no future on the timeline.
"""

from __future__ import annotations

import json

import _paths  # noqa: F401
import streamlit as st
import streamlit.components.v1 as components

from utils import _action_bar, _md_html, _empty_state, metric_tile_html, render_card, soh_status
from design_system import provenance_banner
from _scene_view import scene_iframe_html, theme_from_app

#: What the app says about a value's origin, in the renderer's vocabulary.
_PROVENANCE_LABEL = {
    "measured": ("MEASURED", "#48bb78"),
    "derived": ("DERIVED", "#63b3ed"),
    "fitted": ("FITTED", "#b794f4"),
    "projected": ("PROJECTED", "#f6ad55"),
    "": ("NOT MEASURED", "#718096"),
}


def _spec_for(cell_id, featured_dfs, bundles, graph, theme):
    """Build the scene document for one cell, or None when there is no frame.

    Everything is read from what the rest of the app already loaded — the same
    frame the Workbench charts and the same bundles the forecast comes from — so
    this page cannot show a number the page beside it disagrees with. The
    builder is `src/cell_scene.py`, which is also what `GET /cells/{id}/scene`
    serves and what the tests validate against the published schema.
    """
    from cell_scene import build_cell_scene_from_sources

    return build_cell_scene_from_sources(cell_id, featured_dfs, bundles, graph=graph, theme=theme)


def _vitals(spec: dict) -> None:
    """The five numbers that decide how this cell reads, above the canvas."""
    series = spec["series"]
    last = lambda key: next((v for v in reversed(series[key]) if v is not None), None)  # noqa: E731
    soh = spec["record"]["lastSohPct"]
    label, _css = soh_status(soh) if soh is not None else ("No reading", "")
    capacity = last("capacityAh")
    resistance = last("resistanceOhm")
    temperature = last("temperatureC")
    sop = last("sopPct")
    mechanism = (spec["mechanism"].get("physics") or {}).get("label") or (
        (spec["mechanism"].get("ml") or {}).get("verdict") or "insufficient data"
    )

    tiles = [
        ("State of health", f"{soh:.1f}%" if soh is not None else "—",
         f"{label} · {spec['record']['nCycles']} cycles plotted", "#48bb78" if (soh or 0) >= 90 else "#f6e05e" if (soh or 0) >= 80 else "#fc8181"),
        ("Capacity", f"{capacity:.3f} Ah" if capacity is not None else "—",
         "measured discharge per cycle", "#e2e8f0"),
        ("DC resistance", f"{resistance * 1000:.1f} mΩ" if resistance is not None else "—",
         f"{last('resistanceNormalized'):.2f}× its first reading" if last("resistanceNormalized") else "no baseline", "#e2e8f0"),
        ("Power (SoP)", f"{sop:.0f}%" if sop is not None else "—",
         "peak power vs. fresh", "#fc8181" if (sop or 100) < 70 else "#48bb78"),
        ("Mechanism", mechanism, spec["mechanism"]["emphasisSource"] != "none" and
         f"emphasis from {spec['mechanism']['emphasisSource']}" or "no dominant mechanism named", "#63b3ed"),
    ]
    cols = st.columns(len(tiles))
    for col, (title, value, sub, colour) in zip(cols, tiles):
        with col:
            render_card(metric_tile_html(title, value, sub, value_color=colour, value_size="19px"))
    if temperature is not None:
        st.caption(
            f"Casing tint on the scene maps this cell's measured surface temperature "
            f"({temperature:.1f} °C at the last cycle)."
        )


def _part_card(part: dict) -> None:
    """One anatomy card: value, provenance, meaning, and the law behind it."""
    tag, colour = _PROVENANCE_LABEL.get(part["provenance"], _PROVENANCE_LABEL[""])
    if not part["available"]:
        tag, colour = _PROVENANCE_LABEL[""]
    value = "—" if part["value"] is None else f"{part['value']:,.4g}"
    unit = f" <span style='font-size:11px;color:#a0aec0'>{part['unit']}</span>" if part["unit"] else ""
    reason = (
        f"<div style='font-size:11px;color:#a0aec0;margin-top:6px'>No measurement: {part['unavailableReason']}</div>"
        if not part["available"] else ""
    )
    with st.expander(f"{part['label']} — {value}{' ' + part['unit'] if part['unit'] and part['value'] is not None else ''}"):
        _md_html(
            f"<div style='font-size:12px;color:{colour};letter-spacing:0.06em;font-weight:700'>{tag}</div>"
            f"<div style='font-size:20px;font-weight:800;color:#e2e8f0;margin:4px 0'>{value}{unit}</div>"
            f"<div style='font-size:12px;color:#cbd5e0;line-height:1.55'>{part['meaning']}</div>"
            f"<div style='font-size:11px;color:#a0aec0;margin-top:8px'><b>Law:</b> {part['law']}</div>"
            f"{reason}"
        )


def page_battery3d(cell_ids, featured_dfs, bundles, selected, graph=None) -> None:
    """Render the 3D cell view for the currently selected cell."""
    _action_bar("battery3d")

    if not cell_ids or selected not in featured_dfs:
        _empty_state(
            "Nothing to draw",
            "No cell frame is loaded, so there is no anatomy to bind numbers to. "
            "Load a fleet from the sidebar first.",
        )
        return

    light_mode = bool(st.session_state.get("light_mode", False))
    theme = theme_from_app(light_mode=light_mode)

    from cell_scene import default_theme  # noqa: F401  (documents where the theme comes from)

    with st.spinner("Assembling the cell from its own measurements…"):
        spec = _spec_for(selected, featured_dfs, bundles, graph, theme)
    if spec is None:
        _empty_state("Nothing to draw", f"Cell {selected} has no loaded frame.")
        return

    cell = spec["cell"]
    _md_html(
        f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap'>"
        f"<span style='font-size:24px;font-weight:800;color:#e2e8f0'>{cell['id']}</span>"
        f"<span style='font-size:13px;color:#a0aec0'>{cell['source'] or 'source undeclared'} · "
        f"{cell['chemistry'] or 'chemistry undeclared'}</span>"
        f"<span style='font-size:11px;color:#a0aec0;border:1px solid #2d3748;border-radius:999px;"
        f"padding:1px 8px'>{cell['formFactor']}</span></div>"
    )
    st.caption(
        "Drag to orbit, scroll to zoom, right-drag to pan. Hover any part to read what it is; "
        "the controls in the panel drive the exploded view and this cell's life."
    )
    _md_html(provenance_banner(
        cell["provenance"] if cell["provenance"] in ("measured", "simulated", "synthetic") else "synthetic",
        f"Every part on this scene is bound to a measurement from {cell['source'] or 'this cell'}'s own "
        "record, or says why it carries none. Nothing here is a prediction.",
    ))

    _vitals(spec)

    components.html(scene_iframe_html(spec, height=700), height=700, scrolling=False)

    if spec["physics"]["splitIdentified"]:
        fit_note = (
            f"The fitted two-term law explains this cell's capacity history at r² = "
            f"{spec['physics']['fitR2']:.3f}, and both fade channels fit above their own standard error "
            f"— so the SEI film and the particle loss drawn above are the fitted √n and linear terms. "
            "They are still a fit to capacity data, not a measurement of a film."
        )
    else:
        fit_note = (
            f"r² = {spec['physics']['fitR2']:.3f}. This cell's capacity history cannot separate a √n "
            "lithium-inventory loss from a linear active-material loss, so the film and the particles "
            "carry no number: their geometry is architecture, the disclosed mapping uses the *identified* "
            "total fade, and the mechanism emphasis comes from the classifier instead."
            if spec["physics"]["fitR2"] is not None
            else spec["physics"]["reason"] or spec["physics"]["splitReason"] or ""
        )
    st.info(fit_note)
    if not spec["projection"]["available"]:
        st.warning(f"The scene stops at the last measured cycle. {spec['projection']['reason']}")
    else:
        st.success(
            f"Past the last measured cycle the timeline carries {len(spec['projection']['cycles'])} cycles of "
            f"{spec['projection']['modelLabel']} — with its posterior band on the state gauge. "
            "That half is a projection, and every part holds its last measured state rather than being "
            "invented forward."
        )

    st.markdown("#### The parts")
    left, right = st.columns(2)
    for index, part in enumerate(spec["parts"]):
        with (left if index % 2 == 0 else right):
            _part_card(part)

    with st.expander("What this view does and does not claim"):
        for text in spec["disclosures"]:
            st.markdown(f"- {text}")

    st.download_button(
        "Download this scene (CellSceneSpec JSON)",
        data=json.dumps(spec, indent=2),
        file_name=f"{cell['id']}_cell_scene.json",
        mime="application/json",
        help=(
            "The same document `GET /cells/{id}/scene` serves and `docs/cell_scene.schema.json` "
            "describes — renderable by any host, not only this app."
        ),
    )
