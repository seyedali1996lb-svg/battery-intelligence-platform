"""
Phase 5 — PDF report generation.

Builds a single-page-ish, readable, visually clear PDF summarizing a cell's
key data: identity, SOH/RUL with reliability flags, second-life
recommendation (if available), and the assumption register.

Explicitly labeled a "Demonstration Report" — not a regulatory document.
Uses reportlab (platypus) for full layout control: colored section bars,
state-coded tables, and a disclaimer box up front.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)

NAVY    = colors.HexColor("#1a202c")
SLATE   = colors.HexColor("#2d3748")
MUTED   = colors.HexColor("#718096")
GREEN   = colors.HexColor("#2f855a")
GREEN_BG= colors.HexColor("#eafaf1")
AMBER   = colors.HexColor("#b7791f")
AMBER_BG= colors.HexColor("#fdf6e3")
GRAY_BG = colors.HexColor("#f0f1f3")
RED_BG  = colors.HexColor("#fdecec")
RED     = colors.HexColor("#c0392b")

STATE_COLOUR = {"available": GREEN, "estimated": AMBER, "unavailable": MUTED}


def _pdf_text(s: str) -> str:
    """Sanitize text for reportlab Paragraphs.

    Replaces Unicode subscript ₂ (U+2082) — not in Helvetica's encoding —
    with reportlab's <sub> XML markup so it renders as proper superscript.
    """
    return s.replace("₂", "<sub>2</sub>")
STATE_BG     = {"available": GREEN_BG, "estimated": AMBER_BG, "unavailable": GRAY_BG}
STATE_LABEL  = {"available": "AVAILABLE", "estimated": "ESTIMATE", "unavailable": "N/A IN DEMO"}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("TitleBig", parent=ss["Title"], fontSize=20, leading=24, textColor=NAVY))
    ss.add(ParagraphStyle("SubTitle", parent=ss["Normal"], fontSize=10, textColor=MUTED, spaceAfter=4))
    ss.add(ParagraphStyle("SectionHeader", parent=ss["Heading2"], fontSize=12, leading=15,
                           textColor=colors.white, backColor=NAVY,
                           spaceBefore=14, spaceAfter=6, leftIndent=6, borderPadding=6))
    ss.add(ParagraphStyle("Disclaimer", parent=ss["Normal"], fontSize=9, leading=13, textColor=SLATE))
    ss.add(ParagraphStyle("CellWrap", parent=ss["Normal"], fontSize=9, leading=12, textColor=SLATE))
    ss.add(ParagraphStyle("CellLabel", parent=ss["Normal"], fontSize=9, leading=12,
                           textColor=NAVY, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle("Note", parent=ss["Normal"], fontSize=7.5, leading=10, textColor=MUTED))
    ss.add(ParagraphStyle("Footer", parent=ss["Normal"], fontSize=7.5, textColor=MUTED))
    return ss


def _field_table(fields: list[dict], ss) -> Table:
    rows = [[
        Paragraph("Field", ss["CellLabel"]),
        Paragraph("Value", ss["CellLabel"]),
        Paragraph("Status", ss["CellLabel"]),
    ]]
    bg_cmds = []
    for i, f in enumerate(fields, start=1):
        note = f.get("note")
        value_html = _pdf_text(f["value"]) + (
            f"<br/><font size=7 color='#718096'>{_pdf_text(note)}</font>" if note else ""
        )
        rows.append([
            Paragraph(_pdf_text(f["label"]), ss["CellWrap"]),
            Paragraph(value_html, ss["CellWrap"]),
            Paragraph(
                STATE_LABEL[f["state"]],
                ParagraphStyle("StateCell", fontSize=8, leading=10, fontName="Helvetica-Bold",
                               textColor=STATE_COLOUR[f["state"]]),
            ),
        ])
        bg_cmds.append(("BACKGROUND", (2, i), (2, i), STATE_BG[f["state"]]))

    t = Table(rows, colWidths=[150, 230, 90])
    t.splitByRow = 0  # never split small field tables mid-row
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d4d9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
    ] + bg_cmds
    t.setStyle(TableStyle(style))
    return t


def build_report_pdf(
    passport: dict,
    second_life: dict | None,
    assumptions: dict,
) -> bytes:
    """
    passport: output of passport.build_passport()
    second_life: None if cell is still in primary life, else dict with
                 keys best_app (str), best_fit (str), financials (dict of
                 name -> dollar value) — pass the same shape used on the
                 Consequences page.
    assumptions: consequences.ASSUMPTIONS dict (for the register table)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Battery Passport — {passport['cell_id']}",
    )
    ss = _styles()
    story = []

    # ── Header ──
    story.append(Paragraph("⚡ Battery Intelligence Platform", ss["SubTitle"]))
    story.append(Paragraph(f"Demonstration Report — {passport['cell_id']}", ss["TitleBig"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"Battery Passport Interface, structured around EU Battery Regulation (EU) 2023/1542 data fields",
        ss["SubTitle"],
    ))
    story.append(Spacer(1, 6))

    disclaimer_table = Table([[Paragraph(
        "<b>Not a regulatory document.</b> This report demonstrates the EU Battery Regulation "
        "(2023/1542) data structure using a portfolio project's available pipeline outputs and "
        "literature-cited assumptions. It does not constitute manufacturer certification, a "
        "verified carbon audit, or a regulatory compliance declaration. Fields below are marked "
        "<b>AVAILABLE</b> (pipeline output), <b>ESTIMATE</b> (cited/illustrative assumption), or "
        "<b>N/A IN DEMO</b> (genuinely not present in this demonstration).",
        ss["Disclaimer"],
    )]], colWidths=[470])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 4))

    # ── Field groups ──
    # Wrap each section (header + table) in KeepTogether so a small table is
    # never split across pages. KeepTogether tries to place the whole block on
    # the current page; if it doesn't fit, it pushes everything to the next page.
    # For large tables that exceed a full page, platypus falls back to splitting
    # at row boundaries — this is the correct behaviour for genuinely long tables.
    group_titles = {
        "identity":  "1 · Battery Identity",
        "soh":       "2 · State of Health",
        "lifecycle": "3 · Lifecycle History",
        "carbon":    "4 · Carbon Footprint",
    }
    for key, title in group_titles.items():
        section = [
            Paragraph(title, ss["SectionHeader"]),
            _field_table(passport[key], ss),
            Spacer(1, 4),
        ]
        story.append(KeepTogether(section))

    # ── Second-life recommendation ──
    sl_section = [Paragraph("5 · Second-Life Recommendation", ss["SectionHeader"])]
    if second_life is None:
        sl_section.append(Paragraph(
            "Cell is still in primary life (SOH above the 85% second-life assessment threshold). "
            "No second-life recommendation is generated yet — return once SOH degrades further.",
            ss["CellWrap"],
        ))
    else:
        rows = [[Paragraph("Option", ss["CellLabel"]), Paragraph("Net value", ss["CellLabel"]), Paragraph("Status", ss["CellLabel"])]]
        bg_cmds = []
        for i, (name, val) in enumerate(second_life["financials"].items(), start=1):
            sign = "+" if val > 0 else "−"
            rows.append([
                Paragraph(name, ss["CellWrap"]),
                Paragraph(f"{sign}${abs(val):.2f}", ss["CellWrap"]),
                Paragraph("ESTIMATE", ParagraphStyle("e", fontSize=8, fontName="Helvetica-Bold", textColor=AMBER)),
            ])
            bg_cmds.append(("BACKGROUND", (2, i), (2, i), AMBER_BG))
        t = Table(rows, colWidths=[230, 150, 90])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), SLATE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d4d9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ] + bg_cmds))
        sl_section.append(t)
        sl_section.append(Spacer(1, 4))
        sl_section.append(Paragraph(
            f"Best-fit application: <b>{second_life['best_app']}</b> ({second_life['best_fit']}). "
            "All figures are cited estimates — see assumption register below.",
            ss["CellWrap"],
        ))
    sl_section.append(Spacer(1, 6))
    story.append(KeepTogether(sl_section))

    # ── Assumption register ──
    # This table can be long — wrap only the header + first few rows to avoid
    # orphaning the section header on its own page, then let the rest split naturally.
    assume_rows = [[Paragraph("Assumption", ss["CellLabel"]), Paragraph("Value", ss["CellLabel"]),
                    Paragraph("Label", ss["CellLabel"]), Paragraph("Source", ss["CellLabel"])]]
    bg_cmds = []
    for i, (key, a) in enumerate(assumptions.items(), start=1):
        is_cited = "Cited" in a["label"]
        assume_rows.append([
            Paragraph(key.replace("_", " ").title(), ss["CellWrap"]),
            Paragraph(f"{a['value']:.2f} {a['unit']}", ss["CellWrap"]),
            Paragraph(a["label"], ParagraphStyle("l", fontSize=8, fontName="Helvetica-Bold",
                                                  textColor=AMBER if is_cited else MUTED)),
            Paragraph(a["source"], ss["Note"]),
        ])
        bg_cmds.append(("BACKGROUND", (2, i), (2, i), AMBER_BG if is_cited else GRAY_BG))
    t = Table(assume_rows, colWidths=[95, 70, 75, 230],
              repeatRows=1)   # repeat header row when table splits across pages
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d4d9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ] + bg_cmds))
    # KeepTogether with just the header + table ensures the section title never
    # appears alone at the bottom of a page. Long tables still split at row boundaries.
    story.append(KeepTogether([
        Paragraph("Assumption Register", ss["SectionHeader"]),
        t,
    ]))

    # ── Footer ──
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#d0d4d9")))
    story.append(Spacer(1, 4))
    summ = passport["summary"]
    story.append(Paragraph(
        f"Field summary: {summ['n_available']} available · {summ['n_estimated']} estimated · "
        f"{summ['n_unavailable']} not available in this demonstration ({summ['n_total']} total fields). "
        "Battery Intelligence Platform — portfolio project. Not affiliated with or endorsed by any "
        "regulatory authority.",
        ss["Footer"],
    ))

    doc.build(story)
    return buf.getvalue()
