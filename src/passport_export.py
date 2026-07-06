"""
Machine-readable EU Battery Passport export.

Wraps src/passport.py's build_passport() output into a JSON-LD document.
This is a reasonable-effort structured export — not a GS1 Digital Link /
CIRPASS-certified schema — consistent with build_passport()'s own
docstring: "a data-structure demonstration, not a regulatory submission."

The visual Passport page and the PDF report both consume build_passport()
directly; this module only adds a third, machine-readable consumer of the
same dict, so all three surfaces stay in sync automatically.
"""

import datetime
import hashlib


def document_id(passport: dict, cell_id: str, generated_at: str) -> str:
    """
    A short, deterministic identifier for one export instance — derived from
    the cell ID, generation timestamp, and field-summary counts, so a PDF
    and its JSON-LD counterpart generated in the same export action carry
    the same ID (traceable pair), and any two exports of the same cell at
    different times are distinguishable. This does not assert any
    regulatory certification — it's an audit-trail convenience (which
    export instance is this, was it regenerated since) for whoever files
    the PDF, not a claim about the data itself.
    """
    summ = passport["summary"]
    raw = (
        f"{cell_id}|{generated_at}|{summ['n_available']}|"
        f"{summ['n_estimated']}|{summ['n_unavailable']}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:12].upper()


def _field_entries(fields: list[dict]) -> list[dict]:
    """Map build_passport()'s {"label","value","state","note"} dicts to
    JSON-LD-flavored {"name","value","provenance","note"} entries."""
    out = []
    for f in fields:
        entry = {
            "name": f["label"],
            "value": f["value"],
            "provenance": f["state"],  # "available" | "estimated" | "unavailable"
        }
        if f.get("note"):
            entry["note"] = f["note"]
        out.append(entry)
    return out


def to_json_ld(passport: dict, cell_id: str, doc_id: "str | None" = None) -> dict:
    """
    Convert a build_passport() dict into a JSON-LD document.

    passport: the dict returned by src/passport.py's build_passport()
    cell_id:  the cell identifier (also present as passport["cell_id"])
    doc_id:   pass the same document_id() used for a paired PDF export so
              both artifacts from one export action share an ID; computed
              fresh if omitted (e.g. a standalone JSON-LD-only download).
    """
    summary = passport["summary"]
    generated_at = datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "eubr": "https://eur-lex.europa.eu/eli/reg/2023/1542/",
        },
        "@type": "Product",
        "@id": f"urn:battery-passport:{cell_id}",
        "identifier": cell_id,
        "documentId": doc_id or document_id(passport, cell_id, generated_at),
        "eubr:regulation": "EU 2023/1542",
        "dateExported": generated_at,
        "disclaimer": (
            "Data-structure demonstration only — not a regulatory compliance "
            "submission. Field provenance (available/estimated/unavailable) "
            "indicates data source honesty, not regulatory certification. "
            "documentId is an audit-trail convenience (traces this specific "
            "export instance), not a certification mark."
        ),
        "identity": _field_entries(passport["identity"]),
        "stateOfHealth": _field_entries(passport["soh"]),
        "lifecycle": _field_entries(passport["lifecycle"]),
        "carbonFootprint": _field_entries(passport["carbon"]),
        "completeness": {
            "available": summary["n_available"],
            "estimated": summary["n_estimated"],
            "unavailable": summary["n_unavailable"],
            "total": summary["n_total"],
        },
    }
