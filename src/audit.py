"""
Session-scoped audit log.

Tracks page visits, Copilot queries, and explicit decision events.
All records are stored in st.session_state["audit_log"] as a list of dicts.
Exports to CSV on request.

This is an in-memory log — it is cleared when the session ends.
A production deployment would persist to a database with user identity attached.
"""

from __future__ import annotations

import datetime
import csv
import io
import streamlit as st


_LOG_KEY = "audit_log"


def _log() -> list[dict]:
    if _LOG_KEY not in st.session_state:
        st.session_state[_LOG_KEY] = []
    return st.session_state[_LOG_KEY]


def log_page_view(page: str, cell: str | None = None) -> None:
    _log().append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "user":      st.session_state.get("auth_user", "anonymous"),
        "event":     "page_view",
        "detail":    page,
        "cell":      cell or "",
    })


def log_copilot_query(query_key: str, cell: str, mode: str = "template") -> None:
    _log().append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "user":      st.session_state.get("auth_user", "anonymous"),
        "event":     "copilot_query",
        "detail":    query_key,
        "cell":      cell,
        "mode":      mode,
    })


def log_decision(action: str, cell: str, notes: str = "") -> None:
    _log().append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "user":      st.session_state.get("auth_user", "anonymous"),
        "event":     "decision",
        "detail":    action,
        "cell":      cell,
        "notes":     notes,
    })


def export_csv() -> bytes:
    records = _log()
    if not records:
        return b""
    fields = ["timestamp", "user", "event", "detail", "cell", "notes", "mode"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode()


def get_log() -> list[dict]:
    return list(_log())
