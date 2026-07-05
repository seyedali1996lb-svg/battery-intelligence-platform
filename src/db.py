"""
SQLite persistence layer (via SQLAlchemy, for later low-effort Postgres
portability — swap DB_URL and most of this module is unchanged).

This is shared fleet-team state, not per-individual-user data: the app's
4 logins are shared demo-role accounts (engineer/fleet/compliance/admin),
not per-user accounts, so there is no user_id scoping here — matches how
decision_log/cohort tags already behaved via session_state before this
module existed.
"""

import datetime
import json
import pathlib

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = pathlib.Path(__file__).parent.parent / "data" / "app.db"
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Decision(Base):
    __tablename__ = "decisions"
    id           = Column(String, primary_key=True)
    cell_id      = Column(String, nullable=False)
    action       = Column(String, nullable=False)
    confidence   = Column(String)
    soh_pct      = Column(Float)
    timestamp    = Column(String)
    status       = Column(String, default="Pending")
    outcome_soh  = Column(Float, nullable=True)


class CellCohortTag(Base):
    __tablename__ = "cell_cohort_tags"
    cell_id = Column(String, primary_key=True)
    tag     = Column(String, nullable=False)


class Setting(Base):
    __tablename__ = "settings"
    key   = Column(String, primary_key=True)
    value = Column(Text)  # JSON-encoded


class UploadMeta(Base):
    __tablename__ = "upload_meta"
    id                    = Column(Integer, primary_key=True, autoincrement=True)
    upload_date           = Column(String)
    n_cells               = Column(Integer)
    cell_ids              = Column(Text)  # JSON-encoded list[str]
    joblib_key            = Column(String)


class FailureSignature(Base):
    __tablename__ = "failure_signatures"
    cell_id             = Column(String, primary_key=True)
    source              = Column(String)
    eol_cycle           = Column(Integer)
    soh_at_window_start = Column(Float)
    failure_mode        = Column(String)
    feature_names       = Column(Text)  # JSON-encoded list[str]
    trend_vector        = Column(Text)  # JSON-encoded list[float]


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def save_decision(entry: dict) -> None:
    """Insert a new decision log row. entry keys match the Decision columns."""
    with Session() as s:
        s.merge(Decision(
            id=entry["id"],
            cell_id=entry.get("cell_id"),
            action=entry.get("action"),
            confidence=entry.get("confidence"),
            soh_pct=entry.get("soh_pct"),
            timestamp=entry.get("timestamp"),
            status=entry.get("status", "Pending"),
            outcome_soh=entry.get("outcome_soh"),
        ))
        s.commit()


def load_decisions() -> list[dict]:
    with Session() as s:
        rows = s.query(Decision).order_by(Decision.timestamp.desc()).all()
        return [
            {
                "id": r.id, "cell_id": r.cell_id, "action": r.action,
                "confidence": r.confidence, "soh_pct": r.soh_pct,
                "timestamp": r.timestamp, "status": r.status,
                "outcome_soh": r.outcome_soh,
            }
            for r in rows
        ]


def update_decision(decision_id: str, **fields) -> None:
    with Session() as s:
        row = s.query(Decision).filter(Decision.id == decision_id).one_or_none()
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        s.commit()


# ---------------------------------------------------------------------------
# Cohort tags
# ---------------------------------------------------------------------------

def save_cohort_tag(cell_id: str, tag: str) -> None:
    with Session() as s:
        s.merge(CellCohortTag(cell_id=cell_id, tag=tag))
        s.commit()


def load_cohort_tags() -> dict:
    with Session() as s:
        rows = s.query(CellCohortTag).all()
        return {r.cell_id: r.tag for r in rows}


# ---------------------------------------------------------------------------
# Settings (generic key-value)
# ---------------------------------------------------------------------------

def get_setting(key: str, default=None):
    with Session() as s:
        row = s.query(Setting).filter(Setting.key == key).one_or_none()
        if row is None:
            return default
        try:
            return json.loads(row.value)
        except (TypeError, ValueError):
            return default


def set_setting(key: str, value) -> None:
    with Session() as s:
        s.merge(Setting(key=key, value=json.dumps(value)))
        s.commit()


# ---------------------------------------------------------------------------
# Upload metadata
# ---------------------------------------------------------------------------

def save_upload_meta(meta: dict, joblib_key: str) -> None:
    with Session() as s:
        s.add(UploadMeta(
            upload_date=meta.get("upload_date", datetime.date.today().isoformat()),
            n_cells=meta.get("n_cells"),
            cell_ids=json.dumps(meta.get("cell_ids", [])),
            joblib_key=joblib_key,
        ))
        s.commit()


def load_upload_meta_history() -> list[dict]:
    with Session() as s:
        rows = s.query(UploadMeta).order_by(UploadMeta.id.desc()).all()
        return [
            {
                "id": r.id, "upload_date": r.upload_date, "n_cells": r.n_cells,
                "cell_ids": json.loads(r.cell_ids) if r.cell_ids else [],
                "joblib_key": r.joblib_key,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Failure signatures (trajectory memory)
# ---------------------------------------------------------------------------

def save_failure_signatures(signatures: list) -> None:
    """signatures: list of trajectory_memory.FailureSignature dataclass instances."""
    with Session() as s:
        for sig in signatures:
            s.merge(FailureSignature(
                cell_id=sig.cell_id,
                source=sig.source,
                eol_cycle=sig.eol_cycle,
                soh_at_window_start=sig.soh_at_window_start,
                failure_mode=sig.failure_mode,
                feature_names=json.dumps(list(sig.feature_names)),
                trend_vector=json.dumps(list(float(v) for v in sig.trend_vector)),
            ))
        s.commit()


def load_failure_signatures() -> list:
    """Returns list of trajectory_memory.FailureSignature instances."""
    import numpy as np
    from trajectory_memory import FailureSignature as _FS

    with Session() as s:
        rows = s.query(FailureSignature).all()
        return [
            _FS(
                cell_id=r.cell_id,
                source=r.source,
                eol_cycle=r.eol_cycle,
                soh_at_window_start=r.soh_at_window_start,
                failure_mode=r.failure_mode,
                feature_names=json.loads(r.feature_names),
                trend_vector=np.array(json.loads(r.trend_vector), dtype=float),
            )
            for r in rows
        ]
