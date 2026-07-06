"""
SQLite persistence layer (via SQLAlchemy, for later low-effort Postgres
portability — swap DB_URL and most of this module is unchanged).

Real multi-tenancy: every row in every table belongs to an Organization.
A User authenticates against a username/bcrypt password hash and belongs
to exactly one Organization; every read/write function below takes an
org_id and scopes to it, so two organizations' decisions, cohort tags,
settings, uploads, and failure signatures never mix. A "Demo Org" (id 1,
slug "demo-org") is seeded with the app's original 4 demo accounts so the
public demo link keeps working with the same documented credentials.
"""

import datetime
import json
import os
import pathlib

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import (
    create_engine, inspect, text,
    Column, Integer, String, Float, Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = pathlib.Path(__file__).parent.parent / "data" / "app.db"
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()

DEMO_ORG_SLUG = "demo-org"
_DEMO_ORG_ID = 1

# username → (password, role, display_name), seeded into the Demo Org
DEMO_USERS = {
    "engineer":   ("battery",   "engineer",   "Battery Engineer"),
    "fleet":      ("ops2024",   "fleet",      "Fleet Operations"),
    "compliance": ("eu2024",    "compliance", "Compliance Officer"),
    "admin":      ("admin",     "admin",      "Administrator"),
}

# Setting keys whose values are genuine credentials (API keys/tokens/secrets,
# not IDs or URLs) and are therefore envelope-encrypted at rest in the
# `settings` table — see _get_fernet()/get_setting()/set_setting() below.
_SECRET_SETTING_KEYS = frozenset({
    "vrm_api_token", "circunomics_api_key", "cmms_api_key",
    "webhook_secret", "orion_bms_api_key",
})

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    """Lazily build the Fernet cipher used to encrypt credential-shaped
    Setting values. Same honest demo-grade-secret pattern as JWT_SECRET in
    src/api.py: the fallback key below is fine for local/demo use but must
    be overridden via a real secrets manager before any production deploy —
    anyone with read access to this source can decrypt data encrypted with
    the fallback key."""
    global _fernet
    if _fernet is None:
        _key = os.environ.get(
            "SETTINGS_ENCRYPTION_KEY",
            "4S2b-Ok94fVLdI9xbEQVoIr2Aw3s7Tqo2YAcc1zYUaw=",  # dev-only, insecure
        )
        _fernet = Fernet(_key.encode() if isinstance(_key, str) else _key)
    return _fernet


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Organization(Base):
    __tablename__ = "organizations"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String, nullable=False)
    slug       = Column(String, unique=True, nullable=False)
    created_at = Column(String)


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    org_id        = Column(Integer, nullable=False)
    username      = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role          = Column(String, nullable=False)
    display_name  = Column(String)
    created_at    = Column(String)


class Decision(Base):
    __tablename__ = "decisions"
    id           = Column(String, primary_key=True)
    org_id       = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    cell_id      = Column(String, nullable=False)
    action       = Column(String, nullable=False)
    confidence   = Column(String)
    soh_pct      = Column(Float)
    timestamp    = Column(String)
    status       = Column(String, default="Pending")
    outcome_soh  = Column(Float, nullable=True)


class CellCohortTag(Base):
    __tablename__ = "cell_cohort_tags"
    org_id  = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    cell_id = Column(String, primary_key=True)
    tag     = Column(String, nullable=False)


class Setting(Base):
    __tablename__ = "settings"
    org_id = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    key    = Column(String, primary_key=True)
    value  = Column(Text)  # JSON-encoded


class UploadMeta(Base):
    __tablename__ = "upload_meta"
    id                    = Column(Integer, primary_key=True, autoincrement=True)
    org_id                = Column(Integer, default=_DEMO_ORG_ID)
    upload_date           = Column(String)
    n_cells               = Column(Integer)
    cell_ids              = Column(Text)  # JSON-encoded list[str]
    joblib_key            = Column(String)


class FailureSignature(Base):
    __tablename__ = "failure_signatures"
    org_id              = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    cell_id             = Column(String, primary_key=True)
    source              = Column(String)
    eol_cycle           = Column(Integer)
    soh_at_window_start = Column(Float)
    failure_mode        = Column(String)
    feature_names       = Column(Text)  # JSON-encoded list[str]
    trend_vector        = Column(Text)  # JSON-encoded list[float]


# ---------------------------------------------------------------------------
# Init + migration + seeding
# ---------------------------------------------------------------------------

def _ensure_org_id_column(table_name: str) -> None:
    """
    Additive migration for pre-existing local DBs: Base.metadata.create_all()
    only creates tables that don't exist yet, so a table created before this
    module gained multi-tenancy won't get its new org_id column automatically.
    Adding it here (constant DEFAULT, which SQLite's ADD COLUMN supports)
    preserves existing rows by attaching them to the Demo Org rather than
    losing them.

    Note: on a table migrated this way, SQLite keeps the original CREATE
    TABLE's single-column primary key — it will not retroactively enforce
    the new (org_id, key)-style composite uniqueness declared on the ORM
    model. Harmless for this single-file demo deployment; a fresh DB (or
    a real Postgres migration later) gets the composite PK from the start.
    """
    insp = inspect(engine)
    if table_name not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns(table_name)}
    if "org_id" not in cols:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN org_id INTEGER DEFAULT {_DEMO_ORG_ID}"))


def _seed_demo_org_and_users() -> None:
    """Idempotent: creates the Demo Org + its 4 demo accounts on first run only."""
    with Session() as s:
        if s.query(Organization).filter_by(slug=DEMO_ORG_SLUG).one_or_none() is not None:
            return
        s.add(Organization(
            id=_DEMO_ORG_ID, name="Demo Org", slug=DEMO_ORG_SLUG,
            created_at=datetime.datetime.now().isoformat(),
        ))
        s.commit()
        for username, (password, role, display_name) in DEMO_USERS.items():
            s.add(User(
                org_id=_DEMO_ORG_ID, username=username,
                password_hash=hash_password(password),
                role=role, display_name=display_name,
                created_at=datetime.datetime.now().isoformat(),
            ))
        s.commit()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    for _t in ("decisions", "cell_cohort_tags", "settings", "upload_meta", "failure_signatures"):
        _ensure_org_id_column(_t)
    _seed_demo_org_and_users()


# ---------------------------------------------------------------------------
# Organizations + users (auth)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False


def _slugify(name: str) -> str:
    slug = "-".join(name.strip().lower().split())
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    return slug or "org"


def create_organization_with_admin(org_name: str, username: str, password: str, display_name: str = "") -> dict:
    """
    Self-service signup: creates a brand-new Organization plus its first
    (admin-role) User. Returns {"org_id", "org_name", "user_id"} on success,
    or {"error": "..."} if the username or org slug is already taken.
    """
    with Session() as s:
        if s.query(User).filter_by(username=username.strip().lower()).one_or_none() is not None:
            return {"error": "That username is already taken."}
        base_slug = _slugify(org_name)
        slug = base_slug
        i = 2
        while s.query(Organization).filter_by(slug=slug).one_or_none() is not None:
            slug = f"{base_slug}-{i}"
            i += 1
        org = Organization(name=org_name.strip(), slug=slug, created_at=datetime.datetime.now().isoformat())
        s.add(org)
        s.commit()
        s.refresh(org)
        user = User(
            org_id=org.id, username=username.strip().lower(),
            password_hash=hash_password(password), role="admin",
            display_name=display_name.strip() or username.strip(),
            created_at=datetime.datetime.now().isoformat(),
        )
        s.add(user)
        s.commit()
        s.refresh(user)
        return {"org_id": org.id, "org_name": org.name, "user_id": user.id}


def create_user(org_id: int, username: str, password: str, role: str, display_name: str = "") -> dict:
    """Admin-invites-teammate: adds another user to an existing org."""
    with Session() as s:
        if s.query(User).filter_by(username=username.strip().lower()).one_or_none() is not None:
            return {"error": "That username is already taken."}
        user = User(
            org_id=org_id, username=username.strip().lower(),
            password_hash=hash_password(password), role=role,
            display_name=display_name.strip() or username.strip(),
            created_at=datetime.datetime.now().isoformat(),
        )
        s.add(user)
        s.commit()
        s.refresh(user)
        return {"user_id": user.id}


def get_user_by_username(username: str) -> "dict | None":
    with Session() as s:
        row = s.query(User).filter_by(username=username.strip().lower()).one_or_none()
        if row is None:
            return None
        org = s.query(Organization).filter_by(id=row.org_id).one_or_none()
        return {
            "user_id": row.id, "org_id": row.org_id,
            "org_name": org.name if org else "",
            "username": row.username, "password_hash": row.password_hash,
            "role": row.role, "display_name": row.display_name,
        }


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def save_decision(org_id: int, entry: dict) -> None:
    """Insert a new decision log row. entry keys match the Decision columns."""
    with Session() as s:
        s.merge(Decision(
            id=entry["id"],
            org_id=org_id,
            cell_id=entry.get("cell_id"),
            action=entry.get("action"),
            confidence=entry.get("confidence"),
            soh_pct=entry.get("soh_pct"),
            timestamp=entry.get("timestamp"),
            status=entry.get("status", "Pending"),
            outcome_soh=entry.get("outcome_soh"),
        ))
        s.commit()


def load_decisions(org_id: int) -> list[dict]:
    with Session() as s:
        rows = s.query(Decision).filter_by(org_id=org_id).order_by(Decision.timestamp.desc()).all()
        return [
            {
                "id": r.id, "cell_id": r.cell_id, "action": r.action,
                "confidence": r.confidence, "soh_pct": r.soh_pct,
                "timestamp": r.timestamp, "status": r.status,
                "outcome_soh": r.outcome_soh,
            }
            for r in rows
        ]


def update_decision(org_id: int, decision_id: str, **fields) -> None:
    with Session() as s:
        row = s.query(Decision).filter_by(org_id=org_id, id=decision_id).one_or_none()
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        s.commit()


# ---------------------------------------------------------------------------
# Cohort tags
# ---------------------------------------------------------------------------

def save_cohort_tag(org_id: int, cell_id: str, tag: str) -> None:
    with Session() as s:
        s.merge(CellCohortTag(org_id=org_id, cell_id=cell_id, tag=tag))
        s.commit()


def load_cohort_tags(org_id: int) -> dict:
    with Session() as s:
        rows = s.query(CellCohortTag).filter_by(org_id=org_id).all()
        return {r.cell_id: r.tag for r in rows}


# ---------------------------------------------------------------------------
# Settings (generic key-value, scoped per org)
# ---------------------------------------------------------------------------

def get_setting(org_id: int, key: str, default=None):
    with Session() as s:
        row = s.query(Setting).filter_by(org_id=org_id, key=key).one_or_none()
        if row is None:
            return default
        raw = row.value
        if key in _SECRET_SETTING_KEYS and raw is not None:
            try:
                raw = _get_fernet().decrypt(raw.encode()).decode()
            except InvalidToken:
                # Row predates encryption being added (plaintext JSON) — read
                # it as-is; the next set_setting() call re-encrypts it.
                pass
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default


def set_setting(org_id: int, key: str, value) -> None:
    encoded = json.dumps(value)
    if key in _SECRET_SETTING_KEYS:
        encoded = _get_fernet().encrypt(encoded.encode()).decode()
    with Session() as s:
        s.merge(Setting(org_id=org_id, key=key, value=encoded))
        s.commit()


# ---------------------------------------------------------------------------
# Upload metadata
# ---------------------------------------------------------------------------

def save_upload_meta(org_id: int, meta: dict, joblib_key: str) -> None:
    with Session() as s:
        s.add(UploadMeta(
            org_id=org_id,
            upload_date=meta.get("upload_date", datetime.date.today().isoformat()),
            n_cells=meta.get("n_cells"),
            cell_ids=json.dumps(meta.get("cell_ids", [])),
            joblib_key=joblib_key,
        ))
        s.commit()


def load_upload_meta_history(org_id: int) -> list[dict]:
    with Session() as s:
        rows = s.query(UploadMeta).filter_by(org_id=org_id).order_by(UploadMeta.id.desc()).all()
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

def save_failure_signatures(org_id: int, signatures: list) -> None:
    """signatures: list of trajectory_memory.FailureSignature dataclass instances."""
    with Session() as s:
        for sig in signatures:
            s.merge(FailureSignature(
                org_id=org_id,
                cell_id=sig.cell_id,
                source=sig.source,
                eol_cycle=sig.eol_cycle,
                soh_at_window_start=sig.soh_at_window_start,
                failure_mode=sig.failure_mode,
                feature_names=json.dumps(list(sig.feature_names)),
                trend_vector=json.dumps(list(float(v) for v in sig.trend_vector)),
            ))
        s.commit()


def load_failure_signatures(org_id: int) -> list:
    """Returns list of trajectory_memory.FailureSignature instances."""
    import numpy as np
    from trajectory_memory import FailureSignature as _FS

    with Session() as s:
        rows = s.query(FailureSignature).filter_by(org_id=org_id).all()
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
