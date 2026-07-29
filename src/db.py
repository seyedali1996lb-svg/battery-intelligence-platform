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
import logging
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
_FALLBACK_ENCRYPTION_KEY = "4S2b-Ok94fVLdI9xbEQVoIr2Aw3s7Tqo2YAcc1zYUaw="  # dev-only, insecure
_logger = logging.getLogger(__name__)


def using_fallback_encryption_key() -> bool:
    """True if SETTINGS_ENCRYPTION_KEY isn't set and stored credentials are
    being encrypted with the fallback key that's public in this repo's
    source -- anyone with read access to the source can decrypt them. Used
    by _get_fernet() to log a warning and by the Settings page to show an
    admin-visible banner, so this doesn't stay silent on a real deployment
    the way src/api.py's equivalent JWT_SECRET gap did before it was fixed
    to hard-fail. Unlike that fix, this one can't hard-fail here: this
    module backs the live Streamlit Cloud deployment, and forcing the key
    would lock the deployer out rather than just refuse to start a
    not-yet-deployed API layer -- confirmed with the user that
    SETTINGS_ENCRYPTION_KEY is not currently set there before choosing warn
    over fail."""
    return "SETTINGS_ENCRYPTION_KEY" not in os.environ


def _get_fernet() -> Fernet:
    """Lazily build the Fernet cipher used to encrypt credential-shaped
    Setting values. Same honest demo-grade-secret pattern as JWT_SECRET in
    src/api.py: the fallback key below is fine for local/demo use but must
    be overridden via a real secrets manager before any production deploy —
    anyone with read access to this source can decrypt data encrypted with
    the fallback key."""
    global _fernet
    if _fernet is None:
        if using_fallback_encryption_key():
            _logger.warning(
                "SETTINGS_ENCRYPTION_KEY is not set -- stored credentials "
                "(VRM/Orion/Circunomics/CMMS API keys, webhook secret) are "
                "being encrypted with a fallback key that is public in this "
                "repo's source. Set SETTINGS_ENCRYPTION_KEY via a real "
                "secrets manager before storing real credentials."
            )
        _key = os.environ.get("SETTINGS_ENCRYPTION_KEY", _FALLBACK_ENCRYPTION_KEY)
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
    # Login lockout (Enterprise Readiness audit finding: login.py had no
    # rate-limiting at all -- unlimited password guesses against any
    # username). See is_login_locked_out()/record_failed_login()/
    # reset_login_attempts() below.
    failed_login_attempts = Column(Integer, default=0)
    locked_until           = Column(String, nullable=True)  # ISO datetime, or None


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


class ExperimentRun(Base):
    """One logged GBRT training run — see src/experiment_registry.py for the
    dataclass/orchestration layer built on top of this table (same split as
    FailureSignature above / trajectory_memory.py: this class owns storage
    only). org_id uses the reserved PLATFORM_ORG_ID (0) sentinel — see
    experiment_registry.py's module docstring — for runs trained on the
    platform's own shared reference fleets (NASA/synthetic/Severson), which
    belong to no single tenant; real org_id values are used for runs trained
    on an org's own uploaded data."""
    __tablename__ = "experiment_runs"
    run_id          = Column(String, primary_key=True)
    org_id          = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    dataset         = Column(String, nullable=False)   # "nasa" | "synth" | "severson" | "uploaded" | "nasa_to_severson" ...
    chemistry       = Column(String)                   # e.g. "LiCoO2", "LFP", "NCA"
    feature_set     = Column(Text)                      # JSON-encoded list[str] of columns actually used
    feature_version = Column(String)                    # batlab.features.engineering.FEATURE_VERSION at log time
    hyperparams     = Column(Text)                      # JSON-encoded dict (GBRT_PARAMS)
    seed            = Column(Integer)
    cell_ids        = Column(Text)                      # JSON-encoded list[str] — the LCO population
    n_cells         = Column(Integer)
    n_rows          = Column(Integer)
    soh_mae         = Column(Float)
    soh_r2          = Column(Float)
    rul_mae         = Column(Float)
    rul_r2          = Column(Float)
    rul_reliable    = Column(Integer)                   # 0/1 — SQLite has no native bool
    fold_metrics    = Column(Text)                       # JSON-encoded per-cell LCO breakdown
    git_commit      = Column(String)
    timestamp       = Column(String)
    notes           = Column(Text)


class KGNode(Base):
    """One knowledge-graph node — src/knowledge_graph.py owns the domain
    schema/logic (node types, provenance rules); this table only stores the
    flattened rows produced by its graph_to_rows(). Same split-ownership
    pattern as ExperimentRun above. org_id uses PLATFORM_ORG_ID (imported
    at call time from experiment_registry.py, not redefined here) for the
    graph built from the shared reference fleets."""
    __tablename__ = "kg_nodes"
    org_id    = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    node_type = Column(String, primary_key=True)
    node_id   = Column(String, primary_key=True)
    attrs     = Column(Text)  # JSON-encoded dict


class KGEdge(Base):
    """One knowledge-graph edge. edge_id is a deterministic
    'src_key|dst_key|edge_type' string (see knowledge_graph.graph_to_rows())
    so re-saving an unchanged edge is a no-op merge, not a duplicate row."""
    __tablename__ = "kg_edges"
    org_id    = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    edge_id   = Column(String, primary_key=True)
    edge_type = Column(String, nullable=False)
    src_key   = Column(String, nullable=False)
    dst_key   = Column(String, nullable=False)
    source_fn = Column(String)   # provenance: "module.function" that computed this edge
    doi       = Column(String)   # provenance: literature DOI, when applicable
    attrs     = Column(Text)     # JSON-encoded dict of extra edge attributes


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


def _ensure_login_lockout_columns() -> None:
    """Additive migration for pre-existing local DBs, same pattern as
    _ensure_org_id_column() above -- a users table created before this
    module gained login lockout won't get the new columns automatically
    from Base.metadata.create_all() alone."""
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("users")}
    with engine.begin() as conn:
        if "failed_login_attempts" not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0"))
        if "locked_until" not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN locked_until TEXT"))


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
    for _t in ("decisions", "cell_cohort_tags", "settings", "upload_meta",
               "failure_signatures", "experiment_runs", "kg_nodes", "kg_edges"):
        _ensure_org_id_column(_t)
    _ensure_login_lockout_columns()
    _seed_demo_org_and_users()
    _seed_default_fleet_hierarchy()


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


_MAX_FAILED_LOGIN_ATTEMPTS = 5
_LOCKOUT_MINUTES = 15


def is_login_locked_out(username: str) -> "str | None":
    """Returns the ISO-format lockout-expiry timestamp if this username is
    currently locked out (still in the future), else None. login.py checks
    this before calling verify_password(), so a locked-out account never
    even attempts a bcrypt check."""
    with Session() as s:
        row = s.query(User).filter_by(username=username.strip().lower()).one_or_none()
        if row is None or not row.locked_until:
            return None
        if datetime.datetime.fromisoformat(row.locked_until) > datetime.datetime.now():
            return row.locked_until
        return None


def record_failed_login(username: str) -> None:
    """Increments the failed-attempt counter for this username; locks the
    account for _LOCKOUT_MINUTES once _MAX_FAILED_LOGIN_ATTEMPTS is reached.
    No-op for a username that doesn't exist -- a distinguishable response
    for "wrong password, real user" vs "no such user" would leak which
    usernames are registered."""
    with Session() as s:
        row = s.query(User).filter_by(username=username.strip().lower()).one_or_none()
        if row is None:
            return
        row.failed_login_attempts = (row.failed_login_attempts or 0) + 1
        if row.failed_login_attempts >= _MAX_FAILED_LOGIN_ATTEMPTS:
            row.locked_until = (
                datetime.datetime.now() + datetime.timedelta(minutes=_LOCKOUT_MINUTES)
            ).isoformat()
        s.commit()


def reset_login_attempts(username: str) -> None:
    """Called on successful login -- clears the failed-attempt counter and
    any lockout."""
    with Session() as s:
        row = s.query(User).filter_by(username=username.strip().lower()).one_or_none()
        if row is None:
            return
        row.failed_login_attempts = 0
        row.locked_until = None
        s.commit()


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
        _ensure_default_site_and_fleet(s, org.id)
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


def get_settings(org_id: int, keys: "list[str] | None" = None) -> dict:
    """
    Batched version of get_setting() -- fetches every setting row for
    org_id (optionally filtered to `keys`) in a single query, instead of
    one round-trip per key. Used where a caller needs several settings at
    once (e.g. app/main.py's session hydration on login, which previously
    made 6 separate get_setting() calls).

    Unlike get_setting(), missing/undecodable keys are simply absent from
    the returned dict rather than filled with a per-call default -- callers
    should do `settings.get(key)` (optionally `is not None` to match
    get_setting()'s exact semantics for a stored JSON null) rather than
    assume every requested key is present.
    """
    with Session() as s:
        q = s.query(Setting).filter_by(org_id=org_id)
        if keys is not None:
            q = q.filter(Setting.key.in_(keys))
        rows = q.all()

    result = {}
    for row in rows:
        raw = row.value
        if row.key in _SECRET_SETTING_KEYS and raw is not None:
            try:
                raw = _get_fernet().decrypt(raw.encode()).decode()
            except InvalidToken:
                # Row predates encryption being added (plaintext JSON) — read
                # it as-is; the next set_setting() call re-encrypts it.
                pass
        try:
            result[row.key] = json.loads(raw)
        except (TypeError, ValueError):
            continue
    return result


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


# ---------------------------------------------------------------------------
# Experiment registry (src/experiment_registry.py owns the dataclass/logic)
# ---------------------------------------------------------------------------

def save_experiment_run(org_id: int, entry: dict) -> None:
    """Insert one logged run. entry keys match the ExperimentRun columns
    (feature_set/hyperparams/cell_ids/fold_metrics already JSON-encoded
    strings; rul_reliable already an int 0/1) — see
    experiment_registry.log_run() for the caller that builds this dict."""
    with Session() as s:
        s.merge(ExperimentRun(
            run_id=entry["run_id"],
            org_id=org_id,
            dataset=entry.get("dataset"),
            chemistry=entry.get("chemistry"),
            feature_set=entry.get("feature_set"),
            feature_version=entry.get("feature_version"),
            hyperparams=entry.get("hyperparams"),
            seed=entry.get("seed"),
            cell_ids=entry.get("cell_ids"),
            n_cells=entry.get("n_cells"),
            n_rows=entry.get("n_rows"),
            soh_mae=entry.get("soh_mae"),
            soh_r2=entry.get("soh_r2"),
            rul_mae=entry.get("rul_mae"),
            rul_r2=entry.get("rul_r2"),
            rul_reliable=entry.get("rul_reliable"),
            fold_metrics=entry.get("fold_metrics"),
            git_commit=entry.get("git_commit"),
            timestamp=entry.get("timestamp"),
            notes=entry.get("notes"),
        ))
        s.commit()


def _experiment_run_row_to_dict(r: "ExperimentRun") -> dict:
    return {
        "run_id":          r.run_id,
        "org_id":          r.org_id,
        "dataset":         r.dataset,
        "chemistry":       r.chemistry,
        "feature_set":     json.loads(r.feature_set) if r.feature_set else [],
        "feature_version": r.feature_version,
        "hyperparams":     json.loads(r.hyperparams) if r.hyperparams else {},
        "seed":            r.seed,
        "cell_ids":        json.loads(r.cell_ids) if r.cell_ids else [],
        "n_cells":         r.n_cells,
        "n_rows":          r.n_rows,
        "soh_mae":         r.soh_mae,
        "soh_r2":          r.soh_r2,
        "rul_mae":         r.rul_mae,
        "rul_r2":          r.rul_r2,
        "rul_reliable":    bool(r.rul_reliable),
        "fold_metrics":    json.loads(r.fold_metrics) if r.fold_metrics else {},
        "git_commit":      r.git_commit,
        "timestamp":       r.timestamp,
        "notes":           r.notes,
    }


def load_experiment_runs(org_ids: "int | list[int]") -> list[dict]:
    """Load logged runs for one org_id, or every org_id in a list (used by
    the leaderboard to combine an org's own uploaded-data runs with the
    shared PLATFORM_ORG_ID reference-dataset runs — see
    experiment_registry.py). Newest first."""
    ids = [org_ids] if isinstance(org_ids, int) else list(org_ids)
    with Session() as s:
        rows = (
            s.query(ExperimentRun)
            .filter(ExperimentRun.org_id.in_(ids))
            .order_by(ExperimentRun.timestamp.desc())
            .all()
        )
        return [_experiment_run_row_to_dict(r) for r in rows]


def get_experiment_run(org_id: int, run_id: str) -> "dict | None":
    with Session() as s:
        row = s.query(ExperimentRun).filter_by(org_id=org_id, run_id=run_id).one_or_none()
        return _experiment_run_row_to_dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Knowledge graph (src/knowledge_graph.py owns the schema/domain logic)
# ---------------------------------------------------------------------------

def save_knowledge_graph(org_id: int, node_rows: list, edge_rows: list) -> None:
    """Replace org_id's saved graph snapshot with node_rows/edge_rows (the
    flattened output of knowledge_graph.graph_to_rows()) — a full
    delete-then-insert, not an incremental merge, since a graph rebuild is
    expected to change which edges exist (e.g. a cell's mechanism verdict
    updates), not just add new ones."""
    with Session() as s:
        s.query(KGNode).filter_by(org_id=org_id).delete()
        s.query(KGEdge).filter_by(org_id=org_id).delete()
        for r in node_rows:
            s.add(KGNode(org_id=org_id, node_type=r["node_type"], node_id=r["node_id"], attrs=r["attrs"]))
        for r in edge_rows:
            s.add(KGEdge(
                org_id=org_id, edge_id=r["edge_id"], edge_type=r["edge_type"],
                src_key=r["src_key"], dst_key=r["dst_key"],
                source_fn=r["source_fn"], doi=r["doi"], attrs=r["attrs"],
            ))
        s.commit()


def load_knowledge_graph_rows(org_id: int) -> "tuple[list[dict], list[dict]]":
    """Load org_id's saved graph snapshot as (node_rows, edge_rows) —
    the same shape knowledge_graph.graph_to_rows() produces, ready for
    knowledge_graph.graph_from_rows()."""
    with Session() as s:
        node_rows = [
            {"node_type": r.node_type, "node_id": r.node_id, "attrs": r.attrs}
            for r in s.query(KGNode).filter_by(org_id=org_id).all()
        ]
        edge_rows = [
            {
                "edge_id": r.edge_id, "edge_type": r.edge_type,
                "src_key": r.src_key, "dst_key": r.dst_key,
                "source_fn": r.source_fn, "doi": r.doi, "attrs": r.attrs,
            }
            for r in s.query(KGEdge).filter_by(org_id=org_id).all()
        ]
        return node_rows, edge_rows


# ---------------------------------------------------------------------------
# FleetAsset hierarchy: Organization -> Site -> Fleet -> Pack -> Cell
#
# A persisted, org-scoped asset hierarchy for grouping cells by physical
# location/deployment -- distinct from src/pack_builder.py's Virtual Pack
# Builder (an ephemeral, session-only what-if simulation of series/parallel
# packs; untouched by this hierarchy). Cells are never their own DB rows
# anywhere in this app (they live as DataFrames/model bundles), so PackCell
# links a Pack to an existing string cell_id the same way CellCohortTag
# already does, rather than introducing a new Cell table.
# ---------------------------------------------------------------------------

class Site(Base):
    __tablename__ = "sites"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    org_id     = Column(Integer, nullable=False, default=_DEMO_ORG_ID)
    name       = Column(String, nullable=False)
    created_at = Column(String)


class Fleet(Base):
    __tablename__ = "fleets"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    org_id     = Column(Integer, nullable=False, default=_DEMO_ORG_ID)
    site_id    = Column(Integer, nullable=False)
    name       = Column(String, nullable=False)
    created_at = Column(String)


class Pack(Base):
    __tablename__ = "packs"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    org_id     = Column(Integer, nullable=False, default=_DEMO_ORG_ID)
    fleet_id   = Column(Integer, nullable=False)
    name       = Column(String, nullable=False)
    created_at = Column(String)


class PackCell(Base):
    """One cell assigned to a pack. cell_id is the app's existing string
    cell identifier (NASA/Severson/synthetic/uploaded) -- not a foreign key
    to a Cell table, since none exists anywhere else in this app."""
    __tablename__ = "pack_cells"
    org_id   = Column(Integer, primary_key=True, default=_DEMO_ORG_ID)
    pack_id  = Column(Integer, primary_key=True)
    cell_id  = Column(String, primary_key=True)
    position = Column(Integer)


# ---------------------------------------------------------------------------
# FleetAsset hierarchy -- CRUD
# ---------------------------------------------------------------------------

def create_site(org_id: int, name: str) -> dict:
    with Session() as s:
        site = Site(org_id=org_id, name=name.strip(), created_at=datetime.datetime.now().isoformat())
        s.add(site)
        s.commit()
        s.refresh(site)
        return {"id": site.id, "org_id": site.org_id, "name": site.name, "created_at": site.created_at}


def list_sites(org_id: int) -> list[dict]:
    with Session() as s:
        rows = s.query(Site).filter_by(org_id=org_id).order_by(Site.id).all()
        return [{"id": r.id, "org_id": r.org_id, "name": r.name, "created_at": r.created_at} for r in rows]


def create_fleet(org_id: int, site_id: int, name: str) -> dict:
    with Session() as s:
        fleet = Fleet(org_id=org_id, site_id=site_id, name=name.strip(), created_at=datetime.datetime.now().isoformat())
        s.add(fleet)
        s.commit()
        s.refresh(fleet)
        return {"id": fleet.id, "org_id": fleet.org_id, "site_id": fleet.site_id,
                "name": fleet.name, "created_at": fleet.created_at}


def list_fleets(org_id: int, site_id: "int | None" = None) -> list[dict]:
    with Session() as s:
        q = s.query(Fleet).filter_by(org_id=org_id)
        if site_id is not None:
            q = q.filter_by(site_id=site_id)
        rows = q.order_by(Fleet.id).all()
        return [{"id": r.id, "org_id": r.org_id, "site_id": r.site_id,
                  "name": r.name, "created_at": r.created_at} for r in rows]


def create_pack(org_id: int, fleet_id: int, name: str) -> dict:
    with Session() as s:
        pack = Pack(org_id=org_id, fleet_id=fleet_id, name=name.strip(), created_at=datetime.datetime.now().isoformat())
        s.add(pack)
        s.commit()
        s.refresh(pack)
        return {"id": pack.id, "org_id": pack.org_id, "fleet_id": pack.fleet_id,
                "name": pack.name, "created_at": pack.created_at}


def list_packs(org_id: int, fleet_id: "int | None" = None) -> list[dict]:
    with Session() as s:
        q = s.query(Pack).filter_by(org_id=org_id)
        if fleet_id is not None:
            q = q.filter_by(fleet_id=fleet_id)
        rows = q.order_by(Pack.id).all()
        return [{"id": r.id, "org_id": r.org_id, "fleet_id": r.fleet_id,
                  "name": r.name, "created_at": r.created_at} for r in rows]


def add_cell_to_pack(org_id: int, pack_id: int, cell_id: str, position: "int | None" = None) -> None:
    with Session() as s:
        s.merge(PackCell(org_id=org_id, pack_id=pack_id, cell_id=cell_id, position=position))
        s.commit()


def remove_cell_from_pack(org_id: int, pack_id: int, cell_id: str) -> None:
    with Session() as s:
        row = s.query(PackCell).filter_by(org_id=org_id, pack_id=pack_id, cell_id=cell_id).one_or_none()
        if row is not None:
            s.delete(row)
            s.commit()


def list_pack_cells(org_id: int, pack_id: int) -> list[str]:
    with Session() as s:
        rows = (
            s.query(PackCell)
            .filter_by(org_id=org_id, pack_id=pack_id)
            .order_by(PackCell.position, PackCell.cell_id)
            .all()
        )
        return [r.cell_id for r in rows]


def _ensure_default_site_and_fleet(s, org_id: int) -> None:
    """Idempotent, non-destructive: gives org_id one default Site + one
    default Fleet underneath it if it doesn't have any Site yet. Only ever
    adds rows, never modifies/removes anything -- safe to call for both a
    brand-new org (create_organization_with_admin) and every pre-existing
    org on every init_db() (_seed_default_fleet_hierarchy)."""
    existing = s.query(Site).filter_by(org_id=org_id).first()
    if existing is not None:
        return
    site = Site(org_id=org_id, name="Default Site", created_at=datetime.datetime.now().isoformat())
    s.add(site)
    s.commit()
    s.refresh(site)
    s.add(Fleet(
        org_id=org_id, site_id=site.id, name="Default Fleet",
        created_at=datetime.datetime.now().isoformat(),
    ))
    s.commit()


def _seed_default_fleet_hierarchy() -> None:
    """Additive backfill: every existing Organization gets one default Site
    + Fleet via _ensure_default_site_and_fleet() above. Mirrors
    _seed_demo_org_and_users()'s "create on first run only" pattern."""
    with Session() as s:
        for org in s.query(Organization).all():
            _ensure_default_site_and_fleet(s, org.id)
