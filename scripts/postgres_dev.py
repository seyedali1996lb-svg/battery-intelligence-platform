"""
Project-local PostgreSQL dev instance (validated the migration path:
scripts/migrate_sqlite_to_postgres.py, plus the DATABASE_URL backend).

No PostgreSQL is installed on this machine, so the official Windows
binaries bundled by the `postgresql-binaries` pip package (extracted to
.tools/postgres by this script on first use) provide a real server that
lives entirely inside the project — nothing system-wide, nothing outside
the repo. The data directory is .tools/pgdata (gitignored).

Usage
-----
    python scripts/postgres_dev.py init      # extract binaries + initdb (once)
    python scripts/postgres_dev.py start     # pg_ctl start on port 54329
    python scripts/postgres_dev.py status
    python scripts/postgres_dev.py psql -c "SELECT version();"
    python scripts/postgres_dev.py stop

The default connection for the app layer is:

    postgresql+psycopg2://postgres:postgres@127.0.0.1:54329/battery_platform

(trust auth on 127.0.0.1 for the dev cluster — see initdb's --auth=trust.)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / ".tools"
PG_DIR = TOOLS / "postgres"
PGDATA = TOOLS / "pgdata"
PORT = "54329"
USER = "postgres"
PASS = "postgres"
DB = "battery_platform"
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASS}@127.0.0.1:{PORT}/{DB}"

_BUNDLE = (
    pathlib.Path(sys.prefix)
    / "Lib/site-packages/postgresql_binaries"
    / "postgresql-18.4.0-x86_64-pc-windows-msvc.tar.gz"
)
_PG_BIN = PG_DIR / "postgresql-18.4.0-x86_64-pc-windows-msvc" / "bin"


def _bin(name: str) -> str:
    return str(_PG_BIN / f"{name}.exe")


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print("  " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def _extract() -> None:
    if (_PG_BIN / "postgres.exe").exists():
        print(f"[ok] binaries already extracted at {_PG_BIN}")
        return
    if not _BUNDLE.exists():
        raise SystemExit(
            "postgresql-binaries package not found — run: "
            "pip install -r requirements-dev.txt (dev-only; not in requirements.txt)"
        )
    print(f"[..] extracting {_BUNDLE.name} → {PG_DIR}")
    if PG_DIR.exists():
        shutil.rmtree(PG_DIR)
    PG_DIR.mkdir(parents=True)
    with tarfile.open(_BUNDLE, "r:gz") as tf:
        tf.extractall(PG_DIR, filter="data")
    print("[ok] extracted")


def _cmd_init() -> None:
    _extract()
    if PGDATA.exists():
        print(f"[ok] cluster already initialized at {PGDATA}")
        return
    PGDATA.mkdir(parents=True, exist_ok=True)
    r = _run([_bin("initdb"), "-D", str(PGDATA), "-U", USER, "--auth=trust", "--encoding=UTF8"], check=False)
    if r.returncode != 0:
        raise SystemExit("initdb failed")
    print(f"[ok] initialized. Connection URL:\n    {DATABASE_URL}")


def _cmd_start() -> None:
    if not PGDATA.exists():
        _cmd_init()
    r = _run(
        [_bin("pg_ctl"), "-D", str(PGDATA), "-l", str(TOOLS / "pg.log"),
         "-o", f"-p {PORT} -c listen_addresses=127.0.0.1", "start"],
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit("pg_ctl start failed (see .tools/pg.log)")
    _ensure_db()


def _ensure_db() -> None:
    out = _run(
        [_bin("psql"), "-h", "127.0.0.1", "-p", PORT, "-U", USER, "-d", "postgres",
         "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{DB}'"],
        capture_output=True, text=True, check=False,
    )
    if out.stdout.strip() != "1":
        _run([_bin("createdb"), "-h", "127.0.0.1", "-p", PORT, "-U", USER, DB], check=False)
        print(f"[ok] created database '{DB}'")
    else:
        print(f"[ok] database '{DB}' already exists")


def _cmd_status() -> None:
    r = _run(
        [_bin("pg_isready"), "-h", "127.0.0.1", "-p", PORT],
        capture_output=True, text=True, check=False,
    )
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode == 0:
        _run([_bin("psql"), "-h", "127.0.0.1", "-p", PORT, "-U", USER, "-d", "postgres",
              "-tAc", "SELECT version();"], check=False)


def _cmd_stop() -> None:
    if not PGDATA.exists():
        print("no cluster to stop")
        return
    _run([_bin("pg_ctl"), "-D", str(PGDATA), "stop"], check=False)


def _cmd_psql(args) -> None:
    _run([_bin("psql"), "-h", "127.0.0.1", "-p", PORT, "-U", USER, "-d", DB, *args], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("stop")
    psql_p = sub.add_parser("psql")
    psql_p.add_argument("psql_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd == "init":
        _cmd_init()
    elif args.cmd == "start":
        _cmd_start()
    elif args.cmd == "status":
        _cmd_status()
    elif args.cmd == "stop":
        _cmd_stop()
    elif args.cmd == "psql":
        _cmd_psql(args.psql_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
