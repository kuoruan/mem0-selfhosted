"""Smoke test for the alembic migration chain.

Runs the full migration chain against an **isolated** throwaway database
(``mem0_app_migtest``) so the shared test DB's schema is never clobbered.

Verifies the chain is internally consistent:
- ``upgrade head`` from an empty DB succeeds (every revision's ``upgrade`` works).
- ``downgrade base`` succeeds (every revision's ``downgrade`` works, in reverse).
- ``upgrade head`` again succeeds (downgrade→upgrade round-trip is idempotent).

This catches broken upgrade/downgrade SQL (e.g. a revision that drops a column
another still references, or a non-reversible migration). It does NOT catch
in-place edits of already-applied revisions — that is a release-discipline
issue (never edit an applied migration; add a new revision instead).

Skips when alembic / psycopg / a reachable postgres are unavailable, so it is
a no-op in environments without the Docker test DB.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("alembic", reason="alembic not installed")


_ALEMBIC_INI = "alembic.ini"
_MIG_DB = "mem0_app_migtest"

# Run alembic from server/ so the relative ``script_location = alembic`` in
# alembic.ini (resolved against the CWD, not the ini dir) finds the scripts
# folder. Mirrors the CI step ``cd server && alembic upgrade head``.
_SERVER_DIR = Path(__file__).resolve().parents[2] / "server"


def _pg_conn_params(db_name: str) -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "postgres"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "dbname": db_name,
    }


def _can_connect() -> bool:
    """True if the postgres maintenance DB is reachable."""
    try:
        with psycopg.connect(**_pg_conn_params("postgres"), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _can_connect(),
    reason="A reachable postgres is required",
)


@pytest.fixture()
def isolated_db():
    """Create a throwaway database; drop it on teardown."""
    import uuid

    db_name = f"{_MIG_DB}_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(**_pg_conn_params("postgres"), autocommit=True) as autocommit:
        autocommit.execute(f"DROP DATABASE IF EXISTS {db_name}")
        autocommit.execute(f"CREATE DATABASE {db_name}")
        try:
            yield db_name
        finally:
            autocommit.execute(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)")


def _alembic(*args: str, db_name: str) -> subprocess.CompletedProcess:
    """Run an alembic command against *db_name* (APP_DB_NAME override)."""
    env = dict(os.environ, APP_DB_NAME=db_name)
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", _ALEMBIC_INI, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_SERVER_DIR,
    )


def test_migration_chain_round_trip(isolated_db):
    """upgrade head -> downgrade base -> upgrade head, all exit 0."""
    up1 = _alembic("upgrade", "head", db_name=isolated_db)
    assert up1.returncode == 0, f"upgrade head failed:\n{up1.stderr}\n{up1.stdout}"

    down = _alembic("downgrade", "base", db_name=isolated_db)
    assert down.returncode == 0, f"downgrade base failed:\n{down.stderr}\n{down.stdout}"

    up2 = _alembic("upgrade", "head", db_name=isolated_db)
    assert up2.returncode == 0, f"re-upgrade head failed:\n{up2.stderr}\n{up2.stdout}"


def test_alembic_heads_are_not_branched(isolated_db):
    """A single non-branched head (no divergent revision trees)."""
    heads = _alembic("heads", db_name=isolated_db)
    assert heads.returncode == 0, heads.stderr
    # `alembic heads` prints one head per line (plus possible blank lines).
    head_lines = [
        ln
        for ln in heads.stdout.splitlines()
        if ln.strip() and not ln.startswith(("INFO", "DEBUG", "WARNING", "ERROR"))
    ]
    assert len(head_lines) == 1, f"expected a single head, got: {head_lines}"
