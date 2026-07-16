"""Prepare the Postgres backend for the mem0 API server.

Runs at container start (before ``alembic upgrade head``) to:

1. wait for Postgres to accept connections — retries with an exponential
   backoff (starting at 1s, doubling each retry, capped at ``MAX_SLEEP``) for
   up to ``WAIT_TOTAL`` seconds, then exits non-zero. Permanent errors (bad
   credentials, wrong maintenance database) abort immediately instead of
   waiting out the full timeout.
2. create the application database (``APP_DB_NAME``, default ``mem0_app``) if it
   does not yet exist, so non-compose deploys (e.g. plain ``docker run`` with an
   external Postgres) still get the DB.

All connections go to the maintenance database (``POSTGRES_DB``, default
``postgres``); that DB always exists and is the only one reachable before the
app database is created. Connection parameters come from the same ``POSTGRES_*``
/ ``APP_DB_NAME`` env vars used by ``db._build_database_url``.
"""
import os
import sys
import time

import psycopg
from psycopg import sql

HOST = os.environ.get("POSTGRES_HOST", "postgres")
PORT = os.environ.get("POSTGRES_PORT", "5432")
USER = os.environ.get("POSTGRES_USER", "postgres")
PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
MAINT_DB = os.environ.get("POSTGRES_DB", "postgres")
APP_DB = os.environ.get("APP_DB_NAME", "mem0_app")

WAIT_TOTAL = 60
MAX_SLEEP = 60
CONNECT_TIMEOUT = 2

# SQLSTATEs for permanent config errors — retrying won't help, so fail fast
# instead of holding the container hostage for the whole WAIT_TOTAL window.
FATAL_SQLSTATES = frozenset({
    "28000",  # invalid_authorization_specification (bad user/auth setup)
    "28P01",  # invalid_password
    "3D000",  # invalid_catalog_name (maintenance DB does not exist)
})

# duplicate_database — two containers racing on first deploy; the peer created
# the DB between our SELECT and CREATE. Treat as success (idempotent).
_RACE_DUPLICATE_DATABASE = "42P04"


def _log(message: str, *, error: bool = False) -> None:
    print(f"[db-init] {message}", file=sys.stderr if error else sys.stdout, flush=True)


def _connect(dbname: str, timeout: int) -> psycopg.connection:
    return psycopg.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        dbname=dbname,
        connect_timeout=timeout,
    )


def wait_for_postgres() -> None:
    """Block until Postgres accepts connections, or abort after WAIT_TOTAL seconds."""
    deadline = time.monotonic() + WAIT_TOTAL
    delay = 1
    while True:
        try:
            _connect(MAINT_DB, CONNECT_TIMEOUT).close()
            _log(f"Postgres is reachable at {HOST}:{PORT}.")
            return
        except Exception as exc:
            if (sqlstate := getattr(exc, "sqlstate", None)) in FATAL_SQLSTATES:
                _log(
                    f"Postgres rejected the connection (sqlstate={sqlstate}); this looks "
                    f"like a permanent config error, not a startup race: {exc}",
                    error=True,
                )
                sys.exit(1)
            if time.monotonic() >= deadline:
                _log(f"Postgres at {HOST}:{PORT} still unreachable after {WAIT_TOTAL}s; aborting.", error=True)
                sys.exit(1)
            time.sleep(min(delay, deadline - time.monotonic()))
            delay = min(delay * 2, MAX_SLEEP)


def ensure_app_db() -> None:
    """Create the application database if it does not already exist."""
    conn = _connect(MAINT_DB, 5)
    try:
        conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (APP_DB,))
            if cur.fetchone() is None:
                _log(f'Creating application database "{APP_DB}"...')
                try:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(APP_DB)))
                except psycopg.Error as exc:
                    if getattr(exc, "sqlstate", None) == _RACE_DUPLICATE_DATABASE:
                        _log(f'Application database "{APP_DB}" created concurrently; proceeding.')
                    else:
                        raise
    finally:
        conn.close()


def main() -> None:
    wait_for_postgres()
    ensure_app_db()


if __name__ == "__main__":
    main()
