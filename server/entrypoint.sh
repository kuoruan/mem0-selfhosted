#!/bin/sh
set -e

# Entrypoint for the mem0 API server image.
#
# On a normal server boot (default CMD: `uvicorn main:app ...`), the script:
#   1. prepares the Postgres backend via scripts/prepare_db.py — waits for
#      Postgres to accept connections (exponential backoff from 1s, up to 60s,
#      then aborts; permanent config errors fail fast) and creates the app
#      database (APP_DB_NAME, default `mem0_app`) if missing,
#   2. runs `alembic upgrade head`,
#   then hands control to the server via `exec "$@"`.
#
# Any other command (e.g. `docker run <image> alembic downgrade -1`) skips the
# init block and runs verbatim, so ops/ad-hoc commands are not preceded by a
# migration.

if [ "${1:-}" = "uvicorn" ]; then
    python scripts/prepare_db.py
    echo "[entrypoint] Running alembic upgrade head..."
    alembic upgrade head
fi

exec "$@"
