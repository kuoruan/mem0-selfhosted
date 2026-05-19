"""Background task that periodically prunes stale rows from request_logs."""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from db import SessionLocal
from models import RequestLog

logger = logging.getLogger("mem0.server.bg_tasks")

_MIN_INTERVAL = 60  # seconds — floor to avoid hammering the DB
_BATCH_SIZE = 1000  # rows per DELETE batch
_PRUNE_ADVISORY_LOCK_ID = 0x6D656D30  # "mem0"


def _get_retention_days() -> int:
    raw = os.environ.get("REQUEST_LOG_RETENTION_DAYS", "").strip() or "30"
    try:
        value = int(raw)
    except ValueError:
        logger.warning("REQUEST_LOG_RETENTION_DAYS is not an integer; using default 30")
        return 30
    if value < 1:
        logger.warning("REQUEST_LOG_RETENTION_DAYS=%d is below minimum 1; clamped", value)
        return 1
    return value


def _get_interval_seconds() -> int:
    raw = os.environ.get("REQUEST_LOG_PRUNE_INTERVAL_SECONDS", "").strip()
    try:
        value = int(raw) if raw else 86400
    except ValueError:
        logger.warning("REQUEST_LOG_PRUNE_INTERVAL_SECONDS is not an integer; using default 86400")
        return 86400
    if value < _MIN_INTERVAL:
        logger.warning("REQUEST_LOG_PRUNE_INTERVAL_SECONDS=%d is below minimum %d; clamped", value, _MIN_INTERVAL)
        return _MIN_INTERVAL
    return value


def _is_prune_enabled() -> bool:
    raw = os.environ.get("REQUEST_LOG_PRUNE_ENABLED", "").strip().lower()
    # Enabled by default; opt out by setting to "false" or "0".
    return raw not in ("false", "0", "no")


def _try_advisory_lock(session) -> bool:
    """Try to acquire a non-blocking pg advisory lock for prune dedup.

    Returns True if the lock was acquired, False otherwise (another worker
    is already pruning).
    """
    locked = session.scalar(select(func.pg_try_advisory_lock(_PRUNE_ADVISORY_LOCK_ID)))
    return bool(locked)


def _release_advisory_lock(session) -> None:
    session.execute(select(func.pg_advisory_unlock(_PRUNE_ADVISORY_LOCK_ID)))


async def prune_loop() -> None:
    """Run prune on startup and then every *interval* seconds until cancelled."""
    if not _is_prune_enabled():
        logger.info("Request log auto-prune is disabled (REQUEST_LOG_PRUNE_ENABLED=false).")
        return

    interval = _get_interval_seconds()
    while True:
        try:
            await asyncio.to_thread(_do_prune)
        except asyncio.CancelledError:
            logger.info("Request log prune task cancelled; exiting.")
            break
        except Exception:
            logger.exception("Request log prune failed; will retry next cycle")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Request log prune task cancelled; exiting.")
            break


def _do_prune() -> None:
    retention_days = _get_retention_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    total_removed = 0

    with SessionLocal() as session:
        if not _try_advisory_lock(session):
            logger.info("Another worker holds the prune advisory lock; skipping.")
            return
        try:
            while True:
                # PostgreSQL doesn't support LIMIT on DELETE; use a subquery
                # to batch-delete rows by ID.
                subquery = select(RequestLog.id).where(RequestLog.created_at < cutoff).limit(_BATCH_SIZE)
                result = session.execute(delete(RequestLog).where(RequestLog.id.in_(subquery)))
                session.commit()
                removed = result.rowcount or 0
                total_removed += removed
                if removed < _BATCH_SIZE:
                    break
        except Exception:
            session.rollback()
        finally:
            try:
                _release_advisory_lock(session)
            except Exception:
                logger.warning("Failed to release prune advisory lock", exc_info=True)

    logger.info(
        "Pruned %d request_log rows older than %s (retention=%dd)",
        total_removed,
        cutoff.isoformat(),
        retention_days,
    )
