"""Compatibility background task helpers."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from compat.events import event_cache_update
from compat.responses import normalize_results

logger = logging.getLogger("mem0.server.compat.tasks")


def run_v3_add_memory_task(
    event_id: str,
    messages: List[Dict[str, Any]],
    params: Dict[str, Any],
    memory_instance: Any,
) -> None:
    """Execute add in the background and update synthetic event status."""
    started_at = time.perf_counter()
    try:
        raw = memory_instance.add(messages=messages, **params)
        items = normalize_results(raw)
        finished_iso = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - started_at) * 1000
        event_cache_update(
            event_id,
            status="SUCCEEDED",
            results=items,
            updated_at=finished_iso,
            completed_at=finished_iso,
            latency=latency_ms,
        )
    except Exception as exc:
        logger.exception("v3_add_memory background task failed for event_id=%s", event_id)
        finished_iso = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - started_at) * 1000
        event_cache_update(
            event_id,
            status="FAILED",
            updated_at=finished_iso,
            completed_at=finished_iso,
            latency=latency_ms,
            metadata={"error": str(exc)},
        )
