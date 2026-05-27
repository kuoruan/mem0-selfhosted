"""Synthetic event tracking for the self-hosted server.

The hosted Mem0 platform processes ``add_memory`` calls asynchronously and
exposes ``GET /v1/event/{event_id}`` for polling.  The self-hosted v3 add
endpoint (``POST /v3/memories/add``) follows the same pattern: it enqueues work
via FastAPI ``BackgroundTasks``, returns ``PENDING`` with an ``event_id`` right
away, and updates this in-memory TTL cache when the background job finishes
(``SUCCEEDED`` with results, or ``FAILED`` with error metadata).

Clients should poll ``GET /v1/event/{event_id}`` (or list ``GET /v1/events``)
until the status leaves ``PENDING``.  There is no durable event store; entries
live only in this process-local cache.

Limitations:
- Cache is process-local; multi-worker/pod deployments may return 404 when the
  poll request lands on a different worker than the add request.
- Cached entries include add results for client compatibility and are retained
  only for a short TTL.
- If the process exits before a background add completes, the event may remain
  ``PENDING`` or disappear after TTL expiry.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict, Field

_EVENT_CACHE_TTL_SECONDS = 600  # 10 minutes
_EVENT_CACHE_MAXSIZE = 10_000

_lock = threading.Lock()
_event_cache: TTLCache = TTLCache(maxsize=_EVENT_CACHE_MAXSIZE, ttl=_EVENT_CACHE_TTL_SECONDS)


class CompatEvent(BaseModel):
    """Synthetic event model aligned with docs/openapi.json event fields."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="The unique identifier of the event.")
    event_type: str = Field(description="The type of event, for example ADD or SEARCH.")
    status: Literal["PENDING", "RUNNING", "FAILED", "SUCCEEDED"] = Field(
        description="The current processing status of the event."
    )
    payload: Dict[str, Any] = Field(default_factory=dict, description="The original payload associated with the event.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata associated with the event.")
    results: List[Any] = Field(
        default_factory=list,
        description="Results produced by the event; for ADD this confirms write completion.",
    )
    created_at: str = Field(description="Timestamp when the event was created (ISO 8601).")
    updated_at: str = Field(description="Timestamp when the event was last updated (ISO 8601).")
    started_at: str = Field(description="Timestamp when event processing started (ISO 8601).")
    completed_at: Optional[str] = Field(
        default=None,
        description="Timestamp when event processing completed (ISO 8601).",
    )
    latency: Optional[float] = Field(default=None, description="Processing time in milliseconds.")


def event_cache_put(event_id: str, event_obj: Dict[str, Any]) -> None:
    """Store an event object in the TTL cache."""
    validated = CompatEvent.model_validate(event_obj).model_dump()
    with _lock:
        _event_cache[event_id] = validated


def event_cache_get(event_id: str) -> Optional[Dict[str, Any]]:
    """Return the cached event object, or ``None`` if missing or expired."""
    with _lock:
        event_obj = _event_cache.get(event_id)
    if event_obj is None:
        return None
    return dict(event_obj)


def event_cache_update(event_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Update a cached event in-place and return the updated object.

    Returns ``None`` when the event does not exist (missing or expired).
    """
    with _lock:
        event_obj = _event_cache.get(event_id)
        if event_obj is None:
            return None
        updated = dict(event_obj)
        updated.update(fields)
        validated = CompatEvent.model_validate(updated).model_dump()
        _event_cache[event_id] = validated
        return dict(validated)


def event_cache_all() -> List[Dict[str, Any]]:
    """Return all non-expired event objects sorted by ``created_at`` descending."""
    with _lock:
        items = [dict(item) for item in _event_cache.values()]
    return sorted(items, key=lambda o: o.get("created_at", ""), reverse=True)


def event_cache_clear() -> None:
    """Remove all cached synthetic events."""
    with _lock:
        _event_cache.clear()


def make_event_obj(
    event_id: str,
    results: Any,
    now_iso: Optional[str] = None,
    status: str = "SUCCEEDED",
    completed_at: Optional[str] = None,
    latency: Optional[float] = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a synthetic event object matching the platform's schema."""
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    if completed_at is None and status == "SUCCEEDED":
        completed_at = now_iso
    event = CompatEvent(
        id=event_id,
        event_type="ADD",
        status=status,
        payload={},
        metadata=metadata,
        results=results if isinstance(results, list) else [],
        created_at=now_iso,
        updated_at=now_iso,
        started_at=now_iso,
        completed_at=completed_at,
        latency=latency,
    )
    return event.model_dump()
