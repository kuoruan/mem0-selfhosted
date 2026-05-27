"""Synthetic event tracking for the self-hosted server.

The hosted Mem0 platform processes ``add_memory`` calls asynchronously and
exposes ``GET /v1/event/{event_id}`` for polling.  The self-hosted server
processes adds synchronously, so this module provides a lightweight in-memory
TTL cache that synthesises the same event objects immediately after each add,
allowing clients that poll the event endpoint to observe a ``SUCCEEDED`` status
without any persistence layer.

Limitations:
- Cache is process-local; multi-worker/pod deployments may return 404 when the
    poll request lands on a different worker than the add request.
- Cached entries include add results for client compatibility and are retained
    only for a short TTL.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

_EVENT_CACHE_TTL_SECONDS = 600  # 10 minutes
_EVENT_CACHE_MAXSIZE = 10_000

_lock = threading.Lock()
_event_cache: TTLCache = TTLCache(maxsize=_EVENT_CACHE_MAXSIZE, ttl=_EVENT_CACHE_TTL_SECONDS)


def event_cache_put(event_id: str, event_obj: Dict[str, Any]) -> None:
    """Store an event object in the TTL cache."""
    with _lock:
        _event_cache[event_id] = event_obj


def event_cache_get(event_id: str) -> Optional[Dict[str, Any]]:
    """Return the cached event object, or ``None`` if missing or expired."""
    with _lock:
        return _event_cache.get(event_id)


def event_cache_update(event_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Update a cached event in-place and return the updated object.

    Returns ``None`` when the event does not exist (missing or expired).
    """
    with _lock:
        event_obj = _event_cache.get(event_id)
        if event_obj is None:
            return None
        event_obj.update(fields)
        _event_cache[event_id] = event_obj
        return event_obj


def event_cache_all() -> List[Dict[str, Any]]:
    """Return all non-expired event objects sorted by ``created_at`` descending."""
    with _lock:
        items = list(_event_cache.values())
    return sorted(items, key=lambda o: o.get("created_at", ""), reverse=True)


def event_cache_clear() -> None:
    """Remove all cached synthetic events."""
    with _lock:
        _event_cache.clear()


def event_visible_to_caller(event_obj: Dict[str, Any], auth_user: Any) -> bool:
    """Return True if event should be visible to the current caller."""
    if auth_user is None:
        return True
    auth_user_id = getattr(auth_user, "id", None)
    if auth_user_id is None:
        return True
    owner_user_id = event_obj.get("owner_user_id")
    return owner_user_id is not None and owner_user_id == str(auth_user_id)


def make_event_obj(
    event_id: str,
    results: Any,
    now_iso: Optional[str] = None,
    owner_user_id: Optional[str] = None,
    scope: Optional[Dict[str, str]] = None,
    status: str = "SUCCEEDED",
    completed_at: Optional[str] = None,
    latency: Optional[float] = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a synthetic event object matching the platform's schema."""
    if now_iso is None:
        now_iso = datetime.now(timezone.utc).isoformat()
    scope = scope or {}
    if completed_at is None and status == "SUCCEEDED":
        completed_at = now_iso
    return {
        "id": event_id,
        "event_type": "ADD",
        "status": status,
        "payload": {},
        "metadata": metadata,
        "results": results if isinstance(results, list) else [],
        "created_at": now_iso,
        "updated_at": now_iso,
        "started_at": now_iso,
        "completed_at": completed_at,
        "latency": latency,
        "owner_user_id": owner_user_id,
        "scope": scope,
    }
