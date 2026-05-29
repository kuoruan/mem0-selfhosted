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
- ``owner_user_id`` scopes list/get access to the authenticated user; admin API
  key / auth-disabled mode may see all cached events.
"""

import copy
import threading
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict, Field

from compat.responses import normalize_results
from compat.utils import now_iso as utc_now_iso

_EVENT_CACHE_TTL_SECONDS = 600  # 10 minutes
_EVENT_CACHE_MAXSIZE = 10_000

_lock = threading.Lock()
_event_cache: TTLCache = TTLCache(maxsize=_EVENT_CACHE_MAXSIZE, ttl=_EVENT_CACHE_TTL_SECONDS)

CompatEventStatus = Literal["PENDING", "RUNNING", "FAILED", "SUCCEEDED"]


class CompatEvent(BaseModel):
    """Synthetic event model aligned with docs/openapi.json event fields."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="The unique identifier of the event.")
    event_type: str = Field(description="The type of event, for example ADD or SEARCH.")
    status: CompatEventStatus = Field(description="The current processing status of the event.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="The original payload associated with the event.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata associated with the event."
    )
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
    owner_user_id: Optional[str] = Field(
        default=None,
        description="Authenticated user who created the event (for access control).",
    )

    @classmethod
    def pending(
        cls, event_id: str, *, now_iso: Optional[str] = None, owner_user_id: Optional[str] = None
    ) -> "CompatEvent":
        """Build a queued ADD event returned immediately to the client."""
        ts = utc_now_iso(now_iso)
        return cls(
            id=event_id,
            event_type="ADD",
            status="PENDING",
            payload={},
            metadata=None,
            results=[],
            created_at=ts,
            updated_at=ts,
            started_at=ts,
            completed_at=None,
            latency=None,
            owner_user_id=owner_user_id,
        )

    @classmethod
    def create_add(
        cls,
        event_id: str,
        results: Any,
        *,
        status: CompatEventStatus = "SUCCEEDED",
        now_iso: Optional[str] = None,
        completed_at: Optional[str] = None,
        latency: Optional[float] = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CompatEvent":
        """Build a synthetic ADD event (default ``SUCCEEDED`` after background work)."""
        ts = utc_now_iso(now_iso)
        if completed_at is None and status == "SUCCEEDED":
            completed_at = ts
        return cls(
            id=event_id,
            event_type="ADD",
            status=status,
            payload={},
            metadata=metadata,
            results=normalize_results(results),
            created_at=ts,
            updated_at=ts,
            started_at=ts,
            completed_at=completed_at,
            latency=latency,
        )


def create_pending_add_event(owner_user_id: Optional[str]) -> str:
    """Create a queued ADD event and return its id for client polling."""
    event_id = str(uuid.uuid4())
    now_iso = utc_now_iso()
    event_cache_put(
        event_id,
        CompatEvent.pending(event_id, now_iso=now_iso, owner_user_id=owner_user_id),
    )
    return event_id


def event_cache_put(event_id: str, event: Union[CompatEvent, Dict[str, Any]]) -> None:
    """Store an event object in the TTL cache."""
    validated = event.model_dump() if isinstance(event, CompatEvent) else CompatEvent.model_validate(event).model_dump()
    with _lock:
        _event_cache[event_id] = validated


def event_cache_get(event_id: str) -> Optional[Dict[str, Any]]:
    """Return the cached event object, or ``None`` if missing or expired."""
    with _lock:
        event_obj = _event_cache.get(event_id)
    if event_obj is None:
        return None
    return copy.deepcopy(event_obj)


def event_cache_update(event_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Update a cached event in-place and return the updated object.

    Returns ``None`` when the event does not exist (missing or expired).
    """
    # Owner is set at creation time and must not be reassigned via updates.
    fields = {key: value for key, value in fields.items() if key != "owner_user_id"}
    with _lock:
        event_obj = _event_cache.get(event_id)
        if event_obj is None:
            return None
        updated = CompatEvent.model_validate(event_obj).model_copy(update=fields)
        validated = updated.model_dump()
        _event_cache[event_id] = validated
    return copy.deepcopy(validated)


def event_cache_all() -> List[Dict[str, Any]]:
    """Return all non-expired event objects sorted by ``created_at`` descending."""
    with _lock:
        items = list(_event_cache.values())
    copied = [copy.deepcopy(item) for item in items]
    return sorted(copied, key=lambda o: o.get("created_at", ""), reverse=True)


def event_cache_clear() -> None:
    """Remove all cached synthetic events."""
    with _lock:
        _event_cache.clear()


def resolve_event_owner_id(auth: Any, entity_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve the authenticated owner id to store on synthetic events."""
    if auth is not None:
        owner_id = auth.get("id", None) if isinstance(auth, dict) else getattr(auth, "id", None)
        if owner_id is not None:
            return str(owner_id).strip()
    if entity_params:
        scoped_user = entity_params.get("user_id")
        if scoped_user is not None and str(scoped_user).strip():
            return str(scoped_user).strip()
    return None


def events_visible_to_caller(events: List[Dict[str, Any]], caller_id: Optional[str]) -> List[Dict[str, Any]]:
    """Filter events to those owned by *caller_id*.

    When *caller_id* is ``None`` (admin API key / auth disabled), all events are visible.
    """
    if caller_id is None:
        return events
    return [event for event in events if event.get("owner_user_id") == caller_id]


def event_access_allowed(event: Dict[str, Any], caller_id: Optional[str]) -> bool:
    """Return whether *caller_id* may read *event*."""
    if caller_id is None:
        return True
    owner = event.get("owner_user_id")
    if owner is None:
        return False
    return owner == caller_id


def make_event_obj(
    event_id: str,
    results: Any,
    now_iso: Optional[str] = None,
    status: CompatEventStatus = "SUCCEEDED",
    completed_at: Optional[str] = None,
    latency: Optional[float] = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a synthetic event dict (backward-compatible wrapper around ``CompatEvent.create_add``)."""
    return CompatEvent.create_add(
        event_id,
        results,
        status=status,
        now_iso=now_iso,
        completed_at=completed_at,
        latency=latency,
        metadata=metadata,
    ).model_dump()
