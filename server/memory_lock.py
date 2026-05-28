"""Per-scope and per-memory locks for Memory write operations.

* **Entity scope** (``user_id``, ``agent_id``, ``app_id``, ``run_id``): used for
  ``add``, ``delete_all``, and background v3 add. Same scope serializes; different
  scopes may run concurrently.
* **``memory_id``**: used for single-memory ``update`` / ``delete``. Same id
  serializes; different ids may run concurrently even under the same user scope.
* **Global**: ``reset``, config reload, and cross-scope batch operations.

``add`` for a scope and ``update`` on one memory in that scope are not held under
one lock; clients that require strict ordering should serialize those calls.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, TypeVar

from cachetools import TTLCache

from compat.scope import ENTITY_PARAMS

# Stable key order for lock identity and multi-lock acquisition.
_SCOPE_KEY_ORDER = ("agent_id", "app_id", "run_id", "user_id")

ScopeLockKey = Tuple[Tuple[str, str], ...]
_GLOBAL_LOCK_KEY: ScopeLockKey = ()

_registry_lock = threading.Lock()
_LOCK_REGISTRY_MAX = 10_000
_LOCK_REGISTRY_TTL_SECONDS = 600  # 10 minutes


@dataclass
class _LockRecord:
    lock: threading.RLock
    in_use: int = 0


_locks: Dict[ScopeLockKey, _LockRecord] = {}
_lock_ttl: TTLCache = TTLCache(maxsize=_LOCK_REGISTRY_MAX, ttl=_LOCK_REGISTRY_TTL_SECONDS)

T = TypeVar("T")


def memory_id_lock_key(memory_id: str) -> ScopeLockKey:
    """Build a lock key for a single memory record."""
    if not memory_id or not str(memory_id).strip():
        raise ValueError("memory_id is required for a memory-id lock.")
    return (("memory_id", str(memory_id)),)


def scope_lock_key(entity_scope: Dict[str, str]) -> ScopeLockKey:
    """Build a hashable lock key from entity scope fields present in *entity_scope*."""
    items = tuple(
        (field, str(entity_scope[field]))
        for field in _SCOPE_KEY_ORDER
        if entity_scope.get(field) is not None
    )
    if not items:
        raise ValueError(
            "At least one entity scope field (user_id, agent_id, app_id, run_id) is required for a scoped lock."
        )
    return items


def entity_scope_from_params(params: Dict[str, Any]) -> Dict[str, str]:
    """Extract entity scope fields from add/delete_all-style kwargs."""
    scope: Dict[str, str] = {}
    for key in ENTITY_PARAMS:
        value = params.get(key)
        if value is None:
            continue
        # Writes (add/delete_all) require a concrete scope string. Ignore operator
        # dicts/lists (e.g. {"in": [...]}) to avoid creating unstable lock keys.
        if isinstance(value, (str, int, float, bool)):
            scope[key] = str(value)
    return scope


def _gc_locks() -> None:
    """Expire TTL keys and drop unused lock records."""
    _lock_ttl.expire()
    expired_keys = [key for key in _locks.keys() if key not in _lock_ttl]
    for key in expired_keys:
        record = _locks.get(key)
        if record is not None and record.in_use == 0:
            _locks.pop(key, None)


def _touch_lock_key(key: ScopeLockKey) -> None:
    # Touching keeps the key alive; if TTL evicts it while in_use>0 we keep the record
    # and cleanup will remove it after release.
    _lock_ttl[key] = True


def _get_lock_record(key: ScopeLockKey) -> _LockRecord:
    with _registry_lock:
        _gc_locks()
        record = _locks.get(key)
        if record is None:
            record = _LockRecord(threading.RLock())
            _locks[key] = record
        _touch_lock_key(key)
        return record


@contextmanager
def _hold_record(record: _LockRecord, key: ScopeLockKey) -> Iterator[None]:
    record.lock.acquire()
    with _registry_lock:
        record.in_use += 1
        _touch_lock_key(key)
    try:
        yield
    finally:
        record.lock.release()
        with _registry_lock:
            record.in_use = max(record.in_use - 1, 0)
            _touch_lock_key(key)
            _gc_locks()


@contextmanager
def memory_scope_lock(
    entity_scope: Optional[Dict[str, str]] = None,
    *,
    global_lock: bool = False,
) -> Iterator[None]:
    """Hold the lock for *entity_scope*, or the process-wide lock when *global_lock* is True."""
    key = _GLOBAL_LOCK_KEY if global_lock else scope_lock_key(entity_scope or {})
    record = _get_lock_record(key)
    with _hold_record(record, key):
        yield


@contextmanager
def memory_id_lock(memory_id: str) -> Iterator[None]:
    """Hold the lock for a single ``memory_id`` (update/delete on one record)."""
    key = memory_id_lock_key(memory_id)
    record = _get_lock_record(key)
    with _hold_record(record, key):
        yield


def run_memory_write(
    fn: Callable[[Any], T],
    entity_scope: Optional[Dict[str, str]] = None,
    *,
    global_lock: bool = False,
) -> T:
    """Run ``fn(memory)`` while holding the appropriate scope lock."""
    from server_state import get_memory_instance

    with memory_scope_lock(entity_scope, global_lock=global_lock):
        return fn(get_memory_instance())


def run_memory_write_for_memory_id(
    fn: Callable[[Any], T],
    memory_id: str,
) -> T:
    """Run ``fn(memory)`` under the per-``memory_id`` lock."""
    from server_state import get_memory_instance

    with memory_id_lock(memory_id):
        return fn(get_memory_instance())
