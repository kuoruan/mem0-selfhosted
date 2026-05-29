"""Per-scope and per-memory locks for Memory write operations.

* **Entity scope** (``user_id``, ``agent_id``, ``app_id``, ``run_id``): used for
  ``add``, ``delete_all``, and background v3 add. Same scope serializes; different
  scopes may run concurrently.
* **``memory_id``**: used for single-memory ``update`` / ``delete``. Same id
  serializes; different ids may run concurrently even under the same user scope.
* **Global**: ``reset``, config reload, and cross-scope batch operations.

``add`` for a scope and ``update`` / ``delete`` on one memory in that scope use
different per-key locks and may run concurrently under the same scope. Clients
that require strict ordering (e.g. add then immediately update the same memory)
must serialize those calls at the API layer.
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


class _RWGate:
    """A simple reader/writer gate.

    - Scoped and memory-id writes acquire a *read* slot, allowing concurrency.
    - Global operations acquire a *write* slot, blocking all other operations.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._pending_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._pending_writers > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers = max(self._readers - 1, 0)
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._pending_writers += 1
        try:
            with self._cond:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._pending_writers -= 1
                self._writer = True
            try:
                yield
            finally:
                with self._cond:
                    self._writer = False
                    self._cond.notify_all()
        finally:
            with self._cond:
                if self._pending_writers > 0 and not self._writer:
                    self._pending_writers -= 1
                    self._cond.notify_all()


_GLOBAL_GATE = _RWGate()

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
    return (("memory_id", str(memory_id).strip()),)


def scope_lock_key(entity_scope: Dict[str, str]) -> ScopeLockKey:
    """Build a hashable lock key from entity scope fields present in *entity_scope*."""
    items = tuple(
        (field, str(entity_scope[field]).strip())
        for field in _SCOPE_KEY_ORDER
        if entity_scope.get(field) is not None and str(entity_scope[field]).strip()
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
            trimmed = str(value).strip()
            if trimmed:
                scope[key] = trimmed
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
        record.in_use += 1
        _touch_lock_key(key)
        return record


@contextmanager
def _hold_record(record: _LockRecord, key: ScopeLockKey) -> Iterator[None]:
    acquired = False
    try:
        record.lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
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
    """Hold the lock for *entity_scope*.

    When *global_lock* is True, blocks all other writes (scope + memory-id) for the
    duration of the context.
    """
    scope = entity_scope or {}
    if not global_lock:
        try:
            scoped_key = scope_lock_key(scope)
        except ValueError:
            # Some write operations (e.g. delete-all/reset) may be intentionally unscoped.
            # Treat them as global operations rather than raising a 500.
            global_lock = True
        else:
            # Use the scoped key inside the read gate.
            with _GLOBAL_GATE.read():
                record = _get_lock_record(scoped_key)
                with _hold_record(record, scoped_key):
                    yield
            return

    if global_lock:
        with _GLOBAL_GATE.write():
            yield
        return


@contextmanager
def memory_id_lock(memory_id: str) -> Iterator[None]:
    """Hold the lock for a single ``memory_id`` (update/delete on one record)."""
    with _GLOBAL_GATE.read():
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
