"""Per-scope and per-memory locks for Memory write operations.

Locking model
-------------

* **Global** (``global_lock=True``): blocks every other write (scope + memory-id).
  Rare ops (reset, config reload). Under heavy scoped traffic, global writers may
  wait until in-flight scoped work finishes (writer preference on ``_RWGate``).
* **Entity scope**: one lock per present field; writers sharing any field serialize.
  Fields are always acquired in :data:`LOCK_ACQUIRE_ORDER` (coarse → fine).
* **``memory_id``**: per-record lock; with *entity_scope* (or
  ``run_memory_write_for_memory_id(..., resolve_scope=True)``) ancestor scope
  locks are taken **before** the memory-id lock.

Fixed multi-lock order (never reorder — deadlock risk)
------------------------------------------------------

``user_id`` → ``agent_id`` → ``app_id`` → ``run_id`` → ``memory_id``

All paths use :func:`scope_lock_keys` + optional :func:`memory_id_lock_key` via
:func:`_scoped_resource_lock` (``ExitStack``, same order).

Performance notes
-----------------

* ``resolve_scope=True`` (default on :func:`run_memory_write_for_memory_id`) calls
  ``memory.get(memory_id)`` once before locking. Pass *entity_scope* when already
  loaded to skip the extra read.
* Registry TTL/GC runs at most once per :data:`_GC_MIN_INTERVAL_S` while holding
  ``_registry_lock``. Do not touch ``_lock_ttl`` outside that lock (not thread-safe).
"""

import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, TypeVar

from cachetools import TTLCache

from compat.scope import ENTITY_PARAMS
from server_state import get_memory_instance

# Coarse-to-fine; must match LOCK_ACQUIRE_ORDER. ``scope_lock_key`` sorts by field name.
_SCOPE_FIELDS = ("user_id", "agent_id", "app_id", "run_id")
LOCK_ACQUIRE_ORDER: Tuple[str, ...] = _SCOPE_FIELDS + ("memory_id",)
_SCALAR_SCOPE_TYPES = (str, int, float, bool)

ScopeLockKey = Tuple[Tuple[str, str], ...]

_GC_MIN_INTERVAL_S = 60.0


class _RWGate:
    """A simple reader/writer gate.

    - Scoped and memory-id writes acquire a *read* slot, allowing concurrency.
    - Global operations acquire a *write* slot, blocking all other operations.

    While a writer is queued, new readers block (writer preference). Acquire and
    release for ``write()`` each run in a single ``Condition`` critical section
    so ``_pending_writers`` cannot race with ``read()``.
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
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._pending_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._pending_writers -= 1
                self._writer = True
            except BaseException:
                self._pending_writers -= 1
                self._cond.notify_all()
                raise
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


_GLOBAL_GATE = _RWGate()

_registry_lock = threading.Lock()
_LOCK_REGISTRY_MAX = 10_000
_LOCK_REGISTRY_TTL_SECONDS = 600  # 10 minutes
_last_gc_monotonic = 0.0


@dataclass
class _LockRecord:
    # RLock: defensive if a write path nests on the same key in one thread (not expected).
    lock: threading.RLock
    in_use: int = 0


_locks: Dict[ScopeLockKey, _LockRecord] = {}
_lock_ttl: TTLCache = TTLCache(maxsize=_LOCK_REGISTRY_MAX, ttl=_LOCK_REGISTRY_TTL_SECONDS)

T = TypeVar("T")


def _trimmed_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    trimmed = str(value).strip()
    return trimmed or None


def _scalar_scope_value(value: Any) -> Optional[str]:
    if not isinstance(value, _SCALAR_SCOPE_TYPES):
        return None
    return _trimmed_str(value)


def memory_id_lock_key(memory_id: str) -> ScopeLockKey:
    """Build a lock key for a single memory record."""
    trimmed = _trimmed_str(memory_id)
    if not trimmed:
        raise ValueError("memory_id is required for a memory-id lock.")
    return (("memory_id", trimmed),)


def scope_lock_keys(entity_scope: Dict[str, str]) -> Tuple[ScopeLockKey, ...]:
    """Return per-field lock keys in coarse-to-fine acquisition order."""
    keys = tuple(
        ((field, trimmed),)
        for field in _SCOPE_FIELDS
        if (trimmed := _trimmed_str(entity_scope.get(field)))
    )
    if not keys:
        raise ValueError(
            "At least one entity scope field (user_id, agent_id, app_id, run_id) is required for a scoped lock."
        )
    return keys


def scope_lock_key(entity_scope: Dict[str, str]) -> ScopeLockKey:
    """Flat composite key (all present fields) for identity / tests."""
    items = [
        (field, trimmed)
        for field in _SCOPE_FIELDS
        if (trimmed := _trimmed_str(entity_scope.get(field)))
    ]
    if not items:
        raise ValueError(
            "At least one entity scope field (user_id, agent_id, app_id, run_id) is required for a scoped lock."
        )
    return tuple(sorted(items, key=lambda pair: pair[0]))


def entity_scope_from_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Extract entity scope fields from a memory payload dict."""
    scope: Dict[str, str] = {}
    for key in ENTITY_PARAMS:
        trimmed = _scalar_scope_value(record.get(key))
        if trimmed:
            scope[key] = trimmed
    return scope


def entity_scope_for_memory_id(memory: Any, memory_id: str) -> Optional[Dict[str, str]]:
    """Load *memory_id* and return its entity scope for write locking (or ``None``)."""
    try:
        raw = memory.get(memory_id)
    except Exception:
        return None
    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        return None
    scope = entity_scope_from_record(item)
    return scope or None


def entity_scope_from_params(params: Dict[str, Any]) -> Dict[str, str]:
    """Extract entity scope fields from add/delete_all-style kwargs."""
    scope: Dict[str, str] = {}
    for key in ENTITY_PARAMS:
        trimmed = _scalar_scope_value(params.get(key))
        if trimmed:
            scope[key] = trimmed
    return scope


def _gc_locks() -> None:
    """Expire TTL keys and drop unused lock records."""
    global _last_gc_monotonic
    _last_gc_monotonic = time.monotonic()
    _lock_ttl.expire()
    for key in [k for k in _locks if k not in _lock_ttl]:
        record = _locks.get(key)
        if record and record.in_use == 0:
            _locks.pop(key, None)


def _maybe_gc_locks(*, force: bool = False) -> None:
    """Run registry GC at most once per minute unless *force*."""
    now = time.monotonic()
    if not force and (now - _last_gc_monotonic) < _GC_MIN_INTERVAL_S:
        return
    _gc_locks()


def _touch_lock_key(key: ScopeLockKey) -> None:
    """Caller must hold ``_registry_lock`` (``TTLCache`` is not thread-safe)."""
    _lock_ttl[key] = True


def _get_lock_record(key: ScopeLockKey) -> _LockRecord:
    with _registry_lock:
        _maybe_gc_locks()
        record = _locks.get(key)
        if record is None:
            record = _LockRecord(threading.RLock())
            _locks[key] = record
        record.in_use += 1
        _touch_lock_key(key)
        return record


def _release_lock_record(record: _LockRecord, key: ScopeLockKey) -> None:
    with _registry_lock:
        record.in_use -= 1
        if record.in_use < 0:
            raise RuntimeError(f"lock record in_use underflow for key {key!r}")
        _touch_lock_key(key)
        _maybe_gc_locks()


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
        _release_lock_record(record, key)


@contextmanager
def _scoped_resource_lock(keys: Tuple[ScopeLockKey, ...]) -> Iterator[None]:
    """Acquire *keys* under the global read gate, in list order (see LOCK_ACQUIRE_ORDER)."""
    if not keys:
        raise ValueError("at least one lock key is required")
    with _GLOBAL_GATE.read():
        with ExitStack() as stack:
            for key in keys:
                record = _get_lock_record(key)
                stack.enter_context(_hold_record(record, key))
            yield


@contextmanager
def memory_scope_lock(
    entity_scope: Optional[Dict[str, str]] = None,
    *,
    global_lock: bool = False,
) -> Iterator[None]:
    """Hold the lock for *entity_scope*.

    When *global_lock* is True, blocks all other writes (scope + memory-id) for the
    duration of the context. An empty scope without *global_lock* raises
    ``ValueError`` (fail-fast; do not silently upgrade to a global lock).
    """
    if global_lock:
        with _GLOBAL_GATE.write():
            yield
        return

    keys = scope_lock_keys(entity_scope or {})
    with _scoped_resource_lock(keys):
        yield


def memory_id_lock_keys(
    memory_id: str,
    entity_scope: Optional[Dict[str, str]] = None,
) -> Tuple[ScopeLockKey, ...]:
    """Lock keys for a memory-id write, in :data:`LOCK_ACQUIRE_ORDER`."""
    if entity_scope:
        return scope_lock_keys(entity_scope) + (memory_id_lock_key(memory_id),)
    return (memory_id_lock_key(memory_id),)


@contextmanager
def memory_id_lock(
    memory_id: str,
    entity_scope: Optional[Dict[str, str]] = None,
) -> Iterator[None]:
    """Hold the lock for a single ``memory_id`` (update/delete on one record).

    Keys must follow :data:`LOCK_ACQUIRE_ORDER`. When *entity_scope* is set,
    ancestor scope locks are taken first, then the memory-id lock.
    """
    with _scoped_resource_lock(memory_id_lock_keys(memory_id, entity_scope)):
        yield


def run_memory_write(
    fn: Callable[[Any], T],
    entity_scope: Optional[Dict[str, str]] = None,
    *,
    global_lock: bool = False,
) -> T:
    """Run ``fn(memory)`` while holding the appropriate scope lock."""
    with memory_scope_lock(entity_scope, global_lock=global_lock):
        return fn(get_memory_instance())


def run_memory_write_for_memory_id(
    fn: Callable[[Any], T],
    memory_id: str,
    entity_scope: Optional[Dict[str, str]] = None,
    *,
    resolve_scope: bool = True,
) -> T:
    """Run ``fn(memory)`` under the per-``memory_id`` lock (optional scope ancestors).

    When *resolve_scope* is True and *entity_scope* is omitted, calls
    :func:`entity_scope_for_memory_id` once (``memory.get``) before locking.
    Set *resolve_scope* to False or pass *entity_scope* when the record is already
    loaded.
    """
    memory = get_memory_instance()
    scope = entity_scope
    if scope is None and resolve_scope:
        scope = entity_scope_for_memory_id(memory, memory_id)
    with memory_id_lock(memory_id, entity_scope=scope):
        return fn(memory)
