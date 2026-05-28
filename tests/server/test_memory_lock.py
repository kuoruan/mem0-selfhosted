import threading

import pytest

import server.memory_lock as memory_lock
from server.memory_lock import (
    memory_id_lock,
    memory_id_lock_key,
    memory_scope_lock,
    run_memory_write,
    run_memory_write_for_memory_id,
    scope_lock_key,
)


def test_scope_lock_key_uses_sorted_field_order():
    assert scope_lock_key({"user_id": "u1", "agent_id": "a1"}) == (
        ("agent_id", "a1"),
        ("user_id", "u1"),
    )


def test_scope_lock_key_requires_at_least_one_field():
    with pytest.raises(ValueError, match="At least one entity scope"):
        scope_lock_key({})


def test_different_scopes_run_concurrently():
    order: list[str] = []
    barrier = threading.Barrier(2)

    def work(scope: dict[str, str], label: str) -> None:
        with memory_scope_lock(scope):
            order.append(f"{label}-start")
            barrier.wait(timeout=1)
            order.append(f"{label}-end")

    t1 = threading.Thread(target=work, args=({"user_id": "alice"}, "alice"))
    t2 = threading.Thread(target=work, args=({"user_id": "bob"}, "bob"))
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert "alice-start" in order
    assert "bob-start" in order
    assert order.index("alice-start") < order.index("bob-end")
    assert order.index("bob-start") < order.index("alice-end")


def test_memory_id_lock_key():
    assert memory_id_lock_key("abc-123") == (("memory_id", "abc-123"),)


def test_different_memory_ids_run_concurrently(monkeypatch):
    order: list[str] = []
    barrier = threading.Barrier(2)
    sentinel = object()
    monkeypatch.setattr("server.server_state.get_memory_instance", lambda: sentinel)

    def work(mid: str, label: str) -> None:
        with memory_id_lock(mid):
            order.append(f"{label}-start")
            barrier.wait(timeout=1)
            order.append(f"{label}-end")

    t1 = threading.Thread(target=work, args=("mem-a", "a"))
    t2 = threading.Thread(target=work, args=("mem-b", "b"))
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert "a-start" in order and "b-start" in order
    assert order.index("a-start") < order.index("b-end")
    assert order.index("b-start") < order.index("a-end")


def test_same_memory_id_serializes_via_run_memory_write_for_memory_id(monkeypatch):
    order: list[str] = []
    first_started = threading.Event()
    allow_first_finish = threading.Event()
    sentinel = object()
    monkeypatch.setattr("server.server_state.get_memory_instance", lambda: sentinel)

    def slow(_memory):
        order.append("start")
        if not first_started.is_set():
            first_started.set()
            allow_first_finish.wait(timeout=1)
        order.append("end")
        return None

    def worker():
        run_memory_write_for_memory_id(slow, "mem-1")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    assert first_started.wait(timeout=1)
    allow_first_finish.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert order == ["start", "end", "start", "end"]


def test_lock_record_reserved_before_acquire_survives_gc():
    """in_use must be bumped in _get_lock_record so GC cannot evict before acquire."""
    key = scope_lock_key({"user_id": "gc-survivor"})
    with memory_lock._registry_lock:
        memory_lock._locks.clear()
        memory_lock._lock_ttl.clear()

    record = memory_lock._get_lock_record(key)
    assert record.in_use >= 1

    with memory_lock._registry_lock:
        memory_lock._lock_ttl.pop(key, None)
        memory_lock._gc_locks()
        assert memory_lock._locks.get(key) is record

    with memory_lock._hold_record(record, key):
        pass

    with memory_lock._registry_lock:
        assert record.in_use == 0


def test_same_scope_serializes_writes(monkeypatch):
    order: list[str] = []
    first_started = threading.Event()
    allow_first_finish = threading.Event()
    sentinel = object()
    monkeypatch.setattr("server.server_state.get_memory_instance", lambda: sentinel)

    def slow_write(_memory):
        order.append("start")
        if not first_started.is_set():
            first_started.set()
            allow_first_finish.wait(timeout=1)
        order.append("end")
        return {"results": []}

    scope = {"user_id": "alice"}

    def worker():
        run_memory_write(slow_write, scope)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    assert first_started.wait(timeout=1)
    allow_first_finish.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert order == ["start", "end", "start", "end"]


def test_global_lock_blocks_scoped_and_memory_id_locks():
    """Global operations should serialize against all other write locks."""
    started_scoped = threading.Event()
    allow_scoped_finish = threading.Event()
    global_entered = threading.Event()
    global_exited = threading.Event()

    def scoped_worker():
        with memory_scope_lock({"user_id": "alice"}):
            started_scoped.set()
            allow_scoped_finish.wait(timeout=2)

    def global_worker():
        with memory_scope_lock(global_lock=True):
            global_entered.set()
        global_exited.set()

    t_scoped = threading.Thread(target=scoped_worker)
    t_scoped.start()
    assert started_scoped.wait(timeout=1)

    t_global = threading.Thread(target=global_worker)
    t_global.start()

    # The global lock should not be able to enter while a scoped lock is held.
    assert not global_entered.wait(timeout=0.2)

    allow_scoped_finish.set()
    t_scoped.join(timeout=2)

    assert global_entered.wait(timeout=1)
    t_global.join(timeout=2)
    assert global_exited.is_set()

    started_mem = threading.Event()
    allow_mem_finish = threading.Event()
    global2_entered = threading.Event()

    def mem_worker():
        with memory_id_lock("mem-1"):
            started_mem.set()
            allow_mem_finish.wait(timeout=2)

    def global_worker2():
        with memory_scope_lock(global_lock=True):
            global2_entered.set()

    t_mem = threading.Thread(target=mem_worker)
    t_mem.start()
    assert started_mem.wait(timeout=1)

    t_global2 = threading.Thread(target=global_worker2)
    t_global2.start()
    assert not global2_entered.wait(timeout=0.2)

    allow_mem_finish.set()
    t_mem.join(timeout=2)
    assert global2_entered.wait(timeout=1)
    t_global2.join(timeout=2)


def test_empty_scope_falls_back_to_global_lock():
    """Unscoped write operations should not raise and should behave like global lock."""
    started_scoped = threading.Event()
    allow_scoped_finish = threading.Event()
    empty_entered = threading.Event()

    def scoped_worker():
        with memory_scope_lock({"user_id": "alice"}):
            started_scoped.set()
            allow_scoped_finish.wait(timeout=2)

    def empty_scope_worker():
        with memory_scope_lock({}):
            empty_entered.set()

    t_scoped = threading.Thread(target=scoped_worker)
    t_scoped.start()
    assert started_scoped.wait(timeout=1)

    t_empty = threading.Thread(target=empty_scope_worker)
    t_empty.start()
    # Empty scope should behave like a global lock and thus block until scoped releases.
    assert not empty_entered.wait(timeout=0.2)

    allow_scoped_finish.set()
    t_scoped.join(timeout=2)
    assert empty_entered.wait(timeout=1)
    t_empty.join(timeout=2)
