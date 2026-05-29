import threading

import pytest

import memory_lock
from memory_lock import (
    LOCK_ACQUIRE_ORDER,
    memory_id_lock,
    memory_id_lock_key,
    memory_id_lock_keys,
    memory_scope_lock,
    run_memory_write,
    run_memory_write_for_memory_id,
    scope_lock_key,
    scope_lock_keys,
)


def test_scope_lock_key_uses_sorted_field_order():
    assert scope_lock_key({"user_id": "u1", "agent_id": "a1"}) == (
        ("agent_id", "a1"),
        ("user_id", "u1"),
    )


def test_scope_lock_key_requires_at_least_one_field():
    with pytest.raises(ValueError, match="At least one entity scope"):
        scope_lock_key({})


def test_lock_acquire_order_constant():
    assert LOCK_ACQUIRE_ORDER == ("user_id", "agent_id", "app_id", "run_id", "memory_id")


def test_scope_lock_keys_follow_acquire_order():
    keys = scope_lock_keys(
        {"user_id": "u", "agent_id": "a", "app_id": "p", "run_id": "r"},
    )
    assert [key[0][0] for key in keys] == ["user_id", "agent_id", "app_id", "run_id"]


def test_scope_lock_key_includes_app_id():
    assert scope_lock_key({"user_id": "u1", "app_id": "app-1"}) == (
        ("app_id", "app-1"),
        ("user_id", "u1"),
    )


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


def test_memory_id_lock_keys_follow_acquire_order():
    keys = memory_id_lock_keys(
        "mem-1",
        {"user_id": "u", "agent_id": "a", "app_id": "p", "run_id": "r"},
    )
    assert [key[0][0] for key in keys] == ["user_id", "agent_id", "app_id", "run_id", "memory_id"]


def test_different_memory_ids_run_concurrently(monkeypatch):
    order: list[str] = []
    barrier = threading.Barrier(2)
    sentinel = object()
    monkeypatch.setattr("memory_lock.get_memory_instance", lambda: sentinel)

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
    monkeypatch.setattr("memory_lock.get_memory_instance", lambda: sentinel)

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
    monkeypatch.setattr("memory_lock.get_memory_instance", lambda: sentinel)

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


def test_successful_write_does_not_double_decrement_pending_writers():
    """Releasing one writer must not clear another writer's pending slot."""
    gate = memory_lock._RWGate()
    with gate._cond:
        gate._pending_writers = 1  # simulate another writer already waiting

    with gate.write():
        with gate._cond:
            assert gate._pending_writers == 1
            assert gate._writer

    with gate._cond:
        assert gate._pending_writers == 1
        assert not gate._writer


def test_write_clears_pending_writers_when_wait_aborts():
    """If write() aborts before acquiring, pending must not block readers forever."""
    gate = memory_lock._RWGate()
    gate._readers = 1

    original_wait = gate._cond.wait

    def abort_wait() -> None:
        raise RuntimeError("abort wait")

    gate._cond.wait = abort_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="abort wait"):
        with gate.write():
            pass
    gate._cond.wait = original_wait  # type: ignore[method-assign]

    entered = threading.Event()

    def reader() -> None:
        with gate.read():
            entered.set()

    t = threading.Thread(target=reader)
    t.start()
    assert entered.wait(timeout=1)
    t.join(timeout=1)


def test_global_gate_writer_not_starved_by_continuous_readers():
    """Pending global writes block new readers so write() eventually runs."""
    gate = memory_lock._RWGate()
    global_done = threading.Event()
    stop_readers = threading.Event()

    def reader() -> None:
        while not stop_readers.is_set():
            with gate.read():
                pass

    def global_write() -> None:
        with gate.write():
            global_done.set()

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    for thread in readers:
        thread.start()

    writer = threading.Thread(target=global_write)
    writer.start()
    assert global_done.wait(timeout=2)

    stop_readers.set()
    for thread in readers:
        thread.join(timeout=1)
    writer.join(timeout=1)


def test_memory_id_lock_with_entity_scope_blocks_delete_all(monkeypatch):
    """Per-id write with user scope blocks concurrent delete_all for that user."""
    order: list[str] = []
    update_started = threading.Event()
    allow_update_finish = threading.Event()
    monkeypatch.setattr("memory_lock.get_memory_instance", lambda: object())

    def slow_update(_memory) -> None:
        order.append("update-start")
        update_started.set()
        allow_update_finish.wait(timeout=1)
        order.append("update-end")

    def delete_all_worker() -> None:
        run_memory_write(lambda _memory: order.append("delete-all"), {"user_id": "alice"})

    t_update = threading.Thread(
        target=lambda: run_memory_write_for_memory_id(
            slow_update, "mem-1", entity_scope={"user_id": "alice"}, resolve_scope=False
        )
    )
    t_delete = threading.Thread(target=delete_all_worker)
    t_update.start()
    t_delete.start()
    assert update_started.wait(timeout=1)
    assert "delete-all" not in order
    allow_update_finish.set()
    t_update.join(timeout=2)
    t_delete.join(timeout=2)
    assert order == ["update-start", "update-end", "delete-all"]


def test_concurrent_add_user_and_user_agent_serialize(monkeypatch):
    """run_memory_write on overlapping scopes must not overlap."""
    order: list[str] = []
    gate = threading.Event()
    release = threading.Event()
    monkeypatch.setattr("memory_lock.get_memory_instance", lambda: object())

    def slow_add(_memory) -> None:
        order.append("add-start")
        gate.set()
        release.wait(timeout=1)
        order.append("add-end")

    def user_only() -> None:
        run_memory_write(slow_add, {"user_id": "alice"})

    def user_agent() -> None:
        run_memory_write(slow_add, {"user_id": "alice", "agent_id": "bot"})

    t1 = threading.Thread(target=user_only)
    t2 = threading.Thread(target=user_agent)
    t1.start()
    t2.start()
    assert gate.wait(timeout=1)
    assert "add-start" in order and order.count("add-start") == 1
    release.set()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert order.count("add-start") == 2
    assert order.count("add-end") == 2


def test_lock_record_in_use_returns_to_zero_after_repeated_acquire():
    key = (("user_id", "stress-user"),)
    with memory_lock._registry_lock:
        memory_lock._locks.clear()
        memory_lock._lock_ttl.clear()

    for _ in range(100):
        record = memory_lock._get_lock_record(key)
        with memory_lock._hold_record(record, key):
            pass

    with memory_lock._registry_lock:
        assert memory_lock._locks[key].in_use == 0


def test_resolve_scope_loads_memory_once(monkeypatch):
    calls: list[str] = []
    instance_ids: list[int] = []

    class _Memory:
        def get(self, memory_id: str):
            calls.append(memory_id)
            return {"id": memory_id, "user_id": "alice"}

    mem = _Memory()

    def _get_memory():
        instance_ids.append(id(mem))
        return mem

    monkeypatch.setattr("memory_lock.get_memory_instance", _get_memory)

    run_memory_write_for_memory_id(lambda _m: None, "mem-1", resolve_scope=True)
    assert calls == ["mem-1"]
    assert len(instance_ids) == 1


def test_user_scope_blocks_user_plus_agent_scope():
    """Ancestor user_id lock serializes user-only and user+agent writers."""
    order: list[str] = []
    first_started = threading.Event()
    allow_first_finish = threading.Event()

    def user_only() -> None:
        with memory_scope_lock({"user_id": "alice"}):
            order.append("user-start")
            if not first_started.is_set():
                first_started.set()
                allow_first_finish.wait(timeout=1)
            order.append("user-end")

    def user_and_agent() -> None:
        with memory_scope_lock({"user_id": "alice", "agent_id": "a1"}):
            order.append("ua-start")
            order.append("ua-end")

    t1 = threading.Thread(target=user_only)
    t2 = threading.Thread(target=user_and_agent)
    t1.start()
    t2.start()
    assert first_started.wait(timeout=1)
    allow_first_finish.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert order == ["user-start", "user-end", "ua-start", "ua-end"]


def test_empty_scope_requires_explicit_global_lock():
    """Empty scope without global_lock must fail fast."""
    with pytest.raises(ValueError, match="At least one entity scope"):
        with memory_scope_lock({}):
            pass


def test_empty_scope_with_global_lock_blocks_scoped_writes():
    started_scoped = threading.Event()
    allow_scoped_finish = threading.Event()
    empty_entered = threading.Event()

    def scoped_worker():
        with memory_scope_lock({"user_id": "alice"}):
            started_scoped.set()
            allow_scoped_finish.wait(timeout=2)

    def empty_scope_worker():
        with memory_scope_lock({}, global_lock=True):
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
