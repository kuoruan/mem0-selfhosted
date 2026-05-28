import threading

import pytest

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
