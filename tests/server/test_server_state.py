import json
from unittest.mock import Mock

import pytest

import server.server_state as server_state


def test_initialize_state_skips_missing_optional_config_file(monkeypatch):
    default_config = {"llm": {"provider": "openai", "config": {"api_key": "default"}}}
    memory_instance = object()
    from_config = Mock(return_value=memory_instance)

    monkeypatch.setattr(server_state.Memory, "from_config", from_config)
    monkeypatch.setattr(server_state, "_load_overrides", lambda: {})

    server_state.initialize_state(default_config, config_path="/path/that/does/not/exist.json")

    from_config.assert_called_once_with(default_config)
    assert server_state.get_current_config() == default_config
    assert server_state.get_memory_instance() is memory_instance


def test_load_config_file_expands_env_vars_only_in_leaf_strings(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"llm": {"config": {"api_key": "${BROKEN}"}}, "metadata": ["${BROKEN}", 1]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BROKEN", 'a"b')

    loaded_config = server_state._load_config_file(str(config_path))

    assert loaded_config == {"llm": {"config": {"api_key": 'a"b'}}, "metadata": ['a"b', 1]}


def test_load_config_file_raises_runtime_error_for_invalid_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"llm": ', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid JSON in mem0 config file"):
        server_state._load_config_file(str(config_path))


def test_load_config_file_raises_runtime_error_for_non_object_root(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a JSON object at the root"):
        server_state._load_config_file(str(config_path))


def test_load_config_file_raises_runtime_error_for_unreadable_file(monkeypatch):
    def raise_os_error(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", raise_os_error)

    with pytest.raises(RuntimeError, match="Failed to read mem0 config file"):
        server_state._load_config_file("/tmp/config.json")


def test_list_all_memories_handles_tuple_vector_store_shape(monkeypatch):
    class Row:
        def __init__(self, row_id, payload):
            self.id = row_id
            self.payload = payload

    class VectorStore:
        def list(self, top_k):
            return ([Row("m1", {"data": "hello", "user_id": "u1"})], "next-token")

    class MemoryInstance:
        vector_store = VectorStore()

    monkeypatch.setattr(server_state, "get_memory_instance", lambda: MemoryInstance())

    result = server_state.list_all_memories()

    assert result == {
        "results": [
            {
                "id": "m1",
                "memory": "hello",
                "user_id": "u1",
                "agent_id": None,
                "app_id": None,
                "run_id": None,
                "hash": None,
                "expiration_date": None,
                "metadata": {},
                "created_at": None,
                "updated_at": None,
            }
        ]
    }


def test_serialize_memory_promotes_app_id_out_of_metadata():
    class Row:
        id = "m1"
        payload = {
            "data": "hello",
            "user_id": "u1",
            "agent_id": "a1",
            "app_id": "app1",
            "run_id": "r1",
            "kind": "note",
        }

    result = server_state.serialize_memory(Row())

    assert result["app_id"] == "app1"
    assert result["metadata"] == {"kind": "note"}


def test_list_all_memories_hides_expired_by_default(monkeypatch):
    class Row:
        def __init__(self, row_id, payload):
            self.id = row_id
            self.payload = payload

    class VectorStore:
        def list(self, top_k):
            return [
                Row("active", {"data": "hello", "user_id": "u1", "expiration_date": "2999-01-01"}),
                Row("expired", {"data": "old", "user_id": "u1", "expiration_date": "2000-01-01"}),
            ]

        def col_info(self):
            return {"count": 2}

    class MemoryInstance:
        vector_store = VectorStore()

    monkeypatch.setattr(server_state, "get_memory_instance", lambda: MemoryInstance())

    result = server_state.list_all_memories(limit=None)
    ids = [item["id"] for item in result["results"]]
    assert ids == ["active"]


def test_list_all_memories_uses_object_store_count_when_available(monkeypatch):
    class Row:
        id = "active"
        payload = {"data": "hello", "user_id": "u1"}

    class CollectionInfo:
        points_count = 12_000

    class VectorStore:
        def __init__(self):
            self.top_k = None

        def list(self, top_k):
            self.top_k = top_k
            return [Row()]

        def col_info(self):
            return CollectionInfo()

    vector_store = VectorStore()

    class MemoryInstance:
        pass

    memory_instance = MemoryInstance()
    memory_instance.vector_store = vector_store

    monkeypatch.setattr(server_state, "get_memory_instance", lambda: memory_instance)

    result = server_state.list_all_memories(limit=None)

    assert vector_store.top_k == 12_000
    assert result["results"][0]["id"] == "active"


def test_list_all_memories_includes_expired_when_requested(monkeypatch):
    class Row:
        def __init__(self, row_id, payload):
            self.id = row_id
            self.payload = payload

    class VectorStore:
        def list(self, top_k):
            return [
                Row("active", {"data": "hello", "user_id": "u1", "expiration_date": "2999-01-01"}),
                Row("expired", {"data": "old", "user_id": "u1", "expiration_date": "2000-01-01"}),
            ]

        def col_info(self):
            return {"count": 2}

    class MemoryInstance:
        vector_store = VectorStore()

    monkeypatch.setattr(server_state, "get_memory_instance", lambda: MemoryInstance())

    result = server_state.list_all_memories(limit=None, show_expired=True)
    ids = [item["id"] for item in result["results"]]
    assert ids == ["active", "expired"]
