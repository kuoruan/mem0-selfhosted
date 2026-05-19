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
