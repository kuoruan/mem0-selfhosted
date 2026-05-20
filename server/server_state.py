import json
import logging
import os
import threading
from copy import deepcopy
from typing import Any, Callable, Dict

from mem0 import Memory

_state_lock = threading.RLock()
_current_config: Dict[str, Any] = {}
_memory_instance: Memory | None = None
_session_factory: Callable | None = None


def set_session_factory(factory: Callable) -> None:
    global _session_factory
    _session_factory = factory


def _load_overrides() -> Dict[str, Any]:
    try:
        if _session_factory is None:
            return {}
        from models import Settings

        with _session_factory() as session:
            row = session.get(Settings, "config_overrides")
            if row is None:
                return {}
            return json.loads(row.value)
    except Exception:
        logging.exception("Failed to load config overrides from database")
        return {}


def _save_overrides(overrides: Dict[str, Any]) -> None:
    try:
        if _session_factory is None:
            return
        from models import Settings
        from sqlalchemy.dialects.postgresql import insert

        with _session_factory() as session:
            serialized = json.dumps(overrides)
            stmt = (
                insert(Settings)
                .values(key="config_overrides", value=serialized)
                .on_conflict_do_update(
                    index_elements=[Settings.key],
                    set_={"value": serialized},
                )
            )
            session.execute(stmt)
            session.commit()
    except Exception:
        logging.warning("Failed to persist config overrides to database", exc_info=True)


def _merge_config(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item_value) for key, item_value in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item_value) for item_value in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_config_file(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, encoding="utf-8") as config_file:
            raw_config = config_file.read()
    except OSError as exc:
        raise RuntimeError(f"Failed to read mem0 config file '{config_path}': {exc}") from exc

    try:
        loaded_config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in mem0 config file '{config_path}': {exc}") from exc

    if not isinstance(loaded_config, dict):
        raise RuntimeError(f"Mem0 config file '{config_path}' must be a JSON object at the root.")

    return _expand_env_vars(loaded_config)


def initialize_state(default_config: Dict[str, Any], config_path: str | None = None) -> None:
    global _current_config, _memory_instance
    with _state_lock:
        _current_config = deepcopy(default_config)
        if config_path:
            if os.path.exists(config_path):
                file_overrides = _load_config_file(config_path)
                if file_overrides:
                    _current_config = _merge_config(_current_config, file_overrides)
                    logging.info("Loaded mem0 config overrides from %s", config_path)
            else:
                logging.warning("MEM0_CONFIG_PATH set but file not found: %s", config_path)
        overrides = _load_overrides()
        if overrides:
            _current_config = _merge_config(_current_config, overrides)
        _memory_instance = Memory.from_config(_current_config)


def _config_effectively_changed(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Compare the parts of config that actually affect Memory initialization.

    Only llm, embedder, vector_store, and history_db_path require a restart.
    Changes to top-level keys like 'version' are ignored."""
    _REBUILD_KEYS = {"llm", "embedder", "vector_store", "history_db_path"}
    for key in _REBUILD_KEYS:
        if old.get(key) != new.get(key):
            return True
    return False


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    global _current_config, _memory_instance
    with _state_lock:
        next_config = _merge_config(_current_config, updates)
        if _config_effectively_changed(_current_config, next_config):
            _memory_instance = Memory.from_config(next_config)
        _current_config = next_config
        overrides = _load_overrides()
        overrides = _merge_config(overrides, updates)
        _save_overrides(overrides)
        return deepcopy(_current_config)


def get_current_config() -> Dict[str, Any]:
    with _state_lock:
        return deepcopy(_current_config)


def get_memory_instance() -> Memory:
    with _state_lock:
        if _memory_instance is None:
            raise RuntimeError("Mem0 runtime has not been initialized.")
        return _memory_instance


ALL_MEMORIES_LIMIT = 1000
_RESERVED_PAYLOAD_KEYS = {"data", "user_id", "agent_id", "run_id", "hash", "created_at", "updated_at"}


def serialize_memory(row: Any) -> Dict[str, Any]:
    payload = getattr(row, "payload", None) or {}
    return {
        "id": getattr(row, "id", None),
        "memory": payload.get("data"),
        "user_id": payload.get("user_id"),
        "agent_id": payload.get("agent_id"),
        "run_id": payload.get("run_id"),
        "hash": payload.get("hash"),
        "metadata": {k: v for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS},
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def list_all_memories(limit: int = ALL_MEMORIES_LIMIT) -> Dict[str, Any]:
    results = get_memory_instance().vector_store.list(top_k=limit)
    if not results:
        rows = []
    elif isinstance(results, tuple):
        rows = results[0] if isinstance(results[0], list) else []
    elif isinstance(results, list) and results and isinstance(results[0], list):
        rows = results[0]
    else:
        rows = results
    return {"results": [serialize_memory(row) for row in rows]}
