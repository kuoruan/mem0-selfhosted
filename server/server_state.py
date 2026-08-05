import json
import logging
import os
import threading
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict

from utils.config import load_json_config, merge_config
from utils.helpers import normalize_vector_store_list, safe_count
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


def initialize_state(default_config: Dict[str, Any], config_path: str | None = None) -> None:
    global _current_config, _memory_instance
    with _state_lock:
        _current_config = deepcopy(default_config)
        if config_path:
            if os.path.exists(config_path):
                file_overrides = load_json_config(config_path)
                if file_overrides:
                    _current_config = merge_config(_current_config, file_overrides)
                    logging.info("Loaded mem0 config overrides from %s", config_path)
            else:
                logging.warning("MEM0_CONFIG_PATH set but file not found: %s", config_path)
        overrides = _load_overrides()
        if overrides:
            _current_config = merge_config(_current_config, overrides)
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
    from memory_lock import memory_scope_lock

    global _current_config, _memory_instance
    with memory_scope_lock(global_lock=True):
        with _state_lock:
            next_config = merge_config(_current_config, updates)
            if _config_effectively_changed(_current_config, next_config):
                _memory_instance = Memory.from_config(next_config)
            _current_config = next_config
            overrides = _load_overrides()
            overrides = merge_config(overrides, updates)
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
_RESERVED_PAYLOAD_KEYS = {
    "data",
    "user_id",
    "agent_id",
    "app_id",
    "run_id",
    "hash",
    "created_at",
    "updated_at",
    "expiration_date",
}


def serialize_memory(row: Any) -> Dict[str, Any]:
    payload = getattr(row, "payload", None) or {}
    return {
        "id": getattr(row, "id", None),
        "memory": payload.get("data"),
        "user_id": payload.get("user_id"),
        "agent_id": payload.get("agent_id"),
        "app_id": payload.get("app_id"),
        "run_id": payload.get("run_id"),
        "hash": payload.get("hash"),
        "expiration_date": payload.get("expiration_date"),
        "metadata": {k: v for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS},
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _expiration_date_is_expired(expiration_date: Any) -> bool:
    """Return whether an expiration date has passed.

    Mirrors the OSS Memory SDK behavior for admin all-memory listing without
    depending on private SDK helpers. Invalid or missing dates are treated as
    active so malformed legacy payloads do not break admin reads.
    """
    if not expiration_date:
        return False
    try:
        parsed_date = date.fromisoformat(str(expiration_date))
    except ValueError:
        return False
    return parsed_date < datetime.now(timezone.utc).date()


def list_all_memories(limit: int | None = ALL_MEMORIES_LIMIT, show_expired: bool | None = None) -> Dict[str, Any]:
    memory = get_memory_instance()
    if limit is not None:
        top_k = limit
    else:
        c = safe_count(memory)
        top_k = c if c and c > 0 else ALL_MEMORIES_LIMIT

    # Direct vector_store.list: admin listing has no entity scope, so Memory.get_all()
    # cannot be used (it enforces scope validation). See show_expired note below.
    rows = normalize_vector_store_list(memory.vector_store.list(top_k=top_k))
    memories = [serialize_memory(row) for row in rows]

    if show_expired is not True:
        # Admin all-memory listing has no entity scope, so it cannot call
        # Memory.get_all(show_expired=...) without tripping SDK scope validation.
        memories = [m for m in memories if not _expiration_date_is_expired(m.get("expiration_date"))]
    return {"results": memories}
