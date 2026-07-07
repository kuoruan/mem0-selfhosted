"""Shared configuration utilities for the mem0 server.

Provides helpers for loading and processing JSON configuration files,
including recursive environment-variable expansion.
"""

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def expand_env_vars(value: Any, *, raw_keys: set[str] | None = None) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` references in *value*.

    Walks dicts, lists, and strings.  Non-string scalars are returned
    unchanged.

    When *raw_keys* is provided, dict values whose key name (the final path
    segment) is in *raw_keys* are passed through unchanged — this prevents
    sensitive values that legitimately contain ``$`` (e.g. a ``client_secret``
    of ``pa$$word``) from being mangled by env-var expansion.
    """
    if isinstance(value, dict):
        return {
            k: (v if (raw_keys and k in raw_keys) else expand_env_vars(v, raw_keys=raw_keys))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [expand_env_vars(v, raw_keys=raw_keys) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


_TRUTHY_AFFIRMATIVE = {"1", "true", "yes", "on"}
_TRUTHY_NEGATIVE = {"0", "false", "no", "off", ""}
_IS_TRUTHY_WARNED: set[str] = set()


def is_truthy(value: str | bool | int | float | None) -> bool:
    """Return ``True`` if *value* looks like an affirmative flag.

    Accepts booleans directly.  Strings are lowercased and checked against
    ``{"1", "true", "yes", "on"}``.  Numbers are truthy when non-zero.
    ``None`` / empty / anything else returns ``False``.

    Unrecognized non-empty string values (e.g. ``"enabled"``) are treated as
    ``False`` but emit a one-time warning so a typo such as
    ``MEM0_TELEMETRY=enabled`` does not silently disable telemetry.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in _TRUTHY_AFFIRMATIVE:
            return True
        if lowered in _TRUTHY_NEGATIVE:
            return False
        # Unrecognized non-empty value — warn once per distinct value to avoid
        # high-frequency log spam while still surfacing silent misconfigurations.
        if lowered not in _IS_TRUTHY_WARNED:
            _IS_TRUTHY_WARNED.add(lowered)
            logger.warning(
                "Unrecognized boolean value %r treated as False. "
                "Use one of 1/true/yes/on (true) or 0/false/no/off (false).",
                value,
            )
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def load_json_config(
    path: str | Path,
    *,
    silent: bool = False,
    raw_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    """Load a JSON configuration file with environment-variable expansion.

    Parameters
    ----------
    path:
        Filesystem path to the JSON config file.
    silent:
        If ``True`` and the file cannot be loaded (missing, unreadable,
        or invalid JSON), return ``None`` instead of raising.  Warnings
        and errors are still logged.
    raw_keys:
        Optional set of leaf key names (e.g. ``{"client_secret"}``) whose
        values must be left untouched — no ``$VAR`` expansion is applied.
        Useful for secrets that may legitimately contain ``$`` characters.

    Returns
    -------
    dict | None
        Parsed (and env-expanded) config dictionary, or ``None`` when
        *silent* is ``True`` and loading fails.
    """
    path = Path(path)

    if not path.is_file():
        if silent:
            logger.warning("Config path %s does not exist or is not a file.", path)
            return None
        raise RuntimeError(f"Config file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        if silent:
            logger.error("Failed to load config from %s: %s", path, exc)
            return None
        raise RuntimeError(f"Failed to load config from {path}: {exc}") from exc

    if not isinstance(data, dict):
        if silent:
            logger.error("Config file %s must be a JSON object at the root.", path)
            return None
        raise RuntimeError(f"Config file {path} must be a JSON object at the root.")

    return expand_env_vars(data, raw_keys=raw_keys)


def merge_config(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *updates* into a copy of *base*.

    For keys present in both:
    - If both values are dicts, recurse.
    - Otherwise the *updates* value wins.
    """
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged
