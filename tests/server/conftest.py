"""
Shared test configuration for server/* tests.

server/ modules use flat imports (``from auth import ...``,
``from compat.scope import ...``) because their runtime sys.path includes the
``server/`` directory.  Pytest runs from the repo root where only ``"."`` is in
pythonpath, so we register short-name aliases here once, before any test module
is imported.

- ``auth`` (flat) is a MagicMock so routers load without a DB.
- ``main`` is **not** imported here (module-level ``initialize_state``); load it
  lazily inside fixtures via ``import main`` after patching ``Memory.from_config``.
"""

import importlib
import sys
import warnings
from unittest.mock import MagicMock


def _register_alias(mod_path: str, alias: str) -> None:
    """Import *mod_path* (e.g. ``server.server_state``) and register it in
    ``sys.modules`` under the short *alias* (e.g. ``server_state``).

    Silently catches ``ImportError`` so that one missing dependency does not
    prevent unrelated aliases from being registered.
    """
    try:
        mod = importlib.import_module(mod_path)
        sys.modules.setdefault(alias, mod)
    except ImportError as exc:
        warnings.warn(f"Failed to register alias '{alias}' for '{mod_path}': {exc}", ImportWarning, stacklevel=2)


# -- auth: flat ``auth`` is a MagicMock for router imports. Bcrypt tests load
# the real module lazily via importlib (see TestBcryptHelpers).
sys.modules.setdefault("auth", MagicMock())

# -- Layer 0: no bare-import deps on other server modules --------------------
# utils.config must register before server_state, which imports it at module
# top level; otherwise _register_alias silently swallows the ImportError and
# leaves the server_state alias unset (breaking compat.entities/mcp_server).
for _mod_path, _alias in [
    ("server.errors", "errors"),
    ("server.utils", "utils"),
    ("server.utils.config", "utils.config"),
    ("server.server_state", "server_state"),
    ("server.rate_limit", "rate_limit"),
    ("server.schemas", "schemas"),
    ("server.telemetry", "telemetry"),
]:
    _register_alias(_mod_path, _alias)

# -- Layer 1: compat sub-modules (depend on errors, server_state) ------------
# Order matters: modules with no bare-import deps first, then dependents.
for _mod_path, _alias in [
    ("server.compat", "compat"),
    ("server.compat.utils", "compat.utils"),
    ("server.compat.helpers", "compat.helpers"),
    ("server.compat.scope", "compat.scope"),
    ("server.compat.requests", "compat.requests"),
    ("server.compat.responses", "compat.responses"),
    ("server.compat.decorators", "compat.decorators"),
    ("server.compat.events", "compat.events"),
    ("server.compat.entities", "compat.entities"),
    ("server.compat.tasks", "compat.tasks"),
]:
    _register_alias(_mod_path, _alias)

# -- Layer 2: depends on compat modules + Layer 0 ----------------------------
for _mod_path, _alias in [
    ("server.memory_lock", "memory_lock"),
    ("server.db", "db"),
    ("server.models", "models"),
    ("server.auth_config", "auth_config"),
    ("server.oidc", "oidc"),
    ("server.oidc_state", "oidc_state"),
]:
    _register_alias(_mod_path, _alias)

# -- Layer 3: depends on db, models, auth, compat ----------------------------
for _mod_path, _alias in [
    ("server.bg_tasks", "bg_tasks"),
    ("server.mcp_server", "mcp_server"),
    ("server.routers", "routers"),
    ("server.routers.compat", "routers.compat"),
    ("server.routers.oidc", "routers.oidc"),
    ("server.routers.auth", "routers.auth"),
]:
    _register_alias(_mod_path, _alias)
