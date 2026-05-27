"""
Shared test configuration for server/* tests.

server/ modules use flat imports (``from auth import ...``,
``from compat.scope import ...``) because their runtime sys.path includes the
``server/`` directory.  Pytest runs from the repo root where only ``"."`` is in
pythonpath, so we register short-name aliases here once, before any test module
is imported.  Only ``auth`` is stubbed (MagicMock); all other modules are
aliased from their real ``server.`` package paths.
"""

import sys
from unittest.mock import MagicMock

try:
    # -- Layer 0: no flat-import deps ------------------------------------------
    import server.errors as _errors

    sys.modules.setdefault("errors", _errors)

    import server.server_state as _server_state

    sys.modules.setdefault("server_state", _server_state)

    import server.memory_lock as _memory_lock

    sys.modules.setdefault("memory_lock", _memory_lock)

    import server.rate_limit as _rate_limit

    sys.modules.setdefault("rate_limit", _rate_limit)

    import server.schemas as _schemas

    sys.modules.setdefault("schemas", _schemas)

    import server.telemetry as _telemetry

    sys.modules.setdefault("telemetry", _telemetry)

    # -- auth: must stay a MagicMock — real module needs sqlalchemy/DB ----------
    sys.modules.setdefault("auth", MagicMock())

    # -- Layer 1: compat sub-modules (depend on errors, server_state) ----------
    import server.compat as _compat
    import server.compat.responses as _compat_responses
    import server.compat.scope as _compat_scope

    sys.modules.setdefault("compat", _compat)
    sys.modules.setdefault("compat.scope", _compat_scope)
    sys.modules.setdefault("compat.responses", _compat_responses)

    import server.compat.entities as _compat_entities

    sys.modules.setdefault("compat.entities", _compat_entities)

    import server.compat.decorators as _compat_decorators

    sys.modules.setdefault("compat.decorators", _compat_decorators)

    # -- Layer 2: db / models (models imports db at module level) --------------
    import server.db as _db

    sys.modules.setdefault("db", _db)

    import server.models as _models

    sys.modules.setdefault("models", _models)

    # -- Layer 3: depends on db, models, auth, compat --------------------------
    import server.bg_tasks as _bg_tasks

    sys.modules.setdefault("bg_tasks", _bg_tasks)

    import server.mcp_server as _mcp_server

    sys.modules.setdefault("mcp_server", _mcp_server)

    import server.routers as _routers

    sys.modules.setdefault("routers", _routers)

except ImportError:
    # fastapi not installed — server tests will be skipped via importorskip
    pass
