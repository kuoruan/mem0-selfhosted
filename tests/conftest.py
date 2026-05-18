"""
Shared test configuration for server/* tests.

server/ modules use flat imports (``from auth import ...``,
``from compat.scope import ...``) because their runtime sys.path includes the
``server/`` directory.  Pytest runs from the repo root where only ``"."`` is in
pythonpath, so we register short-name aliases here once, before any test module
is imported.  Real compat sub-modules are wired in where possible; only
``auth``, ``server_state``, and ``compat.decorators`` (which depends on the
server-only ``errors`` module) are stubbed.
"""

import sys
from unittest.mock import MagicMock

try:
    import server.compat.scope
    import server.compat.responses

    sys.modules.setdefault("compat", sys.modules["server.compat"])
    sys.modules.setdefault("compat.scope", sys.modules["server.compat.scope"])
    sys.modules.setdefault("compat.responses", sys.modules["server.compat.responses"])

    # server_state: alias to the real module (no DB deps at import time).
    import server.server_state
    sys.modules.setdefault("server_state", sys.modules["server.server_state"])

    # auth: must stay a MagicMock — the real server.auth imports sqlalchemy/DB
    # at module level and cannot be imported outside a live server environment.
    sys.modules.setdefault("auth", MagicMock())

    # compat.entities needs compat.scope + server_state aliased first
    import server.compat.entities
    sys.modules.setdefault("compat.entities", sys.modules["server.compat.entities"])

    # compat.decorators depends on `errors` (server-only module); stub it with
    # an identity decorator so @upstream_guard leaves route callables intact.
    class _CompatDecorators:
        upstream_guard = staticmethod(lambda fn: fn)

    sys.modules.setdefault("compat.decorators", _CompatDecorators())

except ImportError:
    # fastapi not installed — server tests will be skipped via importorskip
    pass
