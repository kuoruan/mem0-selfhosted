"""
Shared test configuration for server/* tests.

server/ modules use flat imports (``from auth import ...``,
``from compat.scope import ...``) because their runtime sys.path includes the
``server/`` directory.  Pytest runs from the repo root where only ``"."`` is in
pythonpath, so we register short-name aliases here once, before any test module
is imported.

- ``main`` is **not** imported here (module-level ``initialize_state``); load it
  lazily inside fixtures via ``importlib.import_module("server.main")`` after
  patching ``Memory.from_config``. Its flat imports resolve through the aliases
  registered below, so no ``sys.path`` mutation is needed.
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Single source for the test JWT secret. Used for the early env default below
# and by the ``load_app`` fixture.
TEST_JWT_SECRET = "test-jwt-secret-for-tests"

# Ensure AUTH_DISABLED and JWT_SECRET are set early, before any server module
# imports.  The CI workflow sets these at the job level, but make them explicit
# here so local test runs and CI are both self-contained.
os.environ.setdefault("AUTH_DISABLED", "true")
os.environ.setdefault("JWT_SECRET", TEST_JWT_SECRET)


def _stub_default_user(monkeypatch):
    """Ensure a default admin user exists AND stub ``_get_default_user`` to it.

    AUTH_DISABLED resolves the operator via ``_get_default_user(db)``; the shared
    CI/test DB has no users by default, so without one ``resolve_operator`` raises
    401 on every disabled request. The real user is the reliable mechanism (a
    pure mock stub proved flaky under CI); the monkeypatch is belt-and-suspenders
    so the operator resolves even if the DB lookup path is bypassed.
    """
    import server.auth
    from sqlalchemy import select
    from db import SessionLocal
    from models import User

    with SessionLocal() as sess:
        user = sess.scalar(select(User).where(User.email == "default@test.local"))
        if user is None:
            user = User(name="Default User", email="default@test.local", role="admin")
            sess.add(user)
            sess.commit()
            sess.refresh(user)

    monkeypatch.setattr(server.auth, "_get_default_user", lambda db: user)
    return user


@pytest.fixture
def default_user(monkeypatch):
    """AUTH_DISABLED operator stub. Requested by the MCP testbed (which builds a
    bare app without ``load_app``) and any other disabled-mode test."""
    return _stub_default_user(monkeypatch)


@pytest.fixture
def memory_patch():
    """Patch ``mem0.Memory.from_config`` + cache the mock in ``server_state`` so
    the server imports/loads without a real backend. Yields the mock; per-file
    ``_mock_memory`` fixtures customize its return values.

    ``server_state._memory_instance`` is set for the fixture duration so
    ``get_memory_instance()`` returns the mock regardless of whether ``main``
    is reloaded under the patch.
    """
    mock = MagicMock()
    import server.server_state as server_state

    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
        with patch("mem0.Memory.from_config", return_value=mock):
            previous = server_state._memory_instance
            server_state._memory_instance = mock
            try:
                yield mock
            finally:
                server_state._memory_instance = previous


@pytest.fixture
def load_app(monkeypatch):
    """Factory fixture: (re)load ``server/main.py`` under env overrides and
    return its FastAPI app.

    Auth config is injected directly: a stub ``AuthConfig`` is patched onto both
    ``auth.get_auth_config`` (read by ``verify_auth`` on the request path) and
    ``main.get_auth_config`` (read by ``_validate_auth_config`` in lifespan), so
    enabled-mode lifespan tests see the stub rather than the real env/cache.
    No env mutation or cache-clear is needed. In disabled mode the operator is
    additionally stubbed via ``_stub_default_user``.

    Usage::

        def test_x(self, load_app):
            app = load_app({"ADMIN_API_KEY": "..."})

    NOTE: ``reload_main=True`` re-runs ``initialize_state``, which calls
    ``Memory.from_config``. Tests using ``load_app`` must therefore also depend
    on ``memory_patch`` (or a per-file ``_mock_memory`` that wraps it) so the
    mock is installed before the reload — otherwise ``initialize_state`` tries
    to build a real Memory backend and fails.
    """

    def _load(env_overrides: dict, *, reload_main: bool = True, auth_disabled: bool | None = None):
        admin_api_key = env_overrides.get("ADMIN_API_KEY")
        if auth_disabled is None:
            # Empty ADMIN_API_KEY → disabled; set → enabled; absent → enabled.
            auth_disabled = admin_api_key == ""

        import auth
        from auth_config import AuthConfig

        cfg = AuthConfig(
            jwt_secret=TEST_JWT_SECRET,
            admin_api_key=admin_api_key or None,
            auth_disabled=auth_disabled,
        )
        monkeypatch.setattr(auth, "get_auth_config", lambda: cfg)

        if auth_disabled:
            # disabled mode resolves the operator via _get_default_user; stub it
            # (the shared test DB has no users).
            _stub_default_user(monkeypatch)

        server_main = importlib.import_module("server.main")

        if reload_main:
            importlib.reload(server_main)
        # Reload re-binds main.get_auth_config to the freshly imported
        # auth_config.get_auth_config, so patch AFTER the reload or it would be
        # discarded.
        monkeypatch.setattr(server_main, "get_auth_config", lambda: cfg)
        return server_main.app

    return _load


def _register_alias(mod_path: str, alias: str) -> None:
    """Import *mod_path* and alias it in ``sys.modules``; skip on ``ImportError``.

    ImportWarning is ignored by default, so surface the skip on stderr — a silent skip
    leaves a confusing ``ModuleNotFoundError`` at first use.
    """
    try:
        mod = importlib.import_module(mod_path)
        sys.modules.setdefault(alias, mod)
    except ImportError as exc:
        print(f"conftest: skipping alias '{alias}' for '{mod_path}': {exc}", file=sys.stderr)


# Server flat imports (``from auth import ...``) resolve against the ``server/``
# runtime sys.path. Pytest runs from the repo root, so register the short names
# once here, in dependency order, instead of mutating ``sys.path``.
# Each entry is exactly ``"server." + alias`` (``tests.server.helpers`` excepted).
for _mod_path, _alias in [
    ("server.errors", "errors"),
    ("server.entity", "entity"),
    ("server.utils", "utils"),
    ("server.utils.config", "utils.config"),
    ("server.utils.pagination", "utils.pagination"),
    ("server.server_state", "server_state"),
    # memory_lock only needs entity + server_state (both above). compat.tasks
    # flat-imports it, so it must be registered before the compat layer.
    ("server.memory_lock", "memory_lock"),
    ("server.rate_limit", "rate_limit"),
    ("server.schemas", "schemas"),
    ("server.telemetry", "telemetry"),
]:
    _register_alias(_mod_path, _alias)

# Compat sub-modules depend on the aliases above.
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

# Depends on compat + the layer above; order is load-bearing
# (entity_permissions does ``from auth import ...``).
for _mod_path, _alias in [
    ("server.db", "db"),
    ("server.models", "models"),
    ("server.auth_config", "auth_config"),
    ("server.auth", "auth"),
    ("server.entity_permissions", "entity_permissions"),
    ("server.oidc", "oidc"),
    ("server.oidc_state", "oidc_state"),
]:
    _register_alias(_mod_path, _alias)

# Depends on db, models, auth, and compat; ``helpers`` is a test module, not a server one.
for _mod_path, _alias in [
    ("server.bg_tasks", "bg_tasks"),
    ("server.mcp_server", "mcp_server"),
    ("server.routers", "routers"),
    ("server.routers.compat", "routers.compat"),
    ("server.routers.oidc", "routers.oidc"),
    ("server.routers.auth", "routers.auth"),
    # Regular Python module, not a server module; registered last.
    ("tests.server.helpers", "helpers"),
]:
    _register_alias(_mod_path, _alias)
