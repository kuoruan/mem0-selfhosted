"""E2E tests for REST API server authentication.

Tests the actual server/main.py app through FastAPI's TestClient (full ASGI
round-trip) covering:
  - Auth disabled mode (ADMIN_API_KEY unset)
  - Auth enabled mode (ADMIN_API_KEY set)
  - Edge cases: empty keys, near-miss keys, timing-safe comparison, header
    casing, response headers, startup logging, and full CRUD flows through auth.

These are integration tests against the real server DB (Postgres in CI): the
auth-enabled path creates real User + APIKey rows, and the auth-disabled path
resolves the default user from the DB. The mem0 Memory backend is mocked.
"""

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient

from auth_config import AuthConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_memory(memory_patch):
    """Configure the mocked Memory with realistic CRUD return values."""
    memory_patch.get.return_value = {"id": "mem-1", "memory": "test memory", "user_id": "alice"}
    memory_patch.get_all.return_value = [
        {"id": "mem-1", "memory": "test memory", "user_id": "alice"},
    ]
    memory_patch.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    memory_patch.search.return_value = [{"id": "mem-1", "memory": "test", "score": 0.9}]
    memory_patch.update.return_value = {"message": "Memory updated"}
    memory_patch.history.return_value = [{"id": "mem-1", "old_memory": "a", "new_memory": "b"}]
    memory_patch.delete.return_value = None
    memory_patch.delete_all.return_value = {"message": "Memories deleted successfully!"}
    memory_patch.reset.return_value = None
    yield memory_patch


@pytest.fixture
def user_api_key():
    """Create a real dashboard admin + APIKey in the test DB; yield the full key.

    ADMIN_API_KEY is the bootstrap (read-only) principal under the new auth
    model, so memory write endpoints need a real user's key. The APIKey row is
    deleted on teardown to avoid accumulation across tests. (The admin User is
    reused — a partial unique index enforces at-most-one admin row.)"""
    import uuid
    from sqlalchemy import delete, select
    from auth import generate_api_key
    from db import SessionLocal
    from models import APIKey, User

    with SessionLocal() as sess:
        admin = sess.scalar(select(User).where(User.role == "admin").order_by(User.created_at.asc()))
        if admin is None:
            admin = User(name="auth-test", email=f"auth-{uuid.uuid4().hex[:8]}@example.com", role="admin")
            sess.add(admin)
            sess.flush()
        full, prefix, key_hash = generate_api_key()
        sess.add(APIKey(key_prefix=prefix, key_hash=key_hash, label="auth-test", created_by=admin.id))
        sess.commit()

    yield full

    with SessionLocal() as sess:
        sess.execute(delete(APIKey).where(APIKey.key_prefix == prefix))
        sess.commit()


# ---------------------------------------------------------------------------
# Auth disabled (ADMIN_API_KEY not set)
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    """All endpoints should be freely accessible when ADMIN_API_KEY is empty."""

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory, load_app):
        self.app = load_app({"ADMIN_API_KEY": ""})
        self.client = TestClient(self.app)
        self.mock = _mock_memory

    def test_root_redirects_to_docs(self):
        resp = self.client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert "/docs" in resp.headers["location"]

    def test_get_memory_without_key(self):
        resp = self.client.get("/memories/mem-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "mem-1"

    def test_get_all_memories_without_key(self):
        resp = self.client.get("/memories", params={"user_id": "alice"})
        assert resp.status_code == 200

    def test_create_memory_without_key(self):
        resp = self.client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "alice",
        })
        assert resp.status_code == 200

    def test_search_without_key(self):
        resp = self.client.post("/search", json={"query": "pizza", "user_id": "alice"})
        assert resp.status_code == 200

    def test_update_memory_without_key(self):
        resp = self.client.put("/memories/mem-1", json={"text": "updated"})
        assert resp.status_code == 200

    def test_history_without_key(self):
        resp = self.client.get("/memories/mem-1/history")
        assert resp.status_code == 200

    def test_delete_memory_without_key(self):
        resp = self.client.delete("/memories/mem-1")
        assert resp.status_code == 200

    def test_delete_all_without_key(self):
        resp = self.client.delete("/memories", params={"user_id": "alice"})
        assert resp.status_code == 200

    def test_reset_without_key(self):
        resp = self.client.post("/reset")
        assert resp.status_code == 200

    def test_configure_without_key(self):
        self.mock.from_config = MagicMock()
        resp = self.client.post("/configure", json={"version": "v1.1"})
        assert resp.status_code == 200

    def test_unrecognized_key_rejected_even_when_auth_disabled(self):
        """When auth is disabled, an unrecognized X-API-Key is still rejected —
        verify_auth validates any supplied key rather than silently accepting it."""
        resp = self.client.get(
            "/memories/mem-1", headers={"X-API-Key": "some-random-key"}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/configure"),
            ("POST", "/memories"),
            ("GET", "/memories"),
            ("GET", "/memories/test-id"),
            ("POST", "/search"),
            ("PUT", "/memories/test-id"),
            ("GET", "/memories/test-id/history"),
            ("DELETE", "/memories/test-id"),
            ("DELETE", "/memories"),
            ("POST", "/reset"),
        ],
    )
    def test_no_endpoint_returns_401_when_auth_disabled(self, method, path):
        resp = self.client.request(method, path)
        assert resp.status_code != 401, f"{method} {path} should not require auth"


# ---------------------------------------------------------------------------
# Auth enabled (ADMIN_API_KEY set)
# ---------------------------------------------------------------------------

class TestAuthEnabled:
    """All protected endpoints must enforce the API key."""

    API_KEY = "test-secret-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory, load_app, user_api_key):
        self.app = load_app({"ADMIN_API_KEY": self.API_KEY})
        self.client = TestClient(self.app)
        self.mock = _mock_memory
        # ADMIN_API_KEY is bootstrap (read-only); a real user key is needed for
        # write endpoints (create/update/delete memory).
        self.USER_API_KEY = user_api_key

    def _authed_real(self, method, path, **kwargs):
        """Send a request authenticated as the real dashboard user (write access)."""
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.USER_API_KEY
        return self.client.request(method, path, headers=headers, **kwargs)

    # --- Rejection cases ---

    def test_missing_key_returns_401(self):
        resp = self.client.get("/memories/mem-1")
        assert resp.status_code == 401

    def test_missing_key_detail_mentions_header(self):
        resp = self.client.get("/memories/mem-1")
        assert "X-API-Key" in resp.json()["detail"]

    def test_wrong_key_returns_401(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_wrong_key_detail_says_invalid(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": "wrong"})
        assert "Invalid" in resp.json()["detail"]

    def test_empty_string_key_returns_401(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_401_includes_www_authenticate_header(self):
        resp = self.client.get("/memories/mem-1")
        # Both accepted schemes are advertised so the client knows which
        # challenge to answer (Bearer JWT or X-API-Key).
        challenge = resp.headers.get("www-authenticate", "")
        assert "Bearer" in challenge and "ApiKey" in challenge

    def test_near_miss_key_rejected(self):
        """Key that differs by one character should be rejected."""
        near_miss = self.API_KEY[:-1] + ("6" if self.API_KEY[-1] != "6" else "7")
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": near_miss})
        assert resp.status_code == 401

    def test_key_with_extra_whitespace_rejected(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": f" {self.API_KEY} "})
        assert resp.status_code == 401

    def test_key_prefix_rejected(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": self.API_KEY[:5]})
        assert resp.status_code == 401

    def test_key_with_different_case_rejected(self):
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": self.API_KEY.upper()})
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/configure"),
            ("POST", "/memories"),
            ("GET", "/memories"),
            ("GET", "/memories/test-id"),
            ("POST", "/search"),
            ("PUT", "/memories/test-id"),
            ("GET", "/memories/test-id/history"),
            ("DELETE", "/memories/test-id"),
            ("DELETE", "/memories"),
            ("POST", "/reset"),
        ],
    )
    def test_all_endpoints_reject_without_key(self, method, path):
        resp = self.client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} should require auth"

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/configure"),
            ("POST", "/memories"),
            ("GET", "/memories"),
            ("GET", "/memories/test-id"),
            ("POST", "/search"),
            ("PUT", "/memories/test-id"),
            ("GET", "/memories/test-id/history"),
            ("DELETE", "/memories/test-id"),
            ("DELETE", "/memories"),
            ("POST", "/reset"),
        ],
    )
    def test_all_endpoints_reject_wrong_key(self, method, path):
        resp = self.client.request(method, path, headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401, f"{method} {path} should reject wrong key"

    # --- Acceptance cases ---

    def test_root_does_not_require_key(self):
        resp = self.client.get("/", follow_redirects=False)
        assert resp.status_code == 307

    def _authed(self, method, path, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.API_KEY
        return self.client.request(method, path, headers=headers, **kwargs)

    def test_get_memory_with_key(self):
        resp = self._authed("GET", "/memories/mem-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "mem-1"

    def test_get_memory_with_authorization_token(self):
        resp = self.client.get(
            "/memories/mem-1", headers={"Authorization": f"Token {self.API_KEY}"}
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "mem-1"

    def test_get_all_memories_with_key(self):
        resp = self._authed("GET", "/memories", params={"user_id": "alice"})
        assert resp.status_code == 200

    def test_create_memory_with_key(self):
        resp = self._authed_real("POST", "/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "alice",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_search_with_key(self):
        resp = self._authed("POST", "/search", json={"query": "pizza", "user_id": "alice"})
        assert resp.status_code == 200

    def test_update_memory_with_key(self):
        resp = self._authed_real("PUT", "/memories/mem-1", json={"text": "updated"})
        assert resp.status_code == 200

    def test_history_with_key(self):
        resp = self._authed("GET", "/memories/mem-1/history")
        assert resp.status_code == 200

    def test_delete_memory_with_key(self):
        resp = self._authed_real("DELETE", "/memories/mem-1")
        assert resp.status_code == 200

    def test_delete_all_with_key(self):
        resp = self._authed_real("DELETE", "/memories", params={"user_id": "alice"})
        assert resp.status_code == 200

    def test_reset_with_key(self):
        resp = self._authed("POST", "/reset")
        assert resp.status_code == 200

    def test_configure_with_key(self):
        resp = self._authed("POST", "/configure", json={"version": "v1.1"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full CRUD flow through auth
# ---------------------------------------------------------------------------

class TestAuthenticatedCRUDFlow:
    """Verify a complete create → read → search → update → history → delete
    cycle works end-to-end through the auth layer."""

    API_KEY = "flow-test-key-99"

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory, load_app, user_api_key):
        self.app = load_app({"ADMIN_API_KEY": self.API_KEY})
        self.client = TestClient(self.app)
        self.mock = _mock_memory
        # ADMIN_API_KEY is bootstrap (read-only); the full CRUD flow mutates
        # memories, so authenticate as a real dashboard user.
        self.API_KEY = user_api_key

    def _authed(self, method, path, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.API_KEY
        return self.client.request(method, path, headers=headers, **kwargs)

    def test_full_crud_cycle(self):
        # 1. Create
        resp = self._authed("POST", "/memories", json={
            "messages": [{"role": "user", "content": "I love fresh vegetable pizza"}],
            "user_id": "alice",
        })
        assert resp.status_code == 200
        self.mock.add.assert_called_once()

        # 2. Read single
        resp = self._authed("GET", "/memories/mem-1")
        assert resp.status_code == 200
        # resolve_memory_entities also reads memory.get() to derive scope, so get
        # is called more than once; assert on the last call instead of once-only.
        self.mock.get.assert_called_with("mem-1")

        # 3. Read all
        resp = self._authed("GET", "/memories", params={"user_id": "alice"})
        assert resp.status_code == 200
        self.mock.get_all.assert_called_once_with(filters={"user_id": "alice"}, show_expired=False)

        # 4. Search
        resp = self._authed("POST", "/search", json={"query": "pizza", "user_id": "alice"})
        assert resp.status_code == 200
        self.mock.search.assert_called_once()

        # 5. Update
        resp = self._authed("PUT", "/memories/mem-1", json={"text": "updated content"})
        assert resp.status_code == 200
        self.mock.update.assert_called_once()

        # 6. History
        resp = self._authed("GET", "/memories/mem-1/history")
        assert resp.status_code == 200
        self.mock.history.assert_called_once_with(memory_id="mem-1")

        # 7. Delete single
        resp = self._authed("DELETE", "/memories/mem-1")
        assert resp.status_code == 200
        self.mock.delete.assert_called_once_with(memory_id="mem-1")

        # 8. Delete all
        resp = self._authed("DELETE", "/memories", params={"user_id": "alice"})
        assert resp.status_code == 200
        self.mock.delete_all.assert_called_once()

    def test_crud_flow_blocked_without_auth(self):
        """Same flow should fail at every step without the key."""
        endpoints = [
            ("POST", "/memories", {"json": {
                "messages": [{"role": "user", "content": "test"}], "user_id": "alice"
            }}),
            ("GET", "/memories/mem-1", {}),
            ("GET", "/memories", {"params": {"user_id": "alice"}}),
            ("POST", "/search", {"json": {"query": "pizza", "user_id": "alice"}}),
            ("PUT", "/memories/mem-1", {"json": {"data": "x"}}),
            ("GET", "/memories/mem-1/history", {}),
            ("DELETE", "/memories/mem-1", {}),
            ("DELETE", "/memories", {"params": {"user_id": "alice"}}),
            ("POST", "/reset", {}),
        ]
        for method, path, kwargs in endpoints:
            resp = self.client.request(method, path, **kwargs)
            assert resp.status_code == 401, f"Unauthenticated {method} {path} should be 401"
            # Verify the mock was NOT called (auth blocked before reaching handler)
        self.mock.add.assert_not_called()
        self.mock.get.assert_not_called()
        self.mock.search.assert_not_called()
        self.mock.update.assert_not_called()
        self.mock.history.assert_not_called()
        self.mock.delete.assert_not_called()
        self.mock.delete_all.assert_not_called()
        self.mock.reset.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAuthEdgeCases:
    """Boundary conditions and unusual inputs."""

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory, load_app):
        self.mock = _mock_memory
        # Stash the load_app factory so the per-test methods below can build
        # their own apps without each requesting the fixture explicitly.
        self.load_app = load_app

    def test_very_long_api_key(self):
        """Server should handle a very long key without crashing."""
        long_key = "k" * 4096
        app = self.load_app({"ADMIN_API_KEY": long_key})
        client = TestClient(app)
        resp = client.get("/memories/mem-1", headers={"X-API-Key": long_key})
        assert resp.status_code == 200

    def test_special_characters_in_api_key(self):
        """Keys with special ASCII characters should work."""
        special_key = "sk-!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        app = self.load_app({"ADMIN_API_KEY": special_key})
        client = TestClient(app)

        resp = client.get("/memories/mem-1", headers={"X-API-Key": special_key})
        assert resp.status_code == 200

        resp = client.get("/memories/mem-1", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_empty_admin_api_key_disables_auth(self):
        """An empty ADMIN_API_KEY disables auth (no key validated, request passes)."""
        app = self.load_app({"ADMIN_API_KEY": ""})  # empty ADMIN_API_KEY => disabled mode
        client = TestClient(app)
        resp = client.get("/memories/mem-1")
        assert resp.status_code != 401

    def test_switching_from_enabled_to_disabled(self):
        """Simulates a server restart with auth toggled off."""
        # First: auth enabled
        app1 = self.load_app({"ADMIN_API_KEY": "secret"})
        c1 = TestClient(app1)
        assert c1.get("/memories/mem-1").status_code == 401

        # Then: auth disabled
        app2 = self.load_app({"ADMIN_API_KEY": ""})
        c2 = TestClient(app2)
        assert c2.get("/memories/mem-1").status_code != 401

    def test_openapi_schema_accessible_without_key(self):
        """The /docs and /openapi.json endpoints should always be reachable."""
        app = self.load_app({"ADMIN_API_KEY": "secret"})
        client = TestClient(app)

        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema

        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema_documents_auth(self):
        """The OpenAPI schema should mention authentication."""
        app = self.load_app({"ADMIN_API_KEY": "secret"})
        client = TestClient(app)
        schema = client.get("/openapi.json").json()
        assert "Authentication" in schema.get("info", {}).get("description", "")


# ---------------------------------------------------------------------------
# Startup logging
# ---------------------------------------------------------------------------

class TestStartupLogging:
    """Auth config is validated/logged at app startup (lifespan), not import.

    The validator is a plain module-level function (``main._validate_auth_config``)
    that reads ``get_auth_config()`` live, so these tests unit-test it directly by
    patching the config — no import-reload, no env gymnastics. One wiring test
    confirms lifespan actually calls it.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory):
        # Ensures ``main`` is importable (initialize_state runs under the mocked
        # Memory.from_config) even when this class runs in isolation.
        pass

    def _validator(self):
        from server import main

        return main._validate_auth_config()

    def test_warning_when_auth_disabled(self, caplog, monkeypatch):
        from server import main

        monkeypatch.setattr(main, "get_auth_config", lambda: AuthConfig(auth_disabled=True, jwt_secret="x"))
        with caplog.at_level(logging.WARNING):
            self._validator()
        assert any("Protected endpoints are open" in r.message for r in caplog.records)

    def test_runtime_error_when_enabled_without_jwt_secret(self, monkeypatch):
        from server import main

        monkeypatch.setattr(main, "get_auth_config", lambda: AuthConfig(auth_disabled=False, jwt_secret=None))
        with pytest.raises(RuntimeError, match="JWT_SECRET is required"):
            self._validator()

    def test_info_when_auth_enabled(self, caplog, monkeypatch):
        # "Auth config resolved" is logged by load_auth_config on resolution;
        # clear the cache so the next read re-resolves under the env set above.
        monkeypatch.setenv("ADMIN_API_KEY", "a-long-enough-secret-key")
        monkeypatch.setenv("AUTH_DISABLED", "false")
        import auth_config

        auth_config.reload_auth_config()
        with caplog.at_level(logging.INFO):
            auth_config.get_auth_config()
        assert any("Auth config resolved" in r.message for r in caplog.records)

    def test_warning_when_key_too_short(self, caplog, monkeypatch):
        from server import main

        monkeypatch.setattr(
            main, "get_auth_config", lambda: AuthConfig(auth_disabled=False, jwt_secret="x", admin_api_key="short")
        )
        with caplog.at_level(logging.WARNING):
            self._validator()
        assert any("shorter than" in r.message for r in caplog.records)

    def test_lifespan_runs_auth_validation(self, monkeypatch, load_app):
        """Entering the app (lifespan startup) triggers the auth validator."""
        app = load_app({"ADMIN_API_KEY": ""})
        from server import main

        # Patch AFTER load_app: load_app reloads main, which re-binds the
        # validator to the freshly defined function and would discard a patch
        # applied before the reload.
        calls = []
        monkeypatch.setattr(main, "_validate_auth_config", lambda: calls.append(1))
        with TestClient(app):
            pass
        assert calls == [1]


# ---------------------------------------------------------------------------
# Unit tests for bcrypt helper functions (auth module)
# ---------------------------------------------------------------------------

class TestBcryptHelpers:
    """Unit tests for hash_password / verify_password / verify_api_key_hash."""

    @pytest.fixture(autouse=True)
    def _import_auth(self):
        import importlib

        self.auth = importlib.import_module("server.auth")

    def test_hash_password_returns_valid_bcrypt_hash(self):
        h = self.auth.hash_password("secret")
        assert h.startswith("$2b$12$")

    def test_verify_password_correct(self):
        h = self.auth.hash_password("correct-horse")
        assert self.auth.verify_password("correct-horse", h) is True

    def test_verify_password_wrong(self):
        h = self.auth.hash_password("correct-horse")
        assert self.auth.verify_password("wrong", h) is False

    def test_verify_password_malformed_hash_returns_false(self):
        """Corrupt DB hash must not raise — should return False (not 500)."""
        assert self.auth.verify_password("any", "not-a-bcrypt-hash") is False

    def test_verify_password_empty_hash_returns_false(self):
        assert self.auth.verify_password("any", "") is False

    def test_verify_api_key_hash_correct(self):
        _, _, key_hash = self.auth.generate_api_key()
        # generate_api_key hashes the full key; grab it directly for round-trip
        raw = "m0sk_testkey"
        h = self.auth.hash_password(raw)
        assert self.auth.verify_api_key_hash(raw, h) is True

    def test_verify_api_key_hash_wrong(self):
        raw = "m0sk_testkey"
        h = self.auth.hash_password(raw)
        assert self.auth.verify_api_key_hash("m0sk_wrongkey", h) is False

    def test_verify_api_key_hash_malformed_returns_false(self):
        """Corrupt DB hash must not raise — should return False (not 500)."""
        assert self.auth.verify_api_key_hash("m0sk_anykey", "not-a-bcrypt-hash") is False

    def test_generate_api_key_format(self):
        full_key, prefix, key_hash = self.auth.generate_api_key()
        assert full_key.startswith("m0sk_")
        assert prefix == full_key[:12]
        assert key_hash.startswith("$2b$12$")

    def test_generate_api_key_hash_verifies(self):
        full_key, _, key_hash = self.auth.generate_api_key()
        assert self.auth.verify_api_key_hash(full_key, key_hash) is True
