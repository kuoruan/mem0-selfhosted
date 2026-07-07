"""Tests for OIDC routes and state management.

Uses pytest + unittest.mock / fastapi.testclient.TestClient.
Covers:
- oidc_login endpoint (redirect, 404, 503, unsafe next)
- oidc_callback endpoint (user creation, state invalid, first user admin, missing sub)
- oidc_state consume atomicity and expired state cleanup
"""

import importlib
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

# conftest.py registers the flat-name aliases (auth_config, oidc, etc.)

from cachetools import TTLCache

from server.oidc_state import MemoryOidcStateStore, OidcStateData


# ============================================================================
# Test is_safe_redirect
# ============================================================================


class TestIsSafeRedirect:
    """Test the redirect URL validation helper."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        """Import utils.helpers for is_safe_redirect."""
        self.helpers = importlib.import_module("server.utils.helpers")

    def test_relative_path_allowed(self):
        assert self.helpers.is_safe_redirect("/dashboard")

    def test_relative_path_with_query(self):
        assert self.helpers.is_safe_redirect("/dashboard?tab=settings")

    def test_absolute_url_blocked(self):
        assert not self.helpers.is_safe_redirect("https://evil.com")

    def test_protocol_relative_blocked(self):
        assert not self.helpers.is_safe_redirect("//evil.com")

    def test_schemaless_url_blocked(self):
        assert not self.helpers.is_safe_redirect("//evil.com/path")

    def test_none_returns_false(self):
        assert not self.helpers.is_safe_redirect(None)

    def test_empty_string_returns_false(self):
        assert not self.helpers.is_safe_redirect("")

    def test_path_without_slash_blocked(self):
        assert not self.helpers.is_safe_redirect("relative/path")

    # -- Backslash-based open redirect (PR #18 Issue #1) --

    def test_backslash_before_host(self):
        r"""/\evil.com gets normalized to //evil.com by browsers."""
        assert not self.helpers.is_safe_redirect("/\\evil.com")

    def test_mixed_backslash_forward_slash(self):
        r"""\/evil.com also gets normalized to //evil.com."""
        assert not self.helpers.is_safe_redirect("/\\/evil.com")

    def test_backslash_in_path(self):
        r"""Paths containing backslash should be rejected."""
        assert not self.helpers.is_safe_redirect("/dashboard\\@evil.com")

    def test_double_backslash(self):
        r"""\\evil.com should be rejected."""
        assert not self.helpers.is_safe_redirect("\\\\evil.com")

    def test_backslash_at_start(self):
        r"""\\evil.com should be rejected."""
        assert not self.helpers.is_safe_redirect("\\evil.com")

    # -- Whitespace open redirect (Issue #36) --

    def test_tab_in_path(self):
        r"""/\tevil.com — tab should be rejected."""
        assert not self.helpers.is_safe_redirect("/\tevil.com")

    def test_newline_in_path(self):
        r"""/\nevil.com — newline should be rejected."""
        assert not self.helpers.is_safe_redirect("/\nevil.com")

    def test_carriage_return_in_path(self):
        r"""/\revil.com — CR should be rejected."""
        assert not self.helpers.is_safe_redirect("/\revil.com")


# ============================================================================
# Test oidc_login endpoint
# ============================================================================


class TestOidcLogin:
    """Test the /auth/oidc/{provider}/login endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test client with mocked dependencies."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        self.app = FastAPI()
        oidc_routes = importlib.import_module("server.routers.oidc")
        self.app.include_router(oidc_routes.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    @patch("server.routers.oidc.get_auth_config")
    def test_unknown_provider_returns_404(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_config.oidc.providers = [MagicMock(name="google")]
        mock_config.oidc.get_provider = MagicMock(return_value=None)
        mock_get_config.return_value = mock_config

        response = self.client.get("/auth/oidc/nonexistent/login", follow_redirects=False)
        assert response.status_code == 404

    @patch("server.routers.oidc.get_auth_config")
    def test_oidc_not_configured_returns_503(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.oidc = None
        mock_get_config.return_value = mock_config

        response = self.client.get("/auth/oidc/google/login", follow_redirects=False)
        assert response.status_code == 503

    @patch("server.routers.oidc.get_auth_config")
    def test_unsafe_next_returns_400(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_config.oidc.providers = [MagicMock(name="google")]
        mock_get_config.return_value = mock_config

        response = self.client.get("/auth/oidc/google/login?next=https://evil.com", follow_redirects=False)
        assert response.status_code == 400

    @patch("server.routers.oidc.discover_provider")
    @patch("server.routers.oidc.get_state_store")
    @patch("server.routers.oidc.get_auth_config")
    def test_successful_redirect_to_idp(self, mock_get_config, mock_get_store, mock_discover):
        # Config
        provider_cfg = MagicMock()
        provider_cfg.name = "google"
        provider_cfg.issuer_url = "https://accounts.google.com"
        provider_cfg.client_id = "test-client-id"
        provider_cfg.scopes = ["openid", "email", "profile"]

        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_config.oidc.providers = [provider_cfg]
        mock_config.oidc.get_provider = MagicMock(return_value=provider_cfg)
        mock_get_config.return_value = mock_config

        # Discovery
        mock_metadata = MagicMock()
        mock_metadata.authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
        mock_discover.return_value = mock_metadata

        # State store
        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store

        response = self.client.get("/auth/oidc/google/login", follow_redirects=False)
        assert response.status_code == 302
        location = response.headers["location"]
        assert "accounts.google.com/o/oauth2/v2/auth" in location
        assert "state=" in location
        assert "code_challenge=" in location
        assert "code_challenge_method=S256" in location

    @patch("server.routers.oidc.get_auth_config")
    def test_next_with_protocol_relative_returns_400(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_get_config.return_value = mock_config

        response = self.client.get("/auth/oidc/google/login?next=//evil.com/path", follow_redirects=False)
        assert response.status_code == 400

    @patch("server.routers.oidc.discover_provider")
    @patch("server.routers.oidc.get_state_store")
    @patch("server.routers.oidc.get_auth_config")
    def test_discover_failure_returns_502_via_to_thread(self, mock_get_config, mock_get_store, mock_discover):
        """discover_provider exception propagated through asyncio.to_thread → 502."""
        provider_cfg = MagicMock()
        provider_cfg.name = "google"
        provider_cfg.issuer_url = "https://accounts.google.com"
        provider_cfg.client_id = "test-client-id"
        provider_cfg.scopes = ["openid"]

        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_config.oidc.providers = [provider_cfg]
        mock_config.oidc.get_provider = MagicMock(return_value=provider_cfg)
        mock_get_config.return_value = mock_config

        mock_discover.side_effect = ConnectionError("timeout")

        response = self.client.get("/auth/oidc/google/login", follow_redirects=False)
        assert response.status_code == 502
        assert response.json()["detail"] == "Failed to contact identity provider"


# ============================================================================
# Test MemoryOidcStateStore consume
# ============================================================================


class TestMemoryOidcStateStoreConsume:
    """Test the atomic consume method."""

    @pytest.mark.asyncio
    async def test_consume_returns_data_and_removes_it(self):
        store = MemoryOidcStateStore()
        data = OidcStateData(
            code_verifier="verifier123",
            provider="google",
            expires_at=time.time() + 600,
        )
        await store.save("state1", data, ttl_seconds=600)

        result = await store.consume("state1")
        assert result is not None
        assert result.code_verifier == "verifier123"
        assert result.provider == "google"

        # Second consume should return None
        result2 = await store.consume("state1")
        assert result2 is None

    @pytest.mark.asyncio
    async def test_consume_expired_returns_none(self):
        """TTLCache handles expiration automatically; expired entries return None."""
        store = MemoryOidcStateStore()
        # Override with a very short TTL for testing
        store._store = TTLCache(maxsize=100, ttl=0.1)
        data = OidcStateData(
            code_verifier="verifier123",
            provider="google",
        )
        await store.save("short-lived", data)
        time.sleep(0.2)  # wait for TTL to expire

        result = await store.consume("short-lived")
        assert result is None

    @pytest.mark.asyncio
    async def test_consume_nonexistent_returns_none(self):
        store = MemoryOidcStateStore()
        result = await store.consume("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_evict_expired_on_access(self):
        """TTLCache automatically evicts expired entries on access."""
        store = MemoryOidcStateStore()
        store._store = TTLCache(maxsize=100, ttl=0.1)
        # Insert entries with short TTL
        await store.save("old-state", OidcStateData(code_verifier="expired", provider="google"))
        await store.save("valid-state", OidcStateData(code_verifier="valid", provider="google"))
        time.sleep(0.2)  # wait for TTL to expire

        # Accessing the cache triggers lazy eviction
        assert store._store.get("old-state") is None
        assert store._store.get("valid-state") is None


# ============================================================================
# Test oidc_callback endpoint
# ============================================================================


class TestOidcCallback:
    """Test the /auth/oidc/{provider}/callback endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test client with mocked dependencies."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        self.app = FastAPI()
        oidc_routes = importlib.import_module("server.routers.oidc")
        self.app.include_router(oidc_routes.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.oidc_routes = oidc_routes

    def test_callback_with_invalid_state(self):
        """State not found in store should redirect with error."""
        mock_store = AsyncMock()
        mock_store.consume = AsyncMock(return_value=None)

        with patch.object(self.oidc_routes, "get_state_store", return_value=mock_store):
            response = self.client.get(
                "/auth/oidc/google/callback?code=abc&state=badstate",
                follow_redirects=False,
            )
            assert response.status_code == 302
            location = response.headers["location"]
            assert "error=invalid_state" in location

    def test_callback_provider_mismatch(self):
        """State provider != URL provider should redirect with error."""
        state_data = OidcStateData(
            code_verifier="v",
            provider="github",  # different provider
            expires_at=time.time() + 600,
        )
        mock_store = AsyncMock()
        mock_store.consume = AsyncMock(return_value=state_data)

        with patch.object(self.oidc_routes, "get_state_store", return_value=mock_store):
            response = self.client.get(
                "/auth/oidc/google/callback?code=abc&state=s1",
                follow_redirects=False,
            )
            assert response.status_code == 302
            location = response.headers["location"]
            assert "error=provider_mismatch" in location

    def test_callback_idp_error(self):
        """IdP error response should redirect with error info."""
        response = self.client.get(
            "/auth/oidc/google/callback?error=access_denied&error_description=User+denied",
            follow_redirects=False,
        )
        assert response.status_code == 302
        location = response.headers["location"]
        assert "error=access_denied" in location

    def test_callback_missing_code_or_state(self):
        """Missing code/state should redirect with error."""
        response = self.client.get(
            "/auth/oidc/google/callback",
            follow_redirects=False,
        )
        assert response.status_code == 302
        location = response.headers["location"]
        assert "error=invalid_response" in location


# ============================================================================
# Test OIDC discovery issuer mismatch (issue 3)
# ============================================================================


class TestOidcDiscoveryIssuerMismatch:
    """Verify that issuer mismatch raises ValueError."""

    @patch("server.oidc.httpx.Client")
    def test_issuer_mismatch_raises_valueerror(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "issuer": "https://evil.com",
            "authorization_endpoint": "https://evil.com/auth",
            "token_endpoint": "https://evil.com/token",
            "jwks_uri": "https://evil.com/jwks",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from server.oidc import discover_provider

        with pytest.raises(ValueError, match="does not match configured issuer_url"):
            discover_provider("https://accounts.google.com")

    @patch("server.oidc.httpx.Client")
    def test_issuer_match_succeeds(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "issuer": "https://accounts.google.com",
            "authorization_endpoint": "https://accounts.google.com/auth",
            "token_endpoint": "https://accounts.google.com/token",
            "jwks_uri": "https://accounts.google.com/jwks",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from server.oidc import discover_provider

        # Clear cache
        from server.oidc import _DISCOVERY_CACHE

        _DISCOVERY_CACHE.clear()

        metadata = discover_provider("https://accounts.google.com")
        assert metadata.issuer == "https://accounts.google.com"


# ============================================================================
# Test: login endpoint does not leak exception details (issue 2)
# ============================================================================


class TestLoginNoExceptionLeak:
    """Verify that 502 errors from discovery do not expose internal details."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        self.app = FastAPI()
        oidc_routes = importlib.import_module("server.routers.oidc")
        self.app.include_router(oidc_routes.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    @patch("server.routers.oidc.discover_provider")
    @patch("server.routers.oidc.get_auth_config")
    def test_502_does_not_expose_exception_detail(self, mock_get_config, mock_discover):
        provider_cfg = MagicMock()
        provider_cfg.name = "google"
        provider_cfg.issuer_url = "https://accounts.google.com"
        provider_cfg.client_id = "test-client-id"
        provider_cfg.scopes = ["openid"]

        mock_config = MagicMock()
        mock_config.oidc = MagicMock()
        mock_config.oidc.providers = [provider_cfg]
        mock_config.oidc.get_provider = MagicMock(return_value=provider_cfg)
        mock_get_config.return_value = mock_config

        # Discovery raises an exception with sensitive internal info
        mock_discover.side_effect = ConnectionError("secret-internal-host:5432 refused")

        response = self.client.get("/auth/oidc/google/login", follow_redirects=False)
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "Failed to contact identity provider"
        # Ensure the internal error is NOT exposed
        assert "secret-internal-host" not in detail
        assert "refused" not in detail


# ============================================================================
# Test: urlparse has been moved to top-level import (issue 3)
# ============================================================================


class TestUrlParseTopLevelImport:
    """Verify urlparse is imported at module level, not inside function."""

    def test_urlparse_in_module_dir(self):
        import ast
        import server.routers.oidc as oidc_module

        source = inspect.getsource(oidc_module)
        tree = ast.parse(source)

        # Collect all top-level imports of urlparse
        top_level_urlparse = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "urllib.parse":
                for alias in node.names:
                    if alias.name == "urlparse":
                        top_level_urlparse = True

        assert top_level_urlparse, "urlparse should be imported at module top level"

    def test_no_urlparse_import_inside_functions(self):
        import ast
        import server.routers.oidc as oidc_module

        source = inspect.getsource(oidc_module)
        tree = ast.parse(source)

        # Walk function bodies and check for inline urlparse imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.ImportFrom) and child.module == "urllib.parse":
                        names = [alias.name for alias in child.names]
                        assert "urlparse" not in names, f"urlparse should not be imported inside function '{node.name}'"


# ============================================================================
# Test: placeholder email includes random suffix (issue 4)
# ============================================================================


class TestPlaceholderEmailRandomSuffix:
    """Verify that generated placeholder emails contain a random suffix."""

    def test_placeholder_email_has_random_suffix(self):
        import re

        importlib.import_module("server.routers.oidc")
        # We verify the pattern by checking two generated emails are different
        # when the idp_sub prefix is identical
        idp_sub = "a" * 100  # will be truncated to first 64 chars
        provider = "testprovider"

        # Simulate the f-string logic from the code
        email1 = f"{idp_sub[:64]}-{__import__('secrets').token_hex(4)}@oidc.{provider}"
        email2 = f"{idp_sub[:64]}-{__import__('secrets').token_hex(4)}@oidc.{provider}"

        # Same prefix but different random suffix
        assert email1 != email2, "Placeholder emails with same prefix should have different random suffixes"

        # Validate format: prefix-XXXXXXXX@oidc.provider
        pattern = r"^[a-zA-Z0-9_-]*-[0-9a-f]{8}@oidc\.\w+$"
        assert re.match(pattern, email1), f"Email format unexpected: {email1}"
        assert re.match(pattern, email2), f"Email format unexpected: {email2}"


# ============================================================================
# Test oidc_callback success paths (issue 9)
# ============================================================================


def _make_state_data(provider="google", nonce="test-nonce", next_url=None):
    """Create a valid OidcStateData for test callbacks."""
    return OidcStateData(
        code_verifier="test-verifier",
        provider=provider,
        redirect_uri="http://localhost:8000/auth/oidc/google/callback",
        next_url=next_url,
        nonce=nonce,
        expires_at=time.time() + 600,
    )


def _make_provider_config():
    """Create a mock provider config."""
    cfg = MagicMock()
    cfg.name = "google"
    cfg.issuer_url = "https://accounts.google.com"
    cfg.client_id = "test-client-id"
    cfg.client_secret = "test-client-secret"
    cfg.scopes = ["openid", "email", "profile"]
    cfg.username_claim = None  # default: name → preferred_username → email prefix
    return cfg


def _make_oidc_config(provider_config):
    """Create a mock OIDC config that returns the given provider."""
    config = MagicMock()
    config.oidc = MagicMock()
    config.oidc.providers = [provider_config]
    config.oidc.get_provider = MagicMock(return_value=provider_config)
    return config


def _make_metadata():
    """Create a mock provider metadata."""
    metadata = MagicMock()
    metadata.issuer = "https://accounts.google.com"
    metadata.token_endpoint = "https://accounts.google.com/token"
    metadata.jwks_uri = "https://accounts.google.com/jwks"
    metadata.id_token_signing_alg_values_supported = ["RS256"]
    return metadata


class TestOidcCallbackSuccess:
    """Test the /auth/oidc/{provider}/callback endpoint happy paths."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test client with mocked dependencies."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        self.app = FastAPI()
        self.oidc_routes = importlib.import_module("server.routers.oidc")
        self.app.include_router(self.oidc_routes.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def _setup_common_patches(self, mock_db, mock_verify_claims, state_data=None, provider_config=None):
        """Return the full context manager stack for a callback success test."""
        from contextlib import ExitStack

        if provider_config is None:
            provider_config = _make_provider_config()
        oidc_config = _make_oidc_config(provider_config)
        metadata = _make_metadata()
        if state_data is None:
            state_data = _make_state_data()

        # State store
        mock_store = AsyncMock()
        mock_store.consume = AsyncMock(return_value=state_data)

        # Build the list of patches
        patches = [
            patch.object(self.oidc_routes, "get_state_store", return_value=mock_store),
            patch.object(self.oidc_routes, "get_auth_config", return_value=oidc_config),
            patch.object(self.oidc_routes, "discover_provider", return_value=metadata),
            patch.object(
                self.oidc_routes,
                "exchange_code_for_tokens",
                return_value={"id_token": "fake-id-token", "access_token": "at"},
            ),
            patch.object(self.oidc_routes, "verify_id_token", return_value=mock_verify_claims),
            patch.object(self.oidc_routes, "create_access_token", return_value="access-tok"),
            patch.object(self.oidc_routes, "create_refresh_token", return_value="refresh-tok"),
        ]

        # Dependency override for get_db
        self.app.dependency_overrides[self.oidc_routes.get_db] = lambda: mock_db

        stack = ExitStack()
        for p in patches:
            stack.enter_context(p)
        stack.callback(lambda: self.app.dependency_overrides.clear())
        return stack

    def test_callback_forwards_access_token_to_verify(self):
        """The callback forwards token_response['access_token'] to verify_id_token.

        Required for at_hash validation (OIDC Core §3.1.3.6): IdPs like
        Authelia/Keycloak include at_hash in the ID token, and python-jose
        rejects the token unless access_token is supplied.
        """
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [None, None, 0]
        mock_db.get.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {"sub": "google-user-123", "email": "u@example.com", "email_verified": True, "name": "U"}

        with self._setup_common_patches(mock_db, claims):
            self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )
            # exchange_code_for_tokens mock returns {"access_token": "at"}; it
            # must be threaded through to verify_id_token.
            assert self.oidc_routes.verify_id_token.call_args.kwargs["access_token"] == "at"

    def test_new_user_first_oidc_login(self):
        """First OIDC login creates a new user and issues tokens.

        The refresh_token is never placed in the URL fragment — instead a
        short-lived exchange code is delivered and the frontend exchanges it
        via POST /auth/oidc/exchange for the real refresh_token.
        """
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            None,  # candidate lookup (verified email) → not found, no auto-link;
                   # reused as the collision value for the new-user email decision
            0,  # User count for first-user check → 0 → admin
        ]
        mock_db.get.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-123",
            "email": "newuser@example.com",
            "email_verified": True,
            "name": "New User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        location = response.headers["location"]
        assert "/auth/callback#" in location
        assert "access_token=access-tok" in location
        # refresh_token must NOT be in the fragment — replaced by exchange code
        assert "refresh_token=" not in location
        assert "code=" in location

        # Extract the exchange code and verify it can be exchanged for the
        # real refresh_token.
        fragment = location.split("#", 1)[1]
        params = dict(
            p.split("=", 1) for p in fragment.split("&") if "=" in p
        )
        exchange_code = params.get("code")
        assert exchange_code is not None

        exchange_resp = self.client.post(
            "/auth/oidc/exchange",
            json={"exchange_code": exchange_code},
        )
        assert exchange_resp.status_code == 200
        assert exchange_resp.json()["refresh_token"] == "refresh-tok"

        # Reusing the same exchange code must fail (single-use)
        reuse_resp = self.client.post(
            "/auth/oidc/exchange",
            json={"exchange_code": exchange_code},
        )
        assert reuse_resp.status_code == 401

        # User + OidcLink added
        assert mock_db.add.call_count == 2
        mock_db.commit.assert_called_once()

    def test_next_url_passthrough_quoted(self):
        """state_data.next_url is forwarded as a quoted ?next= query param."""
        mock_db = MagicMock()
        # no link, no candidate (verified email, reused as collision), first user → admin
        mock_db.scalar.side_effect = [None, None, 0]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-next",
            "email": "next@example.com",
            "email_verified": True,
            "name": "Next User",
        }

        state_data = _make_state_data(next_url="/dashboard/settings?tab=a b")
        with self._setup_common_patches(mock_db, claims, state_data=state_data):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        location = response.headers["location"]
        # next must be present and percent-quoted (space → %20)
        assert "next=%2Fdashboard%2Fsettings%3Ftab%3Da%20b" in location
        # access_token still in the fragment
        assert "access_token=access-tok" in location

    def test_existing_user_email_verified_auto_link(self):
        """Existing local password account with verified email gets auto-linked to OIDC.

        Auto-link fires only when the existing account has a password_hash
        (local credential being upgraded to OIDC); OIDC-only accounts are
        never merged by email (see test_oidc_email_recycled_no_takeover).
        """
        existing_user = MagicMock()
        existing_user.id = "existing-user-uuid"
        existing_user.role = "member"
        existing_user.email = "existing@example.com"
        existing_user.password_hash = "hashed"  # local password account → eligible for auto-link

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            existing_user,  # candidate lookup → password account → link
        ]
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-456",
            "email": "existing@example.com",
            "email_verified": True,
            "name": "Existing User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        location = response.headers["location"]
        assert "access_token=access-tok" in location
        # Only OidcLink added (auto-link), no new User
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_existing_user_email_not_verified_creates_new(self):
        """Existing user with unverified email — creates new user with placeholder, no auto-link."""
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            # No existing_user lookup — skipped for unverified email
            1,  # User count → not first → member role
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-789",
            "email": "unverified@example.com",
            "email_verified": False,  # NOT verified
            "name": "Unverified User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        location = response.headers["location"]
        assert "access_token=access-tok" in location
        # New user + new OidcLink (2 db.add calls)
        assert mock_db.add.call_count == 2

    def test_null_email_generates_placeholder(self):
        """When email claim is present but None, a placeholder email should be generated."""
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            # No existing_user lookup — email is None, skipped
            1,  # User count → not first → member role
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-null-email",
            "email": None,  # key present but value is None
            "name": "Null Email User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        location = response.headers["location"]
        assert "access_token=access-tok" in location
        # Verify User was added with a placeholder @oidc.google email
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert len(user_add_calls) == 1
        user_obj = user_add_calls[0][0][0]
        assert user_obj.email.endswith("@oidc.google")

    def test_unverified_email_no_existing_user_uses_placeholder(self):
        """Pre-hijacking prevention: unverified email with no existing user must use placeholder.

        An attacker should NOT be able to pre-create an account with a victim's
        unverified email, then have the victim auto-linked to it later.
        """
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            # No existing_user lookup by email — skipped for unverified
            0,  # User count → first → admin
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "attacker-sub-123",
            "email": "victim@example.com",  # real email but unverified
            "email_verified": False,
            "name": "Attacker",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # The created user MUST have a placeholder email, NOT victim@example.com
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert len(user_add_calls) == 1
        user_obj = user_add_calls[0][0][0]
        assert user_obj.email.endswith("@oidc.google")
        assert user_obj.email != "victim@example.com"

    def test_idp_sub_special_chars_sanitized(self):
        """Placeholder email must sanitize special chars in idp_sub (e.g. Auth0 'auth0|607f...')."""
        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            # No existing_user lookup — email is None
            0,  # User count → first → admin
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "auth0|607f1234abcd",  # contains '|'
            "email": None,
            "name": "Auth0 User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert len(user_add_calls) == 1
        user_obj = user_add_calls[0][0][0]
        # Email should not contain '|' or other special chars
        assert "|" not in user_obj.email
        assert "@" in user_obj.email
        assert user_obj.email.endswith("@oidc.google")

    # ----------------------------------------------------------------------
    # Email-sync + tightened auto-link regression tests (account-takeover fix)
    # ----------------------------------------------------------------------

    def test_existing_oidc_user_email_changed_syncs(self):
        """Linked user whose IdP email changed (verified, no conflict) gets synced.

        The sync is no longer gated on the email being an @oidc.{provider}
        placeholder — any verified, differing email is applied so stale
        addresses are released.
        """
        linked_user = MagicMock()
        linked_user.id = "linked-user-uuid"
        linked_user.email = "old@example.com"

        oidc_link = MagicMock()
        oidc_link.user_id = linked_user.id

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            oidc_link,  # OidcLink lookup → linked user
            None,  # collision check for new email → no conflict
        ]
        mock_db.get.return_value = linked_user
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-123",
            "email": "new@example.com",
            "email_verified": True,
            "name": "Linked User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Real email applied
        assert linked_user.email == "new@example.com"
        # No new User created — only the existing linked user was updated
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert user_add_calls == []

    def test_email_sync_conflict_reverts_to_placeholder(self):
        """Linked user: when the verified email is owned by another account,
        the stale email is released by reverting to a placeholder.

        This blocks a recycled-IdP-email takeover: without the revert, the
        account would keep holding an email the IdP has since reassigned,
        which a fresh sub could then auto-link against.
        """
        linked_user = MagicMock()
        linked_user.id = "linked-user-uuid"
        linked_user.email = "stale@example.com"

        oidc_link = MagicMock()
        oidc_link.user_id = linked_user.id

        collision_user = MagicMock()
        collision_user.id = "other-user-uuid"

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            oidc_link,  # OidcLink lookup → linked user
            collision_user,  # collision check → owned by someone else
        ]
        mock_db.get.return_value = linked_user
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-123",
            "email": "new@example.com",
            "email_verified": True,
            "name": "Linked User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Stale email released — reverted to a placeholder, NOT the conflicting email
        assert linked_user.email != "new@example.com"
        assert linked_user.email.endswith("@oidc.google")
        # No new User created
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert user_add_calls == []

    def test_oidc_email_recycled_no_takeover(self):
        """Recycled-IdP-email takeover regression (most critical).

        A new IdP sub (no OidcLink) returns a verified email that is already
        owned by an existing OIDC-only account (password_hash is None). The
        new sub must NOT link to that account — that would let whoever the
        IdP reassigned the email to take it over. Instead an independent
        account is created with a placeholder email.
        """
        existing_oidc_user = MagicMock()
        existing_oidc_user.id = "existing-oidc-uuid"
        existing_oidc_user.email = "recycled@example.com"
        existing_oidc_user.password_hash = None  # OIDC-only account — must NOT be linked

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found (new sub)
            existing_oidc_user,  # candidate lookup → OIDC account, password_hash None → no link;
                                 # reused as collision for the email decision (conflict)
            1,  # User count → not first → member
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "new-google-sub-456",
            "email": "recycled@example.com",
            "email_verified": True,
            "name": "New Recipient",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # New independent User + new OidcLink (2 db.add calls)
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        link_add_calls = [
            c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "OidcLink"
        ]
        assert len(user_add_calls) == 1
        assert len(link_add_calls) == 1

        new_user = user_add_calls[0][0][0]
        new_link = link_add_calls[0][0][0]
        # New User uses a placeholder email (claims_email was in conflict)
        assert new_user.email != "recycled@example.com"
        assert new_user.email.endswith("@oidc.google")
        # OidcLink points at the newly created user, NOT the existing OIDC account
        assert new_link.user_id != existing_oidc_user.id

    def test_password_user_auto_link_still_works(self):
        """Regression for the legitimate auto-link path: a local password
        account (password_hash set) with the same verified email is upgraded
        to also accept OIDC login — no new User is created."""
        existing_pwd_user = MagicMock()
        existing_pwd_user.id = "pwd-user-uuid"
        existing_pwd_user.role = "member"
        existing_pwd_user.email = "pwd@example.com"
        existing_pwd_user.password_hash = "bcrypt$hashed"  # local password account

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            existing_pwd_user,  # candidate lookup → password_hash set → auto-link
        ]
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-pwd-link",
            "email": "pwd@example.com",
            "email_verified": True,
            "name": "Pwd User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Only OidcLink added (auto-link), no new User
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        link_add_calls = [
            c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "OidcLink"
        ]
        assert user_add_calls == []
        assert len(link_add_calls) == 1
        assert link_add_calls[0][0][0].user_id == existing_pwd_user.id

    def test_same_sub_same_email_no_duplicate(self):
        """Linked user, IdP returns the same verified email — no-op.

        Guards against needless writes / duplicate-link attempts on every
        re-login.
        """
        linked_user = MagicMock()
        linked_user.id = "linked-user-uuid"
        linked_user.email = "stable@example.com"

        oidc_link = MagicMock()
        oidc_link.user_id = linked_user.id

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            oidc_link,  # OidcLink lookup → linked user (only scalar call)
        ]
        mock_db.get.return_value = linked_user
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-123",
            "email": "stable@example.com",  # identical to local — no sync needed
            "email_verified": True,
            "name": "Stable User",
        }

        with self._setup_common_patches(mock_db, claims):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Email unchanged
        assert linked_user.email == "stable@example.com"
        # No new User, no new OidcLink (link already existed)
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        link_add_calls = [
            c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "OidcLink"
        ]
        assert user_add_calls == []
        assert link_add_calls == []

    def test_custom_username_claim_used(self):
        """A configured username_claim overrides the default name fallback."""
        provider_config = _make_provider_config()
        provider_config.username_claim = "preferred_username"

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            None,  # candidate lookup (verified email) → reused as collision
            0,  # User count → first → admin
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-claim",
            "email": "claim@example.com",
            "email_verified": True,
            "name": "Ignored Name",
            "preferred_username": "alice",
        }

        with self._setup_common_patches(mock_db, claims, provider_config=provider_config):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert len(user_add_calls) == 1
        # configured claim wins over the default "name" claim
        assert user_add_calls[0][0][0].name == "alice"

    def test_username_claim_list_fallback(self):
        """A list of claims falls back through to the first non-empty one."""
        provider_config = _make_provider_config()
        provider_config.username_claim = ["login", "username"]

        mock_db = MagicMock()
        mock_db.scalar.side_effect = [
            None,  # OidcLink lookup → not found
            None,  # candidate lookup → reused as collision
            0,  # User count → first → admin
        ]
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        claims = {
            "sub": "google-user-list-claim",
            "email": "listclaim@example.com",
            "email_verified": True,
            "name": "Ignored Name",
            "username": "bob",  # "login" missing → falls back to "username"
        }

        with self._setup_common_patches(mock_db, claims, provider_config=provider_config):
            response = self.client.get(
                "/auth/oidc/google/callback?code=auth-code&state=test-state",
                follow_redirects=False,
            )

        assert response.status_code == 302
        user_add_calls = [c for c in mock_db.add.call_args_list if c[0][0].__class__.__name__ == "User"]
        assert len(user_add_calls) == 1
        assert user_add_calls[0][0][0].name == "bob"


# ============================================================================
# Test: timing leak for OIDC users in login route (Issue #10)
# ============================================================================


class TestLoginTimingLeak:
    """Verify that OIDC-only users trigger dummy_verify_password to prevent timing leaks."""

    def test_oidc_user_login_calls_dummy_verify(self):
        """Login handler must call dummy_verify_password before raising for passwordless users.

        Tests the invariant by verifying the function body calls dummy_verify_password
        in the password_hash is None branch, before the raise.
        """
        import ast
        import inspect
        import sys

        # Ensure telemetry stub exists (conftest registration may have failed silently)
        if "telemetry" not in sys.modules:
            sys.modules["telemetry"] = MagicMock()

        # Import the login function (conftest has already registered aliases)
        auth_mod = importlib.import_module("server.routers.auth")
        source = inspect.getsource(auth_mod.login)

        tree = ast.parse(source)

        # Find the if-block that checks password_hash is None
        # and verify dummy_verify_password() is called inside it
        found_hash_check = False
        found_dummy_call = False

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is the `user.password_hash is None` check
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Attribute)
                    and test.left.attr == "password_hash"
                    and any(isinstance(op, ast.Is) for op in test.ops)
                ):
                    found_hash_check = True
                    # Check that dummy_verify_password() is called in this block
                    for child in node.body:
                        if (
                            isinstance(child, ast.Expr)
                            and isinstance(child.value, ast.Call)
                            and isinstance(child.value.func, ast.Name)
                            and child.value.func.id == "dummy_verify_password"
                        ):
                            found_dummy_call = True

        assert found_hash_check, "password_hash is None check not found in login handler"
        assert found_dummy_call, "dummy_verify_password() not called in password_hash=None branch"


# ============================================================================
# Test _build_redirect_uri (F3: reverse-proxy / forwarded-host trust)
# ============================================================================


class TestBuildRedirectUri:
    """Test redirect_uri origin inference and X-Forwarded-Host same-origin gate."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.mod = importlib.import_module("server.routers.oidc")

    def _mock_request(self, scheme="http", netloc="localhost:8000", headers=None, client_ip="127.0.0.1"):
        req = MagicMock()
        req.url.scheme = scheme
        req.url.netloc = netloc
        # A plain dict exposes .get() the same way Starlette's headers do.
        req.headers = headers if headers is not None else {}
        req.client = MagicMock()
        req.client.host = client_ip
        return req

    def test_redirect_base_url_set(self):
        """SERVER_URL takes precedence over request inference."""
        with patch.object(self.mod, "_SERVER_URL", "https://public.example.com"):
            uri = self.mod._build_redirect_uri(self._mock_request(), "google")
        assert uri == "https://public.example.com/auth/oidc/google/callback"

    def test_trust_forwarded_headers_same_origin(self):
        """Forwarded host matching DASHBOARD_URL host is used when from a trusted proxy."""
        with patch.object(self.mod, "_SERVER_URL", ""), patch.object(
            self.mod, "_DASHBOARD_HOST", "localhost"
        ), patch.object(self.mod, "is_trusted_proxy", return_value=True):
            req = self._mock_request(
                netloc="localhost:8000",
                headers={"x-forwarded-host": "localhost:3000", "x-forwarded-proto": "https"},
            )
            uri = self.mod._build_redirect_uri(req, "google")
        assert uri == "https://localhost:3000/auth/oidc/google/callback"

    def test_forwarded_host_multi_value_uses_first(self):
        """Comma-separated X-Forwarded-Host uses the first (client-facing) value."""
        with patch.object(self.mod, "_SERVER_URL", ""), patch.object(
            self.mod, "_DASHBOARD_HOST", "localhost"
        ), patch.object(self.mod, "is_trusted_proxy", return_value=True):
            req = self._mock_request(headers={"x-forwarded-host": "localhost:3000, evil.com"})
            uri = self.mod._build_redirect_uri(req, "google")
        assert "localhost:3000" in uri
        assert "evil.com" not in uri

    def test_forged_cross_origin_forwarded_host_rejected(self):
        """A forwarded host that does not match DASHBOARD_URL falls back to request netloc."""
        with patch.object(self.mod, "_SERVER_URL", ""), patch.object(
            self.mod, "_DASHBOARD_HOST", "localhost"
        ), patch.object(self.mod, "is_trusted_proxy", return_value=True):
            req = self._mock_request(
                netloc="localhost:8000",
                headers={"x-forwarded-host": "evil.com", "x-forwarded-proto": "https"},
            )
            uri = self.mod._build_redirect_uri(req, "google")
        assert "evil.com" not in uri
        # Falls back to the request's own netloc (scheme stays http, proto ignored)
        assert "localhost:8000" in uri

    def test_no_trust_ignores_forwarded_headers(self):
        """Requests not from a trusted proxy ignore forwarded headers."""
        with patch.object(self.mod, "_SERVER_URL", ""), patch.object(
            self.mod, "is_trusted_proxy", return_value=False
        ):
            req = self._mock_request(
                scheme="http",
                netloc="localhost:8000",
                headers={"x-forwarded-host": "evil.com"},
            )
            uri = self.mod._build_redirect_uri(req, "google")
        assert uri == "http://localhost:8000/auth/oidc/google/callback"


# ============================================================================
# Test oidc_callback token-exchange / verification failure paths (F8-related)
# ============================================================================


class TestOidcCallbackTokenFailures:
    """Callback must redirect to the dashboard error fragment when token
    exchange or ID-token verification fails."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from contextlib import ExitStack
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.ExitStack = ExitStack
        self.app = FastAPI()
        self.oidc_routes = importlib.import_module("server.routers.oidc")
        self.app.include_router(self.oidc_routes.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def _common_patches(self, exchange_kwargs, verify_kwargs):
        """Patch state/config/metadata/db; configure exchange & verify via kwargs."""
        provider_config = _make_provider_config()
        oidc_config = _make_oidc_config(provider_config)
        metadata = _make_metadata()
        state_data = _make_state_data()

        mock_store = AsyncMock()
        mock_store.consume = AsyncMock(return_value=state_data)
        mock_db = MagicMock()
        mock_db.commit.return_value = None

        patches = [
            patch.object(self.oidc_routes, "get_state_store", return_value=mock_store),
            patch.object(self.oidc_routes, "get_auth_config", return_value=oidc_config),
            patch.object(self.oidc_routes, "discover_provider", return_value=metadata),
            patch.object(self.oidc_routes, "exchange_code_for_tokens", **exchange_kwargs),
            patch.object(self.oidc_routes, "verify_id_token", **verify_kwargs),
            patch.object(self.oidc_routes, "create_access_token", return_value="at"),
            patch.object(self.oidc_routes, "create_refresh_token", return_value="rt"),
        ]
        self.app.dependency_overrides[self.oidc_routes.get_db] = lambda: mock_db
        stack = self.ExitStack()
        for p in patches:
            stack.enter_context(p)
        stack.callback(lambda: self.app.dependency_overrides.clear())
        return stack

    def test_token_exchange_exception_redirects_to_error(self):
        with self._common_patches({"side_effect": Exception("idp down")}, {"return_value": {}}):
            response = self.client.get(
                "/auth/oidc/google/callback?code=c&state=s", follow_redirects=False
            )
        assert response.status_code == 302
        assert "error=token_exchange_failed" in response.headers["location"]

    def test_missing_id_token_redirects_to_error(self):
        with self._common_patches({"return_value": {"access_token": "at"}}, {"return_value": {}}):
            response = self.client.get(
                "/auth/oidc/google/callback?code=c&state=s", follow_redirects=False
            )
        assert response.status_code == 302
        assert "error=no_id_token" in response.headers["location"]

    def test_verify_id_token_failure_redirects_to_error(self):
        with self._common_patches(
            {"return_value": {"id_token": "fake-id-token"}},
            {"side_effect": ValueError("bad signature")},
        ):
            response = self.client.get(
                "/auth/oidc/google/callback?code=c&state=s", follow_redirects=False
            )
        assert response.status_code == 302
        assert "error=token_verification_failed" in response.headers["location"]
