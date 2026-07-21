"""Integration tests for the ``/auth/*`` router.

Covers the endpoints with zero prior coverage: ``setup-status``, ``register``
(first-admin + registration-closed + password length), ``login`` (success /
wrong password / unknown email / OIDC-only account), ``refresh`` (success,
JTI single-use replay, wrong token type, invalid token), ``me``, ``update_me``
(name / email change / email collision 409 / OIDC-managed email 403),
``change-password`` (success / wrong current / OIDC account 400 / short new),
and ``onboarding-complete``.

Auth-enabled (ADMIN_API_KEY set) so JWT bearer auth is exercised end-to-end.
The rate limiter is disabled so repeated register/login/refresh calls in one
test do not trip the 5/10/20-per-minute limits. Telemetry capture calls are
patched to no-ops so no network/event emission occurs.

Run in the server Docker env (Postgres required)::

    pytest tests/server/test_auth_routes.py
"""

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from auth import hash_password  # noqa: E402
from db import SessionLocal  # noqa: E402
from helpers import (  # noqa: E402
    bearer_header,
    clean_auth_state,
    disable_app_rate_limiters,
)
from models import User  # noqa: E402

API_KEY = "test-secret-key-12345"


def _make_user(*, email, role="member", password=None, auth_provider="local"):
    """Insert a user directly (bypasses /auth/register, which is first-admin-only)."""
    with SessionLocal() as session:
        from sqlalchemy import delete

        session.execute(delete(User).where(User.email == email))
        user = User(
            name=email.split("@")[0],
            email=email,
            password_hash=hash_password(password) if password else None,
            role=role,
            auth_provider=auth_provider,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


class TestAuthRoutes:
    @pytest.fixture(autouse=True)
    def _setup(self, memory_patch, load_app):
        clean_auth_state()
        import rate_limit

        prev_enabled = rate_limit.limiter.enabled
        rate_limit.limiter.enabled = False
        try:
            with patch("routers.auth.capture_admin_registered"), patch("routers.auth.capture_onboarding_completed"):
                self.app = load_app({"ADMIN_API_KEY": API_KEY})
                self.client = TestClient(self.app)
                disable_app_rate_limiters(self.app)
                yield
        finally:
            rate_limit.limiter.enabled = prev_enabled
            clean_auth_state()

    # -- helpers -----------------------------------------------------------
    def _register(self, name="Admin", email="admin@example.com", password="hunter123"):
        return self.client.post(
            "/auth/register",
            json={"name": name, "email": email, "password": password},
        )

    # -- setup-status ------------------------------------------------------
    def test_setup_status_true_when_no_users(self):
        resp = self.client.get("/auth/setup-status")
        assert resp.status_code == 200
        assert resp.json() == {"needsSetup": True}

    def test_setup_status_false_after_register(self):
        self._register()
        assert self.client.get("/auth/setup-status").json() == {"needsSetup": False}

    # -- register ----------------------------------------------------------
    def test_register_first_admin_returns_tokens(self):
        resp = self._register()
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]
        assert body["refresh_token"]
        # The first registered user is an admin.
        with SessionLocal() as s:
            user = s.scalar(select_first_admin())
            assert user is not None and user.role == "admin"

    def test_register_closed_once_a_user_exists(self):
        assert self._register().status_code == 200
        second = self._register(email="other@example.com")
        assert second.status_code == 403
        assert "closed" in second.json()["detail"].lower()

    def test_register_short_password_rejected(self):
        resp = self._register(password="short")
        assert resp.status_code == 400
        assert "at least 8" in resp.json()["detail"].lower()

    # -- login -------------------------------------------------------------
    def test_login_success_returns_tokens(self):
        self._register(email="login@example.com", password="hunter123")
        resp = self.client.post("/auth/login", json={"email": "login@example.com", "password": "hunter123"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["access_token"]

    def test_login_wrong_password(self):
        self._register(email="login@example.com", password="hunter123")
        resp = self.client.post("/auth/login", json={"email": "login@example.com", "password": "wrong-pw-99"})
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()

    def test_login_unknown_email(self):
        resp = self.client.post("/auth/login", json={"email": "nobody@example.com", "password": "hunter123"})
        assert resp.status_code == 401

    def test_login_oidc_account_rejected_with_provider_hint(self):
        _make_user(email="oidc@example.com", auth_provider="google", password=None)
        resp = self.client.post("/auth/login", json={"email": "oidc@example.com", "password": "anything-here"})
        assert resp.status_code == 401
        assert "identity provider" in resp.json()["detail"].lower()

    # -- refresh + JTI single-use -----------------------------------------
    def _refresh_tokens(self, email="admin@example.com"):
        r = self._register(email=email)
        return r.json()["refresh_token"]

    def test_refresh_success_returns_new_tokens(self):
        rt = self._refresh_tokens()
        resp = self.client.post("/auth/refresh", json={"refresh_token": rt})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["access_token"]
        assert body["refresh_token"] != rt  # rotated

    def test_refresh_jti_single_use_replay_rejected(self):
        """Replaying a consumed refresh token must 401 (JTI marked used)."""
        rt = self._refresh_tokens()
        first = self.client.post("/auth/refresh", json={"refresh_token": rt})
        assert first.status_code == 200
        replay = self.client.post("/auth/refresh", json={"refresh_token": rt})
        assert replay.status_code == 401

    def test_refresh_wrong_token_type_rejected(self):
        # An access token (type=access, no jti) must not be accepted as a refresh token.
        reg = self._register(email="typetest@example.com")
        resp = self.client.post("/auth/refresh", json={"refresh_token": reg.json()["access_token"]})
        assert resp.status_code == 401
        assert "type" in resp.json()["detail"].lower()

    def test_refresh_invalid_token_rejected(self):
        resp = self.client.post("/auth/refresh", json={"refresh_token": "not-a-jwt"})
        assert resp.status_code == 401

    # -- me / update_me ----------------------------------------------------
    def test_me_requires_auth(self):
        assert self.client.get("/auth/me").status_code == 401

    def test_me_returns_authenticated_user(self):
        self._register(email="me@example.com", name="Me Person")
        tok = self._login("me@example.com", "hunter123")
        resp = self.client.get("/auth/me", headers=bearer_header(tok))
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "me@example.com"
        assert body["name"] == "Me Person"
        assert body["role"] == "admin"

    def test_update_me_name(self):
        self._register(email="me@example.com")
        tok = self._login("me@example.com", "hunter123")
        resp = self.client.patch("/auth/me", json={"name": "Renamed"}, headers=bearer_header(tok))
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"

    def test_update_me_email_change(self):
        self._register(email="me@example.com")
        tok = self._login("me@example.com", "hunter123")
        resp = self.client.patch("/auth/me", json={"email": "new@example.com"}, headers=bearer_header(tok))
        assert resp.status_code == 200
        assert resp.json()["email"] == "new@example.com"

    def test_update_me_email_collision_409(self):
        self._register(email="a@example.com")
        _make_user(email="b@example.com", password="hunter123")
        tok = self._login("a@example.com", "hunter123")
        resp = self.client.patch("/auth/me", json={"email": "b@example.com"}, headers=bearer_header(tok))
        assert resp.status_code == 409
        assert "already in use" in resp.json()["detail"].lower()

    def test_update_me_oidc_email_change_forbidden(self):
        user = _make_user(email="oidc@example.com", auth_provider="google", password=None)
        # OIDC users have no password -> can't /auth/login; mint an access token directly.
        from auth import create_access_token

        tok = create_access_token(str(user.id), user.role)
        resp = self.client.patch("/auth/me", json={"email": "newoidc@example.com"}, headers=bearer_header(tok))
        assert resp.status_code == 403
        assert "identity provider" in resp.json()["detail"].lower()

    # -- change-password ---------------------------------------------------
    def test_change_password_success(self):
        self._register(email="cp@example.com", password="hunter123")
        tok = self._login("cp@example.com", "hunter123")
        resp = self.client.post(
            "/auth/change-password",
            json={"current_password": "hunter123", "new_password": "newpass456"},
            headers=bearer_header(tok),
        )
        assert resp.status_code == 200, resp.text
        # Old password no longer logs in.
        old = self.client.post("/auth/login", json={"email": "cp@example.com", "password": "hunter123"})
        assert old.status_code == 401
        new = self.client.post("/auth/login", json={"email": "cp@example.com", "password": "newpass456"})
        assert new.status_code == 200

    def test_change_password_wrong_current(self):
        self._register(email="cp@example.com", password="hunter123")
        tok = self._login("cp@example.com", "hunter123")
        resp = self.client.post(
            "/auth/change-password",
            json={"current_password": "wrong-pw-99", "new_password": "newpass456"},
            headers=bearer_header(tok),
        )
        assert resp.status_code == 401

    def test_change_password_oidc_account_400(self):
        user = _make_user(email="oidc@example.com", auth_provider="google", password=None)
        from auth import create_access_token

        tok = create_access_token(str(user.id), user.role)
        resp = self.client.post(
            "/auth/change-password",
            json={"current_password": "x", "new_password": "newpass456"},
            headers=bearer_header(tok),
        )
        assert resp.status_code == 400
        assert "no password" in resp.json()["detail"].lower()

    def test_change_password_short_new(self):
        self._register(email="cp@example.com", password="hunter123")
        tok = self._login("cp@example.com", "hunter123")
        resp = self.client.post(
            "/auth/change-password",
            json={"current_password": "hunter123", "new_password": "short"},
            headers=bearer_header(tok),
        )
        assert resp.status_code == 400

    # -- onboarding --------------------------------------------------------
    def test_onboarding_complete_requires_auth(self):
        assert self.client.post("/auth/onboarding-complete", json={"use_case": "chatbot"}).status_code == 401

    def test_onboarding_complete_ok(self):
        self._register(email="onb@example.com")
        tok = self._login("onb@example.com", "hunter123")
        resp = self.client.post(
            "/auth/onboarding-complete",
            json={"use_case": "chatbot"},
            headers=bearer_header(tok),
        )
        assert resp.status_code == 200

    # -- internal helper ---------------------------------------------------
    def _login(self, email, password):
        r = self.client.post("/auth/login", json={"email": email, "password": password})
        assert r.status_code == 200, r.text
        return r.json()["access_token"]


# --------------------------------------------------------------------------- #
# tiny query helpers (kept module-local to avoid touching other test files)
# --------------------------------------------------------------------------- #
def select_first_admin():
    from sqlalchemy import select

    return select(User).where(User.role == "admin").order_by(User.created_at.asc())