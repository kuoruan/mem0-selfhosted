"""Integration tests for the ``/api-keys`` router (list / create / revoke).

These endpoints had zero prior coverage. Exercises:
- create returns the full key exactly once (never retrievable again),
- list shows the prefix (not the full key) and excludes revoked keys,
- revoke own key succeeds and 404s for a foreign owner's key (IDOR guard),
- invalid key_id uuid -> 404, double-revoke -> 400.

Auth-enabled (JWT bearer). Rate limiter disabled.

Run in the server Docker env:: pytest tests/server/test_api_keys_routes.py
"""

import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from auth import generate_api_key, hash_password  # noqa: E402
from db import SessionLocal  # noqa: E402
from helpers import (  # noqa: E402
    bearer_header,
    clean_auth_state,
    disable_app_rate_limiters,
    register_first_admin,
)
from models import APIKey, User  # noqa: E402

API_KEY = "test-secret-key-12345"


def _make_user_with_key(*, email):
    """Insert a member user directly and create one API key owned by them; return (user_id, key_id)."""
    with SessionLocal() as session:
        session.execute(delete(User).where(User.email == email))
        user = User(name=email.split("@")[0], email=email, password_hash=hash_password("hunter123"), role="member")
        session.add(user)
        session.flush()
        _, prefix, key_hash = generate_api_key()
        key = APIKey(key_prefix=prefix, key_hash=key_hash, label="foreign", created_by=user.id)
        session.add(key)
        session.commit()
        session.refresh(key)
        return user.id, key.id


class TestApiKeysRoutes:
    @pytest.fixture(autouse=True)
    def _setup(self, memory_patch, load_app):
        clean_auth_state()
        import rate_limit

        prev = rate_limit.limiter.enabled
        rate_limit.limiter.enabled = False
        try:
            with patch("routers.auth.capture_admin_registered"), patch("routers.auth.capture_onboarding_completed"):
                self.app = load_app({"ADMIN_API_KEY": API_KEY})
                self.client = TestClient(self.app)
                disable_app_rate_limiters(self.app)
                self.token = register_first_admin(self.client)
                yield
        finally:
            rate_limit.limiter.enabled = prev
            clean_auth_state()

    def test_list_empty(self):
        resp = self.client.get("/api-keys", headers=bearer_header(self.token))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_returns_full_key_once_and_persists_prefix(self):
        resp = self.client.post("/api-keys", json={"label": "ci-key"}, headers=bearer_header(self.token))
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["label"] == "ci-key"
        assert body["key"]  # full key returned once
        assert body["key_prefix"]
        assert body["key"].startswith(body["key_prefix"])

        # The full key never reappears in list — only prefix.
        listed = self.client.get("/api-keys", headers=bearer_header(self.token)).json()
        assert len(listed) == 1
        assert listed[0]["key_prefix"] == body["key_prefix"]
        assert listed[0]["label"] == "ci-key"
        assert "key" not in listed[0]  # no full key

    def test_revoke_own_key(self):
        kid = self.client.post("/api-keys", json={"label": "tokill"}, headers=bearer_header(self.token)).json()["id"]
        resp = self.client.delete(f"/api-keys/{kid}", headers=bearer_header(self.token))
        assert resp.status_code == 200, resp.text
        # Revoked key no longer in list.
        listed = self.client.get("/api-keys", headers=bearer_header(self.token)).json()
        assert all(k["id"] != kid for k in listed)

    def test_revoke_foreign_key_404(self):
        """A user may not revoke another user's API key (owner check -> 404, not 403)."""
        _, foreign_kid = _make_user_with_key(email="other@example.com")
        resp = self.client.delete(f"/api-keys/{foreign_kid}", headers=bearer_header(self.token))
        assert resp.status_code == 404

    def test_revoke_invalid_uuid_404(self):
        resp = self.client.delete("/api-keys/not-a-uuid", headers=bearer_header(self.token))
        assert resp.status_code == 404

    def test_double_revoke_400(self):
        kid = self.client.post("/api-keys", json={"label": "twice"}, headers=bearer_header(self.token)).json()["id"]
        assert self.client.delete(f"/api-keys/{kid}", headers=bearer_header(self.token)).status_code == 200
        second = self.client.delete(f"/api-keys/{kid}", headers=bearer_header(self.token))
        assert second.status_code == 400
        assert "already revoked" in second.json()["detail"].lower()

    def test_revoked_key_excluded_from_list(self):
        kid = self.client.post("/api-keys", json={"label": "a"}, headers=bearer_header(self.token)).json()["id"]
        self.client.post("/api-keys", json={"label": "b"}, headers=bearer_header(self.token))
        self.client.delete(f"/api-keys/{kid}", headers=bearer_header(self.token))
        listed = self.client.get("/api-keys", headers=bearer_header(self.token)).json()
        assert len(listed) == 1
        assert listed[0]["label"] == "b"

    def test_requires_auth(self):
        assert self.client.get("/api-keys").status_code == 401
        assert self.client.post("/api-keys", json={"label": "x"}).status_code == 401

    def test_revoke_nonexistent_uuid_404(self):
        resp = self.client.delete(f"/api-keys/{uuid.uuid4()}", headers=bearer_header(self.token))
        assert resp.status_code == 404
