"""Integration tests for the ``/users`` and ``/requests`` routers.

Both had zero prior coverage. Exercises:

``/users`` (admin-only, paginated):
- admin -> 200 paginated envelope; member -> 403; unauth -> 401.

``/requests`` (request-log listing):
- admin sees all api_key/admin_api_key logs; member sees only their own,
- bearer/none auth-type logs are excluded (only api_key/admin_api_key surface),
- ``limit`` clamped to [1, 200] (out-of-range -> 422).

Auth-enabled (JWT bearer). Rate limiter disabled.

Run in the server Docker env:: pytest tests/server/test_users_requests_routes.py
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from auth import hash_password  # noqa: E402
from db import SessionLocal  # noqa: E402
from helpers import (  # noqa: E402
    bearer_header,
    clean_auth_state,
    disable_app_rate_limiters,
    register_first_admin,
)
from models import RequestLog, User  # noqa: E402

API_KEY = "test-secret-key-12345"


def _make_member(*, email):
    with SessionLocal() as session:
        session.execute(delete(User).where(User.email == email))
        user = User(
            name=email.split("@")[0],
            email=email,
            password_hash=hash_password("hunter123"),
            role="member",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _login(client, email):
    resp = client.post("/auth/login", json={"email": email, "password": "hunter123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _insert_log(*, auth_type, user_id, path="/memories/x", status_code=200):
    with SessionLocal() as session:
        session.add(
            RequestLog(
                method="GET",
                path=path,
                status_code=status_code,
                latency_ms=12.5,
                auth_type=auth_type,
                user_id=user_id,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()


class TestUsersRoutes:
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
                self.admin_token = register_first_admin(self.client)
                self.member = _make_member(email="member@example.com")
                self.member_token = _login(self.client, "member@example.com")
                yield
        finally:
            rate_limit.limiter.enabled = prev
            clean_auth_state()

    def test_admin_can_list(self):
        resp = self.client.get("/users", headers=bearer_header(self.admin_token))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert {"count", "next", "previous", "results"} <= set(body)
        # admin (registered) + member (created) = 2
        assert body["count"] >= 2

    def test_member_forbidden_403(self):
        resp = self.client.get("/users", headers=bearer_header(self.member_token))
        assert resp.status_code == 403

    def test_unauth_401(self):
        assert self.client.get("/users").status_code == 401

    def test_pagination_envelope(self):
        # Create extra members so pagination has multiple pages at page_size=2.
        for i in range(3):
            _make_member(email=f"extra{i}@example.com")
        resp = self.client.get("/users?page=1&page_size=2", headers=bearer_header(self.admin_token))
        body = resp.json()
        assert len(body["results"]) == 2
        assert body["next"] is not None  # more pages
        assert body["previous"] is None
        # Follow the next link's page param.
        resp2 = self.client.get("/users?page=2&page_size=2", headers=bearer_header(self.admin_token))
        body2 = resp2.json()
        assert len(body2["results"]) >= 1
        assert body2["previous"] is not None

    def test_page_size_clamp(self):
        # page_size > 200 -> 422 (le=200); page_size < 1 -> 422 (ge=1).
        assert self.client.get("/users?page_size=500", headers=bearer_header(self.admin_token)).status_code == 422
        assert self.client.get("/users?page_size=0", headers=bearer_header(self.admin_token)).status_code == 422


class TestRequestsRoutes:
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
                self.admin_token = register_first_admin(self.client)
                self.member = _make_member(email="member@example.com")
                self.member_token = _login(self.client, "member@example.com")
                yield
        finally:
            rate_limit.limiter.enabled = prev
            clean_auth_state()

    def _seed_logs(self):
        # Admin-owned api_key log + member-owned api_key log + a bearer log (excluded).
        with SessionLocal() as s:
            admin_id = s.scalar(select(User.id).where(User.email == "admin@example.com"))
        _insert_log(auth_type="api_key", user_id=admin_id, path="/memories/admin")
        _insert_log(auth_type="api_key", user_id=self.member.id, path="/memories/member")
        _insert_log(auth_type="bearer", user_id=self.member.id, path="/memories/bearer")
        _insert_log(auth_type="admin_api_key", user_id=None, path="/configure")
        return admin_id

    def test_admin_sees_all_api_key_logs(self):
        self._seed_logs()
        resp = self.client.get("/requests", headers=bearer_header(self.admin_token))
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        paths = {r["path"] for r in rows}
        assert "/memories/admin" in paths
        assert "/memories/member" in paths
        assert "/configure" in paths
        # bearer-auth logs never surface here.
        assert "/memories/bearer" not in paths

    def test_member_sees_only_own_logs(self):
        self._seed_logs()
        resp = self.client.get("/requests", headers=bearer_header(self.member_token))
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        paths = {r["path"] for r in rows}
        assert paths == {"/memories/member"}, f"member must only see own logs, got {paths}"

    def test_member_does_not_see_admin_logs(self):
        self._seed_logs()
        resp = self.client.get("/requests", headers=bearer_header(self.member_token))
        rows = resp.json()
        assert all(r["path"] != "/memories/admin" for r in rows)
        assert all(r["path"] != "/configure" for r in rows)

    def test_limit_clamp(self):
        self._seed_logs()
        assert self.client.get("/requests?limit=0", headers=bearer_header(self.admin_token)).status_code == 422
        assert self.client.get("/requests?limit=500", headers=bearer_header(self.admin_token)).status_code == 422

    def test_requires_auth(self):
        assert self.client.get("/requests").status_code == 401