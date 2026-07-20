"""Integration tests for the entity-permission bootstrap (admin_api_key) path.

Supplements ``test_entity_permissions.py`` (which unit-tests the multi-user core
logic with SQLite) with REST-level coverage of the ``admin_api_key`` identity:
unclaimed writes, the unclaimed-not-listed-until-recount rule, recount surfacing
``owner_id=NULL`` rows, the bootstrap-cannot-create-entity guard, and admin
bypass on read/search.

The mem0 Memory instance is mocked; the relational DB (entities table) is the
real server DB (Postgres in the Docker test env). Multi-user (JWT) scenarios are
intentionally out of scope here — they are covered by the SQLite unit test, and
this file exercises only the admin_api_key identity (no JWT auth).

Run in the server Docker env: ``pytest tests/server/test_entity_permissions_api.py``.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from db import SessionLocal  # noqa: E402
from models import Entity, EntityPermission, User  # noqa: E402


class _PayloadRow:
    """Vector-store row stand-in exposing ``.payload`` for iter_payloads/recount."""

    def __init__(self, payload):
        self.id = payload.get("data", "row")
        self.payload = payload


API_KEY = "test-secret-key-12345"


@pytest.fixture
def _mock_memory(memory_patch):
    """Configure the mocked Memory.

    ``vector_store.list`` returns rows carrying payloads so ``POST /entities/recount``
    surfaces the unclaimed namespace.
    """
    memory_patch.get.return_value = {
        "id": "mem-1",
        "memory": "test memory",
        "user_id": "alice",
        "agent_id": None,
        "app_id": None,
        "run_id": None,
    }
    memory_patch.get_all.return_value = [{"id": "mem-1", "memory": "test memory", "user_id": "alice"}]
    memory_patch.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    memory_patch.search.return_value = [{"id": "mem-1", "memory": "test memory", "score": 0.9}]
    memory_patch.update.return_value = {"message": "Memory updated"}
    memory_patch.delete.return_value = None
    memory_patch.delete_all.return_value = {"message": "Memories deleted successfully!"}
    memory_patch.vector_store.list.return_value = [_PayloadRow({"user_id": "alice", "data": "test"})]
    yield memory_patch


def _clean_entities():
    """Delete entity_permissions + entities rows (isolation on the shared Postgres)."""
    with SessionLocal() as session:
        session.execute(delete(EntityPermission))
        session.execute(delete(Entity))
        session.commit()


class TestBootstrapEntityPermissions:
    """admin_api_key (bootstrap) path: unclaimed writes, recount, admin bypass."""

    @pytest.fixture(autouse=True)
    def _setup(self, _mock_memory, load_app):
        _clean_entities()
        self.mock = _mock_memory
        self.app = load_app({"ADMIN_API_KEY": API_KEY})
        self.client = TestClient(self.app)
        yield
        _clean_entities()

    @property
    def _auth(self):
        return {"X-API-Key": API_KEY}

    def test_bootstrap_memory_mutation_forbidden(self):
        """admin_api_key cannot author or mutate memories (governance + read-only).

        add / update / delete / delete_all all 403 at the endpoint gate, before
        any vector-store or ownership work.
        """
        # add
        resp = self.client.post(
            "/memories",
            json={
                "messages": [{"role": "user", "content": "I like pizza"}],
                "user_id": "alice",
            },
            headers=self._auth,
        )
        assert resp.status_code == 403, resp.text
        assert "cannot author or mutate memories" in resp.json()["detail"].lower()
        # update
        resp = self.client.put("/memories/mem-1", json={"text": "x"}, headers=self._auth)
        assert resp.status_code == 403, resp.text
        # delete single
        resp = self.client.delete("/memories/mem-1", headers=self._auth)
        assert resp.status_code == 403, resp.text
        # delete_all
        resp = self.client.delete("/memories?user_id=alice", headers=self._auth)
        assert resp.status_code == 403, resp.text

    def test_bootstrap_cannot_create_entity(self):
        """admin_api_key has no real identity to own with -> POST /entities 400."""
        resp = self.client.post(
            "/entities",
            json={"type": "user", "id": "alice"},
            headers=self._auth,
        )
        assert resp.status_code == 400, resp.text
        assert "real owner" in resp.json()["detail"].lower()

    def test_bootstrap_can_create_app(self):
        """admin_api_key can create an app entity owned by a real user (governance).

        The universal bootstrap-400 on POST /entities was relaxed to type-aware:
        bootstrap can create `app` (with a real owner_id) but not `user`.
        """
        with SessionLocal() as session:
            session.execute(delete(User).where(User.email == "app-owner@test.local"))
            owner = User(name="app-owner", email="app-owner@test.local", role="member")
            session.add(owner)
            session.commit()
            session.refresh(owner)
            owner_id = str(owner.id)

        try:
            resp = self.client.post(
                "/entities",
                json={"type": "app", "id": "myapp", "owner_id": owner_id},
                headers=self._auth,
            )
            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["type"] == "app"
            assert body["id"] == "myapp"
            assert body["owner"]["id"] == owner_id
        finally:
            # entities are wiped by the next _setup, but the user is not — clean it.
            with SessionLocal() as session:
                session.execute(delete(Entity).where(Entity.id == "myapp", Entity.type == "app"))
                session.execute(delete(User).where(User.email == "app-owner@test.local"))
                session.commit()

    def test_bootstrap_admin_bypass_get_memory(self):
        """admin_api_key can read any memory (admin bypass), incl. unclaimed scopes."""
        resp = self.client.get("/memories/mem-1", headers=self._auth)
        assert resp.status_code == 200, resp.text

    def test_bootstrap_admin_bypass_search(self):
        """admin_api_key search bypasses query permission (admin)."""
        resp = self.client.post(
            "/search",
            json={"query": "pizza", "user_id": "alice"},
            headers=self._auth,
        )
        assert resp.status_code == 200, resp.text

    def test_delete_nonexistent_entity_is_404(self):
        """DELETE /entities on a never-claimed namespace returns 404 (not 403/silent 200)."""
        resp = self.client.delete("/entities/user/never-claimed", headers=self._auth)
        assert resp.status_code == 404, resp.text

    # ------------------------------------------------------------------ #
    # Pagination envelope, UUID rejection, PATCH rename
    # ------------------------------------------------------------------ #
    def _make_owner(self, email="app-owner@test.local"):
        """Insert a member user for app-entity ownership (wipes a prior row first).

        The caller is responsible for cleaning up both the entities and the user
        in a ``finally`` block — ``_clean_entities`` only wipes the entities table.
        """
        with SessionLocal() as session:
            session.execute(delete(User).where(User.email == email))
            owner = User(name="app-owner", email=email, role="member")
            session.add(owner)
            session.commit()
            session.refresh(owner)
            owner_id = str(owner.id)
        return owner_id

    def _cleanup_owner_entities(self, entity_ids, email="app-owner@test.local"):
        with SessionLocal() as session:
            session.execute(delete(Entity).where(Entity.id.in_(entity_ids)))
            session.execute(delete(User).where(User.email == email))
            session.commit()

    def test_list_entities_returns_paginated_envelope(self):
        """GET /entities returns {count,next,previous,results}, not a bare array."""
        owner_id = self._make_owner()
        try:
            for eid in ("app-a", "app-b"):
                resp = self.client.post(
                    "/entities",
                    json={"type": "app", "id": eid, "owner_id": owner_id},
                    headers=self._auth,
                )
                assert resp.status_code == 201, resp.text

            resp = self.client.get("/entities?page_size=1", headers=self._auth)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert {"count", "next", "previous", "results"} <= set(body)
            assert isinstance(body["results"], list)
            assert body["count"] >= 2
            # page 1 with more to go → next link set, previous absent.
            assert body["next"] is not None
            assert body["previous"] is None
            assert "page=2" in body["next"]
            assert "page_size=1" in body["next"]

            # Admin viewing as another user keeps `view_as` in the next link.
            resp = self.client.get(f"/entities?view_as={owner_id}&page_size=1", headers=self._auth)
            assert resp.status_code == 200, resp.text
            scoped = resp.json()
            assert scoped["count"] == 2
            assert scoped["next"] is not None
            assert f"view_as={owner_id}" in scoped["next"]
        finally:
            self._cleanup_owner_entities(["app-a", "app-b"])

    def test_view_as_forbidden_for_non_admin(self):
        """GET /entities?view_as=<other> is 403 for a non-admin operator.

        Pairs with ``test_list_entities_returns_paginated_envelope`` (admin
        path): the ``view_as`` scope is admin-only, so a non-admin operator
        must be rejected before any scoping work runs. ``resolve_operator`` is
        patched, so the operator object is never read and any UUID triggers
        the guard — no DB row is needed.
        """
        import server.routers.entities as entities_router

        view_as_id = "00000000-0000-0000-0000-000000000001"
        with patch.object(entities_router, "resolve_operator", return_value=(object(), False)):
            resp = self.client.get(
                f"/entities?view_as={view_as_id}",
                headers=self._auth,
            )
        assert resp.status_code == 403, resp.text
        assert "only admins" in resp.json()["detail"].lower()

    def test_view_as_permission_reflects_target_perspective(self):
        """When an admin views as another user, permission badges reflect the
        target's perspective (owner/grant), not the admin's admin-bypass view.

        Regression: the response builder used to pass ``bypass=is_admin`` even
        under ``view_as``, labeling every row ``permission="admin"``. It must
        instead build from the target's identity with ``bypass=False``.
        """
        owner_id = self._make_owner()
        try:
            resp = self.client.post(
                "/entities",
                json={"type": "app", "id": "owned-app", "owner_id": owner_id},
                headers=self._auth,
            )
            assert resp.status_code == 201, resp.text

            resp = self.client.get(f"/entities?view_as={owner_id}", headers=self._auth)
            assert resp.status_code == 200, resp.text
            rows = resp.json()["results"]
            owned = [r for r in rows if r["id"] == "owned-app"]
            assert owned, "owned-app must be visible when viewing as its owner"
            assert owned[0]["permission"] == "owner"
            assert owned[0]["is_owner"] is True
        finally:
            self._cleanup_owner_entities(["owned-app"])

    def test_create_entity_rejects_uuid_user_id(self):
        """POST /entities with a UUID user_id is rejected; own UUID says 'already yours'."""
        other_uuid = "12345678-1234-1234-1234-123456789012"
        resp = self.client.post(
            "/entities",
            json={"type": "user", "id": other_uuid},
            headers=self._auth,
        )
        assert resp.status_code == 400, resp.text
        assert "cannot be created manually" in resp.json()["detail"].lower()

        # _BOOTSTRAP_ADMIN.id == uuid.UUID(int=0) — the operator's own UUID.
        own_uuid = "00000000-0000-0000-0000-000000000000"
        resp = self.client.post(
            "/entities",
            json={"type": "user", "id": own_uuid},
            headers=self._auth,
        )
        assert resp.status_code == 400, resp.text
        assert "already yours" in resp.json()["detail"].lower()

    def test_create_entity_with_name(self):
        """POST /entities {name} stores and returns the display name."""
        owner_id = self._make_owner()
        try:
            resp = self.client.post(
                "/entities",
                json={
                    "type": "app",
                    "id": "named-app",
                    "owner_id": owner_id,
                    "name": "my entity",
                },
                headers=self._auth,
            )
            assert resp.status_code == 201, resp.text
            assert resp.json()["name"] == "my entity"
        finally:
            self._cleanup_owner_entities(["named-app"])

    def test_update_entity_name_and_clear(self):
        """PATCH /entities/{type}/{id} sets, clears, and whitespace-trims the name."""
        owner_id = self._make_owner()
        try:
            create = self.client.post(
                "/entities",
                json={"type": "app", "id": "patch-app", "owner_id": owner_id},
                headers=self._auth,
            )
            assert create.status_code == 201, create.text

            resp = self.client.patch(
                "/entities/app/patch-app",
                json={"name": "X"},
                headers=self._auth,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] == "X"

            # Empty string clears to null.
            resp = self.client.patch(
                "/entities/app/patch-app",
                json={"name": ""},
                headers=self._auth,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] is None

            # Whitespace-only also clears to null (strip() or None).
            resp = self.client.patch(
                "/entities/app/patch-app",
                json={"name": " "},
                headers=self._auth,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] is None

            # Explicit JSON null clears too (REST clearing semantics via
            # model_fields_set — distinguishes "omit" from "set to null").
            resp = self.client.patch(
                "/entities/app/patch-app",
                json={"name": "label"},
                headers=self._auth,
            )
            assert resp.json()["name"] == "label"
            resp = self.client.patch(
                "/entities/app/patch-app",
                json={"name": None},
                headers=self._auth,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] is None
        finally:
            self._cleanup_owner_entities(["patch-app"])
