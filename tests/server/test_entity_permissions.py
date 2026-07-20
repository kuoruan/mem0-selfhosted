"""Unit tests for the entity_permissions authorization service layer (v2).

Covers canonicalization, query branch extraction (OR/AND/NOT), hierarchical user
namespace (prefix matching), agent/run parent-based ownership, app admin-only creation,
permission checks (owner vs admin boundaries), grant/revoke (only user/app), transfer
cascade, and bulk admin validation.
"""

import uuid

import pytest

pytest.importorskip("server.entity_permissions", reason="server modules not installed")

import server.entity_permissions as ep  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from server.db import Base  # noqa: E402
from server.models import Entity, EntityPermission, User  # noqa: E402


class _FakeVectorStore:
    """Minimal vector-store stand-in for delete prescans."""

    def __init__(self, store):
        self._store = store

    def get(self, memory_id):
        payload = self._store.get(memory_id)

        class _Row:
            pass

        row = _Row()
        row.id = memory_id
        row.payload = payload
        return row

    def update(self, memory_id, vector=None, payload=None):
        if memory_id in self._store and payload is not None:
            self._store[memory_id].update(payload)


class FakeMemory:
    """In-memory stand-in for the mem0 Memory instance."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def add(self, memory_id, payload):
        self._store[memory_id] = dict(payload)

    def get(self, memory_id):
        return self._store.get(memory_id)

    def get_all(self, filters=None, top_k=None):
        filters = filters or {}
        return [
            {"id": mid, **payload}
            for mid, payload in self._store.items()
            if all(payload.get(k) == v for k, v in filters.items())
        ]

    def delete_all(self, **kwargs):
        self._store = {
            mid: payload
            for mid, payload in self._store.items()
            if not all(payload.get(k) == v for k, v in kwargs.items())
        }

    @property
    def vector_store(self):
        return _FakeVectorStore(self._store)


@pytest.fixture
def db():
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def fake_memory():
    return FakeMemory()


@pytest.fixture(autouse=True)
def _mock_memory(monkeypatch, fake_memory):
    monkeypatch.setattr(ep, "get_memory_instance", lambda: fake_memory)


@pytest.fixture
def make_user(db):
    counter = {"n": 0}

    def _make(role: str = "member") -> User:
        counter["n"] += 1
        u = User(
            name=f"user{counter['n']}",
            email=f"user{counter['n']}@test.local",
            role=role,
        )
        db.add(u)
        db.flush()
        return u

    return _make


@pytest.fixture
def admin_user(make_user):
    return make_user(role="admin")


# --------------------------------------------------------------------------- #
# canonicalize_entity_id
# --------------------------------------------------------------------------- #
def test_canonicalize_strips_and_rejects_empty():
    assert ep.canonicalize_entity_id("agent", "  bot  ") == "bot"
    with pytest.raises(HTTPException) as exc:
        ep.canonicalize_entity_id("agent", "   ")
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        ep.canonicalize_entity_id("agent", 123)  # type: ignore[arg-type]


def test_canonicalize_user_uuid():
    u = "550e8400-e29b-41d4-a716-446655440000"
    assert ep.canonicalize_entity_id("user", u.upper()) == u
    assert ep.canonicalize_entity_id("user", "alice") == "alice"


def test_canonicalize_user_uuid_sub_namespace():
    """UUID in sub-namespace (e.g. UUID:laptop) has the UUID segment canonicalized."""
    u = "550e8400-e29b-41d4-a716-446655440000"
    assert ep.canonicalize_entity_id("user", f"{u.upper()}:laptop") == f"{u}:laptop"
    assert ep.canonicalize_entity_id("user", f"{u}:laptop:phone") == f"{u}:laptop:phone"
    # Non-UUID first segment is left as-is
    assert ep.canonicalize_entity_id("user", "alice:laptop") == "alice:laptop"


# --------------------------------------------------------------------------- #
# extract_query_scope_branches / check_query_permission
# --------------------------------------------------------------------------- #
def test_branches_flat_and_or_not():
    flat, has_not = ep.extract_query_scope_branches({"user_id": "a", "agent_id": "b"})
    assert flat == [{"user": "a", "agent": "b"}]
    assert has_not is False

    and_tree, _ = ep.extract_query_scope_branches({"AND": [{"user_id": "a"}, {"agent_id": "b"}]})
    assert and_tree == [{"user": "a", "agent": "b"}]

    or_tree, _ = ep.extract_query_scope_branches({"OR": [{"user_id": "a"}, {"agent_id": "b"}]})
    assert {tuple(sorted(d.items())) for d in or_tree} == {
        (("user", "a"),),
        (("agent", "b"),),
    }

    _, has_not = ep.extract_query_scope_branches({"NOT": {"user_id": "x"}})
    assert has_not is True


def test_check_query_permission_not_rejects_non_admin(db, make_user):
    user = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.check_query_permission({"NOT": {"user_id": "x"}}, user.id, db)
    assert exc.value.status_code == 403


def test_check_query_permission_not_admin_passes(db, admin_user):
    ep.check_query_permission({"NOT": {"user_id": "x"}}, admin_user.id, db, bypass=True)


def test_check_query_permission_or_branch_unauthorized(db, make_user):
    user = make_user()
    victim = uuid.uuid4()
    filters = {"OR": [{"user_id": str(user.id)}, {"user_id": str(victim)}]}
    with pytest.raises(HTTPException) as exc:
        ep.check_query_permission(filters, user.id, db)
    assert exc.value.status_code == 403


def test_check_query_permission_own_scope_passes(db, make_user):
    user = make_user()
    ep.check_query_permission({"user_id": str(user.id)}, user.id, db)


# --------------------------------------------------------------------------- #
# ensure_entity_owner — user (hierarchical namespace)
# --------------------------------------------------------------------------- #
def test_ensure_user_entity_first_claim(db, make_user):
    user = make_user()
    entity = ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    assert entity.owner_id == user.id
    assert entity.id == "alice"
    assert entity.parent_pk is None


def test_ensure_user_entity_uuid_reserved(db, make_user):
    a = make_user()
    b = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("user", str(b.id), a.id, db)
    assert exc.value.status_code == 403
    # unregistered UUID is also rejected
    with pytest.raises(HTTPException):
        ep.ensure_entity_owner("user", str(uuid.uuid4()), a.id, db)


def test_ensure_user_entity_sub_namespace(db, make_user):
    """Creating user/A:B:C auto-creates user/A and user/A:B:C."""
    user = make_user()
    entity = ep.ensure_entity_owner("user", "alice:laptop", user.id, db)
    db.commit()
    assert entity.id == "alice:laptop"
    assert entity.owner_id == user.id

    # Top-level should also exist
    top = ep._get_user_entity_or_none("alice", db)
    assert top is not None
    assert top.owner_id == user.id


def test_ensure_user_entity_sub_namespace_prefix_owned_by_other(db, make_user):
    """Cannot create user/A:B if user/A is owned by another user."""
    a = make_user()
    b = make_user()
    ep.ensure_entity_owner("user", "alice", a.id, db)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("user", "alice:laptop", b.id, db)
    assert exc.value.status_code == 403


def test_ensure_user_entity_ui_create_no_colon(db, make_user):
    """UI-created user entities cannot contain ':'."""
    user = make_user()
    entity = ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    assert entity.id == "alice"
    assert ":" not in entity.id


def test_ensure_user_entity_claims_orphaned_toplevel(db, make_user):
    """An orphaned top-level entity (owner=None, e.g. after owner deletion) is
    claimable by a non-admin rather than returning 403."""
    db.add(Entity(type="user", id="alice", owner_id=None))
    db.flush()
    user = make_user()
    entity = ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    assert entity.id == "alice"
    assert entity.owner_id == user.id


def test_orphan_subnamespace_under_owned_toplevel_is_claimable(db, make_user):
    """An orphaned sub-namespace (owner=None) under a top-level entity the caller
    owns is claimable by the caller (orphan is skipped, owned prefix wins)."""
    owner = make_user()
    db.add(Entity(type="user", id="alice", owner_id=owner.id))
    db.add(Entity(type="user", id="alice:laptop", owner_id=None))
    db.flush()
    entity = ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    db.commit()
    assert entity.id == "alice:laptop"
    assert entity.owner_id == owner.id


def test_orphan_subnamespace_under_other_user_toplevel_rejected(db, make_user):
    """An orphaned sub-namespace under another user's top-level entity cannot be
    claimed by a non-owner."""
    owner = make_user()
    other = make_user()
    db.add(Entity(type="user", id="alice", owner_id=owner.id))
    db.add(Entity(type="user", id="alice:laptop", owner_id=None))
    db.flush()
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("user", "alice:laptop", other.id, db)
    assert exc.value.status_code == 403


def test_check_permission_skips_orphan_falls_to_owned_prefix(db, make_user):
    """check_entity_permission on an orphaned sub-namespace follows the longest
    *owned* prefix (orphan skipped, not treated as admin-only)."""
    owner = make_user()
    other = make_user()
    db.add(Entity(type="user", id="alice", owner_id=owner.id))
    db.add(Entity(type="user", id="alice:laptop", owner_id=None))
    db.flush()
    assert ep.check_entity_permission("user", "alice:laptop", owner.id, "read", db) is True
    assert ep.check_entity_permission("user", "alice:laptop", other.id, "read", db) is False


# --------------------------------------------------------------------------- #
# ensure_entity_owner — app (admin only)
# --------------------------------------------------------------------------- #
def test_ensure_app_entity_requires_existing(db, make_user):
    """Non-admin creates app -> 403 if app doesn't exist."""
    user = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("app", "my-repo", user.id, db)
    assert exc.value.status_code == 403


def test_ensure_app_entity_admin_creates(db, make_user):
    """Admin creates app via POST /entities, then writes work."""
    owner = make_user()
    ep._create_entity_row("app", "my-repo", None, owner.id, None, db)
    db.commit()
    # Now owner can write
    entity = ep.ensure_entity_owner("app", "my-repo", owner.id, db)
    assert entity.owner_id == owner.id


# --------------------------------------------------------------------------- #
# ensure_entity_owner — agent/run (parent-based)
# --------------------------------------------------------------------------- #
def test_ensure_agent_entity(db, make_user):
    user = make_user()
    # First create the user entity
    ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    # Then create agent under it
    entity = ep.ensure_entity_owner("agent", "riley", user.id, db, parent_entity_id="alice")
    db.commit()
    assert entity.type == "agent"
    assert entity.id == "riley"
    assert entity.owner_id == user.id
    assert entity.parent_pk is not None


def test_ensure_agent_entity_different_users(db, make_user):
    """Two users can have agent/riley independently."""
    a = make_user()
    b = make_user()
    ep.ensure_entity_owner("user", "alice", a.id, db)
    ep.ensure_entity_owner("user", "bob", b.id, db)
    db.commit()
    agent_a = ep.ensure_entity_owner("agent", "riley", a.id, db, parent_entity_id="alice")
    agent_b = ep.ensure_entity_owner("agent", "riley", b.id, db, parent_entity_id="bob")
    db.commit()
    assert agent_a.pk != agent_b.pk
    assert agent_a.owner_id == a.id
    assert agent_b.owner_id == b.id


def test_ensure_agent_entity_wrong_parent(db, make_user):
    """Cannot create agent under another user's user entity."""
    a = make_user()
    b = make_user()
    ep.ensure_entity_owner("user", "alice", a.id, db)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("agent", "riley", b.id, db, parent_entity_id="alice")
    assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# check_entity_permission / check_memory_scope_permission
# --------------------------------------------------------------------------- #
def test_check_entity_permission_owner(db, make_user):
    user = make_user()
    ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    for level in ("read", "write", "admin"):
        assert ep.check_entity_permission("user", "alice", user.id, level, db)


def test_check_entity_permission_granted_levels(db, make_user):
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "alice", other.id, "write",
        operator_id=owner.id, bypass=False, db=db,
    )
    assert ep.check_entity_permission("user", "alice", other.id, "read", db)
    assert ep.check_entity_permission("user", "alice", other.id, "write", db)
    # admin not granted
    assert not ep.check_entity_permission("user", "alice", other.id, "admin", db)


def test_check_entity_permission_unclaimed(db, make_user):
    user = make_user()
    assert not ep.check_entity_permission("user", "nobody", user.id, "read", db)


def test_check_entity_permission_user_owns_own_uuid(db, make_user):
    user = make_user()
    assert ep.check_entity_permission("user", str(user.id), user.id, "admin", db)


def test_check_entity_permission_user_owns_own_uuid_subnamespace(db, make_user):
    """A user owns sub-namespaces under their own UUID even before any entity
    row is created (lazy first-claim semantics)."""
    user = make_user()
    # No entity row exists for <uuid> or <uuid>:laptop — query must still pass.
    assert ep.check_entity_permission("user", f"{user.id}:laptop", user.id, "write", db)
    assert ep.check_entity_permission("user", f"{user.id}:a:b:c", user.id, "admin", db)
    # A different user must NOT be authorized on this namespace.
    other = make_user()
    assert not ep.check_entity_permission("user", f"{user.id}:laptop", other.id, "read", db)


def test_check_entity_permission_agent(db, make_user):
    user = make_user()
    ep.ensure_entity_owner("user", "alice", user.id, db)
    ep.ensure_entity_owner("agent", "riley", user.id, db, parent_entity_id="alice")
    db.commit()
    assert ep.check_entity_permission("agent", "riley", user.id, "read", db, parent_entity_id="alice")
    assert ep.check_entity_permission("agent", "riley", user.id, "admin", db, parent_entity_id="alice")


def test_check_entity_permission_agent_no_parent_denied(db, make_user):
    user = make_user()
    assert not ep.check_entity_permission("agent", "riley", user.id, "read", db)


def test_check_entity_permission_hierarchical_user_prefix(db, make_user):
    """Owning user/A means owning user/A:B."""
    user = make_user()
    ep.ensure_entity_owner("user", "alice", user.id, db)
    db.commit()
    # user/A:B is not created but prefix matches
    assert ep.check_entity_permission("user", "alice:laptop", user.id, "read", db)


def test_check_entity_permission_grant_inherits_to_subnamespace(db, make_user):
    """A grant on user/alice covers user/alice:laptop (created sub-namespace)."""
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "alice", other.id, "write",
        operator_id=owner.id, bypass=False, db=db,
    )
    # write grant on alice inherits to the created alice:laptop entity
    assert ep.check_entity_permission("user", "alice:laptop", other.id, "write", db)
    assert ep.check_entity_permission("user", "alice:laptop", other.id, "read", db)
    # admin not granted -> denied even on sub-namespace
    assert not ep.check_entity_permission("user", "alice:laptop", other.id, "admin", db)


def test_check_entity_permission_grant_inherits_to_unclaimed_subnamespace(db, make_user):
    """A grant on user/alice covers an unclaimed user/alice:laptop via prefix."""
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "alice", other.id, "read",
        operator_id=owner.id, bypass=False, db=db,
    )
    # alice:laptop entity does not exist; grant on alice still applies via prefix
    assert ep.check_entity_permission("user", "alice:laptop", other.id, "read", db)
    # write not granted -> denied
    assert not ep.check_entity_permission("user", "alice:laptop", other.id, "write", db)


def test_scope_read_or_write_and(db, make_user):
    owner = make_user()
    b = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "bob", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "bob", b.id, "read",
        operator_id=owner.id, bypass=False, db=db,
    )
    scope = {"user": "alice", "app": "my-repo"}
    # B has no permissions on user/alice or app/my-repo -> READ fails
    with pytest.raises(HTTPException) as exc:
        ep.check_memory_scope_permission(scope, b.id, "read", db)
    assert exc.value.status_code == 403


def test_scope_empty_admin_only(db, make_user):
    user = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.check_memory_scope_permission({}, user.id, "read", db)
    assert exc.value.status_code == 403
    admin = make_user(role="admin")
    ep.check_memory_scope_permission({}, admin.id, "read", db, bypass=True)


# --------------------------------------------------------------------------- #
# Cross-user agent/run isolation (agent/run are unique per parent, not global)
# --------------------------------------------------------------------------- #
def test_inject_default_user_id_always_injects(make_user, monkeypatch):
    """user_id is injected whenever missing, even if other entity params exist."""
    user = make_user()
    # conftest stubs `auth` (incl. is_bootstrap_admin) with a MagicMock; pin the
    # real semantics so behavior is deterministic. bootstrap (admin_api_key) is
    # rejected at the endpoint before reaching here, so inject_default_user_id
    # always injects the operator's user_id when missing.
    monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
    # only agent_id -> user_id still injected (prevents cross-user leak)
    params = ep.inject_default_user_id({"agent_id": "riley"}, user)
    assert params["user_id"] == str(user.id)
    assert params["agent_id"] == "riley"
    # explicit user_id is preserved, not overwritten
    params = ep.inject_default_user_id({"user_id": "alice", "agent_id": "riley"}, user)
    assert params["user_id"] == "alice"
    # no params -> user_id injected
    params = ep.inject_default_user_id({}, user)
    assert params == {"user_id": str(user.id)}
    # Even a bootstrap-flagged operator gets injected now (the early-return was
    # removed); bootstrap never reaches here in production, but pin the invariant.
    monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: True)
    params = ep.inject_default_user_id({"agent_id": "riley"}, user)
    assert params == {"agent_id": "riley", "user_id": str(user.id)}


def test_inject_prevents_cross_user_agent_read(db, make_user):
    """A write carrying only agent_id is scoped to the writer via inject, so a
    different user who owns a same-named agent cannot read it."""
    alice = make_user()
    bob = make_user()
    ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
    ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
    ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
    ep.ensure_entity_owner("agent", "riley", bob.id, db, parent_entity_id=str(bob.id))
    db.commit()
    # Stored memory carries alice's user_id (injected) + agent_id.
    scope = {"user": str(alice.id), "agent": "riley"}
    ep.check_memory_scope_permission(scope, alice.id, "read", db)  # alice OK
    with pytest.raises(HTTPException) as exc:
        ep.check_memory_scope_permission(scope, bob.id, "read", db)
    assert exc.value.status_code == 403


def test_get_entity_or_none_scopes_agent_by_parent(db, make_user):
    """agent/run lookups scoped by parent_entity_id return only that parent's entity."""
    alice = make_user()
    bob = make_user()
    ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
    ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
    ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
    ep.ensure_entity_owner("agent", "riley", bob.id, db, parent_entity_id=str(bob.id))
    db.commit()

    a_agent = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(alice.id))
    assert a_agent is not None and a_agent.owner_id == alice.id
    b_agent = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(bob.id))
    assert b_agent is not None and b_agent.owner_id == bob.id
    assert a_agent.pk != b_agent.pk
    # user/app are globally unique -> parent_entity_id ignored
    assert ep.get_entity_or_none("user", str(alice.id), db, parent_entity_id=str(bob.id)) is not None


# --------------------------------------------------------------------------- #
# grant / revoke (user/app only, owner vs admin boundaries)
# --------------------------------------------------------------------------- #
def test_grant_revoke_user(db, make_user):
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "alice", other.id, "write",
        operator_id=owner.id, bypass=False, db=db,
    )
    perms = ep.list_entity_permissions(
        "user", "alice", operator_id=owner.id, bypass=False, db=db,
    )
    assert len(perms) == 1 and perms[0].permission == "write"
    # cannot revoke the owner
    with pytest.raises(HTTPException) as exc:
        ep.revoke_entity_permission(
            "user", "alice", owner.id, operator_id=owner.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 400
    ep.revoke_entity_permission(
        "user", "alice", other.id, operator_id=owner.id, bypass=False, db=db,
    )
    perms = ep.list_entity_permissions(
        "user", "alice", operator_id=owner.id, bypass=False, db=db,
    )
    assert len(perms) == 0


def test_revoke_nonexistent_permission_raises_404(db, make_user):
    """Revoking a permission that does not exist must raise 404, not silently
    return success."""
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.revoke_entity_permission(
            "user", "alice", other.id, operator_id=owner.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 404


def test_grant_admin_only_owner_can(db, make_user):
    """Only owner (not admin grantee) can grant admin."""
    owner = make_user()
    admin_grantee = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    # Owner grants admin to admin_grantee
    ep.grant_entity_permission(
        "user", "alice", admin_grantee.id, "admin",
        operator_id=owner.id, bypass=False, db=db,
    )
    # admin_grantee tries to grant admin to other -> fails
    with pytest.raises(HTTPException) as exc:
        ep.grant_entity_permission(
            "user", "alice", other.id, "admin",
            operator_id=admin_grantee.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 403
    # admin_grantee can grant write
    ep.grant_entity_permission(
        "user", "alice", other.id, "write",
        operator_id=admin_grantee.id, bypass=False, db=db,
    )


def test_grant_denied_for_agent_run(db, make_user):
    owner = make_user()
    other = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id="alice")
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.grant_entity_permission(
            "agent", "riley", other.id, "read",
            operator_id=owner.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------- #
# transfer (user/app, top-level only for user, cascade)
# --------------------------------------------------------------------------- #
def test_transfer_user_cascade(db, make_user):
    owner = make_user()
    new_owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id="alice:laptop")
    db.commit()
    ep.transfer_entity_owner(
        "user", "alice", new_owner.id, operator_id=owner.id, bypass=False, db=db,
    )
    # All entities should now be owned by new_owner
    assert ep.check_entity_permission("user", "alice", new_owner.id, "admin", db)
    assert ep.check_entity_permission("user", "alice:laptop", new_owner.id, "admin", db)
    assert ep.check_entity_permission("agent", "riley", new_owner.id, "admin", db, parent_entity_id="alice:laptop")
    # Old owner keeps admin grant
    assert ep.check_entity_permission("user", "alice", owner.id, "admin", db)


def test_transfer_user_sub_rejected(db, make_user):
    """Cannot transfer a sub-namespace; only top-level."""
    owner = make_user()
    new_owner = make_user()
    ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.transfer_entity_owner(
            "user", "alice:laptop", new_owner.id, operator_id=owner.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 400


def test_transfer_quota(db, make_user, monkeypatch):
    monkeypatch.setattr(ep, "MAX_OWNED_ENTITIES_PER_USER", 1)
    owner = make_user()
    target = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "bob", target.id, db)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        ep.transfer_entity_owner(
            "user", "alice", target.id, operator_id=owner.id, bypass=False, db=db,
        )
    assert exc.value.status_code == 403


def test_transfer_noop_to_current_owner_skips_quota(db, make_user, monkeypatch):
    """No-op transfer (target == current owner) must not trip the quota check
    even when the owner is already at their limit."""
    monkeypatch.setattr(ep, "MAX_OWNED_ENTITIES_PER_USER", 1)
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    # Owner is at quota (owns 1 top-level entity). Transferring to self must not 403.
    ep.transfer_entity_owner(
        "user", "alice", owner.id, operator_id=owner.id, bypass=False, db=db,
    )


# --------------------------------------------------------------------------- #
# owner-only deletion
# --------------------------------------------------------------------------- #
def test_owner_can_delete_admin_cannot(db, make_user):
    owner = make_user()
    admin_grantee = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "alice", admin_grantee.id, "admin",
        operator_id=owner.id, bypass=False, db=db,
    )
    # admin_grantee cannot delete
    with pytest.raises(HTTPException) as exc:
        ep.check_entity_delete_permission("user", "alice", admin_grantee.id, False, db)
    assert exc.value.status_code == 403
    # owner can delete
    entity = ep.check_entity_delete_permission("user", "alice", owner.id, False, db)
    assert entity is not None


def test_global_admin_can_delete(db, admin_user, make_user):
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    entity = ep.check_entity_delete_permission("user", "alice", admin_user.id, True, db)
    assert entity is not None


def test_delete_permission_unclaimed_namespace(db, admin_user, make_user):
    """A missing entity always raises 404, even for a global admin."""
    with pytest.raises(HTTPException) as exc:
        ep.check_entity_delete_permission("user", "ghost", admin_user.id, True, db)
    assert exc.value.status_code == 404
    other = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.check_entity_delete_permission("user", "ghost", other.id, False, db)
    assert exc.value.status_code == 404


def test_delete_permission_scopes_agent_by_parent(db, make_user):
    """agent/run delete permission must be scoped by parent_entity_id.

    MCP ``delete_entities`` previously omitted ``parent_entity_id``, so a non-admin
    deleting their own ``agent/riley`` could resolve another user's same-named
    row first (agent/run are unique per parent, not globally) and 403. REST and
    compat v2 pass ``parent_entity_id``; MCP now does too.
    """
    alice = make_user()
    bob = make_user()
    ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
    ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
    ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
    ep.ensure_entity_owner("agent", "riley", bob.id, db, parent_entity_id=str(bob.id))
    db.commit()

    # Each owner, scoped to their own parent_entity_id, resolves their own agent.
    a_entity = ep.check_entity_delete_permission(
        "agent", "riley", alice.id, False, db,
        parent_entity_id=str(alice.id),
    )
    b_entity = ep.check_entity_delete_permission(
        "agent", "riley", bob.id, False, db,
        parent_entity_id=str(bob.id),
    )
    assert a_entity is not None and a_entity.owner_id == alice.id
    assert b_entity is not None and b_entity.owner_id == bob.id
    # Same name, different owners -> distinct parent-scoped rows.
    assert a_entity.pk != b_entity.pk


def test_prescan_fails_closed_on_vector_error(monkeypatch, fake_memory):
    """A vector-store error during the bulk-delete prescan must surface (503), not
    silently return [] and let validate_bulk_admin_operation pass trivially."""

    def boom(*args, **kwargs):
        raise RuntimeError("vector store down")

    monkeypatch.setattr(fake_memory, "get_all", boom)
    with pytest.raises(HTTPException) as exc:
        ep.list_memory_ids_for_params({"user_id": "alice"})
    assert exc.value.status_code == 503


def test_count_memories_is_advisory_on_vector_error(monkeypatch, fake_memory):
    """count_memories_for_entity is advisory: a vector-store error logs and returns 0."""

    def boom(*args, **kwargs):
        raise RuntimeError("vector store down")

    monkeypatch.setattr(fake_memory, "get_all", boom)
    assert ep.count_memories_for_entity("user", "alice") == 0


def test_count_memories_propagates_programming_errors(monkeypatch, fake_memory):
    """Programming errors (NameError, AttributeError) must propagate, not be
    silently swallowed as 0 — otherwise a refactoring bug that introduces a
    typo in the function body would permanently return 0 counts."""

    def boom(*args, **kwargs):
        raise NameError("undefined variable in vector store call")

    monkeypatch.setattr(fake_memory, "get_all", boom)
    with pytest.raises(NameError):
        ep.count_memories_for_entity("user", "alice")


def test_collect_user_children_cascades_all_descendants(db, make_user):
    """collect_user_children returns sub-namespaces + agent/run descendants,
    in FK-safe deletion order (agent/run first)."""
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    ep.ensure_entity_owner("user", "alice:phone", owner.id, db)
    ep.ensure_entity_owner("agent", "bot", owner.id, db, parent_entity_id="alice")
    ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id="alice:laptop")
    db.commit()

    alice = ep.get_entity_or_none("user", "alice", db)
    children = ep.collect_user_children(alice, db)
    types = [c.type for c in children]
    ids = {c.id for c in children}

    assert ids == {"alice:laptop", "alice:phone", "bot", "riley"}
    # agent/run descendants precede sub-namespaces (FK-safe deletion order)
    assert set(types[:2]) == {"agent"}
    assert set(types[2:]) == {"user"}


def test_collect_user_children_empty_for_leaf(db, make_user):
    """A leaf user entity (no descendants) returns []; a non-user entity returns []."""
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "alice:laptop", owner.id, db)
    db.commit()

    # alice:laptop has no descendants of its own
    laptop = ep.get_entity_or_none("user", "alice:laptop", db)
    assert ep.collect_user_children(laptop, db) == []

    # non-user entity -> [] (agent/run are never parents in the hierarchy)
    ep.ensure_entity_owner("agent", "bot", owner.id, db, parent_entity_id="alice")
    db.commit()
    bot = ep.get_entity_or_none("agent", "bot", db)
    assert ep.collect_user_children(bot, db) == []


def test_subnamespace_match_escapes_like_wildcards(db, make_user):
    """Sub-namespace prefix matching must escape SQL LIKE wildcards so an entity
    name containing '_' / '%' does not match other users' sub-namespaces.

    With unescaped LIKE: 'a_b:%' matches 'aXb:foo' (_ -> X), and 'al%:%' matches
    'alice:foo' (% -> 'ice'). Both would cause cross-entity cascade transfer/delete.
    """
    owner1 = make_user()
    owner2 = make_user()
    ep.ensure_entity_owner("user", "a_b", owner1.id, db)
    ep.ensure_entity_owner("user", "aXb:foo", owner2.id, db)  # aXb top-level + aXb:foo
    ep.ensure_entity_owner("user", "al%", owner1.id, db)
    ep.ensure_entity_owner("user", "alice:foo", owner2.id, db)
    db.commit()

    # Neither a_b nor al% have children; the LIKE wildcards in their names must
    # not match other users' sub-namespaces.
    assert ep.collect_user_children(ep.get_entity_or_none("user", "a_b", db), db) == []
    assert ep.collect_user_children(ep.get_entity_or_none("user", "al%", db), db) == []
    # Sanity: the real parent prefixes still resolve their children.
    assert len(ep.collect_user_children(ep.get_entity_or_none("user", "aXb", db), db)) == 1
    assert len(ep.collect_user_children(ep.get_entity_or_none("user", "alice", db), db)) == 1


# --------------------------------------------------------------------------- #
# quota (top-level user entities only)
# --------------------------------------------------------------------------- #
def test_quota_only_counts_top_level_user(db, make_user, monkeypatch):
    monkeypatch.setattr(ep, "MAX_OWNED_ENTITIES_PER_USER", 2)
    user = make_user()
    ep.ensure_entity_owner("user", "alice", user.id, db)
    ep.ensure_entity_owner("user", "bob", user.id, db)
    db.commit()
    # Sub-namespace should not count toward quota
    ep.ensure_entity_owner("user", "alice:laptop", user.id, db)
    db.commit()
    # Agent should not count toward quota
    ep.ensure_entity_owner("agent", "riley", user.id, db, parent_entity_id="alice:laptop")
    db.commit()
    # Third top-level should fail
    with pytest.raises(HTTPException) as exc:
        ep.ensure_entity_owner("user", "charlie", user.id, db)
    assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# validate_bulk_admin_operation
# --------------------------------------------------------------------------- #
def test_validate_bulk_admin_unauthorized_memory(db, make_user, fake_memory):
    owner = make_user()
    b = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    ep.ensure_entity_owner("user", "bob", owner.id, db)
    db.commit()
    fake_memory.add("m1", {"user_id": "alice", "app_id": "my-repo", "data": "x"})
    ep._create_entity_row("app", "my-repo", None, owner.id, None, db)
    db.commit()
    ep.grant_entity_permission(
        "user", "bob", b.id, "admin", operator_id=owner.id, bypass=False, db=db,
    )
    # B has admin on user/bob but NOT user/alice or app/my-repo -> bulk admin on m1 fails
    with pytest.raises(HTTPException) as exc:
        ep.validate_bulk_admin_operation(["m1"], b.id, db)
    assert exc.value.status_code == 403
    # owner has admin on the full scope -> passes
    ep.validate_bulk_admin_operation(["m1"], owner.id, db)


def test_validate_bulk_admin_admin_short_circuits(db, admin_user, make_user, fake_memory):
    """Bulk validation short-circuits (skips resolve_memory_entities, so a non-existent
    id does not 404) ONLY via the explicit bypass flag — no DB role auto-detection.
    bypass=False resolves + 404s even for role=admin."""
    # bypass flag -> no resolution, no 404 for a bogus id.
    ep.validate_bulk_admin_operation(["does-not-exist"], admin_user.id, db, bypass=True)
    # bypass=False (even role=admin) -> no short-circuit -> resolves -> 404.
    with pytest.raises(HTTPException) as exc:
        ep.validate_bulk_admin_operation(["does-not-exist"], admin_user.id, db, bypass=False)
    assert exc.value.status_code == 404
    # Non-admin (member) -> same 404.
    user = make_user()
    with pytest.raises(HTTPException) as exc:
        ep.validate_bulk_admin_operation(["does-not-exist"], user.id, db, bypass=False)
    assert exc.value.status_code == 404


def test_validate_bulk_admin_scope_hint_app_short_circuits(db, make_user, monkeypatch):
    """App-scoped scope_hint must check the app entity once and skip per-memory
    vector-store lookups (no resolve_memory_entities calls)."""
    owner = make_user()
    ep._create_entity_row("app", "my-repo", None, owner.id, None, db)
    db.commit()
    # If per-memory resolution ran, resolve_memory_entities would raise (no such id);
    # the fast path must never touch it.
    calls = {"n": 0}
    orig = ep.resolve_memory_entities

    def _spy(mid):
        calls["n"] += 1
        return orig(mid)

    monkeypatch.setattr(ep, "resolve_memory_entities", _spy)
    ep.validate_bulk_admin_operation(
        ["nonexistent-1", "nonexistent-2"], owner.id, db,
        scope_hint={"app": "my-repo"},
    )
    assert calls["n"] == 0  # fast path: no per-memory lookups


def test_validate_bulk_admin_scope_hint_non_app_falls_back(db, make_user, fake_memory):
    """scope_hint without app must fall back to per-memory resolution."""
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    fake_memory.add("m1", {"user_id": "alice", "data": "x"})
    # user-scoped hint -> per-memory check (resolves m1's scope = user/alice)
    ep.validate_bulk_admin_operation(
        ["m1"], owner.id, db, scope_hint={"user": "alice"},
    )


def test_bulk_delete_memories_prescans_and_deletes(db, make_user, fake_memory):
    """bulk_delete_memories must prescan, admin-validate, and delete — all
    inside the callable passed to run_memory_write."""
    owner = make_user()
    ep.ensure_entity_owner("user", "alice", owner.id, db)
    db.commit()
    fake_memory.add("m1", {"user_id": "alice", "data": "x"})
    fake_memory.add("m2", {"user_id": "alice", "data": "y"})

    # Verify memories exist
    assert len(fake_memory.get_all({"user_id": "alice"})) == 2

    ep.bulk_delete_memories(
        fake_memory, {"user_id": "alice"}, owner.id, db, bypass=True,
    )

    # Verify memories deleted
    assert len(fake_memory.get_all({"user_id": "alice"})) == 0


# --------------------------------------------------------------------------- #
# resolve_memory_entities
# --------------------------------------------------------------------------- #
def test_resolve_memory_entities_drops_unmapped_scope_fields(fake_memory, monkeypatch):
    import memory_lock as mlock  # noqa: E402

    fake_memory.add("m1", {"user_id": "alice", "data": "x"})
    monkeypatch.setattr(
        mlock, "entity_scope_from_record", lambda record: {"user_id": "alice", "bogus_scope": "z"}
    )
    assert ep.resolve_memory_entities("m1") == {"user": "alice"}


def test_resolve_memory_entities_propagates_non_data_error(fake_memory, monkeypatch):
    def _boom(_mid):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(fake_memory, "get", _boom)
    with pytest.raises(RuntimeError):
        ep.resolve_memory_entities("m1")


def test_resolve_memory_entities_treats_data_error_as_not_found(fake_memory, monkeypatch):
    def _bad(_mid):
        raise KeyError("missing")

    monkeypatch.setattr(fake_memory, "get", _bad)
    with pytest.raises(HTTPException) as exc:
        ep.resolve_memory_entities("m1")
    assert exc.value.status_code == 404


def test_resolve_memory_entities_missing_is_404(fake_memory):
    fake_memory.add("m1", {"user_id": "alice"})
    with pytest.raises(HTTPException) as exc:
        ep.resolve_memory_entities("does-not-exist")
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------- #
# ORM constraints mirror the Alembic migration (007)
# --------------------------------------------------------------------------- #
def test_entity_models_declare_orm_indexes():
    """ORM __table_args__ have the correct indexes."""
    from server.models import Entity, EntityPermission

    ent_indexes = {i.name for i in Entity.__table__.indexes if i.name}
    assert "ix_entities_owner_id" in ent_indexes
    assert "ix_entities_parent_pk" in ent_indexes
    # Partial unique indexes (mirror migration 007)
    assert "uq_entities_type_id_global" in ent_indexes
    assert "uq_entities_type_parent_id" in ent_indexes

    perm_constraints = {c.name for c in EntityPermission.__table__.constraints if c.name}
    perm_indexes = {i.name for i in EntityPermission.__table__.indexes if i.name}
    assert "uq_entity_permissions_entity_grantee" in perm_constraints
    assert "ix_entity_permissions_grantee_id" in perm_indexes


# --------------------------------------------------------------------------- #
# _rewrite_query_filter — app-as-primary-gate filter rewriting
# --------------------------------------------------------------------------- #
class TestRewriteQueryFilter:
    def test_keeps_user_id_when_app_id_present(self, db, make_user):
        """user_id is no longer dropped when app_id is present."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"user_id": "A", "app_id": "x"}, user.id, db
        )
        assert result == {"user_id": "A", "app_id": "x"}

    def test_preserves_user_id_when_no_app_id(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter(
            {"user_id": "A"}, user.id, db
        )
        assert result == {"user_id": "A"}

    def test_keeps_user_id_in_and_branch_with_app_id(self, db, make_user):
        """both user_id and app_id are preserved in AND branches."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "A"}, {"app_id": "x"}]}, user.id, db
        )
        assert "AND" in result
        children = result["AND"]
        has_user = any("user_id" in c and c.get("user_id") == "A" for c in children)
        has_app = any("app_id" in c for c in children)
        assert has_user
        assert has_app

    def test_preserves_user_id_branch_in_or_alongside_app(self, db, make_user):
        """user_id is preserved even in OR branches alongside app."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"OR": [{"user_id": "A", "app_id": "x"}, {"user_id": "B"}]},
            user.id, db,
        )
        assert result == {"OR": [{"user_id": "A", "app_id": "x"}, {"user_id": "B"}]}

    def test_preserves_non_entity_filters(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter(
            {"app_id": "x", "created_at": {"gte": "2024"}}, user.id, db
        )
        assert result == {"app_id": "x", "created_at": {"gte": "2024"}}
        assert "user_id" not in result

    def test_preserves_user_id_inside_metadata(self, db, make_user):
        """user_id inside a metadata block must NOT be stripped (it's a
        user-defined field, not the entity-scope user_id)."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"app_id": "x", "metadata": {"user_id": "custom-tag"}}, user.id, db
        )
        assert result == {
            "app_id": "x",
            "metadata": {"user_id": "custom-tag"},
        }

    def test_wildcard_flat_expands_to_or(self, db, make_user):
        """app_id:* should expand to OR of accessible apps."""
        user = make_user()
        # Create an app entity owned by the user
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        result = ep._rewrite_query_filter({"app_id": "*"}, user.id, db)
        assert result == {"app_id": "my-app"}

    def test_wildcard_zero_apps_raises_403(self, db, make_user):
        user = make_user()
        with pytest.raises(HTTPException) as exc:
            ep._rewrite_query_filter({"app_id": "*"}, user.id, db)
        assert exc.value.status_code == 403

    def test_wildcard_admin_bypass_sees_all_apps(self, db, make_user):
        """A global admin querying app_id:"*" must see ALL apps (admin bypass),
        not just owned/granted — otherwise an admin who owns no apps gets 403."""
        admin = make_user(role="admin")
        other = make_user()
        # Apps owned by another user; admin owns/grants none.
        db.add(Entity(type="app", id="app-a", owner_id=other.id))
        db.add(Entity(type="app", id="app-b", owner_id=other.id))
        db.flush()
        # Direct rewrite with bypass -> expands to all apps (no 403).
        result = ep._rewrite_query_filter({"app_id": "*"}, admin.id, db, bypass=True)
        assert result == {"OR": [{"app_id": "app-a"}, {"app_id": "app-b"}]}
        # Full plumbing: check_query_permission threads bypass through.
        rewritten = ep.check_query_permission({"app_id": "*"}, admin.id, db, bypass=True)
        assert rewritten == {"OR": [{"app_id": "app-a"}, {"app_id": "app-b"}]}

    def test_wildcard_nested_and(self, db, make_user):
        user = make_user()
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        result = ep._rewrite_query_filter(
            {"AND": [{"app_id": "*"}, {"agent_id": "riley"}]}, user.id, db
        )
        assert "AND" in result
        and_children = result["AND"]
        # Should be: AND(app_id=my-app, agent_id=riley)
        # (single-child OR from wildcard expansion was unwrapped by cleanup)
        assert any(isinstance(c, dict) and "app_id" in c and c.get("app_id") == "my-app"
                   for c in and_children)

    def test_wildcard_mixed_dict_with_logical_operators(self, db, make_user):
        """MCP path: mixed dict with both AND and top-level app_id:*."""
        user = make_user()
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        result = ep._rewrite_query_filter(
            {"AND": [{"agent_id": "x"}], "app_id": "*"}, user.id, db
        )
        # Should NOT contain residual "app_id": "*"
        assert "app_id" not in result or result.get("app_id") != "*"
        # Should expand to accessible app
        assert "my-app" in str(result)

    def test_not_subtree_expands_wildcard(self, db, make_user):
        """NOT subtree should also expand app_id:*."""
        user = make_user()
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        result = ep._rewrite_query_filter(
            {"NOT": {"app_id": "*"}}, user.id, db
        )
        assert "NOT" in result
        # The NOT subtree should be expanded (no residual "*")
        assert "*" not in str(result["NOT"])

    def test_user_id_preserved_in_and_with_app_id(self, db, make_user):
        """user_id is preserved alongside app_id in AND branches."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "A"}, {"app_id": "x"}]}, user.id, db
        )
        # Both user_id and app_id should be present
        assert "user_id" in str(result)

    def test_deeply_nested_filter_raises_400(self, db, make_user):
        user = make_user()
        # Build a filter deeper than MAX_DEPTH (10)
        deep = {"app_id": "x"}
        for _ in range(15):
            deep = {"AND": [deep]}
        with pytest.raises(HTTPException) as exc:
            ep._rewrite_query_filter(deep, user.id, db)
        assert exc.value.status_code == 400

    def test_agent_id_leaf_injects_caller_user_id(self, db, make_user):
        """A bare agent_id leaf is scoped to the caller's namespace — agent_id is
        per-parent-user, not global, so without an injected user_id the query
        would leak every user's same-named agent. The injection is per-leaf and
        independent of any sibling user_id in the same AND."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": str(user.id)}, {"agent_id": "riley"}]}, user.id, db
        )
        assert result == {
            "AND": [
                {"user_id": str(user.id)},
                {"agent_id": "riley", "user_id": str(user.id)},
            ]
        }

    def test_agent_id_leaf_with_foreign_user_id_sibling_is_contradictory(self, db, make_user):
        """When a sibling user_id names a *different* user than the caller, the
        per-leaf injection yields user_id=<foreign> AND user_id=<caller> — a
        contradictory filter that matches nothing. This is the security model
        refusing a cross-user agent query, not a bug: agent_id is pinned to the
        caller's namespace regardless of what the caller claims via user_id."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "someone-else"}, {"agent_id": "riley"}]}, user.id, db
        )
        assert result == {
            "AND": [
                {"user_id": "someone-else"},
                {"agent_id": "riley", "user_id": str(user.id)},
            ]
        }

    def test_cleanup_empty_operator_keeps_sibling_keys(self, db, make_user):
        """An empty AND/OR after cleanup must not discard sibling keys
        (created_at, metadata)."""
        user = make_user()
        app = Entity(type="app", id="x", owner_id=user.id)
        db.add(app)
        db.flush()
        # AND with app_id:"*" (expands) + a sibling created_at filter.
        # After rewrite, user_id dropped everywhere; the AND still carries
        # created_at and app_id.
        result = ep._rewrite_query_filter(
            {"created_at": {"gte": "2024"}, "app_id": "x"}, user.id, db
        )
        assert result.get("created_at") == {"gte": "2024"}
        assert result.get("app_id") == "x"

    def test_cleanup_empty_and_with_sibling_preserves_sibling(self, db, make_user):
        """When an AND becomes empty (all children dropped), sibling keys on the
        same node must survive — not return {}."""
        user = make_user()
        # {"AND": [{"user_id": "A"}, {"user_id": "B"}], "app_id": "x"}
        # Post-pass drops both user_id -> AND becomes empty -> must be removed,
        # but app_id (sibling) must survive.
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "A"}, {"user_id": "B"}], "app_id": "x"},
            user.id, db,
        )
        assert result.get("app_id") == "x"

    def test_app_direct_key_preserves_user_id(self, db, make_user):
        """user_id is preserved alongside app_id in all contexts."""
        user = make_user()
        # Single-child AND: user_id should be preserved
        result = ep._rewrite_query_filter(
            {"app_id": "x", "AND": [{"user_id": "alice"}]}, user.id, db,
        )
        assert result.get("app_id") == "x"
        assert "user_id" in str(result)

        # Multi-child AND under app_id: user_id preserved
        result = ep._rewrite_query_filter(
            {"app_id": "x", "AND": [{"user_id": "alice"}, {"user_id": "bob"}]}, user.id, db,
        )
        assert result.get("app_id") == "x"
        assert "user_id" in str(result)

        # NOT sibling of a direct app_id: user_id preserved
        result = ep._rewrite_query_filter(
            {"app_id": "x", "NOT": {"user_id": "alice"}}, user.id, db,
        )
        assert result.get("app_id") == "x"
        assert "NOT" in result

    def test_cleanup_unwrap_preserves_sibling_operators(self, db, make_user):
        """cleanup unwrap preserves AND/OR/NOT; user_id is not stripped."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"app_id": "x", "AND": [{"foo": "bar"}], "NOT": {"user_id": str(user.id)}},
            user.id, db,
        )
        assert result.get("app_id") == "x"
        # user_id from NOT is now preserved (not dropped by app gate)
        assert "user_id" in str(result)
        assert "foo" in str(result)

        # OR sibling: user_id preserved (no longer stripped by app gate).
        result = ep._rewrite_query_filter(
            {"app_id": "x", "AND": [{"foo": "bar"}], "OR": [{"user_id": str(user.id)}]},
            user.id, db,
        )
        assert result.get("app_id") == "x"
        assert "user_id" in str(result)
        assert "foo" in str(result)

    def test_not_subtree_app_id_does_not_trigger_user_id_stripping(self, db, make_user):
        """_filter_has_app_id must NOT recurse into NOT subtrees — app_id inside
        NOT is a negated data filter, not a primary permission gate. An AND
        conjunction whose only app_id is inside a NOT subtree must preserve
        user_id for admin queries (non-admin NOT queries are rejected upstream)."""
        user = make_user()
        # AND with user_id + NOT app_id: the NOT's app_id is negated, so user_id
        # must survive the rewrite (app is NOT the primary gate here).
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "A"}, {"NOT": {"app_id": "x"}}]},
            user.id, db, bypass=True,
        )
        assert "user_id" in str(result)
        assert "A" in str(result)

    def test_not_subtree_app_id_does_not_block_direct_app_id_gate(self, db, make_user):
        """When an AND has both a direct app_id AND a NOT with app_id, the direct
        app_id still serves as the primary gate and user_id is dropped."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"app_id": "x"}, {"NOT": {"app_id": "y"}}]},
            user.id, db, bypass=True,
        )
        # app_id "x" is a direct gate → user_id should be dropped from siblings
        assert "app_id" in str(result)
        assert "x" in str(result)
        # NOT subtree should survive (it's a data filter, not stripped)
        # but user_id should be gone from the node
        assert "user_id" not in str(result)

    def test_wildcard_inside_not_multi_app_raises_400(self, db, make_user):
        """app_id: * inside a NOT subtree with multiple accessible apps must be
        rejected — the expansion would produce NOT OR which most vector stores
        do not support."""
        user = make_user()
        db.add(Entity(type="app", id="app-a", owner_id=user.id))
        db.add(Entity(type="app", id="app-b", owner_id=user.id))
        db.flush()
        with pytest.raises(HTTPException) as exc:
            ep._rewrite_query_filter(
                {"NOT": {"app_id": "*"}}, user.id, db, bypass=True,
            )
        assert exc.value.status_code == 400
        assert "NOT" in exc.value.detail or "multiple" in exc.value.detail.lower()

    def test_none_logical_operator_is_treated_as_empty(self, db, make_user):
        """A logical operator with value None (e.g., {"AND": None}) must be
        treated as if the operator were absent, not produce a malformed filter
        with "AND": None."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": None, "user_id": "A"}, user.id, db
        )
        # Should produce a clean filter with user_id, not retain "AND": None
        assert result == {"user_id": "A"}


def test_get_accessible_apps_returns_owned_and_granted(db, make_user):
    """_get_accessible_apps must return both owned and explicitly granted apps
    in a single query."""
    owner = make_user()
    grantee = make_user()
    db.add(Entity(type="app", id="owned-app", owner_id=owner.id))
    db.add(Entity(type="app", id="shared-app", owner_id=owner.id))
    db.flush()
    shared = db.scalar(
        select(Entity).where(Entity.type == "app", Entity.id == "shared-app")
    )
    db.add(EntityPermission(
        entity_pk=shared.pk, grantee_id=grantee.id, permission="read",
    ))
    db.flush()

    apps = ep._get_accessible_apps(owner.id, db)
    assert "owned-app" in apps
    assert "shared-app" in apps
    assert len(apps) == 2

    apps = ep._get_accessible_apps(grantee.id, db)
    assert "shared-app" in apps
    assert "owned-app" not in apps
    assert len(apps) == 1


# --------------------------------------------------------------------------- #
# check_memory_scope_permission — app as primary gate
# --------------------------------------------------------------------------- #
class TestAppPrimaryGate:
    def test_read_with_app_and_user_permission_required(self, db, make_user):
        """ both app read AND user read are required when both are in scope."""
        owner = make_user()
        reader = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=reader.id, permission="read"))
        db.flush()
        # reader has app read but NOT user read for "some-other-user" -> 403
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"user": "some-other-user", "app": "project-x"},
                reader.id, "read", db,
            )
        assert exc.value.status_code == 403

    def test_write_with_app_and_user_permission_required(self, db, make_user):
        """write also requires both app and user permission."""
        owner = make_user()
        writer = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=writer.id, permission="write"))
        db.flush()
        # writer has app write but NOT user write -> 403
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"user": "some-other-user", "app": "project-x"},
                writer.id, "write", db,
            )
        assert exc.value.status_code == 403

    def test_write_without_app_permission_raises_403(self, db, make_user):
        owner = make_user()
        writer = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        # No grant for writer
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"user": "some-other-user", "app": "project-x"},
                writer.id, "write", db,
            )
        assert exc.value.status_code == 403

    def test_no_app_in_scope_falls_back_to_original_logic(self, db, make_user):
        """Scope without app should use original AND/OR logic."""
        owner = make_user()
        other = make_user()
        # other does NOT own user/A, so check should fail for write
        with pytest.raises(HTTPException):
            ep.check_memory_scope_permission(
                {"user": str(owner.id)}, other.id, "write", db,
            )

    def test_read_with_app_owner_but_not_user_owner_fails(self, db, make_user):
        """app owner also needs user read permission for the scoped user."""
        owner = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        # owner owns app but not the user entity "some-other-user" -> 403
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"user": "some-other-user", "app": "project-x"},
                owner.id, "read", db,
            )
        assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# check_query_permission — returns rewritten filter
# --------------------------------------------------------------------------- #
class TestCheckQueryPermission:
    def test_returns_rewritten_filter_with_user_id_preserved(self, db, make_user):
        """user_id is preserved in rewritten filter."""
        user = make_user()
        app = Entity(type="app", id="x", owner_id=user.id)
        db.add(app)
        db.flush()
        # Use user's own UUID so user read check passes
        result = ep.check_query_permission(
            {"user_id": str(user.id), "app_id": "x"}, user.id, db
        )
        assert isinstance(result, dict)
        assert "user_id" in result
        assert "app_id" in result

    def test_fails_with_app_read_but_no_user_read(self, db, make_user):
        """app read alone is insufficient; user read is also required."""
        owner = make_user()
        reader = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=reader.id, permission="read"))
        db.flush()
        # reader has app read but not user read for "A" -> 403
        with pytest.raises(HTTPException) as exc:
            ep.check_query_permission(
                {"user_id": "A", "app_id": "project-x"}, reader.id, db
            )
        assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# authorize_write — app scope skips user/agent/run in Pass 1
# --------------------------------------------------------------------------- #
class TestAuthorizeWriteApp:
    def test_pass1_dual_check_requires_user_write_too(self, db, make_user, monkeypatch):
        """authorize_write also requires user permission when user is in scope."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        owner = make_user()
        writer = make_user()
        # Create user entity owned by someone else
        user_entity = Entity(type="user", id="alice", owner_id=owner.id)
        db.add(user_entity)
        # Create app entity with writer having write permission
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=writer.id, permission="write"))
        db.flush()
        # writer has app write but NOT user write for "alice" -> 403
        with pytest.raises(HTTPException) as exc:
            ep.authorize_write(
                {"user_id": "alice", "app_id": "project-x"}, writer, db, bypass=False
            )
        assert exc.value.status_code == 403

    def test_pass1_checks_app_entity_when_app_present(self, db, make_user, monkeypatch):
        """When app is in scope, app entity must have write permission."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        owner = make_user()
        writer = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        # No write grant for writer
        with pytest.raises(HTTPException) as exc:
            ep.authorize_write(
                {"user_id": "alice", "app_id": "project-x"}, writer, db, bypass=False
            )
        assert exc.value.status_code == 403

    def test_pass1_passes_with_user_write_grant(self, db, make_user, monkeypatch):
        """authorize_write passes when caller has write grants on both app and user entities."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        owner = make_user()
        writer = make_user()
        # Create user entity with writer having write grant
        user_entity = Entity(type="user", id="alice", owner_id=owner.id)
        db.add(user_entity)
        # Create app entity with writer having write grant
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=user_entity.pk, grantee_id=writer.id, permission="write"))
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=writer.id, permission="write"))
        db.flush()
        # writer has both app write and user write grants -> should pass
        ep.authorize_write(
            {"user_id": "alice", "app_id": "project-x"}, writer, db, bypass=False
        )


# --------------------------------------------------------------------------- #
# inject_default_user_id — skip_if_has_app
# --------------------------------------------------------------------------- #
class TestInjectDefaultUserId:
    def test_skip_if_has_app_when_app_present(self, db, make_user, monkeypatch):
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        user = make_user()
        result = ep.inject_default_user_id(
            {"app_id": "project-x"}, user, skip_if_has_app=True
        )
        assert "user_id" not in result

    def test_skip_if_has_app_injects_when_no_app(self, db, make_user, monkeypatch):
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        user = make_user()
        result = ep.inject_default_user_id({"agent_id": "riley"}, user)
        assert result["user_id"] == str(user.id)

    def test_skip_if_has_app_false_always_injects(self, db, make_user, monkeypatch):
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        user = make_user()
        result = ep.inject_default_user_id(
            {"app_id": "project-x"}, user, skip_if_has_app=False
        )
        assert result["user_id"] == str(user.id)


# --------------------------------------------------------------------------- #
# Security regressions: cross-user leaks / revocation bypass
# --------------------------------------------------------------------------- #
class TestSecurityRegressions:
    # --- Fix 1: read-path user_id injection (agent_id/run_id-only leak) -----
    def test_read_query_injects_caller_user_id_for_agent_only(self, db, make_user, monkeypatch):
        """An agent_id/run_id-only read query must be scoped to the caller's
        user namespace so the vector store does not return other users'
        same-named agent/run memories. Mirrors inject_default_user_id.

        conftest stubs ``auth`` (incl. is_bootstrap_admin) with a MagicMock; pin
        the real semantics so the bootstrap guard is deterministic."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        alice = make_user()
        bob = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
        ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
        ep.ensure_entity_owner("agent", "riley", bob.id, db, parent_entity_id=str(bob.id))
        ep.ensure_entity_owner("run", "r1", bob.id, db, parent_entity_id=str(bob.id))
        db.commit()
        # bob's agent_id-only query is scoped to bob's user_id
        out = ep.check_query_permission({"agent_id": "riley"}, bob.id, db)
        assert out.get("user_id") == str(bob.id)
        assert out.get("agent_id") == "riley"
        # run_id-only is scoped too
        out = ep.check_query_permission({"run_id": "r1"}, bob.id, db)
        assert out.get("user_id") == str(bob.id)

    def test_read_query_no_inject_for_bootstrap(self, db, monkeypatch):
        """Bootstrap admin keeps the unscoped admin bypass (no user_id injected)."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: True)
        out = ep.check_query_permission(
            {"agent_id": "riley"}, uuid.UUID(int=0), db, bypass=True
        )
        assert "user_id" not in out

    def test_read_query_no_inject_when_app_present(self, db, make_user, monkeypatch):
        """app-scoped queries stay global (user_id dropped, not injected)."""
        monkeypatch.setattr(ep, "is_bootstrap_admin", lambda _uid: False)
        owner = make_user()
        reader = make_user()
        app = Entity(type="app", id="project-x", owner_id=owner.id)
        db.add(app)
        db.flush()
        from server.models import EntityPermission
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=reader.id, permission="read"))
        db.flush()
        out = ep.check_query_permission(
            {"app_id": "project-x", "agent_id": "riley"}, reader.id, db
        )
        assert "user_id" not in out

    # --- Fix 2: UUID shortcut respects transfer + revocation ----------------
    def test_uuid_namespace_denied_after_transfer_and_revoke(self, db, make_user):
        """After a UUID namespace is transferred and the previous owner's admin
        grant is revoked, the previous owner must lose access — the UUID
        shortcut must not bypass the new ownership state."""
        alice = make_user()
        bob = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        db.commit()
        assert ep.check_entity_permission("user", str(alice.id), alice.id, "admin", db)
        # transfer to bob; alice keeps an explicit (revocable) admin grant
        ep.transfer_entity_owner(
            "user", str(alice.id), bob.id,
            operator_id=alice.id, bypass=False, db=db,
        )
        assert ep.check_entity_permission("user", str(alice.id), alice.id, "read", db)
        # bob (new owner) revokes alice's admin grant
        ep.revoke_entity_permission(
            "user", str(alice.id), alice.id,
            operator_id=bob.id, bypass=False, db=db,
        )
        # alice must now be denied (was the bug: UUID shortcut kept granting)
        assert not ep.check_entity_permission("user", str(alice.id), alice.id, "read", db)
        assert not ep.check_entity_permission("user", str(alice.id), alice.id, "admin", db)
        # bob still owns the namespace
        assert ep.check_entity_permission("user", str(alice.id), bob.id, "admin", db)

    def test_uuid_namespace_self_owned_when_row_exists(self, db, make_user):
        """The UUID shortcut still grants self-ownership when the namespace row
        exists and is owned by the caller (regression guard for Fix 2)."""
        alice = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        db.commit()
        assert ep.check_entity_permission("user", str(alice.id), alice.id, "admin", db)
        assert ep.check_entity_permission("user", f"{alice.id}:laptop", alice.id, "write", db)

    # --- Fix 3b: prefix inheritance effective-owner (cross-user leak) -------
    def test_longer_prefix_owned_by_other_blocks_shorter_owned_prefix(self, db, make_user):
        """A longer prefix owned by another user must not be bypassed by
        ownership of a shorter prefix. Simulates the corrupted/orphaned
        namespace state that arises from a non-cascading parent delete."""
        alice = make_user()
        bob = make_user()
        # bob owns the top-level "alice"; alice owns the sub-namespace
        # "alice:laptop" (cannot arise via the normal API; set up directly).
        db.add(Entity(type="user", id="alice", owner_id=bob.id))
        db.add(Entity(type="user", id="alice:laptop", owner_id=alice.id))
        db.commit()
        # bob owns "alice" but must NOT read alice's private "alice:laptop"
        assert not ep.check_entity_permission("user", "alice:laptop", bob.id, "read", db)
        assert not ep.check_entity_permission("user", "alice:laptop", bob.id, "admin", db)
        # alice still owns her sub-namespace
        assert ep.check_entity_permission("user", "alice:laptop", alice.id, "read", db)
        # bob still reads his own top-level "alice"
        assert ep.check_entity_permission("user", "alice", bob.id, "read", db)

    def test_grant_on_mismatched_owner_prefix_not_applied(self, db, make_user):
        """A grant on a prefix owned by a different user must NOT apply to the
        longer prefix (cross-user leak via grant path)."""
        alice = make_user()
        bob = make_user()
        charlie = make_user()
        # bob owns "alice"; alice owns "alice:laptop" (corrupted state)
        db.add(Entity(type="user", id="alice", owner_id=bob.id))
        db.add(Entity(type="user", id="alice:laptop", owner_id=alice.id))
        db.commit()
        # bob grants charlie read on "alice"
        alice_entity = db.scalar(
            select(Entity).where(Entity.type == "user", Entity.id == "alice")
        )
        db.add(EntityPermission(
            entity_pk=alice_entity.pk, grantee_id=charlie.id, permission="read",
        ))
        db.commit()
        # charlie can read "alice" via the grant
        assert ep.check_entity_permission("user", "alice", charlie.id, "read", db)
        # charlie must NOT read "alice:laptop" — the grant is on bob's prefix,
        # but "alice:laptop" is owned by alice (different effective_owner)
        assert not ep.check_entity_permission("user", "alice:laptop", charlie.id, "read", db)


# --------------------------------------------------------------------------- #
# get_entity_or_none – agent/run scoping
# --------------------------------------------------------------------------- #
class TestGetEntityOrNone:
    """agent/run are unique per parent user, not globally. ``get_entity_or_none``
    accepts ``parent_entity_id`` (the parent user entity's string id) to scope
    the lookup to a specific parent namespace. user/app are globally unique and
    ignore the filter."""

    def test_agent_scoped_to_parent(self, db, make_user):
        """Two users each own agent/riley → scoped lookup returns the correct one."""
        alice = make_user()
        bob = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
        ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
        ep.ensure_entity_owner("agent", "riley", bob.id, db, parent_entity_id=str(bob.id))
        db.commit()

        a = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(alice.id))
        b = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(bob.id))
        assert a is not None and a.owner_id == alice.id
        assert b is not None and b.owner_id == bob.id
        assert a.pk != b.pk

    def test_agent_unscoped_returns_any(self, db, make_user):
        """Without parent_entity_id, agent lookup returns any matching row."""
        alice = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
        db.commit()
        entity = ep.get_entity_or_none("agent", "riley", db)
        assert entity is not None
        assert entity.type == "agent" and entity.id == "riley"

    def test_agent_wrong_parent_returns_none(self, db, make_user):
        """Scoped lookup for another user's parent namespace returns None."""
        alice = make_user()
        bob = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        ep.ensure_entity_owner("agent", "riley", alice.id, db, parent_entity_id=str(alice.id))
        db.commit()
        assert ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(bob.id)) is None

    def test_user_ignores_parent_filter(self, db, make_user):
        """user entities are globally unique; parent_entity_id filter is not applied."""
        owner = make_user()
        ep.ensure_entity_owner("user", "alice", owner.id, db)
        db.commit()
        other = make_user()
        entity = ep.get_entity_or_none("user", "alice", db, parent_entity_id=str(other.id))
        assert entity is not None
        assert entity.id == "alice"

    def test_run_scoped_to_parent(self, db, make_user):
        """run entities are also scoped (same as agent)."""
        alice = make_user()
        bob = make_user()
        ep.ensure_entity_owner("user", str(alice.id), alice.id, db)
        ep.ensure_entity_owner("run", "run-1", alice.id, db, parent_entity_id=str(alice.id))
        ep.ensure_entity_owner("user", str(bob.id), bob.id, db)
        ep.ensure_entity_owner("run", "run-1", bob.id, db, parent_entity_id=str(bob.id))
        db.commit()

        a = ep.get_entity_or_none("run", "run-1", db, parent_entity_id=str(alice.id))
        b = ep.get_entity_or_none("run", "run-1", db, parent_entity_id=str(bob.id))
        assert a is not None and a.owner_id == alice.id
        assert b is not None and b.owner_id == bob.id

    def test_parent_entity_id_resolves_correct_namespace(self, db, make_user):
        """One user owns two namespaces, each with agent/riley → parent_entity_id
        disambiguates which agent row is returned."""
        owner = make_user()
        ep.ensure_entity_owner("user", str(owner.id), owner.id, db)
        ep.ensure_entity_owner("user", f"{owner.id}:laptop", owner.id, db)
        db.commit()
        ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id=str(owner.id))
        ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id=f"{owner.id}:laptop")
        db.commit()

        top = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=str(owner.id))
        sub = ep.get_entity_or_none("agent", "riley", db, parent_entity_id=f"{owner.id}:laptop")
        assert top is not None and sub is not None
        assert top.pk != sub.pk
        from models import Entity
        top_parent = db.get(Entity, top.parent_pk)
        sub_parent = db.get(Entity, sub.parent_pk)
        assert top_parent.id == str(owner.id)
        assert sub_parent.id == f"{owner.id}:laptop"

    def test_parent_entity_id_nonexistent_returns_none(self, db, make_user):
        """parent_entity_id pointing to a non-existent user entity returns None."""
        owner = make_user()
        ep.ensure_entity_owner("user", str(owner.id), owner.id, db)
        ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id=str(owner.id))
        db.commit()
        assert ep.get_entity_or_none("agent", "riley", db, parent_entity_id="nonexistent") is None

    def test_parent_entity_id_ignored_for_user_type(self, db, make_user):
        """parent_entity_id is ignored for globally-unique user entities."""
        owner = make_user()
        ep.ensure_entity_owner("user", "alice", owner.id, db)
        db.commit()
        entity = ep.get_entity_or_none("user", "alice", db, parent_entity_id="some-parent")
        assert entity is not None
        assert entity.id == "alice"

    def test_check_entity_delete_permission_with_parent_entity_id(self, db, make_user):
        """check_entity_delete_permission passes parent_entity_id through to
        get_entity_or_none and resolves the correct scoped entity."""
        owner = make_user()
        ep.ensure_entity_owner("user", str(owner.id), owner.id, db)
        ep.ensure_entity_owner("user", f"{owner.id}:laptop", owner.id, db)
        db.commit()
        ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id=str(owner.id))
        ep.ensure_entity_owner("agent", "riley", owner.id, db, parent_entity_id=f"{owner.id}:laptop")
        db.commit()

        top = ep.check_entity_delete_permission(
            "agent", "riley", owner.id, False, db,
            parent_entity_id=str(owner.id),
        )
        sub = ep.check_entity_delete_permission(
            "agent", "riley", owner.id, False, db,
            parent_entity_id=f"{owner.id}:laptop",
        )
        assert top is not None and sub is not None
        assert top.pk != sub.pk


# --------------------------------------------------------------------------- #
# Client-zone no-admin-bypass (allow_role_admin_lookup=False)
# --------------------------------------------------------------------------- #
class TestClientZoneNoAdminBypass:
    """Permission functions honor ONLY the explicit bypass flag (no DB role
    auto-detection). compat/mcp (client zone) pass bypass=False -> role=admin
    is governed by ownership/grants (no bypass). main.py/entities.py (management
    zone) pass bypass=True -> role=admin bypasses. Bootstrap (nil UUID,
    bypass=True) bypasses via the flag.

    authorize_write's UUID-reservation/quota skip + the app_id="*" wildcard are
    governed by the bypass flag (client zone = bypass=False) — API-level.
    """

    def test_check_entity_permission_bypass_flag_only(self, db, make_user):
        admin = make_user(role="admin")
        other = make_user()
        db.add(Entity(type="user", id="alice", owner_id=other.id))
        db.flush()
        # bypass=True (management zone): role=admin bypasses.
        assert ep.check_entity_permission("user", "alice", admin.id, "read", db, bypass=True)
        # bypass=False (client zone): role=admin NOT bypassed -> False.
        assert not ep.check_entity_permission("user", "alice", admin.id, "read", db, bypass=False)
        # bootstrap (nil UUID, bypass=True) bypasses via the flag.
        assert ep.check_entity_permission("user", "alice", uuid.UUID(int=0), "read", db, bypass=True)

    def test_get_visible_entities_bypass_flag_only(self, db, make_user):
        admin = make_user(role="admin")
        other = make_user()
        db.add(Entity(type="user", id="alice", owner_id=other.id))
        db.add(Entity(type="user", id="bob", owner_id=admin.id))
        db.flush()
        # bypass=True: role=admin sees ALL entities.
        assert len(ep.get_visible_entities(admin.id, db, bypass=True)) == 2
        # bypass=False: role=admin sees only owned (bob).
        visible = ep.get_visible_entities(admin.id, db, bypass=False)
        assert [e.id for e in visible] == ["bob"]

    def test_validate_bulk_admin_bypass_flag_only(self, db, admin_user, fake_memory):
        # bypass=True: short-circuits -> bogus id does not 404.
        ep.validate_bulk_admin_operation(["does-not-exist"], admin_user.id, db, bypass=True)
        # bypass=False (even role=admin): no short-circuit -> resolves -> 404.
        with pytest.raises(HTTPException) as exc:
            ep.validate_bulk_admin_operation(["does-not-exist"], admin_user.id, db, bypass=False)
        assert exc.value.status_code == 404

    def test_check_query_permission_bypass_flag_only_not_query(self, db, make_user):
        admin = make_user(role="admin")
        flt = {"AND": [{"user_id": "alice"}, {"NOT": {"user_id": "bob"}}]}
        # bypass=True: NOT-query passes.
        ep.check_query_permission(flt, admin.id, db, bypass=True)
        # bypass=False: NOT-query -> 403.
        with pytest.raises(HTTPException) as exc:
            ep.check_query_permission(flt, admin.id, db, bypass=False)
        assert exc.value.status_code == 403
        assert "NOT operators" in exc.value.detail

    def test_check_memory_scope_permission_bypass_flag_only_empty_scope(self, db, admin_user):
        # bypass=True: empty scope -> pass.
        ep.check_memory_scope_permission({}, admin_user.id, "read", db, bypass=True)
        # bypass=False: empty scope -> 403.
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission({}, admin_user.id, "read", db, bypass=False)
        assert exc.value.status_code == 403
        assert "no entity scope" in exc.value.detail.lower()

    def test_wildcard_bypass_false_strips_role_admin(self, db, make_user):
        """role=admin with bypass=False: app_id='*' expands to owned/granted
        apps only (vs bypass=True which sees ALL apps)."""
        admin = make_user(role="admin")
        other = make_user()
        db.add(Entity(type="app", id="app-a", owner_id=other.id))
        db.add(Entity(type="app", id="app-b", owner_id=other.id))
        db.flush()
        # bypass=True: admin sees ALL apps.
        out = ep._rewrite_query_filter({"app_id": "*"}, admin.id, db, bypass=True)
        assert out == {"OR": [{"app_id": "app-a"}, {"app_id": "app-b"}]}
        # bypass=False: admin owns/granted none -> 403.
        with pytest.raises(HTTPException) as exc:
            ep._rewrite_query_filter({"app_id": "*"}, admin.id, db, bypass=False)
        assert exc.value.status_code == 403
        assert "No accessible app scope" in exc.value.detail
        # Grant admin read on app-a -> wildcard lists only app-a.
        db.add(EntityPermission(entity_pk=db.scalar(
            select(Entity.pk).where(Entity.type == "app", Entity.id == "app-a")
        ), grantee_id=admin.id, permission="read"))
        db.flush()
        out = ep._rewrite_query_filter({"app_id": "*"}, admin.id, db, bypass=False)
        assert out == {"app_id": "app-a"}

    def test_ensure_entity_owner_uuid_reservation_bypass_false(self, db, make_user):
        """role=admin with bypass=False cannot claim another user's UUID namespace
        (UUID reservation no longer skipped)."""
        admin = make_user(role="admin")
        other = make_user()
        # bypass=True: admin can claim user/<other's UUID> (UUID reservation skipped).
        ep.ensure_entity_owner("user", str(other.id), admin.id, db, bypass=True)
        # bypass=False: admin cannot claim another user's UUID namespace -> 403.
        other2 = make_user()
        with pytest.raises(HTTPException) as exc:
            ep.ensure_entity_owner("user", str(other2.id), admin.id, db, bypass=False)
        assert exc.value.status_code == 403
        assert "reserved for the user" in exc.value.detail


# --------------------------------------------------------------------------- #
# _resolve_user_owner — longest-prefix batch query
# --------------------------------------------------------------------------- #
class TestResolveUserOwner:
    def test_returns_owner_for_hierarchical_user_id(self, db, make_user):
        """alice:laptop finds alice's owner via prefix match."""
        owner = make_user()
        db.add(Entity(type="user", id="alice", owner_id=owner.id))
        db.flush()

        owner_id, entity = ep._resolve_user_owner("alice:laptop", db)
        assert owner_id == owner.id
        assert entity.id == "alice"

    def test_returns_owner_for_exact_match(self, db, make_user):
        """Exact user id finds its own owner."""
        owner = make_user()
        db.add(Entity(type="user", id="bob", owner_id=owner.id))
        db.flush()

        owner_id, entity = ep._resolve_user_owner("bob", db)
        assert owner_id == owner.id
        assert entity.id == "bob"

    def test_skips_orphaned_prefix(self, db, make_user):
        """Orphaned prefix (owner_id=None) is skipped; next shorter owned prefix wins."""
        owner = make_user()
        db.add(Entity(type="user", id="alice", owner_id=None))  # orphan
        db.add(Entity(type="user", id="alice:laptop", owner_id=owner.id))
        db.flush()

        owner_id, entity = ep._resolve_user_owner("alice:laptop:agent1", db)
        # alice:laptop is owned, so it wins (alice is orphaned and skipped)
        assert owner_id == owner.id
        assert entity.id == "alice:laptop"

    def test_returns_none_when_no_prefix_exists(self, db):
        """No matching entities at all."""
        owner_id, entity = ep._resolve_user_owner("nonexistent:child", db)
        assert owner_id is None
        assert entity is None

    def test_returns_none_when_all_prefixes_orphaned(self, db):
        """All matching prefixes are orphaned."""
        db.add(Entity(type="user", id="alice", owner_id=None))
        db.flush()

        owner_id, entity = ep._resolve_user_owner("alice:laptop", db)
        assert owner_id is None
        assert entity is None

    def test_batch_query_uses_in_clause(self, db, mocker):
        """Batch query fires a single IN clause, not one query per prefix."""
        spy = mocker.spy(ep, "select")
        db.add(Entity(type="user", id="alice", owner_id=uuid.uuid4()))
        db.flush()

        ep._resolve_user_owner("alice:laptop:agent1", db)
        # select() should be called exactly once for the batch IN query
        assert spy.call_count == 1


# --------------------------------------------------------------------------- #
# _assert_can_manage — is_owner short-circuit
# --------------------------------------------------------------------------- #
class TestAssertCanManage:
    def test_owner_short_circuits_skips_permission_check(self, db, make_user, mocker):
        """When operator is the owner, check_entity_permission is never called."""
        owner = make_user()
        entity = Entity(type="app", id="my-app", owner_id=owner.id)
        db.add(entity)
        db.flush()

        spy = mocker.spy(ep, "check_entity_permission")
        is_owner = ep._assert_can_manage(
            entity, "app", "my-app", owner.id, bypass=False, db=db,
        )
        assert is_owner is True
        spy.assert_not_called()

    def test_bypass_true_short_circuits(self, db, make_user, mocker):
        """bypass=True skips both owner check and permission check."""
        entity = Entity(type="app", id="shared-app", owner_id=None)
        db.add(entity)
        db.flush()

        spy = mocker.spy(ep, "check_entity_permission")
        is_owner = ep._assert_can_manage(
            entity, "app", "shared-app", None, bypass=True, db=db,
        )
        assert is_owner is False
        spy.assert_not_called()

    def test_non_owner_falls_through_to_permission_check(self, db, make_user):
        """Non-owner with admin grant passes the permission check."""
        owner = make_user()
        other = make_user()
        entity = Entity(type="app", id="app-x", owner_id=owner.id)
        db.add(entity)
        db.flush()  # populate entity.pk
        db.add(EntityPermission(
            entity_pk=entity.pk, grantee_id=other.id, permission="admin",
        ))
        db.flush()

        is_owner = ep._assert_can_manage(
            entity, "app", "app-x", other.id, bypass=False, db=db,
        )
        assert is_owner is False

    def test_non_owner_no_permission_raises_403(self, db, make_user):
        """Non-owner without admin grant gets 403."""
        owner = make_user()
        other = make_user()
        entity = Entity(type="app", id="app-y", owner_id=owner.id)
        db.add(entity)
        db.flush()

        with pytest.raises(HTTPException) as exc:
            ep._assert_can_manage(entity, "app", "app-y", other.id, bypass=False, db=db)
        assert exc.value.status_code == 403
        assert "Only entity owner/admin can manage permissions" in exc.value.detail


# --------------------------------------------------------------------------- #
# is_owner_or_global_admin
# --------------------------------------------------------------------------- #
class TestIsOwnerOrGlobalAdmin:
    def test_owner_returns_true(self):
        """Owner passes regardless of bypass."""
        owner_id = uuid.uuid4()
        entity = Entity(type="user", id="alice", owner_id=owner_id)
        assert ep.is_owner_or_global_admin(entity, owner_id, bypass=False) is True
        assert ep.is_owner_or_global_admin(entity, owner_id, bypass=True) is True

    def test_bypass_true_returns_true(self):
        """Global admin bypass passes even for non-owner."""
        entity = Entity(type="user", id="alice", owner_id=uuid.uuid4())
        assert ep.is_owner_or_global_admin(entity, uuid.uuid4(), bypass=True) is True

    def test_non_owner_no_bypass_returns_false(self):
        """Non-owner without bypass fails."""
        entity = Entity(type="user", id="alice", owner_id=uuid.uuid4())
        assert ep.is_owner_or_global_admin(entity, uuid.uuid4(), bypass=False) is False

    def test_orphan_entity_no_bypass_returns_false(self):
        """Orphan entity (owner_id=None) fails for non-bypass."""
        entity = Entity(type="user", id="alice", owner_id=None)
        assert ep.is_owner_or_global_admin(entity, uuid.uuid4(), bypass=False) is False

    def test_orphan_entity_bypass_true_returns_true(self):
        """Orphan entity passes with bypass."""
        entity = Entity(type="user", id="alice", owner_id=None)
        assert ep.is_owner_or_global_admin(entity, uuid.uuid4(), bypass=True) is True

    def test_granted_admin_returns_false(self):
        """Explicit admin grant does NOT pass — only owner or global admin."""
        owner_id = uuid.uuid4()
        grantee_id = uuid.uuid4()
        entity = Entity(type="app", id="app-x", owner_id=owner_id)
        # granted admin ≠ owner, bypass=False → False
        assert ep.is_owner_or_global_admin(entity, grantee_id, bypass=False) is False


# --------------------------------------------------------------------------- #
# get_visible_entities_paginated
# --------------------------------------------------------------------------- #
class TestGetVisibleEntitiesPaginated:
    """Pagination, owned-first ordering, type filter, and visibility scoping."""

    def _seed(self, db, owner):
        """A mix of owned + foreign user/agent entities for ordering checks."""
        owned_user = Entity(type="user", id="default", owner_id=owner.id)
        owned_sub = Entity(type="user", id="default:laptop", owner_id=owner.id)
        foreign_user = Entity(type="user", id="bob", owner_id=uuid.uuid4())
        owned_agent = Entity(type="agent", id="bot", owner_id=owner.id)
        db.add_all([owned_user, owned_sub, foreign_user, owned_agent])
        db.flush()
        return owned_user, owned_sub, foreign_user, owned_agent

    def test_owned_entities_sort_first(self, db, make_user):
        """Owned entities precede foreign ones regardless of id alphabet."""
        owner = make_user()
        self._seed(db, owner)
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, page=1, page_size=100
        )
        assert total == 4
        ids = [e.id for e in items]
        # The three owned entities (default, default:laptop, bot) come before bob.
        assert ids.index("default") < ids.index("bob")
        assert ids.index("default:laptop") < ids.index("bob")
        assert ids.index("bot") < ids.index("bob")

    def test_entity_type_filter_restricts_to_user(self, db, make_user):
        """entity_type='user' drops agent/run rows."""
        owner = make_user()
        self._seed(db, owner)
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, entity_type="user", page=1, page_size=100
        )
        assert total == 3  # default, default:laptop, bob (agent 'bot' excluded)
        assert {e.type for e in items} == {"user"}

    def test_pagination_slice_and_total(self, db, make_user):
        """page_size limits the slice while total reflects the full count."""
        owner = make_user()
        # 5 owned user entities
        for i in range(5):
            db.add(Entity(type="user", id=f"u{i}", owner_id=owner.id))
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, entity_type="user", page=1, page_size=2
        )
        assert total == 5
        assert len(items) == 2
        # Page 2 returns the next slice.
        items2, _ = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, entity_type="user", page=2, page_size=2
        )
        assert len(items2) == 2
        assert {e.id for e in items}.isdisjoint({e.id for e in items2})

    def test_non_admin_sees_only_owned_and_granted(self, db, make_user):
        """Without bypass, a user sees owned + explicitly granted entities only."""
        owner = make_user()
        other = make_user()
        db.add(Entity(type="user", id="alice", owner_id=owner.id))
        granted = Entity(type="app", id="shared-app", owner_id=other.id)
        db.add(granted)
        db.add(Entity(type="user", id="carol", owner_id=other.id))  # invisible
        db.flush()
        # Grant 'other' read on owner's alice entity.
        db.add(EntityPermission(
            entity_pk=granted.pk, grantee_id=owner.id, permission="read",
        ))
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=False, page=1, page_size=100
        )
        ids = {e.id for e in items}
        # alice (owned) + shared-app (granted); carol is invisible.
        assert ids == {"alice", "shared-app"}
        assert total == 2

    def test_empty_state(self, db, make_user):
        """A fresh user with no entities gets an empty page and zero total."""
        user = make_user()
        items, total = ep.get_visible_entities_paginated(
            user.id, db, bypass=False, page=1, page_size=100
        )
        assert items == []
        assert total == 0

    def test_unowned_only_filters_to_null_owner(self, db, make_user):
        """unowned_only=True returns only entities with owner_id IS NULL."""
        owner = make_user()
        db.add(Entity(type="user", id="alice", owner_id=owner.id))
        db.add(Entity(type="app", id="orphan-app", owner_id=None))
        db.add(Entity(type="user", id="orphan-user", owner_id=None))
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, unowned_only=True, page=1, page_size=100
        )
        assert total == 2
        assert {e.id for e in items} == {"orphan-app", "orphan-user"}

    def test_unowned_only_combined_with_type_filter(self, db, make_user):
        """unowned_only + entity_type combines both filters."""
        owner = make_user()
        db.add(Entity(type="app", id="orphan-app", owner_id=None))
        db.add(Entity(type="user", id="orphan-user", owner_id=None))
        db.flush()

        items, total = ep.get_visible_entities_paginated(
            owner.id, db, bypass=True, unowned_only=True, entity_type="user",
            page=1, page_size=100,
        )
        assert total == 1
        assert items[0].id == "orphan-user"


# --------------------------------------------------------------------------- #
# user_id wildcard
# --------------------------------------------------------------------------- #
class TestGetAccessibleUsers:
    def test_returns_owned_user_entities(self, db, make_user):
        user = make_user()
        db.add(Entity(type="user", id="alice", owner_id=user.id))
        db.add(Entity(type="user", id="bob", owner_id=user.id))
        db.flush()

        result = ep._get_accessible_users(user.id, db)
        # Own UUID is always included, plus owned entities
        assert set(result) == {str(user.id), "alice", "bob"}

    def test_returns_granted_user_entities(self, db, make_user):
        user = make_user()
        owner = make_user()
        ent = Entity(type="user", id="alice", owner_id=owner.id)
        db.add(ent)
        db.flush()
        db.add(EntityPermission(entity_pk=ent.pk, grantee_id=user.id, permission="read"))
        db.flush()

        result = ep._get_accessible_users(user.id, db)
        assert "alice" in result

    def test_excludes_orphaned_entities(self, db, make_user):
        user = make_user()
        db.add(Entity(type="user", id="orphan", owner_id=None))
        db.flush()

        result = ep._get_accessible_users(user.id, db)
        assert "orphan" not in result

    def test_includes_own_uuid_even_without_entity_row(self, db, make_user):
        """New user with no entity rows must still see their own UUID namespace."""
        user = make_user()
        result = ep._get_accessible_users(user.id, db)
        assert str(user.id) in result

    def test_returns_sub_namespaces(self, db, make_user):
        user = make_user()
        db.add(Entity(type="user", id="alice", owner_id=user.id))
        db.add(Entity(type="user", id="alice:laptop", owner_id=user.id))
        db.flush()

        result = ep._get_accessible_users(user.id, db)
        assert set(result) == {str(user.id), "alice", "alice:laptop"}

    def test_empty_for_user_with_no_access(self, db, make_user):
        """User with no owned entities and no grants, apart from own UUID."""
        user = make_user()
        result = ep._get_accessible_users(user.id, db)
        # Only own UUID (implicit access), no other entities
        assert result == [str(user.id)]


class TestUserWildcard:
    def test_admin_alone_drops_user_constraint(self, db, make_user):
        admin = make_user(role="admin")
        result = ep._rewrite_query_filter(
            {"user_id": "*"}, admin.id, db, bypass=True
        )
        # admin bypass: user_id="*" → no user constraint (empty filter)
        assert result == {}

    def test_admin_with_app_id_drops_user_constraint(self, db, make_user):
        admin = make_user(role="admin")
        result = ep._rewrite_query_filter(
            {"user_id": "*", "app_id": "x"}, admin.id, db, bypass=True
        )
        assert result == {"app_id": "x"}

    def test_member_expands_to_accessible_users(self, db, make_user):
        user = make_user()
        db.add(Entity(type="user", id="alice", owner_id=user.id))
        db.add(Entity(type="user", id="bob", owner_id=user.id))
        db.flush()

        result = ep._rewrite_query_filter(
            {"user_id": "*"}, user.id, db
        )
        # Should expand to OR of accessible user entities
        assert "OR" in result
        user_ids = {child["user_id"] for child in result["OR"]}
        assert user_ids == {str(user.id), "alice", "bob"}

    def test_member_new_user_returns_own_uuid(self, db, make_user):
        """New member with no entity rows still gets their own UUID."""
        user = make_user()
        result = ep._rewrite_query_filter(
            {"user_id": "*"}, user.id, db
        )
        assert result == {"user_id": str(user.id)}

    def test_member_no_accessible_users_raises_403(self, db, make_user):
        """If _get_accessible_users returns empty (no own UUID either), 403."""
        user = make_user()
        # Force empty by patching — but actually _get_accessible_users always
        # includes own UUID. This test verifies the guard is wired correctly:
        # if the list is somehow empty, we should 403.
        import unittest.mock as mock
        with mock.patch.object(ep, "_get_accessible_users", return_value=[]):
            with pytest.raises(HTTPException) as exc:
                ep._rewrite_query_filter({"user_id": "*"}, user.id, db)
            assert exc.value.status_code == 403

    def test_member_with_app_id_combines_and(self, db, make_user):
        user = make_user()
        db.add(Entity(type="user", id="alice", owner_id=user.id))
        db.flush()

        result = ep._rewrite_query_filter(
            {"user_id": "*", "app_id": "x"}, user.id, db
        )
        assert "AND" in result
        # Should be AND(OR(users), app_id=x)
        assert len(result["AND"]) == 2

    def test_admin_with_agent_id_preserves_agent(self, db, make_user):
        admin = make_user(role="admin")
        result = ep._rewrite_query_filter(
            {"user_id": "*", "agent_id": "riley"}, admin.id, db, bypass=True
        )
        assert result == {"agent_id": "riley"}


# --------------------------------------------------------------------------- #
# EntityResponse permission field
# --------------------------------------------------------------------------- #
class TestEntityResponsePermission:
    def test_owner_entity_gets_owner_permission(self, db, make_user):
        from server.routers.entities import _entities_to_response_batch

        user = make_user()
        ent = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(ent)
        db.flush()

        results = _entities_to_response_batch([ent], user.id, db, bypass=False)
        assert results[0].permission == "owner"

    def test_granted_entity_gets_permission_level(self, db, make_user):
        from server.routers.entities import _entities_to_response_batch

        owner = make_user()
        grantee = make_user()
        ent = Entity(type="app", id="shared-app", owner_id=owner.id)
        db.add(ent)
        db.flush()
        db.add(EntityPermission(entity_pk=ent.pk, grantee_id=grantee.id, permission="write"))
        db.flush()

        results = _entities_to_response_batch([ent], grantee.id, db, bypass=False)
        assert results[0].permission == "write"

    def test_bypass_admin_gets_admin_permission(self, db, make_user):
        from server.routers.entities import _entities_to_response_batch

        admin = make_user(role="admin")
        owner = make_user()
        ent = Entity(type="app", id="some-app", owner_id=owner.id)
        db.add(ent)
        db.flush()

        results = _entities_to_response_batch([ent], admin.id, db, bypass=True)
        assert results[0].permission == "admin"

    def test_no_permission_visible_entity_gets_none(self, db, make_user):
        from server.routers.entities import _entities_to_response_batch

        viewer = make_user()
        owner = make_user()
        ent = Entity(type="app", id="other-app", owner_id=owner.id)
        db.add(ent)
        db.flush()
        # viewer has no grant and is not owner — but the entity could be
        # visible (e.g. via admin bypass listing). permission should be None.

        results = _entities_to_response_batch([ent], viewer.id, db, bypass=False)
        assert results[0].permission is None

    def test_permission_field_in_response_model(self, db, make_user):
        from server.routers.entities import EntityResponse

        user = make_user()
        ent = Entity(type="app", id="test-app", owner_id=user.id)
        db.add(ent)
        db.flush()

        # Verify the field exists and accepts the expected values
        resp = EntityResponse(
            id=ent.id,
            type=ent.type,
            name=ent.name,
            permission="owner",
        )
        assert resp.permission == "owner"


# --------------------------------------------------------------------------- #
# user_id preserved + dual app+user permission check
# --------------------------------------------------------------------------- #
class TestRewriteQueryFilterNoDrop:
    """After the user_id dropping removal, _rewrite_query_filter must never drop user_id when app_id is present."""

    def test_keeps_user_id_with_app_id(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter(
            {"user_id": "A", "app_id": "x"}, user.id, db
        )
        assert "user_id" in result
        assert "app_id" in result

    def test_keeps_user_id_in_and_branch(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter(
            {"AND": [{"user_id": "A"}, {"app_id": "x"}]}, user.id, db
        )
        # Both should be preserved — no dropping
        assert "AND" in result
        children = result["AND"]
        has_user = any("user_id" in c and c.get("user_id") == "A" for c in children)
        has_app = any("app_id" in c for c in children)
        assert has_user
        assert has_app

    def test_keeps_user_id_with_app_id_and_agent_id(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter(
            {"user_id": "A", "app_id": "x", "agent_id": "g"}, user.id, db
        )
        assert result.get("user_id") == "A"
        assert result.get("app_id") == "x"
        assert result.get("agent_id") == "g"

    def test_app_id_alone_unchanged(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter({"app_id": "x"}, user.id, db)
        assert result == {"app_id": "x"}

    def test_user_id_alone_unchanged(self, db, make_user):
        user = make_user()
        result = ep._rewrite_query_filter({"user_id": "A"}, user.id, db)
        assert result == {"user_id": "A"}


class TestCheckMemoryScopePermissionDualCheck:
    """After the user_id dropping removal, check_memory_scope_permission checks both app AND user when both present."""

    def test_dual_check_app_and_user_read(self, db, make_user):
        """When scope has both app and user, both must pass read check."""
        user = make_user()
        # Create app entity owned by user
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        # User owns their own UUID namespace implicitly
        # Should pass: user owns both app and their own user namespace
        ep.check_memory_scope_permission(
            {"app": "my-app", "user": str(user.id)}, user.id, "read", db
        )

    def test_dual_check_fails_when_user_read_fails(self, db, make_user):
        """When app read passes but user read fails, should raise 403."""
        owner = make_user()
        viewer = make_user()
        app = Entity(type="app", id="my-app", owner_id=owner.id)
        db.add(app)
        db.flush()
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=viewer.id, permission="read"))
        db.flush()
        # viewer has app read but no user read for owner's namespace
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"app": "my-app", "user": str(owner.id)}, viewer.id, "read", db
            )
        assert exc.value.status_code == 403

    def test_app_only_still_works(self, db, make_user):
        """Scope with only app (no user) still works as before."""
        user = make_user()
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        ep.check_memory_scope_permission(
            {"app": "my-app"}, user.id, "read", db
        )

    def test_app_with_agent_skips_user_check(self, db, make_user):
        """Scope with app+agent (no user) only checks app."""
        user = make_user()
        app = Entity(type="app", id="my-app", owner_id=user.id)
        db.add(app)
        db.flush()
        ep.check_memory_scope_permission(
            {"app": "my-app", "agent": "riley"}, user.id, "read", db
        )

    def test_write_path_also_dual_checks(self, db, make_user):
        """Write/admin paths also do dual check (stricter, not a regression)."""
        owner = make_user()
        other = make_user()
        app = Entity(type="app", id="my-app", owner_id=owner.id)
        db.add(app)
        db.flush()
        db.add(EntityPermission(entity_pk=app.pk, grantee_id=other.id, permission="write"))
        db.flush()
        # other has app write but no user write for owner's namespace
        with pytest.raises(HTTPException) as exc:
            ep.check_memory_scope_permission(
                {"app": "my-app", "user": str(owner.id)}, other.id, "write", db
            )
        assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# is_wildcard / strip_user_id_for_app_gate / entity id validation
# --------------------------------------------------------------------------- #
class TestIsWildcard:
    def test_matches_star(self):
        from utils.helpers import is_wildcard
        assert is_wildcard("*") is True

    def test_rejects_other_strings(self):
        from utils.helpers import is_wildcard
        assert is_wildcard("alice") is False
        assert is_wildcard("") is False
        assert is_wildcard("**") is False

    def test_rejects_non_string(self):
        from utils.helpers import is_wildcard
        assert is_wildcard(None) is False
        assert is_wildcard(42) is False


class TestStripUserIdForAppGate:
    def test_strips_user_id_when_app_id_present(self):
        result = ep.strip_user_id_for_app_gate({"user_id": "alice", "app_id": "x"})
        assert result == {"app_id": "x"}

    def test_preserves_user_id_when_no_app_id(self):
        result = ep.strip_user_id_for_app_gate({"user_id": "alice"})
        assert result == {"user_id": "alice"}

    def test_preserves_other_keys(self):
        result = ep.strip_user_id_for_app_gate(
            {"user_id": "alice", "app_id": "x", "agent_id": "riley", "created_at": {"gte": "2024"}}
        )
        assert result == {"app_id": "x", "agent_id": "riley", "created_at": {"gte": "2024"}}

    def test_handles_non_dict(self):
        assert ep.strip_user_id_for_app_gate([]) == []
        assert ep.strip_user_id_for_app_gate("string") == "string"


class TestEnsureUserEntity:
    """Regression: _ensure_user_entity should NOT reject short IDs or wildcards
    (that validation belongs in the API layer)."""

    def test_short_id_still_works(self, db, make_user):
        """A 2-char user_id is valid on the write path."""
        user = make_user()
        entity = ep._ensure_user_entity("ab", user.id, db)
        assert entity.id == "ab"
        assert entity.owner_id == user.id

    def test_uuid_still_works(self, db, make_user):
        user = make_user()
        entity = ep._ensure_user_entity(str(user.id), user.id, db)
        assert entity.id == str(user.id)
