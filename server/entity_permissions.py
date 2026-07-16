"""Entity-level ownership and permission isolation for the self-hosted server (v2).

Every memory scope (``user_id`` / ``agent_id`` / ``app_id`` / ``run_id``) is treated
as an *entity namespace*.

Entity hierarchy:
- user : global namespace with ``:``-delimited sub-namespace; top-level user entities
  are first-claim (write-time auto-create). Sub-namespaces (e.g. ``A:B``) are auto-created
  on write if the caller owns the first-level prefix (``A``).
- app  : global namespace, created by admin (not first-claim). Supports explicit grants.
- agent: user-scoped via ``parent_pk`` -> user entity. Auto-created on first write.
  No explicit grants.
- run  : user-scoped via ``parent_pk`` -> user entity. Auto-created on first write.
  No explicit grants.

Permission model:
- Single memory READ   : OR  — read on *any one* entity in the memory's scope.
- Single memory WRITE  : AND — write on *all* entities in the scope.
- Single memory DELETE  : AND — admin on *all* entities in the scope.
- Query (search/list)  : per *branch* (each OR branch must independently pass READ).
- Bulk destructive      : prescan every matched memory and AND-check ADMIN on each.

Owner vs admin:
- owner (owner_user_id): full control — read/write, grant/revoke read+write, delete entity,
  transfer owner.
- admin (granted): read/write + grant/revoke read+write. Cannot delete entity, grant admin,
  or transfer owner.
"""

import logging
import os
import uuid
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import case, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import determine_user, is_bootstrap_admin
from entity import (
    ENTITY_PARAMS,
    FIELD_TO_TYPE,
    SCOPED_ENTITY_TYPES,
    TYPE_TO_FIELD,
    canonicalize_entity_id,
    is_scoped_entity_type,
    params_to_entities,
    top_level_user_id,
    user_prefixes,
)
from models import Entity, EntityPermission, User
from server_state import get_memory_instance
from utils.helpers import normalize_results, unwrap_result

logger = logging.getLogger(__name__)

_LEVELS = {"read": 1, "write": 2, "admin": 3}

# Only non-admin, top-level user entities count toward the quota.
MAX_OWNED_ENTITIES_PER_USER = int(os.environ.get("MAX_OWNED_ENTITIES_PER_USER", "10"))

# Vector-store scan cap for first-claim counts / delete prescans.
_SCAN_TOP_K = 1_000_000


# --------------------------------------------------------------------------- #
# Operator resolution (FastAPI adapter)
# --------------------------------------------------------------------------- #
def resolve_operator(request: Request, auth: User | None, db: Session) -> tuple[User, bool]:
    """Resolve the acting user and whether they have the admin bypass.

    Returns ``(user, is_admin)``:

    - real user (bearer / API key) -> (user, user.role == "admin")
    - admin_api_key               -> (bootstrap admin, True): admin bypass; callers
      must skip ``ensure_entity_owner`` (does not own any entity row).
    - auth_disabled               -> (_get_default_user(db), default.role == "admin"):
      a real user that owns entities normally.
    """
    result = determine_user(auth, getattr(request.state, "auth_type", "none"), db)
    if result is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return result


def reject_bootstrap_memory_mutation(operator: User) -> None:
    """Forbid the admin_api_key bypass from authoring or mutating memories.

    Bootstrap is a governance + read-only principal: it manages entities (create
    app, grant/revoke/transfer, delete entity) and reads all data, but never
    authors or mutates scoped memories. Memories must be authored by a real user
    so every memory has a scoped owner. Call this right after ``resolve_operator``
    on every memory-mutation endpoint (add / update / delete / delete_all).

    Gates on ``is_bootstrap_admin`` (the nil-UUID sentinel), NOT ``is_admin`` —
    real ``role="admin"`` dashboard users have ``bypass=True`` but
    ``is_bootstrap_admin=False`` and must remain able to author memories.
    """
    if is_bootstrap_admin(operator.id):
        raise HTTPException(
            status_code=403,
            detail="admin_api_key cannot author or mutate memories; use a real user API key or JWT session.",
        )


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #
# ``canonicalize_entity_id`` lives in the ``entity`` primitives module.


# --------------------------------------------------------------------------- #
# User entity helpers (hierarchical namespace)
# --------------------------------------------------------------------------- #
# ``top_level_user_id`` / ``user_prefixes`` live in the ``entity`` primitives module.


def _subnamespace_prefix_condition(parent_id: str):
    """SQL condition matching user sub-namespaces of *parent_id* — ``Entity.id``
    starts with ``parent_id + ':'`` — with LIKE wildcards escaped.

    Entity ids are user-controlled and may contain ``%`` / ``_`` (e.g. ``john_doe``);
    an unescaped ``startswith`` would compile to ``LIKE 'john_doe:%'`` where ``_``
    matches any char, letting ``user/john_doe`` match ``user/johnX:foo`` and trigger
    cross-entity cascade transfer/deletion. We escape ``\\`` / ``%`` / ``_`` so the
    prefix is matched literally.
    """
    escaped = (parent_id + ":").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return Entity.id.like(escaped + "%", escape="\\")


def _get_user_entity_or_none(entity_id: str, db: Session) -> Entity | None:
    """Look up a user entity by exact entity_id."""
    eid = canonicalize_entity_id("user", entity_id)
    return db.scalar(
        select(Entity).where(Entity.type == "user", Entity.id == eid)
    )


def _resolve_parent_user_id(scope: dict[str, str], user_id: uuid.UUID) -> str | None:
    """Resolve the parent user namespace for agent/run checks.

    Prefer an explicit ``user`` in the memory scope; fall back to the operator's
    own id (bootstrap admin has no real id, so it gets ``None`` — agent/run under
    a bootstrap write are not parent-resolvable).
    """
    return scope.get("user") or (str(user_id) if not is_bootstrap_admin(user_id) else None)


def _resolve_user_owner(entity_id: str, db: Session) -> tuple[uuid.UUID | None, Entity | None]:
    """Find the owning user for a user entity_id via longest-prefix match.

    Returns ``(owner_user_id, matched_entity)`` for the longest *owned* prefix
    (skips orphaned entities, ``owner_user_id is None``). If no owned prefix
    matches, returns ``(None, None)``. Skipping orphans keeps this consistent with
    ``check_entity_permission`` (orphan = admin-only, fall through to shorter
    prefixes) and avoids the longer-orphan masking a shorter owned prefix.
    """
    prefixes = [canonicalize_entity_id("user", p) for p in user_prefixes(entity_id)]
    entities = {
        e.id: e
        for e in db.scalars(
            select(Entity).where(Entity.type == "user", Entity.id.in_(prefixes))
        ).all()
    }
    for prefix in prefixes:
        entity = entities.get(prefix)
        if entity is not None and entity.owner_user_id is not None:
            return entity.owner_user_id, entity
    return None, None


# --------------------------------------------------------------------------- #
# Vector-store helpers
# --------------------------------------------------------------------------- #
def _result_id(row: Any) -> str | None:
    if row is None:
        return None
    if isinstance(row, dict):
        mid = row.get("id")
    else:
        mid = getattr(row, "id", None)
    return str(mid) if mid is not None else None


def entity_parent_user_id(entity: Entity, db: Session) -> str | None:
    """For an agent/run entity, the parent user entity's id (used to scope
    vector-store queries/counts since same-named agent/run are unique per parent).
    Returns ``None`` for user/app or when the parent row no longer exists."""
    if is_scoped_entity_type(entity.type) and entity.parent_pk is not None:
        parent = db.get(Entity, entity.parent_pk)
        if parent is not None:
            return parent.id
    return None


def entity_filter_params(entity: Entity, db: Session) -> dict[str, str]:
    """Flat entity-param filter matching *entity*'s vector-store memories, scoped
    by the parent user_id for agent/run (so a same-named agent/run owned by another
    user is not matched)."""
    params: dict[str, str] = {TYPE_TO_FIELD[entity.type]: entity.id}
    parent_user_id = entity_parent_user_id(entity, db)
    if parent_user_id is not None:
        params["user_id"] = parent_user_id
    return params


def count_memories_for_entity(
    entity_type: str,
    entity_id: str,
    *,
    parent_user_id: str | None = None,
) -> int:
    """Count memories currently in the vector store for this entity.

    For agent/run types, ``parent_user_id`` should be provided to scope the
    count to the owning user's namespace (different users may have
    agent/run entities with the same id). Counts are advisory (not a security
    path), so a vector-store error is logged and reported as 0 rather than
    failing the request.
    """
    field = TYPE_TO_FIELD[entity_type]
    filters: dict[str, Any] = {field: entity_id}
    if is_scoped_entity_type(entity_type) and parent_user_id:
        filters["user_id"] = parent_user_id
    try:
        raw = get_memory_instance().get_all(filters=filters, top_k=_SCAN_TOP_K)
    except Exception as exc:
        if isinstance(exc, (NameError, AttributeError, SyntaxError)):
            raise
        logger.exception("count_memories_for_entity: vector-store scan failed for %s/%s", entity_type, entity_id)
        return 0
    return len(normalize_results(raw))


def list_memory_ids_for_params(entity_params: dict[str, Any]) -> list[str]:
    """Return memory ids matching a flat entity-param filter (bulk-delete prescan).

    This is the bulk-delete prescan, a security-critical path: returning an empty
    list on a transient vector-store error would let ``validate_bulk_admin_operation``
    pass trivially and the subsequent delete proceed without authorization. Fail
    closed by raising 503 instead — REST's ``except HTTPException: raise`` passes it
    through and MCP's ``_mcp_raise`` converts it to ValueError.
    """
    if not entity_params:
        return []
    try:
        raw = get_memory_instance().get_all(filters=dict(entity_params), top_k=_SCAN_TOP_K)
    except Exception:
        logger.exception("list_memory_ids_for_params: prescan failed for %r", entity_params)
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable during delete prescan; the delete was aborted.",
        )
    ids: list[str] = []
    for row in normalize_results(raw):
        mid = _result_id(row)
        if mid is not None:
            ids.append(mid)
    return ids


# --------------------------------------------------------------------------- #
# Entity lookup helpers
# --------------------------------------------------------------------------- #
def get_entity_or_none(
    entity_type: str,
    entity_id: str,
    db: Session,
    *,
    parent_entity_id: str | None = None,
) -> Entity | None:
    """Look up an entity by (type, id).

    agent/run are unique per parent user, not globally. Pass ``parent_entity_id``
    (the parent user entity's string id, e.g. ``"<uuid>"`` or ``"<uuid>:laptop"``)
    to scope the lookup to a specific parent namespace. user/app are globally
    unique and ignore the filter.
    """
    eid = canonicalize_entity_id(entity_type, entity_id)
    stmt = select(Entity).where(Entity.type == entity_type, Entity.id == eid)
    if is_scoped_entity_type(entity_type) and parent_entity_id is not None:
        parent_eid = canonicalize_entity_id("user", parent_entity_id)
        parent_stmt = select(Entity.pk).where(Entity.type == "user", Entity.id == parent_eid)
        stmt = stmt.where(Entity.parent_pk == parent_stmt.scalar_subquery())
    return db.scalar(stmt)


def _get_entity_or_404(
    entity_type: str,
    entity_id: str,
    db: Session,
) -> Entity:
    entity = get_entity_or_none(entity_type, entity_id, db)
    if entity is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_type}/{entity_id}' not found.",
        )
    return entity

def _get_scoped_entity_or_none(
    entity_type: str,
    parent_pk: uuid.UUID,
    entity_id: str,
    db: Session,
) -> Entity | None:
    """Look up an agent/run entity by (parent_pk, entity_id).

    agent/run are unique per parent user, not globally, so the lookup is scoped
    to *parent_pk*. *entity_type* must be ``"agent"`` or ``"run"``.
    """
    eid = canonicalize_entity_id(entity_type, entity_id)
    return db.scalar(
        select(Entity).where(
            Entity.type == entity_type,
            Entity.parent_pk == parent_pk,
            Entity.id == eid,
        )
    )


def count_owned_entities(user_id: uuid.UUID, db: Session) -> int:
    """Number of top-level user entities (no ``:``) a user owns.

    Only top-level user entities count toward the quota. agent/run/app are excluded.
    """
    return (
        db.scalar(
            select(func.count())
            .select_from(Entity)
            .where(
                Entity.owner_user_id == user_id,
                Entity.type == "user",
                ~Entity.id.contains(":"),
            )
        )
        or 0
    )


# --------------------------------------------------------------------------- #
# Ownership (first-claim / auto-create)
# --------------------------------------------------------------------------- #
def _create_entity_row(
    entity_type: str,
    entity_id: str,
    entity_name: str | None,
    owner_user_id: uuid.UUID,
    parent_pk: uuid.UUID | None,
    db: Session,
) -> Entity:
    """Create an entity row (with savepoint for race safety)."""
    try:
        with db.begin_nested():
            entity = Entity(
                type=entity_type,
                id=entity_id,
                owner_user_id=owner_user_id,
                parent_pk=parent_pk,
                name=entity_name,
            )
            db.add(entity)
            db.flush()
        return entity
    except IntegrityError:
        raise


def _assert_uuid_reservation(entity_id: str, user_id: uuid.UUID, bypass: bool) -> None:
    """Reject claiming a UUID-namespaced user entity owned by a different user.

    A user entity whose canonicalized id parses as a UUID is reserved for the
    user with that UUID. Callers pass already-canonicalized ids (and top-level
    segments thereof), so a parseable id is in canonical UUID form — no extra
    canonical-equality check is needed.
    """
    try:
        parsed = uuid.UUID(entity_id)
    except (ValueError, TypeError):
        return
    if parsed != user_id and not bypass:
        raise HTTPException(
            status_code=403,
            detail=f"Entity 'user/{entity_id}' is reserved for the user with that ID and cannot be claimed.",
        )


def _first_claim_toplevel(
    entity_id: str,
    user_id: uuid.UUID,
    bypass: bool,
    db: Session,
) -> None:
    """First-claim the top-level segment of *entity_id* (no owned prefix exists).

    Enforces the UUID-reservation and quota rules, then creates the top-level
    entity. Race-safe: a concurrent first-claim is reconciled via the partial
    unique index (orphan rows are claimed; rows owned by another user 403).
    """
    top = top_level_user_id(entity_id)
    if top != entity_id:
        # The top-level segment must not be another user's UUID namespace.
        _assert_uuid_reservation(top, user_id, bypass)

    if not bypass and count_owned_entities(user_id, db) >= MAX_OWNED_ENTITIES_PER_USER:
        raise HTTPException(
            status_code=403,
            detail=(
                f"You already own the maximum number of entities ({MAX_OWNED_ENTITIES_PER_USER}). "
                "Ask an admin to raise the limit, or transfer/delete entities you no longer need."
            ),
        )

    try:
        _create_entity_row("user", top, None, user_id, None, db)
    except IntegrityError:
        # Race: another request created the top-level entity first.
        existing_top = _get_user_entity_or_none(top, db)
        if existing_top is None:
            raise
        if existing_top.owner_user_id is None:
            # Orphaned top-level entity (e.g. owner deleted): claim it.
            existing_top.owner_user_id = user_id
            db.flush()
        elif existing_top.owner_user_id != user_id and not bypass:
            raise HTTPException(
                status_code=403,
                detail=f"Entity 'user/{top}' is already owned by another user.",
            )


def _claim_or_get_user_entity(
    entity_id: str,
    user_id: uuid.UUID,
    db: Session,
) -> Entity | None:
    """Return the user entity if it exists, claiming it if orphaned (owner=None)."""
    existing = _get_user_entity_or_none(entity_id, db)
    if existing is not None and existing.owner_user_id is None and user_id is not None:
        existing.owner_user_id = user_id
        db.flush()
    return existing


def _ensure_user_entity(
    entity_id: str,
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
) -> Entity:
    """Ensure a user entity exists: first-claim if new, or verify ownership.

    - If the entity exists (exact match), return it.
    - If a prefix is owned by the caller, create the exact entity_id.
    - If no prefix exists, create the top-level entity (first-claim) + the exact entity_id.
    - UUID entity_ids are reserved for the user with that UUID.
    """
    entity_id = canonicalize_entity_id("user", entity_id)

    # Exact match (already owned) short-circuits.
    existing = _get_user_entity_or_none(entity_id, db)
    if existing is not None and existing.owner_user_id is not None:
        return existing

    # UUID entity_id: only the user with that UUID can claim it.
    _assert_uuid_reservation(entity_id, user_id, bypass)

    # Resolve the longest owned prefix: if one exists the caller must own it,
    # otherwise first-claim the top-level segment.
    owner_id, matched_entity = _resolve_user_owner(entity_id, db)
    if owner_id is not None:
        if owner_id != user_id and not bypass:
            raise HTTPException(
                status_code=403,
                detail=f"Entity 'user/{entity_id}' falls under 'user/{matched_entity.id}' which is owned by another user.",
            )
    else:
        _first_claim_toplevel(entity_id, user_id, bypass, db)

    # Create / claim the exact entity_id when it is a sub-namespace.
    if entity_id != top_level_user_id(entity_id):
        existing = _claim_or_get_user_entity(entity_id, user_id, db)
        if existing is not None:
            return existing
        try:
            return _create_entity_row("user", entity_id, None, user_id, None, db)
        except IntegrityError:
            existing = _claim_or_get_user_entity(entity_id, user_id, db)
            if existing is None:
                raise
            return existing

    # entity_id == top-level; already created above.
    existing = _get_user_entity_or_none(entity_id, db)
    if existing is not None:
        return existing
    raise HTTPException(status_code=500, detail="Failed to create entity.")


def _ensure_agent_or_run_entity(
    entity_type: str,
    entity_id: str,
    parent_entity: Entity,
    db: Session,
) -> Entity:
    """Auto-create an agent/run entity under the given user parent entity."""
    entity_id = canonicalize_entity_id(entity_type, entity_id)
    owner_user_id = parent_entity.owner_user_id
    parent_pk = parent_entity.pk

    existing = _get_scoped_entity_or_none(entity_type, parent_pk, entity_id, db)
    if existing is not None:
        return existing

    try:
        return _create_entity_row(entity_type, entity_id, None, owner_user_id, parent_pk, db)
    except IntegrityError:
        existing = _get_scoped_entity_or_none(entity_type, parent_pk, entity_id, db)
        if existing is None:
            raise
        return existing


def ensure_entity_owner(
    entity_type: str,
    entity_id: str,
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
    parent_entity_id: str | None = None,
) -> Entity:
    """Create or verify an entity for the given type and id.

    - user: hierarchical namespace, first-claim for top-level.
    - app: must already exist (created by admin). Non-admin gets 403.
    - agent/run: auto-created under the user entity specified by ``parent_entity_id``.
    """
    if entity_type == "user":
        return _ensure_user_entity(entity_id, user_id, db, bypass=bypass)

    if entity_type == "app":
        entity = get_entity_or_none("app", entity_id, db)
        if entity is None:
            raise HTTPException(
                status_code=403,
                detail=f"App '{entity_id}' has not been created yet. Ask an admin to create it.",
            )
        if entity.owner_user_id is None:
            raise HTTPException(
                status_code=403,
                detail=f"App '{entity_id}' has no owner yet. Ask an admin to assign one.",
            )
        return entity

    if is_scoped_entity_type(entity_type):
        # Resolve parent user entity
        if parent_entity_id is None:
            parent_entity_id = _resolve_parent_user_id({}, user_id)
            if parent_entity_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot resolve parent user for agent/run without a real user identity.",
                )
        parent_entity = ensure_entity_owner("user", parent_entity_id, user_id, db, bypass=bypass)
        # Verify caller owns the parent user entity
        if not bypass and parent_entity.owner_user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail=f"You do not own 'user/{parent_entity_id}', so you cannot create {entity_type}/{entity_id} under it.",
            )
        return _ensure_agent_or_run_entity(entity_type, entity_id, parent_entity, db)

    raise HTTPException(status_code=400, detail=f"Unknown entity type: '{entity_type}'.")


def create_app_entity(entity_id: str, owner_user_id: uuid.UUID, db: Session, *, name: str | None = None) -> Entity:
    """Create an app entity owned by *owner_user_id* (admin-only manual creation).

    Validates that *owner_user_id* is an existing user (404 otherwise) and that
    the app does not already exist (409 otherwise). Race-safe via the partial
    unique index. Commits. The caller is responsible for the admin-role check.
    """
    if db.get(User, owner_user_id) is None:
        raise HTTPException(status_code=404, detail=f"User '{owner_user_id}' not found.")
    try:
        entity = _create_entity_row("app", entity_id, name, owner_user_id, None, db)
    except IntegrityError:
        if get_entity_or_none("app", entity_id, db) is not None:
            raise HTTPException(status_code=409, detail=f"App '{entity_id}' already exists.")
        raise
    db.commit()
    return entity


# --------------------------------------------------------------------------- #
# Permission checks
# --------------------------------------------------------------------------- #
_MAX_FILTER_DEPTH = 10


# --------------------------------------------------------------------------- #
# Query filter rewriting (app-as-primary-gate)
# --------------------------------------------------------------------------- #
def _filter_has_app_id(filters: Any) -> bool:
    """Return True if *filters* contains ``app_id`` anywhere in the tree."""
    if isinstance(filters, dict):
        if "app_id" in filters:
            return True
        for key in ("AND", "OR"):
            if _filter_has_app_id(filters.get(key)):
                return True
    elif isinstance(filters, list):
        return any(_filter_has_app_id(item) for item in filters)
    return False


def _get_accessible_apps(user_id: uuid.UUID, db: Session) -> list[str]:
    """Return app entity ids a **non-admin** user can read (owner or explicitly granted).

    Non-orphaned apps only — orphaned entities are admin-only. Admins never reach
    here: ``app_id:"*"`` for an admin is resolved to *all* apps at the wildcard
    expansion site (``_rewrite_query_filter``'s ``_get_apps`` helper), since
    "accessible apps" for an admin is the entire app set, not a permission-scoped
    subset.
    """
    # Owned or explicitly granted (non-orphaned), combined in one round-trip.
    owned = select(Entity.id).where(
        Entity.type == "app",
        Entity.owner_user_id == user_id,
    )
    granted = (
        select(Entity.id)
        .join(EntityPermission, EntityPermission.entity_pk == Entity.pk)
        .where(
            Entity.type == "app",
            Entity.owner_user_id.is_not(None),
            EntityPermission.user_id == user_id,
            EntityPermission.permission.in_(["read", "write", "admin"]),
        )
    )
    return sorted(set(db.scalars(owned.union(granted)).all()))


def _rewrite_query_filter(
    filters: Any,
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
) -> Any:
    """Rewrite a query filter for the app-as-primary-gate model.

    Rules (applied in order):

    1. ``app_id: "*"`` → expanded to ``{"OR": [{"app_id": "a"}, ...]}``
       (403 if no accessible apps).
    2. If a node directly contains ``app_id`` (not ``"*"``), drop ``user_id``
       from that node.
    3. Empty ``{}`` children are removed from AND/OR; empty AND/OR → ``{}``.
    4. Recursion depth is capped at ``_MAX_FILTER_DEPTH`` (400 on overflow).
    5. NOT subtrees are recursively rewritten (admin may pass ``NOT`` + ``"*"``).
    6. Mixed dicts (both logical operators AND top-level entity fields) are
       handled by processing logical operators first, then applying rule 2
       to the top-level fields.

    Within an AND combination (a flat dict, or an explicit ``AND`` list), if any
    sibling carries ``app_id``, ``user_id`` is a pure data tag and is dropped from
    every sibling — this handles ``{"AND": [{"user_id": "A"}, {"app_id": "x"}]}``.
    OR siblings are intentionally left intact: each is a legitimate alternative
    branch, so ``{"OR": [{"user_id": "A"}, {"app_id": "x"}]}`` keeps both.

    7. For a non-bootstrap caller, a leaf carrying ``agent_id``/``run_id`` but no
       ``user_id`` and no ``app_id`` gets the caller's ``user_id`` injected.
       agent/run are unique per parent user (not globally), so without this the
       vector-store query would match every user's same-named agent/run memories
       (cross-user leak). Mirrors the write path's ``inject_default_user_id``.
    """
    if not isinstance(filters, dict) or not filters:
        return filters

    accessible_apps: list[str] | None = None  # lazy

    def _get_apps() -> list[str]:
        nonlocal accessible_apps
        if accessible_apps is None:
            if bypass:
                # Admin has full access, so "*" spans every app. mem0 filters
                # have no "field exists" operator, so "all memories with an
                # app_id" can only be expressed by enumerating the app ids. The
                # admin decision belongs here at wildcard expansion, not inside
                # the non-admin _get_accessible_apps helper.
                accessible_apps = sorted(
                    db.scalars(select(Entity.id).where(Entity.type == "app")).all()
                )
            else:
                accessible_apps = _get_accessible_apps(user_id, db)
        return accessible_apps

    def _drop_user_id(node: Any, in_metadata: bool = False) -> Any:
        """Strip entity-scope ``user_id`` from *node* (a ``metadata`` block's
        ``user_id`` is user-defined and left intact)."""
        if isinstance(node, list):
            return [_drop_user_id(child, in_metadata) for child in node]
        if not isinstance(node, dict):
            return node
        result = {
            k: _drop_user_id(v, in_metadata or k == "metadata")
            for k, v in node.items()
        }
        if not in_metadata:
            result.pop("user_id", None)
        return result

    def _walk(node: Any, depth: int = 0, in_not: bool = False) -> Any:
        if depth > _MAX_FILTER_DEPTH:
            raise HTTPException(
                status_code=400,
                detail="Filter tree is too deeply nested.",
            )

        if node is None:
            return {}

        if isinstance(node, list):
            # AND list: rewrite each child, filter out empty {}
            result = []
            for child in node:
                rewritten = _walk(child, depth + 1, in_not=in_not)
                if isinstance(rewritten, dict) and not rewritten:
                    continue
                if isinstance(rewritten, list):
                    result.extend(rewritten)
                else:
                    result.append(rewritten)
            # app-as-primary-gate within this AND conjunction: if any sibling
            # carries app_id, user_id is a pure data tag across the whole
            # conjunction — drop it from every sibling. OR siblings are left
            # intact (each is a legitimate alternative branch).
            if any(_filter_has_app_id(child) for child in result):
                result = [_drop_user_id(child) for child in result]
                result = [child for child in result if not (isinstance(child, dict) and not child)]
            if len(result) == 0:
                return {}
            if len(result) == 1:
                return result[0]
            return result

        if not isinstance(node, dict):
            return node

        # --- Process logical operators first ---
        rewritten_and = None
        rewritten_or = None
        rewritten_not = None

        if "AND" in node:
            rewritten_and = _walk(node["AND"], depth + 1, in_not=in_not)
            # An empty AND is handled by the _cleanup post-pass, which pops the
            # operator while keeping sibling keys — do NOT early-return {} here
            # (that would discard siblings like app_id/created_at).

        if "OR" in node:
            or_children = node["OR"]
            if isinstance(or_children, list):
                children = []
                for child in or_children:
                    rewritten = _walk(child, depth + 1, in_not=in_not)
                    if isinstance(rewritten, dict) and not rewritten:
                        continue
                    if isinstance(rewritten, list):
                        children.extend(rewritten)
                    else:
                        children.append(rewritten)
                if not children:
                    rewritten_or = {}
                elif len(children) == 1:
                    rewritten_or = children[0]
                else:
                    rewritten_or = children
            else:
                rewritten_or = _walk(or_children, depth + 1, in_not=in_not)

        if "NOT" in node:
            rewritten_not = _walk(node["NOT"], depth + 1, in_not=True)

        # --- Build result ---
        result = dict(node)
        if rewritten_and is not None:
            result["AND"] = rewritten_and
        if rewritten_or is not None:
            result["OR"] = rewritten_or
        if rewritten_not is not None:
            result["NOT"] = rewritten_not

        # --- Rule 2 & Rule 1: handle top-level entity fields ---
        # Recursive: a nested app_id (e.g. inside an AND-subtree of this mixed
        # dict) also makes the top-level user_id a data tag.
        has_app = _filter_has_app_id(result)

        # When app_id is a *direct* key, every other key is in an implicit AND
        # with it, so user_id is a data tag across the whole node. Strip user_id
        # from sibling logical-operator subtrees (AND/OR/NOT) too — otherwise
        # _cleanup could unwrap a single-child operator and promote user_id back
        # to this level alongside app_id (violating app-as-primary-gate).
        if "app_id" in result:
            for key in list(result.keys()):
                # app_id is the gate; metadata is user-defined data (its inner
                # user_id is not entity scope). Strip user_id from everything else.
                if key not in ("app_id", "metadata"):
                    result[key] = _drop_user_id(result[key])
            result.pop("user_id", None)

            if result.get("app_id") == "*":
                apps = _get_apps()
                if not apps:
                    raise HTTPException(
                        status_code=403,
                        detail="No accessible app scope for wildcard query.",
                    )
                if in_not and len(apps) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail="app_id wildcard ('*') inside NOT with multiple apps produces an unsupported filter.",
                    )
                expanded: dict[str, Any] = {"OR": [{"app_id": app} for app in apps]}
                # Remove the wildcard (user_id already stripped above).
                del result["app_id"]
                has_logical = rewritten_and is not None or rewritten_or is not None or rewritten_not is not None
                if has_logical or result:
                    return {"AND": [expanded, result]}
                return expanded

            return result

        if has_app:
            # app_id is nested (e.g. inside an AND-subtree), not a direct key:
            # the top-level user_id is still a data tag and must be dropped.
            result.pop("user_id", None)
        else:
            # Rule 7: no app scope — an agent_id/run_id-only query is ambiguous
            # across user namespaces (agent/run are unique per parent user, not
            # globally), so scope it to the caller's namespace. Bootstrap admin
            # keeps the unscoped admin bypass. Recursion injects into nested
            # AND/OR leaves too.
            if (
                not is_bootstrap_admin(user_id)
                and ("agent_id" in result or "run_id" in result)
                and "user_id" not in result
            ):
                result["user_id"] = str(user_id)

        return result

    rewritten = _walk(filters)

    # --- Cleanup: remove empty dicts and unwrap single-child AND/OR ---
    def _cleanup(node: Any) -> Any:
        if isinstance(node, list):
            result = []
            for child in node:
                c = _cleanup(child)
                if isinstance(c, dict) and not c:
                    continue
                result.append(c)
            if len(result) == 0:
                return {}
            if len(result) == 1:
                return result[0]
            return result
        if not isinstance(node, dict):
            return node
        result = dict(node)
        for op in ("AND", "OR"):
            if op in result:
                cleaned = _cleanup(result[op])
                if isinstance(cleaned, dict) and not cleaned:
                    # Operator became empty: drop just the op key, keep sibling
                    # keys (e.g. created_at, metadata) rather than discarding
                    # the whole node.
                    result.pop(op, None)
                    continue
                # Unwrap single-child AND/OR
                if isinstance(cleaned, dict) and op not in cleaned:
                    # Unwrap: merge the child's keys with the parent's non-op keys.
                    # Recurse so carried sibling operators (OR/NOT and their
                    # children) are cleaned too — a bare `return merged` here used
                    # to skip them via this early return path.
                    merged = dict(cleaned)
                    merged.update({k: v for k, v in result.items() if k != op})
                    return _cleanup(merged)
                result[op] = cleaned
        if "NOT" in result:
            cleaned = _cleanup(result["NOT"])
            if isinstance(cleaned, dict) and not cleaned:
                # NOT became empty (its only term was a data-tag user_id stripped
                # under app-as-primary-gate): drop the residue, keep sibling keys.
                result.pop("NOT", None)
            else:
                result["NOT"] = cleaned
        return result

    rewritten = _cleanup(rewritten)

    return rewritten


def check_entity_permission(
    entity_type: str,
    entity_id: str,
    user_id: uuid.UUID,
    required: str,
    db: Session,
    *,
    bypass: bool = False,
    parent_entity_id: str | None = None,
) -> bool:
    """Whether ``user_id`` has ``required`` permission on the entity. Never raises."""
    entity_id = canonicalize_entity_id(entity_type, entity_id)

    if bypass:
        return True

    # ``user`` type: a user fully owns their own UUID namespace and any
    # sub-namespace under it (e.g. ``<uuid>:laptop``), even before the entity
    # rows are lazily created on first write — UNLESS the namespace has been
    # transferred to another user, in which case access is governed by the
    # hierarchical ownership/grant check below (so the new owner can revoke it).
    if entity_type == "user":
        try:
            if uuid.UUID(top_level_user_id(entity_id)) == user_id:
                top_entity = _get_user_entity_or_none(top_level_user_id(entity_id), db)
                if (
                    top_entity is None
                    or top_entity.owner_user_id is None
                    or top_entity.owner_user_id == user_id
                ):
                    return True
                # Transferred to another user: fall through to the hierarchical
                # ownership/grant check below.
        except (ValueError, TypeError):
            pass

    # For agent/run: look up via parent_pk
    if is_scoped_entity_type(entity_type):
        if parent_entity_id is None:
            return False
        parent_entity = _get_user_entity_or_none(parent_entity_id, db)
        if parent_entity is None:
            return False
        entity = _get_scoped_entity_or_none(entity_type, parent_entity.pk, entity_id, db)
        if entity is None:
            return False
        if entity.owner_user_id == user_id:
            return True
        # agent/run do not support explicit grants
        return False

    # user: hierarchical — ownership and explicit grants inherit across all
    # prefixes (owning/grant on user/alice covers user/alice:laptop). Longest
    # existing prefix wins. Fetch all prefix rows in one query (avoids an N+1 for
    # deep namespaces like A:B:C:D:E).
    if entity_type == "user":
        canonical_prefixes = [canonicalize_entity_id("user", p) for p in user_prefixes(entity_id)]
        prefix_entities = {
            e.id: e
            for e in db.scalars(
                select(Entity).where(Entity.type == "user", Entity.id.in_(canonical_prefixes))
            ).all()
        }
        # Batch-fetch the grants for all prefix entities in one query (avoids a
        # per-prefix N+1 on deep namespaces like A:B:C:D:E).
        prefix_grants: dict[uuid.UUID, EntityPermission] = {}
        if prefix_entities:
            prefix_grants = {
                p.entity_pk: p
                for p in db.scalars(
                    select(EntityPermission).where(
                        EntityPermission.entity_pk.in_([e.pk for e in prefix_entities.values()]),
                        EntityPermission.user_id == user_id,
                    )
                ).all()
            }
        # Effective owner = owner of the longest existing, non-orphan prefix. In
        # a clean namespace every existing owned prefix shares this owner (a sub-
        # namespace cannot be created under another user's namespace), so owning
        # any prefix grants the whole subtree. Differing owners only arise from
        # corrupted/orphaned state — in that case ownership of a shorter prefix
        # must NOT bypass a longer prefix owned by another user (cross-user leak).
        effective_owner: uuid.UUID | None = None
        for prefix in canonical_prefixes:  # longest-first
            prefix_entity = prefix_entities.get(prefix)
            if prefix_entity is not None and prefix_entity.owner_user_id is not None:
                effective_owner = prefix_entity.owner_user_id
                break
        if effective_owner == user_id:
            return True
        # Grant path: a sufficient grant on any existing non-orphan prefix covers
        # this entity and all of its sub-namespaces (including unclaimed ones).
        # A grant on a prefix owned by a different user must NOT apply — the
        # sub-namespace API (ensure_entity_owner) prevents this state in normal
        # operation, but defense-in-depth guards against DB corruption, manual
        # inserts, or migration artifacts (same as the effective_owner guard
        # above for the ownership path).
        for prefix in canonical_prefixes:  # longest-first
            prefix_entity = prefix_entities.get(prefix)
            if prefix_entity is None or prefix_entity.owner_user_id is None:
                continue  # missing or orphaned: skip, keep scanning for a grant
            if prefix_entity.owner_user_id != effective_owner:
                continue  # defense-in-depth: mismatched owner → skip grant
            perm = prefix_grants.get(prefix_entity.pk)
            if perm is not None and _LEVELS.get(perm.permission, 0) >= _LEVELS.get(required, 0):
                return True
        return False

    # app: exact entity, ownership + explicit grants (no hierarchy)
    entity = get_entity_or_none(entity_type, entity_id, db)
    if entity is None:
        return False  # unclaimed entity: non-admins cannot access
    if entity.owner_user_id is None:
        return False  # owner deleted / not yet assigned: admin only
    if entity.owner_user_id == user_id:
        return True

    # Check explicit grants
    perm = db.scalar(
        select(EntityPermission).where(
            EntityPermission.entity_pk == entity.pk,
            EntityPermission.user_id == user_id,
        )
    )
    if perm is None:
        return False
    return _LEVELS.get(perm.permission, 0) >= _LEVELS.get(required, 0)


def check_memory_scope_permission(
    scope: dict[str, str],
    user_id: uuid.UUID,
    required: str,
    db: Session,
    *,
    bypass: bool = False,
) -> None:
    """Authorize a single memory's scope.

    When scope contains ``app``, only the app entity is checked (app is the
    primary permission gate). Otherwise READ=OR, WRITE/ADMIN=AND applies.
    Empty scope: admin only.
    """
    if not scope:
        if not bypass:
            raise HTTPException(status_code=403, detail="Memory has no entity scope; admin only.")
        return

    # --- app as primary gate ---
    if "app" in scope:
        ok = check_entity_permission(
            "app", scope["app"], user_id, required, db,
            bypass=bypass,
        )
        if not ok:
            raise HTTPException(
                status_code=403,
                detail=f"You do not have '{required}' permission for app '{scope['app']}'.",
            )
        return

    # --- original logic: no app in scope ---
    parent_entity_id = _resolve_parent_user_id(scope, user_id)

    results = [
        check_entity_permission(
            et, eid, user_id, required, db,
            bypass=bypass,
            parent_entity_id=parent_entity_id if is_scoped_entity_type(et) else None,
        )
        for et, eid in scope.items()
    ]
    ok = any(results) if required == "read" else all(results)
    if not ok:
        scope_str = ", ".join(f"{et}/{eid}" for et, eid in scope.items())
        raise HTTPException(
            status_code=403,
            detail=f"You do not have '{required}' permission for entity scope [{scope_str}].",
        )


# --------------------------------------------------------------------------- #
# Query (filter-tree) authorization
# --------------------------------------------------------------------------- #
def extract_query_scope_branches(filters: Any) -> tuple[list[dict[str, str]], bool]:
    """Extract positive entity-scope branches from a filter tree.

    Returns ``(branches, has_not)``. Each branch is ``{entity_type: entity_id}``.

    - flat dict / AND list : merged into one branch
    - OR                   : one branch per child (recursively)
    - NOT                  : contributes no positive scope, but sets ``has_not`` anywhere
      in the tree (NOT can match memories outside every positive branch, so callers
      must reject non-admin NOT queries wholesale).
    """
    has_not = False

    def scope_from_flat(node: dict) -> dict[str, str]:
        scope: dict[str, str] = {}
        for field in ENTITY_PARAMS:
            value = node.get(field)
            if isinstance(value, str) and value.strip():
                scope[FIELD_TO_TYPE[field]] = value.strip()
        return scope

    def merge(a: dict[str, str], b: dict[str, str]) -> dict[str, str] | None:
        if not a:
            return dict(b)
        if not b:
            return dict(a)
        out = dict(a)
        for key, value in b.items():
            if key in out and out[key] != value:
                return None  # conflicting entity value -> impossible branch
            out[key] = value
        return out

    def and_combine(branch_lists: list[list[dict[str, str]]]) -> list[dict[str, str]]:
        acc = [{}]
        for branches in branch_lists:
            nxt: list[dict[str, str]] = []
            for a in acc:
                for b in branches:
                    merged = merge(a, b)
                    if merged is not None:
                        nxt.append(merged)
            acc = nxt if nxt else [{}]
        return acc

    def walk(node: Any, depth: int = 0) -> list[dict[str, str]]:
        if depth > _MAX_FILTER_DEPTH:
            raise HTTPException(
                status_code=400,
                detail="Filter tree is too deeply nested.",
            )
        nonlocal has_not
        if isinstance(node, list):
            return and_combine([walk(child, depth + 1) for child in node if isinstance(child, (dict, list))])
        if not isinstance(node, dict):
            return [{}]

        local = scope_from_flat(node)
        or_sub = node.get("OR")
        and_sub = node.get("AND")
        not_sub = node.get("NOT")

        branches: list[dict[str, str]] = [local]

        if or_sub is not None:
            or_children = or_sub if isinstance(or_sub, list) else [or_sub]
            or_branches: list[dict[str, str]] = []
            for child in or_children:
                or_branches.extend(walk(child, depth + 1))
            if local:
                combined = [m for m in (merge(local, b) for b in or_branches) if m is not None]
                branches = combined if combined else [local]
            else:
                branches = or_branches

        if and_sub is not None:
            and_children = and_sub if isinstance(and_sub, list) else [and_sub]
            and_branches = and_combine([walk(child, depth + 1) for child in and_children])
            combined = [
                m
                for b in branches
                for ab in and_branches
                for m in [merge(b, ab)]
                if m is not None
            ]
            branches = combined if combined else [{}]

        if not_sub is not None:
            has_not = True
            walk(not_sub, depth + 1)  # propagate has_not for nested NOTs; positive scope discarded

        return branches

    if not isinstance(filters, dict) or not filters:
        return [], has_not
    return walk(filters), has_not


def check_query_permission(
    filters: Any,
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
) -> dict[str, Any]:
    """Authorize a query and return the rewritten filter for the vector store.

    Steps:

    1. Rewrite filter (drop ``user_id`` where ``app_id`` is present, expand
       ``app_id: "*"`` to all accessible apps).
    2. Extract positive scope branches from the rewritten filter.
    3. Check each branch with the single-memory READ rule.
    4. Return the rewritten filter.

    NOT-containing queries are rejected for non-admins wholesale. No positive
    scope -> admin only.
    """
    # Step 1: Rewrite filter
    rewritten = _rewrite_query_filter(filters, user_id, db, bypass=bypass)

    # Step 2: Extract branches from rewritten filter
    branches, has_not = extract_query_scope_branches(rewritten)
    if has_not and not bypass:
        raise HTTPException(
            status_code=403,
            detail="Queries containing NOT operators require admin privileges.",
        )
    if not branches:
        if not bypass:
            raise HTTPException(
                status_code=403,
                detail="Query has no positive entity scope; admin only.",
            )
        return rewritten

    # Step 3: Check each branch
    for branch in branches:
        check_memory_scope_permission(branch, user_id, "read", db, bypass=bypass)

    # Step 4: Return rewritten filter
    return rewritten


# --------------------------------------------------------------------------- #
# Per-memory scope resolution
# --------------------------------------------------------------------------- #
def resolve_memory_entities(memory_id: str) -> dict[str, str]:
    """Return ``{entity_type: entity_id}`` for a memory (404 if not found).

    Distinguishes "not found" (404) from "exists but no entity scope" (returns ``{}``,
    which ``check_memory_scope_permission`` treats as admin-only).
    """
    from memory_lock import entity_scope_from_record

    memory = get_memory_instance()
    try:
        raw = memory.get(memory_id)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        logger.warning("resolve_memory_entities: data error reading %s: %s", memory_id, exc)
        raw = None
    item = unwrap_result(raw)
    if not isinstance(item, dict):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    field_scope = entity_scope_from_record(item)
    return {
        FIELD_TO_TYPE[field]: value
        for field, value in field_scope.items()
        if field in FIELD_TO_TYPE
    }


def validate_bulk_admin_operation(
    memory_ids: list[str],
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
    scope_hint: dict[str, str] | None = None,
) -> None:
    """Bulk destructive ops must AND-check ADMIN on every matched memory's full scope.

    Fast path: when *scope_hint* contains ``app`` (e.g., app-scoped
    ``delete_all``), the app-primary gate makes the admin check identical for
    every matched memory, so we check once instead of issuing one vector-store
    lookup per memory (which can be millions for large namespaces). Non-app
    scopes fall back to per-memory resolution because each memory may carry
    different agent/run scopes that also require admin.
    """
    # Admins have global access — every memory is in their admin scope — so the
    # per-memory admin check (and the resolve_memory_entities vector-store read
    # it requires) is entirely redundant. Short-circuit before the loop.
    if bypass:
        return
    if scope_hint and "app" in scope_hint:
        check_memory_scope_permission(scope_hint, user_id, "admin", db, bypass=bypass)
        return
    for memory_id in memory_ids:
        scope = resolve_memory_entities(memory_id)
        check_memory_scope_permission(scope, user_id, "admin", db, bypass=bypass)


def bulk_delete_memories(
    memory,
    params: dict[str, Any],
    operator_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
) -> None:
    """Prescan, admin-validate, and delete — call inside ``run_memory_write``.

    The prescan runs inside the scope lock so a concurrent cross-scope write
    cannot sneak a memory into the delete set after validation (TOCTOU).

    ``list_memory_ids_for_params`` raises ``HTTPException(503)`` on vector-store
    failure; MCP callers must wrap with ``_mcp_raise`` to convert to ``ValueError``.
    """
    memory_ids = list_memory_ids_for_params(params)
    scope_hint = params_to_entities(params)
    validate_bulk_admin_operation(
        memory_ids, operator_id, db, bypass=bypass, scope_hint=scope_hint,
    )
    memory.delete_all(**params)


# --------------------------------------------------------------------------- #
# Write-path authorization (shared by REST compat, main, and MCP)
# --------------------------------------------------------------------------- #
# ``params_to_entities`` lives in the ``entity`` primitives module.


def inject_default_user_id(
    entity_params: dict[str, Any],
    operator: User,
    *,
    skip_if_has_app: bool = False,
) -> dict[str, Any]:
    """Scope the write to the caller by injecting their user_id if missing.

    agent/run/app entities live under a user namespace, so every write must
    carry the caller's user_id — otherwise the stored memory has no user scope
    and a different user who owns a same-named agent/run could read or modify
    it (cross-user leak). Bootstrap (admin_api_key) is rejected at the endpoint
    before reaching here, so this always injects for the real-user operator.

    When *skip_if_has_app* is True and ``app_id`` is present in *entity_params*,
    user_id injection is skipped (app-scoped operations do not need user_id
    for scoping, and injecting it would narrow delete_all / vector-store queries).
    """
    if skip_if_has_app and "app_id" in entity_params and entity_params["app_id"]:
        return entity_params
    if "user_id" not in entity_params or not entity_params["user_id"]:
        entity_params = dict(entity_params)
        entity_params["user_id"] = str(operator.id)
    return entity_params


def authorize_write(
    entity_params: dict[str, Any],
    operator: User,
    db: Session,
    *,
    bypass: bool = False,
) -> None:
    """Two-pass write authorization + first-claim ownership, then a full-scope write check.

    Pass 1 rejects early if any *existing* entity in the scope lacks write — so a
    request that would 403 cannot first claim the brand-new namespaces in its scope.
    Pass 2 claims brand-new namespaces. Finally the whole scope is AND-checked.

    When ``app`` is present in the scope, only the app entity is checked in Pass 1
    (app is the primary permission gate); user/agent/run entities are still created
    in Pass 2 as data tags.

    Bootstrap (admin_api_key) is rejected at the endpoint before reaching here, so
    this always runs the full two-pass + AND-check for a real-user operator.
    """
    scope = params_to_entities(entity_params)
    parent_entity_id = _resolve_parent_user_id(scope, operator.id)

    has_app = "app" in scope

    # Pass 1: check existing entities for write permission
    for entity_type, entity_id in scope.items():
        # When app is present, only check app entity (skip user/agent/run)
        if has_app and entity_type != "app":
            continue

        if is_scoped_entity_type(entity_type):
            if parent_entity_id:
                parent_entity = _get_user_entity_or_none(parent_entity_id, db)
                if parent_entity is None:
                    continue
                existing = _get_scoped_entity_or_none(entity_type, parent_entity.pk, entity_id, db)
            else:
                existing = None
        else:
            existing = get_entity_or_none(entity_type, entity_id, db)

        if existing is not None and not check_entity_permission(
            entity_type, entity_id, operator.id, "write", db, bypass=bypass,
            parent_entity_id=parent_entity_id if is_scoped_entity_type(entity_type) else None,
        ):
            raise HTTPException(
                status_code=403,
                detail=f"You do not have 'write' permission for entity scope [{entity_type}/{entity_id}].",
            )

    # Pass 2: create/claim entities
    for entity_type, entity_id in scope.items():
        ensure_entity_owner(
            entity_type, entity_id, operator.id, db,
            bypass=bypass,
            parent_entity_id=parent_entity_id if is_scoped_entity_type(entity_type) else None,
        )

    db.commit()
    check_memory_scope_permission(scope, operator.id, "write", db, bypass=bypass)


# --------------------------------------------------------------------------- #
# Grant / revoke / list
# --------------------------------------------------------------------------- #
def _assert_can_manage(
    entity: Entity,
    entity_type: str,
    entity_id: str,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
) -> bool:
    """Permission gate shared by grant/revoke/list: operator must be the owner or
    hold admin (global admin or an explicit admin grant). Returns ``is_owner`` so
    callers can apply the stricter owner-only rule (grant/revoke admin). Raises
    403 otherwise.
    """
    is_owner = entity.owner_user_id is not None and entity.owner_user_id == operator_id
    operator_has_admin = bypass or is_owner or (
        operator_id is not None
        and check_entity_permission(entity_type, entity_id, operator_id, "admin", db)
    )
    if not (is_owner or operator_has_admin):
        raise HTTPException(
            status_code=403,
            detail="Only entity owner/admin can manage permissions.",
        )
    return is_owner


def is_owner_or_global_admin(
    entity: Entity,
    operator_id: uuid.UUID | None,
    bypass: bool,
) -> bool:
    """Owner-or-global-admin tier (excludes granted admin).

    Strictest management tier: entity deletion and ownership transfer require
    the operator to be the owner or a global admin — an explicit admin grant is
    not enough. Contrast with ``_assert_can_manage`` (grant/revoke/list), which
    also accepts an explicit admin grant.
    """
    return bypass or (
        entity.owner_user_id is not None and entity.owner_user_id == operator_id
    )


def upsert_permission(
    entity_pk: uuid.UUID,
    user_id: uuid.UUID,
    permission: str,
    *,
    granted_by: uuid.UUID | None,
    db: Session,
) -> EntityPermission:
    """Insert or update a permission row (DB-agnostic; race-safe via rollback+retry)."""
    existing = db.scalar(
        select(EntityPermission).where(
            EntityPermission.entity_pk == entity_pk,
            EntityPermission.user_id == user_id,
        )
    )
    if existing is None:
        try:
            with db.begin_nested():
                perm = EntityPermission(
                    entity_pk=entity_pk,
                    user_id=user_id,
                    permission=permission,
                    granted_by=granted_by,
                )
                db.add(perm)
                db.flush()
            return perm
        except IntegrityError:
            existing = db.scalar(
                select(EntityPermission).where(
                    EntityPermission.entity_pk == entity_pk,
                    EntityPermission.user_id == user_id,
                )
            )
            if existing is None:
                raise
    existing.permission = permission
    existing.granted_by = granted_by
    db.flush()
    return existing


def grant_entity_permission(
    entity_type: str,
    entity_id: str,
    grantee_user_id: uuid.UUID,
    permission: str,
    *,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
) -> EntityPermission:
    """Grant read/write/admin to a user.

    - Only user and app entities support explicit grants.
    - Only owner can grant admin; admin can grant read/write.
    """
    if permission not in _LEVELS:
        raise HTTPException(status_code=400, detail="Permission must be one of: read, write, admin.")

    if is_scoped_entity_type(entity_type):
        raise HTTPException(
            status_code=400,
            detail=f"Entity type '{entity_type}' does not support explicit permission grants.",
        )

    entity = _get_entity_or_404(entity_type, entity_id, db)

    # Determine what level the operator has
    is_owner = _assert_can_manage(entity, entity_type, entity_id, operator_id, bypass, db)

    # Only owner can grant admin
    if permission == "admin" and not is_owner and not bypass:
        raise HTTPException(
            status_code=403,
            detail="Only the entity owner can grant admin permissions.",
        )

    grantee = db.get(User, grantee_user_id)
    if grantee is None:
        raise HTTPException(status_code=404, detail=f"User '{grantee_user_id}' not found.")

    granted_by = None if is_bootstrap_admin(operator_id) else operator_id
    perm = upsert_permission(entity.pk, grantee_user_id, permission, granted_by=granted_by, db=db)
    db.commit()
    return perm


def revoke_entity_permission(
    entity_type: str,
    entity_id: str,
    grantee_user_id: uuid.UUID,
    *,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
) -> None:
    """Revoke a granted permission. Cannot revoke the owner."""
    if is_scoped_entity_type(entity_type):
        raise HTTPException(
            status_code=400,
            detail=f"Entity type '{entity_type}' does not support explicit permission grants.",
        )

    entity = _get_entity_or_404(entity_type, entity_id, db)
    if entity.owner_user_id is not None and entity.owner_user_id == grantee_user_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot revoke the owner's permissions.",
        )

    is_owner = _assert_can_manage(entity, entity_type, entity_id, operator_id, bypass, db)

    # Only owner can revoke admin grants
    perm = db.scalar(
        select(EntityPermission).where(
            EntityPermission.entity_pk == entity.pk,
            EntityPermission.user_id == grantee_user_id,
        )
    )
    if perm is not None:
        if perm.permission == "admin" and not is_owner and not bypass:
            raise HTTPException(
                status_code=403,
                detail="Only the entity owner can revoke admin permissions.",
            )
        db.delete(perm)
        db.commit()
    else:
        raise HTTPException(
            status_code=404,
            detail=f"User '{grantee_user_id}' does not have any permission on '{entity_type}/{entity_id}'.",
        )


def list_entity_permissions(
    entity_type: str,
    entity_id: str,
    *,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
) -> list[EntityPermission]:
    if is_scoped_entity_type(entity_type):
        raise HTTPException(
            status_code=400,
            detail=f"Entity type '{entity_type}' does not support explicit permission grants.",
        )

    entity = _get_entity_or_404(entity_type, entity_id, db)
    _assert_can_manage(entity, entity_type, entity_id, operator_id, bypass, db)
    return (
        db.execute(
            select(EntityPermission)
            .where(EntityPermission.entity_pk == entity.pk)
            .order_by(EntityPermission.created_at.asc())
        )
        .scalars()
        .all()
    )


# --------------------------------------------------------------------------- #
# Transfer ownership
# --------------------------------------------------------------------------- #
def transfer_entity_owner(
    entity_type: str,
    entity_id: str,
    new_owner_user_id: uuid.UUID,
    *,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
) -> Entity:
    """Transfer entity ownership.

    - Only user and app entities support transfer.
    - Only the owner or global admin can transfer.
    - For user entities, only top-level transfer is allowed; all sub-entities
      (user sub-namespaces, agent, run) have their owner_user_id cascaded.
    - The previous owner keeps an explicit admin grant.
    """
    if is_scoped_entity_type(entity_type):
        raise HTTPException(
            status_code=400,
            detail=f"Entity type '{entity_type}' does not support ownership transfer.",
        )

    entity = _get_entity_or_404(entity_type, entity_id, db)

    if not is_owner_or_global_admin(entity, operator_id, bypass):
        raise HTTPException(
            status_code=403,
            detail="Only entity owner/admin can transfer ownership.",
        )

    # For user: must be top-level (no `:`)
    if entity_type == "user" and ":" in entity.id:
        raise HTTPException(
            status_code=400,
            detail="Only top-level user entities can be transferred. Transfer the parent entity instead.",
        )

    new_owner = db.get(User, new_owner_user_id)
    if new_owner is None:
        raise HTTPException(status_code=404, detail=f"User '{new_owner_user_id}' not found.")

    # Quota check for non-admin recipients. Skip when the target is already the
    # owner (no-op transfer / re-assignment) — count_owned_entities would include
    # this entity and incorrectly 403 if the current owner is at their limit.
    if new_owner.role != "admin" and entity_type == "user" and entity.owner_user_id != new_owner_user_id:
        if count_owned_entities(new_owner_user_id, db) >= MAX_OWNED_ENTITIES_PER_USER:
            raise HTTPException(
                status_code=403,
                detail=f"Target user already owns the maximum number of entities ({MAX_OWNED_ENTITIES_PER_USER}).",
            )

    previous_owner_id = entity.owner_user_id

    # For user entities: cascade to all sub-entities
    if entity_type == "user":
        # Find all user sub-entities with this prefix (escaped LIKE — see
        # _subnamespace_prefix_condition).
        sub_users = (
            db.execute(
                select(Entity).where(
                    Entity.type == "user",
                    _subnamespace_prefix_condition(entity.id),
                )
            )
            .scalars()
            .all()
        )
        for sub in sub_users:
            sub.owner_user_id = new_owner_user_id

        # Find all agent/run entities whose parent_pk points to any of these user entities
        affected_user_ids = [entity.pk] + [sub.pk for sub in sub_users]
        children = (
            db.execute(
                select(Entity).where(
                    Entity.type.in_(SCOPED_ENTITY_TYPES),
                    Entity.parent_pk.in_(affected_user_ids),
                )
            )
            .scalars()
            .all()
        )
        for child in children:
            child.owner_user_id = new_owner_user_id

    # Transfer the entity itself
    entity.owner_user_id = new_owner_user_id

    # Grant previous owner explicit admin
    if previous_owner_id and previous_owner_id != new_owner_user_id:
        granted_by = None if is_bootstrap_admin(operator_id) else operator_id
        upsert_permission(entity.pk, previous_owner_id, "admin", granted_by=granted_by, db=db)

    db.commit()
    return entity


# --------------------------------------------------------------------------- #
# Visibility (list)
# --------------------------------------------------------------------------- #
def get_visible_entities(
    user_id: uuid.UUID, db: Session, *, bypass: bool = False
) -> list[Entity]:
    """Entities the user can see: owned + explicitly granted (admins see all)."""
    if bypass:
        return db.execute(select(Entity)).scalars().all()

    permitted_ids = select(EntityPermission.entity_pk).where(EntityPermission.user_id == user_id)
    return (
        db.execute(
            select(Entity).where(
                or_(
                    Entity.owner_user_id == user_id,
                    Entity.owner_user_id.is_not(None) & Entity.pk.in_(permitted_ids),
                )
            )
        )
        .scalars()
        .all()
    )


def get_visible_entities_paginated(
    user_id: uuid.UUID,
    db: Session,
    *,
    bypass: bool = False,
    entity_type: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> tuple[list[Entity], int]:
    """Paginated variant of :func:`get_visible_entities`.

    Returns ``(page_items, total)``. Ordering puts entities owned by ``user_id``
    first (so the caller's own namespaces surface at the top of the list), then by
    type and id. ``entity_type`` optionally restricts to one type (e.g. ``"user"``)
    so the memories-page user picker can fetch only user entities.
    """
    visibility = or_(
        Entity.owner_user_id == user_id,
        Entity.owner_user_id.is_not(None)
        & Entity.pk.in_(
            select(EntityPermission.entity_pk).where(EntityPermission.user_id == user_id)
        ),
    )
    where_clause = [] if bypass else [visibility]
    if entity_type is not None:
        where_clause.append(Entity.type == entity_type)

    # Owned-first ordering: 0 for owned, 1 for everything else.
    owned_first = case((Entity.owner_user_id == user_id, 0), else_=1)

    base = select(Entity)
    if where_clause:
        base = base.where(*where_clause)

    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    items = (
        db.execute(
            base.order_by(owned_first, Entity.type, Entity.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .scalars()
        .all()
    )
    return items, total


# --------------------------------------------------------------------------- #
# Child-entity collection (cascade deletion safety)
# --------------------------------------------------------------------------- #
def collect_user_children(entity: Entity, db: Session) -> list[Entity]:
    """For a user entity, return all descendant entities (sub-namespaces + agent/run).

    Sub-namespaces (``alice:laptop``) are matched by ``id`` prefix; agent/run
    descendants are matched via ``parent_pk`` on the parent and every sub-namespace.
    Returns ``[]`` for non-user entities.

    Order is FK-safe for deletion: agent/run descendants first, then sub-namespaces
    (agent/run rows reference user rows via ``parent_pk``).
    """
    if entity.type != "user":
        return []

    sub_namespaces = list(
        db.execute(
            select(Entity).where(
                Entity.type == "user",
                _subnamespace_prefix_condition(entity.id),
            )
        )
        .scalars()
        .all()
    )
    all_user_pks = [entity.pk] + [s.pk for s in sub_namespaces]
    agent_run_children = list(
        db.execute(
            select(Entity).where(
                Entity.type.in_(SCOPED_ENTITY_TYPES),
                Entity.parent_pk.in_(all_user_pks),
            )
        )
        .scalars()
        .all()
    )
    return agent_run_children + sub_namespaces


# --------------------------------------------------------------------------- #
# Owner-only deletion check
# --------------------------------------------------------------------------- #
def check_entity_delete_permission(
    entity_type: str,
    entity_id: str,
    operator_id: uuid.UUID | None,
    bypass: bool,
    db: Session,
    *,
    parent_entity_id: str | None = None,
) -> Entity | None:
    """Verify the operator can delete this entity: owner or global admin only.

    A granted admin cannot delete entities (design: delete = owner only). Returns
    the entity on success, or raises 404 if no entity row exists.

    Pass ``parent_entity_id`` for agent/run so the lookup is scoped to that
    parent user namespace (agent/run are unique per parent, not globally).
    """
    entity = get_entity_or_none(
        entity_type, entity_id, db,
        parent_entity_id=parent_entity_id,
    )
    if entity is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_type}/{entity_id}' not found.",
        )
    if is_owner_or_global_admin(entity, operator_id, bypass):
        return entity
    raise HTTPException(
        status_code=403,
        detail="Only the entity owner can delete this entity.",
    )
