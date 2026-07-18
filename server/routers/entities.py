import logging
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from auth import is_bootstrap_admin, verify_auth
from db import get_db
from entity import EntityType, is_scoped_entity_type
from entity_permissions import (
    check_entity_delete_permission,
    check_entity_permission,
    collect_user_children,
    count_memories_for_entity,
    create_app_entity,
    ensure_entity_owner,
    entity_filter_params,
    get_parent_entity_id,
    get_entity_or_none,
    get_visible_entities_paginated,
    grant_entity_permission,
    list_entity_permissions,
    list_memory_ids_for_params,
    resolve_operator,
    revoke_entity_permission,
    transfer_entity_owner,
    validate_bulk_admin_operation,
)
from errors import upstream_error
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from memory_lock import run_memory_write
from models import Entity, EntityPermission, User
from pydantic import BaseModel, Field
from schemas import MessageResponse, UserInfo
from sqlalchemy import select
from sqlalchemy.orm import Session
from utils.helpers import is_wildcard
from utils.pagination import paginate_response

router = APIRouter(prefix="/entities", tags=["entities"])

logger = logging.getLogger(__name__)

PermissionType = Literal["read", "write", "admin"]


# --------------------------------------------------------------------------- #
# Response / input models
# --------------------------------------------------------------------------- #
class ParentEntityInfo(BaseModel):
    """Minimal parent entity reference for agent/run entities."""

    id: str
    type: EntityType
    name: Optional[str] = None


class EntityResponse(BaseModel):
    """Entity summary. ``id`` is the external identifier; ``name`` is the display name."""

    id: str
    type: EntityType
    name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    owner: Optional[UserInfo] = None
    parent: Optional[ParentEntityInfo] = None
    is_owner: bool = False
    permission: Optional[Literal["owner", "admin", "write", "read"]] = None


class EntityPermissionResponse(BaseModel):
    id: str
    grantee: UserInfo
    permission: PermissionType
    grantor: Optional[UserInfo] = None
    created_at: datetime


class EntityListResponse(BaseModel):
    """Paginated envelope mirroring ``compat.responses.paginate_response``."""

    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: list[EntityResponse]


class CreateEntityInput(BaseModel):
    type: EntityType
    id: str
    owner_id: Optional[str] = None  # admin only, for app creation
    name: Optional[str] = Field(None, max_length=255)


class GrantPermissionInput(BaseModel):
    grantee_id: str
    permission: PermissionType


class TransferOwnerInput(BaseModel):
    owner_id: str


class UpdateEntityInput(BaseModel):
    name: Optional[str] = Field(None, max_length=255)


def _user_info(user_id: uuid.UUID, db: Session) -> UserInfo:
    user = db.get(User, user_id)
    if user is not None:
        return UserInfo(id=str(user.id), name=user.name, email=user.email)
    return UserInfo(id=str(user_id), name="Unknown", email="")


def _user_info_batch(user_ids: set[uuid.UUID], db: Session) -> dict[uuid.UUID, UserInfo]:
    """Batch-fetch users for populating permission/entity responses."""
    if not user_ids:
        return {}
    users = {
        u.id: UserInfo(id=str(u.id), name=u.name, email=u.email)
        for u in db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()
    }
    for uid in user_ids:
        if uid not in users:
            users[uid] = UserInfo(id=str(uid), name="Unknown", email="")
    return users


def _entities_to_response_batch(
    entities: list[Entity], operator_id: uuid.UUID, db: Session, *, bypass: bool = False
) -> list[EntityResponse]:
    """Batch-resolve parent entities, owner users, and operator permissions."""
    # Collect parent PKs and owner user IDs
    parent_pks: set[uuid.UUID] = set()
    owner_ids: set[uuid.UUID] = set()
    grant_pks: set[uuid.UUID] = set()
    for e in entities:
        if e.parent_pk is not None:
            parent_pks.add(e.parent_pk)
        if e.owner_id is not None:
            owner_ids.add(e.owner_id)
        if e.owner_id != operator_id:
            grant_pks.add(e.pk)

    # Batch-fetch parents
    parents: dict[uuid.UUID, Entity] = {}
    if parent_pks:
        parents = {p.pk: p for p in db.execute(select(Entity).where(Entity.pk.in_(parent_pks))).scalars().all()}

    # Batch-fetch owners
    owners = _user_info_batch(owner_ids, db)

    # Batch-fetch explicit grants for non-owned entities
    grants: dict[uuid.UUID, str] = {}
    if grant_pks:
        grants = {
            p.entity_pk: p.permission
            for p in db.execute(
                select(EntityPermission).where(
                    EntityPermission.entity_pk.in_(grant_pks),
                    EntityPermission.grantee_id == operator_id,
                    EntityPermission.permission.in_(["read", "write", "admin"]),
                )
            ).scalars().all()
        }

    # Build responses
    result: list[EntityResponse] = []
    for entity in entities:
        parent: Optional[ParentEntityInfo] = None
        if entity.parent_pk is not None:
            parent_entity = parents.get(entity.parent_pk)
            if parent_entity is not None:
                parent = ParentEntityInfo(
                    id=parent_entity.id,
                    type=parent_entity.type,
                    name=parent_entity.name,
                )
        # Resolve permission
        permission: Optional[Literal["owner", "admin", "write", "read"]] = None
        if entity.owner_id == operator_id:
            permission = "owner"
        elif entity.pk in grants:
            permission = grants[entity.pk]
        elif bypass:
            permission = "admin"

        result.append(
            EntityResponse(
                id=entity.id,
                type=entity.type,
                name=entity.name,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                owner=owners.get(entity.owner_id) if entity.owner_id else None,
                parent=parent,
                is_owner=entity.owner_id == operator_id,
                permission=permission,
            )
        )
    return result


def _entity_to_response(
    entity: Entity, operator_id: uuid.UUID, db: Session, *, bypass: bool = False
) -> EntityResponse:
    """Single-entity convenience wrapper around ``_entities_to_response_batch``."""
    return _entities_to_response_batch([entity], operator_id, db, bypass=bypass)[0]


def _permission_to_response(perm: EntityPermission, db: Session) -> EntityPermissionResponse:
    grantor: Optional[UserInfo] = None
    if perm.grantor_id is not None:
        grantor = _user_info(perm.grantor_id, db)

    return EntityPermissionResponse(
        id=str(perm.id),
        grantee=_user_info(perm.grantee_id, db),
        permission=perm.permission,
        grantor=grantor,
        created_at=perm.created_at,
    )


def _permissions_to_response_batch(perms: list[EntityPermission], db: Session) -> list[EntityPermissionResponse]:
    """Batch-resolve all user references in one query instead of N+1."""
    user_ids: set[uuid.UUID] = set()
    for p in perms:
        if p.grantee_id is not None:
            user_ids.add(p.grantee_id)
        if p.grantor_id is not None:
            user_ids.add(p.grantor_id)
    users = _user_info_batch(user_ids, db)

    return [
        EntityPermissionResponse(
            id=str(p.id),
            grantee=users[p.grantee_id],
            permission=p.permission,
            grantor=users.get(p.grantor_id) if p.grantor_id else None,
            created_at=p.created_at,
        )
        for p in perms
    ]


def _parse_uuid(value: str, *, label: str = "UUID") -> uuid.UUID:
    """Parse a UUID string, raising 400 on invalid input."""
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{value}'.")


# --------------------------------------------------------------------------- #
# List / create
# --------------------------------------------------------------------------- #
@router.get("", response_model=EntityListResponse)
def list_entities(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    entity_type: Optional[EntityType] = Query(
        None, alias="type", description="Filter to one entity type (e.g. 'user')."
    ),
    view_as: Optional[str] = Query(None, description="Admin-only: view entities visible to this user (UUID)."),
    unowned_only: bool = Query(False, description="Only return entities without an owner."),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Page through visible entities. Owned entities are sorted first.

    Pagination is done at the DB level (``LIMIT``/``OFFSET`` + ``COUNT``). The
    optional ``type`` filter (query param ``?type=user``) narrows to a single
    type — the memories-page user picker uses ``type=user`` to fetch only user
    namespaces.

    When ``view_as`` is provided (admin only), the list is scoped to entities
    visible to that user (owned + explicitly granted). Without it, behaviour is
    unchanged: admins see all, non-admins see their own.

    ``unowned_only`` returns entities whose ``owner_id`` is NULL — useful for
    surfacing unclaimed namespaces on the dashboard.
    """
    operator, is_admin = resolve_operator(request, auth, db)

    if view_as is not None:
        if not is_admin:
            raise HTTPException(status_code=403, detail="Only admins can scope to another user.")
        target_id = _parse_uuid(view_as, label="view_as")
        items, total = get_visible_entities_paginated(
            target_id,
            db,
            bypass=False,
            entity_type=entity_type,
            unowned_only=unowned_only,
            page=page,
            page_size=page_size,
        )
    else:
        items, total = get_visible_entities_paginated(
            operator.id,
            db,
            bypass=is_admin,
            entity_type=entity_type,
            unowned_only=unowned_only,
            page=page,
            page_size=page_size,
        )
    results = _entities_to_response_batch(items, operator.id, db, bypass=is_admin)
    return paginate_response(request, results, page, page_size, total=total)


@router.post("", response_model=EntityResponse, status_code=201)
def create_entity(
    body: CreateEntityInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, is_admin = resolve_operator(request, auth, db)

    if is_scoped_entity_type(body.type):
        raise HTTPException(
            status_code=400,
            detail=f"Entity type '{body.type}' cannot be manually created. It is auto-created on first write.",
        )

    entity_id = body.id.strip()
    entity_name = (body.name or "").strip() or None

    # "*" is reserved as a wildcard in query filters and cannot be used as an entity id.
    if is_wildcard(entity_id):
        raise HTTPException(
            status_code=400,
            detail="Entity id '*' is reserved for wildcard queries and cannot be used.",
        )

    # entity id must be at least 3 characters.
    if len(entity_id) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Entity id must be at least 3 characters, got '{entity_id}'.",
        )

    # user: UI-created, entity_id cannot contain ':'
    if body.type == "user":
        if ":" in entity_id:
            raise HTTPException(
                status_code=400,
                detail="User entity_id cannot contain ':'. Sub-namespaces are created automatically on write.",
            )
        # UUID entity_ids are not allowed for manual creation.
        try:
            parsed_uuid = uuid.UUID(entity_id)
        except (ValueError, TypeError):
            parsed_uuid = None
        if parsed_uuid is not None:
            if parsed_uuid == operator.id:
                raise HTTPException(
                    status_code=400,
                    detail="Your own user_id is already yours by default; no entity needs to be created.",
                )
            raise HTTPException(
                status_code=400,
                detail="UUID user_ids cannot be created manually. Use a non-UUID identifier (e.g. 'alice').",
            )
        if is_bootstrap_admin(operator.id):
            raise HTTPException(
                status_code=400,
                detail="Cannot create a user entity without a real owner.",
            )
        entity = ensure_entity_owner(body.type, entity_id, operator.id, db, bypass=is_admin)
        if entity.owner_id != operator.id:
            raise HTTPException(
                status_code=403,
                detail=f"Entity '{body.type}/{entity_id}' is already owned by another user.",
            )
        if entity_name and entity.name is None:
            entity.name = entity_name
        db.commit()
        return _entity_to_response(entity, operator.id, db, bypass=is_admin)

    # app: admin only, must specify owner_id
    if body.type == "app":
        if not is_admin:
            raise HTTPException(
                status_code=403,
                detail="Only administrators can create app entities.",
            )
        if body.owner_id is None:
            raise HTTPException(
                status_code=400,
                detail="owner_id is required when creating an app entity.",
            )
        owner_id = _parse_uuid(body.owner_id, label="owner_id")
        entity = create_app_entity(entity_id, owner_id, db, name=entity_name)
        return _entity_to_response(entity, operator.id, db, bypass=is_admin)

    raise HTTPException(status_code=400, detail=f"Unknown entity type: '{body.type}'.")


@router.get("/{entity_type}/{entity_id}/count", summary="Count entity memories in real-time")
def count_entity_memories(
    entity_type: EntityType,
    entity_id: str,
    request: Request,
    parent_id: str | None = None,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Return the current memory count for this entity by scanning the vector store.

    For agent/run entities (unique per parent user), ``parent_id`` scopes the
    count to a specific user namespace (e.g. ``"<uuid>"`` or ``"<uuid>:laptop"``).
    Non-admins are always scoped to their own namespace; admins may pass
    ``parent_id`` to count another user's entity. Bootstrap admin without
    ``parent_id`` returns a global count across all users.
    """
    operator, is_admin = resolve_operator(request, auth, db)

    # Bootstrap admin without parent_id: global count across all users.
    if is_bootstrap_admin(operator.id) and not parent_id:
        count = count_memories_for_entity(entity_type, entity_id, parent_entity_id=None)
        return {"total_memories": count}

    # Resolve the parent entity id for scoped agent/run lookup.
    if is_admin and parent_id is not None:
        resolved_parent = parent_id
    else:
        resolved_parent = parent_id or str(operator.id)

    entity = get_entity_or_none(
        entity_type,
        entity_id,
        db,
        parent_entity_id=resolved_parent,
    )
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_type}/{entity_id}' not found.")
    # For agent/run: resolve the parent namespace to scope both the permission
    # check and the vector store query (agent/run are unique per parent, not globally).
    parent_entity_id = get_parent_entity_id(entity, db)
    if not check_entity_permission(
        entity_type,
        entity_id,
        operator.id,
        "read",
        db,
        bypass=is_admin,
        parent_entity_id=parent_entity_id,
    ):
        raise HTTPException(status_code=403, detail="You do not have read permission for this entity.")
    count = count_memories_for_entity(entity_type, entity_id, parent_entity_id=parent_entity_id)
    return {"total_memories": count}


@router.get("/{entity_type}/{entity_id}/permissions", response_model=list[EntityPermissionResponse])
def list_permissions(
    entity_type: EntityType,
    entity_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, is_admin = resolve_operator(request, auth, db)
    perms = list_entity_permissions(
        entity_type,
        entity_id,
        operator_id=operator.id,
        bypass=is_admin,
        db=db,
    )
    return _permissions_to_response_batch(perms, db)


@router.post("/{entity_type}/{entity_id}/permissions", response_model=EntityPermissionResponse)
def grant_permission(
    entity_type: EntityType,
    entity_id: str,
    body: GrantPermissionInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, is_admin = resolve_operator(request, auth, db)
    grantee_id = _parse_uuid(body.grantee_id, label="grantee_id")
    perm = grant_entity_permission(
        entity_type,
        entity_id,
        grantee_id,
        body.permission,
        operator_id=operator.id,
        bypass=is_admin,
        db=db,
    )
    return _permission_to_response(perm, db)


@router.delete("/{entity_type}/{entity_id}/permissions/{grantee_id}", response_model=MessageResponse)
def revoke_permission(
    entity_type: EntityType,
    entity_id: str,
    grantee_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, is_admin = resolve_operator(request, auth, db)
    grantee_id = _parse_uuid(grantee_id, label="grantee_id")
    revoke_entity_permission(
        entity_type,
        entity_id,
        grantee_id,
        operator_id=operator.id,
        bypass=is_admin,
        db=db,
    )
    return MessageResponse(message="Permission revoked")


# --------------------------------------------------------------------------- #
# Transfer / delete
# --------------------------------------------------------------------------- #
@router.post("/{entity_type}/{entity_id}/transfer-owner", response_model=EntityResponse)
def transfer_owner_endpoint(
    entity_type: EntityType,
    entity_id: str,
    body: TransferOwnerInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, is_admin = resolve_operator(request, auth, db)
    new_owner_id = _parse_uuid(body.owner_id, label="owner_id")
    entity = transfer_entity_owner(
        entity_type,
        entity_id,
        new_owner_id,
        operator_id=operator.id,
        bypass=is_admin,
        db=db,
    )
    return _entity_to_response(entity, operator.id, db, bypass=is_admin)


@router.delete("/{entity_type}/{entity_id}", response_model=MessageResponse)
def delete_entity(
    entity_type: EntityType,
    entity_id: str,
    request: Request,
    parent_id: str | None = None,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Delete an entity namespace: prescan matched memories, verify owner permission,
    clear the vector store, then drop the DB row (cascades entity_permissions).

    For agent/run entities (unique per parent user), ``parent_id`` scopes the
    lookup to a specific user namespace (e.g. ``"<uuid>"`` or ``"<uuid>:laptop"``).
    Non-admins are always scoped to their own namespace; admins may pass
    ``parent_id`` to delete another user's entity. Bootstrap admin without
    ``parent_id`` matches any entity of that type/name.
    """
    operator, is_admin = resolve_operator(request, auth, db)

    if is_admin and parent_id is not None:
        resolved_parent = parent_id
    elif is_bootstrap_admin(operator.id):
        resolved_parent = parent_id  # None → unscoped, or explicit scope
    else:
        resolved_parent = parent_id or str(operator.id)

    entity = check_entity_delete_permission(
        entity_type,
        entity_id,
        operator.id,
        is_admin,
        db,
        parent_entity_id=resolved_parent,
    )

    # For user entities: guard against child namespaces, cascade-delete for admins.
    if entity_type == "user":
        children = collect_user_children(entity, db)
        if children and not is_admin:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete 'user/{entity_id}' because it has sub-namespaces or agent/run entities. Delete them first.",
            )
        # Admin: cascade-delete child memories + entity rows. agent/run children
        # must be scoped by their parent entity_id so a same-named agent/run owned
        # by another user is not swept into the deletion.
        for child in children:
            child_params: dict[str, Any] = entity_filter_params(child, db)
            try:
                run_memory_write(
                    lambda m, dp=child_params: m.delete_all(**dp),
                    child_params,
                )
            except HTTPException:
                raise
            except Exception:
                logger.exception("delete_entity: child vector-store delete failed for %s/%s", child.type, child.id)
                raise upstream_error()
            db.delete(child)

    # For agent/run: scope the vector-store scan by parent entity_id to avoid
    # matching other users' memories with the same agent/run id.
    delete_params: dict[str, Any] = entity_filter_params(entity, db)

    def _delete_all(memory):
        memory_ids = list_memory_ids_for_params(delete_params)
        validate_bulk_admin_operation(memory_ids, operator.id, db, bypass=is_admin)
        memory.delete_all(**delete_params)

    try:
        run_memory_write(_delete_all, delete_params)
    except HTTPException:
        raise
    except Exception:
        logger.exception("delete_entity: vector-store delete failed for %s/%s", entity_type, entity_id)
        raise upstream_error()

    # NOTE: vector-store deletions above are irreversible. If db.commit() fails
    # (e.g. connection lost), memories are already gone but entity rows survive
    # in the DB — the namespace is occupied without data. The scope lock
    # acquired by run_memory_write prevents concurrent writes during the
    # deletion, but the two storage backends (vector store + RDB) are not
    # transactional with each other.
    db.delete(entity)
    db.commit()
    return MessageResponse(message="Entity deleted")


@router.patch("/{entity_type}/{entity_id}", response_model=EntityResponse)
def update_entity(
    entity_type: EntityType,
    entity_id: str,
    body: UpdateEntityInput,
    request: Request,
    parent_id: str | None = None,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Update an entity's display name. Only the owner or an admin may edit.

    For agent/run entities, ``parent_id`` scopes the lookup to a specific
    user namespace (same as delete/count).
    """
    operator, is_admin = resolve_operator(request, auth, db)

    # Resolve parent namespace for agent/run (mirror count_entity_memories/delete).
    if is_admin and parent_id is not None:
        resolved_parent = parent_id
    elif is_bootstrap_admin(operator.id):
        resolved_parent = parent_id
    else:
        resolved_parent = parent_id or str(operator.id)

    entity = get_entity_or_none(entity_type, entity_id, db, parent_entity_id=resolved_parent)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_type}/{entity_id}' not found.")

    # Only the owner or a system admin may edit.
    if entity.owner_id != operator.id and not is_admin:
        raise HTTPException(status_code=403, detail="Only the owner or an admin can edit this entity.")

    if "name" in body.model_fields_set:
        entity.name = (body.name or "").strip() or None
    db.commit()
    return _entity_to_response(entity, operator.id, db, bypass=is_admin)
