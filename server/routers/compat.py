"""Client-compatible versioned API endpoints.

These routes expose the versioned paths used by ``MemoryClient`` and align the
self-hosted server as closely as practical with ``docs/openapi.json``.

Covered endpoints
-----------------
    GET    /v1/ping

    GET    /v1/memories
    POST   /v1/memories
    DELETE /v1/memories
    GET    /v1/memories/{entity_type}/{entity_id}
    GET    /v1/memories/{memory_id}
    PUT    /v1/memories/{memory_id}
    DELETE /v1/memories/{memory_id}
    GET    /v1/memories/{memory_id}/history
    POST   /v1/memories/search

    POST   /v2/memories
    POST   /v2/memories/search

    POST   /v3/memories
    POST   /v3/memories/add
    POST   /v3/memories/search

    GET    /v1/entities
    GET    /v1/entities/filters
    GET    /v2/entities/{entity_type}/{entity_id}
    DELETE /v2/entities/{entity_type}/{entity_id}

    GET    /v1/events
    GET    /v1/event/{event_id}
    GET    /v1/memories/events

    PUT    /v1/batch
    DELETE /v1/batch

Stub endpoints (501 Not Implemented)
-------------------------------------
    GET    /api/v1/orgs/organizations/{org_id}/projects/
    POST   /api/v1/orgs/organizations/{org_id}/projects/
    GET    /api/v1/orgs/organizations/{org_id}/projects/{project_id}/
    PATCH  /api/v1/orgs/organizations/{org_id}/projects/{project_id}/
    DELETE /api/v1/orgs/organizations/{org_id}/projects/{project_id}/
    GET    /api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/
    POST   /api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/
    PUT    /api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/
    DELETE /api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/
"""

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auth import is_bootstrap_admin, verify_auth
from compat.decorators import upstream_guard
from compat.requests import RequestMeta, request_meta
from compat.entities import CompatEntity
from compat.helpers import (
    build_search_kwargs,
    normalize_results,
    normalize_results_dict,
    paginated_get_all,
    resolve_existing,
)
from compat.metadata import (
    build_extraction_prompt,
    build_v3_add_extra_metadata,
    merge_v1_add_metadata,
    merge_v3_add_metadata,
    merge_update_metadata,
)
from compat.utils import drop_none
from compat.responses import (
    apply_fields,
    warn_ignored_compat_params,
    pending_add_response,
    sync_add_response,
    unsupported_api_error,
)
from utils.pagination import paginate_response
from compat.events import (
    create_pending_add_event,
    event_access_allowed,
    event_cache_all,
    event_cache_get,
    events_visible_to_caller,
    resolve_event_owner_id,
)
from compat.scope import (
    append_search_convenience_filters,
    build_list_filters,
    build_search_filters,
    collect_direct_entity_params,
    get_entity_field,
    require_entity_scope,
)
from compat.tasks import run_v3_add_memory_task
from memory_lock import (
    entity_scope_from_params,
    run_memory_write,
    run_memory_write_for_memory_id,
)
from server_state import get_memory_instance
from sqlalchemy.orm import Session
from models import Entity
from db import get_db
from entity import VALID_ENTITY_TYPES, canonicalize_entity_id, is_scoped_entity_type
from entity_permissions import (
    authorize_write,
    bulk_delete_memories,
    check_entity_delete_permission,
    check_memory_scope_permission,
    check_query_permission,
    collect_user_children,
    strip_user_id_for_app_gate,
    get_visible_entities,
    inject_default_user_id,
    list_memory_ids_for_params,
    reject_bootstrap_memory_mutation,
    resolve_memory_entities,
    resolve_operator,
    validate_bulk_admin_operation,
)

logger = logging.getLogger("mem0.server.compat")

router = APIRouter(tags=["Client API"])


class MemoryAddInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]] = Field(
        description="An array of message objects representing the content of the memory. "
        "Each message object typically contains 'role' and 'content' fields, where 'role' "
        "indicates the sender ('user' or 'assistant') and 'content' contains the actual message text. "
        "This structure allows for the representation of conversations or multi-part memories."
    )
    user_id: Optional[str] = Field(
        default=None, description="The unique identifier of the user associated with this memory."
    )
    agent_id: Optional[str] = Field(
        default=None, description="The unique identifier of the agent associated with this memory."
    )
    app_id: Optional[str] = Field(
        default=None,
        description="The unique identifier of the application.",
    )
    run_id: Optional[str] = Field(
        default=None, description="The unique identifier of the run associated with this memory."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with the memory. Best practice for incorporating additional "
        "information is through metadata (e.g. location, time, ids, etc.). During retrieval, you can either use "
        "these metadata alongside the query to fetch relevant memories or retrieve memories based on the query "
        "first and then refine the results using metadata during post-processing.",
    )
    infer: Optional[bool] = Field(
        default=None, description="Whether to infer the memories or directly store the messages."
    )
    categories: Optional[List[str]] = Field(default=None, description="A list of categories to tag the memory with.")


class MemorySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The query to search for in the memory.")
    user_id: Optional[str] = Field(default=None, description="The user ID associated with the memory.")
    agent_id: Optional[str] = Field(default=None, description="The agent ID associated with the memory.")
    app_id: Optional[str] = Field(
        default=None,
        description="The app ID associated with the memory.",
    )
    run_id: Optional[str] = Field(default=None, description="The run ID associated with the memory.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter results to memories matching this metadata."
    )
    top_k: Optional[int] = Field(default=None, description="The number of top results to return.")
    threshold: Optional[float] = Field(
        default=None, description="The minimum similarity threshold for returned results."
    )
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank the memories.")
    fields: Optional[List[str]] = Field(default=None, description="Restrict the fields returned per memory.")
    show_expired: Optional[bool] = Field(
        default=None,
        description="When true, include memories whose `expiration_date` has passed. Expired memories are hidden by default.",
    )
    latest_only: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; not processed by the self-hosted server.",
    )


class MemoryUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Optional[str] = Field(default=None, description="The updated text content of the memory.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata associated with the memory."
    )
    timestamp: Optional[int] = Field(
        default=None, description="Unix epoch seconds to backdate created_at on the stored memory."
    )
    expiration_date: Optional[date] = Field(
        default=None,
        description="Expiration date in YYYY-MM-DD format, or null to clear.",
    )


class MemoryBatchUpdateItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: str = Field(description="ID of the memory to update.")
    text: Optional[str] = Field(default=None, description="New text content.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata.")


class MemoryBatchUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memories: List[MemoryBatchUpdateItem] = Field(description="List of memories to update.")


class MemoryBatchDeleteItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: str = Field(description="ID of the memory to delete.")


class MemoryBatchDeleteLegacyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memories: List[MemoryBatchDeleteItem] = Field(description="List of memories to delete (legacy format).")

    @model_validator(mode="after")
    def _validate_memories_not_empty(self) -> "MemoryBatchDeleteLegacyInput":
        if not self.memories:
            raise ValueError("'memories' must not be empty.")
        return self


class MemoryBatchDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_ids: List[str] = Field(description="List of memory IDs to delete.")

    @model_validator(mode="after")
    def _validate_memory_ids_not_empty(self) -> "MemoryBatchDeleteInput":
        if not self.memory_ids:
            raise ValueError("'memory_ids' must not be empty.")
        return self


class MemoryGetInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary of filters to apply to retrieve memories. Available fields are: "
        "user_id, agent_id, app_id, run_id, created_at, updated_at, categories, keywords. "
        "Supports logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains, *). "
        "For categories field, use 'contains' for partial matching "
        '(e.g., {"categories": {"contains": "finance"}}) or \'in\' for exact matching '
        '(e.g., {"categories": {"in": ["personal_information"]}}).',
    )
    start_date: Optional[str] = Field(
        default=None, description="Only return memories created on or after this ISO 8601 date."
    )
    end_date: Optional[str] = Field(
        default=None, description="Only return memories created on or before this ISO 8601 date."
    )
    categories: Optional[List[str]] = Field(
        default=None, description="Filter results to memories tagged with any of these categories."
    )
    show_expired: Optional[bool] = Field(
        default=None,
        description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
    )
    fields: Optional[List[str]] = Field(default=None, description="Restrict the fields returned per memory.")
    latest_only: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; not processed by the self-hosted server.",
    )


class MemoryGetInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Entity and metadata filters. Must include at least one entity ID "
        "(`user_id`, `agent_id`, `app_id` or `run_id`). Supports `AND`, `OR`, `NOT`, and "
        "comparison operators (`in`, `gte`, `lte`, `gt`, `lt`, `contains`, `icontains`, `ne`).",
    )
    start_date: Optional[str] = Field(
        default=None, description="Only return memories created on or after this ISO 8601 date."
    )
    end_date: Optional[str] = Field(
        default=None, description="Only return memories created on or before this ISO 8601 date."
    )
    categories: Optional[List[str]] = Field(
        default=None, description="Filter results to memories tagged with any of these categories."
    )
    show_expired: Optional[bool] = Field(
        default=None,
        description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
    )
    latest_only: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; not processed by the self-hosted server.",
    )


class MemorySearchInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The query to search for in the memory.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary of filters to apply to the search. Available fields are: "
        "user_id, agent_id, app_id, run_id, created_at, updated_at, categories, keywords. "
        "Supports logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains). "
        "For categories field, use 'contains' for partial matching "
        '(e.g., {"categories": {"contains": "finance"}}) or \'in\' for exact matching '
        '(e.g., {"categories": {"in": ["personal_information"]}}).',
    )
    top_k: Optional[int] = Field(default=None, description="The number of top results to return.")
    threshold: Optional[float] = Field(
        default=None, description="The minimum similarity threshold for returned results."
    )
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank the memories.")
    user_id: Optional[str] = Field(
        default=None, description="The user ID associated with the memory (also accepted inside filters)."
    )
    agent_id: Optional[str] = Field(
        default=None, description="The agent ID associated with the memory (also accepted inside filters)."
    )
    app_id: Optional[str] = Field(
        default=None, description="The app ID associated with the memory (also accepted inside filters)."
    )
    run_id: Optional[str] = Field(
        default=None, description="The run ID associated with the memory (also accepted inside filters)."
    )
    fields: Optional[List[str]] = Field(default=None, description="Restrict the fields returned per memory.")
    show_expired: Optional[bool] = Field(
        default=None,
        description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
    )
    latest_only: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; not processed by the self-hosted server.",
    )


class MemoryAddInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]] = Field(
        description="Conversation messages to extract memories from. "
        "Each object must have 'role' ('user', 'assistant', or 'system') and 'content' keys."
    )
    user_id: Optional[str] = Field(default=None, description="Scope memories to this user.")
    agent_id: Optional[str] = Field(default=None, description="Scope memories to this agent.")
    app_id: Optional[str] = Field(default=None, description="Scope memories to this app / project.")
    run_id: Optional[str] = Field(default=None, description="Scope memories to this session / run.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="User-supplied metadata to attach to each extracted memory."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Filters containing entity IDs (e.g. {'user_id': '...'})."
    )
    infer: Optional[bool] = Field(
        default=None,
        description=(
            "When `false`, stores each message verbatim without the extraction LLM and returns "
            "`results` synchronously. When `true` or omitted, runs extraction asynchronously and "
            "returns `event_id` + `status: PENDING`."
        ),
    )
    custom_categories: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Category catalog for this call. Stored in memory metadata.",
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Project-level instructions that guide extraction for this call."
    )
    agent_custom_instructions: Optional[str] = Field(
        default=None,
        description="Extraction instructions for agent-scoped memories. Takes precedence over `custom_instructions` for this call when `agent_id` is present and the value is non-empty. No project-level persistence (self-hosted has no project settings API).",
    )
    includes: Optional[str] = Field(
        default=None,
        description="Free-text hint of what to include during extraction, e.g. 'vehicles'.",
    )
    excludes: Optional[str] = Field(
        default=None,
        description="Free-text hint of what to exclude during extraction, e.g. 'politics'.",
    )
    structured_data_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Schema for structured data extraction. Not supported on the self-hosted server; accepted for wire compatibility.",
    )
    timestamp: Optional[int] = Field(
        default=None, description="Unix epoch seconds used to backdate created_at on the stored memories."
    )
    source: Optional[str] = Field(
        default=None, description="Source identifier for the memory (e.g. 'OPENCLAW'). Stored in metadata."
    )
    deduced_memories: Optional[List[str]] = Field(
        default=None,
        description="Pre-extracted facts stored individually as memories when infer=False; ignored otherwise.",
    )
    expiration_date: Optional[date] = Field(
        default=None,
        description="Optional expiration date in YYYY-MM-DD format. After this date, memories are hidden from search and get-all unless `show_expired` is true.",
    )


class MemorySearchInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Natural-language search query.")
    user_id: Optional[str] = Field(default=None, description="The user ID associated with the memory.")
    agent_id: Optional[str] = Field(default=None, description="The agent ID associated with the memory.")
    app_id: Optional[str] = Field(default=None, description="The app ID associated with the memory.")
    run_id: Optional[str] = Field(default=None, description="The run ID associated with the memory.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Entity and metadata filters. Must include at least one entity ID "
        "(`user_id`, `agent_id`, `app_id` or `run_id`). Supports `AND`, `OR`, `NOT`, and "
        "comparison operators (`in`, `gte`, `lte`, `gt`, `lt`, `contains`, `icontains`, `ne`).",
    )
    top_k: Optional[int] = Field(default=None, description="Number of results to return.")
    threshold: Optional[float] = Field(
        default=None, description="Minimum semantic relevance score. Pass `0.0` to disable filtering."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter results to memories matching this metadata."
    )
    rerank: Optional[bool] = Field(
        default=None, description="Apply the managed reranker for better ordering (adds latency)."
    )
    fields: Optional[List[str]] = Field(default=None, description="Restrict the fields returned per memory.")
    categories: Optional[List[str]] = Field(
        default=None, description="Filter results to memories tagged with any of these categories."
    )
    output_format: Optional[str] = Field(
        default=None,
        description='Response format. `v1.1` (default) returns `{"results": [...]}`. '
        "`v1.0` returns a flat array `[{...}]` for backwards compatibility.",
    )
    show_expired: Optional[bool] = Field(
        default=None,
        description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
    )
    reference_date: Optional[Any] = Field(
        default=None,
        description="Date and time to simulate the search from. Accepts a Unix epoch, YYYY-MM-DD, or ISO datetime.",
    )
    latest_only: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; not processed by the self-hosted server.",
    )
    keyword_search: Optional[bool] = Field(
        default=None,
        description="Accepted for compatibility; the self-hosted server already runs hybrid (semantic + keyword) search.",
    )


@router.get("/v1/ping/", include_in_schema=False)
@router.get("/v1/ping", summary="Ping / validate API key")
def ping(_auth=Depends(verify_auth)):
    """Used by ``MemoryClient`` to validate the API key on initialisation.

    Returns ``org_id`` and ``project_id`` so that ``MemoryClient.project``
    initialises without raising ``ValueError``.
    """
    user_email = getattr(_auth, "email", None) if _auth else None
    return {
        "status": "ok",
        "message": "pong",
        "user_email": user_email,
        "org_id": "local",
        "project_id": "default",
    }


@router.get("/v1/memories/", include_in_schema=False)
@router.get("/v1/memories", summary="Get all memories (v1)")
@upstream_guard
def v1_list_memories(
    request: Request,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    show_expired: Optional[bool] = Query(default=None),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """List memories (v1 compat).

    When both ``user_id`` and ``app_id`` are provided, ``user_id`` is stripped
    and only the app permission is checked (app-primary-gate). See
    ``strip_user_id_for_app_gate``."""
    operator, _ = resolve_operator(request, auth, db)
    show_expired_flag = show_expired if isinstance(show_expired, bool) else None
    filters = drop_none({"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id})

    if not filters:
        if is_bootstrap_admin(operator):
            raise HTTPException(
                status_code=400,
                detail="admin_api_key requires an explicit scope (user_id, agent_id, app_id, or run_id).",
            )
        filters = {"user_id": str(operator.id)}
    filters = strip_user_id_for_app_gate(filters)
    filters = check_query_permission(filters, operator.id, db)
    kwargs: Dict[str, Any] = {"filters": filters}
    if show_expired_flag is not None:
        kwargs["show_expired"] = show_expired_flag
    raw = get_memory_instance().get_all(**kwargs)
    return normalize_results(raw)


@router.post("/v1/memories/", include_in_schema=False)
@router.post("/v1/memories", summary="Add memories (v1)")
@upstream_guard
def v1_add_memories(
    body: MemoryAddInput,
    request: Request,
    meta: RequestMeta = Depends(request_meta),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    entity_params = collect_direct_entity_params(
        user_id=body.user_id,
        agent_id=body.agent_id,
        app_id=body.app_id,
        run_id=body.run_id,
    )
    entity_params = inject_default_user_id(entity_params, operator)
    if not entity_params:
        raise HTTPException(
            status_code=400,
            detail="One of the filters: user_id, agent_id, app_id or run_id is required!",
        )
    authorize_write(entity_params, operator, db)
    params = drop_none({**entity_params, "metadata": body.metadata})
    if body.infer is not None:
        params["infer"] = body.infer

    params["metadata"] = merge_v1_add_metadata(
        params.get("metadata"),
        source=meta.source,
        platform=meta.platform,
        categories=body.categories,
    )

    raw = run_memory_write(lambda memory: memory.add(messages=body.messages, **params), entity_params)
    return normalize_results(raw)


@router.get("/v1/memories/{memory_id}/", include_in_schema=False)
@router.get("/v1/memories/{memory_id}", summary="Get a memory (v1)")
@upstream_guard
def v1_get_memory(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "read", db)
    return resolve_existing(get_memory_instance(), memory_id)


@router.put("/v1/memories/{memory_id}", include_in_schema=False)
@router.put("/v1/memories/{memory_id}/", summary="Update a memory (v1)")
@upstream_guard
def v1_update_memory(
    memory_id: str,
    body: MemoryUpdateInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    has_expiration_update = "expiration_date" in body.model_fields_set
    if body.text is None and body.metadata is None and body.timestamp is None and not has_expiration_update:
        raise HTTPException(
            status_code=400,
            detail="At least one of text, metadata, timestamp, or expiration_date must be provided for update.",
        )
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "write", db)
    # Forward only the fields the caller explicitly set (mirrors main.update_memory).
    params: Dict[str, Any] = {"memory_id": memory_id}
    if body.text is not None:
        params["data"] = body.text
    try:
        metadata = merge_update_metadata(body.metadata, body.timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if metadata is not None:
        params["metadata"] = metadata
    if has_expiration_update:
        params["expiration_date"] = body.expiration_date
    try:
        return run_memory_write_for_memory_id(
            lambda memory: memory.update(**params),
            memory_id,
        )
    except ValueError as e:
        # "not found" → 404 (matches main._client_error and the prior
        # resolve_existing behaviour); any other ValueError falls through to
        # @upstream_guard, which maps it to 400.
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
        raise


@router.delete("/v1/memories/{memory_id}/", include_in_schema=False)
@router.delete("/v1/memories/{memory_id}", summary="Delete a memory (v1)")
@upstream_guard
def v1_delete_memory(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "write", db)
    try:
        result = run_memory_write_for_memory_id(lambda memory: memory.delete(memory_id=memory_id), memory_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    return result


@router.get("/v1/memories/{memory_id}/history/", include_in_schema=False)
@router.get("/v1/memories/{memory_id}/history", summary="Get memory history (v1)")
@upstream_guard
def v1_memory_history(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "read", db)
    raw = get_memory_instance().history(memory_id=memory_id)
    return normalize_results(raw)


@router.get("/v1/memories/{entity_type}/{entity_id}/", include_in_schema=False)
@router.get("/v1/memories/{entity_type}/{entity_id}", summary="Get memories for an entity (v1)")
@upstream_guard
def v1_get_entity_memories(
    entity_type: str,
    entity_id: str,
    request: Request,
    show_expired: Optional[bool] = Query(default=None),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Get memories for an entity (v1 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present in
    filters, ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    field = get_entity_field(entity_type)  # validates entity_type (400 on invalid)
    filters: Dict[str, Any] = {field: entity_id}
    filters = strip_user_id_for_app_gate(filters)
    filters = check_query_permission(filters, operator.id, db)
    kwargs: Dict[str, Any] = {"filters": filters}
    if isinstance(show_expired, bool):
        kwargs["show_expired"] = show_expired
    raw = get_memory_instance().get_all(**kwargs)
    return normalize_results(raw)


@router.post("/v1/memories/search/", include_in_schema=False)
@router.post("/v1/memories/search", summary="Search memories (v1)")
@upstream_guard
def v1_search_memories(
    body: MemorySearchInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Search memories (v1 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present,
    ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    warn_ignored_compat_params("v1_search_memories", latest_only=body.latest_only)
    effective_filters = build_search_filters(
        user_id=body.user_id,
        agent_id=body.agent_id,
        app_id=body.app_id,
        run_id=body.run_id,
        detail="At least one of the filters: agent_id, user_id, app_id or run_id is required!",
    )
    effective_filters = append_search_convenience_filters(effective_filters, metadata=body.metadata)
    effective_filters = strip_user_id_for_app_gate(effective_filters)
    effective_filters = check_query_permission(effective_filters, operator.id, db)

    raw = get_memory_instance().search(
        query=body.query,
        **build_search_kwargs(effective_filters, body.top_k, body.threshold, body.rerank, body.show_expired),
    )
    return apply_fields(normalize_results(raw), body.fields)


@router.delete("/v1/memories/", include_in_schema=False)
@router.delete("/v1/memories", summary="Delete all memories (v1)")
@upstream_guard
def v1_delete_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[str] = None,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    # ``filters`` is a legacy query-string JSON blob (not the structured dict
    # used in v2/v3 body endpoints). Only parse it when no explicit entity
    # params are given, to avoid silently overriding explicit args.
    if filters and not any((user_id, agent_id, app_id, run_id)):
        try:
            filters_dict = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in 'filters' query parameter.")
        if not isinstance(filters_dict, dict):
            raise HTTPException(status_code=400, detail="'filters' query parameter must be a JSON object.")
        user_id = user_id or filters_dict.get("user_id")
        agent_id = agent_id or filters_dict.get("agent_id")
        app_id = app_id or filters_dict.get("app_id")
        run_id = run_id or filters_dict.get("run_id")

    params = drop_none({"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id})
    if not params:
        raise HTTPException(
            status_code=400,
            detail="One of the filters: user_id, agent_id, app_id or run_id is required!",
        )
    # Scope agent/run/app-only deletes to the caller's user namespace (mirrors the
    # write path's inject_default_user_id). Without this, delete_all(agent_id=riley)
    # matches every user's same-named agent and the prescan fails the whole batch
    # (403) — so a user could write but not delete their own agent memories.
    params = inject_default_user_id(params, operator, skip_if_has_app=True)
    return run_memory_write(
        lambda m: bulk_delete_memories(m, params, operator.id, db),
        entity_scope_from_params(params),
    )


@router.get("/v1/events/", include_in_schema=False)
@router.get("/v1/events", summary="List events (v1)")
def v1_list_events(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    auth=Depends(verify_auth),
):
    """Retrieve all events for the current project.

    Returns events from the in-memory TTL cache populated by add operations.
    """
    caller_id = resolve_event_owner_id(auth)
    user_events = events_visible_to_caller(event_cache_all(), caller_id)
    return paginate_response(request, user_events, page, page_size)


@router.get("/v1/event/{event_id}/", include_in_schema=False)
@router.get("/v1/event/{event_id}", summary="Get event details (v1)")
def v1_get_event(event_id: str, auth=Depends(verify_auth)):
    """Retrieve details of a specific event by its ID.

    Returns the SUCCEEDED event object from the in-memory TTL cache.
    """
    obj = event_cache_get(event_id)
    caller_id = resolve_event_owner_id(auth)
    if obj is None or not event_access_allowed(obj, caller_id):
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
    return obj


@router.get("/v1/memories/events/", include_in_schema=False)
@router.get("/v1/memories/events", summary="List memory events (v1)")
def v1_list_memory_events(_auth=Depends(verify_auth)):
    """Retrieve memory-level events.

    Memory-level change events (create/update/delete per memory) are not tracked
    by the self-hosted server. Returns an empty paginated response.
    """
    return {"count": 0, "next": None, "previous": None, "results": []}


@router.put("/v1/batch/", include_in_schema=False)
@router.put("/v1/batch", summary="Batch update memories (v1)")
@upstream_guard
def v1_batch_update(
    body: MemoryBatchUpdateInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    # Validate items: each must have at least text or metadata.
    invalid: List[str] = [item.memory_id for item in body.memories if item.text is None and item.metadata is None]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Items missing both 'text' and 'metadata': {invalid}")
    # Cap updates per request: each update still issues vector-store reads
    # internally (lock scope resolution + _update_memory), so bound the work.
    if len(body.memories) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Too many updates ({len(body.memories)}). Maximum is 100 per request.",
        )
    # Pre-validate write permission on ALL items before updating any (all-or-nothing;
    # an unauthorized item fails the whole request instead of being silently skipped).
    for item in body.memories:
        scope = resolve_memory_entities(item.memory_id)
        check_memory_scope_permission(scope, operator.id, "write", db)

    updated_count = 0
    for item in body.memories:
        try:
            run_memory_write_for_memory_id(
                lambda memory, it=item: memory.update(memory_id=it.memory_id, data=it.text, metadata=it.metadata),
                item.memory_id,
            )
            updated_count += 1
        except ValueError:
            # Memory vanished between pre-validation and update (concurrent
            # delete); not a permission issue, so skip rather than fail.
            continue
    return {"message": f"Memories updated successfully, count: {updated_count}."}


@router.delete("/v1/batch/", include_in_schema=False)
@router.delete("/v1/batch", summary="Batch delete memories (v1)")
@upstream_guard
def v1_batch_delete(
    body: MemoryBatchDeleteLegacyInput | MemoryBatchDeleteInput,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    memory_ids = (
        body.memory_ids if isinstance(body, MemoryBatchDeleteInput) else [item.memory_id for item in body.memories]
    )
    if len(memory_ids) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be deleted in a single request")
    # Pre-validate admin permission on ALL memory_ids before deleting any: an
    # unauthorized item fails the whole request (all-or-nothing) instead of being
    # silently skipped. resolve_memory_entities raises 404 for a missing memory;
    # check_memory_scope_permission raises 403 for insufficient permission.
    for memory_id in memory_ids:
        scope = resolve_memory_entities(memory_id)
        check_memory_scope_permission(scope, operator.id, "write", db)

    deleted_count = 0
    for memory_id in memory_ids:
        try:
            run_memory_write_for_memory_id(
                lambda memory, mid=memory_id: memory.delete(memory_id=mid),
                memory_id,
            )
            deleted_count += 1
        except ValueError:
            # Memory vanished between pre-validation and delete (concurrent
            # delete); not a permission issue, so skip rather than fail.
            continue
    return {"message": f"Memories deleted successfully, count: {deleted_count}."}


@router.get("/v1/entities/", include_in_schema=False)
@router.get("/v1/entities", summary="List entities (v1)")
@upstream_guard
def v1_list_entities(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Return entities in the SDK-compatible envelope while preserving spec fields.

    Visibility is filtered by ownership/grants (get_visible_entities); unclaimed
    namespaces are not visible to non-admins. The hosted spec documents an array
    response, but ``MemoryClient.users()`` and ``delete_users()`` read
    ``response["results"]``. Keep that envelope here and include the spec fields
    on each entity item.
    """
    operator, _ = resolve_operator(request, auth, db)
    entities = get_visible_entities(operator.id, db)
    all_results = [
        CompatEntity.from_bucket(
            ent.type,
            ent.id,
            created_at=ent.created_at,
            updated_at=ent.updated_at,
            entity_name=ent.name,
        )
        for ent in sorted(entities, key=lambda e: (e.type, e.id))
    ]
    return paginate_response(request, all_results, page, page_size)


@router.get("/v1/entities/filters/", include_in_schema=False)
@router.get("/v1/entities/filters", summary="List supported entity filters (v1)")
def v1_list_entity_filters(_auth=Depends(verify_auth)):
    return {"results": sorted(VALID_ENTITY_TYPES)}


@router.post("/v2/memories/", include_in_schema=False)
@router.post("/v2/memories", summary="Get all memories (v2)")
@upstream_guard
def v2_list_memories(
    request: Request,
    body: MemoryGetInputV2,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """List memories (v2 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present in
    filters, ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    warn_ignored_compat_params("v2_list_memories", latest_only=body.latest_only)
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="One of the filters: user_id, agent_id, app_id or run_id is required!",
    )
    filters = build_list_filters(body, entity_params)
    filters = strip_user_id_for_app_gate(filters)
    filters = check_query_permission(filters, operator.id, db)
    kwargs: Dict[str, Any] = {"filters": filters}
    if body.show_expired is not None:
        kwargs["show_expired"] = body.show_expired
    response = paginated_get_all(request, page, page_size, **kwargs)
    response["results"] = apply_fields(response["results"], body.fields)
    return response


@router.post("/v2/memories/search/", include_in_schema=False)
@router.post("/v2/memories/search", summary="Search memories (v2)")
@upstream_guard
def v2_search_memories(
    body: MemorySearchInputV2,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Search memories (v2 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present,
    ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    warn_ignored_compat_params("v2_search_memories", latest_only=body.latest_only)
    effective_filters = build_search_filters(
        user_id=body.user_id,
        agent_id=body.agent_id,
        app_id=body.app_id,
        run_id=body.run_id,
        filters=body.filters,
        detail="At least one of the filters: agent_id, user_id, app_id or run_id is required!",
    )
    effective_filters = strip_user_id_for_app_gate(effective_filters)
    effective_filters = check_query_permission(effective_filters, operator.id, db)
    raw = get_memory_instance().search(
        query=body.query,
        **build_search_kwargs(effective_filters, body.top_k, body.threshold, body.rerank, body.show_expired),
    )
    # NOTE: docs/openapi.json declares a bare array response, but MemoryClient
    # reads response["results"]. We intentionally return the envelope here.
    response = normalize_results_dict(raw)
    response["results"] = apply_fields(response["results"], body.fields)
    return response


@router.get("/v2/entities/{entity_type}/{entity_id}/", include_in_schema=False)
@router.get("/v2/entities/{entity_type}/{entity_id}", summary="Get entity details (v2)")
@upstream_guard
def v2_get_entity(
    entity_type: str,
    entity_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    get_entity_field(entity_type)  # validate entity_type early
    eid = canonicalize_entity_id(entity_type, entity_id)
    for ent in get_visible_entities(operator.id, db):
        if ent.type == entity_type and ent.id == eid:
            return CompatEntity.from_bucket(
                ent.type,
                ent.id,
                created_at=ent.created_at,
                updated_at=ent.updated_at,
                entity_name=ent.name,
            )
    raise HTTPException(status_code=404, detail=f"Entity '{entity_type}/{entity_id}' not found.")


@router.delete("/v1/entities/{entity_type}/{entity_id}/", include_in_schema=False, status_code=204)
@router.delete("/v2/entities/{entity_type}/{entity_id}/", include_in_schema=False, status_code=204)
@router.delete("/v2/entities/{entity_type}/{entity_id}", summary="Delete entity (v2)", status_code=204)
@upstream_guard
def v2_delete_entity(
    entity_type: str,
    entity_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    # Client zone: no bypass. All operations are scoped to the caller's own
    # namespace. Bootstrap admin cannot delete entities here (parent_entity_id
    # resolves to None → check_entity_delete_permission returns 403/404).
    field = get_entity_field(entity_type)  # validate entity_type early
    # Owner-only delete: a granted admin cannot delete entities (design: delete =
    # owner only). agent/run are unique per parent, not globally: scope
    # the by-name lookup to the caller.
    parent_entity_id = None if is_bootstrap_admin(operator.id) else str(operator.id)
    entity = check_entity_delete_permission(
        entity_type,
        entity_id,
        operator.id,
        False,
        db,
        parent_entity_id=parent_entity_id,
    )
    # Client zone: user entities with children cannot be deleted — the
    # caller must delete sub-namespaces and agent/run children first.
    # Cascade-delete is handled by the management zone (DELETE /entities/...).
    if entity is not None and entity_type == "user":
        if collect_user_children(entity, db):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete 'user/{entity_id}' because it has sub-namespaces or agent/run entities. Delete them first.",
            )
    # For agent/run: resolve parent entity id to scope the vector-store scan/delete
    # (otherwise a same-named agent/run owned by another user is matched).
    parent_entity_id = None
    if entity is not None and is_scoped_entity_type(entity_type) and entity.parent_pk is not None:
        parent_entity = db.get(Entity, entity.parent_pk)
        if parent_entity is not None:
            parent_entity_id = parent_entity.id
    # Bulk destructive: prescan + admin-validate + delete inside the scope lock (TOCTOU).
    delete_params: dict[str, Any] = {field: entity_id}
    if parent_entity_id is not None:
        delete_params["user_id"] = parent_entity_id

    def _delete_all(memory):
        memory_ids = list_memory_ids_for_params(delete_params)
        validate_bulk_admin_operation(memory_ids, operator.id, db)
        memory.delete_all(**delete_params)

    try:
        run_memory_write(_delete_all, delete_params)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_type}/{entity_id}' not found.")
    # Drop the DB row so the namespace is released for re-claim (mirrors
    # DELETE /entities/{type}/{id}); cascade clears entity_permissions.
    if entity is not None:
        db.delete(entity)
        db.commit()


@router.post("/v3/memories/add/", include_in_schema=False)
@router.post("/v3/memories/add", summary="Add memory (v3)")
@upstream_guard
def v3_add_memory(
    body: MemoryAddInputV3,
    background_tasks: BackgroundTasks,
    request: Request,
    meta: RequestMeta = Depends(request_meta),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    operator, _ = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    entity_params = collect_direct_entity_params(
        filters=body.filters,
        user_id=body.user_id,
        agent_id=body.agent_id,
        app_id=body.app_id,
        run_id=body.run_id,
    )
    entity_params = inject_default_user_id(entity_params, operator)
    if not entity_params:
        raise HTTPException(
            status_code=400,
            detail="One of the filters: user_id, agent_id, app_id or run_id is required!",
        )
    # Permission + ownership must be committed synchronously before the async task
    # is dispatched.
    authorize_write(entity_params, operator, db)
    params: Dict[str, Any] = drop_none(
        {
            **entity_params,
            "metadata": body.metadata,
        }
    )
    extra_md = build_v3_add_extra_metadata(
        custom_categories=body.custom_categories,
        source=body.source,
    )

    try:
        params["metadata"] = merge_v3_add_metadata(
            params.get("metadata"),
            source=meta.source,
            platform=meta.platform,
            timestamp=body.timestamp,
            extra_metadata=extra_md,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if body.expiration_date is not None:
        params["expiration_date"] = body.expiration_date

    if body.structured_data_schema is not None:
        logger.warning("v3_add_memory: structured_data_schema unsupported on self-hosted server")

    extraction_prompt = build_extraction_prompt(
        custom_instructions=body.custom_instructions,
        agent_custom_instructions=body.agent_custom_instructions,
        includes=body.includes,
        excludes=body.excludes,
        has_agent_scope=entity_params.get("agent_id") is not None,
    )
    if extraction_prompt and body.infer is not False:
        params["prompt"] = extraction_prompt

    # infer=False + deduced_memories: store each fact as its own memory.
    effective_messages = body.messages
    if body.infer is False and body.deduced_memories:
        effective_messages = [{"role": "user", "content": fact} for fact in body.deduced_memories]

    # Hybrid write path (same semantics as MCP add_memory):
    #   infer=False — fast path: no LLM, sync response with memory ids.
    #   infer=True or omitted — slow path: extraction + dedup, async + event_id (hosted v3 default).
    if body.infer is False:
        params["infer"] = False
        raw = run_memory_write(
            lambda memory: memory.add(messages=effective_messages, **params),
            entity_params,
        )
        return sync_add_response(raw)
    elif body.infer is True:
        params["infer"] = True

    event_id = create_pending_add_event(resolve_event_owner_id(auth, entity_params))
    background_tasks.add_task(run_v3_add_memory_task, event_id, effective_messages, params)
    return pending_add_response(event_id)


@router.post("/v3/memories/", include_in_schema=False)
@router.post("/v3/memories", summary="Get all memories (v3)")
@upstream_guard
def v3_get_all_memories(
    request: Request,
    body: MemoryGetInputV3,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """List memories (v3 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present in
    filters, ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    warn_ignored_compat_params("v3_get_all_memories", latest_only=body.latest_only)
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="One of the filters: user_id, agent_id, app_id or run_id is required!",
    )
    filters = build_list_filters(body, entity_params)
    filters = strip_user_id_for_app_gate(filters)
    filters = check_query_permission(filters, operator.id, db)
    kwargs: Dict[str, Any] = {"filters": filters}
    if body.show_expired is not None:
        kwargs["show_expired"] = body.show_expired
    return paginated_get_all(request, page, page_size, **kwargs)


@router.post("/v3/memories/search/", include_in_schema=False)
@router.post("/v3/memories/search", summary="Search memories (v3)")
@upstream_guard
def v3_search_memories(
    body: MemorySearchInputV3,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Search memories (v3 compat).

    App-primary-gate: when both ``user_id`` and ``app_id`` are present,
    ``user_id`` is stripped (see ``strip_user_id_for_app_gate``)."""
    operator, _ = resolve_operator(request, auth, db)
    warn_ignored_compat_params(
        "v3_search_memories",
        latest_only=body.latest_only,
        reference_date=body.reference_date,
    )
    effective_filters = build_search_filters(
        user_id=body.user_id,
        agent_id=body.agent_id,
        app_id=body.app_id,
        run_id=body.run_id,
        filters=body.filters,
        detail="At least one of the filters: agent_id, user_id, app_id or run_id is required!",
    )
    effective_filters = append_search_convenience_filters(
        effective_filters,
        categories=body.categories,
        metadata=body.metadata,
    )
    effective_filters = strip_user_id_for_app_gate(effective_filters)
    effective_filters = check_query_permission(effective_filters, operator.id, db)
    raw = get_memory_instance().search(
        query=body.query,
        **build_search_kwargs(effective_filters, body.top_k, body.threshold, body.rerank, body.show_expired),
    )

    items = apply_fields(normalize_results(raw), body.fields)
    if body.output_format == "v1.0":
        return items
    return {"results": items}


# ---------------------------------------------------------------------------
# Project API stubs — return 501 for all project management endpoints.
# MemoryClient.project calls these; the self-hosted server does not implement
# a full project management API.
# ---------------------------------------------------------------------------


@router.get("/api/v1/orgs/organizations/{org_id}/projects/", include_in_schema=False)
@router.post("/api/v1/orgs/organizations/{org_id}/projects/", include_in_schema=False)
async def project_list_create(org_id: str) -> None:
    raise unsupported_api_error()


@router.get("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/", include_in_schema=False)
@router.patch("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/", include_in_schema=False)
@router.delete("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/", include_in_schema=False)
async def project_detail(org_id: str, project_id: str) -> None:
    raise unsupported_api_error()


@router.get("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/", include_in_schema=False)
@router.post("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/", include_in_schema=False)
@router.put("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/", include_in_schema=False)
@router.delete("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/", include_in_schema=False)
async def project_members(org_id: str, project_id: str) -> None:
    raise unsupported_api_error()


# ---------------------------------------------------------------------------
# Feedback  — 501 stub; self-hosted server does not implement feedback.
# ---------------------------------------------------------------------------


@router.post("/v1/feedback/", include_in_schema=False)
async def feedback() -> None:
    raise unsupported_api_error()


# ---------------------------------------------------------------------------
# Exports  — 501 stubs; self-hosted server does not implement export jobs.
# ---------------------------------------------------------------------------


@router.post("/v1/exports/", include_in_schema=False)
async def create_memory_export() -> None:
    raise unsupported_api_error()


@router.post("/v1/exports/get/", include_in_schema=False)
async def get_memory_export() -> None:
    raise unsupported_api_error()


# ---------------------------------------------------------------------------
# Summary  — 501 stub.
# ---------------------------------------------------------------------------


@router.post("/v1/summary/", include_in_schema=False)
async def get_summary() -> None:
    raise unsupported_api_error()


# ---------------------------------------------------------------------------
# Webhooks  — 501 stubs; self-hosted server does not implement webhooks.
# ---------------------------------------------------------------------------


@router.get("/api/v1/webhooks/projects/{project_id}/", include_in_schema=False)
@router.post("/api/v1/webhooks/projects/{project_id}/", include_in_schema=False)
async def webhooks_project(project_id: str) -> None:
    raise unsupported_api_error()


@router.put("/api/v1/webhooks/{webhook_id}/", include_in_schema=False)
@router.delete("/api/v1/webhooks/{webhook_id}/", include_in_schema=False)
async def webhook_detail(webhook_id: str) -> None:
    raise unsupported_api_error()
