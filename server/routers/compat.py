"""Client-compatible versioned API endpoints.

These routes expose the versioned paths used by ``MemoryClient`` and align the
self-hosted server as closely as practical with ``docs/openapi.json``.

Covered endpoints
-----------------
    GET    /v1/ping/

    GET    /v1/memories/
    POST   /v1/memories/
    DELETE /v1/memories/
    GET    /v1/memories/{entity_type}/{entity_id}/
    GET    /v1/memories/{memory_id}/
    PUT    /v1/memories/{memory_id}/
    DELETE /v1/memories/{memory_id}/
    GET    /v1/memories/{memory_id}/history/
    POST   /v1/memories/search/

    POST   /v2/memories/
    POST   /v2/memories/search/

    POST   /v3/memories/
    POST   /v3/memories/add/
    POST   /v3/memories/search/

    GET    /v1/entities/
    GET    /v1/entities/filters/
    GET    /v2/entities/{entity_type}/{entity_id}/
    DELETE /v2/entities/{entity_type}/{entity_id}/

    PUT    /v1/batch/
    DELETE /v1/batch/
"""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from auth import verify_auth
from compat.decorators import upstream_guard
from compat.entities import list_entities_payload
from compat.responses import drop_none, normalize_results
from compat.scope import (
    VALID_ENTITY_TYPES,
    collect_entity_params,
    get_entity_field,
    require_entity_scope,
)
from server_state import get_memory_instance, list_all_memories

router = APIRouter(tags=["Client API"])

class MemoryAddInput(BaseModel):
    messages: List[Dict[str, Any]] = Field(description="Array of message objects with 'role' and 'content' keys.")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier to scope the memory.")
    user_id: Optional[str] = Field(default=None, description="User identifier to scope the memory.")
    app_id: Optional[str] = Field(default=None, description="App identifier to scope the memory.")
    run_id: Optional[str] = Field(default=None, description="Run identifier to scope the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata to attach to the memory.")
    infer: Optional[bool] = Field(default=None, description="When False, store messages verbatim without LLM fact extraction.")
    categories: Optional[List[str]] = Field(default=None, description="Categories to assign to the memory.")


class MemorySearchInput(BaseModel):
    query: str = Field(description="The search query string.")
    agent_id: Optional[str] = Field(default=None, description="Agent ID to filter memories by.")
    user_id: Optional[str] = Field(default=None, description="User ID to filter memories by.")
    app_id: Optional[str] = Field(default=None, description="App ID to filter memories by.")
    run_id: Optional[str] = Field(default=None, description="Run ID to filter memories by.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for the search.")
    top_k: Optional[int] = Field(default=None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold (0.0–1.0).")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters with entity IDs and operators.")


class MemoryUpdateInput(BaseModel):
    text: Optional[str] = Field(default=None, description="New text content for the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata for the memory.")
    timestamp: Optional[Any] = Field(default=None, description="Unix timestamp for the memory.")


class MemoryBatchUpdateItem(BaseModel):
    memory_id: str = Field(description="ID of the memory to update.")
    text: Optional[str] = Field(default=None, description="New text content.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata.")


class MemoryBatchUpdateInput(BaseModel):
    memories: List[MemoryBatchUpdateItem] = Field(description="List of memories to update.")


class MemoryBatchDeleteItem(BaseModel):
    memory_id: str = Field(description="ID of the memory to delete.")


class MemoryBatchDeleteLegacyInput(BaseModel):
    memories: List[MemoryBatchDeleteItem] = Field(description="List of memories to delete (legacy format).")


class MemoryBatchDeleteInput(BaseModel):
    memory_ids: List[str] = Field(description="List of memory IDs to delete.")


class MemoryGetInputV2(BaseModel):
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters with entity IDs and operators (AND, OR, gte, lte, contains, etc.).",
    )
    start_date: Optional[str] = Field(default=None, description="Only return memories created on or after this ISO 8601 date.")
    end_date: Optional[str] = Field(default=None, description="Only return memories created on or before this ISO 8601 date.")
    categories: Optional[List[str]] = Field(default=None, description="Filter memories by categories.")


class MemorySearchInputV2(BaseModel):
    query: str = Field(description="The search query string.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters with entity IDs and operators.")
    top_k: Optional[int] = Field(default=None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold (0.0–1.0).")
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank results.")
    user_id: Optional[str] = Field(default=None, description="User ID to filter by (also accepted inside filters).")
    agent_id: Optional[str] = Field(default=None, description="Agent ID to filter by (also accepted inside filters).")
    app_id: Optional[str] = Field(default=None, description="App ID to filter by (also accepted inside filters).")
    run_id: Optional[str] = Field(default=None, description="Run ID to filter by (also accepted inside filters).")
    fields: Optional[List[str]] = Field(default=None, description="Field names to include in the response.")


class MemoryAddInputV3(BaseModel):
    messages: List[Dict[str, Any]] = Field(description="Array of message objects with 'role' and 'content' keys.")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier to scope the memory.")
    user_id: Optional[str] = Field(default=None, description="User identifier to scope the memory.")
    app_id: Optional[str] = Field(default=None, description="App identifier to scope the memory.")
    run_id: Optional[str] = Field(default=None, description="Run identifier to scope the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata to attach to the memory.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters containing entity IDs (e.g. {'user_id': '...'}).")
    infer: Optional[bool] = Field(default=None, description="When False, store messages verbatim without LLM fact extraction.")
    custom_categories: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Custom category definitions with name and description."
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Project-specific guidelines for handling and organizing memories."
    )
    structured_data_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Schema for structured data extraction from the memory."
    )
    timestamp: Optional[int] = Field(default=None, description="Unix timestamp for the memory.")


class MemorySearchInputV3(BaseModel):
    query: str = Field(description="The search query string.")
    agent_id: Optional[str] = Field(default=None, description="Agent ID to filter by.")
    user_id: Optional[str] = Field(default=None, description="User ID to filter by.")
    app_id: Optional[str] = Field(default=None, description="App ID to filter by.")
    run_id: Optional[str] = Field(default=None, description="Run ID to filter by.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters with entity IDs and operators.")
    top_k: Optional[int] = Field(default=None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold (0.0–1.0).")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for the search.")
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank results.")
    fields: Optional[List[str]] = Field(default=None, description="Field names to include in the response.")
    categories: Optional[List[str]] = Field(default=None, description="Categories to filter by.")


def _build_page_url(request: Request, *, page: int, page_size: int) -> str:
    params = dict(request.query_params)
    params["page"] = str(page)
    params["page_size"] = str(page_size)
    return f"{request.url.path}?{urlencode(params)}"


@router.get("/v1/ping/", summary="Ping / validate API key")
def ping(_auth=Depends(verify_auth)):
    """Used by ``MemoryClient`` to validate the API key on initialisation."""
    user_email = getattr(_auth, "email", None) if _auth else None
    return {"status": "ok", "message": "pong", "user_email": user_email}


@router.get("/v1/memories/", summary="Get all memories (v1)")
@upstream_guard
def v1_list_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    filters = drop_none({"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id})
    raw = get_memory_instance().get_all(filters=filters) if filters else list_all_memories()
    return normalize_results(raw)


@router.post("/v1/memories/", summary="Add memories (v1)")
@upstream_guard
def v1_add_memories(body: MemoryAddInput, _auth=Depends(verify_auth)):
    entity_params = collect_entity_params(
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id, app_id=body.app_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    params = drop_none({**entity_params, "metadata": body.metadata})
    if body.infer is not None:
        params["infer"] = body.infer
    if body.categories:
        meta = params.get("metadata") or {}
        meta["categories"] = body.categories
        params["metadata"] = meta
    result = get_memory_instance().add(messages=body.messages, **params)
    return normalize_results(result)


@router.get("/v1/memories/{memory_id}/", summary="Get a memory (v1)")
@upstream_guard
def v1_get_memory(memory_id: str, _auth=Depends(verify_auth)):
    return get_memory_instance().get(memory_id)


@router.put("/v1/memories/{memory_id}/", summary="Update a memory (v1)")
@upstream_guard
def v1_update_memory(memory_id: str, body: MemoryUpdateInput, _auth=Depends(verify_auth)):
    if not any([body.text, body.metadata, body.timestamp]):
        raise HTTPException(
            status_code=400,
            detail="At least one of text, metadata, or timestamp must be provided for update.",
        )
    metadata = body.metadata
    if body.timestamp is not None:
        metadata = {**(metadata or {}), "timestamp": body.timestamp}
    return get_memory_instance().update(
        memory_id=memory_id,
        data=body.text,
        metadata=metadata,
    )


@router.delete("/v1/memories/{memory_id}/", summary="Delete a memory (v1)")
@upstream_guard
def v1_delete_memory(memory_id: str, _auth=Depends(verify_auth)):
    get_memory_instance().delete(memory_id=memory_id)
    return {"message": "Memory deleted successfully."}


@router.get("/v1/memories/{memory_id}/history/", summary="Get memory history (v1)")
@upstream_guard
def v1_memory_history(memory_id: str, _auth=Depends(verify_auth)):
    return get_memory_instance().history(memory_id=memory_id)


@router.get("/v1/memories/{entity_type}/{entity_id}/", summary="Get memories for an entity (v1)")
@upstream_guard
def v1_get_entity_memories(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    raw = get_memory_instance().get_all(filters={get_entity_field(entity_type): entity_id})
    return normalize_results(raw)


@router.post("/v1/memories/search/", summary="Search memories (v1)")
@upstream_guard
def v1_search_memories(body: MemorySearchInput, _auth=Depends(verify_auth)):
    entity_params = collect_entity_params(
        filters=body.filters,
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id, app_id=body.app_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    search_kwargs: Dict[str, Any] = {"filters": entity_params}
    if body.top_k is not None:
        search_kwargs["top_k"] = body.top_k
    if body.threshold is not None:
        search_kwargs["threshold"] = body.threshold
    result = get_memory_instance().search(query=body.query, **search_kwargs)
    return normalize_results(result)


@router.delete("/v1/memories/", summary="Delete all memories (v1)")
@upstream_guard
def v1_delete_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    # ``filters`` is a legacy query-string JSON blob (not the structured dict
    # used in v2/v3 body endpoints). Only parse it when no explicit entity
    # params are given, to avoid silently overriding explicit args.
    if filters and not any([user_id, agent_id, app_id, run_id]):
        try:
            filters_dict = json.loads(filters)
            user_id = user_id or filters_dict.get("user_id")
            agent_id = agent_id or filters_dict.get("agent_id")
            app_id = app_id or filters_dict.get("app_id")
            run_id = run_id or filters_dict.get("run_id")
        except (json.JSONDecodeError, AttributeError):
            pass

    params = drop_none({"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id})
    if not params:
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, app_id, run_id) is required.",
        )
    # SDK delete_all only accepts user_id, agent_id, run_id — filter app_id
    sdk_params = {k: v for k, v in params.items() if k in ("user_id", "agent_id", "run_id")}
    get_memory_instance().delete_all(**sdk_params)
    return {"message": "All memories deleted successfully."}


@router.put("/v1/batch/", summary="Batch update memories (v1)")
@upstream_guard
def v1_batch_update(body: MemoryBatchUpdateInput, _auth=Depends(verify_auth)):
    if len(body.memories) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be updated in a single request")
    mem = get_memory_instance()
    for item in body.memories:
        mem.update(memory_id=item.memory_id, data=item.text, metadata=item.metadata)
    return {"message": f"Successfully updated {len(body.memories)} memories"}


@router.delete("/v1/batch/", summary="Batch delete memories (v1)")
@upstream_guard
def v1_batch_delete(
    body: MemoryBatchDeleteLegacyInput | MemoryBatchDeleteInput,
    _auth=Depends(verify_auth),
):
    mem = get_memory_instance()
    memory_ids = body.memory_ids if isinstance(body, MemoryBatchDeleteInput) else [item.memory_id for item in body.memories]
    if len(memory_ids) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be deleted in a single request")
    for memory_id in memory_ids:
        mem.delete(memory_id=memory_id)
    return {"message": f"Successfully deleted {len(memory_ids)} memories"}


@router.get("/v1/entities/", summary="List entities (v1)")
def v1_list_entities(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    _auth=Depends(verify_auth),
):
    """Return entities in the SDK-compatible envelope while preserving spec fields.

    The hosted spec documents an array response, but ``MemoryClient.users()`` and
    ``delete_users()`` read ``response["results"]``. Keep that envelope here and
    include the spec fields on each entity item.
    """
    all_results = list_entities_payload()
    total = len(all_results)
    start = (page - 1) * page_size
    page_results = all_results[start : start + page_size]
    return {"count": total, "results": page_results}


@router.get("/v1/entities/filters/", summary="List supported entity filters (v1)")
def v1_list_entity_filters(_auth=Depends(verify_auth)):
    return {"results": sorted(VALID_ENTITY_TYPES)}


@router.post("/v2/memories/", summary="Get all memories (v2)")
@upstream_guard
def v2_list_memories(
    request: Request,
    body: MemoryGetInputV2,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    _auth=Depends(verify_auth),
):
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    sdk_filters = dict(entity_params)
    date_filter: Dict[str, str] = {}
    if body.start_date:
        date_filter["gte"] = body.start_date
    if body.end_date:
        date_filter["lte"] = body.end_date
    if date_filter:
        sdk_filters["created_at"] = date_filter
    if body.categories:
        sdk_filters["categories"] = {"contains": body.categories}
    raw = get_memory_instance().get_all(filters=sdk_filters)
    items = normalize_results(raw)

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]
    next_url = _build_page_url(request, page=page + 1, page_size=page_size) if start + page_size < total else None
    previous_url = _build_page_url(request, page=page - 1, page_size=page_size) if page > 1 else None

    return {
        "count": total,
        "next": next_url,
        "previous": previous_url,
        "results": page_items,
    }


@router.post("/v2/memories/search/", summary="Search memories (v2)")
@upstream_guard
def v2_search_memories(body: MemorySearchInputV2, _auth=Depends(verify_auth)):
    entity_params = collect_entity_params(
        filters=body.filters,
        user_id=body.user_id, agent_id=body.agent_id, app_id=body.app_id, run_id=body.run_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    search_kwargs: Dict[str, Any] = {"filters": entity_params}
    if body.top_k is not None:
        search_kwargs["top_k"] = body.top_k
    if body.threshold is not None:
        search_kwargs["threshold"] = body.threshold
    if body.rerank is not None:
        search_kwargs["rerank"] = body.rerank
    result = get_memory_instance().search(query=body.query, **search_kwargs)
    return normalize_results(result)


@router.get("/v2/entities/{entity_type}/{entity_id}/", summary="Get entity details (v2)")
@upstream_guard
def v2_get_entity(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    get_entity_field(entity_type)  # validate entity_type early
    for entity in list_entities_payload():
        if entity["type"] == entity_type and entity["id"] == entity_id:
            return entity
    raise HTTPException(status_code=404, detail="Entity not found.")


@router.delete("/v2/entities/{entity_type}/{entity_id}/", summary="Delete entity (v2)", status_code=204)
@upstream_guard
def v2_delete_entity(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    get_memory_instance().delete_all(**{get_entity_field(entity_type): entity_id})
    return Response(status_code=204)


@router.post("/v3/memories/add/", summary="Add memory (v3)")
@upstream_guard
def v3_add_memory(body: MemoryAddInputV3, _auth=Depends(verify_auth)):
    entity_params = collect_entity_params(
        filters=body.filters,
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id, app_id=body.app_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    params: Dict[str, Any] = drop_none({
        **entity_params,
        "metadata": body.metadata,
        "infer": body.infer,
    })
    # Merge platform-only fields into metadata for self-hosted preservation
    extra_meta = drop_none({
        "custom_categories": body.custom_categories,
        "custom_instructions": body.custom_instructions,
        "structured_data_schema": body.structured_data_schema,
        "timestamp": body.timestamp,
    })
    if extra_meta:
        meta = params.get("metadata") or {}
        meta.update(extra_meta)
        params["metadata"] = meta
    result = get_memory_instance().add(messages=body.messages, **params)
    return JSONResponse(content=result)


@router.post("/v3/memories/", summary="Get all memories (v3)")
@upstream_guard
def v3_get_all_memories(
    request: Request,
    body: MemoryGetInputV2,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    _auth=Depends(verify_auth),
):
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    sdk_filters = dict(entity_params)
    date_filter: Dict[str, str] = {}
    if body.start_date:
        date_filter["gte"] = body.start_date
    if body.end_date:
        date_filter["lte"] = body.end_date
    if date_filter:
        sdk_filters["created_at"] = date_filter
    if body.categories:
        sdk_filters["categories"] = {"contains": body.categories}
    raw = get_memory_instance().get_all(filters=sdk_filters)
    items = normalize_results(raw)

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]
    next_url = _build_page_url(request, page=page + 1, page_size=page_size) if start + page_size < total else None
    previous_url = _build_page_url(request, page=page - 1, page_size=page_size) if page > 1 else None

    return {
        "count": total,
        "next": next_url,
        "previous": previous_url,
        "results": page_items,
    }


@router.post("/v3/memories/search/", summary="Search memories (v3)")
@upstream_guard
def v3_search_memories(body: MemorySearchInputV3, _auth=Depends(verify_auth)):
    entity_params = collect_entity_params(
        filters=body.filters,
        user_id=body.user_id, agent_id=body.agent_id, app_id=body.app_id, run_id=body.run_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    search_kwargs: Dict[str, Any] = {"filters": entity_params}
    if body.top_k is not None:
        search_kwargs["top_k"] = body.top_k
    if body.threshold is not None:
        search_kwargs["threshold"] = body.threshold
    if body.rerank is not None:
        search_kwargs["rerank"] = body.rerank
    result = get_memory_instance().search(query=body.query, **search_kwargs)
    if isinstance(result, list):
        return {"results": result}
    return result
