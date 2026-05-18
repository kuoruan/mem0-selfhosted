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
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from auth import verify_auth
from compat.decorators import upstream_guard
from compat.entities import list_entities_payload
from compat.responses import drop_none, normalize_results
from compat.scope import (
    VALID_ENTITY_TYPES,
    build_search_filters,
    collect_entity_params,
    get_entity_field,
    reject_app_id,
    require_entity_scope,
)
from server_state import get_memory_instance, list_all_memories

router = APIRouter(tags=["Client API"])


class MemoryAddInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]] = Field(
        description="An array of message objects representing the content of the memory. "
        "Each message object typically contains 'role' and 'content' fields, where 'role' "
        "indicates the sender ('user' or 'assistant') and 'content' contains the actual message text. "
        "This structure allows for the representation of conversations or multi-part memories."
    )
    agent_id: Optional[str] = Field(default=None, description="The unique identifier of the agent associated with this memory.")
    user_id: Optional[str] = Field(default=None, description="The unique identifier of the user associated with this memory.")
    app_id: Optional[str] = Field(default=None, description="The unique identifier of the application. Not supported by the self-hosted server (returns 501).")
    run_id: Optional[str] = Field(default=None, description="The unique identifier of the run associated with this memory.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with the memory. Best practice for incorporating additional "
        "information is through metadata (e.g. location, time, ids, etc.). During retrieval, you can either use "
        "these metadata alongside the query to fetch relevant memories or retrieve memories based on the query "
        "first and then refine the results using metadata during post-processing.",
    )
    infer: Optional[bool] = Field(default=None, description="Whether to infer the memories or directly store the messages.")
    categories: Optional[List[str]] = Field(default=None, description="A list of categories to tag the memory with.")


class MemorySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The query to search for in the memory.")
    agent_id: Optional[str] = Field(default=None, description="The agent ID associated with the memory.")
    user_id: Optional[str] = Field(default=None, description="The user ID associated with the memory.")
    app_id: Optional[str] = Field(default=None, description="The app ID associated with the memory. Not supported by the self-hosted server (returns 501).")
    run_id: Optional[str] = Field(default=None, description="The run ID associated with the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata associated with the memory.")
    top_k: Optional[int] = Field(default=None, description="The number of top results to return.")
    threshold: Optional[float] = Field(default=None, description="The minimum similarity threshold for returned results.")
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank the memories.")
    fields: Optional[List[str]] = Field(default=None, description="A list of field names to include in the response. If not provided, all fields will be returned.")


class MemoryUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Optional[str] = Field(default=None, description="New text content for the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata for the memory.")
    timestamp: Optional[Any] = Field(default=None, description="Unix timestamp for the memory.")


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


class MemoryBatchDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_ids: List[str] = Field(description="List of memory IDs to delete.")


class MemoryGetInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary of filters to apply to retrieve memories. Available fields are: "
        "user_id, agent_id, run_id, created_at, updated_at, categories, keywords. "
        "Supports logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains, *). "
        "For categories field, use 'contains' for partial matching "
        "(e.g., {\"categories\": {\"contains\": \"finance\"}}) or 'in' for exact matching "
        "(e.g., {\"categories\": {\"in\": [\"personal_information\"]}}).",
    )
    start_date: Optional[str] = Field(default=None, description="Only return memories created on or after this ISO 8601 date.")
    end_date: Optional[str] = Field(default=None, description="Only return memories created on or before this ISO 8601 date.")
    categories: Optional[List[str]] = Field(default=None, description="A list of categories to filter the memories by.")


class MemorySearchInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The query to search for in the memory.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary of filters to apply to the search. Available fields are: "
        "user_id, agent_id, run_id, created_at, updated_at, categories, keywords. "
        "Supports logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains). "
        "For categories field, use 'contains' for partial matching "
        "(e.g., {\"categories\": {\"contains\": \"finance\"}}) or 'in' for exact matching "
        "(e.g., {\"categories\": {\"in\": [\"personal_information\"]}}).",
    )
    top_k: Optional[int] = Field(default=None, description="The number of top results to return.")
    threshold: Optional[float] = Field(default=None, description="The minimum similarity threshold for returned results.")
    rerank: Optional[bool] = Field(default=None, description="Whether to rerank the memories.")
    user_id: Optional[str] = Field(default=None, description="The user ID associated with the memory (also accepted inside filters).")
    agent_id: Optional[str] = Field(default=None, description="The agent ID associated with the memory (also accepted inside filters).")
    app_id: Optional[str] = Field(default=None, description="Not supported by the self-hosted server (returns 501).")
    run_id: Optional[str] = Field(default=None, description="The run ID associated with the memory (also accepted inside filters).")
    fields: Optional[List[str]] = Field(default=None, description="A list of field names to include in the response. If not provided, all fields will be returned.")


class MemoryAddInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]] = Field(description="Conversation messages to extract memories from. "
        "Each object must have 'role' ('user', 'assistant', or 'system') and 'content' keys.")
    agent_id: Optional[str] = Field(default=None, description="Scope memories to this agent.")
    user_id: Optional[str] = Field(default=None, description="Scope memories to this user.")
    app_id: Optional[str] = Field(default=None, description="Not supported by the self-hosted server (returns 501).")
    run_id: Optional[str] = Field(default=None, description="Scope memories to this session / run.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="User-supplied metadata to attach to each extracted memory."
    )
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters containing entity IDs (e.g. {'user_id': '...'}).")
    infer: Optional[bool] = Field(
        default=None, description="When `false`, stores each message verbatim without running the extraction LLM."
    )
    custom_categories: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="A list of categories with category name and its description."
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Project-level instructions that guide extraction for this call."
    )
    structured_data_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Schema for structured data extraction from the memory."
    )
    timestamp: Optional[int] = Field(default=None, description="The timestamp of the memory. Format: Unix timestamp")
    source: Optional[str] = Field(default=None, description="Source identifier for the memory (e.g. 'OPENCLAW'). Stored in metadata.")
    deduced_memories: Optional[List[Any]] = Field(
        default=None, description="Pre-extracted fact strings used by agentic harnesses when infer=False. Stored in metadata."
    )


class MemorySearchInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Natural-language search query.")
    agent_id: Optional[str] = Field(default=None, description="The agent ID associated with the memory.")
    user_id: Optional[str] = Field(default=None, description="The user ID associated with the memory.")
    app_id: Optional[str] = Field(default=None, description="Not supported by the self-hosted server (returns 501).")
    run_id: Optional[str] = Field(default=None, description="The run ID associated with the memory.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Entity and metadata filters. Must include at least one entity ID "
        "(`user_id`, `agent_id`, or `run_id`). Supports `AND`, `OR`, `NOT`, and "
        "comparison operators (`in`, `gte`, `lte`, `gt`, `lt`, `contains`, `icontains`, `ne`).",
    )
    top_k: Optional[int] = Field(default=None, description="Number of results to return.")
    threshold: Optional[float] = Field(
        default=None, description="Minimum semantic relevance score. Pass `0.0` to disable filtering."
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata associated with the memory.")
    rerank: Optional[bool] = Field(default=None, description="Apply the managed reranker for better ordering (adds latency).")
    fields: Optional[List[str]] = Field(
        default=None, description="A list of field names to include in the response. If not provided, all fields will be returned."
    )
    categories: Optional[List[str]] = Field(default=None, description="A list of categories to filter the memories by.")
    output_format: Optional[str] = Field(
        default=None,
        description="Response format. `v1.1` (default) returns `{\"results\": [...]}`. "
        "`v1.0` returns a flat array `[{...}]` for backwards compatibility.",
    )


def _build_page_url(request: Request, *, page: int, page_size: int) -> str:
    params = dict(request.query_params)
    params["page"] = str(page)
    params["page_size"] = str(page_size)
    return f"{request.url.path}?{urlencode(params)}"


def _build_list_filters(
    body: "MemoryGetInputV2",
    entity_params: Dict[str, str],
) -> Dict[str, Any]:
    """Build SDK filter dict for get_all from a MemoryGetInputV2 body.

    Starts from body.filters so non-entity conditions (e.g. created_at) are
    preserved, then merges date / categories convenience fields for flat format.
    setdefault avoids overriding conditions already present in body.filters.
    """
    sdk_filters: Dict[str, Any] = dict(body.filters) if body.filters else dict(entity_params)
    if "AND" not in sdk_filters and "OR" not in sdk_filters:
        date_filter: Dict[str, str] = {}
        if body.start_date:
            date_filter["gte"] = body.start_date
        if body.end_date:
            date_filter["lte"] = body.end_date
        if date_filter:
            sdk_filters.setdefault("created_at", date_filter)
        if body.categories:
            sdk_filters.setdefault("categories", {"contains": body.categories})
    return sdk_filters


def _paginate_response(
    request: Request,
    items: List[Any],
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """Wrap a list of items in the SDK-compatible pagination envelope."""
    total = len(items)
    start = (page - 1) * page_size
    return {
        "count": total,
        "next": _build_page_url(request, page=page + 1, page_size=page_size) if start + page_size < total else None,
        "previous": _build_page_url(request, page=page - 1, page_size=page_size) if page > 1 else None,
        "results": items[start: start + page_size],
    }


def _warn_unsupported_fields(fields: Optional[List[str]], endpoint: str) -> None:
    """Log a warning when 'fields' projection is requested but not supported by the OSS SDK."""
    if fields:
        logging.warning(
            "%s: 'fields' projection is not supported by the OSS SDK "
            "and will be ignored. Requested fields: %s",
            endpoint,
            fields,
        )


def _build_search_kwargs(
    filters: Dict[str, Any],
    top_k: Optional[int],
    threshold: Optional[float],
    rerank: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for Memory.search() from common request fields."""
    kwargs: Dict[str, Any] = {"filters": filters}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if threshold is not None:
        kwargs["threshold"] = threshold
    if rerank is not None:
        kwargs["rerank"] = rerank
    return kwargs


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
    run_id: Optional[str] = None,
    app_id: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    if app_id is not None:
        raise HTTPException(status_code=501, detail="'app_id' is not supported by the self-hosted server.")
    filters = drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    raw = get_memory_instance().get_all(filters=filters) if filters else list_all_memories()
    return normalize_results(raw)


@router.post("/v1/memories/", summary="Add memories (v1)")
@upstream_guard
def v1_add_memories(body: MemoryAddInput, _auth=Depends(verify_auth)):
    reject_app_id(body.app_id)
    entity_params = collect_entity_params(
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    params = drop_none({**entity_params, "metadata": body.metadata})
    if body.infer is not None:
        params["infer"] = body.infer
    if body.categories:
        meta = params.get("metadata") or {}
        meta.setdefault("categories", body.categories)
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
    if body.text is None and body.metadata is None and body.timestamp is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of text, metadata, or timestamp must be provided for update.",
        )
    metadata = body.metadata
    if body.timestamp is not None:
        metadata = {**(metadata or {}), "timestamp": body.timestamp}
    if body.text is not None:
        return get_memory_instance().update(
            memory_id=memory_id,
            data=body.text,
            metadata=metadata,
        )
    # text is None — metadata-only update: read existing, merge metadata, write back
    mem = get_memory_instance()
    existing_raw = mem.get(memory_id)
    # Some SDK versions return a list-of-one instead of a plain dict.
    existing = existing_raw[0] if isinstance(existing_raw, list) and existing_raw else existing_raw
    if not isinstance(existing, dict):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    existing_text = existing.get("memory") or existing.get("text") or ""
    existing_metadata = existing.get("metadata") or {}
    merged_metadata = {**existing_metadata, **(metadata or {})}
    return mem.update(memory_id=memory_id, data=existing_text, metadata=merged_metadata)


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
    reject_app_id(body.app_id)
    _warn_unsupported_fields(body.fields, "v1_search_memories")
    entity_params = collect_entity_params(
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    search_filters: Dict[str, Any] = {**entity_params}
    if body.metadata:
        # Merge client-supplied metadata filters; entity params take precedence.
        for k, v in body.metadata.items():
            search_filters.setdefault(k, v)
    result = get_memory_instance().search(
        query=body.query, **_build_search_kwargs(search_filters, body.top_k, body.threshold, body.rerank)
    )
    return normalize_results(result)


@router.delete("/v1/memories/", summary="Delete all memories (v1)")
@upstream_guard
def v1_delete_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    app_id: Optional[str] = None,
    filters: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    if app_id is not None:
        raise HTTPException(status_code=501, detail="'app_id' is not supported by the self-hosted server.")
    # ``filters`` is a legacy query-string JSON blob (not the structured dict
    # used in v2/v3 body endpoints). Only parse it when no explicit entity
    # params are given, to avoid silently overriding explicit args.
    if filters and not any([user_id, agent_id, run_id]):
        try:
            filters_dict = json.loads(filters)
            user_id = user_id or filters_dict.get("user_id")
            agent_id = agent_id or filters_dict.get("agent_id")
            run_id = run_id or filters_dict.get("run_id")
            if filters_dict.get("app_id"):
                raise HTTPException(status_code=501, detail="'app_id' is not supported by the self-hosted server.")
        except json.JSONDecodeError:
            pass
        except AttributeError:
            pass

    params = drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    if not params:
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )
    get_memory_instance().delete_all(**params)
    return {"message": "All memories deleted successfully."}


@router.put("/v1/batch/", summary="Batch update memories (v1)")
@upstream_guard
def v1_batch_update(body: MemoryBatchUpdateInput, _auth=Depends(verify_auth)):
    if len(body.memories) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be updated in a single request")
    mem = get_memory_instance()
    updated_count = 0
    for item in body.memories:
        if item.text is not None:
            mem.update(memory_id=item.memory_id, data=item.text, metadata=item.metadata)
            updated_count += 1
        elif item.metadata is not None:
            # NOTE: metadata-only updates require fetching the existing text first.
            # This is an inherent N+1 limitation: the OSS SDK has no bulk-get API.
            existing_raw = mem.get(item.memory_id)
            existing = existing_raw[0] if isinstance(existing_raw, list) and existing_raw else existing_raw
            if isinstance(existing, dict):
                existing_text = existing.get("memory") or existing.get("text") or ""
                existing_metadata = existing.get("metadata") or {}
                merged_metadata = {**existing_metadata, **item.metadata}
                mem.update(memory_id=item.memory_id, data=existing_text, metadata=merged_metadata)
                updated_count += 1
    return {"message": f"Successfully updated {updated_count} memories"}


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
        detail="filters must include at least one entity ID (user_id, agent_id, or run_id).",
    )
    raw = get_memory_instance().get_all(filters=_build_list_filters(body, entity_params))
    # NOTE: Pagination is performed in-memory. The OSS SDK's get_all() does not yet
    # support server-side limit/offset. Known limitation for very large datasets.
    # NOTE: docs/openapi.json declares this endpoint as returning a bare array, but
    # MemoryClient parses the pagination envelope {count, next, previous, results}.
    # We intentionally diverge from openapi.json to remain client-compatible.
    return _paginate_response(request, normalize_results(raw), page, page_size)


@router.post("/v2/memories/search/", summary="Search memories (v2)")
@upstream_guard
def v2_search_memories(body: MemorySearchInputV2, _auth=Depends(verify_auth)):
    reject_app_id(body.app_id)
    _warn_unsupported_fields(body.fields, "v2_search_memories")
    effective_filters = build_search_filters(
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id,
        filters=body.filters,
        detail="At least one entity ID is required.",
    )
    result = get_memory_instance().search(
        query=body.query, **_build_search_kwargs(effective_filters, body.top_k, body.threshold, body.rerank)
    )
    # NOTE: docs/openapi.json declares a bare array response, but MemoryClient
    # reads response["results"]. We intentionally return the envelope here.
    return {"results": normalize_results(result)}


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
    reject_app_id(body.app_id)
    entity_params = collect_entity_params(
        filters=body.filters,
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id,
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
        "source": body.source,
        "deduced_memories": body.deduced_memories,
    })
    if extra_meta:
        meta = params.get("metadata") or {}
        meta.update(extra_meta)
        params["metadata"] = meta
    result = get_memory_instance().add(messages=body.messages, **params)
    return result


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
        detail="filters must include at least one entity ID (user_id, agent_id, or run_id).",
    )
    raw = get_memory_instance().get_all(filters=_build_list_filters(body, entity_params))
    # NOTE: Pagination is performed in-memory. The OSS SDK's get_all() does not yet
    # support server-side limit/offset. Known limitation for very large datasets.
    return _paginate_response(request, normalize_results(raw), page, page_size)


@router.post("/v3/memories/search/", summary="Search memories (v3)")
@upstream_guard
def v3_search_memories(body: MemorySearchInputV3, _auth=Depends(verify_auth)):
    reject_app_id(body.app_id)
    _warn_unsupported_fields(body.fields, "v3_search_memories")
    effective_filters = build_search_filters(
        user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id,
        filters=body.filters,
        detail="At least one entity ID is required.",
    )
    # Merge convenience fields for flat (non-AND/OR/NOT) dicts.
    if "AND" not in effective_filters and "OR" not in effective_filters and "NOT" not in effective_filters:
        if body.categories and "categories" not in effective_filters:
            effective_filters["categories"] = {"contains": body.categories}
        if body.metadata:
            # Merge metadata filters; entity params and explicit filters take precedence.
            for k, v in body.metadata.items():
                effective_filters.setdefault(k, v)
    result = get_memory_instance().search(
        query=body.query, **_build_search_kwargs(effective_filters, body.top_k, body.threshold, body.rerank)
    )
    if body.output_format == "v1.0":
        # Legacy flat-array format requested by the caller.
        return result if isinstance(result, list) else (result or {}).get("results", [])
    if isinstance(result, list):
        return {"results": result}
    return result
