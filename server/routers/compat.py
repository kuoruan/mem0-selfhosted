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
from pydantic import BaseModel

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
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    app_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchInput(BaseModel):
    query: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None


class MemoryUpdateInput(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[Any] = None


class MemoryBatchUpdateItem(BaseModel):
    memory_id: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryBatchUpdateInput(BaseModel):
    memories: List[MemoryBatchUpdateItem]


class MemoryBatchDeleteItem(BaseModel):
    memory_id: str


class MemoryBatchDeleteLegacyInput(BaseModel):
    memories: List[MemoryBatchDeleteItem]


class MemoryBatchDeleteInput(BaseModel):
    memory_ids: List[str]


class MemoryGetInputV2(BaseModel):
    filters: Optional[Dict[str, Any]] = None


class MemorySearchInputV2(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None


class MemoryAddInputV3(BaseModel):
    messages: List[Dict[str, Any]]
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    app_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = None
    memory_type: Optional[str] = None
    prompt: Optional[str] = None


class MemorySearchInputV3(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None


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
    result = get_memory_instance().add(messages=body.messages, **params)
    return normalize_results(result)


@router.get("/v1/memories/{memory_id}/", summary="Get a memory (v1)")
@upstream_guard
def v1_get_memory(memory_id: str, _auth=Depends(verify_auth)):
    return get_memory_instance().get(memory_id)


@router.put("/v1/memories/{memory_id}/", summary="Update a memory (v1)")
@upstream_guard
def v1_update_memory(memory_id: str, body: MemoryUpdateInput, _auth=Depends(verify_auth)):
    if not any([body.text, body.metadata]):
        raise HTTPException(
            status_code=400,
            detail="At least one of text or metadata must be provided for update.",
        )
    return get_memory_instance().update(
        memory_id=memory_id,
        data=body.text,
        metadata=body.metadata,
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
    params = drop_none({**entity_params, "top_k": body.top_k, "threshold": body.threshold})
    result = get_memory_instance().search(query=body.query, **params)
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
    get_memory_instance().delete_all(**params)
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
def v1_list_entities(_auth=Depends(verify_auth)):
    """Return entities in the SDK-compatible envelope while preserving spec fields.

    The hosted spec documents an array response, but ``MemoryClient.users()`` and
    ``delete_users()`` read ``response["results"]``. Keep that envelope here and
    include the spec fields on each entity item.
    """
    results = list_entities_payload()
    return {"count": len(results), "results": results}


@router.get("/v1/entities/filters/", summary="List supported entity filters (v1)")
def v1_list_entity_filters(_auth=Depends(verify_auth)):
    return {"results": sorted(VALID_ENTITY_TYPES)}


@router.post("/v2/memories/", summary="Get all memories (v2)")
@upstream_guard
def v2_list_memories(body: MemoryGetInputV2, _auth=Depends(verify_auth)):
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    raw = get_memory_instance().get_all(filters=entity_params)
    return normalize_results(raw)


@router.post("/v2/memories/search/", summary="Search memories (v2)")
@upstream_guard
def v2_search_memories(body: MemorySearchInputV2, _auth=Depends(verify_auth)):
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    params = drop_none({**entity_params, "top_k": body.top_k})
    result = get_memory_instance().search(query=body.query, **params)
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
        "memory_type": body.memory_type,
        "prompt": body.prompt,
    })
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
    raw = get_memory_instance().get_all(filters=entity_params)
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
    entity_params = require_entity_scope(
        filters=body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    params: Dict[str, Any] = drop_none({**entity_params, "top_k": body.top_k, "threshold": body.threshold})
    result = get_memory_instance().search(query=body.query, **params)
    if isinstance(result, list):
        return {"results": result}
    return result
