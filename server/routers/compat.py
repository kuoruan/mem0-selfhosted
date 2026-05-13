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
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from auth import verify_auth
from errors import upstream_error
from routers.entities import _iter_payloads, _parse_timestamp
from server_state import get_memory_instance, list_all_memories

router = APIRouter(tags=["Client API"])

# Entity param names that may appear either at top-level or inside ``filters``.
_ENTITY_PARAMS = frozenset({"user_id", "agent_id", "app_id", "run_id"})
CompatEntityType = str
COMPAT_TYPE_TO_FIELD: dict[str, str] = {
    "user": "user_id",
    "agent": "agent_id",
    "app": "app_id",
    "run": "run_id",
}
VALID_ENTITY_TYPES = frozenset(COMPAT_TYPE_TO_FIELD)


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------


class V3AddMemoryBody(BaseModel):
    messages: List[Dict[str, Any]]
    # New-style: entity params inside filters dict
    filters: Optional[Dict[str, Any]] = None
    # Old-style (kwargs): entity params at top level – kept for back-compat
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    app_id: Optional[str] = None
    # Common add options
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = None
    memory_type: Optional[str] = None
    prompt: Optional[str] = None


class V3GetAllBody(BaseModel):
    filters: Optional[Dict[str, Any]] = None


class V3SearchBody(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None


class V1MemoryUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[Any] = None


class BatchMemoryUpdateItem(BaseModel):
    memory_id: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchUpdateBody(BaseModel):
    memories: List[BatchMemoryUpdateItem]


class BatchDeleteItem(BaseModel):
    memory_id: str


class BatchDeleteBody(BaseModel):
    memories: List[BatchDeleteItem]


class BatchDeleteIdsBody(BaseModel):
    memory_ids: List[str]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _entity_params_from(
    body_filters: Optional[Dict[str, Any]],
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve entity params, preferring top-level values over filters dict."""
    merged: Dict[str, Any] = {}
    if body_filters:
        merged.update({k: v for k, v in body_filters.items() if k in _ENTITY_PARAMS and v is not None})
    for key, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id), ("app_id", app_id)):
        if val is not None:
            merged[key] = val
    return merged


def _require_entity_scope(filters: Optional[Dict[str, Any]], *, detail: str) -> Dict[str, Any]:
    entity_params = _entity_params_from(filters)
    if not entity_params:
        raise HTTPException(status_code=400, detail=detail)
    return entity_params


def _normalize_results(raw: Any) -> List[Any]:
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    return []


def _build_page_url(request: Request, *, page: int, page_size: int) -> str:
    params = dict(request.query_params)
    params["page"] = str(page)
    params["page_size"] = str(page_size)
    return f"{request.url.path}?{urlencode(params)}"


def _get_entity_field(entity_type: str) -> str:
    field = COMPAT_TYPE_TO_FIELD.get(entity_type)
    if field is None:
        raise HTTPException(status_code=400, detail="Invalid entity type")
    return field


def _list_entities_payload() -> List[Dict[str, Any]]:
    buckets: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"total_memories": 0, "created_at": None, "updated_at": None, "metadata": {}}
    )

    for payload in _iter_payloads():
        created = _parse_timestamp(payload.get("created_at"))
        updated = _parse_timestamp(payload.get("updated_at")) or created

        for entity_type, field in COMPAT_TYPE_TO_FIELD.items():
            value = payload.get(field)
            if not value:
                continue
            bucket = buckets[(entity_type, str(value))]
            bucket["total_memories"] += 1
            if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                bucket["created_at"] = created
            if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                bucket["updated_at"] = updated

    return [
        {
            "id": entity_id,
            "name": entity_id,
            "type": entity_type,
            "total_memories": data["total_memories"],
            "created_at": data["created_at"].isoformat() if data["created_at"] else None,
            "updated_at": data["updated_at"].isoformat() if data["updated_at"] else None,
            "owner": "self-hosted",
            "organization": "self-hosted",
            "metadata": data["metadata"],
        }
        for (entity_type, entity_id), data in sorted(
            buckets.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/v1/ping/", summary="Ping / validate API key")
def ping(_auth=Depends(verify_auth)):
    """Used by ``MemoryClient`` to validate the API key on initialisation."""
    user_email = getattr(_auth, "email", None) if _auth else None
    return {"message": "pong", "user_email": user_email}


# --- Add -------------------------------------------------------------------


@router.post("/v3/memories/add/", summary="Add memory (v3)")
def v3_add_memory(body: V3AddMemoryBody, _auth=Depends(verify_auth)):
    entity_params = _entity_params_from(
        body.filters,
        user_id=body.user_id,
        agent_id=body.agent_id,
        run_id=body.run_id,
        app_id=body.app_id,
    )
    params: Dict[str, Any] = {
        k: v
        for k, v in {
            **entity_params,
            "metadata": body.metadata,
            "infer": body.infer,
            "memory_type": body.memory_type,
            "prompt": body.prompt,
        }.items()
        if v is not None
    }
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    try:
        result = get_memory_instance().add(messages=body.messages, **params)
        return JSONResponse(content=result)
    except Exception:
        raise upstream_error()


# --- Get all ---------------------------------------------------------------


@router.post("/v3/memories/", summary="Get all memories (v3)")
def v3_get_all_memories(
    request: Request,
    body: V3GetAllBody,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    _auth=Depends(verify_auth),
):
    entity_params = _require_entity_scope(
        body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    try:
        raw = get_memory_instance().get_all(filters=entity_params)
        items = _normalize_results(raw)

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
    except Exception:
        raise upstream_error()


# --- Search ----------------------------------------------------------------


@router.post("/v3/memories/search/", summary="Search memories (v3)")
def v3_search_memories(body: V3SearchBody, _auth=Depends(verify_auth)):
    entity_params = _require_entity_scope(
        body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    params: Dict[str, Any] = {
        k: v
        for k, v in {
            **entity_params,
            "top_k": body.top_k,
            "threshold": body.threshold,
        }.items()
        if v is not None
    }
    try:
        result = get_memory_instance().search(query=body.query, **params)
        # Normalise to {"results": [...]}
        if isinstance(result, list):
            return {"results": result}
        return result
    except Exception:
        raise upstream_error()


# --- Single memory CRUD ----------------------------------------------------


@router.get("/v1/memories/{memory_id}/history/", summary="Get memory history (v1)")
def v1_memory_history(memory_id: str, _auth=Depends(verify_auth)):
    try:
        return get_memory_instance().history(memory_id=memory_id)
    except Exception:
        raise upstream_error()


@router.get("/v1/memories/{entity_type}/{entity_id}/", summary="Get memories for an entity (v1)")
def v1_get_entity_memories(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    try:
        raw = get_memory_instance().get_all(filters={_get_entity_field(entity_type): entity_id})
        return _normalize_results(raw)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@router.get("/v1/memories/{memory_id}/", summary="Get a memory (v1)")
def v1_get_memory(memory_id: str, _auth=Depends(verify_auth)):
    try:
        return get_memory_instance().get(memory_id)
    except Exception:
        raise upstream_error()


@router.put("/v1/memories/{memory_id}/", summary="Update a memory (v1)")
def v1_update_memory(memory_id: str, body: V1MemoryUpdate, _auth=Depends(verify_auth)):
    if not any([body.text, body.metadata]):
        raise HTTPException(
            status_code=400,
            detail="At least one of text or metadata must be provided for update.",
        )
    try:
        return get_memory_instance().update(
            memory_id=memory_id,
            data=body.text,
            metadata=body.metadata,
        )
    except Exception:
        raise upstream_error()


@router.delete(
    "/v1/memories/{memory_id}/",
    summary="Delete a memory (v1)",
    status_code=204,
)
def v1_delete_memory(memory_id: str, _auth=Depends(verify_auth)):
    try:
        get_memory_instance().delete(memory_id=memory_id)
        return Response(status_code=204)
    except Exception:
        raise upstream_error()


# --- Delete all ------------------------------------------------------------


@router.delete("/v1/memories/", summary="Delete all memories (v1)", status_code=204)
def v1_delete_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    # Fallback: client may serialise filters dict as JSON string in query
    filters: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    # Attempt to expand entity params from a JSON-encoded ``filters`` query param
    if filters and not any([user_id, agent_id, run_id]):
        try:
            filters_dict = json.loads(filters)
            user_id = filters_dict.get("user_id")
            agent_id = filters_dict.get("agent_id")
            app_id = filters_dict.get("app_id")
            run_id = filters_dict.get("run_id")
        except (json.JSONDecodeError, AttributeError):
            pass

    if not any([user_id, agent_id, app_id, run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, app_id, run_id) is required.",
        )

    params = {
        k: v
        for k, v in {"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id}.items()
        if v is not None
    }
    try:
        get_memory_instance().delete_all(**params)
        return Response(status_code=204)
    except Exception:
        raise upstream_error()


@router.get("/v1/memories/", summary="Get all memories (v1)")
def v1_list_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    _auth=Depends(verify_auth),
):
    filters = {k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "app_id": app_id, "run_id": run_id}.items() if v is not None}
    try:
        raw = get_memory_instance().get_all(filters=filters) if filters else list_all_memories()
        return _normalize_results(raw)
    except Exception:
        raise upstream_error()


@router.post("/v1/memories/", summary="Add memories (v1)")
def v1_add_memories(body: V3AddMemoryBody, _auth=Depends(verify_auth)):
    entity_params = _entity_params_from(
        body.filters,
        user_id=body.user_id,
        agent_id=body.agent_id,
        run_id=body.run_id,
        app_id=body.app_id,
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    params = {
        k: v
        for k, v in {**entity_params, "metadata": body.metadata, "infer": body.infer, "prompt": body.prompt}.items()
        if v is not None
    }
    try:
        result = get_memory_instance().add(messages=body.messages, **params)
        return _normalize_results(result)
    except Exception:
        raise upstream_error()


@router.post("/v2/memories/", summary="Get all memories (v2)")
def v2_list_memories(body: V3GetAllBody, _auth=Depends(verify_auth)):
    entity_params = _require_entity_scope(
        body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    try:
        raw = get_memory_instance().get_all(filters=entity_params)
        return _normalize_results(raw)
    except Exception:
        raise upstream_error()


@router.post("/v1/memories/search/", summary="Search memories (v1)")
def v1_search_memories(body: V3AddMemoryBody | V3SearchBody, _auth=Depends(verify_auth)):
    query = getattr(body, "query", None)
    if not query:
        raise HTTPException(status_code=400, detail="query is required.")
    entity_params = _entity_params_from(
        getattr(body, "filters", None),
        user_id=getattr(body, "user_id", None),
        agent_id=getattr(body, "agent_id", None),
        run_id=getattr(body, "run_id", None),
        app_id=getattr(body, "app_id", None),
    )
    if not entity_params:
        raise HTTPException(status_code=400, detail="At least one entity ID is required.")
    params = {
        k: v
        for k, v in {
            **entity_params,
            "top_k": getattr(body, "top_k", None),
            "threshold": getattr(body, "threshold", None),
        }.items()
        if v is not None
    }
    try:
        result = get_memory_instance().search(query=query, **params)
        return _normalize_results(result)
    except Exception:
        raise upstream_error()


@router.post("/v2/memories/search/", summary="Search memories (v2)")
def v2_search_memories(body: V3SearchBody, _auth=Depends(verify_auth)):
    entity_params = _require_entity_scope(
        body.filters,
        detail="filters must include at least one entity ID (user_id, agent_id, app_id, or run_id).",
    )
    params = {k: v for k, v in {**entity_params, "top_k": body.top_k, "threshold": body.threshold}.items() if v is not None}
    try:
        result = get_memory_instance().search(query=body.query, **params)
        return _normalize_results(result)
    except Exception:
        raise upstream_error()


# --- Entities --------------------------------------------------------------


@router.get("/v1/entities/", summary="List entities (v1)")
def v1_list_entities(_auth=Depends(verify_auth)):
    """Return entities in the SDK-compatible envelope while preserving spec fields.

    The hosted spec documents an array response, but ``MemoryClient.users()`` and
    ``delete_users()`` read ``response["results"]``. Keep that envelope here and
    include the spec fields on each entity item.
    """
    results = _list_entities_payload()
    return {"count": len(results), "results": results}


@router.get("/v1/entities/filters/", summary="List supported entity filters (v1)")
def v1_list_entity_filters(_auth=Depends(verify_auth)):
    return {"results": sorted(_ENTITY_PARAMS)}


@router.delete(
    "/v2/entities/{entity_type}/{entity_id}/",
    summary="Delete entity (v2)",
    status_code=204,
)
def v2_delete_entity(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    try:
        get_memory_instance().delete_all(**{_get_entity_field(entity_type): entity_id})
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()
    return Response(status_code=204)


@router.get("/v2/entities/{entity_type}/{entity_id}/", summary="Get entity details (v2)")
def v2_get_entity(entity_type: str, entity_id: str, _auth=Depends(verify_auth)):
    field = _get_entity_field(entity_type)
    for entity in _list_entities_payload():
        if entity["type"] == entity_type and entity["id"] == entity_id:
            return entity
    try:
        raw = get_memory_instance().get_all(filters={field: entity_id})
        memories = _normalize_results(raw)
        if memories:
            return {
                "id": entity_id,
                "name": entity_id,
                "type": entity_type,
                "total_memories": len(memories),
                "created_at": None,
                "updated_at": None,
                "owner": "self-hosted",
                "organization": "self-hosted",
                "metadata": {},
            }
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()
    raise HTTPException(status_code=404, detail="Entity not found.")


# --- Batch operations ------------------------------------------------------


@router.put("/v1/batch/", summary="Batch update memories (v1)")
def v1_batch_update(body: BatchUpdateBody, _auth=Depends(verify_auth)):
    if len(body.memories) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be updated in a single request")
    mem = get_memory_instance()
    for item in body.memories:
        try:
            mem.update(memory_id=item.memory_id, data=item.text, metadata=item.metadata)
        except Exception:
            raise upstream_error()
    return {"message": f"Successfully updated {len(body.memories)} memories"}


@router.delete("/v1/batch/", summary="Batch delete memories (v1)")
def v1_batch_delete(
    body: BatchDeleteBody | BatchDeleteIdsBody,
    _auth=Depends(verify_auth),
):
    mem = get_memory_instance()
    memory_ids = body.memory_ids if isinstance(body, BatchDeleteIdsBody) else [item.memory_id for item in body.memories]
    if len(memory_ids) > 1000:
        raise HTTPException(status_code=400, detail="Maximum of 1000 memories can be deleted in a single request")
    for memory_id in memory_ids:
        try:
            mem.delete(memory_id=memory_id)
        except Exception:
            raise upstream_error()
    return {"message": f"Successfully deleted {len(memory_ids)} memories"}
