import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from typing import Annotated, Any, AsyncIterator, Callable, Iterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.types import Receive, Scope, Send

from auth import determine_user, is_bootstrap_admin, verify_auth
from compat.entities import CompatEntity
from entity import FIELD_TO_TYPE, is_scoped_entity_type
from compat.events import (
    create_pending_add_event,
    event_access_allowed,
    event_cache_all,
    event_cache_get,
    events_visible_to_caller,
    resolve_event_owner_id,
)
from compat.helpers import build_search_kwargs, normalize_results, normalize_results_dict
from utils.helpers import safe_count
from compat.requests import request_meta
from compat.responses import (
    pending_add_response,
    resolve_optional_pagination,
    sync_add_response,
)
from compat.scope import collect_direct_entity_params
from compat.tasks import run_v3_add_memory_task
from memory_lock import run_memory_write, run_memory_write_for_memory_id
from server_state import get_memory_instance
import server_state
from entity_permissions import (
    bulk_delete_memories,
    collect_user_children,
    get_entity_or_none,
    authorize_write,
    check_entity_delete_permission,
    check_memory_scope_permission,
    check_query_permission,
    entity_filter_params,
    get_visible_entities,
    inject_default_user_id,
    strip_user_id_for_app_gate,
    list_memory_ids_for_params,
    reject_bootstrap_memory_mutation,
    resolve_memory_entities,
    validate_bulk_admin_operation,
)

logger = logging.getLogger("mem0.server.mcp")

auth_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_user_id", default=None)
platform_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_platform", default=None)
mem0_source_var: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_mem0_source", default="MCP")
# Full operator object + auth_type, set in _mcp_request_context, so tools (which
# bypass FastAPI DI) can resolve (operator, is_admin) without the Request/User args.
mcp_user_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar("mcp_user_object", default=None)
mcp_auth_type_var: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_auth_type", default="none")

# Background pool for infer=True adds only (LLM extraction can take seconds).
_ADD_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mcp-add-memory")

# Sentinel distinguishing "expiration_date omitted" (preserve existing) from an
# explicit None (clear) on the MCP tool's flat function arg. FastMCP does not
# expose "was this arg provided", so a non-None default is the only way to tell.
# Wrapped in Field(default_factory=...) so Pydantic does not try to JSON-serialize
# the sentinel object when generating the tool's input schema (which a bare
# ``= _UNSET`` default would trigger a non-serializable-default warning for).
_UNSET = object()
_EXPIRATION_DATE_DEFAULT = Field(default_factory=lambda: _UNSET)

mcp = FastMCP("mem0", json_response=True, stateless_http=True)
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Endpoints"])

_MCP_LIFESPAN_INSTALLED_STATE = "mem0_mcp_lifespan_installed"
_MCP_SESSION_MANAGER_STATE = "mem0_mcp_session_manager"


def _mcp_auth_user_id() -> str | None:
    """Authenticated user id from the current MCP request context."""
    return auth_user_id_var.get()


@contextmanager
def _mcp_db() -> Iterator[Any]:
    """Open a DB session from the server's session factory (MCP tools bypass FastAPI DI)."""
    factory = server_state._session_factory
    if factory is None:
        raise RuntimeError("DB session factory not initialized.")
    session = factory()
    try:
        yield session
    finally:
        session.close()


def _mcp_resolve_operator(db: Any) -> tuple[Any, bool]:
    """Resolve the acting MCP operator from per-request contextvars.

    Delegates to the shared core in entity_permissions; only the exception type
    differs (ValueError for MCP vs HTTPException for REST).
    """
    result = determine_user(mcp_user_var.get(), mcp_auth_type_var.get(), db)
    if result is None:
        raise ValueError("Authentication required.")
    return result


def _mcp_raise(check: Callable[[], Any]) -> Any:
    """Run a permission check, converting HTTPException to ValueError for MCP tools.

    MCP tools use ValueError for client-visible errors (FastMCP surfaces it as an
    isError response); the service layer raises HTTPException, so adapt here.
    Returns the check's result so callers can capture rewritten filters, etc.
    """
    try:
        return check()
    except HTTPException as exc:
        raise ValueError(exc.detail) from exc


def _mcp_resolve_memory_entities(memory_id: str) -> dict[str, str]:
    try:
        return resolve_memory_entities(memory_id)
    except HTTPException as exc:
        raise ValueError(exc.detail) from exc


@mcp.tool(
    description=(
        "Store a new preference, fact, or conversation snippet. "
        "Requires at least one: user_id, agent_id, app_id or run_id. "
        "When infer=False, returns memory results immediately. "
        "When infer=True or omitted, returns event_id — poll get_event_status until SUCCEEDED."
    ),
    annotations=ToolAnnotations(readOnlyHint=False),
)
def add_memory(
    text: Annotated[
        Optional[str],
        Field(default=None, description="Plain sentence summarizing what to store. Required when messages is omitted."),
    ] = None,
    messages: Annotated[
        Optional[list[dict[str, str]]],
        Field(
            default=None,
            description="Structured conversation history with `role`/`content`. Use when you have multiple turns.",
        ),
    ] = None,
    user_id: Annotated[
        Optional[str], Field(default=None, description="Override the default user scope for this write.")
    ] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent identifier.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Optional app identifier.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Optional run identifier.")] = None,
    infer: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Extract structured memories via LLM (default true). Set false to store raw text without extraction.",
        ),
    ] = None,
    metadata: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Attach arbitrary metadata JSON to the memory.")
    ] = None,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
    expiration_date: Annotated[
        Optional[date],
        Field(
            default=None,
            description="Optional expiration date in YYYY-MM-DD format. After this date, memories are hidden from search and get_all unless show_expired is True.",
        ),
    ] = None,
) -> dict[str, Any]:
    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        _mcp_raise(lambda: reject_bootstrap_memory_mutation(operator))
        entity_params = collect_direct_entity_params(
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )
        entity_params = inject_default_user_id(entity_params, operator)
        if not entity_params:
            raise ValueError("One of the filters: user_id, agent_id, app_id or run_id is required!")
        # Permission + ownership committed synchronously before any async dispatch.
        _mcp_raise(lambda: authorize_write(entity_params, operator, db))
    if messages is None and text is None:
        raise ValueError("Provide either text or messages before calling add_memory.")
    conversation = messages if messages is not None else [{"role": "user", "content": text}]
    add_kwargs: dict[str, Any] = {**entity_params}

    # Three-layer metadata precedence, matching REST merge_v3_add_metadata:
    # request-header source/platform (x-mem0-*) fill missing keys, caller
    # metadata is preserved, and an explicit top-level `source` arg wins.
    merged_metadata: dict[str, Any] = dict(metadata or {})
    if header_source := mem0_source_var.get():
        merged_metadata.setdefault("source", header_source)
    if platform := platform_var.get():
        merged_metadata.setdefault("platform", platform)
    if source is not None:
        merged_metadata["source"] = source
    add_kwargs["metadata"] = merged_metadata

    if expiration_date is not None:
        add_kwargs["expiration_date"] = expiration_date

    # Hybrid write path (same semantics as REST POST /v3/memories/add/):
    #   infer=False — fast path: no LLM, sync response with memory ids.
    #   infer=True  — slow path: extraction + dedup, async + event_id.
    if infer is False:
        add_kwargs["infer"] = False
        raw = run_memory_write(
            lambda memory: memory.add(messages=conversation, **add_kwargs),
            entity_params,
        )
        return sync_add_response(raw)

    if infer is True:
        add_kwargs["infer"] = True

    event_id = create_pending_add_event(_mcp_auth_user_id() or resolve_event_owner_id(None, entity_params))
    _ADD_EXECUTOR.submit(run_v3_add_memory_task, event_id, conversation, add_kwargs)
    return pending_add_response(event_id)


@mcp.tool(
    description="""Run a semantic search over existing memories.

Use filters to narrow results. Common filter patterns:
- Single user: {"AND": [{"user_id": "john"}]}
- Agent memories: {"AND": [{"agent_id": "agent_name"}]}
- Recent memories: {"AND": [{"user_id": "john"}, {"created_at": {"gte": "2024-01-01"}}]}

user_id is automatically added to filters if not provided.""",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def search_memories(
    query: Annotated[str, Field(description="Natural language description of what to find.")],
    user_id: Annotated[Optional[str], Field(default=None, description="Limit search to this user's memories.")] = None,
    agent_id: Annotated[
        Optional[str], Field(default=None, description="Limit search to this agent's memories.")
    ] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Limit search to this app's memories.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Limit search to this run's memories.")] = None,
    filters: Annotated[
        Optional[dict[str, Any]],
        Field(default=None, description="Additional filter clauses (user_id injected automatically)."),
    ] = None,
    top_k: Annotated[
        Optional[int], Field(default=None, description="Number of results to return (1-1000, default 10).")
    ] = None,
    threshold: Annotated[
        Optional[float],
        Field(
            default=None, description="Minimum similarity score (0.0-1.0). Default 0.1; pass 0.0 to disable filtering."
        ),
    ] = None,
    rerank: Annotated[
        Optional[bool],
        Field(
            default=None, description="Re-rank results for better relevance (adds 200-400ms latency). Default false."
        ),
    ] = None,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
    show_expired: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
        ),
    ] = None,
) -> dict[str, Any]:
    # `source` is accepted for parity with the platform MCP. The self-hosted
    # event model only tracks ADD events (see compat.events), so read/delete
    # paths have no SEARCH/GET_ALL/DELETE_ALL event to tag — record intent only.
    # When both `user_id` and `app_id` are present, `user_id` is stripped (app-primary-gate).
    if source is not None:
        logger.debug("search_memories: source=%s (advisory; self-hosted tracks ADD events only)", source)

    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        entity_params = collect_direct_entity_params(
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
            filters=filters,
        )
        if not entity_params:
            if is_bootstrap_admin(operator):
                raise ValueError(
                    "admin_api_key requires an explicit scope (user_id, agent_id, app_id, or run_id)."
                )
            entity_params = {"user_id": str(operator.id)}
        scoped_filters: dict[str, Any] = dict(filters) if filters else {}
        scoped_filters.update(entity_params)
        scoped_filters = strip_user_id_for_app_gate(scoped_filters)
        scoped_filters = _mcp_raise(lambda: check_query_permission(scoped_filters, operator.id, db))

    raw = get_memory_instance().search(
        query=query,
        **build_search_kwargs(scoped_filters, top_k, threshold, rerank, show_expired),
    )
    return normalize_results_dict(raw)


@mcp.tool(
    description="""Page through memories using filters instead of search.

Use filters to list specific memories. Common filter patterns:
- Single user: {"AND": [{"user_id": "john"}]}
- Agent memories: {"AND": [{"agent_id": "agent_name"}]}

Pagination: Use page (1-indexed) and page_size for browsing results.
user_id is automatically added to filters if not provided. When both
user_id and app_id are present, user_id is stripped (app-primary-gate
— see ``strip_user_id_for_app_gate``).""",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def get_memories(
    user_id: Annotated[Optional[str], Field(default=None, description="List memories for this user.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="List memories for this agent.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="List memories for this app.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="List memories for this run.")] = None,
    filters: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Structured filters; user_id injected automatically.")
    ] = None,
    page: Annotated[int, Field(default=1, ge=1, description="1-indexed page number (default 1).")] = 1,
    page_size: Annotated[
        int, Field(default=10, ge=1, le=100, description="Number of memories per page (default 10, max 100).")
    ] = 10,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
    show_expired: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default.",
        ),
    ] = None,
) -> dict[str, Any]:
    if source is not None:
        logger.debug("get_memories: source=%s (advisory; self-hosted tracks ADD events only)", source)

    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        entity_params = collect_direct_entity_params(
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
            filters=filters,
        )
        if not entity_params:
            if is_bootstrap_admin(operator):
                raise ValueError(
                    "admin_api_key requires an explicit scope (user_id, agent_id, app_id, or run_id)."
                )
            entity_params = {"user_id": str(operator.id)}
        scoped_filters: dict[str, Any] = dict(filters) if filters else {}
        scoped_filters.update(entity_params)
        scoped_filters = strip_user_id_for_app_gate(scoped_filters)
        scoped_filters = _mcp_raise(lambda: check_query_permission(scoped_filters, operator.id, db))

    get_all_kwargs: dict[str, Any] = {"filters": scoped_filters}
    if show_expired is not None:
        get_all_kwargs["show_expired"] = show_expired

    memory = get_memory_instance()
    clamped_page = max(1, page)
    clamped_page_size = min(max(1, page_size), 100)
    start = (clamped_page - 1) * clamped_page_size
    get_all_kwargs["top_k"] = clamped_page_size
    get_all_kwargs["skip"] = start
    raw = memory.get_all(**get_all_kwargs)
    results = normalize_results(raw)
    # count() is advisory and may be unsupported/ignored; fall back to a scanned
    # lower bound (exact on the final page). MCP has no next link, so clients
    # detect the end via len(results) < page_size.
    c = safe_count(memory, get_all_kwargs["filters"])
    total = c if c is not None else start + len(results)
    return {"count": total, "results": results}


@mcp.tool(
    description="Fetch a single memory by ID.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def get_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
) -> dict[str, Any]:
    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        scope = _mcp_resolve_memory_entities(memory_id)
        _mcp_raise(lambda: check_memory_scope_permission(scope, operator.id, "read", db))
    result = get_memory_instance().get(memory_id)
    if result is None:
        raise ValueError(f"Memory '{memory_id}' not found.")
    return result


@mcp.tool(
    description="Update an existing memory's text and/or metadata.",
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True),
)
def update_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to update.")],
    text: Annotated[Optional[str], Field(default=None, description="Replacement text for the memory.")] = None,
    metadata: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Metadata to merge into the memory.")
    ] = None,
    expiration_date: Annotated[
        Optional[date],
        Field(description="Optional expiration date in YYYY-MM-DD format, or null to clear."),
    ] = _EXPIRATION_DATE_DEFAULT,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
) -> dict[str, Any]:
    expiration_date_omitted = expiration_date is _UNSET
    if text is None and metadata is None and source is None and expiration_date_omitted:
        raise ValueError("Provide text, metadata, source or expiration_date.")

    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        _mcp_raise(lambda: reject_bootstrap_memory_mutation(operator))
        scope = _mcp_resolve_memory_entities(memory_id)
        _mcp_raise(lambda: check_memory_scope_permission(scope, operator.id, "write", db))

    update_kwargs: dict[str, Any] = {"memory_id": memory_id}
    if text is not None:
        update_kwargs["data"] = text
    if metadata is not None or source is not None:
        # Caller metadata is preserved; an explicit top-level `source` arg wins
        # over a same-named key in the metadata bag. (Unlike add_memory, update
        # has no request-header source layer.)
        merged_metadata: dict[str, Any] = dict(metadata or {})
        if source is not None:
            merged_metadata["source"] = source
        update_kwargs["metadata"] = merged_metadata
    if not expiration_date_omitted:
        update_kwargs["expiration_date"] = expiration_date

    return run_memory_write_for_memory_id(lambda memory: memory.update(**update_kwargs), memory_id)


@mcp.tool(
    description="Delete one memory after the user confirms its memory_id.",
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True),
)
def delete_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
) -> dict[str, Any]:
    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        _mcp_raise(lambda: reject_bootstrap_memory_mutation(operator))
        scope = _mcp_resolve_memory_entities(memory_id)
        _mcp_raise(lambda: check_memory_scope_permission(scope, operator.id, "write", db))
    result = run_memory_write_for_memory_id(lambda memory: memory.delete(memory_id), memory_id)
    return result


@mcp.tool(
    description="Delete every memory in the given user/agent/app/run scope but keep the entity.",
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True),
)
def delete_all_memories(
    user_id: Annotated[
        Optional[str], Field(default=None, description="User scope to delete; defaults to server user.")
    ] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent scope to delete.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Optional app scope to delete.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Optional run scope to delete.")] = None,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
) -> dict[str, Any]:
    if source is not None:
        logger.debug("delete_all_memories: source=%s (advisory; self-hosted tracks ADD events only)", source)

    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        _mcp_raise(lambda: reject_bootstrap_memory_mutation(operator))
        entity_params = collect_direct_entity_params(
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )
        if not entity_params:
            raise ValueError("One of the filters: user_id, agent_id, app_id or run_id is required!")
        # Scope to caller's user namespace for agent/run-only deletes (skip for
        # app-scoped deletes — injecting user_id would narrow the delete scope).
        entity_params = inject_default_user_id(entity_params, operator, skip_if_has_app=True)
        operator_id = operator.id

    def _delete_all(memory):
        with _mcp_db() as db:
            _mcp_raise(lambda: bulk_delete_memories(memory, entity_params, operator_id, db))

    return run_memory_write(_delete_all, entity_params)


@mcp.tool(
    description="Remove an entity and cascade-delete its memories.",
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True),
)
def delete_entities(
    user_id: Annotated[Optional[str], Field(default=None, description="Delete this user and its memories.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Delete this agent and its memories.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Delete this app and its memories.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Delete this run and its memories.")] = None,
) -> dict[str, Any]:
    selected = list(
        collect_direct_entity_params(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id).items()
    )
    if not selected:
        raise ValueError("Provide user_id, agent_id, app_id or run_id before calling delete_entities.")
    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        # Client zone: no bypass. All operations are scoped to the caller's own
        # namespace. Bootstrap admin cannot delete entities here (parent_entity_id
        # resolves to None → check_entity_delete_permission returns 403/404).
        operator_id = operator.id
        # Owner-only delete check + child-safety check: must pass for ALL entities
        # before any deletion, so a denied entity/child aborts the whole call.
        # A granted admin cannot delete entities (design: delete = owner only).
        for key, value in selected:
            entity_type = FIELD_TO_TYPE[key]
            parent_entity_id = None
            if is_scoped_entity_type(entity_type):
                parent_entity_id = user_id or (str(operator_id) if not is_bootstrap_admin(operator_id) else None)
            entity = _mcp_raise(
                lambda et=entity_type, v=value, p=parent_entity_id: check_entity_delete_permission(
                    et, v, operator_id, False, db,
                    parent_entity_id=p,
                )
            )
            if entity is not None and entity.type == "user":
                if collect_user_children(entity, db):
                    raise ValueError(
                        f"Cannot delete 'user/{value}' because it has sub-namespaces or "
                        "agent/run entities. Delete them first."
                    )
    # Client zone: user entities with children cannot be deleted — the
    # caller must delete sub-namespaces and agent/run children first.
    # Cascade-delete is handled by the management zone (DELETE /entities/...).
    # Bulk destructive: prescan + admin-validate + delete inside the scope
    # lock so a concurrent cross-scope write cannot sneak a memory in (TOCTOU).
    for key, value in selected:
        def _delete_one(memory, k=key, v=value):
            with _mcp_db() as db:
                entity_type = FIELD_TO_TYPE[k]
                parent_entity_id = None
                if is_scoped_entity_type(entity_type):
                    parent_entity_id = user_id or (str(operator_id) if not is_bootstrap_admin(operator_id) else None)
                entity = get_entity_or_none(
                    entity_type, v, db,
                    parent_entity_id=parent_entity_id,
                )
                # For agent/run: scope vector-store scan by parent user_id.
                delete_params: dict[str, Any] = entity_filter_params(entity, db) if entity is not None else {k: v}
                memory_ids = _mcp_raise(lambda: list_memory_ids_for_params(delete_params))
                _mcp_raise(lambda: validate_bulk_admin_operation(memory_ids, operator_id, db))
            memory.delete_all(**delete_params)
        run_memory_write(_delete_one, {key: value})
    # Drop DB rows so namespaces are released for re-claim (cascade clears
    # permissions).
    with _mcp_db() as db:
        for key, value in selected:
            entity_type = FIELD_TO_TYPE[key]
            parent_entity_id = None
            if is_scoped_entity_type(entity_type):
                parent_entity_id = user_id or (str(operator_id) if not is_bootstrap_admin(operator_id) else None)
            entity = get_entity_or_none(
                entity_type, value, db,
                parent_entity_id=parent_entity_id,
            )
            if entity is not None:
                db.delete(entity)
        db.commit()
    return {"message": f"Entities deleted successfully, count: {len(selected)}."}


@mcp.tool(
    description="List which users/agents/apps/runs currently hold memories.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def list_entities() -> dict[str, Any]:
    with _mcp_db() as db:
        operator, _ = _mcp_resolve_operator(db)
        entities = get_visible_entities(operator.id, db)
        results = [
            CompatEntity.from_bucket(
                ent.type,
                ent.id,
                created_at=ent.created_at,
                updated_at=ent.updated_at,
                entity_name=ent.name,
            )
            for ent in sorted(entities, key=lambda e: (e.type, e.id))
        ]
    return {"count": len(results), "results": [item.model_dump() for item in results]}


@mcp.tool(
    description="List memory operation events with optional filters and pagination.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def list_events(
    event_type: Annotated[
        Optional[str],
        Field(default=None, description="Filter by type: ADD, SEARCH, UPDATE, DELETE, GET_ALL, DELETE_ALL."),
    ] = None,
    page: Annotated[Optional[int], Field(default=None, description="1-indexed page number.")] = None,
    page_size: Annotated[
        Optional[int], Field(default=None, description="Events per page (default 50, max 100).")
    ] = None,
) -> dict[str, Any]:
    items = events_visible_to_caller(event_cache_all(), _mcp_auth_user_id())
    if event_type:
        items = [item for item in items if item.get("event_type") == event_type]
    pagination = resolve_optional_pagination(page, page_size)
    if pagination:
        effective_page, effective_page_size = pagination
        start = (effective_page - 1) * effective_page_size
        page_items = items[start : start + effective_page_size]
        return {"count": len(items), "results": page_items}
    return {"count": len(items), "results": items}


@mcp.tool(
    description="Check the status of a specific memory operation event by its ID.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def get_event_status(
    event_id: Annotated[str, Field(description="UUID of the event to check.")],
) -> dict[str, Any]:
    obj = event_cache_get(event_id)
    if obj is None or not event_access_allowed(obj, _mcp_auth_user_id()):
        raise ValueError(f"Event '{event_id}' not found.")
    return obj


@mcp.prompt()
def memory_assistant() -> str:
    return """You are using the Mem0 MCP server (self-hosted) for long-term memory management.

Quick Start:
1. Store memories: Use add_memory to save facts, preferences, or conversations
2. Search memories: Use search_memories for semantic queries
3. List memories: Use get_memories for filtered browsing
4. Update/Delete: Use update_memory and delete_memory for modifications
5. List events: Use list_events to see recent memory operations
6. Check status: Use get_event_status to poll async operation status

Filter Examples:
- User memories: {"AND": [{"user_id": "john"}]}
- Agent memories: {"AND": [{"agent_id": "agent_name"}]}
- Recent only: {"AND": [{"user_id": "john"}, {"created_at": {"gte": "2024-01-01"}}]}

Search Defaults:
- threshold defaults to 0.1 (pass 0.0 to disable similarity filtering)
- rerank defaults to false (set true for better relevance, adds 200-400ms)

Tips:
- user_id is automatically added to filters
- Use "*" as wildcard for any non-null value
- Combine filters with AND/OR/NOT for complex queries
- Use infer=false in add_memory to skip LLM extraction and store raw text"""


class _McpStreamableResponse(Response):
    """Bridge the MCP session manager's ASGI app into a Starlette ``Response``.

    Starlette calls ``await response(scope, receive, send)`` *after* the route
    handler returns, with the middleware-chain-wrapped ``send`` (so CORS etc.
    apply to the transport's own ``http.response.start``). Per-request
    contextvars must therefore be (re)established here, inside ``__call__``,
    so they are live when the session manager dispatches the per-request
    server task — anyio copies the caller's context to task-group children, so
    tools invoked in that task see ``auth_user_id_var`` / ``platform_var`` /
    ``mem0_source_var``.

    Status code and headers come from the transport via the real ``send``;
    ``__call__`` never calls ``super()``, so the defaults set in ``__init__``
    are unused. This replaces a capture-and-rebuild buffer that defeated SSE
    streaming and poked at ``Response.raw_headers`` with a direct pass-through.
    """

    def __init__(
        self,
        request: Request,
        user: Any,
        session_manager: StreamableHTTPSessionManager,
    ) -> None:
        super().__init__(status_code=200)
        self._request = request
        self._user = user
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:  # type: ignore[override]
        with _mcp_request_context(self._request, self._user):
            try:
                await self._session_manager.handle_request(scope, receive, send)
            except Exception:
                logger.exception("MCP streamable transport error")
                raise


def _install_mcp_lifespan(app: FastAPI) -> None:
    if getattr(app.state, _MCP_LIFESPAN_INSTALLED_STATE, False):
        return

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_mcp(app: FastAPI) -> AsyncIterator[None]:
        async with original_lifespan(app):
            # Fresh session manager per lifespan cycle: run() can only be
            # called once per instance, so reusing one across cycles (TestClient
            # with-blocks, in-process reload) raises RuntimeError. Mirrors mcp's
            # ctor (json_response=True, stateless_http=True); stored on app.state
            # for the route to read at request time.
            session_manager = StreamableHTTPSessionManager(
                app=mcp._mcp_server,
                json_response=True,
                stateless=True,
            )
            setattr(app.state, _MCP_SESSION_MANAGER_STATE, session_manager)
            async with session_manager.run():
                yield

    app.router.lifespan_context = lifespan_with_mcp
    setattr(app.state, _MCP_LIFESPAN_INSTALLED_STATE, True)


@contextmanager
def _mcp_request_context(request: Request, user: Any) -> Iterator[None]:
    # Resolve header metadata before setting any contextvar so a failure here
    # cannot leave auth_user_id_var set without a matching reset.
    meta = request_meta(request)
    auth_token = auth_user_id_var.set(str(user.id) if user is not None else None)
    user_token = mcp_user_var.set(user)
    auth_type_token = mcp_auth_type_var.set(getattr(request.state, "auth_type", "none"))
    platform_token = platform_var.set(meta.platform or meta.ua_tool_name)
    source_token = mem0_source_var.set(meta.source or "MCP")

    try:
        yield
    finally:
        auth_user_id_var.reset(auth_token)
        mcp_user_var.reset(user_token)
        mcp_auth_type_var.reset(auth_type_token)
        platform_var.reset(platform_token)
        mem0_source_var.reset(source_token)


@mcp_router.api_route("/", methods=["GET", "POST", "DELETE"], include_in_schema=False)
@mcp_router.api_route("", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def handle_streamable_http(request: Request, user=Depends(verify_auth)):
    return _McpStreamableResponse(request, user, getattr(request.app.state, _MCP_SESSION_MANAGER_STATE))


def setup_mcp_server(app: FastAPI) -> None:
    _install_mcp_lifespan(app)
    app.include_router(mcp_router)
