import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, AsyncIterator, Iterator, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import Field

from auth import verify_auth
from compat.entities import list_entities_payload
from compat.events import (
    create_pending_add_event,
    event_access_allowed,
    event_cache_all,
    event_cache_get,
    events_visible_to_caller,
    resolve_event_owner_id,
)
from compat.helpers import UNSET, build_search_kwargs, normalize_results, normalize_results_dict
from compat.requests import request_meta
from compat.responses import (
    pending_add_response,
    resolve_optional_pagination,
    sync_add_response,
)
from compat.scope import build_search_filters, collect_direct_entity_params, require_entity_scope
from compat.tasks import run_v3_add_memory_task
from memory_lock import run_memory_write, run_memory_write_for_memory_id
from server_state import get_memory_instance

logger = logging.getLogger("mem0.server.mcp")

auth_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_user_id", default=None)
platform_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_platform", default=None)
mem0_source_var: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_mem0_source", default="MCP")

# Background pool for infer=True adds only (LLM extraction can take seconds).
_ADD_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mcp-add-memory")
_EXPIRATION_DATE_DEFAULT = Field(default_factory=lambda: UNSET)

mcp = FastMCP("mem0", json_response=True, stateless_http=True)
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Endpoints"])

_MCP_LIFESPAN_INSTALLED_STATE = "mem0_mcp_lifespan_installed"
_MCP_SESSION_MANAGER_STATE = "mem0_mcp_session_manager"


def _mcp_auth_user_id() -> str | None:
    """Authenticated user id from the current MCP request context."""
    return auth_user_id_var.get()


def _new_mcp_session_manager() -> StreamableHTTPSessionManager:
    return StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=True,
        stateless=True,
    )


def _mcp_session_manager(app: FastAPI) -> StreamableHTTPSessionManager:
    return getattr(app.state, _MCP_SESSION_MANAGER_STATE)


@mcp.tool(
    description=(
        "Store a new preference, fact, or conversation snippet. "
        "Requires at least one: user_id, agent_id, app_id or run_id. "
        "When infer=False, returns memory results immediately. "
        "When infer=True or omitted, returns event_id — poll get_event_status until SUCCEEDED."
    )
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
        Optional[str],
        Field(default=None, description="Optional expiration date in YYYY-MM-DD format. After this date, memories are hidden from search and get_all unless show_expired is True."),
    ] = None,
) -> dict[str, Any]:
    scope = require_entity_scope(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        fallback_user_id=_mcp_auth_user_id(),
    )
    if messages is None and text is None:
        raise ValueError("Provide either text or messages before calling add_memory.")
    conversation = messages if messages is not None else [{"role": "user", "content": text}]
    add_kwargs: dict[str, Any] = {**scope}

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
            scope,
        )
        return sync_add_response(raw)

    if infer is True:
        add_kwargs["infer"] = True

    event_id = create_pending_add_event(_mcp_auth_user_id() or resolve_event_owner_id(None, scope))
    _ADD_EXECUTOR.submit(run_v3_add_memory_task, event_id, conversation, add_kwargs)
    return pending_add_response(event_id)


@mcp.tool(
    description="""Run a semantic search over existing memories.

Use filters to narrow results. Common filter patterns:
- Single user: {"AND": [{"user_id": "john"}]}
- Agent memories: {"AND": [{"agent_id": "agent_name"}]}
- Recent memories: {"AND": [{"user_id": "john"}, {"created_at": {"gte": "2024-01-01"}}]}

user_id is automatically added to filters if not provided."""
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
        Optional[float], Field(default=None, description="Minimum similarity score (0.0-1.0). Default 0.1; pass 0.0 to disable filtering.")
    ] = None,
    rerank: Annotated[
        Optional[bool],
        Field(default=None, description="Re-rank results for better relevance (adds 200-400ms latency). Default false."),
    ] = None,
    show_expired: Annotated[
        Optional[bool],
        Field(default=None, description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default."),
    ] = None,
) -> dict[str, Any]:
    scoped_filters = build_search_filters(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
        fallback_user_id=_mcp_auth_user_id(),
    )

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
user_id is automatically added to filters if not provided."""
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
    show_expired: Annotated[
        Optional[bool],
        Field(default=None, description="When true, include memories whose expiration_date has passed. Expired memories are hidden by default."),
    ] = None,
) -> dict[str, Any]:
    scoped_filters = build_search_filters(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
        fallback_user_id=_mcp_auth_user_id(),
    )

    get_all_kwargs: dict[str, Any] = {"filters": scoped_filters}
    if show_expired is not None:
        get_all_kwargs["show_expired"] = show_expired

    raw = get_memory_instance().get_all(**get_all_kwargs)
    items = normalize_results(raw)
    clamped_page = max(1, page)
    clamped_page_size = min(max(1, page_size), 100)
    start = (clamped_page - 1) * clamped_page_size
    return {
        "count": len(items),
        "results": items[start : start + clamped_page_size],
    }


@mcp.tool(description="Fetch a single memory by ID.")
def get_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
) -> dict[str, Any]:
    result = get_memory_instance().get(memory_id)
    if result is None:
        raise ValueError(f"Memory '{memory_id}' not found.")
    return result


@mcp.tool(description="Update an existing memory's text and/or metadata.")
def update_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to update.")],
    text: Annotated[
        Optional[str], Field(default=None, description="Replacement text for the memory.")
    ] = None,
    metadata: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Metadata to merge into the memory.")
    ] = None,
    expiration_date: Annotated[
        Optional[str],
        Field(description="Optional expiration date in YYYY-MM-DD format, or null to clear."),
    ] = _EXPIRATION_DATE_DEFAULT,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
) -> dict[str, Any]:
    expiration_date_omitted = expiration_date is UNSET
    if text is None and metadata is None and source is None and expiration_date_omitted:
        raise ValueError("Provide text, metadata, source or expiration_date.")

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


@mcp.tool(description="Delete one memory after the user confirms its memory_id.")
def delete_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
) -> dict[str, Any]:
    return run_memory_write_for_memory_id(lambda memory: memory.delete(memory_id), memory_id)


@mcp.tool(description="Delete every memory in the given user/agent/app/run scope but keep the entity.")
def delete_all_memories(
    user_id: Annotated[
        Optional[str], Field(default=None, description="User scope to delete; defaults to server user.")
    ] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent scope to delete.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Optional app scope to delete.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Optional run scope to delete.")] = None,
) -> dict[str, Any]:
    scope = require_entity_scope(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        fallback_user_id=_mcp_auth_user_id(),
    )

    return run_memory_write(lambda memory: memory.delete_all(**scope), scope)


@mcp.tool(description="Remove an entity and cascade-delete its memories.")
def delete_entities(
    user_id: Annotated[Optional[str], Field(default=None, description="Delete this user and its memories.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Delete this agent and its memories.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Delete this app and its memories.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Delete this run and its memories.")] = None,
) -> dict[str, Any]:
    selected = list(collect_direct_entity_params(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id).items())
    if not selected:
        raise ValueError("Provide user_id, agent_id, app_id or run_id before calling delete_entities.")
    for key, value in selected:
        run_memory_write(lambda memory, k=key, v=value: memory.delete_all(**{k: v}), {key: value})
    return {"message": f"Entities deleted successfully, count: {len(selected)}."}


@mcp.tool(description="List which users/agents/apps/runs currently hold memories.")
def list_entities() -> dict[str, Any]:
    results = list_entities_payload()
    return {"count": len(results), "results": [item.model_dump() for item in results]}


@mcp.tool(description="List memory operation events with optional filters and pagination.")
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


@mcp.tool(description="Check the status of a specific memory operation event by its ID.")
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


async def _run_streamable_transport(request: Request) -> Response:
    response_started = False
    response_status = 200
    response_headers: list[tuple[bytes, bytes]] = []
    response_body = bytearray()

    async def capture_send(message):
        nonlocal response_started, response_status
        if message["type"] == "http.response.start":
            response_started = True
            response_status = message["status"]
            response_headers.extend(message.get("headers", []))
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    try:
        await _mcp_session_manager(request.app).handle_request(request.scope, request.receive, capture_send)
    except Exception:
        logger.exception("MCP streamable transport error")
        return Response(status_code=500, content=b"Internal MCP transport error")

    if not response_started:
        return Response(status_code=500, content=b"Transport did not produce a response")

    response = Response(
        content=bytes(response_body),
        status_code=response_status,
    )
    response.raw_headers = response_headers
    return response


def _install_mcp_lifespan(app: FastAPI) -> None:
    if getattr(app.state, _MCP_LIFESPAN_INSTALLED_STATE, False):
        return

    session_manager = _new_mcp_session_manager()
    setattr(app.state, _MCP_SESSION_MANAGER_STATE, session_manager)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_mcp(app: FastAPI) -> AsyncIterator[None]:
        async with original_lifespan(app):
            async with session_manager.run():
                yield

    app.router.lifespan_context = lifespan_with_mcp
    setattr(app.state, _MCP_LIFESPAN_INSTALLED_STATE, True)


@contextmanager
def _mcp_request_context(request: Request, user: Any) -> Iterator[None]:
    auth_token = auth_user_id_var.set(str(user.id) if user is not None else None)
    meta = request_meta(request)
    platform_token = platform_var.set(meta.platform or meta.ua_tool_name)
    source_token = mem0_source_var.set(meta.source or "MCP")

    try:
        yield
    finally:
        auth_user_id_var.reset(auth_token)
        platform_var.reset(platform_token)
        mem0_source_var.reset(source_token)


@mcp_router.api_route(
    "/", methods=["GET", "POST", "DELETE"], include_in_schema=False, operation_id="handle_streamable_http_slash"
)
@mcp_router.api_route("", methods=["GET", "POST", "DELETE"], summary="MCP Endpoint")
async def handle_streamable_http(request: Request, user=Depends(verify_auth)):
    with _mcp_request_context(request, user):
        return await _run_streamable_transport(request)


def setup_mcp_server(app: FastAPI) -> None:
    _install_mcp_lifespan(app)
    app.include_router(mcp_router)
