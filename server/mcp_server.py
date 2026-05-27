import contextvars
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport
from pydantic import Field

from auth import verify_auth
from compat.entities import list_entities_payload
from compat.events import event_cache_all, event_cache_get, event_cache_put, make_event_obj
from compat.requests import request_meta
from compat.responses import normalize_results, normalize_results_dict
from compat.scope import build_search_filters, collect_entity_params, require_entity_scope
from compat.tasks import run_v3_add_memory_task
from memory_lock import run_memory_write, run_memory_write_for_memory_id
from server_state import get_memory_instance

logger = logging.getLogger("mem0.server.mcp")

auth_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_user_id", default=None)
platform_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_platform", default=None)
mem0_source_var: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_mem0_source", default="MCP")

mcp = FastMCP("mem0")
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Endpoints"])


def _fallback_uid() -> str | None:
    return auth_user_id_var.get()


@mcp.tool(
    description=(
        "Store a new preference, fact, or conversation snippet. "
        "Requires at least one: user_id, agent_id, app_id or run_id. "
        "Returns an event_id for async polling via get_event_status."
    )
)
def add_memory(
    text: Annotated[str, Field(description="Plain sentence summarizing what to store.")],
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
        bool, Field(default=True, description="When False, store text verbatim without LLM fact extraction.")
    ] = True,
    metadata: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Attach arbitrary metadata JSON to the memory.")
    ] = None,
    source: Annotated[
        Optional[str],
        Field(default=None, description="Event source tag (defaults to MCP if omitted)."),
    ] = None,
) -> dict[str, Any]:
    scope = require_entity_scope(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        fallback_user_id=_fallback_uid(),
    )
    conversation = messages if messages is not None else [{"role": "user", "content": text}]
    add_kwargs: dict[str, Any] = {**scope}
    base_md: dict[str, Any] = {"source": source or mem0_source_var.get()}

    if platform := platform_var.get():
        base_md["platform"] = platform
    add_kwargs["metadata"] = {**base_md, **(metadata or {})}
    if not infer:
        add_kwargs["infer"] = False

    event_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    event_cache_put(
        event_id,
        make_event_obj(
            event_id,
            [],
            now_iso=now_iso,
            status="PENDING",
            completed_at=None,
            latency=None,
        ),
    )
    threading.Thread(
        target=run_v3_add_memory_task,
        args=(event_id, conversation, add_kwargs),
        daemon=True,
        name=f"mcp-add-memory-{event_id}",
    ).start()

    return {
        "message": "Memory processing has been queued for background execution.",
        "event_id": event_id,
        "status": "PENDING",
    }


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
        Optional[float], Field(default=None, description="Minimum semantic relevance score (0.0–1.0).")
    ] = None,
) -> dict[str, Any]:
    scoped_filters = build_search_filters(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
        fallback_user_id=_fallback_uid(),
    )
    search_kwargs: dict[str, Any] = {"filters": scoped_filters}
    if top_k is not None:
        search_kwargs["top_k"] = top_k
    if threshold is not None:
        search_kwargs["threshold"] = threshold

    raw = get_memory_instance().search(query=query, **search_kwargs)
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
    page: Annotated[Optional[int], Field(default=None, description="1-indexed page number when paginating.")] = None,
    page_size: Annotated[
        Optional[int], Field(default=None, description="Number of memories per page (default 10).")
    ] = None,
) -> dict[str, Any]:
    scoped_filters = build_search_filters(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
        fallback_user_id=_fallback_uid(),
    )

    raw = get_memory_instance().get_all(filters=scoped_filters)
    items = normalize_results(raw)
    if page and page_size:
        start = max(page - 1, 0) * page_size
        return {
            "count": len(items),
            "results": items[start : start + page_size],
        }
    return {"count": len(items), "results": items}


@mcp.tool(description="Fetch a single memory by ID.")
def get_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
) -> dict[str, Any]:
    result = get_memory_instance().get(memory_id)
    if result is None:
        raise ValueError(f"Memory '{memory_id}' not found.")
    return result


@mcp.tool(description="Overwrite an existing memory's text.")
def update_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to overwrite.")],
    text: Annotated[str, Field(description="Replacement text for the memory.")],
    metadata: Annotated[
        Optional[dict[str, Any]], Field(default=None, description="Optional metadata to attach to the updated memory.")
    ] = None,
) -> dict[str, Any]:
    update_kwargs: dict[str, Any] = {"memory_id": memory_id, "data": text}
    if metadata is not None:
        update_kwargs["metadata"] = metadata

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
        fallback_user_id=_fallback_uid(),
    )

    return run_memory_write(lambda memory: memory.delete_all(**scope), scope)


@mcp.tool(description="Remove an entity and cascade-delete its memories.")
def delete_entities(
    user_id: Annotated[Optional[str], Field(default=None, description="Delete this user and its memories.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Delete this agent and its memories.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Delete this app and its memories.")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Delete this run and its memories.")] = None,
) -> dict[str, Any]:
    selected = list(collect_entity_params(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id).items())
    if not selected:
        raise ValueError("Provide user_id, agent_id, app_id or run_id before calling delete_entities.")
    for key, value in selected:
        run_memory_write(lambda memory, k=key, v=value: memory.delete_all(**{k: v}), {key: value})
    return {"message": f"Entities deleted successfully, count: {len(selected)}."}


@mcp.tool(description="List which users/agents/apps/runs currently hold memories.")
def list_entities() -> dict[str, Any]:
    results = list_entities_payload()
    return {"count": len(results), "results": results}


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
    items = event_cache_all()
    if event_type:
        items = [item for item in items if item.get("event_type") == event_type]
    if page and page_size:
        start = max(page - 1, 0) * page_size
        page_items = items[start : start + page_size]
        return {"count": len(items), "results": page_items}
    return {"count": len(items), "results": items}


@mcp.tool(description="Check the status of a specific memory operation event by its ID.")
def get_event_status(
    event_id: Annotated[str, Field(description="UUID of the event to check.")],
) -> dict[str, Any]:
    obj = event_cache_get(event_id)
    if obj is None:
        raise ValueError(f"Event '{event_id}' not found.")
    return obj


@mcp.prompt()
def memory_assistant() -> str:
    return """You are using the Mem0 MCP server for long-term memory management.

Quick Start:
1. Store memories: Use add_memory (returns event_id), then poll get_event_status until SUCCEEDED
2. Search memories: Use search_memories for semantic queries
3. List memories: Use get_memories for filtered browsing
4. Update/Delete: Use update_memory and delete_memory for modifications
5. List entities: Use list_entities to see all users, agents, and runs
6. Track writes: Use list_events to browse recent add/search operations

Tips:
- user_id is automatically added to filters
- Use "*" as wildcard for any non-null value
- Use delete_entities to remove an entity and all its memories
- Use get_memories with page and page_size for paginated results
- Combine filters with AND/OR/NOT for complex queries"""


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

    transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True)
    try:
        async with anyio.create_task_group() as tg:

            async def run_server(*, task_status=anyio.TASK_STATUS_IGNORED):
                async with transport.connect() as (read_stream, write_stream):
                    task_status.started()
                    await mcp._mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp._mcp_server.create_initialization_options(),
                        stateless=True,
                    )

            await tg.start(run_server)
            await transport.handle_request(request.scope, request.receive, capture_send)
            await transport.terminate()
            tg.cancel_scope.cancel()
    except Exception:
        logger.exception("MCP streamable transport error")
        return Response(status_code=500, content=b"Internal MCP transport error")

    if not response_started:
        return Response(status_code=500, content=b"Transport did not produce a response")

    return Response(
        content=bytes(response_body),
        status_code=response_status,
        headers={key.decode(): value.decode() for key, value in response_headers},
    )


@mcp_router.api_route(
    "/", methods=["GET", "POST", "DELETE"], include_in_schema=False, operation_id="handle_streamable_http_slash"
)
@mcp_router.api_route("", methods=["GET", "POST", "DELETE"], summary="MCP Endpoint")
async def handle_streamable_http(request: Request, user=Depends(verify_auth)):
    auth_token = auth_user_id_var.set(str(user.id) if user is not None else None)

    meta = request_meta(request)

    platform_token = platform_var.set(meta.platform or meta.ua_tool_name)
    source_token = mem0_source_var.set(meta.source or "MCP")

    try:
        return await _run_streamable_transport(request)
    finally:
        auth_user_id_var.reset(auth_token)
        platform_var.reset(platform_token)
        mem0_source_var.reset(source_token)


def setup_mcp_server(app: FastAPI) -> None:
    mcp._mcp_server.name = "mem0-mcp"
    app.include_router(mcp_router)
