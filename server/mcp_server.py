import contextvars
import logging
from typing import Annotated, Any, Optional

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport
from pydantic import Field

from auth import verify_auth
from compat.entities import list_entities_payload
from compat.responses import normalize_results_dict
from compat.scope import build_search_filters, collect_entity_params, reject_app_id, require_entity_scope
from server_state import get_memory_instance

logger = logging.getLogger("mem0.server.mcp")

auth_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("mcp_user_id", default=None)
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_client_name", default="mem0-mcp")

mcp = FastMCP("mem0")
mcp_router = APIRouter(prefix="/mcp")


def _fallback_uid() -> str | None:
    return auth_user_id_var.get()


@mcp.tool(description="Store a new preference, fact, or conversation snippet. Requires at least one: user_id, agent_id, or run_id.")
def add_memory(
    text: Annotated[str, Field(description="Plain sentence summarizing what to store.")],
    messages: Annotated[
        Optional[list[dict[str, str]]],
        Field(default=None, description="Structured conversation history with `role`/`content`. Use when you have multiple turns."),
    ] = None,
    user_id: Annotated[Optional[str], Field(default=None, description="Override the default user scope for this write.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent identifier.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Not supported by the self-hosted server (returns 501).")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Optional run identifier.")] = None,
    infer: Annotated[bool, Field(default=True, description="When False, store text verbatim without LLM fact extraction.")] = True,
    metadata: Annotated[Optional[dict[str, Any]], Field(default=None, description="Attach arbitrary metadata JSON to the memory.")] = None,
) -> dict[str, Any]:
    reject_app_id(app_id)
    scope = require_entity_scope(
        user_id=user_id, agent_id=agent_id, run_id=run_id,
        fallback_user_id=_fallback_uid(),
    )
    conversation = messages if messages is not None else [{"role": "user", "content": text}]
    add_kwargs: dict[str, Any] = {**scope}
    if metadata is not None:
        add_kwargs["metadata"] = metadata
    if not infer:
        add_kwargs["infer"] = False
    return normalize_results_dict(get_memory_instance().add(messages=conversation, **add_kwargs))


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
    agent_id: Annotated[Optional[str], Field(default=None, description="Limit search to this agent's memories.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Not supported by the self-hosted server (returns 501).")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Limit search to this run's memories.")] = None,
    filters: Annotated[Optional[dict[str, Any]], Field(default=None, description="Additional filter clauses (user_id injected automatically).")] = None,
    top_k: Annotated[Optional[int], Field(default=None, description="Number of results to return (1-1000, default 10).")] = None,
    threshold: Annotated[Optional[float], Field(default=None, description="Minimum semantic relevance score (0.0–1.0).")] = None,
) -> dict[str, Any]:
    reject_app_id(app_id)
    scoped_filters = build_search_filters(
        user_id=user_id, agent_id=agent_id, run_id=run_id,
        filters=filters, fallback_user_id=_fallback_uid(),
    )
    search_kwargs: dict[str, Any] = {"filters": scoped_filters}
    if top_k is not None:
        search_kwargs["top_k"] = top_k
    if threshold is not None:
        search_kwargs["threshold"] = threshold
    return normalize_results_dict(get_memory_instance().search(query=query, **search_kwargs))


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
    app_id: Annotated[Optional[str], Field(default=None, description="Not supported by the self-hosted server (returns 501).")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="List memories for this run.")] = None,
    filters: Annotated[Optional[dict[str, Any]], Field(default=None, description="Structured filters; user_id injected automatically.")] = None,
    page: Annotated[Optional[int], Field(default=None, description="1-indexed page number when paginating.")] = None,
    page_size: Annotated[Optional[int], Field(default=None, description="Number of memories per page (default 10).")] = None,
) -> dict[str, Any]:
    reject_app_id(app_id)
    scoped_filters = build_search_filters(
        user_id=user_id, agent_id=agent_id, run_id=run_id,
        filters=filters, fallback_user_id=_fallback_uid(),
    )
    result = normalize_results_dict(get_memory_instance().get_all(filters=scoped_filters))
    items = result["results"]
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
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    if not isinstance(result, dict):
        logger.warning(
            "get_memory returned unexpected type %s for memory_id=%s",
            type(result).__name__, memory_id,
        )
        return {}
    return result


@mcp.tool(description="Overwrite an existing memory's text.")
def update_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to overwrite.")],
    text: Annotated[str, Field(description="Replacement text for the memory.")],
    metadata: Annotated[Optional[dict[str, Any]], Field(default=None, description="Optional metadata to attach to the updated memory.")] = None,
) -> dict[str, Any]:
    update_kwargs: dict[str, Any] = {"memory_id": memory_id, "data": text}
    if metadata is not None:
        update_kwargs["metadata"] = metadata
    result = get_memory_instance().update(**update_kwargs)
    if not isinstance(result, dict):
        logger.warning(
            "update_memory returned unexpected type %s for memory_id=%s",
            type(result).__name__, memory_id,
        )
        return {"message": "Memory updated successfully"}
    return result


@mcp.tool(description="Delete one memory after the user confirms its memory_id.")
def delete_memory(
    memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
) -> dict[str, Any]:
    get_memory_instance().delete(memory_id)
    return {"message": "Memory deleted successfully"}


@mcp.tool(description="Delete every memory in the given user/agent/run but keep the entity.")
def delete_all_memories(
    user_id: Annotated[Optional[str], Field(default=None, description="User scope to delete; defaults to server user.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent scope to delete.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Not supported by the self-hosted server (returns 501).")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Optional run scope to delete.")] = None,
) -> dict[str, Any]:
    reject_app_id(app_id)
    scope = require_entity_scope(
        user_id=user_id, agent_id=agent_id, run_id=run_id,
        fallback_user_id=_fallback_uid(),
    )
    result = get_memory_instance().delete_all(**scope)
    return result if isinstance(result, dict) else {"message": "Memories deleted successfully"}


@mcp.tool(description="Remove an entity and cascade-delete its memories.")
def delete_entities(
    user_id: Annotated[Optional[str], Field(default=None, description="Delete this user and its memories.")] = None,
    agent_id: Annotated[Optional[str], Field(default=None, description="Delete this agent and its memories.")] = None,
    app_id: Annotated[Optional[str], Field(default=None, description="Not supported by the self-hosted server (returns 501).")] = None,
    run_id: Annotated[Optional[str], Field(default=None, description="Delete this run and its memories.")] = None,
) -> dict[str, Any]:
    reject_app_id(app_id)
    selected = list(collect_entity_params(user_id=user_id, agent_id=agent_id, run_id=run_id).items())
    if not selected:
        raise HTTPException(status_code=400, detail="Provide user_id, agent_id, or run_id before calling delete_entities.")
    memory = get_memory_instance()
    for key, value in selected:
        memory.delete_all(**{key: value})
    suffix = "y" if len(selected) == 1 else "ies"
    return {"message": f"Deleted {len(selected)} entit{suffix} successfully"}


@mcp.tool(description="List which users/agents/apps/runs currently hold memories.")
def list_entities() -> dict[str, Any]:
    results = list_entities_payload()
    return {"count": len(results), "results": results}


@mcp.prompt()
def memory_assistant() -> str:
    return """You are using the Mem0 MCP server for long-term memory management.

Quick Start:
1. Store memories: Use add_memory to save facts, preferences, or conversations
2. Search memories: Use search_memories for semantic queries
3. List memories: Use get_memories for filtered browsing
4. Update/Delete: Use update_memory and delete_memory for modifications
5. List entities: Use list_entities to see all users, agents, and runs

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

    if not response_started:
        return Response(status_code=500, content=b"Transport did not produce a response")

    return Response(
        content=bytes(response_body),
        status_code=response_status,
        headers={key.decode(): value.decode() for key, value in response_headers},
    )


@mcp_router.api_route("", methods=["GET", "POST", "DELETE"])
@mcp_router.api_route("/", methods=["GET", "POST", "DELETE"])
async def handle_streamable_http(request: Request, user=Depends(verify_auth)):
    auth_token = auth_user_id_var.set(str(user.id) if user is not None else None)
    client_token = client_name_var.set(
        request.headers.get("x-mcp-client-name") or request.headers.get("user-agent") or "mem0-mcp"
    )
    try:
        return await _run_streamable_transport(request)
    finally:
        auth_user_id_var.reset(auth_token)
        client_name_var.reset(client_token)


def setup_mcp_server(app: FastAPI) -> None:
    mcp._mcp_server.name = "mem0-mcp"
    app.include_router(mcp_router)
