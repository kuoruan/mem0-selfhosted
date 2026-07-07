# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.25.4"]
# ///
"""Mem0 MCP stdio → streamable-http proxy.

Forwards stdin/stdout (editor) requests to the upstream MCP server at
``MEM0_MCP_URL`` (default: ``https://mcp.mem0.ai/mcp/``), authenticated
via ``Authorization: Token <MEM0_API_KEY>``.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server import Server
from mcp.server.stdio import stdio_server

from _identity import resolve_mcp_url, resolve_api_key

logger = logging.getLogger("mem0.mcp_proxy")


def _auth_headers() -> dict[str, str]:
    """Return ``Authorization: Token <api_key>`` resolved via the standard identity chain."""
    key = resolve_api_key()
    return {"Authorization": f"Token {key}"} if key else {}


async def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    # streamablehttp_client yields (read_stream, write_stream, get_session_id).
    async with streamablehttp_client(resolve_mcp_url(), headers=_auth_headers()) as (read, write, _get_sid):
        async with ClientSession(read, write) as client:
            init = await client.initialize()
            server = Server(init.serverInfo.name, version=init.serverInfo.version)
            caps = init.capabilities

            # Raw handlers: bypass decorators to faithfully forward the full
            # upstream response (including meta, nextCursor, etc.). Decorators
            # reconstruct result objects from the handler return value, which
            # loses fields and (for call_tool) mangles the result entirely.

            if caps.tools:
                async def _list_tools(req: types.ListToolsRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.list_tools(cursor=req.params.cursor if req.params else None)
                    )

                async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.call_tool(name=req.params.name, arguments=req.params.arguments)
                    )

                server.request_handlers[types.ListToolsRequest] = _list_tools
                server.request_handlers[types.CallToolRequest] = _call_tool

            if caps.prompts:
                async def _list_prompts(req: types.ListPromptsRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.list_prompts(cursor=req.params.cursor if req.params else None)
                    )

                async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.get_prompt(name=req.params.name, arguments=req.params.arguments)
                    )

                server.request_handlers[types.ListPromptsRequest] = _list_prompts
                server.request_handlers[types.GetPromptRequest] = _get_prompt

            if caps.resources:
                async def _list_resources(req: types.ListResourcesRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.list_resources(cursor=req.params.cursor if req.params else None)
                    )

                async def _list_resource_templates(
                    req: types.ListResourceTemplatesRequest,
                ) -> types.ServerResult:
                    return types.ServerResult(
                        await client.list_resource_templates(cursor=req.params.cursor if req.params else None)
                    )

                async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
                    return types.ServerResult(await client.read_resource(uri=req.params.uri))

                server.request_handlers[types.ListResourcesRequest] = _list_resources
                server.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates
                server.request_handlers[types.ReadResourceRequest] = _read_resource

            if caps.completions:
                async def _complete(req: types.CompleteRequest) -> types.ServerResult:
                    return types.ServerResult(
                        await client.complete(
                            ref=req.params.ref,
                            argument=req.params.argument.model_dump(),
                            context_arguments=req.params.context.arguments if req.params.context else None,
                        )
                    )

                server.request_handlers[types.CompleteRequest] = _complete

            async with stdio_server() as (stdin, stdout):
                await server.run(stdin, stdout, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("proxy failed: %s: %s", type(e).__name__, e)
        sys.exit(1)
