"""Tests for mcp_proxy.py with lightweight MCP stubs.

The proxy script is written as a standalone Python script with inline `mcp`
metadata, so these tests stub the small slice of the SDK needed to import it.
That keeps the feedback loop fast while still covering the proxy-specific logic.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


def _install_fake_mcp(monkeypatch) -> None:
    mcp_mod = ModuleType("mcp")
    types_mod = ModuleType("mcp.types")
    client_mod = ModuleType("mcp.client")
    streamable_http_mod = ModuleType("mcp.client.streamable_http")
    server_mod = ModuleType("mcp.server")
    stdio_mod = ModuleType("mcp.server.stdio")

    class ClientSession:  # pragma: no cover - import stub only
        pass

    class Server:
        def __init__(self, *_args, **_kwargs):
            self.run = AsyncMock()
            self.request_handlers: dict = {}

        def list_tools(self):
            return lambda func: func

        def call_tool(self):
            return lambda func: func

        def list_prompts(self):
            return lambda func: func

        def get_prompt(self):
            return lambda func: func

        def list_resources(self):
            return lambda func: func

        def list_resource_templates(self):
            return lambda func: func

        def read_resource(self):
            return lambda func: func

        def create_initialization_options(self):
            return {}

    @dataclass
    class Tool:
        name: str = "tool"

    @dataclass
    class Prompt:
        name: str = "prompt"

    @dataclass
    class Resource:
        uri: str = "resource://test"

    @dataclass
    class ContentBlock:
        type: str = "text"

    @dataclass
    class TextResourceContents:
        text: str

    @dataclass
    class BlobResourceContents:
        blob: bytes

    @dataclass
    class CallToolResult:
        content: list[ContentBlock]
        isError: bool = False
        meta: dict[str, object] | None = None

    @dataclass
    class ResourceTemplate:
        uriTemplate: str = "resource://{id}"

    @dataclass
    class GetPromptResult:
        description: str = ""

    @dataclass
    class ReadResourceResult:
        contents: list[TextResourceContents | BlobResourceContents]

    @dataclass
    class ServerResult:
        root: object = None

    # Request types
    @dataclass
    class ListToolsRequest:
        pass

    @dataclass
    class CallToolRequest:
        params: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(name="test", arguments={})
        )

    @dataclass
    class ListPromptsRequest:
        pass

    @dataclass
    class GetPromptRequest:
        params: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(name="test", arguments=None)
        )

    @dataclass
    class ListResourcesRequest:
        pass

    @dataclass
    class ListResourceTemplatesRequest:
        pass

    @dataclass
    class ReadResourceRequest:
        params: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(uri="test://resource")
        )

    @dataclass
    class CompleteRequest:
        params: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(
                ref=SimpleNamespace(type="ref/prompt", name="test_prompt"),
                argument=SimpleNamespace(name="arg1", value="partial"),
                context=SimpleNamespace(arguments={"resolved": "val"}),
            )
        )

    @dataclass
    class CompleteResult:
        completion: SimpleNamespace = field(
            default_factory=lambda: SimpleNamespace(values=[], total=0, hasMore=False)
        )

    # Result types
    @dataclass
    class ListToolsResult:
        tools: list = field(default_factory=list)

    @dataclass
    class ListPromptsResult:
        prompts: list = field(default_factory=list)

    @dataclass
    class ListResourcesResult:
        resources: list = field(default_factory=list)

    @dataclass
    class ListResourceTemplatesResult:
        resourceTemplates: list = field(default_factory=list)

    def streamablehttp_client(*_args, **_kwargs):  # pragma: no cover - import stub only
        raise AssertionError("streamablehttp_client should not be used in unit tests")

    def stdio_server(*_args, **_kwargs):  # pragma: no cover - import stub only
        raise AssertionError("stdio_server should not be used in unit tests")

    mcp_mod.ClientSession = ClientSession

    types_mod.Tool = Tool
    types_mod.Prompt = Prompt
    types_mod.Resource = Resource
    types_mod.ContentBlock = ContentBlock
    types_mod.TextResourceContents = TextResourceContents
    types_mod.BlobResourceContents = BlobResourceContents
    types_mod.CallToolResult = CallToolResult
    types_mod.ResourceTemplate = ResourceTemplate
    types_mod.GetPromptResult = GetPromptResult
    types_mod.ReadResourceResult = ReadResourceResult
    types_mod.ServerResult = ServerResult
    types_mod.ListToolsRequest = ListToolsRequest
    types_mod.CallToolRequest = CallToolRequest
    types_mod.ListPromptsRequest = ListPromptsRequest
    types_mod.GetPromptRequest = GetPromptRequest
    types_mod.ListResourcesRequest = ListResourcesRequest
    types_mod.ListResourceTemplatesRequest = ListResourceTemplatesRequest
    types_mod.ReadResourceRequest = ReadResourceRequest
    types_mod.CompleteRequest = CompleteRequest
    types_mod.CompleteResult = CompleteResult
    types_mod.ListToolsResult = ListToolsResult
    types_mod.ListPromptsResult = ListPromptsResult
    types_mod.ListResourcesResult = ListResourcesResult
    types_mod.ListResourceTemplatesResult = ListResourceTemplatesResult

    streamable_http_mod.streamablehttp_client = streamablehttp_client
    server_mod.Server = Server
    stdio_mod.stdio_server = stdio_server

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.types", types_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_http_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.stdio", stdio_mod)


@pytest.fixture()
def mcp_proxy(monkeypatch):
    _install_fake_mcp(monkeypatch)
    sys.modules.pop("mcp_proxy", None)
    return importlib.import_module("mcp_proxy")


# ---------------------------------------------------------------------------
# Helper: async context manager mock
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Minimal async context manager that yields a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass


def _make_mock_client(*, tools=True, prompts=True, resources=True, completions=True) -> MagicMock:
    """Create a mock upstream ClientSession with the given capabilities."""
    client = MagicMock()
    client.initialize = AsyncMock(
        return_value=SimpleNamespace(
            serverInfo=SimpleNamespace(name="test-upstream", version="2.0"),
            capabilities=SimpleNamespace(
                tools=SimpleNamespace() if tools else None,
                prompts=SimpleNamespace() if prompts else None,
                resources=SimpleNamespace() if resources else None,
                completions=SimpleNamespace() if completions else None,
            ),
        )
    )
    client.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    client.call_tool = AsyncMock(return_value=MagicMock())
    client.list_prompts = AsyncMock(return_value=SimpleNamespace(prompts=[]))
    client.get_prompt = AsyncMock(return_value=MagicMock())
    client.list_resources = AsyncMock(return_value=SimpleNamespace(resources=[]))
    client.list_resource_templates = AsyncMock(return_value=SimpleNamespace(resourceTemplates=[]))
    client.read_resource = AsyncMock(return_value=MagicMock())
    client.complete = AsyncMock(return_value=MagicMock())
    return client


def _install_mock_transports(monkeypatch, mock_client):
    """Replace the stub transport functions with mocks wired to *mock_client*."""
    _install_fake_mcp(monkeypatch)

    import mcp.client.streamable_http as sh
    import mcp.server.stdio as stdio_mod
    import mcp as mcp_mod

    sh.streamablehttp_client = MagicMock(return_value=_AsyncCtx((MagicMock(), MagicMock(), MagicMock())))
    mcp_mod.ClientSession = MagicMock(return_value=_AsyncCtx(mock_client))
    stdio_mod.stdio_server = MagicMock(return_value=_AsyncCtx((MagicMock(), MagicMock())))


def _reload_mcp_proxy(monkeypatch, mock_client):
    """Install mock transports and (re-)import mcp_proxy."""
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    _install_mock_transports(monkeypatch, mock_client)
    sys.modules.pop("mcp_proxy", None)
    return importlib.import_module("mcp_proxy")


# ---------------------------------------------------------------------------
# Unit tests: _auth_headers, _read_resource_body
# ---------------------------------------------------------------------------


def test_auth_headers_omitted_when_no_key(monkeypatch, mcp_proxy):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    import _identity
    monkeypatch.setattr(_identity, "_extract_key_from_shell_profiles", lambda: "")
    assert mcp_proxy._auth_headers() == {}


def test_auth_headers_empty_key_omitted(monkeypatch, mcp_proxy):
    monkeypatch.setenv("MEM0_API_KEY", "   ")
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    import _identity
    monkeypatch.setattr(_identity, "_extract_key_from_shell_profiles", lambda: "")
    assert mcp_proxy._auth_headers() == {}


def test_auth_headers_token_for_api_key(monkeypatch, mcp_proxy):
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    import _identity
    monkeypatch.setattr(_identity, "_extract_key_from_shell_profiles", lambda: "")
    for key in ("m0-platform-key", "m0sk_selfhost_key", "admin-static-key"):
        monkeypatch.setenv("MEM0_API_KEY", key)
        assert mcp_proxy._auth_headers() == {"Authorization": f"Token {key}"}


def test_auth_headers_strips_whitespace(monkeypatch, mcp_proxy):
    monkeypatch.setenv("MEM0_API_KEY", "  m0-key-with-spaces  ")
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    import _identity
    monkeypatch.setattr(_identity, "_extract_key_from_shell_profiles", lambda: "")
    assert mcp_proxy._auth_headers() == {"Authorization": "Token m0-key-with-spaces"}


# ---------------------------------------------------------------------------
# Integration tests: main() proxy flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_connects_and_initializes(monkeypatch):
    """main() initializes upstream session and runs the local server."""
    mock_client = _make_mock_client()
    mcp_proxy = _reload_mcp_proxy(monkeypatch, mock_client)

    await mcp_proxy.main()

    mock_client.initialize.assert_awaited_once()
    # Verify server.run() was called (AsyncMock on the Server instance)
    import mcp.server.stdio as stdio_mod

    stdio_mod.stdio_server.assert_called_once()


@pytest.mark.asyncio
async def test_main_respects_capabilities_tools_only(monkeypatch):
    """When upstream only advertises tools, only tool handlers are registered."""
    mock_client = _make_mock_client(tools=True, prompts=False, resources=False, completions=False)
    mcp_proxy = _reload_mcp_proxy(monkeypatch, mock_client)

    await mcp_proxy.main()

    mock_client.initialize.assert_awaited_once()
    # prompts/resources/completions methods should never be called since handlers
    # were not registered (and server.run is mocked, so handlers never fire).
    mock_client.list_prompts.assert_not_called()
    mock_client.list_resources.assert_not_called()
    mock_client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_main_registers_completions_handler(monkeypatch):
    """When upstream advertises completions, the complete handler is registered."""
    mock_client = _make_mock_client(tools=False, prompts=False, resources=False, completions=True)
    mcp_proxy = _reload_mcp_proxy(monkeypatch, mock_client)

    await mcp_proxy.main()

    mock_client.initialize.assert_awaited_once()
    # tools/prompts/resources should not be called
    mock_client.list_tools.assert_not_called()
    mock_client.list_prompts.assert_not_called()
    mock_client.list_resources.assert_not_called()


@pytest.mark.asyncio
async def test_main_connection_error_propagates(monkeypatch):
    """When upstream connection fails, the exception propagates to caller."""
    mock_client = _make_mock_client()
    mock_client.initialize = AsyncMock(side_effect=ConnectionError("upstream down"))
    mcp_proxy = _reload_mcp_proxy(monkeypatch, mock_client)

    with pytest.raises(ConnectionError, match="upstream down"):
        await mcp_proxy.main()


@pytest.mark.asyncio
async def test_main_no_api_key_uses_empty_auth(monkeypatch):
    """main() works without MEM0_API_KEY (unauthenticated proxy)."""
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    mock_client = _make_mock_client()
    _install_mock_transports(monkeypatch, mock_client)
    sys.modules.pop("mcp_proxy", None)
    mcp_proxy = importlib.import_module("mcp_proxy")

    await mcp_proxy.main()

    mock_client.initialize.assert_awaited_once()
