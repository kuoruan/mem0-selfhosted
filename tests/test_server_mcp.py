import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("mcp", reason="mcp not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

mcp_server = importlib.import_module("mcp_server")

MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _jsonrpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }


def _initialize_payload(req_id: int = 1) -> dict:
    return _jsonrpc(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        },
        req_id=req_id,
    )


@pytest.fixture
def mcp_testbed(monkeypatch):
    module = importlib.reload(mcp_server)

    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "saved"}]}
    mock_memory.get.return_value = {"id": "mem-1", "memory": "saved"}
    mock_memory.get_all.return_value = [{"id": "mem-1", "memory": "saved", "user_id": "alice"}]
    mock_memory.search.return_value = [{"id": "mem-1", "memory": "saved", "score": 0.9}]
    mock_memory.update.return_value = {"message": "updated"}
    mock_memory.delete.return_value = None
    mock_memory.delete_all.return_value = {"message": "deleted"}

    monkeypatch.setattr(module, "get_memory_instance", lambda: mock_memory)

    app = FastAPI()
    app.include_router(module.mcp_router)
    app.dependency_overrides[module.verify_auth] = lambda: None

    client = TestClient(app)
    return module, client, mock_memory


def _initialize_client(client: TestClient, headers: dict | None = None) -> None:
    response = client.post("/mcp", json=_initialize_payload(), headers=headers or MCP_HEADERS)
    assert response.status_code == 200
    assert response.json()["result"]["protocolVersion"] == "2025-03-26"


def test_tools_list_exposes_expected_toolset(mcp_testbed):
    _, client, _ = mcp_testbed
    _initialize_client(client)

    response = client.post("/mcp", json=_jsonrpc("tools/list", req_id=2), headers=MCP_HEADERS)

    assert response.status_code == 200
    tool_items = response.json()["result"]["tools"]
    tools = {tool["name"] for tool in tool_items}
    assert {
        "add_memory",
        "search_memories",
        "get_memories",
        "get_memory",
        "update_memory",
        "delete_memory",
        "delete_all_memories",
        "delete_entities",
        "list_entities",
    }.issubset(tools)
    assert "list_events" not in tools
    assert "get_event_status" not in tools
    descriptions = {tool["name"]: tool["description"] for tool in tool_items}
    assert descriptions["add_memory"].startswith("Store a new preference")
    assert "user_id is automatically added to filters" in descriptions["search_memories"]
    assert "user_id is automatically added to filters" in descriptions["get_memories"]


def test_add_memory_tool_uses_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    response = client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "remember this", "user_id": "alice"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    structured = response.json()["result"]["structuredContent"]
    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id="alice",
    )
    assert structured["results"][0]["id"] == "mem-1"


def test_add_memory_requires_scope(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    response = client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "no scope"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    # Tool raises HTTPException(400) — MCP layer returns it as an error
    result = response.json()["result"]
    assert result.get("isError") is True
    mock_memory.add.assert_not_called()


@pytest.fixture
def mcp_testbed_authed(monkeypatch):
    """Like mcp_testbed but verify_auth returns a real User-like object with a known id."""
    import uuid

    module = importlib.reload(mcp_server)

    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "saved"}]}
    monkeypatch.setattr(module, "get_memory_instance", lambda: mock_memory)

    auth_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    mock_user = MagicMock()
    mock_user.id = auth_user_id

    app = FastAPI()
    app.include_router(module.mcp_router)
    app.dependency_overrides[module.verify_auth] = lambda: mock_user

    client = TestClient(app)
    return module, client, mock_memory, str(auth_user_id)


def test_add_memory_defaults_user_id_to_auth_user(mcp_testbed_authed):
    _, client, mock_memory, auth_uid = mcp_testbed_authed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "remember this"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id=auth_uid,
    )


def test_add_memory_infer_false_passes_flag(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "verbatim", "user_id": "alice", "infer": False}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "verbatim"}],
        user_id="alice",
        infer=False,
    )


def test_add_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "decision made", "user_id": "alice", "metadata": {"type": "decision"}}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "decision made"}],
        user_id="alice",
        metadata={"type": "decision"},
    )


def test_search_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "search_memories", "arguments": {"query": "test", "user_id": "alice"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.search.assert_called_once_with(query="test", filters={"user_id": "alice"})


def test_get_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "get_memories", "arguments": {"user_id": "alice"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.get_all.assert_called_once_with(filters={"user_id": "alice"})


def test_update_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    _initialize_client(client)

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "update_memory", "arguments": {"memory_id": "mem-1", "text": "new text", "metadata": {"type": "revised"}}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", data="new text", metadata={"type": "revised"})


def test_client_name_context_is_taken_from_header(mcp_testbed):
    module, client, _ = mcp_testbed
    captured: dict[str, str | None] = {}

    @module.mcp.tool(name="__test_client_name", description="test only")
    def _capture_client_name() -> dict[str, str | None]:
        captured["client_name"] = module.client_name_var.get(None)
        return {"client_name": captured["client_name"]}

    headers = {**MCP_HEADERS, "x-mcp-client-name": "cursor-test"}

    try:
        _initialize_client(client, headers=headers)
        response = client.post(
            "/mcp",
            json=_jsonrpc("tools/call", {"name": "__test_client_name", "arguments": {}}, req_id=2),
            headers=headers,
        )
        assert response.status_code == 200
        structured = response.json()["result"]["structuredContent"]
        assert captured["client_name"] == "cursor-test"
        assert structured["client_name"] == "cursor-test"
    finally:
        module.mcp._tool_manager._tools.pop("__test_client_name", None)
