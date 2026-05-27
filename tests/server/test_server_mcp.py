import importlib
import uuid
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("mcp", reason="mcp not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.mcp_server as mcp_server

# MCP_HEADERS with a User-Agent prefixed with "Mozilla" so it's skipped as generic,
# avoiding platform injection in tests that don't explicitly test that feature.
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "User-Agent": "Mozilla"}


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

    monkeypatch.setattr("server.server_state.get_memory_instance", lambda: mock_memory)

    app = FastAPI()
    app.include_router(module.mcp_router)
    app.dependency_overrides[module.verify_auth] = lambda: None

    client = TestClient(app)
    _initialize_client(client)
    return module, client, mock_memory


def _initialize_client(client: TestClient, headers: dict | None = None) -> None:
    response = client.post("/mcp", json=_initialize_payload(), headers=headers or MCP_HEADERS)
    assert response.status_code == 200
    assert response.json()["result"]["protocolVersion"] == "2025-03-26"


def test_tools_list_exposes_expected_toolset(mcp_testbed):
    _, client, _ = mcp_testbed

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

    response = client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call", {"name": "add_memory", "arguments": {"text": "remember this", "user_id": "alice"}}, req_id=2
        ),
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    structured = response.json()["result"]["structuredContent"]
    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id="alice",
        metadata={"source": "MCP"},
    )
    assert structured["results"][0]["id"] == "mem-1"


def test_add_memory_requires_scope(mcp_testbed):
    _, client, mock_memory = mcp_testbed

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
    module = importlib.reload(mcp_server)

    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "saved"}]}
    monkeypatch.setattr("server.server_state.get_memory_instance", lambda: mock_memory)

    auth_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    mock_user = MagicMock()
    mock_user.id = auth_user_id

    app = FastAPI()
    app.include_router(module.mcp_router)
    app.dependency_overrides[module.verify_auth] = lambda: mock_user

    client = TestClient(app)
    _initialize_client(client)
    return module, client, mock_memory, str(auth_user_id)


def test_add_memory_defaults_user_id_to_auth_user(mcp_testbed_authed):
    _, client, mock_memory, auth_uid = mcp_testbed_authed

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "remember this"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id=auth_uid,
        metadata={"source": "MCP"},
    )


def test_add_memory_infer_false_passes_flag(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {"name": "add_memory", "arguments": {"text": "verbatim", "user_id": "alice", "infer": False}},
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "verbatim"}],
        user_id="alice",
        metadata={"source": "MCP"},
        infer=False,
    )


def test_add_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {
                "name": "add_memory",
                "arguments": {"text": "decision made", "user_id": "alice", "metadata": {"type": "decision"}},
            },
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "decision made"}],
        user_id="alice",
        metadata={"source": "MCP", "type": "decision"},
    )


def test_search_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call", {"name": "search_memories", "arguments": {"query": "test", "user_id": "alice"}}, req_id=2
        ),
        headers=MCP_HEADERS,
    )

    mock_memory.search.assert_called_once_with(query="test", filters={"user_id": "alice"})


def test_get_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "get_memories", "arguments": {"user_id": "alice"}}, req_id=2),
        headers=MCP_HEADERS,
    )

    mock_memory.get_all.assert_called_once_with(filters={"user_id": "alice"})


def test_get_memory_non_dict_returns_empty(mcp_testbed):
    """When SDK get() returns a non-dict, MCP reports a tool error (Pydantic validation)."""
    module, client, mock_memory = mcp_testbed
    mock_memory.get.return_value = ["not", "a", "dict"]

    response = client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {"name": "get_memory", "arguments": {"memory_id": "mem-x"}},
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    mock_memory.get.assert_called_once_with("mem-x")
    result = response.json()["result"]
    # Non-dict SDK output causes Pydantic validation error in MCP framework
    assert result.get("isError") is True


def test_update_memory_non_dict_returns_fallback(mcp_testbed):
    """When SDK update() returns a non-dict, MCP reports a tool error (Pydantic validation)."""
    module, client, mock_memory = mcp_testbed
    mock_memory.update.return_value = "ok"

    response = client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {"name": "update_memory", "arguments": {"memory_id": "mem-x", "text": "new"}},
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    mock_memory.update.assert_called_once_with(memory_id="mem-x", data="new")
    result = response.json()["result"]
    # Non-dict SDK output causes Pydantic validation error in MCP framework
    assert result.get("isError") is True


def test_normalize_list_result_shapes():
    """_normalize_list_result should handle all documented backend return shapes."""
    from compat.entities import _normalize_list_result

    # Empty / falsy
    assert _normalize_list_result(None) == []
    assert _normalize_list_result([]) == []

    # PGVector / Chroma: nested list
    row = MagicMock(payload={"foo": "bar"})
    assert _normalize_list_result([[row]]) == [row]

    # Qdrant: tuple of (rows, offset)
    assert _normalize_list_result(([row], "next_offset")) == [row]

    # Qdrant edge: tuple with non-list first element
    assert _normalize_list_result((None, "offset")) == []
    assert _normalize_list_result(("not-a-list", 0)) == []

    # Flat list
    assert _normalize_list_result([row]) == [row]


def test_iter_payloads_skips_none_rows():
    """iter_payloads should skip None entries in the rows list."""
    from unittest.mock import patch
    from compat.entities import iter_payloads

    row = MagicMock(payload={"data": 1})
    mock_memory = MagicMock()
    mock_memory.vector_store.list.return_value = [row, None, MagicMock(payload={"data": 2})]

    with patch("compat.entities.get_memory_instance", return_value=mock_memory):
        payloads = iter_payloads()

    assert payloads == [{"data": 1}, {"data": 2}]
    assert len(payloads) == 2


def test_update_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {
                "name": "update_memory",
                "arguments": {"memory_id": "mem-1", "text": "new text", "metadata": {"type": "revised"}},
            },
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", data="new text", metadata={"type": "revised"})


def test_platform_context_is_taken_from_header(mcp_testbed):
    module, client, _ = mcp_testbed
    captured: dict[str, str | None] = {}

    @module.mcp.tool(name="__test_platform", description="test only")
    def _capture_platform() -> dict[str, str | None]:
        captured["platform"] = module.platform_var.get(None)
        return {"platform": captured["platform"]}

    headers = {**MCP_HEADERS, "x-mem0-platform": "cursor"}

    try:
        response = client.post(
            "/mcp",
            json=_jsonrpc("tools/call", {"name": "__test_platform", "arguments": {}}, req_id=2),
            headers=headers,
        )
        assert response.status_code == 200
        structured = response.json()["result"]["structuredContent"]
        assert captured["platform"] == "cursor"
        assert structured["platform"] == "cursor"
    finally:
        module.mcp._tool_manager._tools.pop("__test_platform", None)
