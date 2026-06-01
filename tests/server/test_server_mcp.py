import importlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("mcp", reason="mcp not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import mcp_server
from compat.events import event_cache_clear, event_cache_put, make_event_obj

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


def _call_tool(client: TestClient, name: str, arguments: dict | None = None, *, req_id: int = 2) -> dict:
    response = client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": name, "arguments": arguments or {}}, req_id=req_id),
        headers=MCP_HEADERS,
    )
    assert response.status_code == 200
    return response.json()["result"]


def _structured(client: TestClient, name: str, arguments: dict | None = None, *, req_id: int = 2) -> dict:
    result = _call_tool(client, name, arguments, req_id=req_id)
    assert not result.get("isError"), result
    return result["structuredContent"]


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=False, name=None):
        self._target = target
        self._args = args

    def start(self):
        if self._target:
            self._target(*self._args)

class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return MagicMock()


@pytest.fixture(autouse=True)
def _clear_event_cache():
    event_cache_clear()
    yield
    event_cache_clear()


@pytest.fixture
def mcp_testbed(monkeypatch):
    module = importlib.reload(mcp_server)
    event_cache_clear()

    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "saved"}]}
    mock_memory.get.return_value = {"id": "mem-1", "memory": "saved"}
    mock_memory.get_all.return_value = [{"id": "mem-1", "memory": "saved", "user_id": "alice"}]
    mock_memory.search.return_value = [{"id": "mem-1", "memory": "saved", "score": 0.9}]
    mock_memory.update.return_value = {"message": "updated"}
    mock_memory.delete.return_value = {"message": "Memory deleted successfully!"}
    mock_memory.delete_all.return_value = {"message": "deleted"}

    def get_memory():
        return mock_memory

    monkeypatch.setattr(module, "get_memory_instance", get_memory)
    monkeypatch.setattr("server_state.get_memory_instance", get_memory)
    monkeypatch.setattr("memory_lock.get_memory_instance", get_memory)
    monkeypatch.setattr(module, "_ADD_EXECUTOR", _ImmediateExecutor())

    app = FastAPI()
    app.include_router(module.mcp_router)

    def _verify_auth_override():
        return None

    app.dependency_overrides[module.verify_auth] = _verify_auth_override

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
        "list_events",
        "get_event_status",
    }.issubset(tools)
    descriptions = {tool["name"]: tool["description"] for tool in tool_items}
    assert descriptions["add_memory"].startswith("Store a new preference")
    assert "infer=False" in descriptions["add_memory"]
    assert "get_event_status" in descriptions["add_memory"]
    assert "user_id is automatically added to filters" in descriptions["search_memories"]
    assert "user_id is automatically added to filters" in descriptions["get_memories"]


def test_add_memory_infer_false_returns_results_immediately(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(
        client,
        "add_memory",
        {"text": "verbatim fact", "user_id": "alice", "infer": False},
    )
    assert structured["results"][0]["id"] == "mem-1"
    assert structured["event_id"] is None
    assert structured["status"] == "SUCCEEDED"
    mock_memory.add.assert_called_once()


def test_add_memory_tool_uses_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(
        client, "add_memory", {"text": "remember this", "user_id": "alice", "infer": True},
    )
    assert structured["status"] == "PENDING"
    assert structured["event_id"]

    event = _structured(client, "get_event_status", {"event_id": structured["event_id"]}, req_id=3)
    assert event["status"] == "SUCCEEDED"
    assert event["results"][0]["id"] == "mem-1"
    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id="alice",
        metadata={"source": "MCP"},
        infer=True,
    )


def test_add_memory_requires_scope(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    result = _call_tool(client, "add_memory", {"text": "no scope"})
    assert result.get("isError") is True
    mock_memory.add.assert_not_called()


def test_add_memory_uses_messages_when_provided(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    _structured(
        client,
        "add_memory",
        {"text": "ignored", "user_id": "alice", "messages": messages},
    )

    mock_memory.add.assert_called_once()
    assert mock_memory.add.call_args.kwargs["messages"] == messages


def test_add_memory_infer_false_failure_surfaces_as_tool_error(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.add.side_effect = RuntimeError("add failed")

    result = _call_tool(
        client,
        "add_memory",
        {"text": "boom", "user_id": "alice", "infer": False},
    )
    assert result.get("isError") is True
    assert "add failed" in result["content"][0]["text"]


def test_add_memory_failure_updates_event_status(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.add.side_effect = RuntimeError("add failed")

    structured = _structured(
        client,
        "add_memory",
        {"text": "boom", "user_id": "alice", "infer": True},
    )
    event = _structured(client, "get_event_status", {"event_id": structured["event_id"]}, req_id=3)

    assert event["status"] == "FAILED"
    assert "add failed" in event["metadata"]["error"]


def test_get_event_status_not_found(mcp_testbed):
    _, client, _ = mcp_testbed

    result = _call_tool(client, "get_event_status", {"event_id": "00000000-0000-0000-0000-000000000099"})
    assert result.get("isError") is True


@pytest.fixture
def mcp_testbed_authed(monkeypatch):
    """Like mcp_testbed but verify_auth returns a real User-like object with a known id."""
    module = importlib.reload(mcp_server)

    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "saved"}]}

    def get_memory():
        return mock_memory

    monkeypatch.setattr(module, "get_memory_instance", get_memory)
    monkeypatch.setattr("server_state.get_memory_instance", get_memory)
    monkeypatch.setattr("memory_lock.get_memory_instance", get_memory)
    monkeypatch.setattr(module, "_ADD_EXECUTOR", _ImmediateExecutor())

    auth_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    mock_user = MagicMock()
    mock_user.id = auth_user_id

    app = FastAPI()
    app.include_router(module.mcp_router)

    def _verify_auth_override():
        return mock_user

    app.dependency_overrides[module.verify_auth] = _verify_auth_override

    client = TestClient(app)
    _initialize_client(client)
    return module, client, mock_memory, str(auth_user_id)


def test_list_events_filters_by_authenticated_user(mcp_testbed_authed):
    _, client, _, auth_uid = mcp_testbed_authed
    now = "2026-01-01T00:00:00+00:00"
    event_cache_put(
        "e1",
        {**make_event_obj("e1", [], now_iso=now, status="SUCCEEDED"), "owner_user_id": auth_uid},
    )
    event_cache_put(
        "e2",
        {
            **make_event_obj("e2", [], now_iso="2026-01-02T00:00:00+00:00", status="SUCCEEDED"),
            "owner_user_id": "other-user",
        },
    )

    listed = _structured(client, "list_events")
    assert listed["count"] == 1
    assert listed["results"][0]["id"] == "e1"


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
        infer=True,
    )


def test_add_memory_infer_false_passes_flag(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(
        client,
        "add_memory",
        {"text": "verbatim", "user_id": "alice", "infer": False},
    )

    assert structured["results"][0]["id"] == "mem-1"
    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "verbatim"}],
        user_id="alice",
        metadata={"source": "MCP"},
        infer=False,
    )


def test_add_memory_with_custom_source(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    client.post(
        "/mcp",
        json=_jsonrpc(
            "tools/call",
            {
                "name": "add_memory",
                "arguments": {"text": "tagged", "user_id": "alice", "source": "cursor"},
            },
            req_id=2,
        ),
        headers=MCP_HEADERS,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "tagged"}],
        user_id="alice",
        metadata={"source": "cursor"},
        infer=True,
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
        infer=True,
    )


def test_list_events_filter_and_pagination(mcp_testbed):
    _, client, _ = mcp_testbed
    now = "2026-01-01T00:00:00+00:00"
    event_cache_put("e1", {**make_event_obj("e1", [], now_iso=now, status="SUCCEEDED"), "owner_user_id": "alice"})
    event_cache_put(
        "e2",
        {
            **make_event_obj("e2", [], now_iso="2026-01-02T00:00:00+00:00", status="SUCCEEDED"),
            "owner_user_id": "bob",
        },
    )
    event_cache_put(
        "e3",
        {
            **make_event_obj("e3", [], now_iso="2026-01-03T00:00:00+00:00", status="PENDING"),
            "owner_user_id": "alice",
        },
    )

    listed = _structured(client, "list_events")
    assert listed["count"] == 3
    assert len(listed["results"]) == 3

    paged = _structured(client, "list_events", {"page": 1, "page_size": 2})
    assert paged["count"] == 3
    assert len(paged["results"]) == 2


def test_prompts_get_memory_assistant(mcp_testbed):
    _, client, _ = mcp_testbed

    response = client.post("/mcp", json=_jsonrpc("prompts/get", {"name": "memory_assistant"}, req_id=2), headers=MCP_HEADERS)
    assert response.status_code == 200
    messages = response.json()["result"]["messages"]
    assert any("add_memory" in msg.get("content", {}).get("text", "") for msg in messages)


def test_get_memory_success(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(client, "get_memory", {"memory_id": "mem-1"})
    assert structured["id"] == "mem-1"
    mock_memory.get.assert_called_once_with("mem-1")


def test_get_memory_not_found(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get.return_value = None

    result = _call_tool(client, "get_memory", {"memory_id": "missing"})
    assert result.get("isError") is True


def test_search_memories_passes_top_k_and_threshold(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "search_memories",
        {"query": "prefs", "user_id": "alice", "top_k": 5, "threshold": 0.8},
    )

    mock_memory.search.assert_called_once_with(
        query="prefs",
        filters={"user_id": "alice"},
        top_k=5,
        threshold=0.8,
    )


def test_get_memories_pagination(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [
        {"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(5)
    ]

    structured = _structured(client, "get_memories", {"user_id": "alice", "page": 2, "page_size": 2})

    assert structured["count"] == 5
    assert len(structured["results"]) == 2
    assert structured["results"][0]["id"] == "mem-2"


def test_get_memories_page_without_page_size_uses_defaults(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [
        {"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(25)
    ]

    structured = _structured(client, "get_memories", {"user_id": "alice", "page": 1})

    assert structured["count"] == 25
    assert len(structured["results"]) == 10


def test_list_events_page_without_page_size_uses_defaults(mcp_testbed):
    _, client, _ = mcp_testbed
    now = "2026-01-01T00:00:00+00:00"
    for i in range(3):
        event_cache_put(f"e{i}", make_event_obj(f"e{i}", [], now_iso=now, status="SUCCEEDED"))

    paged = _structured(client, "list_events", {"page": 2, "page_size": 2})
    assert paged["count"] == 3
    assert len(paged["results"]) == 1

    page_only = _structured(client, "list_events", {"page": 1})
    assert page_only["count"] == 3
    assert len(page_only["results"]) == 3


def test_delete_memory_invokes_sdk(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(client, "delete_memory", {"memory_id": "mem-1"})
    mock_memory.delete.assert_called_once_with("mem-1")
    assert structured["message"] == "Memory deleted successfully!"


def test_delete_all_memories_scoped(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(client, "delete_all_memories", {"user_id": "alice", "agent_id": "bot"})
    mock_memory.delete_all.assert_called_once_with(user_id="alice", agent_id="bot")


def test_delete_entities_requires_scope(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    result = _call_tool(client, "delete_entities", {})
    assert result.get("isError") is True
    mock_memory.delete_all.assert_not_called()


def test_delete_entities_calls_delete_all_per_entity(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(client, "delete_entities", {"user_id": "alice", "agent_id": "bot"})
    assert structured["message"] == "Entities deleted successfully, count: 2."
    assert mock_memory.delete_all.call_count == 2


def test_list_entities_returns_payload(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    row = MagicMock(
        payload={
            "user_id": "alice",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        }
    )
    mock_memory.vector_store.list.return_value = [row]

    with patch("compat.entities.get_memory_instance", return_value=mock_memory):
        structured = _structured(client, "list_entities")
    assert structured["count"] == 1
    assert structured["results"][0]["name"] == "alice"


def test_source_from_x_mem0_source_header(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    headers = {**MCP_HEADERS, "x-mem0-source": "CURSOR"}

    client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": "add_memory", "arguments": {"text": "hdr", "user_id": "alice"}}, req_id=2),
        headers=headers,
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "hdr"}],
        user_id="alice",
        metadata={"source": "CURSOR"},
        infer=True,
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
    """normalize_vector_store_list should handle all documented backend return shapes."""
    from compat.entities import normalize_vector_store_list

    # Empty / falsy
    assert normalize_vector_store_list(None) == []
    assert normalize_vector_store_list([]) == []

    # PGVector / Chroma: nested list
    row = MagicMock(payload={"foo": "bar"})
    assert normalize_vector_store_list([[row]]) == [row]

    # Qdrant: tuple of (rows, offset)
    assert normalize_vector_store_list(([row], "next_offset")) == [row]

    # Qdrant edge: tuple with non-list first element
    assert normalize_vector_store_list((None, "offset")) == []
    assert normalize_vector_store_list(("not-a-list", 0)) == []

    # Flat list
    assert normalize_vector_store_list([row]) == [row]


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
