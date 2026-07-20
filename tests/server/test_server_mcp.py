import importlib
import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("mcp", reason="mcp not installed")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import server.mcp_server as mcp_server
from server.compat.entities import iter_payloads, normalize_vector_store_list
from server.compat.events import event_cache_clear, event_cache_put, make_event_obj

# User-Agent "Mozilla" is treated as a generic client (no platform injection),
# so tests that don't exercise platform/source headers stay unaffected by them.
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "User-Agent": "Mozilla"}
AUTH_USER_ID = "00000000-0000-0000-0000-000000000001"


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


def _call_tool(
    client: TestClient,
    name: str,
    arguments: dict | None = None,
    *,
    req_id: int = 2,
    headers: dict | None = None,
) -> dict:
    """Invoke an MCP tool over JSON-RPC and return the result (asserting HTTP 200)."""
    response = client.post(
        "/mcp",
        json=_jsonrpc("tools/call", {"name": name, "arguments": arguments or {}}, req_id=req_id),
        headers=headers or MCP_HEADERS,
    )
    assert response.status_code == 200
    return response.json()["result"]


def _structured(
    client: TestClient,
    name: str,
    arguments: dict | None = None,
    *,
    req_id: int = 2,
    headers: dict | None = None,
) -> dict:
    """Like _call_tool but asserts the tool succeeded and returns its structuredContent."""
    result = _call_tool(client, name, arguments, req_id=req_id, headers=headers)
    assert not result.get("isError"), result
    return result["structuredContent"]


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return MagicMock()


# ---------------------------------------------------------------------------
# Transport & lifespan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_mcp_server_wraps_existing_lifespan(monkeypatch):
    module = importlib.reload(mcp_server)
    events = []

    @asynccontextmanager
    async def original_lifespan(app):
        events.append("app-start")
        yield
        events.append("app-stop")

    class _FakeSessionManager:
        # The lifespan constructs the session manager inline (per cycle), so
        # patching the class captures both construction and run() ordering.
        def __init__(self, **kwargs):
            pass

        @asynccontextmanager
        async def run(self):
            events.append("mcp-start")
            yield
            events.append("mcp-stop")

    monkeypatch.setattr(module, "StreamableHTTPSessionManager", _FakeSessionManager)

    app = FastAPI(lifespan=original_lifespan)
    module.setup_mcp_server(app)

    async with app.router.lifespan_context(app):
        events.append("inside")

    assert events == ["app-start", "mcp-start", "inside", "mcp-stop", "app-stop"]


@pytest.mark.asyncio
async def test_mcp_streamable_response_delegates_to_session_manager(monkeypatch):
    """_McpStreamableResponse passes the real ASGI send straight through to the
    session manager (no capture buffer), so status/headers/body reach the client
    verbatim, including duplicate headers."""
    module = importlib.reload(mcp_server)
    sent = []

    async def send(message):
        sent.append(message)

    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b'{"jsonrpc":"2.0"}', "more_body": False}

    class _FakeSessionManager:
        async def handle_request(self, scope, receive, send):
            self.received_send = send
            self.received_scope = scope
            message = await receive()
            assert message["body"] == b'{"jsonrpc":"2.0"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 202,
                    "headers": [(b"x-repeat", b"one"), (b"x-repeat", b"two")],
                }
            )
            await send({"type": "http.response.body", "body": b"accepted"})

    fake_sm = _FakeSessionManager()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "root_path": "",
        "query_string": b"",
        "scheme": "http",
        "server": ("localhost", 8000),
        "client": ("127.0.0.1", 50000),
        "http_version": "1.1",
    }
    request = Request(scope, receive)

    response = module._McpStreamableResponse(request, user=None, session_manager=fake_sm)
    await response(scope, receive, send)

    assert receive_calls == 1
    assert fake_sm.received_scope["path"] == "/mcp"
    # Real send pass-through: handle_request got the same callable, and its
    # messages arrived verbatim (duplicate headers preserved).
    assert fake_sm.received_send is send
    assert sent[0] == {
        "type": "http.response.start",
        "status": 202,
        "headers": [(b"x-repeat", b"one"), (b"x-repeat", b"two")],
    }
    assert sent[1] == {"type": "http.response.body", "body": b"accepted"}


@pytest.mark.asyncio
async def test_setup_mcp_server_uses_per_app_session_managers(monkeypatch):
    """Each FastAPI app gets its own session manager. Construction is per
    lifespan cycle (not at install time), so drive each app's lifespan once."""
    module = importlib.reload(mcp_server)
    managers = []

    class _FakeSessionManager:
        def __init__(self, **kwargs):
            managers.append(self)

        @asynccontextmanager
        async def run(self):
            yield

    monkeypatch.setattr(module, "StreamableHTTPSessionManager", _FakeSessionManager)

    app_one = FastAPI()
    app_two = FastAPI()

    module.setup_mcp_server(app_one)
    module.setup_mcp_server(app_two)

    async with app_one.router.lifespan_context(app_one):
        pass
    async with app_two.router.lifespan_context(app_two):
        pass

    assert len(managers) == 2
    assert getattr(app_one.state, module._MCP_SESSION_MANAGER_STATE) is managers[0]
    assert getattr(app_two.state, module._MCP_SESSION_MANAGER_STATE) is managers[1]
    assert getattr(app_one.state, module._MCP_SESSION_MANAGER_STATE) is not getattr(
        app_two.state,
        module._MCP_SESSION_MANAGER_STATE,
    )


class _RecordingSessionManager:
    """Records construction kwargs and run() calls for lifespan assertions."""

    instances: list = []

    def __init__(self, **kwargs):
        self.__class__.instances.append(self)
        self.init_kwargs = kwargs
        self.run_calls = 0

    @asynccontextmanager
    async def run(self):
        self.run_calls += 1
        yield


@pytest.mark.asyncio
async def test_lifespan_creates_fresh_session_manager_per_cycle(monkeypatch):
    """The SDK's run() is once-per-instance; reusing one across lifespan cycles
    raises RuntimeError. Each cycle must construct a fresh manager (the fix)."""
    module = importlib.reload(mcp_server)
    _RecordingSessionManager.instances = []
    monkeypatch.setattr(module, "StreamableHTTPSessionManager", _RecordingSessionManager)

    app = FastAPI()
    module.setup_mcp_server(app)

    state_during_cycle = []
    for _ in range(3):
        async with app.router.lifespan_context(app):
            state_during_cycle.append(getattr(app.state, module._MCP_SESSION_MANAGER_STATE))

    instances = _RecordingSessionManager.instances
    assert len(instances) == 3
    assert len({id(s) for s in instances}) == 3  # three distinct instances
    assert all(s.run_calls == 1 for s in instances)  # run() exactly once each
    # app.state pointed at the cycle's own manager while it was active
    for captured, instance in zip(state_during_cycle, instances):
        assert captured is instance


@pytest.mark.asyncio
async def test_lifespan_session_manager_kwargs_mirror_mcp_settings(monkeypatch):
    """Per-cycle construction passes app=mcp._mcp_server and the json/stateless
    flags matching the FastMCP constructor."""
    module = importlib.reload(mcp_server)
    _RecordingSessionManager.instances = []
    monkeypatch.setattr(module, "StreamableHTTPSessionManager", _RecordingSessionManager)

    app = FastAPI()
    module.setup_mcp_server(app)

    async with app.router.lifespan_context(app):
        pass

    assert len(_RecordingSessionManager.instances) == 1
    kwargs = _RecordingSessionManager.instances[0].init_kwargs
    assert kwargs["app"] is module.mcp._mcp_server
    assert kwargs["json_response"] is True
    assert kwargs["stateless"] is True


@pytest.mark.asyncio
async def test_setup_mcp_server_idempotent_no_double_wrap(monkeypatch):
    """Two setup calls on the same app must not double-wrap the lifespan
    (installed-state guard), so a single cycle still constructs one manager."""
    module = importlib.reload(mcp_server)
    _RecordingSessionManager.instances = []
    monkeypatch.setattr(module, "StreamableHTTPSessionManager", _RecordingSessionManager)

    app = FastAPI()
    module.setup_mcp_server(app)
    module.setup_mcp_server(app)

    async with app.router.lifespan_context(app):
        pass

    assert len(_RecordingSessionManager.instances) == 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_event_cache():
    event_cache_clear()
    yield
    event_cache_clear()


def _build_testbed(monkeypatch, *, auth_user_id: str | None = None):
    """Build a FastAPI app backed by a mocked Memory (caller owns the TestClient).

    *auth_user_id* selects the verify_auth override: None -> admin/no-user;
    a UUID string -> a mock user with that id (tools see str(id) as the scope).
    """
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

    import sys as _sys
    from server.db import SessionLocal

    # mcp_server and its flat-import deps (memory_lock / server_state /
    # entity_permissions) hold their own get_memory_instance references, and under
    # importlib.reload the flat short-name modules can be distinct objects from the
    # server.* alias targets conftest registered. Patch every form present so MCP
    # tools (and run_memory_write_for_memory_id) reach the mocked Memory.
    def _patch_gmi(_mod):
        if _mod is not None and hasattr(_mod, "get_memory_instance"):
            monkeypatch.setattr(_mod, "get_memory_instance", get_memory, raising=False)

    monkeypatch.setattr(module, "get_memory_instance", get_memory)
    for _name in (
        "memory_lock", "server.memory_lock",
        "server_state", "server.server_state",
        "entity_permissions", "server.entity_permissions",
    ):
        _patch_gmi(_sys.modules.get(_name))
    monkeypatch.setattr(module, "_ADD_EXECUTOR", _ImmediateExecutor())

    # Permission/scope guards query the DB; stub them so MCP tools reach the
    # mocked Memory. These tests exercise tool wiring / param-forwarding, not
    # authorization (that lives in test_entity_permissions*). NOTE:
    # reject_bootstrap_memory_mutation is intentionally left real — the default
    # operator below is a non-bootstrap user, so it never rejects; the
    # *_requires_scope tests opt into the bootstrap path explicitly.
    for _guard in (
        "check_memory_scope_permission",
        "authorize_write",
        "check_entity_delete_permission",
        "validate_bulk_admin_operation",
    ):
        monkeypatch.setattr(module, _guard, lambda *a, **k: None)
    monkeypatch.setattr(module, "check_query_permission", lambda filters, *a, **k: filters)
    monkeypatch.setattr(module, "resolve_memory_entities", lambda memory_id: {})
    monkeypatch.setattr(module, "get_entity_or_none", lambda *a, **k: None)
    monkeypatch.setattr(module, "get_visible_entities", lambda *a, **k: [])
    # bulk_delete_memories queries the DB; forward to the mocked SDK call so
    # delete_all_memories tests assert on Memory.delete_all.
    monkeypatch.setattr(
        module,
        "bulk_delete_memories",
        lambda memory, params, op_id, db: memory.delete_all(**params),
    )

    # MCP tools open DB sessions via server_state._session_factory (bypassing
    # FastAPI DI). main.py sets it on import; these tests build a bare FastAPI
    # app without main, so wire up a real SessionLocal pointed at the test DB.
    _ss = _sys.modules.get("server_state") or _sys.modules.get("server.server_state")
    _ss.set_session_factory(SessionLocal)

    app = FastAPI()
    module.setup_mcp_server(app)

    # verify_auth is overridden below, but that override has no Request and so
    # can't set request.state.auth_type. _mcp_request_context reads it to populate
    # mcp_auth_type_var; default it to "disabled" so determine_user resolves the
    # AUTH_DISABLED default user. (For the authed path user is non-None, which
    # determine_user short-circuits on before consulting auth_type.)
    @app.middleware("http")
    async def _mark_disabled(request, call_next):
        request.state.auth_type = "disabled"
        return await call_next(request)

    if auth_user_id is None:
        app.dependency_overrides[module.verify_auth] = lambda: None
    else:
        mock_user = MagicMock()
        mock_user.id = uuid.UUID(auth_user_id)
        app.dependency_overrides[module.verify_auth] = lambda: mock_user

    return module, app, mock_memory, auth_user_id


@pytest.fixture
def mcp_testbed(monkeypatch, default_user):
    module, app, mock_memory, _ = _build_testbed(monkeypatch)
    with TestClient(app) as client:
        _initialize_client(client)
        yield module, client, mock_memory


def _initialize_client(client: TestClient, headers: dict | None = None) -> None:
    response = client.post("/mcp", json=_initialize_payload(), headers=headers or MCP_HEADERS)
    assert response.status_code == 200
    assert response.json()["result"]["protocolVersion"] == "2025-03-26"


@pytest.fixture
def mcp_testbed_authed(monkeypatch, default_user):
    """Like mcp_testbed but verify_auth returns a real User-like object with a known id."""
    module, app, mock_memory, uid = _build_testbed(monkeypatch, auth_user_id=AUTH_USER_ID)
    with TestClient(app) as client:
        _initialize_client(client)
        yield module, client, mock_memory, uid


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


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
    assert "app_id" in descriptions["add_memory"]
    assert "user_id is automatically added to filters" in descriptions["search_memories"]
    assert "user_id is automatically added to filters" in descriptions["get_memories"]
    assert "Update an existing memory" in descriptions["update_memory"]


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------


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
    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "verbatim fact"}],
        user_id="alice",
        metadata={"source": "MCP"},
        infer=False,
    )


def test_add_memory_tool_uses_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(
        client,
        "add_memory",
        {"text": "remember this", "user_id": "alice", "infer": True},
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


def test_add_memory_requires_scope(mcp_testbed, monkeypatch):
    module, client, mock_memory = mcp_testbed
    from auth import _BOOTSTRAP_ADMIN

    # Bootstrap admin cannot author memories (reject_bootstrap_memory_mutation),
    # which is the "no auth user / no scope" rejection this test asserts.
    monkeypatch.setattr(module, "_mcp_resolve_operator", lambda db: (_BOOTSTRAP_ADMIN, True))

    result = _call_tool(client, "add_memory", {"text": "no scope"})
    assert result.get("isError") is True
    mock_memory.add.assert_not_called()


def test_mcp_resolve_operator_raises_when_no_auth_context():
    """No auth context (user=None, auth_type="none") → determine_user returns
    None → _mcp_resolve_operator raises ValueError("Authentication required."),
    which FastMCP surfaces as an isError response.

    Covers the "no auth_type → ValueError" path that the ``*_requires_scope``
    tests skip — they exercise the bootstrap-admin rejection path instead.
    """
    u = mcp_server.mcp_user_var.set(None)
    t = mcp_server.mcp_auth_type_var.set("none")
    try:
        with pytest.raises(ValueError, match="Authentication required."):
            mcp_server._mcp_resolve_operator(None)
    finally:
        mcp_server.mcp_user_var.reset(u)
        mcp_server.mcp_auth_type_var.reset(t)


def test_add_memory_neither_text_nor_messages_rejected(mcp_testbed):
    """Scope present but no content to store is a tool error."""
    _, client, mock_memory = mcp_testbed

    result = _call_tool(client, "add_memory", {"user_id": "alice"})
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


def test_add_memory_allows_messages_without_text(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    _structured(
        client,
        "add_memory",
        {"user_id": "alice", "messages": messages},
    )

    mock_memory.add.assert_called_once()
    assert mock_memory.add.call_args.kwargs["messages"] == messages


def test_add_memory_messages_win_over_text(mcp_testbed):
    """When both text and messages are given, messages is used and text ignored."""
    _, client, mock_memory = mcp_testbed
    messages = [{"role": "user", "content": "from-messages"}]

    _structured(
        client,
        "add_memory",
        {"text": "from-text", "user_id": "alice", "messages": messages, "infer": False},
    )

    mock_memory.add.assert_called_once()
    # `is` would fail: MCP deserializes tool args over JSON-RPC, so the messages
    # list the tool receives is a reconstructed copy. Compare by equality, and
    # confirm the text arg was dropped in favour of messages.
    assert mock_memory.add.call_args.kwargs["messages"] == messages
    assert mock_memory.add.call_args.kwargs["messages"][0]["content"] == "from-messages"


def test_add_memory_default_infer_omits_infer_flag(mcp_testbed):
    """When infer is not passed, infer should not appear in add kwargs."""
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "add_memory",
        {"text": "fact", "user_id": "alice"},
    )

    mock_memory.add.assert_called_once()
    assert "infer" not in mock_memory.add.call_args.kwargs


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


def test_add_memory_defaults_user_id_to_auth_user(mcp_testbed_authed):
    _, client, mock_memory, auth_uid = mcp_testbed_authed

    _call_tool(client, "add_memory", {"text": "remember this"})

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "remember this"}],
        user_id=auth_uid,
        metadata={"source": "MCP"},
    )


def test_add_memory_with_custom_source(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(client, "add_memory", {"text": "tagged", "user_id": "alice", "source": "cursor"})

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "tagged"}],
        user_id="alice",
        metadata={"source": "cursor"},
    )


def test_add_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "add_memory",
        {"text": "decision made", "user_id": "alice", "metadata": {"type": "decision"}},
    )

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "decision made"}],
        user_id="alice",
        metadata={"source": "MCP", "type": "decision"},
    )


def test_add_memory_top_level_source_wins_over_metadata_source(mcp_testbed):
    """An explicit top-level source arg wins over a same-named key in metadata."""
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "add_memory",
        {
            "text": "x",
            "user_id": "alice",
            "source": "cursor",
            "metadata": {"source": "spoof", "extra": 1},
        },
    )

    mock_memory.add.assert_called_once()
    md = mock_memory.add.call_args.kwargs["metadata"]
    assert md["source"] == "cursor"
    assert md["extra"] == 1
    assert "spoof" not in md.values()


def test_add_memory_metadata_source_wins_over_header_source(mcp_testbed):
    """Caller metadata.source beats the request-header source (setdefault, matching REST)."""
    _, client, mock_memory = mcp_testbed
    headers = {**MCP_HEADERS, "x-mem0-source": "CURSOR"}

    _call_tool(
        client,
        "add_memory",
        {"text": "x", "user_id": "alice", "metadata": {"source": "from-metadata"}},
        headers=headers,
    )

    mock_memory.add.assert_called_once()
    assert mock_memory.add.call_args.kwargs["metadata"]["source"] == "from-metadata"


def test_add_memory_header_platform_written_to_metadata(mcp_testbed):
    """Request-header platform (x-mem0-platform) is recorded in metadata when not set by caller."""
    _, client, mock_memory = mcp_testbed
    headers = {**MCP_HEADERS, "x-mem0-platform": "cursor"}

    _call_tool(client, "add_memory", {"text": "x", "user_id": "alice"}, headers=headers)

    mock_memory.add.assert_called_once()
    assert mock_memory.add.call_args.kwargs["metadata"]["platform"] == "cursor"


def test_add_memory_metadata_platform_wins_over_header(mcp_testbed):
    """Caller metadata.platform beats the request-header platform (setdefault, matching REST)."""
    _, client, mock_memory = mcp_testbed
    headers = {**MCP_HEADERS, "x-mem0-platform": "cursor"}

    _call_tool(
        client,
        "add_memory",
        {"text": "x", "user_id": "alice", "metadata": {"platform": "from-metadata"}},
        headers=headers,
    )

    mock_memory.add.assert_called_once()
    assert mock_memory.add.call_args.kwargs["metadata"]["platform"] == "from-metadata"


def test_source_from_x_mem0_source_header(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    headers = {**MCP_HEADERS, "x-mem0-source": "CURSOR"}

    _call_tool(client, "add_memory", {"text": "hdr", "user_id": "alice"}, headers=headers)

    mock_memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "hdr"}],
        user_id="alice",
        metadata={"source": "CURSOR"},
    )


def test_add_memory_default_infer_passes_expiration_date(mcp_testbed):
    """expiration_date is forwarded on the default async add path."""
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "add_memory",
        {"text": "session note", "user_id": "alice", "expiration_date": "2099-12-31"},
    )

    mock_memory.add.assert_called_once()
    call_kwargs = mock_memory.add.call_args.kwargs
    assert call_kwargs["expiration_date"] == "2099-12-31"
    assert "expiration_date" not in (call_kwargs.get("metadata") or {})


def test_add_memory_expiration_date_not_passed_when_none(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "add_memory",
        {"text": "fact", "user_id": "alice", "infer": False},
    )

    mock_memory.add.assert_called_once()
    assert "expiration_date" not in mock_memory.add.call_args.kwargs


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------


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


def test_search_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(client, "search_memories", {"query": "test", "user_id": "alice"})

    mock_memory.search.assert_called_once_with(query="test", filters={"user_id": "alice"})


def test_search_memories_requires_scope(mcp_testbed, monkeypatch):
    """search_memories with no entity scope and no auth user is rejected."""
    module, client, mock_memory = mcp_testbed
    from auth import _BOOTSTRAP_ADMIN

    monkeypatch.setattr(module, "_mcp_resolve_operator", lambda db: (_BOOTSTRAP_ADMIN, True))

    result = _call_tool(client, "search_memories", {"query": "anything"})
    assert result.get("isError") is True
    mock_memory.search.assert_not_called()


def test_search_memories_passes_rerank(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "search_memories",
        {"query": "hello", "user_id": "alice", "rerank": True},
    )

    mock_memory.search.assert_called_once()
    assert mock_memory.search.call_args.kwargs["rerank"] is True


def test_search_memories_passes_show_expired(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "search_memories",
        {"query": "hello", "user_id": "alice", "show_expired": True},
    )

    mock_memory.search.assert_called_once()
    assert mock_memory.search.call_args.kwargs["show_expired"] is True


def test_search_memories_omits_optional_kwargs_when_unset(mcp_testbed):
    """rerank and show_expired are absent from SDK kwargs when not provided."""
    _, client, mock_memory = mcp_testbed

    _structured(client, "search_memories", {"query": "hello", "user_id": "alice"})

    mock_memory.search.assert_called_once()
    kwargs = mock_memory.search.call_args.kwargs
    assert "rerank" not in kwargs
    assert "show_expired" not in kwargs


def test_search_memories_source_param_is_advisory(mcp_testbed):
    """source on read paths is accepted for parity but not forwarded to the SDK."""
    _, client, mock_memory = mcp_testbed

    _structured(client, "search_memories", {"query": "x", "user_id": "alice", "source": "cursor"})

    mock_memory.search.assert_called_once()
    assert "source" not in mock_memory.search.call_args.kwargs


# ---------------------------------------------------------------------------
# get_memories
# ---------------------------------------------------------------------------


def test_get_memories_pagination(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [{"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(5)]

    structured = _structured(client, "get_memories", {"user_id": "alice", "page": 2, "page_size": 2})

    assert structured["count"] == 5
    assert len(structured["results"]) == 2
    assert structured["results"][0]["id"] == "mem-2"


def test_get_memories_page_without_page_size_uses_defaults(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [{"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(25)]

    structured = _structured(client, "get_memories", {"user_id": "alice", "page": 1})

    assert structured["count"] == 25
    assert len(structured["results"]) == 10


def test_get_memories_without_pagination_params_uses_default_first_page(mcp_testbed):
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [{"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(25)]

    structured = _structured(client, "get_memories", {"user_id": "alice"})

    assert structured["count"] == 25
    assert len(structured["results"]) == 10
    assert structured["results"][0]["id"] == "mem-0"


def test_get_memories_page_beyond_range_returns_empty(mcp_testbed):
    """A page past the last item yields empty results but count stays the total."""
    _, client, mock_memory = mcp_testbed
    mock_memory.get_all.return_value = [{"id": f"mem-{i}", "memory": f"m{i}", "user_id": "alice"} for i in range(5)]

    structured = _structured(client, "get_memories", {"user_id": "alice", "page": 10, "page_size": 2})

    assert structured["count"] == 5
    assert structured["results"] == []


def test_get_memories_with_explicit_user_id(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(client, "get_memories", {"user_id": "alice"})

    mock_memory.get_all.assert_called_once_with(filters={"user_id": "alice"})


def test_get_memories_requires_scope(mcp_testbed, monkeypatch):
    module, client, mock_memory = mcp_testbed
    from auth import _BOOTSTRAP_ADMIN

    monkeypatch.setattr(module, "_mcp_resolve_operator", lambda db: (_BOOTSTRAP_ADMIN, True))

    result = _call_tool(client, "get_memories")
    assert result.get("isError") is True
    mock_memory.get_all.assert_not_called()


def test_get_memories_passes_show_expired(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "get_memories",
        {"user_id": "alice", "show_expired": True},
    )

    mock_memory.get_all.assert_called_once()
    assert mock_memory.get_all.call_args.kwargs["show_expired"] is True


def test_get_memories_show_expired_defaults_to_none(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(
        client,
        "get_memories",
        {"user_id": "alice"},
    )

    mock_memory.get_all.assert_called_once()
    assert "show_expired" not in mock_memory.get_all.call_args.kwargs


def test_get_memories_source_param_is_advisory(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _structured(client, "get_memories", {"user_id": "alice", "source": "cursor"})

    mock_memory.get_all.assert_called_once()
    assert "source" not in mock_memory.get_all.call_args.kwargs


# ---------------------------------------------------------------------------
# get_memory
# ---------------------------------------------------------------------------


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


def test_get_memory_non_dict_surfaces_as_tool_error(mcp_testbed):
    """When SDK get() returns a non-dict, MCP reports a tool error (Pydantic validation)."""
    _, client, mock_memory = mcp_testbed
    mock_memory.get.return_value = ["not", "a", "dict"]

    result = _call_tool(client, "get_memory", {"memory_id": "mem-x"})

    mock_memory.get.assert_called_once_with("mem-x")
    assert result.get("isError") is True


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


def test_update_memory_with_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "text": "new text", "metadata": {"type": "revised"}},
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", data="new text", metadata={"type": "revised"})


def test_update_memory_passes_expiration_date(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "text": "updated", "expiration_date": "2099-12-31"},
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", data="updated", expiration_date="2099-12-31")


def test_update_memory_merges_source_into_metadata(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "text": "updated", "source": "cursor"},
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", data="updated", metadata={"source": "cursor"})


def test_update_memory_source_and_metadata_merged(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "text": "updated", "source": "cursor", "metadata": {"type": "note"}},
    )

    mock_memory.update.assert_called_once_with(
        memory_id="mem-1", data="updated", metadata={"source": "cursor", "type": "note"}
    )


def test_update_memory_top_level_source_wins_over_metadata_source(mcp_testbed):
    """An explicit top-level source beats a same-named key inside the metadata bag."""
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {
            "memory_id": "mem-1",
            "source": "cursor",
            "metadata": {"source": "spoof", "type": "note"},
        },
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", metadata={"source": "cursor", "type": "note"})


def test_update_memory_text_optional(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "metadata": {"type": "note"}},
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", metadata={"type": "note"})


def test_update_memory_null_expiration_date_clears(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1", "expiration_date": None},
    )

    mock_memory.update.assert_called_once_with(memory_id="mem-1", expiration_date=None)


def test_update_memory_noop_rejected(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    result = _call_tool(
        client,
        "update_memory",
        {"memory_id": "mem-1"},
    )

    assert result.get("isError") is True
    mock_memory.update.assert_not_called()


def test_update_memory_non_dict_surfaces_as_tool_error(mcp_testbed):
    """When SDK update() returns a non-dict, MCP reports a tool error (Pydantic validation)."""
    _, client, mock_memory = mcp_testbed
    mock_memory.update.return_value = "ok"

    result = _call_tool(client, "update_memory", {"memory_id": "mem-x", "text": "new"})

    mock_memory.update.assert_called_once_with(memory_id="mem-x", data="new")
    assert result.get("isError") is True


# ---------------------------------------------------------------------------
# delete_memory / delete_all_memories / delete_entities
# ---------------------------------------------------------------------------


def test_delete_memory_invokes_sdk(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(client, "delete_memory", {"memory_id": "mem-1"})
    mock_memory.delete.assert_called_once_with("mem-1")
    assert structured["message"] == "Memory deleted successfully!"


def test_delete_all_memories_scoped(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(client, "delete_all_memories", {"user_id": "alice", "agent_id": "bot"})
    mock_memory.delete_all.assert_called_once_with(user_id="alice", agent_id="bot")


def test_delete_all_memories_requires_scope(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    result = _call_tool(client, "delete_all_memories")
    assert result.get("isError") is True
    mock_memory.delete_all.assert_not_called()


def test_delete_all_memories_source_param_is_advisory(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    _call_tool(client, "delete_all_memories", {"user_id": "alice", "source": "cursor"})

    mock_memory.delete_all.assert_called_once()
    assert "source" not in mock_memory.delete_all.call_args.kwargs


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


def test_delete_entities_single_entity(mcp_testbed):
    _, client, mock_memory = mcp_testbed

    structured = _structured(client, "delete_entities", {"user_id": "alice"})
    assert structured["message"] == "Entities deleted successfully, count: 1."
    assert mock_memory.delete_all.call_count == 1


# ---------------------------------------------------------------------------
# list_entities
# ---------------------------------------------------------------------------


def test_list_entities_returns_payload(mcp_testbed):
    module, client, mock_memory = mcp_testbed
    from server.models import Entity

    entities = [Entity(type="user", id="alice", name="alice")]
    with patch.object(module, "get_visible_entities", return_value=entities):
        structured = _structured(client, "list_entities")
    assert structured["count"] == 1
    assert structured["results"][0]["name"] == "alice"


# ---------------------------------------------------------------------------
# list_events / get_event_status
# ---------------------------------------------------------------------------


def test_list_events_filter_and_pagination(mcp_testbed):
    _, client, _ = mcp_testbed
    now = "2026-01-01T00:00:00+00:00"
    event_cache_put("e1", {**make_event_obj("e1", [], now_iso=now, status="SUCCEEDED"), "owner_id": "alice"})
    event_cache_put(
        "e2",
        {
            **make_event_obj("e2", [], now_iso="2026-01-02T00:00:00+00:00", status="SUCCEEDED"),
            "owner_id": "bob",
        },
    )
    event_cache_put(
        "e3",
        {
            **make_event_obj("e3", [], now_iso="2026-01-03T00:00:00+00:00", status="PENDING"),
            "owner_id": "alice",
        },
    )

    listed = _structured(client, "list_events")
    assert listed["count"] == 3
    assert len(listed["results"]) == 3

    paged = _structured(client, "list_events", {"page": 1, "page_size": 2})
    assert paged["count"] == 3
    assert len(paged["results"]) == 2


def test_list_events_filters_by_authenticated_user(mcp_testbed_authed):
    _, client, _, auth_uid = mcp_testbed_authed
    now = "2026-01-01T00:00:00+00:00"
    event_cache_put(
        "e1",
        {**make_event_obj("e1", [], now_iso=now, status="SUCCEEDED"), "owner_id": auth_uid},
    )
    event_cache_put(
        "e2",
        {
            **make_event_obj("e2", [], now_iso="2026-01-02T00:00:00+00:00", status="SUCCEEDED"),
            "owner_id": "other-user",
        },
    )

    listed = _structured(client, "list_events")
    assert listed["count"] == 1
    assert listed["results"][0]["id"] == "e1"


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


def test_list_events_filters_by_event_type(mcp_testbed):
    _, client, _ = mcp_testbed
    now = "2026-01-01T00:00:00+00:00"
    add_event = make_event_obj("e-add", [], now_iso=now, status="SUCCEEDED")
    search_event = make_event_obj("e-search", [], now_iso=now, status="SUCCEEDED")
    search_event["event_type"] = "SEARCH"
    event_cache_put("e-add", add_event)
    event_cache_put("e-search", search_event)

    filtered = _structured(client, "list_events", {"event_type": "ADD"})
    assert filtered["count"] == 1
    assert filtered["results"][0]["id"] == "e-add"


def test_get_event_status_not_found(mcp_testbed):
    _, client, _ = mcp_testbed

    result = _call_tool(client, "get_event_status", {"event_id": "00000000-0000-0000-0000-000000000099"})
    assert result.get("isError") is True


def test_get_event_status_denied_for_other_user_event(mcp_testbed_authed):
    """An event owned by another user is reported as not found to this caller."""
    _, client, _, _ = mcp_testbed_authed
    event = {
        **make_event_obj("e-other", [], now_iso="2026-01-01T00:00:00+00:00", status="SUCCEEDED"),
        "owner_id": "someone-else",
    }
    event_cache_put("e-other", event)

    result = _call_tool(client, "get_event_status", {"event_id": "e-other"})
    assert result.get("isError") is True


# ---------------------------------------------------------------------------
# prompts & per-request context
# ---------------------------------------------------------------------------


def test_prompts_get_memory_assistant(mcp_testbed):
    _, client, _ = mcp_testbed

    response = client.post(
        "/mcp", json=_jsonrpc("prompts/get", {"name": "memory_assistant"}, req_id=2), headers=MCP_HEADERS
    )
    assert response.status_code == 200
    messages = response.json()["result"]["messages"]
    assert any("add_memory" in msg.get("content", {}).get("text", "") for msg in messages)


def test_platform_context_is_taken_from_header(mcp_testbed):
    module, client, _ = mcp_testbed
    captured: dict[str, str | None] = {}

    @module.mcp.tool(name="__test_platform", description="test only")
    def _capture_platform() -> dict[str, str | None]:
        captured["platform"] = module.platform_var.get(None)
        return {"platform": captured["platform"]}

    headers = {**MCP_HEADERS, "x-mem0-platform": "cursor"}

    try:
        structured = _structured(client, "__test_platform", {}, headers=headers)
        assert captured["platform"] == "cursor"
        assert structured["platform"] == "cursor"
    finally:
        module.mcp._tool_manager._tools.pop("__test_platform", None)


# ---------------------------------------------------------------------------
# compat helpers (entities normalization)
# ---------------------------------------------------------------------------


def test_normalize_list_result_shapes():
    """normalize_vector_store_list should handle all documented backend return shapes."""
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
    row = MagicMock(payload={"data": 1})
    mock_memory = MagicMock()
    mock_memory.vector_store.list.return_value = [row, None, MagicMock(payload={"data": 2})]

    with patch("server.compat.entities.get_memory_instance", return_value=mock_memory):
        payloads = iter_payloads()

    assert payloads == [{"data": 1}, {"data": 2}]
    assert len(payloads) == 2
