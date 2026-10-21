from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parents[1]
CORE_ROOT = HOST.parent / "agent-plugin-core"
sys.path.insert(0, str(CORE_ROOT))
from build.build import build  # noqa: E402

SPEC = importlib.util.spec_from_file_location("cursor_adapter", HOST / "hooks" / "adapter.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.mark.parametrize(
    ("event", "action"),
    [
        ("sessionStart", "session-start"),
        ("beforeSubmitPrompt", "user-prompt"),
        ("postToolUse", "post-tool"),
        ("postToolUseFailure", "post-tool-failure"),
        ("afterAgentResponse", "assistant-stop"),
        ("stop", "stop"),
        ("sessionEnd", "session-end"),
        ("preCompact", "pre-compact"),
    ],
)
def test_normalizes_cursor_events(event: str, action: str) -> None:
    normalized = adapter.normalize(
        {
            "conversation_id": "conversation-1",
            "workspace_roots": ["/worktree"],
            "tool_output": "done",
            "text": "assistant response",
        },
        event,
    )

    assert normalized["action"] == action
    assert normalized["payload"]["session_id"] == "conversation-1"
    assert normalized["payload"]["cwd"] == "/worktree"
    assert normalized["payload"]["tool_response"] == "done"
    assert normalized["payload"]["last_assistant_message"] == "assistant response"


def test_after_agent_response_does_not_return_internal_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.hook_runner, "default_record_stop", lambda store, payload: (object(), "session"))

    assert adapter._record_response(object(), {}) is None


def test_cursor_hooks_use_native_flat_entries() -> None:
    hooks = json.loads((HOST / "hooks" / "hooks.json").read_text(encoding="utf-8"))

    assert hooks["version"] == 1
    assert not {"subagentStart", "subagentStop"} & hooks["hooks"].keys()
    assert set(hooks["hooks"]) >= {
        "sessionStart",
        "beforeSubmitPrompt",
        "postToolUse",
        "stop",
        "sessionEnd",
    }
    assert all("hooks" not in entry for entries in hooks["hooks"].values() for entry in entries)


def test_native_cursor_bundle_is_self_contained(tmp_path: Path) -> None:
    root = build("cursor", "native", tmp_path / "cursor")

    manifest = json.loads((root / ".cursor-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["hooks"] == "./hooks/hooks.json"
    assert "agents" not in manifest
    assert (root / "hooks" / "adapter.py").is_file()
    assert not (root / "agents").exists()
    assert not any(path.is_symlink() for path in root.rglob("*"))


def test_selfhosted_api_url_reaches_proxy_and_hooks(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A self-hosted `api_url` set in the plugin options must reach both the
    platform proxy and the hook runtime; otherwise the self-hosted key is sent
    to the public api.mem0.ai / mcp.mem0.ai defaults."""
    mcp = json.loads((HOST / "mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["mem0-platform"]["env"]["PLUGIN_OPTION_API_URL"] == "${api_url}"

    hooks = json.loads((HOST / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    session_start = hooks["hooks"]["sessionStart"][0]["command"]
    assert 'PLUGIN_OPTION_API_URL="${api_url}"' in session_start

    monkeypatch.setattr(adapter.hook_runner, "run", lambda **kwargs: 0)
    monkeypatch.setattr(adapter, "configure_harness", lambda *a, **k: None)
    monkeypatch.setattr(adapter.telemetry, "init", lambda *a, **k: None)
    monkeypatch.setenv("PLUGIN_OPTION_API_URL", "http://mem0.internal:8888")
    monkeypatch.setattr(sys, "argv", ["adapter.py", "sessionStart"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))

    adapter.main()

    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["env"]
    assert env["PLUGIN_OPTION_API_URL"] == "http://mem0.internal:8888"
