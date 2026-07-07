"""Tests for write-path app_id migration and API key resolution.

Verifies that all scripts writing to the Mem0 API:
1. Pass app_id as a top-level parameter (not in metadata)
2. Do NOT include project_id in metadata
3. Include branch in metadata when available
4. Use resolve_api_key() for key resolution with userConfig fallback
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_auto_import_post_memory_uses_app_id():
    """auto_import.post_memory sends app_id top-level, not metadata.project_id."""
    from auto_import import post_memory

    captured = {}

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        captured.update(body)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = post_memory(
            api_key="test-key",
            content="test content",
            user_id="testuser",
            filename="CLAUDE.md",
            project_id="my-project",
            branch="main",
        )

    assert result is True
    assert captured["app_id"] == "my-project"
    assert captured["user_id"] == "testuser"
    assert "project_id" not in captured.get("metadata", {})
    assert captured["metadata"]["type"] == "project_profile"
    assert captured["metadata"]["branch"] == "main"
    assert captured["infer"] is False


def test_auto_import_post_memory_omits_empty_branch():
    """auto_import.post_memory skips branch in metadata when empty."""
    from auto_import import post_memory

    captured = {}

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        captured.update(body)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        post_memory("key", "content", "user", "FILE.md", "proj", branch="")

    assert "branch" not in captured.get("metadata", {})


def test_on_pre_compact_store_memory_uses_app_id():
    """on_pre_compact.store_memory sends app_id top-level."""
    from on_pre_compact import store_memory

    captured = {}

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        captured.update(body)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = store_memory(
            api_key="test-key",
            content="session state content",
            user_id="testuser",
            source="pre-compaction",
            session_id="sess-123",
            project_id="my-project",
            branch="feat/auth",
        )

    assert result is True
    assert captured["app_id"] == "my-project"
    assert captured["user_id"] == "testuser"
    assert "project_id" not in captured.get("metadata", {})
    assert captured["metadata"]["type"] == "session_state"
    assert captured["metadata"]["source"] == "pre-compaction"
    assert captured["metadata"]["branch"] == "feat/auth"
    assert "expiration_date" in captured


def test_capture_compact_summary_store_uses_app_id():
    """capture_compact_summary.store_summary sends app_id top-level."""
    from capture_compact_summary import store_summary

    captured = {}

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        captured.update(body)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = store_summary(
            api_key="test-key",
            summary="compact summary text",
            user_id="testuser",
            session_id="sess-456",
            project_id="my-project",
            branch="main",
        )

    assert result is True
    assert captured["app_id"] == "my-project"
    assert captured["user_id"] == "testuser"
    assert "project_id" not in captured.get("metadata", {})
    assert captured["metadata"]["type"] == "compact_summary"
    assert captured["metadata"]["branch"] == "main"
    assert captured["infer"] is True
    assert "expiration_date" in captured


def test_no_metadata_project_id_anywhere():
    """Ensure none of the write functions put project_id in metadata."""
    from auto_import import post_memory
    from capture_compact_summary import store_summary
    from on_pre_compact import store_memory

    bodies = []

    def mock_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        bodies.append(body)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        post_memory("k", "c", "u", "f", "proj", "br")
        store_memory("k", "c", "u", "src", "sid", "proj", "br")
        store_summary("k", "s", "u", "sid", "proj", "br")

    for i, body in enumerate(bodies):
        metadata = body.get("metadata", {})
        assert "project_id" not in metadata, f"Write function #{i} still has metadata.project_id"
        assert body.get("app_id") == "proj", f"Write function #{i} missing app_id top-level"


def test_resolve_api_key_prefers_env_var(monkeypatch):
    """resolve_api_key returns MEM0_API_KEY when both are set."""
    from _identity import resolve_api_key

    monkeypatch.setenv("MEM0_API_KEY", "direct-key")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", "fallback-key")
    assert resolve_api_key() == "direct-key"


def test_resolve_api_key_falls_back_to_plugin_option(monkeypatch):
    """resolve_api_key falls back to CLAUDE_PLUGIN_OPTION_MEM0_API_KEY."""
    from _identity import resolve_api_key

    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", "fallback-key")
    assert resolve_api_key() == "fallback-key"


def test_resolve_api_key_returns_empty_when_neither_set(monkeypatch):
    """resolve_api_key returns empty string when no key is available."""
    from _identity import resolve_api_key

    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    monkeypatch.setattr("_identity._extract_key_from_shell_profiles", lambda: "")
    assert resolve_api_key() == ""


def test_resolve_api_key_falls_back_to_shell_profile(monkeypatch):
    """resolve_api_key extracts key from shell profile when env vars are empty."""
    from _identity import resolve_api_key

    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MEM0_API_KEY", raising=False)
    monkeypatch.setattr("_identity._extract_key_from_shell_profiles", lambda: "m0-from-profile")
    assert resolve_api_key() == "m0-from-profile"


# ---------------------------------------------------------------------------
# resolve_mcp_url / resolve_api_url
# ---------------------------------------------------------------------------


def test_resolve_mcp_url_prefers_env_var(monkeypatch):
    from _identity import resolve_mcp_url

    monkeypatch.setenv("MEM0_MCP_URL", "http://custom:8888/mcp/")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MCP_URL", "http://plugin/mcp/")
    assert resolve_mcp_url() == "http://custom:8888/mcp/"


def test_resolve_mcp_url_falls_back_to_plugin_option(monkeypatch):
    from _identity import resolve_mcp_url

    monkeypatch.delenv("MEM0_MCP_URL", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MCP_URL", "http://plugin/mcp/")
    assert resolve_mcp_url() == "http://plugin/mcp/"


def test_resolve_mcp_url_default_when_unset(monkeypatch):
    from _identity import resolve_mcp_url

    monkeypatch.delenv("MEM0_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MCP_URL", raising=False)
    assert resolve_mcp_url() == "https://mcp.mem0.ai/mcp/"


def test_resolve_api_url_prefers_env_var(monkeypatch):
    from _identity import resolve_api_url

    monkeypatch.setenv("MEM0_API_URL", "http://custom:8888")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_URL", "http://plugin/api")
    assert resolve_api_url() == "http://custom:8888"


def test_resolve_api_url_falls_back_to_plugin_option(monkeypatch):
    from _identity import resolve_api_url

    monkeypatch.delenv("MEM0_API_URL", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_URL", "http://plugin/api")
    assert resolve_api_url() == "http://plugin/api"


def test_resolve_api_url_default_when_unset(monkeypatch):
    from _identity import resolve_api_url

    monkeypatch.delenv("MEM0_API_URL", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_URL", raising=False)
    assert resolve_api_url() == "https://api.mem0.ai"


def test_resolve_mcp_url_rejects_invalid_scheme(monkeypatch, caplog):
    from _identity import resolve_mcp_url

    monkeypatch.setenv("MEM0_MCP_URL", "ftp://bad-scheme/mcp/")
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_MCP_URL", raising=False)
    with caplog.at_level("WARNING", logger="mem0._identity"):
        assert resolve_mcp_url() == "https://mcp.mem0.ai/mcp/"
    assert "does not start with http:// or https://" in caplog.text


def test_resolve_api_url_rejects_invalid_scheme(monkeypatch, caplog):
    from _identity import resolve_api_url

    monkeypatch.setenv("MEM0_API_URL", "file:///etc/passwd")
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_URL", raising=False)
    with caplog.at_level("WARNING", logger="mem0._identity"):
        assert resolve_api_url() == "https://api.mem0.ai"
    assert "does not start with http:// or https://" in caplog.text


def test_resolve_mcp_url_plugin_option_rejects_invalid_scheme(monkeypatch, caplog):
    from _identity import resolve_mcp_url

    monkeypatch.delenv("MEM0_MCP_URL", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MCP_URL", "gopher://old/mcp/")
    with caplog.at_level("WARNING", logger="mem0._identity"):
        assert resolve_mcp_url() == "https://mcp.mem0.ai/mcp/"
    assert "CLAUDE_PLUGIN_OPTION_MCP_URL does not start with" in caplog.text


def test_resolve_api_url_plugin_option_rejects_invalid_scheme(monkeypatch, caplog):
    from _identity import resolve_api_url

    monkeypatch.delenv("MEM0_API_URL", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_URL", "ws://websocket/api")
    with caplog.at_level("WARNING", logger="mem0._identity"):
        assert resolve_api_url() == "https://api.mem0.ai"
    assert "CLAUDE_PLUGIN_OPTION_API_URL does not start with" in caplog.text
