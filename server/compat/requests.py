"""Per-request metadata extracted from well-known Mem0 HTTP headers.

Usage::

    from compat.requests import RequestMeta, request_meta

    @router.post("/v1/memories")
    def add(body: ..., meta: RequestMeta = Depends(request_meta)):
        if meta.source:
            ...
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

KNOWN_CLIENTS: dict[str, tuple[str, ...]] = {
    # --- AI coding agents & CLI tools ---
    "claude-code": ("claude-code", "claudecode", "claude-cli"),  # Anthropic – terminal agent
    "codex": ("codex",),  # OpenAI – coding agent
    "aider": ("aider",),  # open-source CLI agent
    "devin": ("devin",),  # Cognition – autonomous agent
    # --- AI IDEs & editors ---
    "cursor": ("cursor",),  # Cursor Inc. – agent-first IDE
    "windsurf": ("windsurf", "codeium"),  # Codeium – AI IDE
    "trae": ("trae",),  # ByteDance – AI IDE
    "zed": ("zed",),  # Zed Industries – fast editor with AI
    "replit": ("replit",),  # Replit – cloud IDE + agent
    # --- IDE extensions & plugins ---
    "copilot": ("copilot", "github-copilot"),  # GitHub/Microsoft – IDE assistant
    "cline": ("cline",),  # open-source VS Code agent
    "continue": ("continue",),  # Continue – open-source autocomplete
    "roo-code": ("roo-code", "roo-code-"),  # Roo Code – VS Code extension
    "cody": ("cody", "sourcegraph"),  # Sourcegraph – code intelligence
    "augment": ("augment",),  # Augment Code – coding assistant
    "amazon-q": ("amazon-q", "amazonq"),  # AWS – IDE assistant
    "tabnine": ("tabnine",),  # Tabnine – code completion
    "supermaven": ("supermaven",),  # Supermaven – fast completion
    "pearai": ("pearai",),  # PearAI – open-source AI editor
    "jetbrains-ai": ("junie", "jetbrains-ai"),  # JetBrains – AI assistant
    # --- China-based AI coding tools ---
    "lingma": ("lingma", "tongyi"),  # Alibaba – 通义灵码
    "comate": ("comate",),  # Baidu – 文心快码
    "codebuddy": ("codebuddy",),  # Tencent – CodeBuddy
    "codegeex": ("codegeex",),  # Zhipu AI – CodeGeeX
    "marscode": ("marscode",),  # ByteDance – 豆包 MarsCode
}


@dataclass(frozen=True)
class RequestMeta:
    """Typed container for Mem0-specific HTTP header values."""

    source: Optional[str] = None
    """Normalised (upper-cased) value of ``X-Mem0-Source`` — the access channel through which mem0 is called
    (e.g. ``"CLI"``, ``"MCP"``, ``"OPENCLAW"``). Distinct from :attr:`platform`, which identifies the specific
    host tool within that channel."""

    client_language: Optional[str] = None
    """Value of ``X-Mem0-Client-Language`` — the language of the mem0 SDK making the request (e.g. ``"python"``, ``"node"``)."""

    client_version: Optional[str] = None
    """Value of ``X-Mem0-Client-Version`` — the version of the mem0 SDK making the request (e.g. ``"1.2.3"``)."""

    platform: Optional[str] = None
    """Normalised (lower-cased) value of ``X-Mem0-Platform`` — the host tool invoking mem0 (e.g. ``"claude-code"``, ``"cursor"``, ``"codex"``)."""

    ua_tool_name: Optional[str] = None
    """AI coding tool name detected from the ``User-Agent`` header via :func:`_parse_ua_tool_name` (e.g. ``"claude-code"``, ``"cursor"``), or ``None``."""


def _parse_ua_tool_name(user_agent: Optional[str]) -> Optional[str]:
    """Detect known AI clients from a ``User-Agent`` string.

    Checks for patterns listed in :data:`KNOWN_CLIENTS`.  Returns the
    canonical client name, or ``None`` when *user_agent* is empty, ``None``,
    or does not contain a recognised client.
    """
    if not user_agent:
        return None

    ua = user_agent.lower()
    for name, patterns in KNOWN_CLIENTS.items():
        for pattern in patterns:
            if pattern in ua:
                return name

    return None


def _get_header(request: Request, name: str) -> Optional[str]:
    """Fetch a header value, strip whitespace, and return ``None`` if empty."""
    value = request.headers.get(name)
    if not value:
        return None
    value = value.strip()
    return value if value else None


def request_meta(request: Request) -> RequestMeta:
    """FastAPI dependency — parse well-known Mem0 request headers into :class:`RequestMeta`."""
    raw_source = _get_header(request, "x-mem0-source")
    raw_platform = _get_header(request, "x-mem0-platform")
    raw_user_agent = _get_header(request, "user-agent")
    return RequestMeta(
        source=raw_source.upper() if raw_source else None,
        client_language=_get_header(request, "x-mem0-client-language"),
        client_version=_get_header(request, "x-mem0-client-version"),
        platform=raw_platform.lower() if raw_platform else None,
        ua_tool_name=_parse_ua_tool_name(raw_user_agent) if raw_user_agent else None,
    )
