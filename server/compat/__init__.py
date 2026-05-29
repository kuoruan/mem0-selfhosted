"""Shared layer for the client-compatible REST router and MCP server.

``routers/compat.py`` and ``mcp_server.py`` should delegate reusable logic here
instead of duplicating it. Import from submodules directly — this package does
not re-export symbols.

Module map (by responsibility, not an API index)
------------------------------------------------
scope
    Entity scoping (``user_id`` / ``agent_id`` / ``app_id`` / ``run_id``),
    filter trees, and validation shared by read and write paths.

entities
    Discover and aggregate entities from vector-store payloads for list/detail
    APIs (compat router, ``routers/entities``, MCP).

helpers
    Cross-route ``Memory`` helpers — SDK result normalisation, search kwargs,
    fetch-or-404, update merge.

responses
    Pagination wrappers and HTTP/MCP response bodies (status envelopes, 501
    stubs, unsupported-field warnings).

events
    Process-local synthetic event cache for deferred writes — models, TTL
    storage, access control, and helpers to register pollable events.

tasks
    Background workers that complete deferred operations and update the event
    cache.

metadata
    Rules for merging caller metadata with request headers and version-specific
    add/update fields.

requests
    Per-request context derived from Mem0 client HTTP headers.

decorators
    Shared guards and exception mapping (e.g. upstream provider errors).

utils
    Generic, domain-free helpers (timestamps, ``drop_none`` for kwargs).

When adding code, place it by *what it does*, not by which router called it
first. Update this doc only when a new submodule appears or a module's role
changes — not when individual functions move within the package.
"""
