"""Shared layer for the client-compatible API (``routers/compat``) and MCP server.

Consolidates logic previously duplicated across ``routers/compat.py`` and
``mcp_server.py``. Import from submodules directly (this package does not
re-export symbols).

Modules
-------
scope
    Entity-parameter collection, scope resolution, and filter-tree merging.
utils
    Generic helpers only (ISO timestamps); no domain-specific logic.
responses
    Normalise SDK return values, pagination helpers, and list/dict envelopes.
decorators
    ``upstream_guard`` — map unhandled exceptions to HTTP errors.
entities
    Entity-listing aggregation (vector-store scan, bucket roll-up,
    ``CompatEntity``); shared by the compat router, ``routers/entities``, and
    MCP ``list_entities``.
events
    In-process TTL cache and ``CompatEvent`` models for async v3 add polling
    (``GET /v1/event/{id}``, ``GET /v1/events``).
tasks
    Background workers (e.g. ``run_v3_add_memory_task``) that update the event
    cache after writes complete.
helpers
    Router-oriented Memory helpers (search kwargs, fetch/update merge).
metadata
    Metadata merge rules for v1/v2/v3 add and update routes.
requests
    ``RequestMeta`` / ``request_meta`` — per-request fields from Mem0 HTTP
    headers (source, platform, categories).
"""
