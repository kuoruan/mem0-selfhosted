"""Shared utilities for the client-compatible API (``routers/compat``) and MCP server.

This package consolidates logic previously duplicated across ``routers/compat.py``
and ``mcp_server.py``:

- **scope** – entity-parameter collection, scope resolution, and filter merging.
- **responses** – normalising SDK return values into consistent list / dict shapes.
- **decorators** – ``upstream_guard`` for converting unhandled exceptions to 502s.
- **entities** – entity-listing aggregation used by both the compat router and the
  MCP ``list_entities`` tool (breaks the former ``mcp_server → routers.compat``
  reverse dependency).
"""
