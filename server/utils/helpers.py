"""General-purpose helper utilities for the mem0 server."""

import re
from typing import Any
from urllib.parse import urlparse

_WILDCARD = "*"


def is_wildcard(value: Any) -> bool:
    """Return True if *value* is the wildcard sentinel ``"*"``."""
    return value == _WILDCARD


def is_http_url(url: str) -> bool:
    """Return ``True`` if *url* starts with ``http://`` or ``https://`` (case-insensitive)."""
    return url.lower().startswith(("http://", "https://"))


def is_safe_redirect(url: str | None) -> bool:
    """Return ``True`` if *url* is a safe relative redirect target.

    Only relative paths (no scheme, no netloc) are allowed. Rejects whitespace,
    control characters, and backslashes that browsers normalize into open-redirect
    vectors.
    """
    if not url:
        return False
    # Block whitespace/control characters which some browsers normalize, enabling open redirect
    if any(c.isspace() for c in url):
        return False
    # Block backslashes: browsers normalize \ to /, enabling open redirect
    if "\\" in url:
        return False
    parsed = urlparse(url)
    # Must be a relative path: no scheme, no netloc
    if parsed.scheme or parsed.netloc:
        return False
    # Must start with /
    if not url.startswith("/"):
        return False
    return True


def sanitize_for_log(value: str) -> str:
    """Reduce *value* to ``[A-Za-z0-9_.-]`` so untrusted text cannot forge log entries.

    Newlines and control characters in user-supplied input (e.g. an OIDC provider
    path segment) are replaced with ``_`` before reaching a log line.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def normalize_results(raw: Any) -> list[Any]:
    """Normalise SDK / vector-store output to a plain ``list``.

    Accepts ``{"results": [...]}``, a bare ``list``, or anything else
    (returned as an empty list).
    """
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    return []


def unwrap_result(raw: Any) -> Any:
    """Unwrap a single result from ``mem.get()``: the first element if *raw* is a
    non-empty list, otherwise *raw* itself.
    """
    if isinstance(raw, list) and raw:
        return raw[0]
    return raw


def normalize_vector_store_list(raw: Any) -> list[Any]:
    """Unwrap vector store ``list()`` output to a flat list of rows.

    Handles the three return formats:
    - tuple (qdrant: ``(points, next_offset)``)
    - list-of-lists (most stores: ``List[List[OutputData]]``)
    - flat list
    """
    if not raw:
        return []
    if isinstance(raw, tuple):
        return raw[0] if isinstance(raw[0], list) else []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return raw[0]
    if isinstance(raw, list):
        return raw
    return []


def paginate_vector_store(store, *, filters=None, batch_size=1000):
    """Yield batches from vector store ``list()`` using skip pagination.

    Handles qdrant cursor-based pagination and numeric skip pagination.
    Each batch is a flat list of rows (unwrapped via
    ``normalize_vector_store_list``).
    """
    skip = 0
    while True:
        raw = store.list(filters=filters, top_k=batch_size, skip=skip)
        batch = normalize_vector_store_list(raw)
        if not batch:
            return
        yield batch
        if isinstance(raw, tuple) and len(raw) == 2:
            skip = raw[1] if raw[1] is not None else None
        elif len(batch) < batch_size:
            return
        else:
            skip += len(batch)
        if skip is None:
            return
