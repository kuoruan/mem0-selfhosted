"""General-purpose helper utilities for the mem0 server."""

import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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


def extract_memory_id(row: Any) -> str | None:
    """Extract the memory id from a vector-store row.

    Handles dict rows (``id`` or ``_id`` key) and object rows (``.id`` attr).

    Note: duplicated from ``mem0.memory.utils.extract_memory_id`` to avoid
    coupling the server package to the OSS SDK's internal module layout.
    Keep both copies in sync when changing the extraction logic.
    """
    if row is None:
        return None
    if isinstance(row, dict):
        mid = row.get("id")
        if mid is None:
            mid = row.get("_id")
    else:
        mid = getattr(row, "id", None)
    return str(mid) if mid is not None else None


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

    Handles qdrant cursor-based pagination (2-tuple with cursor offset) and
    numeric skip pagination.  Detects stores that ignore ``skip`` (same first
    id across batches) to avoid infinite loops.  Each batch is a flat list of
    rows (unwrapped via ``normalize_vector_store_list``).
    """
    skip = 0
    prev_first_id = None
    while True:
        raw = store.list(filters=filters, top_k=batch_size, skip=skip)
        batch = normalize_vector_store_list(raw)
        if not batch:
            return
        # Detect stores that ignore skip: same first id → infinite loop
        first_id = extract_memory_id(batch[0])
        if first_id == prev_first_id:
            return
        prev_first_id = first_id
        yield batch
        # Advance offset: cursor (qdrant 2-tuple) or numeric skip
        if isinstance(raw, tuple) and len(raw) == 2:
            cursor = raw[1]
            if cursor is None:
                return
            skip = cursor
        elif len(batch) < batch_size:
            return
        else:
            skip += len(batch)


def safe_count(memory, filters=None) -> Optional[int]:
    """Count memories via ``memory.count()``, returning ``None`` on transient failure.

    Programming errors (NameError/AttributeError/SyntaxError/TypeError) are
    re-raised; any other exception (store unavailable, timeout, ...) is logged
    and returns ``None`` so list endpoints degrade gracefully instead of 500ing.
    The result is normalized to ``int | None`` (defensive against stores that
    return a non-int).

    Caution: ``AttributeError`` is re-raised on the assumption that the store
    client is not a lazy proxy. If a store uses a connection-pool proxy that
    surfaces transient connection failures as ``AttributeError``, this will
    500 instead of degrading. Re-evaluate if such a store is introduced.
    """
    try:
        c = memory.count(filters=filters)
    except (NameError, AttributeError, SyntaxError, TypeError):
        raise
    except Exception:
        logger.exception("safe_count: count() failed for filters=%r", filters)
        return None
    return c if isinstance(c, int) else None
