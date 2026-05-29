"""HTTP/MCP response envelopes for the client-compatible API.

Pagination wrappers, add-route status bodies, and other response shapes built
after SDK values are normalised (see ``helpers``).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi import Request

from compat.helpers import normalize_results_dict

API_UNSUPPORTED_DETAIL = "This API is not supported by the self-hosted server."
logger = logging.getLogger("mem0.server.compat.responses")


def unsupported_api_error() -> HTTPException:
    """Return a fresh 501 exception for unsupported self-hosted endpoints."""
    return HTTPException(status_code=501, detail=API_UNSUPPORTED_DETAIL)


def sync_add_response(raw: Any) -> Dict[str, Any]:
    """Envelope for synchronous v3/MCP add (``infer=False``)."""
    return normalize_results_dict(
        raw, extra={"message": "Memory added successfully.", "event_id": None, "status": "SUCCEEDED"}
    )


def pending_add_response(event_id: str) -> Dict[str, Any]:
    """Envelope returned immediately when add is queued for background processing."""
    return {
        "message": "Memory processing has been queued for background execution.",
        "event_id": event_id,
        "status": "PENDING",
    }


def resolve_optional_pagination(
    page: Optional[int],
    page_size: Optional[int],
    *,
    default_page: int = 1,
    default_page_size: int = 50,
    max_page_size: int = 100,
) -> Optional[tuple[int, int]]:
    """Resolve MCP-style optional pagination params.

    Returns ``None`` when neither *page* nor *page_size* is given (return all items).
    When either is provided, defaults missing values to *default_page* / *default_page_size*
    and clamps *page_size* to ``[1, max_page_size]``.
    """
    if page is None and page_size is None:
        return None
    effective_page = default_page if page is None else max(1, page)
    raw_size = default_page_size if page_size is None else page_size
    effective_page_size = min(max(1, raw_size), max_page_size)
    return effective_page, effective_page_size


def build_page_url(request: Request, *, page: int, page_size: int) -> str:
    return str(request.url.include_query_params(page=page, page_size=page_size))


def paginate_response(
    request: Request,
    items: List[Any],
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """Wrap a list of items in the SDK-compatible pagination envelope."""
    total = len(items)
    start = (page - 1) * page_size
    return {
        "count": total,
        "next": build_page_url(request, page=page + 1, page_size=page_size) if start + page_size < total else None,
        "previous": build_page_url(request, page=page - 1, page_size=page_size) if page > 1 else None,
        "results": items[start : start + page_size],
    }


def warn_unsupported_fields(fields: Optional[List[str]], endpoint: str) -> None:
    """Log a warning when 'fields' projection is requested but not supported by the OSS SDK."""
    if fields:
        logger.warning(
            "%s: 'fields' projection is not supported by the OSS SDK and will be ignored. Requested fields: %s",
            endpoint,
            fields,
        )
