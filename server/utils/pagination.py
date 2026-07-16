"""Generic pagination envelope shared across the REST API.

``paginate_response`` wraps a page of items in the ``{count, next, previous,
results}`` envelope used by ``/entities``, ``/users``, and the compat API.
Moved out of ``compat.responses`` so non-compat routers don't pull the whole
compat response module.
"""

from typing import Any, Dict, List

from fastapi import Request


def build_page_url(request: Request, *, page: int, page_size: int) -> str:
    return str(request.url.include_query_params(page=page, page_size=page_size))


def paginate_response(
    request: Request,
    items: List[Any],
    page: int,
    page_size: int,
    *,
    total: int | None = None,
) -> Dict[str, Any]:
    """Wrap a list of items in the SDK-compatible pagination envelope.

    When *total* is provided, *items* is treated as the already-paginated page
    slice (e.g. produced by a DB-level ``LIMIT``/``OFFSET``) and is not re-sliced;
    otherwise the full list is sliced here and *total* is derived from its length.
    """
    start = (page - 1) * page_size
    if total is None:
        total = len(items)
        page_items = items[start : start + page_size]
    else:
        page_items = items
    return {
        "count": total,
        "next": build_page_url(request, page=page + 1, page_size=page_size)
        if start + page_size < total
        else None,
        "previous": build_page_url(request, page=page - 1, page_size=page_size)
        if page > 1
        else None,
        "results": page_items,
    }
