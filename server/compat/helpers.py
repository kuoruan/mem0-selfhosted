"""Helper utilities shared by compat routers."""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from server_state import get_memory_instance
from utils.helpers import normalize_results, safe_count, unwrap_result
from utils.pagination import paginate_response


def normalize_results_dict(raw: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalise SDK output to ``{"results": [...]}`` and merge *extra* into the result.

    If *raw* is already a dict, its existing fields are preserved and only
    ``results`` is normalised; *extra* is applied last and may override any key.
    """
    if isinstance(raw, dict):
        base: Dict[str, Any] = {**raw, "results": normalize_results(raw)}
    else:
        base = {"results": normalize_results(raw)}
    if extra:
        base.update(extra)
    return base


def build_search_kwargs(
    filters: Dict[str, Any],
    top_k: Optional[int],
    threshold: Optional[float],
    rerank: Optional[bool] = None,
    show_expired: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for Memory.search() from common request fields."""
    kwargs: Dict[str, Any] = {"filters": filters}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if threshold is not None:
        kwargs["threshold"] = threshold
    if rerank is not None:
        kwargs["rerank"] = rerank
    if show_expired is not None:
        kwargs["show_expired"] = show_expired
    return kwargs


def resolve_existing(mem: Any, memory_id: str) -> Dict[str, Any]:
    """Fetch an existing memory and return its dict, or raise 404."""
    raw = mem.get(memory_id)
    item = unwrap_result(raw)
    if not isinstance(item, dict):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    return item


def paginated_get_all(
    request: Request,
    page: int,
    page_size: int,
    *,
    filters: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return a page of memories using server-side skip pagination.

    *filters* is required (keyword-only).  Optional keys (``show_expired``, …)
    are forwarded to ``memory.get_all()``.

    Note: ``skip`` is a visible-row offset (post-expiry-filter), so deep pages
    cost O(page * page_size / batch_size) store round-trips — a known
    trade-off of post-filter pagination.
    """
    memory = get_memory_instance()
    start = (page - 1) * page_size
    # Fetch one extra row to detect whether a next page exists, independent of
    # the (advisory) count().
    kwargs["top_k"] = page_size + 1
    kwargs["skip"] = start
    raw = memory.get_all(filters=filters, **kwargs)
    fetched = normalize_results(raw)
    results = fetched[:page_size]
    has_more = len(fetched) > page_size
    # count() is advisory: surface it when available; otherwise fall back to a
    # scanned lower bound (start + fetched rows). On the final page this
    # converges to the exact visible total.
    c = safe_count(memory, filters)
    total = c if c is not None else start + len(fetched)
    return paginate_response(request, results, page, page_size, total=total, has_more=has_more)
