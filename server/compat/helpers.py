"""Helper utilities shared by compat routers."""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from utils.helpers import normalize_results, unwrap_result


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
