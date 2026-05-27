"""Helper utilities shared by compat routers."""

from typing import Any, Dict, Optional

from fastapi import HTTPException


def build_search_kwargs(
    filters: Dict[str, Any],
    top_k: Optional[int],
    threshold: Optional[float],
    rerank: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build keyword arguments for Memory.search() from common request fields."""
    kwargs: Dict[str, Any] = {"filters": filters}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if threshold is not None:
        kwargs["threshold"] = threshold
    if rerank is not None:
        kwargs["rerank"] = rerank
    return kwargs


def resolve_existing(mem: Any, memory_id: str) -> Dict[str, Any]:
    """Fetch an existing memory and return its dict, or raise 404."""
    raw = mem.get(memory_id)
    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    return item


def merge_and_update(
    mem: Any, memory_id: str, *, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """Read current memory, merge text/metadata changes, write back."""
    existing = resolve_existing(mem, memory_id)
    final_text = text if text is not None else (existing.get("memory") or existing.get("text") or "")
    merged = {**(existing.get("metadata") or {}), **(metadata or {})}
    return mem.update(memory_id=memory_id, data=final_text, metadata=merged)
