"""Metadata merge helpers for compat routes."""

from typing import Any, Dict, List, Optional

from compat.responses import drop_none


def merge_v1_add_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    source: Optional[str],
    platform: Optional[str],
    categories: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """Merge v1 add metadata using the original three-layer precedence.

    Priority (low -> high):
    1) Header-injected values (``source``/``platform``) only fill missing keys.
       This is why ``setdefault`` is used.
    2) Existing ``metadata`` from the request body is preserved.
    3) Explicit v1 body field ``categories`` always wins and overwrites.
    """
    if not source and not platform and not categories:
        return metadata

    merged: Dict[str, Any] = dict(metadata or {})
    if source:
        merged.setdefault("source", source)
    if platform:
        merged.setdefault("platform", platform)
    if categories:
        merged["categories"] = categories
    return merged


def build_v3_add_extra_metadata(
    *,
    custom_categories: Optional[List[Dict[str, Any]]],
    custom_instructions: Optional[str],
    structured_data_schema: Optional[Dict[str, Any]],
    timestamp: Optional[int],
    source: Optional[str],
    deduced_memories: Optional[List[Any]],
) -> Dict[str, Any]:
    """Build v3 explicit metadata fields from request body."""
    return drop_none(
        {
            "custom_categories": custom_categories,
            "custom_instructions": custom_instructions,
            "structured_data_schema": structured_data_schema,
            "timestamp": timestamp,
            "source": source,
            "deduced_memories": deduced_memories,
        }
    )


def merge_v3_add_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    source: Optional[str],
    platform: Optional[str],
    extra_metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge v3 add metadata using the original three-layer precedence.

    Priority (low -> high):
    1) Header-injected values (``source``/``platform``) only fill missing keys.
       This is why ``setdefault`` is used.
    2) Existing ``metadata`` from the request body is preserved.
    3) ``extra_metadata`` from dedicated v3 body fields always wins via ``update``.
    """
    if not source and not platform and not extra_metadata:
        return metadata

    merged: Dict[str, Any] = dict(metadata or {})
    if source:
        merged.setdefault("source", source)
    if platform:
        merged.setdefault("platform", platform)
    if extra_metadata:
        merged.update(extra_metadata)
    return merged
