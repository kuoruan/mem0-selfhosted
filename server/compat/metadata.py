"""Metadata merge helpers for compat routes."""

from typing import Any, Dict, List, Optional

from compat.utils import drop_none


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
    if source is None and platform is None and categories is None:
        return metadata

    merged: Dict[str, Any] = dict(metadata or {})
    if source is not None:
        merged.setdefault("source", source)
    if platform is not None:
        merged.setdefault("platform", platform)
    if categories is not None:
        merged["categories"] = categories
    return merged


def build_v3_add_extra_metadata(
    *,
    custom_categories: Optional[List[Dict[str, Any]]],
    source: Optional[str],
) -> Dict[str, Any]:
    """Build the v3 add body fields that belong in metadata."""
    return drop_none(
        {
            "custom_categories": custom_categories,
            "source": source,
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
    if source is None and platform is None and extra_metadata is None:
        return metadata

    merged: Dict[str, Any] = dict(metadata or {})
    if source is not None:
        merged.setdefault("source", source)
    if platform is not None:
        merged.setdefault("platform", platform)
    if extra_metadata is not None:
        merged.update(extra_metadata)
    return merged


def build_extraction_prompt(
    *,
    custom_instructions: Optional[str],
    agent_custom_instructions: Optional[str],
    includes: Optional[str],
    excludes: Optional[str],
    has_agent_scope: bool,
) -> Optional[str]:
    """Merge instruction body fields into a single ``prompt`` for ``Memory.add()``.

    ``agent_custom_instructions`` wins when ``has_agent_scope`` is true;
    ``includes``/``excludes`` are appended as constraints.
    """
    base = agent_custom_instructions if (has_agent_scope and agent_custom_instructions) else custom_instructions

    constraints: List[str] = []
    if includes:
        constraints.append(f"Include only: {includes}")
    if excludes:
        constraints.append(f"Exclude: {excludes}")

    if not base and not constraints:
        return None

    parts: List[str] = []
    if base:
        parts.append(base)
    if constraints:
        parts.append("Extraction constraints:\n- " + "\n- ".join(constraints))
    return "\n\n".join(parts)
