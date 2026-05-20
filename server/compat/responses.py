"""Response-shaping helpers for REST and MCP handlers.

Normalises the varied return shapes from the ``Memory`` SDK into consistent
list or dict formats expected by client SDKs and the MCP protocol.
"""

from typing import Any, Dict, List

from fastapi import HTTPException

API_UNSUPPORTED = HTTPException(
    status_code=501,
    detail="This API is not supported by the self-hosted server.",
)


def drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *d* with all ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


def normalize_results(raw: Any) -> List[Any]:
    """Normalise SDK output to a plain ``list``.

    Accepts ``{"results": [...]}``, a bare ``list``, or anything else
    (returned as an empty list).
    """
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    return []


def normalize_results_dict(raw: Any) -> Dict[str, Any]:
    """Normalise SDK output to ``{"results": [...]}``."""
    if isinstance(raw, dict) and "results" in raw:
        return raw
    if isinstance(raw, list):
        return {"results": raw}
    return {"results": []}
