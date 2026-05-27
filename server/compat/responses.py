"""Response-shaping helpers for REST and MCP handlers.

Normalises the varied return shapes from the ``Memory`` SDK into consistent
list or dict formats expected by client SDKs and the MCP protocol.
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

API_UNSUPPORTED_DETAIL = "This API is not supported by the self-hosted server."


def unsupported_api_error() -> HTTPException:
    """Return a fresh 501 exception for unsupported self-hosted endpoints."""
    return HTTPException(status_code=501, detail=API_UNSUPPORTED_DETAIL)


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
