"""Generic utilities shared across compat modules (no domain-specific logic)."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *d* with all ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}


def iso_timestamp_or_now(timestamp: Optional[str] = None) -> str:
    """Return *timestamp* when provided, otherwise the current UTC time as ISO-8601."""
    return timestamp or datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string to ``datetime``, or ``None`` when missing/invalid."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def format_iso_timestamp(value: Optional[datetime]) -> Optional[str]:
    """Format *value* as ISO-8601, or ``None`` when *value* is missing."""
    return value.isoformat() if value else None
