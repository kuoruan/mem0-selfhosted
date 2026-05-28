"""Generic utilities shared across compat modules (no domain-specific logic)."""

from datetime import datetime, timezone
from typing import Any, Optional


def now_iso(timestamp: Optional[str] = None) -> str:
    """Return an ISO-8601 UTC timestamp string, or *timestamp* when provided."""
    return timestamp or datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    """Format a datetime as ISO-8601, or ``None`` when *value* is missing."""
    return value.isoformat() if value else None
