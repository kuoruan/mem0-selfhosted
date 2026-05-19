"""Entity-listing aggregation shared by the compat router and MCP server.

Breaks the former ``mcp_server → routers.compat`` reverse dependency by
providing ``list_entities_payload`` in a neutral location.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from compat.scope import COMPAT_TYPE_TO_FIELD
from server_state import get_memory_instance

SCAN_LIMIT = 10_000


def _normalize_list_result(raw: Any) -> list:
    """Unpack different ``vector_store.list()`` return shapes into a flat list of rows.

    Backend return shapes:
    - PGVector / Chroma: ``[[OutputData, …]]`` — list containing one list of rows
    - Qdrant: ``([ScoredPoint, …], next_page_offset)`` — tuple of (rows, offset)
    - Others: flat ``[row, …]`` or empty ``[]``
    """
    if not raw:
        return []
    if isinstance(raw, tuple):
        return raw[0] if isinstance(raw[0], list) else []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return raw[0]
    if isinstance(raw, list):
        return raw
    return []


def iter_payloads() -> list[dict[str, Any]]:
    """Return raw vector-store payloads for all stored memories."""
    rows = _normalize_list_result(get_memory_instance().vector_store.list(top_k=SCAN_LIMIT))
    return [getattr(row, "payload", None) or {} for row in rows if row is not None]


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def list_entities_payload() -> list[dict[str, Any]]:
    """Aggregate memory counts and timestamps by entity (user / agent / app / run).

    Returns a list of dicts compatible with the ``MemoryClient`` SDK envelope
    (includes ``owner``, ``organization``, ``metadata`` fields).
    """
    buckets: dict[Any, dict[str, Any]] = defaultdict(
        lambda: {"total_memories": 0, "created_at": None, "updated_at": None, "metadata": {}}
    )

    for payload in iter_payloads():
        created = parse_timestamp(payload.get("created_at"))
        updated = parse_timestamp(payload.get("updated_at")) or created

        for entity_type, field in COMPAT_TYPE_TO_FIELD.items():
            value = payload.get(field)
            if not value:
                continue
            bucket = buckets[(entity_type, str(value))]
            bucket["total_memories"] += 1
            if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                bucket["created_at"] = created
            if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                bucket["updated_at"] = updated

    return [
        {
            "id": entity_id,
            "name": entity_id,
            "type": entity_type,
            "total_memories": data["total_memories"],
            "created_at": data["created_at"].isoformat() if data["created_at"] else None,
            "updated_at": data["updated_at"].isoformat() if data["updated_at"] else None,
            "owner": "self-hosted",
            "organization": "self-hosted",
            "metadata": data["metadata"],
        }
        for (entity_type, entity_id), data in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
