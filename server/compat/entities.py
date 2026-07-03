"""Entity-listing aggregation shared by the compat router and MCP server.

Breaks the former ``mcp_server → routers.compat`` reverse dependency by
providing ``list_entities_payload`` in a neutral location.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from compat.scope import COMPAT_TYPE_TO_FIELD
from compat.utils import format_iso_timestamp, parse_iso_timestamp
from server_state import get_memory_instance

CompatEntityType = Literal["user", "agent", "app", "run"]

SCAN_LIMIT = 10_000
MAX_AUTOSCAN_LIMIT = 1_000_000

logger = logging.getLogger("mem0.server.compat.entities")


class CompatEntity(BaseModel):
    """Entity summary aligned with ``GET /v1/entities`` and MemoryClient envelopes."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier for the entity.")
    name: str = Field(description="Display name of the entity.")
    type: CompatEntityType = Field(description="Entity kind: user, agent, app, or run.")
    total_memories: int = Field(description="Total memories associated with this entity.")
    created_at: Optional[str] = Field(default=None, description="Earliest memory timestamp (ISO 8601).")
    updated_at: Optional[str] = Field(default=None, description="Latest memory timestamp (ISO 8601).")
    owner: str = Field(default="self-hosted", description="Owner label for hosted API compatibility.")
    organization: str = Field(default="self-hosted", description="Organization label for hosted API compatibility.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional entity metadata.")

    @classmethod
    def from_bucket(
        cls,
        entity_type: CompatEntityType,
        entity_id: str,
        *,
        total_memories: int,
        created_at: Optional[datetime],
        updated_at: Optional[datetime],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CompatEntity":
        return cls(
            id=entity_id,
            name=entity_id,
            type=entity_type,
            total_memories=total_memories,
            created_at=format_iso_timestamp(created_at),
            updated_at=format_iso_timestamp(updated_at),
            metadata=metadata or {},
        )


def normalize_vector_store_list(raw: Any) -> list:
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


def _vector_store_row_count(vector_store: Any) -> Optional[int]:
    """Best-effort count lookup from ``vector_store.col_info()``."""
    col_info = getattr(vector_store, "col_info", None)
    if not callable(col_info):
        return None
    try:
        info = col_info()
    except Exception:
        return None
    for key in ("count", "points_count", "vectors_count"):
        count = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
        if isinstance(count, int) and count > 0:
            return count
    return None


def _effective_scan_limit(vector_store: Any, requested_limit: int) -> int:
    """Resolve scan limit using store count when available."""
    store_count = _vector_store_row_count(vector_store)
    if store_count is None:
        return max(1, requested_limit)
    return max(1, max(requested_limit, store_count))


def _is_truncated_page(raw: Any) -> bool:
    """Return whether ``vector_store.list()`` likely returned a paged/truncated result."""
    return isinstance(raw, tuple) and len(raw) > 1 and raw[1] is not None


def _scan_rows_with_adaptive_limit(vector_store: Any, initial_limit: int) -> list[Any]:
    """Read rows with adaptive top_k retries when backend signals truncation.

    Some stores return ``(rows, next_offset)`` but do not expose an offset argument
    on ``list()``. In that case, progressively increase ``top_k`` to reduce
    truncation risk.
    """
    store_count = _vector_store_row_count(vector_store)
    current_limit = _effective_scan_limit(vector_store, initial_limit)
    attempts = 0

    while True:
        raw_rows = vector_store.list(top_k=current_limit)
        rows = normalize_vector_store_list(raw_rows)
        if not _is_truncated_page(raw_rows):
            return rows

        attempts += 1
        logger.warning(
            "vector_store.list() returned a paged result while building entities; "
            "retrying with larger top_k=%s (attempt=%s)",
            current_limit,
            attempts,
        )

        if store_count is not None and current_limit >= store_count:
            logger.warning(
                "entity aggregation may still be truncated because backend paging "
                "requires cursor continuation and list() has no offset support"
            )
            return rows

        next_limit = current_limit * 2
        if store_count is not None:
            next_limit = min(next_limit, store_count)
        else:
            next_limit = min(next_limit, MAX_AUTOSCAN_LIMIT)

        if next_limit <= current_limit:
            logger.warning(
                "entity aggregation may be truncated; adaptive top_k retries reached limit=%s",
                current_limit,
            )
            return rows
        current_limit = next_limit


def iter_payloads(*, limit: int = SCAN_LIMIT) -> list[dict[str, Any]]:
    """Return raw vector-store payloads for all stored memories."""
    vector_store = get_memory_instance().vector_store
    rows = _scan_rows_with_adaptive_limit(vector_store, limit)
    return [getattr(row, "payload", None) or {} for row in rows if row is not None]


def aggregate_entity_buckets(
    payloads: Iterable[dict[str, Any]],
    type_to_field: Mapping[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate memory counts and created/updated timestamps by entity type and id."""
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"total_memories": 0, "created_at": None, "updated_at": None}
    )

    for payload in payloads:
        created = parse_iso_timestamp(payload.get("created_at"))
        updated = parse_iso_timestamp(payload.get("updated_at")) or created

        for entity_type, field in type_to_field.items():
            value = payload.get(field)
            if not value:
                continue
            bucket = buckets[(entity_type, str(value))]
            bucket["total_memories"] += 1
            if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                bucket["created_at"] = created
            if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                bucket["updated_at"] = updated

    return dict(buckets)


def list_entities_payload() -> list[CompatEntity]:
    """Aggregate memory counts and timestamps by entity (user / agent / app / run).

    Returns validated models compatible with the hosted platform entity schema.
    """
    buckets = aggregate_entity_buckets(iter_payloads(), COMPAT_TYPE_TO_FIELD)

    return [
        CompatEntity.from_bucket(
            entity_type,
            entity_id,
            total_memories=data["total_memories"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata={},
        )
        for (entity_type, entity_id), data in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
