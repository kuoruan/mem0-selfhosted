"""Entity-listing aggregation shared by the compat router and MCP server.

Breaks the former ``mcp_server → routers.compat`` reverse dependency by
providing ``list_entities_payload`` in a neutral location.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

from entity import EntityType, TYPE_TO_FIELD
from pydantic import BaseModel, ConfigDict, Field

from compat.utils import format_iso_timestamp, parse_iso_timestamp
from utils.helpers import paginate_vector_store
from server_state import get_memory_instance

SCAN_LIMIT = 10_000

logger = logging.getLogger("mem0.server.compat.entities")


class CompatEntity(BaseModel):
    """Entity summary aligned with ``GET /v1/entities`` and MemoryClient envelopes."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier for the entity.")
    name: str = Field(description="Display name of the entity.")
    type: EntityType = Field(description="Entity kind: user, agent, app, or run.")
    created_at: Optional[str] = Field(default=None, description="Earliest memory timestamp (ISO 8601).")
    updated_at: Optional[str] = Field(default=None, description="Latest memory timestamp (ISO 8601).")
    owner: str = Field(default="self-hosted", description="Owner label for hosted API compatibility.")
    organization: str = Field(default="self-hosted", description="Organization label for hosted API compatibility.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional entity metadata.")

    @classmethod
    def from_bucket(
        cls,
        entity_type: EntityType,
        entity_id: str,
        entity_name: Optional[str] = None,
        *,
        created_at: Optional[datetime],
        updated_at: Optional[datetime],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CompatEntity":
        return cls(
            id=entity_id,
            name=entity_name or entity_id,
            type=entity_type,
            created_at=format_iso_timestamp(created_at),
            updated_at=format_iso_timestamp(updated_at),
            metadata=metadata or {},
        )


def _scan_rows_with_adaptive_limit(vector_store: Any, initial_limit: int) -> list[Any]:
    """Read all rows using ``skip`` pagination."""
    all_rows: list[Any] = []
    for batch in paginate_vector_store(vector_store, batch_size=initial_limit):
        all_rows.extend(batch)
    return all_rows


def iter_payloads(*, limit: int = SCAN_LIMIT) -> list[dict[str, Any]]:
    """Return raw vector-store payloads for all stored memories."""
    # Direct vector_store access: needs raw payloads (not Memory's wrapped format)
    # across all memories with no entity scope; Memory exposes no equivalent.
    vector_store = get_memory_instance().vector_store
    rows = _scan_rows_with_adaptive_limit(vector_store, limit)
    return [getattr(row, "payload", None) or {} for row in rows if row is not None]


def aggregate_entity_buckets(
    payloads: Iterable[dict[str, Any]],
    type_to_field: Mapping[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate memory counts and created/updated timestamps by entity type and id."""
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"created_at": None, "updated_at": None}
    )

    for payload in payloads:
        created = parse_iso_timestamp(payload.get("created_at"))
        updated = parse_iso_timestamp(payload.get("updated_at")) or created

        for entity_type, field in type_to_field.items():
            value = payload.get(field)
            if not value:
                continue
            bucket = buckets[(entity_type, str(value))]
            if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                bucket["created_at"] = created
            if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                bucket["updated_at"] = updated

    return dict(buckets)


def list_entities_payload() -> list[CompatEntity]:
    """Aggregate memory counts and timestamps by entity (user / agent / app / run).

    Returns validated models compatible with the hosted platform entity schema.
    """
    buckets = aggregate_entity_buckets(iter_payloads(), TYPE_TO_FIELD)

    return [
        CompatEntity.from_bucket(
            entity_type,
            entity_id,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata={},
        )
        for (entity_type, entity_id), data in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
