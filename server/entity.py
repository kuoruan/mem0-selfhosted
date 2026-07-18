"""Entity domain primitives for the self-hosted server.

Central, dependency-light definition of the entity model shared across
``compat``, ``entity_permissions``, ``routers`` and ``memory_lock``:

- The four entity namespaces (``user`` / ``agent`` / ``app`` / ``run``) and their
  payload field names (``TYPE_TO_FIELD`` / ``FIELD_TO_TYPE`` / ``ENTITY_PARAMS``).
- Parent-scoped type grouping (``SCOPED_ENTITY_TYPES`` / ``is_scoped_entity_type``).
- Entity-id normalization (``canonicalize_entity_id``).
- ``:``-delimited user sub-namespace parsing (``top_level_user_id`` /
  ``user_prefixes``).
- Field-keyed params <-> ``{entity_type: entity_id}`` conversion
  (``params_to_entities``).

Kept free of DB / model / server-state imports so any module can depend on it
without creating a cycle. Ownership, permission and persistence logic lives in
``entity_permissions``; this module holds only the entity domain primitives.
"""

import uuid
from typing import Any, Literal

from fastapi import HTTPException

EntityType = Literal["user", "agent", "app", "run"]

# Entity type -> payload field name.
TYPE_TO_FIELD: dict[EntityType, str] = {
    "user": "user_id",
    "agent": "agent_id",
    "app": "app_id",
    "run": "run_id",
}

# Payload field name -> entity type (inverse of TYPE_TO_FIELD).
FIELD_TO_TYPE: dict[str, EntityType] = {field: etype for etype, field in TYPE_TO_FIELD.items()}

# All valid entity types (the keys of TYPE_TO_FIELD).
VALID_ENTITY_TYPES: frozenset[str] = frozenset(TYPE_TO_FIELD)

# All entity payload field names (the values of TYPE_TO_FIELD).
ENTITY_PARAMS: frozenset[str] = frozenset(TYPE_TO_FIELD.values())

# Entity types unique per parent user (not globally): ``agent`` / ``run``. They
# are auto-created on first write under a user entity, need a ``parent_entity_id``
# to scope lookups/counts, and do not support explicit grants or ownership
# transfer. Contrast with ``user`` / ``app`` (globally unique, grantable).
SCOPED_ENTITY_TYPES: frozenset[str] = frozenset({"agent", "run"})


def is_scoped_entity_type(entity_type: str) -> bool:
    """Whether *entity_type* is parent-scoped (``agent`` / ``run``)."""
    return entity_type in SCOPED_ENTITY_TYPES


def canonicalize_entity_id(entity_type: str, entity_id: str) -> str:
    """Normalize an entity id: strip, reject empty, canonicalize UUIDs for ``user``."""
    if entity_type not in TYPE_TO_FIELD:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity type '{entity_type}'. Must be one of: {', '.join(sorted(TYPE_TO_FIELD))}.",
        )
    if not isinstance(entity_id, str):
        raise HTTPException(status_code=400, detail="Entity id must be a string.")
    normalized = entity_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Entity id cannot be empty.")
    if entity_type == "user":
        parts = normalized.split(":")
        try:
            parts[0] = str(uuid.UUID(parts[0]))
        except (ValueError, TypeError):
            pass
        return ":".join(parts)
    return normalized


def top_level_user_id(entity_id: str) -> str:
    """Return the first ``:``-delimited segment of a user entity_id."""
    return entity_id.split(":")[0]


def user_prefixes(entity_id: str) -> list[str]:
    """Return all ``:``-delimited prefixes of *entity_id* from longest to shortest.

    ``A:B:C`` -> ``["A:B:C", "A:B", "A"]``
    """
    parts = entity_id.split(":")
    prefixes = []
    for i in range(len(parts), 0, -1):
        prefixes.append(":".join(parts[:i]))
    return prefixes


def params_to_entities(entity_params: dict[str, Any]) -> dict[str, str]:
    """Convert a field-keyed entity-param dict to ``{entity_type: entity_id}``.

    Non-entity fields (anything not in ``FIELD_TO_TYPE``) are dropped.
    """
    return {FIELD_TO_TYPE[field]: value for field, value in entity_params.items() if field in FIELD_TO_TYPE}
