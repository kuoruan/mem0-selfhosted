from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import verify_auth
from compat.entities import aggregate_entity_buckets, iter_payloads
from errors import upstream_error
from schemas import MessageResponse
from memory_lock import run_memory_write

router = APIRouter(prefix="/entities", tags=["entities"])

SCAN_LIMIT = 10_000

EntityType = Literal["user", "agent", "run"]
TYPE_TO_FIELD: dict[EntityType, str] = {"user": "user_id", "agent": "agent_id", "run": "run_id"}


class Entity(BaseModel):
    id: str
    type: EntityType
    total_memories: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@router.get("", response_model=list[Entity])
def list_entities(_auth=Depends(verify_auth)):
    buckets = aggregate_entity_buckets(iter_payloads(limit=SCAN_LIMIT), TYPE_TO_FIELD)

    return [
        Entity(id=entity_id, type=entity_type, **data)
        for (entity_type, entity_id), data in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


@router.delete("/{entity_type}/{entity_id}", response_model=MessageResponse)
def delete_entity(entity_type: EntityType, entity_id: str, _auth=Depends(verify_auth)):
    try:
        run_memory_write(
            lambda memory: memory.delete_all(**{TYPE_TO_FIELD[entity_type]: entity_id}),
            {TYPE_TO_FIELD[entity_type]: entity_id},
        )
    except Exception:
        raise upstream_error()
    return MessageResponse(message="Entity deleted")
