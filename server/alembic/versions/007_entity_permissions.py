"""Create entities and entity_permissions tables

Entity-level ownership and permission isolation: each (entity_type, id) namespace
is owned by a dashboard user; other users can be granted read/write/admin
permissions. See docs/entity-permissions-plan.md.

Hierarchy:
- user/app: globally unique id, parent_pk is NULL.
- agent/run: unique per parent_pk (the user entity), owner_user_id is copied from parent.

Column naming:
- pk: internal UUID primary key
- id: external identifier string (e.g. "alice", "riley")
- name: display name (optional)
- parent_pk: FK to entities.pk for agent/run -> user hierarchy

Revision ID: 007
Revises: 006
Create Date: 2026-07-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column("pk", sa.Uuid(), primary_key=True),
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("owner_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "parent_pk",
            sa.Uuid(),
            sa.ForeignKey("entities.pk", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "type IN ('user', 'agent', 'app', 'run')",
            name="ck_entities_type",
        ),
    )
    op.create_index("ix_entities_owner_user_id", "entities", ["owner_user_id"])
    op.create_index("ix_entities_parent_pk", "entities", ["parent_pk"])

    # Partial unique indexes: user/app are globally unique; agent/run are unique per parent.
    op.create_index(
        "uq_entities_type_id_global",
        "entities",
        ["type", "id"],
        unique=True,
        postgresql_where=sa.text("type IN ('user', 'app')"),
        sqlite_where=sa.text("type IN ('user', 'app')"),
    )
    op.create_index(
        "uq_entities_type_parent_id",
        "entities",
        ["type", "parent_pk", "id"],
        unique=True,
        postgresql_where=sa.text("type IN ('agent', 'run')"),
        sqlite_where=sa.text("type IN ('agent', 'run')"),
    )

    op.create_table(
        "entity_permissions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "entity_pk",
            sa.Uuid(),
            sa.ForeignKey("entities.pk", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("permission", sa.String(16), nullable=False),
        sa.Column("granted_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "permission IN ('read', 'write', 'admin')",
            name="ck_entity_permissions_permission",
        ),
        sa.UniqueConstraint("entity_pk", "user_id", name="uq_entity_permissions_entity_user"),
    )
    op.create_index("ix_entity_permissions_user_id", "entity_permissions", ["user_id"])


def downgrade() -> None:
    op.drop_table("entity_permissions")
    op.drop_table("entities")