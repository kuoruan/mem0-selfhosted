"""Create users, api_keys, and oidc_links tables

Revision ID: 001
Revises: None
Create Date: 2026-04-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        # password_hash is nullable: OIDC users have no local password.
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("auth_provider", sa.String(50), nullable=False, server_default="local"),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("key_prefix", sa.String(12), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # oidc_links: IdP identity → local user mapping for OIDC login.
    op.create_table(
        "oidc_links",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("idp_issuer", sa.String(512), nullable=False),
        sa.Column("idp_sub", sa.String(512), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("idp_issuer", "idp_sub", name="uq_oidc_links_issuer_sub"),
    )
    op.create_index("ix_oidc_links_user_id", "oidc_links", ["user_id"])


def downgrade() -> None:
    op.drop_table("oidc_links")
    op.drop_table("api_keys")
    op.drop_table("users")
