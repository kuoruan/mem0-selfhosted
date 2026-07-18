import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(50), default="local", server_default="local")
    role: Mapped[str] = mapped_column(String(20), default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    key_prefix: Mapped[str] = mapped_column(String(12))
    key_hash: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    method: Mapped[str] = mapped_column(String(16))
    path: Mapped[str] = mapped_column(String(512))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    auth_type: Mapped[str] = mapped_column(String(32), default="none")
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RefreshTokenJti(Base):
    __tablename__ = "refresh_token_jtis"

    jti: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Settings(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


class OidcLink(Base):
    __tablename__ = "oidc_links"
    __table_args__ = (UniqueConstraint("idp_issuer", "idp_sub", name="uq_oidc_links_issuer_sub"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    provider: Mapped[str] = mapped_column(String(100))
    idp_issuer: Mapped[str] = mapped_column(String(512))
    idp_sub: Mapped[str] = mapped_column(String(512))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Entity(Base):
    """An entity namespace (user/agent/app/run).

    ``id`` is the external identifier (e.g. ``alice``, ``riley``).
    ``pk`` is the internal UUID primary key.
    """

    __tablename__ = "entities"
    __table_args__ = (
        Index("ix_entities_owner_id", "owner_id"),
        Index("ix_entities_parent_pk", "parent_pk"),
        # Partial unique indexes (mirror migration 007): user/app are globally
        # unique on (type, id); agent/run are unique per parent on
        # (type, parent_pk, id). Duplicates raise IntegrityError, which the
        # service layer uses for race-safe first-claim / orphan re-claim.
        Index(
            "uq_entities_type_id_global",
            "type",
            "id",
            unique=True,
            postgresql_where=text("type IN ('user', 'app')"),
            sqlite_where=text("type IN ('user', 'app')"),
        ),
        Index(
            "uq_entities_type_parent_id",
            "type",
            "parent_pk",
            "id",
            unique=True,
            postgresql_where=text("type IN ('agent', 'run')"),
            sqlite_where=text("type IN ('agent', 'run')"),
        ),
    )

    pk: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    id: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(20))
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    parent_pk: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.pk", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class EntityPermission(Base):
    """An explicit grant (read/write/admin) on an Entity to a dashboard user."""

    __tablename__ = "entity_permissions"
    __table_args__ = (
        UniqueConstraint("entity_pk", "grantee_id", name="uq_entity_permissions_entity_grantee"),
        Index("ix_entity_permissions_grantee_id", "grantee_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_new_uuid)
    entity_pk: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.pk", ondelete="CASCADE"))
    grantee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    permission: Mapped[str] = mapped_column(String(16))
    grantor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
