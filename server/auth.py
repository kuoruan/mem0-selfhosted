import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from db import get_db
from models import APIKey, RefreshTokenJti, User

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").lower() in {"1", "true", "yes", "on"}

# Pre-computed bcrypt hash of b"dummy" (rounds=12), used only for timing-safe dummy verification.
DUMMY_HASH: bytes = b"$2b$12$k/g9O8usX37dgo75GqFaG.nC5QjJnh5e9NhW43zoWPjoaDl21gB1q"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False


def dummy_verify_password() -> None:
    """Burn the same bcrypt cycles as a real verify so login timing doesn't leak whether an email exists."""
    bcrypt.checkpw(b"dummy", DUMMY_HASH)


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, prefix, hash)."""
    raw = secrets.token_urlsafe(32)
    full_key = f"m0sk_{raw}"
    prefix = full_key[:12]
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt(rounds=12)).decode()
    return full_key, prefix, key_hash


def verify_api_key_hash(plain_key: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain_key.encode(), hashed.encode())
    except ValueError:
        return False


def _get_secret() -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured.")
    return JWT_SECRET


def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, db: Session) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    jti = uuid.uuid4()
    db.add(RefreshTokenJti(jti=jti, user_id=uuid.UUID(user_id), expires_at=expire))
    db.commit()
    payload = {"sub": user_id, "exp": expire, "jti": str(jti), "type": "refresh"}
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def consume_refresh_jti(jti: str, db: Session) -> None:
    """Atomically mark a refresh token's jti as used. Raises 401 if missing, already used, or expired.

    The conditional UPDATE closes the read-check-write race: concurrent replays of the same
    token race on a single row, so at most one update affects a row and the rest see rowcount 0.
    """
    try:
        jti_uuid = uuid.UUID(jti)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid.")
    now = datetime.now(timezone.utc)
    result = db.execute(
        update(RefreshTokenJti)
        .where(
            RefreshTokenJti.jti == jti_uuid,
            RefreshTokenJti.used_at.is_(None),
            RefreshTokenJti.expires_at > now,
        )
        .values(used_at=now)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=401, detail="Refresh token is no longer valid.")
    db.commit()


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _authorization_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization")
    scheme, credentials = get_authorization_scheme_param(authorization)
    if scheme.lower() == "token" and credentials:
        return credentials
    return None


async def _api_key_or_token(
    x_api_key: str | None = Depends(api_key_header),
    token: str | None = Depends(_authorization_token),
) -> str | None:
    if x_api_key is not None:
        return x_api_key

    return token


def _is_jwt(token: str) -> bool:
    return token.startswith("eyJ") and token.count(".") == 2


def _mark_auth_type(request: Request, auth_type: str) -> None:
    request.state.auth_type = auth_type


def _get_default_user(db: Session) -> User | None:
    return db.scalar(select(User).order_by(User.created_at.asc()))


def _resolve_user_from_jwt(token: str, db: Session) -> User:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type.")
    user = db.get(User, payload.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def _resolve_user_from_api_key(key: str, db: Session) -> User:
    prefix = key[:12] if len(key) >= 12 else key
    candidates = (
        db.execute(select(APIKey).where(APIKey.key_prefix == prefix, APIKey.revoked_at.is_(None))).scalars().all()
    )

    for candidate in candidates:
        if verify_api_key_hash(key, candidate.key_hash):
            candidate.last_used_at = datetime.now(timezone.utc)
            db.commit()
            user = db.get(User, candidate.created_by)
            if user is None:
                raise HTTPException(status_code=401, detail="API key owner not found.")
            return user

    raise HTTPException(status_code=401, detail="Invalid API key.")


async def verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Depends(_api_key_or_token),
    db: Session = Depends(get_db),
) -> User | None:
    """Authenticate via JWT, X-API-Key, or legacy ADMIN_API_KEY. Returns User or None."""
    if credentials is not None:
        if _is_jwt(credentials.credentials):
            _mark_auth_type(request, "bearer")
            return _resolve_user_from_jwt(credentials.credentials, db)
        # Not a JWT — try as API key below
        if x_api_key is None:
            x_api_key = credentials.credentials

    if x_api_key is not None:
        if ADMIN_API_KEY and secrets.compare_digest(x_api_key, ADMIN_API_KEY):
            _mark_auth_type(request, "admin_api_key")
            return None
        _mark_auth_type(request, "api_key")
        return _resolve_user_from_api_key(x_api_key, db)

    if AUTH_DISABLED:
        _mark_auth_type(request, "disabled")
        return None

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide a Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(
    request: Request,
    user: User | None = Depends(verify_auth),
    db: Session = Depends(get_db),
) -> User:
    """Like verify_auth but guarantees a non-None User. Use for endpoints that require auth."""
    if user is None:
        if getattr(request.state, "auth_type", "none") in {"admin_api_key", "disabled"}:
            default_user = _get_default_user(db)
            if default_user is not None:
                return default_user
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user
