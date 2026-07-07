"""Pluggable short-lived storage for the OIDC Authorization Code Flow.

Holds two single-use, TTL-bounded artifacts:
- PKCE/nonce auth-flow state (``OidcStateStore``)
- the post-callback exchange-code → refresh-token map (``OidcExchangeStore``)

Both default to an in-memory ``TTLCache`` (suitable for single-instance
deployments). To run multi-instance, subclass the ABCs against a shared backend
(Redis/DB) and select it via ``OIDC_STATE_STORE`` (only ``memory`` ships today).
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cachetools import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class OidcStateData:
    """Data persisted during an OIDC auth flow."""

    code_verifier: str
    provider: str
    redirect_uri: str | None = None
    next_url: str | None = None
    nonce: str | None = None
    expires_at: float = 0.0


class OidcStateStore(ABC):
    """Abstract base for OIDC state storage."""

    @abstractmethod
    async def save(self, state: str, data: OidcStateData, ttl_seconds: int = 600) -> None:
        """Save state data with a TTL (default 10 minutes)."""

    @abstractmethod
    async def load(self, state: str) -> OidcStateData | None:
        """Load and return state data, or None if expired/missing."""

    @abstractmethod
    async def delete(self, state: str) -> None:
        """Remove state data after use."""

    async def consume(self, state: str) -> OidcStateData | None:
        """Atomically load and delete state data.

        Returns the data if found (and not expired), or ``None``.
        The default implementation calls :meth:`load` then :meth:`delete`,
        but concrete stores should override this for true atomicity.
        """
        data = await self.load(state)
        if data is not None:
            await self.delete(state)
        return data


class MemoryOidcStateStore(OidcStateStore):
    """In-memory state storage with TTL expiry and size limiting.

    Uses TTLCache to prevent unbounded memory growth (DoS mitigation).
    Suitable for single-instance deployments.
    """

    def __init__(self) -> None:
        self._store: TTLCache = TTLCache(maxsize=10000, ttl=600)

    async def save(self, state: str, data: OidcStateData, ttl_seconds: int = 600) -> None:
        # Stamp an app-level expiry so future DB/Redis backends (which may not
        # have native TTL) can reject stale state via OidcStateData.expires_at.
        data.expires_at = time.time() + ttl_seconds
        self._store[state] = data

    async def load(self, state: str) -> OidcStateData | None:
        data = self._store.get(state)
        if data is not None and data.expires_at and time.time() > data.expires_at:
            return None
        return data

    async def delete(self, state: str) -> None:
        self._store.pop(state, None)

    async def consume(self, state: str) -> OidcStateData | None:
        """Atomically load and delete state data using a single ``pop``.

        Also enforces the app-level ``expires_at`` as a secondary check so a
        future store backend that does not evict on TTL still rejects stale
        state.
        """
        data = self._store.pop(state, None)
        if data is not None and data.expires_at and time.time() > data.expires_at:
            return None
        return data


class OidcExchangeStore(ABC):
    """Abstract base for the short-lived exchange-code → refresh-token store.

    The OIDC callback hides the long-lived refresh_token behind a one-time,
    TTL-bounded exchange code; the frontend redeems it at ``/auth/oidc/exchange``.
    """

    @abstractmethod
    async def save(self, code: str, refresh_token: str, ttl_seconds: int = 60) -> None:
        """Store an exchange code → refresh_token mapping with a TTL."""

    @abstractmethod
    async def consume(self, code: str) -> str | None:
        """Atomically return and remove the refresh_token, or ``None`` if missing/expired."""


class MemoryOidcExchangeStore(OidcExchangeStore):
    """In-memory exchange-code store with TTL expiry and size limiting."""

    def __init__(self) -> None:
        self._store: TTLCache = TTLCache(maxsize=10000, ttl=60)

    async def save(self, code: str, refresh_token: str, ttl_seconds: int = 60) -> None:
        # Pair the token with an app-level expiry so consume() rejects stale
        # entries even if a future backend does not evict on TTL.
        self._store[code] = (refresh_token, time.time() + ttl_seconds)

    async def consume(self, code: str) -> str | None:
        entry = self._store.pop(code, None)
        if entry is None:
            return None
        refresh_token, expires_at = entry
        if expires_at and time.time() > expires_at:
            return None
        return refresh_token


def _backend() -> str:
    return os.environ.get("OIDC_STATE_STORE", "memory").lower()


def create_state_store() -> OidcStateStore:
    """Factory: create the state store for the configured backend.

    Only ``memory`` is implemented today; add Redis/DB by subclassing
    ``OidcStateStore`` and extending this factory.
    """
    if _backend() == "memory":
        return MemoryOidcStateStore()
    logger.warning("Unknown OIDC_STATE_STORE=%s, falling back to 'memory'.", _backend())
    return MemoryOidcStateStore()


def create_exchange_store() -> OidcExchangeStore:
    """Factory: create the exchange-code store for the configured backend.

    Shares the ``OIDC_STATE_STORE`` selector with :func:`create_state_store` so a
    future shared backend (Redis/DB) swaps both transient stores together.
    """
    if _backend() == "memory":
        return MemoryOidcExchangeStore()
    logger.warning("Unknown OIDC_STATE_STORE=%s, falling back to 'memory'.", _backend())
    return MemoryOidcExchangeStore()


# Module-level singletons
_state_store: OidcStateStore | None = None
_exchange_store: OidcExchangeStore | None = None


def get_state_store() -> OidcStateStore:
    """Return the (cached) state store singleton."""
    global _state_store
    if _state_store is None:
        _state_store = create_state_store()
    return _state_store


def get_exchange_store() -> OidcExchangeStore:
    """Return the (cached) exchange-code store singleton."""
    global _exchange_store
    if _exchange_store is None:
        _exchange_store = create_exchange_store()
    return _exchange_store
