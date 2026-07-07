"""Unit tests for server/oidc_state.py.

The PKCE state store's consume / single-use / TTL behavior is already covered
in test_oidc_routes.py (TestStateStore). This file covers what had no direct
coverage:

- ``MemoryOidcExchangeStore`` — the post-callback exchange-code → refresh_token
  store (including the 60s TTL regression: the old ``_EXCHANGE_CACHE`` was a
  plain dict with no expiry, so abandoned flows leaked forever).
- ``create_state_store`` / ``create_exchange_store`` factories and the
  ``get_state_store`` / ``get_exchange_store`` singletons.
"""

import time

import pytest

pytest.importorskip("cachetools", reason="cachetools not installed")

from cachetools import TTLCache  # noqa: E402

from server.oidc_state import (  # noqa: E402
    MemoryOidcExchangeStore,
    MemoryOidcStateStore,
    create_exchange_store,
    create_state_store,
    get_exchange_store,
    get_state_store,
)


@pytest.fixture
def reset_store_singletons(monkeypatch):
    """Reset the module-level store singletons + OIDC_STATE_STORE around each test."""
    import server.oidc_state as oidc_state

    monkeypatch.delenv("OIDC_STATE_STORE", raising=False)
    oidc_state._state_store = None
    oidc_state._exchange_store = None
    yield
    oidc_state._state_store = None
    oidc_state._exchange_store = None


# ============================================================================
# MemoryOidcExchangeStore
# ============================================================================


class TestMemoryOidcExchangeStore:
    @pytest.mark.asyncio
    async def test_save_and_consume_returns_token(self):
        store = MemoryOidcExchangeStore()
        await store.save("code1", "refresh-token-abc")
        assert await store.consume("code1") == "refresh-token-abc"

    @pytest.mark.asyncio
    async def test_consume_is_single_use(self):
        store = MemoryOidcExchangeStore()
        await store.save("code1", "rt")
        assert await store.consume("code1") == "rt"
        # A second consume (replay) must fail.
        assert await store.consume("code1") is None

    @pytest.mark.asyncio
    async def test_consume_unknown_code_returns_none(self):
        store = MemoryOidcExchangeStore()
        assert await store.consume("never-saved") is None

    @pytest.mark.asyncio
    async def test_consume_expired_returns_none(self):
        """Regression: exchange codes must expire (~60s TTL), not live forever.

        The original ``_EXCHANGE_CACHE`` was a bare ``dict`` whose entries were
        only removed on consume — a user who abandoned the flow left the
        refresh_token mapping in memory indefinitely (unbounded growth / DoS).
        The store is now backed by a TTLCache plus an app-level expiry check.
        """
        store = MemoryOidcExchangeStore()
        store._store = TTLCache(maxsize=100, ttl=0.1)
        await store.save("short-lived", "rt")
        time.sleep(0.2)  # past TTL
        assert await store.consume("short-lived") is None

    @pytest.mark.asyncio
    async def test_app_level_expiry_rejects_stale_entry(self):
        """consume() also enforces the app-level expiry stamped at save() time,
        so a future backend without native TTL eviction still rejects stale codes."""
        store = MemoryOidcExchangeStore()
        await store.save("c", "rt", ttl_seconds=60)
        # Backdate the stamped expiry past the TTLCache window without waiting.
        token, _expires_at = store._store["c"]
        store._store["c"] = (token, time.time() - 1)
        assert await store.consume("c") is None


# ============================================================================
# Factories + singletons
# ============================================================================


class TestStoreFactories:
    def test_create_state_store_default_is_memory(self, reset_store_singletons):
        assert isinstance(create_state_store(), MemoryOidcStateStore)

    def test_create_exchange_store_default_is_memory(self, reset_store_singletons):
        assert isinstance(create_exchange_store(), MemoryOidcExchangeStore)

    def test_create_exchange_store_explicit_memory(self, reset_store_singletons, monkeypatch):
        monkeypatch.setenv("OIDC_STATE_STORE", "memory")
        assert isinstance(create_exchange_store(), MemoryOidcExchangeStore)

    def test_create_exchange_store_unknown_backend_falls_back(self, reset_store_singletons, monkeypatch, caplog):
        monkeypatch.setenv("OIDC_STATE_STORE", "redis-unimplemented")
        with caplog.at_level("WARNING"):
            store = create_exchange_store()
        assert isinstance(store, MemoryOidcExchangeStore)
        assert any("OIDC_STATE_STORE" in r.message and "redis-unimplemented" in r.message for r in caplog.records)

    def test_create_state_store_unknown_backend_falls_back(self, reset_store_singletons, monkeypatch, caplog):
        monkeypatch.setenv("OIDC_STATE_STORE", "redis-unimplemented")
        with caplog.at_level("WARNING"):
            store = create_state_store()
        assert isinstance(store, MemoryOidcStateStore)
        assert any("OIDC_STATE_STORE" in r.message for r in caplog.records)


class TestStoreSingletons:
    def test_get_exchange_store_returns_singleton(self, reset_store_singletons):
        import server.oidc_state as oidc_state

        first = get_exchange_store()
        second = oidc_state.get_exchange_store()
        assert first is second
        assert isinstance(first, MemoryOidcExchangeStore)

    def test_get_state_store_returns_singleton(self, reset_store_singletons):
        import server.oidc_state as oidc_state

        first = get_state_store()
        second = oidc_state.get_state_store()
        assert first is second
        assert isinstance(first, MemoryOidcStateStore)

    def test_get_exchange_store_distinct_from_state_store(self, reset_store_singletons):
        """The two transient stores are independent singletons."""
        assert get_exchange_store() is not get_state_store()
