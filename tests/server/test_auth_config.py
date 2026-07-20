"""Tests for server/auth_config.py — load_auth_config, get_auth_config, OIDC parsing."""

import json
from pathlib import Path

import pytest

# auth_config uses flat imports internally; conftest.py registers aliases.
# We import the module under a distinct name so we can reload after env patching.
import server.auth_config as _ac


def _reload_module() -> None:
    """Force-reload auth_config so module-level env vars are re-read."""
    import importlib

    importlib.reload(_ac)


# ============================================================================
# OIDCProviderConfig / OIDCConfig dataclasses
# ============================================================================


class TestOIDCProviderConfig:
    def test_display_name_auto(self) -> None:
        p = _ac.OIDCProviderConfig(
            name="google_oauth",
            issuer_url="https://accounts.google.com",
            client_id="id",
            client_secret="secret",
        )
        assert p.display_name == "Google Oauth"

    def test_display_name_explicit(self) -> None:
        p = _ac.OIDCProviderConfig(
            name="google",
            issuer_url="https://accounts.google.com",
            client_id="id",
            client_secret="secret",
            display_name="Google Sign-In",
        )
        assert p.display_name == "Google Sign-In"

    def test_default_scopes(self) -> None:
        p = _ac.OIDCProviderConfig(name="test", issuer_url="https://example.com", client_id="id", client_secret="s")
        assert p.scopes == ["openid", "email", "profile"]

    def test_custom_scopes(self) -> None:
        p = _ac.OIDCProviderConfig(
            name="test",
            issuer_url="https://example.com",
            client_id="id",
            client_secret="s",
            scopes=["openid", "custom"],
        )
        assert p.scopes == ["openid", "custom"]


class TestOIDCConfig:
    def test_get_provider_found(self) -> None:
        p = _ac.OIDCProviderConfig(name="google", issuer_url="https://example.com", client_id="id", client_secret="s")
        cfg = _ac.OIDCConfig(providers=[p])
        assert cfg.get_provider("google") is p

    def test_get_provider_not_found(self) -> None:
        cfg = _ac.OIDCConfig(providers=[])
        assert cfg.get_provider("missing") is None


# ============================================================================
# _parse_config
# ============================================================================


class TestParseConfig:
    def test_minimal(self) -> None:
        cfg = _ac._parse_config({})
        assert cfg.jwt_secret is None
        assert cfg.admin_api_key is None
        assert cfg.auth_disabled is None
        assert cfg.oidc is None

    def test_flat_fields(self) -> None:
        cfg = _ac._parse_config({"jwt_secret": "s", "admin_api_key": "k", "auth_disabled": True})
        assert cfg.jwt_secret == "s"
        assert cfg.admin_api_key == "k"
        assert cfg.auth_disabled is True

    def test_auth_disabled_string_coercion(self) -> None:
        cfg = _ac._parse_config({"auth_disabled": "true"})
        assert cfg.auth_disabled is True

    def test_auth_disabled_string_false(self) -> None:
        cfg = _ac._parse_config({"auth_disabled": "no"})
        assert cfg.auth_disabled is False

    def test_oidc_providers(self) -> None:
        cfg = _ac._parse_config(
            {
                "oidc": {
                    "providers": [
                        {
                            "name": "google",
                            "issuer_url": "https://accounts.google.com",
                            "client_id": "cid",
                            "client_secret": "csecret",
                            "display_name": "Google",
                        }
                    ]
                }
            }
        )
        assert cfg.oidc is not None
        assert len(cfg.oidc.providers) == 1
        assert cfg.oidc.providers[0].name == "google"
        assert cfg.oidc.providers[0].issuer_url == "https://accounts.google.com"


# ============================================================================
# load_auth_config — env-only (no JSON file)
# ============================================================================


class TestLoadAuthConfigEnvOnly:
    """Default config loaded purely from env vars."""

    def test_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JWT_SECRET", "my-secret")
        monkeypatch.setenv("ADMIN_API_KEY", "my-key")
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("AUTH_CONFIG_PATH", raising=False)
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.jwt_secret == "my-secret"
        assert cfg.admin_api_key == "my-key"
        assert cfg.auth_disabled is False

    def test_auth_disabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTH_DISABLED", "true")
        monkeypatch.delenv("AUTH_CONFIG_PATH", raising=False)
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.auth_disabled is True

    def test_no_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("AUTH_CONFIG_PATH", raising=False)
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.jwt_secret == ""
        assert cfg.admin_api_key == ""
        assert cfg.auth_disabled is False
        assert cfg.oidc is None


# ============================================================================
# load_auth_config — JSON file overrides
# ============================================================================


class TestLoadAuthConfigJsonOverride:
    """JSON file values override env-var defaults."""

    def test_json_overrides_jwt_secret(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "auth.json"
        cfg_file.write_text(json.dumps({"jwt_secret": "from-file"}))
        monkeypatch.setenv("JWT_SECRET", "from-env")
        monkeypatch.setenv("AUTH_CONFIG_PATH", str(cfg_file))
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.jwt_secret == "from-file"

    def test_json_overrides_admin_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "auth.json"
        cfg_file.write_text(json.dumps({"admin_api_key": "key-from-file"}))
        monkeypatch.setenv("ADMIN_API_KEY", "key-from-env")
        monkeypatch.setenv("AUTH_CONFIG_PATH", str(cfg_file))
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.admin_api_key == "key-from-file"

    def test_json_provides_oidc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_file = tmp_path / "auth.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "oidc": {
                        "providers": [
                            {
                                "name": "github",
                                "issuer_url": "https://token.actions.githubusercontent.com",
                                "client_id": "cid",
                                "client_secret": "csecret",
                            }
                        ]
                    }
                }
            )
        )
        monkeypatch.setenv("AUTH_CONFIG_PATH", str(cfg_file))
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        _reload_module()

        cfg = _ac.load_auth_config()
        assert cfg.oidc is not None
        assert len(cfg.oidc.providers) == 1
        assert cfg.oidc.providers[0].name == "github"

    def test_missing_json_file_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When AUTH_CONFIG_PATH is explicitly set to a missing file, it should raise."""
        monkeypatch.setenv("JWT_SECRET", "env-secret")
        monkeypatch.setenv("AUTH_CONFIG_PATH", "/nonexistent/path/auth.json")
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        _reload_module()

        with pytest.raises(RuntimeError, match="Config file not found"):
            _ac.load_auth_config()


# ============================================================================
# get_auth_config — caching
# ============================================================================


class TestGetAuthConfig:
    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("AUTH_CONFIG_PATH", raising=False)
        _reload_module()

        # Reset cache
        _ac.reload_auth_config()

        first = _ac.get_auth_config()
        second = _ac.get_auth_config()
        assert first is second

    def test_always_returns_auth_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("AUTH_CONFIG_PATH", raising=False)
        _reload_module()

        _ac.reload_auth_config()
        cfg = _ac.get_auth_config()
        assert isinstance(cfg, _ac.AuthConfig)
