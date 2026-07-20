"""Unified authentication configuration loader.

Loads auth settings from a JSON file (AUTH_CONFIG_PATH) with layered override:
  JSON value > environment variable > default.

JSON format:
{
  "jwt_secret": "optional-override",
  "admin_api_key": "optional-override",
  "auth_disabled": false,
  "oidc": {
    "providers": [
      {
        "name": "google",
        "issuer_url": "https://accounts.google.com",
        "client_id": "...",
        "client_secret": "...",
        "display_name": "Google",
        "scopes": ["openid", "email", "profile"],
        "username_claim": "preferred_username"
      }
    ]
  }
}

Optional per-provider keys:
  - "username_claim": claim name (str) or list of names used for the local
    display name. Falls back through the list to the first non-empty claim;
    when unset, defaults to "name" -> "preferred_username" -> email prefix.
"""

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from utils.config import is_truthy, load_json_config, merge_config

logger = logging.getLogger(__name__)

DEFAULT_OIDC_SCOPES = ["openid", "email", "profile"]


def _env_defaults() -> dict[str, Any]:
    """Read auth-related env vars live, at config-resolution time.

    Read here rather than as module-level constants so the config reflects the
    environment when it is resolved, not whatever it was when ``auth_config``
    was first imported. ``reload_auth_config()`` clears the cache to force a
    re-resolve.
    """
    return {
        "jwt_secret": os.environ.get("JWT_SECRET", ""),
        "admin_api_key": os.environ.get("ADMIN_API_KEY", ""),
        "auth_disabled": is_truthy(os.environ.get("AUTH_DISABLED", "")),
        "oidc": None,
    }


@dataclass
class OIDCProviderConfig:
    name: str
    issuer_url: str
    client_id: str
    client_secret: str
    display_name: str | None = None
    scopes: list[str] = field(default_factory=lambda: list(DEFAULT_OIDC_SCOPES))
    username_claim: str | list[str] | None = None

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").replace("-", " ").title()
        # Enforce https:// for the issuer so tokens/code are not exchanged over a
        # plaintext transport. OIDC_ALLOW_HTTP_ISSUER=true opts in (local development).
        if not self.issuer_url.startswith("https://") and not is_truthy(
            os.environ.get("OIDC_ALLOW_HTTP_ISSUER", "")
        ):
            raise RuntimeError(
                f"OIDC issuer_url must use https:// (got {self.issuer_url!r}). "
                "Set OIDC_ALLOW_HTTP_ISSUER=true only for local development."
            )

        if self.username_claim is not None and not (
            isinstance(self.username_claim, str)
            or (isinstance(self.username_claim, list) and all(isinstance(c, str) for c in self.username_claim))
        ):
            raise RuntimeError(
                f"OIDC provider {self.name}: username_claim must be a string or list of strings"
            )


@dataclass
class OIDCConfig:
    providers: list[OIDCProviderConfig] = field(default_factory=list)

    def get_provider(self, name: str) -> OIDCProviderConfig | None:
        for p in self.providers:
            if p.name == name:
                return p
        return None


@dataclass
class AuthConfig:
    jwt_secret: str | None = None
    admin_api_key: str | None = None
    auth_disabled: bool | None = None
    oidc: OIDCConfig | None = None


def _parse_provider(data: dict[str, Any]) -> OIDCProviderConfig:
    try:
        return OIDCProviderConfig(
            name=data["name"],
            issuer_url=data["issuer_url"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            display_name=data.get("display_name"),
            scopes=data.get("scopes", list(DEFAULT_OIDC_SCOPES)),
            username_claim=data.get("username_claim"),
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required OIDC provider configuration key: {exc}") from exc


def _parse_config(data: dict[str, Any]) -> AuthConfig:
    oidc = None
    if "oidc" in data and isinstance(data["oidc"], dict):
        providers_data = data["oidc"].get("providers", [])
        if isinstance(providers_data, list):
            providers = [_parse_provider(p) for p in providers_data if isinstance(p, dict)]
        else:
            providers = []
        oidc = OIDCConfig(providers=providers)

    auth_disabled = data.get("auth_disabled")
    if isinstance(auth_disabled, str):
        auth_disabled = is_truthy(auth_disabled)

    return AuthConfig(
        jwt_secret=data.get("jwt_secret"),
        admin_api_key=data.get("admin_api_key"),
        auth_disabled=auth_disabled,
        oidc=oidc,
    )


def load_auth_config() -> AuthConfig:
    """Load auth configuration.

    If ``AUTH_CONFIG_PATH`` is set and the file exists, its values are
    merged on top of the defaults (which already include env-var values).
    Otherwise the defaults are used as-is.
    """
    config_path = os.environ.get("AUTH_CONFIG_PATH", "").strip()
    data: dict[str, Any] | None = None

    if config_path:
        # Sensitive fields may legitimately contain '$' (e.g. a client_secret
        # like "pa$$word"); skip env-var expansion for them.
        data = load_json_config(
            config_path,
            silent=False,
            raw_keys={"client_secret", "jwt_secret", "admin_api_key"},
        )
        if data is not None:
            logger.info("Auth config loaded from %s", config_path)

    merged = merge_config(_env_defaults(), data or {})
    config = _parse_config(merged)
    provider_names = [p.name for p in (config.oidc.providers if config.oidc else [])]
    logger.info(
        "Auth config resolved (jwt_secret=%s, admin_api_key=%s, oidc_providers=%s)",
        "set" if config.jwt_secret else "unset",
        "set" if config.admin_api_key else "unset",
        provider_names,
    )
    return config


@functools.lru_cache(maxsize=1)
def get_auth_config() -> AuthConfig:
    """Return the cached auth config, loading it on first access.

    Cached for the process lifetime; call ``reload_auth_config()`` to force a
    re-resolve after changing ``JWT_SECRET`` / ``AUTH_CONFIG_PATH`` / the OIDC
    provider config.
    """
    return load_auth_config()


def reload_auth_config() -> None:
    """Clear the cached env-based config so the next ``get_auth_config()`` call
    re-resolves from env / ``AUTH_CONFIG_PATH``. Use after changing ``JWT_SECRET``
    / ``AUTH_CONFIG_PATH`` / the OIDC provider config."""
    get_auth_config.cache_clear()
