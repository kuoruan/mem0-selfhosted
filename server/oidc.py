"""OIDC Discovery client and ID Token verification.

Handles:
- Provider metadata discovery via /.well-known/openid-configuration
- JWKS public key fetching and caching
- ID Token signature and claims verification
- PKCE code_verifier / code_challenge generation
"""

import base64
import hashlib
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx
from cachetools import TTLCache
from jose import JWTError, jwt

from utils.config import is_truthy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def generate_code_verifier() -> str:
    """Generate a code_verifier per RFC 7636 §4.1 (43–128 chars, unreserved)."""
    return secrets.token_urlsafe(64)


def generate_nonce() -> str:
    """Generate a random nonce for OIDC replay-protection."""
    return secrets.token_urlsafe(32)


def generate_code_challenge(verifier: str) -> str:
    """S256 code_challenge per RFC 7636 §4.2."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# Provider metadata discovery
# ---------------------------------------------------------------------------

_DISCOVERY_CACHE: TTLCache = TTLCache(maxsize=16, ttl=24 * 60 * 60)  # 24 hours
_DISCOVERY_LOCK = threading.Lock()


@dataclass
class ProviderMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    end_session_endpoint: str | None = None
    id_token_signing_alg_values_supported: list[str] = field(default_factory=lambda: ["RS256"])


def discover_provider(issuer_url: str, *, force: bool = False) -> ProviderMetadata:
    """Fetch and parse the OIDC discovery document.

    Results are cached for 24 hours unless *force* is True.
    Thread-safe: uses a lock to protect the TTLCache.
    """
    if not force:
        with _DISCOVERY_LOCK:
            cached = _DISCOVERY_CACHE.get(issuer_url)
        if cached is not None:
            return cached

    discovery_url = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(discovery_url)
    resp.raise_for_status()
    doc = resp.json()

    metadata = ProviderMetadata(
        issuer=doc["issuer"],
        authorization_endpoint=doc["authorization_endpoint"],
        token_endpoint=doc["token_endpoint"],
        jwks_uri=doc["jwks_uri"],
        end_session_endpoint=doc.get("end_session_endpoint"),
        id_token_signing_alg_values_supported=doc.get("id_token_signing_alg_values_supported", ["RS256"]),
    )

    # Validate that the issuer matches what we requested (normalize trailing slashes)
    if metadata.issuer.rstrip("/") != issuer_url.rstrip("/"):
        raise ValueError(f"Discovered issuer {metadata.issuer!r} does not match configured issuer_url {issuer_url!r}.")

    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE[issuer_url] = metadata
    return metadata


# ---------------------------------------------------------------------------
# JWKS handling
# ---------------------------------------------------------------------------

_JWKS_CACHE: TTLCache = TTLCache(maxsize=16, ttl=24 * 60 * 60)  # 24 hours
_JWKS_LOCK = threading.Lock()


def _fetch_jwks(jwks_uri: str, *, force: bool = False) -> dict[str, Any]:
    """Fetch and cache the JWKS document.

    Thread-safe: uses a lock to protect the TTLCache.
    """
    if not force:
        with _JWKS_LOCK:
            cached = _JWKS_CACHE.get(jwks_uri)
        if cached is not None:
            return cached

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(jwks_uri)
    resp.raise_for_status()
    data = resp.json()
    with _JWKS_LOCK:
        _JWKS_CACHE[jwks_uri] = data
    return data


# ---------------------------------------------------------------------------
# ID Token verification
# ---------------------------------------------------------------------------


def verify_id_token(
    id_token: str,
    client_id: str,
    issuer: str,
    jwks_uri: str,
    algorithms: list[str] | None = None,
    *,
    nonce: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Verify an OIDC ID Token and return its claims.

    Validates:
    - Signature against the IdP's JWKS
    - iss claim matches the expected issuer
    - aud claim contains our client_id
    - exp claim is in the future
    - nonce matches the one sent in the authorization request (if provided)
    - at_hash matches the access_token (OIDC Core §3.1.3.6) when the IdP
      includes it. python-jose mandates access_token whenever at_hash is
      present, so callers MUST forward the access_token from the token
      response — otherwise tokens carrying at_hash (Authelia, Keycloak,
      Auth0, …) are rejected with "No access_token provided to compare
      against at_hash claim".
    """
    if algorithms is None:
        algorithms = ["RS256"]
    else:
        # CWE-347: reject 'none' (signature bypass) and symmetric algorithms
        # (HS256 etc. — algorithm confusion attack using public key as shared secret)
        algorithms = [alg for alg in algorithms if alg.lower() != "none" and not alg.upper().startswith("HS")]
    if not algorithms:
        # The IdP advertised only disallowed algs (e.g. only 'none' / HS*). Refuse
        # outright with a clear message rather than passing an empty allowlist to
        # jwt.decode (which would reject the token with a less diagnosable error).
        raise ValueError(
            "ID token lists no acceptable signing algorithm; only asymmetric "
            "algorithms (e.g. RS256/ES256) are permitted."
        )

    # Intersect with the operator-configured allowlist so a compromised or
    # misconfigured IdP cannot force a less-preferred algorithm. Default RS256.
    allowed = os.environ.get("OIDC_ALLOWED_SIGNING_ALGS", "RS256")
    allowlist = {a.strip() for a in allowed.split(",") if a.strip()}
    if allowlist:
        algorithms = [a for a in algorithms if a in allowlist]
    if not algorithms:
        raise ValueError(f"ID token signing algorithm not in OIDC_ALLOWED_SIGNING_ALGS={sorted(allowlist)}.")

    # Decode header to get kid
    try:
        unverified_header = jwt.get_unverified_header(id_token)
    except JWTError as exc:
        raise ValueError(f"Invalid ID token header: {exc}") from exc
    kid = unverified_header.get("kid")
    if not kid:
        raise ValueError("ID token header missing 'kid' claim")

    # Fetch JWKS and find the matching key.
    # If the key is not found, force-refresh the JWKS cache (the IdP may have
    # rotated signing keys) and retry once.
    try:
        jwks_data = _fetch_jwks(jwks_uri)
    except Exception as exc:
        raise ValueError(f"Failed to fetch JWKS: {exc}") from exc

    def _find_key(data: dict[str, Any]) -> dict[str, Any]:
        for k in data.get("keys", []):
            if k.get("kid") == kid:
                return k
        return {}

    rsa_key = _find_key(jwks_data)
    if not rsa_key:
        logger.info("kid=%s not found in cached JWKS, force-refreshing", kid)
        try:
            jwks_data = _fetch_jwks(jwks_uri, force=True)
        except Exception as exc:
            raise ValueError(f"Failed to fetch JWKS (force refresh): {exc}") from exc
        rsa_key = _find_key(jwks_data)

    if not rsa_key:
        raise ValueError(f"No matching key found in JWKS for kid={kid}")

    try:
        claims = jwt.decode(
            id_token,
            rsa_key,
            algorithms=algorithms,
            audience=client_id,
            issuer=issuer,
            access_token=access_token,
            options={"leeway": 120},  # tolerate up to 120s clock skew
        )
    except JWTError as exc:
        raise ValueError(f"ID token verification failed: {exc}") from exc

    # Validate nonce if provided
    if nonce is not None:
        token_nonce = claims.get("nonce")
        if token_nonce != nonce:
            raise ValueError(f"ID token nonce mismatch: expected {nonce!r}, got {token_nonce!r}")

    # Validate azp (authorized party) per OIDC Core §3.1.3.7.3:
    # - When aud is a list (multi-audience token), azp MUST be present and
    #   MUST equal our client_id (otherwise a token minted for another client
    #   could be replayed against us).
    # - When aud is a single string, azp is optional but if present MUST equal
    #   client_id.
    aud = claims.get("aud")
    token_azp = claims.get("azp")
    if isinstance(aud, list):
        if not token_azp:
            raise ValueError("ID token with multiple audiences is missing required 'azp' claim")
        if token_azp != client_id:
            raise ValueError(f"ID token azp mismatch: expected {client_id!r}, got {token_azp!r}")
    elif token_azp is not None and token_azp != client_id:
        raise ValueError(f"ID token azp mismatch: expected {client_id!r}, got {token_azp!r}")

    return claims


# ---------------------------------------------------------------------------
# Token exchange (authorization code → tokens)
# ---------------------------------------------------------------------------


def exchange_code_for_tokens(
    *,
    token_endpoint: str,
    code: str,
    redirect_uri: str,
    client_id: str,
    client_secret: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Exchange an authorization code for tokens using PKCE."""
    # Refuse to send the client_secret over plaintext http unless explicitly
    # exempted (OIDC_ALLOW_HTTP_ISSUER=true, intended for local test IdPs only).
    if token_endpoint.startswith("http://") and not is_truthy(os.environ.get("OIDC_ALLOW_HTTP_ISSUER", "")):
        raise ValueError(
            f"Refusing token exchange over http:// token_endpoint ({token_endpoint!r}). "
            "Set OIDC_ALLOW_HTTP_ISSUER=true only for local testing."
        )
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
                "code_verifier": code_verifier,
            },
        )
    resp.raise_for_status()
    return resp.json()
