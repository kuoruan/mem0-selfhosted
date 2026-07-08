"""OIDC authentication routes.

Provides endpoints for:
- Listing configured OIDC providers
- Initiating Authorization Code Flow + PKCE
- Handling the IdP callback and issuing local JWT tokens

All routes are unauthenticated (public).
"""

import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from urllib.parse import quote, urlencode, urlparse

from auth import FIRST_USER_ADVISORY_LOCK_ID, create_access_token, create_refresh_token
from auth_config import OIDCProviderConfig, get_auth_config
from db import get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from models import OidcLink, User
from oidc import (
    discover_provider,
    exchange_code_for_tokens,
    verify_id_token,
)
from oidc_state import OidcStateData, get_exchange_store, get_state_store
from pydantic import BaseModel
from rate_limit import is_trusted_proxy, limiter
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from utils.config import is_truthy
from utils.helpers import is_http_url, is_safe_redirect, sanitize_for_log
from utils.pkce import generate_code_challenge, generate_code_verifier, generate_nonce

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/oidc", tags=["auth"])

# DASHBOARD_URL may be a comma-separated list (for CORS origins in main.py);
# OIDC redirects require a single origin — use the first one.
DASHBOARD_URL = (os.environ.get("DASHBOARD_URL", "") or "http://localhost:3000").split(",")[0].strip().rstrip("/")

# Advisory lock ID for serializing first-user checks during OIDC registration
# lives in auth.py (shared with /auth/register). Must differ from
# bg_tasks._PRUNE_ADVISORY_LOCK_ID (0x6D656D30).

if not is_http_url(DASHBOARD_URL):
    raise ValueError(f"DASHBOARD_URL must start with http:// or https://, got: {DASHBOARD_URL!r}")

# SERVER_URL is optional — validate only when explicitly set.
_SERVER_URL = os.environ.get("SERVER_URL", "").rstrip("/")
if _SERVER_URL and not is_http_url(_SERVER_URL):
    raise ValueError(f"SERVER_URL must start with http:// or https://, got: {_SERVER_URL!r}")

# Hostname of the configured DASHBOARD_URL, used to gate X-Forwarded-Host trust
# so an attacker cannot redirect the OIDC callback to a foreign origin.
_DASHBOARD_HOST = (urlparse(DASHBOARD_URL).hostname or "").lower()


def _forwarded_host_matches_dashboard(candidate: str) -> bool:
    """Return True if *candidate* (an X-Forwarded-Host value) shares the
    configured DASHBOARD_URL hostname.

    Same-hostname comparison (port-agnostic) lets a reverse proxy legitimately
    forward ``app.example.com:443`` while rejecting a forged ``evil.com``.
    """
    try:
        candidate_host = (urlparse("http://" + candidate).hostname or "").lower()
    except ValueError:
        return False
    return bool(_DASHBOARD_HOST) and candidate_host == _DASHBOARD_HOST


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OIDCProviderInfo(BaseModel):
    name: str
    display_name: str


class OIDCProvidersResponse(BaseModel):
    providers: list[OIDCProviderInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_oidc_config():
    """Get OIDC config or raise 503 if not configured."""
    config = get_auth_config()
    if not config or not config.oidc or not config.oidc.providers:
        raise HTTPException(status_code=503, detail="OIDC is not configured.")
    return config.oidc


def _build_redirect_uri(request: Request, provider: str) -> str:
    """Build the redirect_uri for the OIDC callback.

    Uses SERVER_URL if set, otherwise infers from the request. Forwarded
    headers (X-Forwarded-Host / X-Forwarded-Proto) are honored only when the
    request came from a trusted proxy (FORWARDED_ALLOW_IPS — the same allowlist
    uvicorn/rate_limit use) AND the forwarded host matches the configured
    DASHBOARD_URL hostname. A forged cross-origin ``X-Forwarded-Host`` is
    rejected and falls back to the request's own netloc so the IdP redirect
    cannot be hijacked.
    """
    base = _SERVER_URL
    if not base:
        scheme = request.url.scheme
        host = request.url.netloc
        if is_trusted_proxy(request.client.host if request.client else ""):
            forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
            if forwarded_host:
                # Multiple proxies produce comma-separated values; use the first (client-facing)
                candidate = forwarded_host.split(",")[0].strip()
                if _forwarded_host_matches_dashboard(candidate):
                    host = candidate
                else:
                    logger.warning(
                        "Rejecting X-Forwarded-Host %s: does not match DASHBOARD_URL host %s",
                        sanitize_for_log(candidate),
                        _DASHBOARD_HOST,
                    )
                    # fall back to request.url.netloc (already set above)
            forwarded_proto = request.headers.get("x-forwarded-proto")
            if forwarded_proto:
                scheme = forwarded_proto.split(",")[0].strip()
        base = f"{scheme}://{host}"
    return f"{base}/auth/oidc/{provider}/callback"


def _make_placeholder_email(idp_sub: str, provider: str) -> str:
    """Generate a unique ``@oidc.{provider}`` placeholder email.

    Used when the IdP's real email can't be trusted onto the local account —
    either because it is unverified, or because another local account already
    owns it. Uses a SHA-256 digest of ``idp_sub`` (16 hex chars) to keep the
    local part short and avoid exposing the raw sub.
    """
    digest = hashlib.sha256(idp_sub.encode()).hexdigest()[:8]
    return f"{digest}-{secrets.token_hex(2)}@oidc.{provider}"


def _callback_error_redirect(error: str, description: str = "") -> RedirectResponse:
    """Build a 302 to the dashboard callback carrying an OIDC error in the fragment.

    Centralizes the ``#error=...&error_description=...`` redirect used at every
    failure point of the OIDC callback. ``description`` is URL-encoded and capped
    at 200 chars here, so call sites stay readable plain text.
    """
    fragment = f"error={quote(error, safe='')}"
    if description:
        fragment += f"&error_description={quote(description[:200], safe='')}"
    return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)


class OidcCallbackError(Exception):
    """Recoverable OIDC callback failure carrying a dashboard error code + description.

    Raised by callback helpers (e.g. :func:`_find_or_create_user`) so the caller can
    translate the failure into a single ``_callback_error_redirect`` rather than each
    helper having to know about HTTP responses.
    """

    def __init__(self, error: str, description: str = "") -> None:
        self.error = error
        self.description = description
        super().__init__(error)


def _find_or_create_user(
    db: Session,
    claims: dict,
    *,
    provider: str,
    idp_issuer: str,
    provider_config: OIDCProviderConfig,
    idp_sub: str,
) -> User:
    """Find the local user linked to this OIDC identity, or create/link one.

    Security invariants (see inline comments): two OIDC identities never merge by
    email — a recycled IdP email cannot take over an existing OIDC account. The
    user is added and flushed (so ``user.id`` is populated); the caller owns the
    commit and handles :class:`IntegrityError` from concurrent creation. Raises
    :class:`OidcCallbackError` for the unrecoverable "link exists but user gone" case.
    """
    oidc_link = db.scalar(select(OidcLink).where(OidcLink.idp_issuer == idp_issuer, OidcLink.idp_sub == idp_sub))

    if oidc_link:
        user = db.get(User, oidc_link.user_id)
        if not user:
            logger.error("OIDC link exists but user %s not found", oidc_link.user_id)
            raise OidcCallbackError("user_not_found", "Linked user account not found")

        # Sync email for pure-OIDC users (no local password). The IdP is the
        # sole authority for email; mem0 does not allow these users to edit
        # their email via PATCH /auth/me.
        if user.password_hash is None:
            claims_email = claims.get("email")
            if (
                claims_email
                and is_truthy(claims.get("email_verified"))
                and user.email != claims_email
            ):
                collision = db.scalar(
                    select(User).where(func.lower(User.email) == claims_email.lower(), User.id != user.id)
                )
                if not collision:
                    logger.info("Syncing email for OIDC user %s: %s → %s", user.id, user.email, claims_email)
                    user.email = claims_email
                else:
                    logger.warning(
                        "OIDC user %s email %s now in use by user %s, keeping current email %s",
                        user.id,
                        claims_email,
                        collision.id,
                        user.email,
                    )

        return user

    # New OIDC identity — link to an existing local account ONLY when that
    # account has a password_hash (a local credential being upgraded to
    # also accept OIDC). Two OIDC identities must never merge by email: a
    # recycled IdP email would otherwise let a fresh sub take over the
    # original owner's existing OIDC account. The ``auth_provider``
    # equality check is intentionally NOT used — it does not defend
    # against same-IdP email recycling.
    email_verified = is_truthy(claims.get("email_verified"))
    claims_email = claims.get("email")

    existing_user = None
    candidate = None
    if claims_email and email_verified:
        candidate = db.scalar(select(User).where(func.lower(User.email) == claims_email.lower()))
        if candidate is not None and candidate.password_hash is not None:
            existing_user = candidate

    if existing_user:
        # Link the existing local (password) account to this OIDC identity
        db.add(
            OidcLink(
                provider=provider,
                idp_issuer=idp_issuer,
                idp_sub=idp_sub,
                user_id=existing_user.id,
            )
        )
        logger.info("Linked existing user %s to OIDC provider %s", existing_user.id, provider)
        return existing_user

    # No existing account to link — create a new user. Use the IdP's verified
    # email only when no other account already owns it (otherwise the unique
    # constraint would fire and, worse, we'd shadow another OIDC account).
    # Otherwise fall back to a placeholder. Reuse the candidate lookup from the
    # auto-link check above (same email query in this transaction) instead of
    # issuing a duplicate.
    collision = candidate
    if claims_email and email_verified and not collision:
        email = claims_email
    else:
        if claims_email and not email_verified:
            logger.warning("Using placeholder email: email_verified is False for %s", claims_email)
        elif claims_email and email_verified and collision:
            logger.warning(
                "Using placeholder email: verified email %s already in use by user %s",
                claims_email,
                collision.id,
            )
        email = _make_placeholder_email(idp_sub, provider)

    configured = provider_config.username_claim
    if configured:
        _claim_list = configured if isinstance(configured, list) else [configured]
        name = next((claims.get(c) for c in _claim_list if claims.get(c)), None)
    else:
        name = claims.get("name") or claims.get("preferred_username")
    name = name or email.split("@")[0]

    # First user gets admin role; subsequent users get member.
    # Serialize concurrent first-user checks with pg_advisory_xact_lock.
    db.execute(text(f"SELECT pg_advisory_xact_lock({FIRST_USER_ADVISORY_LOCK_ID})"))
    is_first_user = db.scalar(select(func.count(User.id)).select_from(User)) == 0
    user = User(
        name=name,
        email=email,
        password_hash=None,
        auth_provider=provider,
        role="admin" if is_first_user else "member",
    )
    db.add(user)
    db.flush()  # get user.id

    db.add(
        OidcLink(
            provider=provider,
            idp_issuer=idp_issuer,
            idp_sub=idp_sub,
            user_id=user.id,
        )
    )
    logger.info("Created new user %s via OIDC provider %s", user.id, provider)
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/providers", response_model=OIDCProvidersResponse)
async def list_providers():
    """Return the list of configured OIDC identity providers."""
    config = get_auth_config()
    if not config or not config.oidc or not config.oidc.providers:
        return OIDCProvidersResponse(providers=[])

    providers = [OIDCProviderInfo(name=p.name, display_name=p.display_name or p.name) for p in config.oidc.providers]
    return OIDCProvidersResponse(providers=providers)


@router.get("/{provider}/login")
@limiter.limit("20/minute")
async def oidc_login(provider: str, request: Request, next: str | None = None):
    """Initiate OIDC Authorization Code Flow with PKCE.

    Redirects the user agent to the IdP's authorization endpoint.
    Accepts an optional `next` query parameter for post-login redirect.
    """
    # Validate next parameter to prevent open redirect
    if next and not is_safe_redirect(next):
        raise HTTPException(status_code=400, detail="Invalid redirect URL. Only relative paths are allowed.")

    oidc_config = _get_oidc_config()
    provider_config = oidc_config.get_provider(provider)
    if not provider_config:
        raise HTTPException(status_code=404, detail=f"Unknown OIDC provider: {provider}")

    # Discover provider endpoints (run in thread pool to avoid blocking event loop)
    try:
        metadata = await asyncio.to_thread(discover_provider, provider_config.issuer_url)
    except Exception as exc:
        logger.error("OIDC discovery failed for %s: %s", provider, exc)
        raise HTTPException(status_code=502, detail="Failed to contact identity provider")

    # Generate PKCE + state + nonce
    state = os.urandom(32).hex()
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    nonce = generate_nonce()

    redirect_uri = _build_redirect_uri(request, provider)

    # Save state to store
    store = get_state_store()
    await store.save(
        state,
        OidcStateData(
            code_verifier=code_verifier,
            provider=provider,
            redirect_uri=redirect_uri,
            next_url=next,
            nonce=nonce,
        ),
        ttl_seconds=600,
    )

    # Build authorization URL
    params = {
        "response_type": "code",
        "client_id": provider_config.client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(provider_config.scopes),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "nonce": nonce,
    }
    separator = "&" if "?" in metadata.authorization_endpoint else "?"
    authorize_url = f"{metadata.authorization_endpoint}{separator}{urlencode(params)}"

    return RedirectResponse(url=authorize_url, status_code=302)


@router.get("/{provider}/callback")
@limiter.limit("60/minute")
async def oidc_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_db),
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle the IdP callback after user authentication.

    On success: redirects to Dashboard with tokens in URL fragment.
    On failure: redirects to Dashboard with error in URL fragment.
    """
    # Handle IdP-reported errors
    if error:
        logger.warning(
            "OIDC callback error from %s: %s (%s)",
            sanitize_for_log(provider),
            error,
            error_description,
        )
        return _callback_error_redirect(error, str(error_description or ""))

    if not code or not state:
        return _callback_error_redirect("invalid_response", "Missing code or state parameter")

    # Retrieve and atomically consume state
    store = get_state_store()
    state_data = await store.consume(state)
    if not state_data:
        logger.warning(
            "OIDC callback with invalid or expired state for provider %s",
            sanitize_for_log(provider),
        )
        return _callback_error_redirect("invalid_state", "Session expired, please try again")

    # Validate that the provider matches
    if state_data.provider != provider:
        logger.warning("OIDC callback provider mismatch: expected %s, got %s", state_data.provider, provider)
        return _callback_error_redirect("provider_mismatch", "Provider mismatch, please try again")

    # Get provider config
    oidc_config = _get_oidc_config()
    provider_config = oidc_config.get_provider(provider)
    if not provider_config:
        return _callback_error_redirect("unknown_provider", "Provider not configured")

    # Discover provider endpoints (run in thread pool to avoid blocking event loop)
    try:
        metadata = await asyncio.to_thread(discover_provider, provider_config.issuer_url)
    except Exception as exc:
        logger.error("OIDC discovery failed during callback for %s: %s", provider, exc)
        return _callback_error_redirect("discovery_failed", "Failed to contact identity provider")

    # Exchange code for tokens (run in thread pool to avoid blocking event loop)
    redirect_uri = state_data.redirect_uri or _build_redirect_uri(request, provider)
    try:
        token_response = await asyncio.to_thread(
            exchange_code_for_tokens,
            token_endpoint=metadata.token_endpoint,
            code=code,
            redirect_uri=redirect_uri,
            client_id=provider_config.client_id,
            client_secret=provider_config.client_secret,
            code_verifier=state_data.code_verifier,
        )
    except Exception as exc:
        logger.error("OIDC token exchange failed for %s: %s", provider, exc)
        return _callback_error_redirect("token_exchange_failed", "Token exchange failed")

    id_token_str = token_response.get("id_token")
    if not id_token_str:
        return _callback_error_redirect("no_id_token", "Identity provider did not return an ID token")
    # access_token is required for at_hash validation (OIDC Core §3.1.3.6)
    # when the IdP includes that claim (Authelia, Keycloak, …). python-jose
    # rejects the ID token if access_token is missing in that case.
    access_token_str = token_response.get("access_token")

    # Verify ID token (run in thread pool to avoid blocking event loop)
    try:
        claims = await asyncio.to_thread(
            verify_id_token,
            id_token=id_token_str,
            client_id=provider_config.client_id,
            issuer=metadata.issuer,
            jwks_uri=metadata.jwks_uri,
            algorithms=metadata.id_token_signing_alg_values_supported,
            nonce=state_data.nonce,
            access_token=access_token_str,
        )
    except ValueError as exc:
        logger.error("OIDC ID token verification failed for %s: %s", provider, exc)
        return _callback_error_redirect("token_verification_failed", "ID token verification failed")

    idp_sub = claims.get("sub")
    if not idp_sub:
        return _callback_error_redirect("missing_sub", "ID token missing 'sub' claim")
    idp_sub = str(idp_sub)

    # Find or create local user. Raises OidcCallbackError on the rare
    # "link exists but user deleted" case; IntegrityError from concurrent
    # creation surfaces at the commit below.
    idp_issuer = metadata.issuer
    try:
        user = _find_or_create_user(
            db,
            claims,
            provider=provider,
            idp_issuer=idp_issuer,
            provider_config=provider_config,
            idp_sub=idp_sub,
        )
    except OidcCallbackError as exc:
        return _callback_error_redirect(exc.error, exc.description)

    # Update last login & commit — wrap in try/except for concurrent conflicts
    user.last_login_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        logger.error("OIDC user creation conflict for %s: %s", provider, exc)
        return _callback_error_redirect("user_creation_conflict", "Account creation conflict, please retry")

    # Issue local tokens
    access_token = create_access_token(str(user.id), user.role)
    refresh_token = create_refresh_token(str(user.id), db)

    # Store the refresh_token server-side behind a short-lived, single-use
    # exchange code so it never appears in the browser URL fragment (visible in
    # history, extensions, and screenshots). The code lives in OidcExchangeStore
    # (60s TTL, single use — see oidc_state.py).
    exchange_code = secrets.token_urlsafe(32)
    await get_exchange_store().save(exchange_code, refresh_token)

    # Build callback URL with optional next redirect
    callback_base = f"{DASHBOARD_URL}/auth/callback"
    if state_data.next_url:
        callback_base += f"?next={quote(state_data.next_url, safe='')}"
    callback_url = f"{callback_base}#access_token={access_token}&code={exchange_code}&token_type=bearer"
    return RedirectResponse(url=callback_url, status_code=302)


class ExchangeCodeRequest(BaseModel):
    exchange_code: str


class ExchangeCodeResponse(BaseModel):
    refresh_token: str


@router.post("/exchange", response_model=ExchangeCodeResponse)
@limiter.limit("20/minute")
async def exchange_code(request: Request, body: ExchangeCodeRequest):
    """Exchange a short-lived OIDC callback code for the real refresh_token.

    The callback delivers a one-time exchange code (60s TTL, single use) in the
    URL fragment instead of the long-lived refresh_token. The frontend sends it
    here to obtain the refresh_token, which is then stored as an httpOnly cookie
    by the Next.js API layer.
    """
    refresh_token = await get_exchange_store().consume(body.exchange_code)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Invalid or expired exchange code.")
    return ExchangeCodeResponse(refresh_token=refresh_token)
