"""OIDC authentication routes.

Provides endpoints for:
- Listing configured OIDC providers
- Initiating Authorization Code Flow + PKCE
- Handling the IdP callback and issuing local JWT tokens

All routes are unauthenticated (public).
"""

import asyncio
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from urllib.parse import quote, urlencode, urlparse

from auth import FIRST_USER_ADVISORY_LOCK_ID, create_access_token, create_refresh_token
from auth_config import get_auth_config
from db import get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from models import OidcLink, User
from oidc import (
    discover_provider,
    exchange_code_for_tokens,
    generate_code_challenge,
    generate_code_verifier,
    generate_nonce,
    verify_id_token,
)
from oidc_state import OidcStateData, get_exchange_store, get_state_store
from pydantic import BaseModel
from rate_limit import is_trusted_proxy, limiter
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from utils.config import is_truthy
from utils.helpers import is_http_url

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


def _sanitize_provider(name: str) -> str:
    """Strip log-injection characters from a provider path parameter.

    ``provider`` is an unvalidated URL path segment; before it reaches a log
    line it is reduced to ``[A-Za-z0-9_.-]`` so newlines/control chars cannot
    forge log entries.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


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


def _is_safe_redirect(url: str | None) -> bool:
    """Return True if *url* is a safe relative redirect target.

    Only relative paths (no scheme, no netloc) are allowed.
    """
    if not url:
        return False
    # Block whitespace/control characters which some browsers normalize, enabling open redirect
    if any(c.isspace() for c in url):
        return False
    # Block backslashes: browsers normalize \ to /, enabling open redirect
    if "\\" in url:
        return False
    parsed = urlparse(url)
    # Must be a relative path: no scheme, no netloc
    if parsed.scheme or parsed.netloc:
        return False
    # Must start with /
    if not url.startswith("/"):
        return False
    return True


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
                        _sanitize_provider(candidate),
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
    owns it. The random suffix ensures successive calls produce distinct
    addresses so the ``users.email`` unique index is never violated, even when
    a stale address is released after an IdP recycles it.
    """
    clean_sub = "".join(c if c.isalnum() or c in ".-_" else "_" for c in idp_sub[:64])
    return f"{clean_sub}-{secrets.token_hex(4)}@oidc.{provider}"


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
    if next and not _is_safe_redirect(next):
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
            _sanitize_provider(provider),
            error,
            error_description,
        )
        fragment = f"error={quote(error, safe='')}"
        if error_description:
            fragment += f"&error_description={quote(str(error_description)[:200], safe='')}"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    if not code or not state:
        fragment = "error=invalid_response&error_description=Missing%20code%20or%20state%20parameter"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    # Retrieve and atomically consume state
    store = get_state_store()
    state_data = await store.consume(state)
    if not state_data:
        logger.warning(
            "OIDC callback with invalid or expired state for provider %s",
            _sanitize_provider(provider),
        )
        fragment = "error=invalid_state&error_description=Session%20expired%2C%20please%20try%20again"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    # Validate that the provider matches
    if state_data.provider != provider:
        logger.warning("OIDC callback provider mismatch: expected %s, got %s", state_data.provider, provider)
        fragment = "error=provider_mismatch&error_description=Provider%20mismatch%2C%20please%20try%20again"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    # Get provider config
    oidc_config = _get_oidc_config()
    provider_config = oidc_config.get_provider(provider)
    if not provider_config:
        fragment = "error=unknown_provider&error_description=Provider%20not%20configured"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    # Discover provider endpoints (run in thread pool to avoid blocking event loop)
    try:
        metadata = await asyncio.to_thread(discover_provider, provider_config.issuer_url)
    except Exception as exc:
        logger.error("OIDC discovery failed during callback for %s: %s", provider, exc)
        fragment = "error=discovery_failed&error_description=Failed%20to%20contact%20identity%20provider"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

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
        fragment = "error=token_exchange_failed&error_description=Token%20exchange%20failed"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    id_token_str = token_response.get("id_token")
    if not id_token_str:
        fragment = "error=no_id_token&error_description=Identity%20provider%20did%20not%20return%20an%20ID%20token"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)
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
        fragment = "error=token_verification_failed&error_description=ID%20token%20verification%20failed"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

    idp_sub = claims.get("sub")
    if not idp_sub:
        fragment = "error=missing_sub&error_description=ID%20token%20missing%20'sub'%20claim"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)
    idp_sub = str(idp_sub)

    # Find or create local user
    idp_issuer = metadata.issuer
    oidc_link = db.scalar(select(OidcLink).where(OidcLink.idp_issuer == idp_issuer, OidcLink.idp_sub == idp_sub))

    if oidc_link:
        # Existing user — look up
        user = db.get(User, oidc_link.user_id)
        if not user:
            logger.error("OIDC link exists but user %s not found", oidc_link.user_id)
            fragment = "error=user_not_found&error_description=Linked%20user%20account%20not%20found"
            return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

        # Sync real email: when the IdP returns a verified email that differs,
        # update the local account so it stays aligned with the IdP and any
        # stale address is released. A recycled IdP email can otherwise be
        # abused via auto-link to shadow this account.
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
                logger.info("Syncing email for user %s: %s → %s", user.id, user.email, claims_email)
                user.email = claims_email
            else:
                # Verified email is owned by another local account; fall back
                # to a placeholder so the stale address is released and cannot
                # be exploited via auto-link if the IdP recycles it.
                placeholder = _make_placeholder_email(idp_sub, provider)
                logger.warning(
                    "Email %s for user %s already owned by user %s; reverting to placeholder",
                    claims_email,
                    user.id,
                    collision.id,
                )
                user.email = placeholder

    else:
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
            oidc_link = OidcLink(
                provider=provider,
                idp_issuer=idp_issuer,
                idp_sub=idp_sub,
                user_id=existing_user.id,
            )
            db.add(oidc_link)
            user = existing_user
            logger.info("Linked existing user %s to OIDC provider %s", user.id, provider)

        if not existing_user:
            # Pick the email for the new user. Use the IdP's verified email
            # only when no other account already owns it (otherwise the unique
            # constraint would fire and, worse, we'd shadow another OIDC
            # account). Otherwise fall back to a placeholder.
            # Reuse the candidate lookup from the auto-link check above (same
            # email query in this transaction) instead of issuing a duplicate.
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

            # First user gets admin role; subsequent users get member
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

            oidc_link = OidcLink(
                provider=provider,
                idp_issuer=idp_issuer,
                idp_sub=idp_sub,
                user_id=user.id,
            )
            db.add(oidc_link)
            logger.info("Created new user %s via OIDC provider %s", user.id, provider)

    # Update last login & commit — wrap in try/except for concurrent conflicts
    user.last_login_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        logger.error("OIDC user creation conflict for %s: %s", provider, exc)
        fragment = "error=user_creation_conflict&error_description=Account%20creation%20conflict%2C%20please%20retry"
        return RedirectResponse(url=f"{DASHBOARD_URL}/auth/callback#{fragment}", status_code=302)

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
