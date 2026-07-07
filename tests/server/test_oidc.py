"""Tests for core OIDC helpers in server/oidc.py.

Covers:
- PKCE helpers: code_verifier, code_challenge, nonce
- Provider metadata discovery (caching, issuer validation)
- JWKS fetching and force-refresh
- ID Token verification (signature, claims, JWK rotation fallback)
- Token exchange (authorization code → tokens)
"""

import base64
from unittest.mock import MagicMock, patch

import importlib

import pytest

pytest.importorskip("cachetools", reason="cachetools not installed")
pytest.importorskip("jose", reason="python-jose not installed")
pytest.importorskip("httpx", reason="httpx not installed")

# Load the real oidc module by its full path (not relying on conftest aliases).
oidc = importlib.import_module("server.oidc")
pkce = importlib.import_module("server.utils.pkce")


# ============================================================================
# PKCE helpers
# ============================================================================


class TestPkce:
    """Tests for PKCE (RFC 7636) helpers."""

    def test_code_verifier_length(self):
        v = pkce.generate_code_verifier()
        # token_urlsafe(64) → ceil(64*6/8) = 48 bytes → 64 base64url chars
        assert 43 <= len(v) <= 128

    def test_code_verifier_is_urlsafe(self):
        v = pkce.generate_code_verifier()
        # Only unreserved characters: A-Z a-z 0-9 - _
        assert all(c.isalnum() or c in "-_" for c in v)

    def test_code_verifier_is_random(self):
        a = pkce.generate_code_verifier()
        b = pkce.generate_code_verifier()
        assert a != b

    def test_nonce_is_random(self):
        a = pkce.generate_nonce()
        b = pkce.generate_nonce()
        assert a != b
        assert len(a) > 0

    def test_code_challenge_no_padding(self):
        verifier = pkce.generate_code_verifier()
        challenge = pkce.generate_code_challenge(verifier)
        # S256 code_challenge must NOT end with '=' (RFC 7636 §4.2)
        assert not challenge.endswith("=")

    def test_code_challenge_is_base64url(self):
        verifier = "test-verifier-for-pkce-testing-12345"
        challenge = pkce.generate_code_challenge(verifier)
        # Should be valid base64url (decode then encode gives same string)
        decoded = base64.urlsafe_b64decode(challenge + "===")
        assert len(decoded) == 32  # SHA-256 digest is exactly 32 bytes

    def test_code_challenge_deterministic(self):
        verifier = "fixed-verifier-value"
        a = pkce.generate_code_challenge(verifier)
        b = pkce.generate_code_challenge(verifier)
        assert a == b

    def test_code_challenge_different_for_different_verifiers(self):
        a = pkce.generate_code_challenge("verifier-a")
        b = pkce.generate_code_challenge("verifier-b")
        assert a != b


# ============================================================================
# Provider metadata discovery
# ============================================================================


class TestDiscovery:
    """Tests for discover_provider()."""

    def test_happy_path(self):
        """Successful discovery returns ProviderMetadata."""
        issuer = "https://accounts.example.com"
        discovery_doc = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
            "end_session_endpoint": f"{issuer}/logout",
            "id_token_signing_alg_values_supported": ["RS256", "ES256"],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = discovery_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            metadata = oidc.discover_provider(issuer, force=True)

        expected_url = issuer + "/.well-known/openid-configuration"
        mock_get.assert_called_once_with(expected_url)
        assert metadata.issuer == issuer
        assert metadata.authorization_endpoint == discovery_doc["authorization_endpoint"]
        assert metadata.token_endpoint == discovery_doc["token_endpoint"]
        assert metadata.jwks_uri == discovery_doc["jwks_uri"]
        assert metadata.end_session_endpoint == discovery_doc["end_session_endpoint"]
        assert metadata.id_token_signing_alg_values_supported == ["RS256", "ES256"]

    def test_issuer_mismatch_raises(self):
        """When discovered issuer != requested, raise ValueError."""
        issuer = "https://accounts.example.com"
        discovery_doc = {
            "issuer": "https://other.example.com",  # mismatch
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = discovery_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp):
            with pytest.raises(ValueError, match="does not match"):
                oidc.discover_provider(issuer, force=True)

    def test_default_signing_algs(self):
        """When id_token_signing_alg_values_supported is missing, default to RS256."""
        issuer = "https://accounts.example.com"
        discovery_doc = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
            # no id_token_signing_alg_values_supported
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = discovery_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp):
            metadata = oidc.discover_provider(issuer, force=True)

        assert metadata.id_token_signing_alg_values_supported == ["RS256"]

    def test_cache_hit_avoids_http_call(self):
        """Second call with same issuer returns cached result without HTTP."""
        issuer = "https://accounts.cached.example.com"
        discovery_doc = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = discovery_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            m1 = oidc.discover_provider(issuer, force=True)
            m2 = oidc.discover_provider(issuer)  # should hit cache

        # Only one HTTP call made (the initial force=True)
        assert mock_get.call_count == 1
        assert m1.issuer == m2.issuer
        assert m1.jwks_uri == m2.jwks_uri

    def test_force_bypasses_cache(self):
        """force=True re-fetches even if cached."""
        issuer = "https://accounts.force.example.com"
        discovery_doc = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "jwks_uri": f"{issuer}/jwks",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = discovery_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            oidc.discover_provider(issuer, force=True)
            oidc.discover_provider(issuer, force=True)

        assert mock_get.call_count == 2


# ============================================================================
# JWKS handling
# ============================================================================


class TestJwks:
    """Tests for _fetch_jwks()."""

    def test_happy_path(self):
        uri = "https://accounts.example.com/jwks"
        jwks_doc = {"keys": [{"kid": "k1", "kty": "RSA"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            data = oidc._fetch_jwks(uri, force=True)

        mock_get.assert_called_once_with(uri)
        assert data == jwks_doc

    def test_cache_hit(self):
        uri = "https://accounts.example.com/cached-jwks"
        jwks_doc = {"keys": [{"kid": "k1", "kty": "RSA"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            oidc._fetch_jwks(uri, force=True)
            oidc._fetch_jwks(uri)  # cache hit

        assert mock_get.call_count == 1

    def test_force_bypasses_cache(self):
        uri = "https://accounts.example.com/force-jwks"
        jwks_doc = {"keys": [{"kid": "k1", "kty": "RSA"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks_doc
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "get", return_value=mock_resp) as mock_get:
            oidc._fetch_jwks(uri, force=True)
            oidc._fetch_jwks(uri, force=True)

        assert mock_get.call_count == 2


# ============================================================================
# ID Token verification
# ============================================================================


def _generate_rsa_key_and_jwks(kid: str = "test-kid"):
    """Generate a real RSA key pair and corresponding JWKS representation.

    Returns (private_pem, public_jwks_dict).
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Serialize private key to PEM
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")

    # Build JWKS from public key
    from jose import jwk

    public_jwk = jwk.construct(private_key.public_key(), algorithm="RS256").to_dict()
    public_jwk["kid"] = kid
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"

    return private_pem, {"keys": [public_jwk]}


def _sign_id_token(private_pem: str, claims: dict, kid: str = "test-kid") -> str:
    """Sign an ID token (JWT) with the given RSA private key."""
    from jose import jwt as jose_jwt

    headers = {"kid": kid}
    return jose_jwt.encode(claims, private_pem, algorithm="RS256", headers=headers)


def _compute_at_hash(access_token: str, alg: str = "RS256") -> str:
    """Compute the OIDC ``at_hash`` claim value for an access_token.

    Mirrors python-jose's ``_validate_at_hash``: hash the access_token with the
    algorithm's hash function (RS256 → SHA-256), take the leftmost half of the
    digest, and base64url-encode without padding.
    """
    import hashlib

    hashalg = {"RS256": "sha256", "RS384": "sha384", "RS512": "sha512"}[alg]
    digest = hashlib.new(hashalg, access_token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest[: len(digest) // 2]).rstrip(b"=").decode("ascii")


class TestVerifyIdToken:
    """Tests for verify_id_token()."""

    def test_happy_path(self):
        """Valid ID token is verified and claims returned."""
        private_pem, jwks = _generate_rsa_key_and_jwks("valid-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "user-123",
            "aud": "my-client-id",
            "exp": 9999999999,
            "iat": 1000000000,
            "email": "user@example.com",
        }
        id_token = _sign_id_token(private_pem, claims, kid="valid-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            result = oidc.verify_id_token(
                id_token=id_token,
                client_id="my-client-id",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
                algorithms=["RS256"],
            )

        assert result["sub"] == "user-123"
        assert result["email"] == "user@example.com"

    def test_at_hash_validated_with_access_token(self):
        """ID token carrying at_hash verifies when the matching access_token is supplied.

        Regression: python-jose mandates access_token whenever at_hash is present
        (Authelia/Keycloak/Auth0 include it). Before access_token was wired
        through, such logins failed with "No access_token provided to compare
        against at_hash claim".
        """
        private_pem, jwks = _generate_rsa_key_and_jwks("at-hash-kid")
        access_token = "real-access-token-value"
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "cid",
            "exp": 9999999999,
            "at_hash": _compute_at_hash(access_token),
        }
        id_token = _sign_id_token(private_pem, claims, kid="at-hash-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            result = oidc.verify_id_token(
                id_token=id_token,
                client_id="cid",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
                algorithms=["RS256"],
                access_token=access_token,
            )

        assert result["sub"] == "u1"
        assert result["at_hash"] == _compute_at_hash(access_token)

    def test_at_hash_without_access_token_raises(self):
        """ID token with at_hash is rejected when access_token is not forwarded.

        Guards against silently bypassing at_hash validation (e.g. via
        options={'verify_at_hash': False}); the binding between ID token and
        access_token (OIDC Core §3.1.3.6) must stay enforced.
        """
        private_pem, jwks = _generate_rsa_key_and_jwks("at-hash-noat-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "cid",
            "exp": 9999999999,
            "at_hash": _compute_at_hash("real-access-token-value"),
        }
        id_token = _sign_id_token(private_pem, claims, kid="at-hash-noat-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="No access_token provided"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                    algorithms=["RS256"],
                )

    def test_missing_kid_raises(self):
        """ID token without 'kid' in header raises ValueError."""
        private_pem, jwks = _generate_rsa_key_and_jwks()
        claims = {"iss": "https://accounts.example.com", "sub": "u1", "aud": "cid", "exp": 9999999999}
        id_token = _sign_id_token(private_pem, claims, kid="test-kid")

        # Strip the header kid by patching get_unverified_header
        with patch.object(oidc.jwt, "get_unverified_header", return_value={}):
            with pytest.raises(ValueError, match="missing 'kid'"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_malformed_token_header_raises_valueerror(self):
        """Malformed ID token (invalid JWT header) raises ValueError, not JWTError."""
        with pytest.raises(ValueError, match="Invalid ID token header"):
            oidc.verify_id_token(
                id_token="not-a-valid-jwt",
                client_id="cid",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
            )

    def test_none_algorithm_filtered_out(self):
        """'none' algorithm must be stripped from allowed algorithms (CWE-347)."""
        private_pem, jwks = _generate_rsa_key_and_jwks("none-test-kid")
        claims = {"iss": "https://accounts.example.com", "sub": "u1", "aud": "cid", "exp": 9999999999}
        id_token = _sign_id_token(private_pem, claims, kid="none-test-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks), \
             patch.object(oidc.jwt, "decode", wraps=oidc.jwt.decode) as mock_decode:
            oidc.verify_id_token(
                id_token=id_token,
                client_id="cid",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
                algorithms=["RS256", "none", "None"],
            )

        # Verify that "none" was filtered out before being passed to jwt.decode
        call_algorithms = mock_decode.call_args[1].get("algorithms") or mock_decode.call_args[0][3] if len(mock_decode.call_args[0]) > 3 else mock_decode.call_args[1]["algorithms"]
        assert "none" not in call_algorithms
        assert "None" not in call_algorithms
        assert "RS256" in call_algorithms

    def test_only_none_algorithm_results_in_empty_list(self):
        """If only 'none' variants are provided, algorithms list becomes empty → error."""
        private_pem, jwks = _generate_rsa_key_and_jwks("only-none-kid")
        claims = {"iss": "https://accounts.example.com", "sub": "u1", "aud": "cid", "exp": 9999999999}
        id_token = _sign_id_token(private_pem, claims, kid="only-none-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            # After filtering all 'none' variants, algorithms list is empty
            with pytest.raises((ValueError, Exception)):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                    algorithms=["none", "None", "NONE"],
                )

    def test_kid_not_found_after_refresh_raises(self):
        """When kid is missing from both cached and refreshed JWKS, raise ValueError."""
        private_pem, jwks = _generate_rsa_key_and_jwks("real-kid")
        claims = {"iss": "https://accounts.example.com", "sub": "u1", "aud": "cid", "exp": 9999999999}
        id_token = _sign_id_token(private_pem, claims, kid="real-kid")

        # JWKS doesn't contain the kid — won't match on either attempt
        wrong_jwks = {"keys": [{"kid": "other-kid", "kty": "RSA", "n": "..", "e": ".."}]}

        with patch.object(oidc, "_fetch_jwks", return_value=wrong_jwks):
            with pytest.raises(ValueError, match="No matching key found"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_jwk_rotation_fallback_success(self):
        """When kid is not in cached JWKS but found after force-refresh, verification succeeds."""
        private_pem, correct_jwks = _generate_rsa_key_and_jwks("rotated-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "user-456",
            "aud": "cid",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="rotated-kid")

        # Stale JWKS without the key, fresh JWKS with the key
        stale_jwks = {"keys": [{"kid": "old-kid", "kty": "RSA", "n": "..", "e": ".."}]}

        call_count = 0

        def _fetch_jwks_side_effect(uri, *, force=False):
            nonlocal call_count
            call_count += 1
            if force:
                return correct_jwks
            return stale_jwks

        with patch.object(oidc, "_fetch_jwks", side_effect=_fetch_jwks_side_effect):
            result = oidc.verify_id_token(
                id_token=id_token,
                client_id="cid",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
                algorithms=["RS256"],
            )

        assert result["sub"] == "user-456"
        assert call_count == 2  # first from cache (miss), then force-refresh (hit)

    def test_nonce_mismatch_raises(self):
        """When nonce doesn't match, ValueError is raised."""
        private_pem, jwks = _generate_rsa_key_and_jwks("nonce-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "cid",
            "exp": 9999999999,
            "nonce": "idp-nonce-123",
        }
        id_token = _sign_id_token(private_pem, claims, kid="nonce-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="nonce mismatch"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                    nonce="wrong-nonce",
                )

    def test_nonce_match_passes(self):
        """When nonce matches, no error raised."""
        private_pem, jwks = _generate_rsa_key_and_jwks("match-nonce-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "cid",
            "exp": 9999999999,
            "nonce": "correct-nonce",
        }
        id_token = _sign_id_token(private_pem, claims, kid="match-nonce-kid")

        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            result = oidc.verify_id_token(
                id_token=id_token,
                client_id="cid",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
                nonce="correct-nonce",
            )

        assert result["sub"] == "u1"

    def test_jwks_fetch_error_raises_valueerror(self):
        """Network/JSON errors from _fetch_jwks must be wrapped as ValueError."""
        private_pem, jwks = _generate_rsa_key_and_jwks("net-error-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "cid",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="net-error-kid")

        # _fetch_jwks raises a network error → verify_id_token should convert to ValueError
        with patch.object(oidc, "_fetch_jwks", side_effect=ConnectionError("network unreachable")):
            with pytest.raises(ValueError, match="Failed to fetch JWKS"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="cid",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )


# ============================================================================
# Token exchange
# ============================================================================


class TestExchangeCodeForTokens:
    """Tests for exchange_code_for_tokens()."""

    def test_happy_path(self):
        token_response = {
            "access_token": "at-123",
            "id_token": "id-456",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = token_response
        mock_resp.raise_for_status.return_value = None

        with patch.object(oidc.httpx.Client, "post", return_value=mock_resp) as mock_post:
            result = oidc.exchange_code_for_tokens(
                token_endpoint="https://idp.example.com/token",
                code="auth-code-xyz",
                redirect_uri="https://myapp.example.com/callback",
                client_id="my-client",
                client_secret="my-secret",
                code_verifier="verifier-123",
            )

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "authorization_code"
        assert kwargs["data"]["code"] == "auth-code-xyz"
        assert kwargs["data"]["code_verifier"] == "verifier-123"
        assert result == token_response

    def test_http_token_endpoint_rejected(self, monkeypatch):
        """Refuse to send client_secret over http:// without exemption (F2)."""
        monkeypatch.delenv("OIDC_ALLOW_HTTP_ISSUER", raising=False)
        with pytest.raises(ValueError, match="http://"):
            oidc.exchange_code_for_tokens(
                token_endpoint="http://idp.local:8080/token",
                code="c",
                redirect_uri="r",
                client_id="cid",
                client_secret="secret",
                code_verifier="v",
            )

    def test_http_token_endpoint_allowed_with_exemption(self, monkeypatch):
        """OIDC_ALLOW_HTTP_ISSUER=true permits http:// for local test IdPs (F2)."""
        monkeypatch.setenv("OIDC_ALLOW_HTTP_ISSUER", "true")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id_token": "x"}
        mock_resp.raise_for_status.return_value = None
        with patch.object(oidc.httpx.Client, "post", return_value=mock_resp) as mock_post:
            result = oidc.exchange_code_for_tokens(
                token_endpoint="http://idp.local:8080/token",
                code="c",
                redirect_uri="r",
                client_id="cid",
                client_secret="secret",
                code_verifier="v",
            )
        assert mock_post.call_count == 1
        assert result == {"id_token": "x"}


# ============================================================================
# ID Token verification — claim failure paths (F7)
# ============================================================================


class TestVerifyIdTokenClaimFailures:
    """verify_id_token must reject tokens with wrong aud, iss, or expired exp."""

    def test_aud_mismatch_raises(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("aud-mismatch-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "wrong-client-id",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="aud-mismatch-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="ID token verification failed"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="my-client-id",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_iss_mismatch_raises(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("iss-mismatch-kid")
        claims = {
            "iss": "https://wrong-issuer.example.com",
            "sub": "u1",
            "aud": "my-client-id",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="iss-mismatch-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="ID token verification failed"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="my-client-id",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_expired_token_raises(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("expired-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": "my-client-id",
            "exp": 1,  # far in the past, beyond the 120s leeway
            "iat": 1,
        }
        id_token = _sign_id_token(private_pem, claims, kid="expired-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="ID token verification failed"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="my-client-id",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )


# ============================================================================
# ID Token verification — azp (authorized party) validation (F7)
# ============================================================================


class TestVerifyIdTokenAzp:
    """verify_id_token azp validation per OIDC Core §3.1.3.7.3."""

    def test_azp_mismatch_raises(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("azp-mismatch-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": ["my-client-id", "other-client"],
            "azp": "wrong-client",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="azp-mismatch-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="azp mismatch"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="my-client-id",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_multi_audience_missing_azp_raises(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("no-azp-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "u1",
            "aud": ["my-client-id", "other-client"],
            "exp": 9999999999,
            # no azp — multi-audience tokens MUST carry azp
        }
        id_token = _sign_id_token(private_pem, claims, kid="no-azp-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            with pytest.raises(ValueError, match="missing required 'azp'"):
                oidc.verify_id_token(
                    id_token=id_token,
                    client_id="my-client-id",
                    issuer="https://accounts.example.com",
                    jwks_uri="https://accounts.example.com/jwks",
                )

    def test_azp_equals_client_id_passes(self):
        private_pem, jwks = _generate_rsa_key_and_jwks("azp-ok-kid")
        claims = {
            "iss": "https://accounts.example.com",
            "sub": "azp-user",
            "aud": ["my-client-id", "other-client"],
            "azp": "my-client-id",
            "exp": 9999999999,
        }
        id_token = _sign_id_token(private_pem, claims, kid="azp-ok-kid")
        with patch.object(oidc, "_fetch_jwks", return_value=jwks):
            result = oidc.verify_id_token(
                id_token=id_token,
                client_id="my-client-id",
                issuer="https://accounts.example.com",
                jwks_uri="https://accounts.example.com/jwks",
            )
        assert result["sub"] == "azp-user"
