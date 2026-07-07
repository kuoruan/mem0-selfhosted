"""PKCE (Proof Key for Code Exchange, RFC 7636) helpers.

Used by the OIDC Authorization Code Flow to bind the authorization request to
the token request, preventing authorization-code interception.
"""

import base64
import hashlib
import secrets


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
