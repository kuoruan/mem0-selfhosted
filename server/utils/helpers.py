"""General-purpose helper utilities for the mem0 server."""

import re
from urllib.parse import urlparse


def is_http_url(url: str) -> bool:
    """Return ``True`` if *url* starts with ``http://`` or ``https://`` (case-insensitive)."""
    return url.lower().startswith(("http://", "https://"))


def is_safe_redirect(url: str | None) -> bool:
    """Return ``True`` if *url* is a safe relative redirect target.

    Only relative paths (no scheme, no netloc) are allowed. Rejects whitespace,
    control characters, and backslashes that browsers normalize into open-redirect
    vectors.
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


def sanitize_for_log(value: str) -> str:
    """Reduce *value* to ``[A-Za-z0-9_.-]`` so untrusted text cannot forge log entries.

    Newlines and control characters in user-supplied input (e.g. an OIDC provider
    path segment) are replaced with ``_`` before reaching a log line.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)
