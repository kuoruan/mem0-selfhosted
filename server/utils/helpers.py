"""General-purpose helper utilities for the mem0 server."""


def is_http_url(url: str) -> bool:
    """Return ``True`` if *url* starts with ``http://`` or ``https://`` (case-insensitive)."""
    return url.lower().startswith(("http://", "https://"))
