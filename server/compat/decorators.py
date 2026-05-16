"""Shared decorators for server route handlers."""

import functools

from fastapi import HTTPException

from errors import upstream_error


def upstream_guard(func):
    """Decorator that converts unhandled exceptions into 502 ``UpstreamError``.

    ``HTTPException`` subclasses are re-raised as-is so that explicit 4xx
    responses from handler code are preserved.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception:
            raise upstream_error()

    return wrapper
