"""Shared decorators for server route handlers."""

import functools

from errors import upstream_error
from fastapi import HTTPException

from mem0.exceptions import ValidationError as Mem0ValidationError


def upstream_guard(func):
    """Decorator that converts unhandled exceptions into appropriate HTTP errors.

    Exception mapping:
      ``HTTPException`` — re-raised as-is (preserves explicit 4xx from handlers).
      ``Mem0ValidationError`` / ``ValueError`` — converted to 400 (input validation).
      Everything else — converted to 502 ``UpstreamError`` via ``_classify``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        # Mem0ValidationError: SDK input validation (e.g. invalid messages format).
        # ValueError: SDK filter/param validation (e.g. unsupported operator, missing entity).
        # Both are always client errors. Handler bugs typically raise TypeError/AttributeError
        # which still fall through to the generic handler below.
        except (Mem0ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            raise upstream_error()

    return wrapper
