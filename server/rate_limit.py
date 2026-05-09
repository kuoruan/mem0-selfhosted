import ipaddress
import logging
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

_TRUST_XFF = os.environ.get("TRUST_X_FORWARDED_FOR", "").strip().lower() in ("true", "1", "yes")


def _get_real_ip(request: Request) -> str:
    """Return the client IP, optionally respecting X-Forwarded-For.

    Enable by setting TRUST_X_FORWARDED_FOR=true. Only use when the server
    runs behind a trusted reverse proxy that sanitises this header.
    """
    if _TRUST_XFF:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            first_ip = forwarded.split(",")[0].strip()
            if first_ip:
                try:
                    ipaddress.ip_address(first_ip)
                    return first_ip
                except ValueError:
                    logger.warning("Invalid IP in X-Forwarded-For: %s", first_ip)
    return get_remote_address(request)


limiter = Limiter(key_func=_get_real_ip)
