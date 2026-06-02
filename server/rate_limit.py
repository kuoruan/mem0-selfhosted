import ipaddress
import logging
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

_FORWARDED_ALLOW_IPS = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1").strip()
_ALWAYS_TRUST = _FORWARDED_ALLOW_IPS == "*"

# Pre-parse the allow-list once at import time.
_trusted_exact: set[str] = set()
_trusted_nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []

if not _ALWAYS_TRUST:
    for entry in _FORWARDED_ALLOW_IPS.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                _trusted_nets.append(ipaddress.ip_network(entry, strict=False))
            else:
                addr = ipaddress.ip_address(entry)  # validate + canonicalize
                _trusted_exact.add(str(addr))
        except ValueError:
            logger.warning("Invalid entry in FORWARDED_ALLOW_IPS: %s", entry)


def _parse_ip(ip_str: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Parse *ip_str* to an ``ipaddress`` object, stripping brackets, ports, or zone index if present."""
    if not isinstance(ip_str, str):
        raise ValueError("IP address must be a string")
    ip_str = ip_str.strip()
    # Strip brackets and port from bracketed IPv6 (e.g. "[::1]:8080" → "::1")
    if ip_str.startswith("["):
        end_bracket = ip_str.find("]")
        if end_bracket != -1:
            ip_str = ip_str[1:end_bracket]
    # Strip port from IPv4 (e.g. "1.2.3.4:8080" → "1.2.3.4")
    elif ip_str.count(":") == 1 and "." in ip_str:
        ip_str = ip_str.split(":")[0]
    # Strip zone index (e.g. "::1%lo" → "::1")
    zone_idx = ip_str.find("%")
    if zone_idx != -1:
        ip_str = ip_str[:zone_idx]
    return ipaddress.ip_address(ip_str)


def _is_trusted_proxy(remote_ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if *remote_ip* is a trusted proxy per FORWARDED_ALLOW_IPS."""
    if _ALWAYS_TRUST:
        return True
    if isinstance(remote_ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        addr = remote_ip
    else:
        try:
            addr = _parse_ip(remote_ip)
        except ValueError:
            return False
    if str(addr) in _trusted_exact:
        return True
    if _trusted_nets:
        return any(addr in net for net in _trusted_nets if addr.version == net.version)
    return False


def _get_real_ip(request: Request) -> str:
    """Return the client IP, respecting X-Forwarded-For from trusted proxies.

    Uses the same FORWARDED_ALLOW_IPS env var as uvicorn.  When set to ``"*"``
    all proxies are trusted and the leftmost X-Forwarded-For IP is returned.
    Otherwise the header is traversed from right to left (each proxy appends
    to the list), skipping trusted hosts until the first untrusted IP.  This
    mirrors uvicorn's ``ProxyHeadersMiddleware`` logic.
    """
    remote_ip = get_remote_address(request)
    if not _is_trusted_proxy(remote_ip):
        return remote_ip

    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return remote_ip

    hosts = [h.strip() for h in forwarded.split(",") if h.strip()]
    if not hosts:
        return remote_ip

    if _ALWAYS_TRUST:
        try:
            return str(_parse_ip(hosts[0]))
        except ValueError:
            logger.warning("Invalid IP in X-Forwarded-For: %s", hosts[0])
            return remote_ip

    # Traverse from right to left, skipping trusted proxies.
    last_parsed_addr = None
    for host in reversed(hosts):
        try:
            addr = _parse_ip(host)
        except ValueError:
            logger.warning("Invalid IP in X-Forwarded-For: %s", host)
            return remote_ip
        if not _is_trusted_proxy(addr):
            return str(addr)
        last_parsed_addr = addr

    # All hosts are trusted — fall back to the leftmost.
    return str(last_parsed_addr) if last_parsed_addr is not None else remote_ip


limiter = Limiter(key_func=_get_real_ip)
