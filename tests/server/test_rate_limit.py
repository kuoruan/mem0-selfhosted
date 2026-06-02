import importlib
import ipaddress
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("slowapi", reason="slowapi not installed")

import server.rate_limit as rate_limit


# ---------------------------------------------------------------------------
# _is_trusted_proxy – wildcard
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_wildcard_returns_true(monkeypatch):
    monkeypatch.setattr(rate_limit, "_ALWAYS_TRUST", True)
    assert rate_limit._is_trusted_proxy("10.0.0.1") is True
    assert rate_limit._is_trusted_proxy("192.168.1.1") is True
    assert rate_limit._is_trusted_proxy("::1") is True


# ---------------------------------------------------------------------------
# _is_trusted_proxy – exact IP
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_exact_match(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "127.0.0.1,10.0.0.1")
    monkeypatch.setattr(rate_limit, "_trusted_exact", {"127.0.0.1", "10.0.0.1"})
    monkeypatch.setattr(rate_limit, "_trusted_nets", [])

    assert rate_limit._is_trusted_proxy("127.0.0.1") is True
    assert rate_limit._is_trusted_proxy("10.0.0.1") is True
    assert rate_limit._is_trusted_proxy("10.0.0.2") is False


# ---------------------------------------------------------------------------
# _is_trusted_proxy – CIDR
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_cidr_match(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("10.0.0.0/8")])

    assert rate_limit._is_trusted_proxy("10.1.2.3") is True
    assert rate_limit._is_trusted_proxy("10.255.255.255") is True
    assert rate_limit._is_trusted_proxy("192.168.1.1") is False


def test_is_trusted_proxy_cidr_ipv6_match(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "fd00::/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("fd00::/8")])

    assert rate_limit._is_trusted_proxy("fd12:3456::1") is True
    assert rate_limit._is_trusted_proxy("2001:db8::1") is False


# ---------------------------------------------------------------------------
# _is_trusted_proxy – exact takes priority over CIDR (short-circuits)
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_exact_before_cidr(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", {"10.0.0.1"})
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("10.0.0.0/8")])

    assert rate_limit._is_trusted_proxy("10.0.0.1") is True
    assert rate_limit._is_trusted_proxy("10.0.0.2") is True  # via CIDR


# ---------------------------------------------------------------------------
# _is_trusted_proxy – empty allow-list (no matches)
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_empty_list(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [])

    assert rate_limit._is_trusted_proxy("127.0.0.1") is False
    assert rate_limit._is_trusted_proxy("10.0.0.1") is False


# ---------------------------------------------------------------------------
# _is_trusted_proxy – mixed address families (regression: C1 TypeError)
# ---------------------------------------------------------------------------
def test_is_trusted_proxy_ipv6_against_ipv4_nets_returns_false(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("10.0.0.0/8")])

    assert rate_limit._is_trusted_proxy("::1") is False


def test_is_trusted_proxy_ipv4_against_ipv6_nets_returns_false(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "fd00::/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("fd00::/8")])

    assert rate_limit._is_trusted_proxy("10.0.0.1") is False


def test_is_trusted_proxy_invalid_ip_returns_false(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    monkeypatch.setattr(rate_limit, "_trusted_exact", set())
    monkeypatch.setattr(rate_limit, "_trusted_nets", [ipaddress.ip_network("10.0.0.0/8")])

    assert rate_limit._is_trusted_proxy("not-an-ip") is False
    assert rate_limit._is_trusted_proxy(None) is False
    assert rate_limit._is_trusted_proxy(123) is False


def test_is_trusted_proxy_zone_index_stripped(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "::1")
    monkeypatch.setattr(rate_limit, "_trusted_exact", {"::1"})
    monkeypatch.setattr(rate_limit, "_trusted_nets", [])

    assert rate_limit._is_trusted_proxy("::1%lo") is True


def test_is_trusted_proxy_port_and_brackets_stripped(monkeypatch):
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "127.0.0.1,::1")
    monkeypatch.setattr(rate_limit, "_trusted_exact", {"127.0.0.1", "::1"})
    monkeypatch.setattr(rate_limit, "_trusted_nets", [])

    assert rate_limit._is_trusted_proxy("127.0.0.1:8080") is True
    assert rate_limit._is_trusted_proxy("[::1]") is True
    assert rate_limit._is_trusted_proxy("[::1]:8080") is True


# ---------------------------------------------------------------------------
# _parse_ip – port / bracket / zone stripping
# ---------------------------------------------------------------------------
def test_parse_ip_ipv4_with_port():
    assert str(rate_limit._parse_ip("1.2.3.4:8080")) == "1.2.3.4"
    assert str(rate_limit._parse_ip("10.0.0.1:3000")) == "10.0.0.1"


def test_parse_ip_bracketed_ipv6():
    assert str(rate_limit._parse_ip("[::1]")) == "::1"
    assert str(rate_limit._parse_ip("[2001:db8::1]")) == "2001:db8::1"


def test_parse_ip_bracketed_ipv6_with_port():
    assert str(rate_limit._parse_ip("[::1]:8080")) == "::1"
    assert str(rate_limit._parse_ip("[2001:db8::1]:443")) == "2001:db8::1"


def test_parse_ip_zone_index():
    assert str(rate_limit._parse_ip("::1%lo")) == "::1"
    assert str(rate_limit._parse_ip("fe80::1%eth0")) == "fe80::1"


def test_parse_ip_preserves_canonical():
    """Valid IPs without ports/brackets/zone are returned in canonical form."""
    assert str(rate_limit._parse_ip("::1")) == "::1"
    assert str(rate_limit._parse_ip("127.0.0.1")) == "127.0.0.1"
    assert str(rate_limit._parse_ip("0:0:0:0:0:0:0:1")) == "::1"


def test_parse_ip_invalid_inputs():
    with pytest.raises(ValueError):
        rate_limit._parse_ip("not-an-ip:8080")
    with pytest.raises(ValueError):
        rate_limit._parse_ip(None)


# ---------------------------------------------------------------------------
# _get_real_ip – no proxy (simple request)
# ---------------------------------------------------------------------------
def test_get_real_ip_no_proxy(monkeypatch):
    """When the remote IP is untrusted, X-Forwarded-For is ignored."""
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: ip == "10.0.0.1")

    mock_request = MagicMock()
    mock_request.headers.get.return_value = "1.2.3.4"  # x-forwarded-for

    with patch("server.rate_limit.get_remote_address", return_value="5.5.5.5"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "5.5.5.5"  # untrusted, use remote IP


# ---------------------------------------------------------------------------
# _get_real_ip – trusted proxy with X-Forwarded-For
# ---------------------------------------------------------------------------
def test_get_real_ip_trusted_proxy_returns_first_untrusted(monkeypatch):
    """Traverse X-Forwarded-For right-to-left, return the first untrusted host."""
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.1")
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: str(ip) == "10.0.0.1")

    mock_request = MagicMock()
    mock_request.headers.get.return_value = "1.2.3.4, 6.7.8.9"

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "6.7.8.9"  # rightmost untrusted (only direct proxy is trusted)


def test_get_real_ip_all_hosts_trusted_returns_leftmost(monkeypatch):
    """When all X-Forwarded-For hosts are trusted, fall back to the leftmost."""
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.1,6.7.8.9,1.2.3.4")
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: str(ip) in ("10.0.0.1", "6.7.8.9", "1.2.3.4"))

    mock_request = MagicMock()
    mock_request.headers.get.return_value = "1.2.3.4, 6.7.8.9, 10.0.0.1"

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "1.2.3.4"  # all trusted → leftmost


def test_get_real_ip_returns_canonical_form(monkeypatch):
    """Returned IPs are canonicalized to a consistent rate-limit key."""
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.1")
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: str(ip) == "10.0.0.1")

    mock_request = MagicMock()
    # X-Forwarded-For with both canonical and non-canonical IPv6 forms
    mock_request.headers.get.return_value = "0:0:0:0:0:0:0:1, ::1"

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "::1"  # canonical, not "0:0:0:0:0:0:0:1"


def test_get_real_ip_multiple_trusted_proxies_returns_first_untrusted(monkeypatch):
    """With multiple trusted chain proxies, stop at the first untrusted."""
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.1,6.7.8.9")
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: str(ip) in ("10.0.0.1", "6.7.8.9"))

    mock_request = MagicMock()
    mock_request.headers.get.return_value = "1.2.3.4, 6.7.8.9, 10.0.0.1"

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "1.2.3.4"  # first untrusted after right-to-left scan


# ---------------------------------------------------------------------------
# _get_real_ip – trusted proxy, invalid X-Forwarded-For falls back
# ---------------------------------------------------------------------------
def test_get_real_ip_trusted_proxy_invalid_xff_stops_on_invalid(monkeypatch):
    """Invalid IP in X-Forwarded-For falls back to secure remote_ip."""
    monkeypatch.setattr(rate_limit, "_FORWARDED_ALLOW_IPS", "10.0.0.1")
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: str(ip) == "10.0.0.1")

    mock_request = MagicMock()
    mock_request.headers.get.return_value = "1.2.3.4, not-an-ip"

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "10.0.0.1"  # fall back to remote_ip, not the invalid string


# ---------------------------------------------------------------------------
# _get_real_ip – trusted proxy, empty X-Forwarded-For
# ---------------------------------------------------------------------------
def test_get_real_ip_trusted_proxy_empty_xff_falls_back(monkeypatch):
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: ip == "10.0.0.1")

    mock_request = MagicMock()
    mock_request.headers.get.return_value = ""

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "10.0.0.1"


# ---------------------------------------------------------------------------
# _get_real_ip – trusted proxy, no X-Forwarded-For header
# ---------------------------------------------------------------------------
def test_get_real_ip_trusted_proxy_no_xff_header(monkeypatch):
    monkeypatch.setattr(rate_limit, "_is_trusted_proxy", lambda ip: ip == "10.0.0.1")

    mock_request = MagicMock()
    mock_request.headers.get.return_value = None

    with patch("server.rate_limit.get_remote_address", return_value="10.0.0.1"):
        result = rate_limit._get_real_ip(mock_request)

    assert result == "10.0.0.1"


# ---------------------------------------------------------------------------
# Module-level pre-parsing – correct parsing at import time
# ---------------------------------------------------------------------------
class TestModuleLevelParsing:
    """Verify that the allow-list is parsed correctly when the module is imported."""

    @pytest.fixture(autouse=True)
    def _reload_clean(self):
        """Avoid leaking state between tests."""
        yield
        importlib.reload(rate_limit)

    def test_wildcard_skips_parsing(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == set()
        assert rate_limit._trusted_nets == []

    def test_exact_ips_parsed(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1, 10.0.0.1")
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == {"127.0.0.1", "10.0.0.1"}
        assert rate_limit._trusted_nets == []

    def test_cidr_parsed(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8, 192.168.0.0/16")
        importlib.reload(rate_limit)
        nets = [str(n) for n in rate_limit._trusted_nets]
        assert "10.0.0.0/8" in nets
        assert "192.168.0.0/16" in nets
        assert rate_limit._trusted_exact == set()

    def test_mixed_exact_and_cidr_parsed(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1,10.0.0.0/8")
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == {"127.0.0.1"}
        nets = [str(n) for n in rate_limit._trusted_nets]
        assert "10.0.0.0/8" in nets

    def test_invalid_entries_skipped(self, monkeypatch, caplog):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "not-an-ip, 127.0.0.1, also-bad")
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == {"127.0.0.1"}
        assert rate_limit._trusted_nets == []
        # Warnings should be logged for invalid entries
        assert "Invalid entry in FORWARDED_ALLOW_IPS: not-an-ip" in caplog.text
        assert "Invalid entry in FORWARDED_ALLOW_IPS: also-bad" in caplog.text

    def test_empty_string_produces_empty_sets(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "")
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == set()
        assert rate_limit._trusted_nets == []

    def test_default_ip_parsed(self, monkeypatch):
        monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
        importlib.reload(rate_limit)
        assert rate_limit._trusted_exact == {"127.0.0.1"}
        assert rate_limit._trusted_nets == []

    def test_wildcard_with_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "  *  ")
        importlib.reload(rate_limit)
        assert rate_limit._FORWARDED_ALLOW_IPS == "*"
        assert rate_limit._is_trusted_proxy("10.0.0.1") is True
