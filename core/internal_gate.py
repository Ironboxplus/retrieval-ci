"""Pure decision logic for the 4090 compute box's internal-access gate.

Design (so 内网直连依然免认证，便于局域网使用):
  * LAN/intranet clients — private, NON-loopback IPs (10/8, 172.16/12, 192.168/16,
    incl. the lab 10.100.100.x overlay) — are trusted WITHOUT a token. Direct
    intranet access stays frictionless and free.
  * The PUBLIC path arrives via the reverse tunnel and lands on uvicorn as a
    LOOPBACK peer (127.0.0.1), so loopback is deliberately NOT treated as LAN: it
    must present the shared X-Internal-Token that the public gateway injects.
  * OPTIONS preflight + exempt paths (health checks) always pass.

Kept framework-free so it's unit-tested without importing the heavy main app.
"""
from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Container


def is_private_lan(host: str | None) -> bool:
    """True for a private, routable-on-LAN address — explicitly NOT loopback and
    NOT link-local (those are the tunnel / non-trustworthy cases)."""
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private and not ip.is_loopback and not ip.is_link_local


def gate_allows(*, method: str, path: str, client_host: str | None,
                token_header: str | None, expected_token: str,
                exempt: Container[str], trust_lan: bool = True) -> bool:
    """Whether a request may pass the internal gate."""
    if method == "OPTIONS" or path in exempt:
        return True
    if trust_lan and is_private_lan(client_host):
        return True
    if expected_token and token_header and hmac.compare_digest(token_header, expected_token):
        return True
    return False
