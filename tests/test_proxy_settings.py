"""TDD: the proxy is MANDATORY — ProxySettings always routes external traffic
through the configured proxy, with NO direct-connection escape hatch.

Rationale (2026-06-16): the old ``enabled`` flag plus ``... if settings.proxy.
enabled else None`` fallbacks silently sent traffic DIRECT whenever the proxy was
"disabled" (e.g. ``PROXY__ENABLED=false`` on the 4090 box). That made open-venue
fetches (USENIX/DBLP/ACM) fail "Network is unreachable" instead of going through
the 10808 proxy on Arc. Per the user decision every external fetch MUST use the
proxy; the no-proxy code paths are deleted. These tests pin that contract.
"""
from __future__ import annotations

from config import ProxySettings


def test_proxy_settings_has_no_enabled_flag():
    # The global escape hatch is gone — there is no way to turn the proxy off.
    assert not hasattr(ProxySettings(), "enabled")


def test_get_proxy_dict_always_returns_configured_proxy():
    p = ProxySettings()
    assert p.get_proxy_dict() == {"http": p.http_proxy, "https": p.https_proxy}


def test_get_browser_proxy_always_returns_server():
    p = ProxySettings()
    assert p.get_browser_proxy() == {"server": p.http_proxy}


def test_default_proxy_is_10808():
    p = ProxySettings()
    assert p.http_proxy == "http://127.0.0.1:10808"
    assert p.https_proxy == "http://127.0.0.1:10808"
