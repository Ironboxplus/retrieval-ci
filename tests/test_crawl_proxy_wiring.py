"""TDD: settings.proxy must flow into EnhancedCrawler when use_proxy=True.

Bug discovered 2026-05-20: `_build_enhanced_crawler` sets
`config.proxy_config.enabled = use_proxy` but never populates
`config.proxy_config.proxies` from `settings.proxy.http_proxy`. So the runtime
contract "use_proxy=True → traffic actually routed through configured proxy"
silently breaks — `get_next_proxy()` returns None, browser launches direct.

Contract pinned here:
1. use_proxy=False → proxy_config.enabled=False AND .proxies=[]
2. use_proxy=True + settings.proxy.enabled + settings.proxy.http_proxy set →
   proxy_config.enabled=True AND .proxies contains the configured URL
3. get_next_proxy() returns a non-None string when use_proxy=True
"""
from __future__ import annotations

import pytest


def test_use_proxy_false_keeps_proxy_config_empty():
    from app.tasks.crawl_tasks import _build_enhanced_crawler

    crawler = _build_enhanced_crawler(use_proxy=False, max_concurrent=1)
    pc = crawler.config.proxy_config
    assert pc.enabled is False
    assert pc.proxies == [], (
        f"use_proxy=False must not inject proxies, got {pc.proxies}"
    )
    assert pc.get_next_proxy() is None


def test_use_proxy_true_injects_settings_proxy_url():
    """use_proxy=True must propagate settings.proxy.http_proxy into
    proxy_config.proxies. The proxy is always configured (no enabled flag)."""
    from app.tasks.crawl_tasks import _build_enhanced_crawler
    from config import settings

    assert settings.proxy.http_proxy, "test precondition: settings.proxy.http_proxy must be set"

    crawler = _build_enhanced_crawler(use_proxy=True, max_concurrent=1)
    pc = crawler.config.proxy_config

    assert pc.enabled is True
    assert len(pc.proxies) > 0, (
        f"BUG: use_proxy=True but no proxies injected from settings.proxy.http_proxy "
        f"(crawler will launch direct connection silently)"
    )
    assert settings.proxy.http_proxy in pc.proxies, (
        f"settings.proxy.http_proxy={settings.proxy.http_proxy!r} not in injected "
        f"proxies={pc.proxies!r}"
    )

    next_proxy = pc.get_next_proxy()
    assert next_proxy is not None
    assert next_proxy.startswith(("http://", "socks5://")), (
        f"Injected proxy must be playwright-compatible URL, got {next_proxy!r}"
    )
