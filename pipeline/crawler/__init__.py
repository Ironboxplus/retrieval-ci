"""
Crawler 模块 - 网页爬虫功能

提供基础爬虫和增强版爬虫（带Captcha检测和重试机制）
"""
from __future__ import annotations

from importlib import import_module

from .captcha_detector import (
    CaptchaDetector,
    CaptchaDetectionResult,
    CaptchaType,
)
from .retry_handler import (
    RetryHandler,
    RetryConfig,
    RateLimiter,
    BackoffStrategy,
    RetryableError,
    NetworkError,
    TimeoutError,
    RateLimitError,
    CaptchaError,
    BlockedError,
)

# The standalone medical downloader reuses only CaptchaDetector.  Importing a
# submodule still executes this package initializer, so browser crawler exports
# must stay lazy: the .88 retrieval runtime intentionally does not install the
# optional browser_use stack just to fetch a PDF through its HTTP proxy.
_LAZY_EXPORTS = {
    "BrowserCrawler": (".browser_crawler", "BrowserCrawler"),
    "CrawlResult": (".browser_crawler", "CrawlResult"),
    "EnhancedCrawler": (".enhanced_crawler", "EnhancedCrawler"),
    "EnhancedCrawlResult": (".enhanced_crawler", "EnhancedCrawlResult"),
    "CrawlerConfig": (".enhanced_crawler", "CrawlerConfig"),
    "ProxyConfig": (".enhanced_crawler", "ProxyConfig"),
    "CrawlStatusType": (".enhanced_crawler", "CrawlStatusType"),
    "crawl_with_retry": (".enhanced_crawler", "crawl_with_retry"),
    "batch_crawl": (".enhanced_crawler", "batch_crawl"),
}


def __getattr__(name: str):
    """Load browser-dependent crawler exports only when callers request them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, export_name = target
    value = getattr(import_module(module_name, __name__), export_name)
    globals()[name] = value
    return value

__all__ = [
    # 基础爬虫
    "BrowserCrawler",
    "CrawlResult",
    # Captcha检测
    "CaptchaDetector",
    "CaptchaDetectionResult",
    "CaptchaType",
    # 重试处理
    "RetryHandler",
    "RetryConfig",
    "RateLimiter",
    "BackoffStrategy",
    "RetryableError",
    "NetworkError",
    "TimeoutError",
    "RateLimitError",
    "CaptchaError",
    "BlockedError",
    # 增强版爬虫
    "EnhancedCrawler",
    "EnhancedCrawlResult",
    "CrawlerConfig",
    "ProxyConfig",
    "CrawlStatusType",
    "crawl_with_retry",
    "batch_crawl",
]
