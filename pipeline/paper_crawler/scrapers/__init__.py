"""
Scrapers - 各数据源爬虫模块

支持的数据源:
- OpenReview (ICLR, NeurIPS等会议)
- ACL Anthology (ACL, EMNLP, NAACL等NLP会议)
- NeurIPS
- Semantic Scholar API
- arXiv
- Google Scholar (引用数获取)

基类提供:
- 统一的抽象接口
- 通用重试逻辑（带指数退避）
- 速率限制（令牌桶算法）
- 完善的错误处理
- 进度日志
- 支持 Playwright 和 Chrome MCP 浏览器自动化
"""

from .base_scraper import (
    # 基类
    BaseScraper,
    SyncScraperWrapper,

    # 结果类
    CrawlResult,

    # 异常类
    ScraperError,
    RateLimitError,
    AuthenticationError,
    NotFoundError,
    ParseError,
    NetworkError,
    BrowserError,

    # 配置类
    RateLimiter,
    RetryPolicy,
    BrowserBackend,
    BrowserContext,

    # 装饰器
    retry_async,
)

# ArxivScraper可以直接导入（不依赖可选包）
from .arxiv_scraper import ArxivScraper
from .openreview_scraper import OpenReviewScraper
from .acl_scraper import ACLScraper
from .neurips_scraper import NeurIPSScraper
from .semantic_scholar import SemanticScholarScraper
from .google_scholar_scraper import (
    GoogleScholarScraper,
    GoogleScholarConfig,
    FetchMethod,
    CaptchaError,
    get_citation_count,
    get_citation_count_batch,
)


# 工厂函数（保留用于向后兼容）
def get_openreview_scraper():
    """获取OpenReview爬虫类"""
    return OpenReviewScraper


def get_acl_scraper():
    """获取ACL爬虫类"""
    return ACLScraper


def get_neurips_scraper():
    """获取NeurIPS爬虫类"""
    return NeurIPSScraper


def get_semantic_scholar_scraper():
    """获取Semantic Scholar爬虫类"""
    return SemanticScholarScraper


def get_arxiv_scraper():
    """获取arXiv爬虫类"""
    return ArxivScraper


def get_google_scholar_scraper():
    """获取Google Scholar爬虫类"""
    return GoogleScholarScraper


__all__ = [
    # 基类
    "BaseScraper",
    "SyncScraperWrapper",

    # 结果类
    "CrawlResult",

    # 异常类
    "ScraperError",
    "RateLimitError",
    "AuthenticationError",
    "NotFoundError",
    "ParseError",
    "NetworkError",
    "BrowserError",
    "CaptchaError",

    # 配置类
    "RateLimiter",
    "RetryPolicy",
    "BrowserBackend",
    "BrowserContext",
    "GoogleScholarConfig",
    "FetchMethod",

    # 装饰器
    "retry_async",

    # 爬虫类
    "ArxivScraper",
    "OpenReviewScraper",
    "ACLScraper",
    "NeurIPSScraper",
    "SemanticScholarScraper",
    "GoogleScholarScraper",

    # 便捷函数
    "get_citation_count",
    "get_citation_count_batch",

    # 工厂函数
    "get_openreview_scraper",
    "get_acl_scraper",
    "get_neurips_scraper",
    "get_semantic_scholar_scraper",
    "get_arxiv_scraper",
    "get_google_scholar_scraper",
]
