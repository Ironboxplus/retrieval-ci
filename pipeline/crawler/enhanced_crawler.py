"""
增强版爬虫 - 集成Captcha检测、重试机制和代理轮换
"""
import asyncio
import json
import os
import random
import time
import logging
from typing import List, Dict, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from playwright.async_api import async_playwright, Page, Response, Browser, BrowserContext

from config import settings
from app.models.schemas import CrawlStatus
from pipeline.crawler.captcha_detector import CaptchaDetector, CaptchaDetectionResult, CaptchaType
from pipeline.crawler.retry_handler import (
    RetryHandler, RetryConfig, RateLimiter, BackoffStrategy,
    CaptchaError, BlockedError, NetworkError, TimeoutError as RetryTimeoutError
)

logger = logging.getLogger(__name__)


class CrawlStatusType(Enum):
    """爬取状态类型"""
    PENDING = "pending"
    CRAWLING = "crawling"
    SUCCESS = "success"
    FAILED = "failed"
    CAPTCHA = "captcha"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class EnhancedCrawlResult:
    """增强版爬取结果"""
    url: str
    title: str
    content: str
    html: str
    status: CrawlStatusType
    status_code: Optional[int] = None
    error: Optional[str] = None
    captcha_detected: bool = False
    captcha_type: Optional[CaptchaType] = None
    retry_count: int = 0
    crawl_time: float = 0.0
    crawled_at: datetime = None
    proxy_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.crawled_at is None:
            self.crawled_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "url": self.url,
            "title": self.title,
            "content_length": len(self.content),
            "status": self.status.value,
            "status_code": self.status_code,
            "error": self.error,
            "captcha_detected": self.captcha_detected,
            "captcha_type": self.captcha_type.value if self.captcha_type else None,
            "retry_count": self.retry_count,
            "crawl_time": self.crawl_time,
            "crawled_at": self.crawled_at.isoformat() if self.crawled_at else None,
            "proxy_used": self.proxy_used,
            "metadata": self.metadata,
        }


@dataclass
class ProxyConfig:
    """代理配置"""
    enabled: bool = False
    proxies: List[str] = field(default_factory=list)
    rotation_strategy: str = "round_robin"  # round_robin, random, failover
    current_index: int = 0
    failed_proxies: Set[str] = field(default_factory=set)
    max_failures_per_proxy: int = 3
    proxy_failures: Dict[str, int] = field(default_factory=dict)

    def get_next_proxy(self) -> Optional[str]:
        """获取下一个代理"""
        if not self.enabled or not self.proxies:
            return None

        available = [p for p in self.proxies if p not in self.failed_proxies]
        if not available:
            # 重置失败列表
            self.failed_proxies.clear()
            self.proxy_failures.clear()
            available = self.proxies

        if self.rotation_strategy == "random":
            return random.choice(available)
        elif self.rotation_strategy == "failover":
            return available[0]
        else:  # round_robin
            proxy = available[self.current_index % len(available)]
            self.current_index += 1
            return proxy

    def mark_failed(self, proxy: str):
        """标记代理失败"""
        self.proxy_failures[proxy] = self.proxy_failures.get(proxy, 0) + 1
        if self.proxy_failures[proxy] >= self.max_failures_per_proxy:
            self.failed_proxies.add(proxy)
            logger.warning(f"Proxy marked as failed: {proxy}")


@dataclass
class CrawlerConfig:
    """爬虫配置"""
    # 基本配置
    headless: bool = True
    timeout: int = 30000  # ms
    max_concurrent: int = 5
    delay_between_requests: float = 1.0  # seconds

    # 重试配置
    retry_config: RetryConfig = field(default_factory=lambda: RetryConfig(
        max_retries=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
        initial_delay=2.0,
        max_delay=30.0,
    ))

    # 频率限制
    rate_limit_enabled: bool = True
    requests_per_second: float = 1.0
    burst_size: int = 3

    # 代理配置
    proxy_config: ProxyConfig = field(default_factory=ProxyConfig)

    # Captcha处理
    detect_captcha: bool = True
    pause_on_captcha: bool = True
    captcha_wait_time: float = 60.0  # 检测到验证码后等待时间

    # 日志配置
    log_level: str = "INFO"
    save_failed_html: bool = True
    failed_html_dir: str = ""

    def __post_init__(self):
        if not self.failed_html_dir:
            self.failed_html_dir = os.path.join(settings.data_dir, "failed_crawls")


class EnhancedCrawler:
    """增强版浏览器爬虫"""

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        use_proxy: bool = False
    ):
        """
        初始化增强版爬虫

        Args:
            config: 爬虫配置
            use_proxy: 是否使用代理
        """
        self.config = config or CrawlerConfig()
        if use_proxy:
            self.config.proxy_config.enabled = True

        self.captcha_detector = CaptchaDetector()
        self.retry_handler = RetryHandler(self.config.retry_config)
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.requests_per_second,
            burst_size=self.config.burst_size
        ) if self.config.rate_limit_enabled else None

        self.results: List[EnhancedCrawlResult] = []
        self.cancelled: bool = False
        self.status_callback: Optional[Callable[[str, CrawlStatusType, str], None]] = None

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "captcha_detected": 0,
            "blocked": 0,
            "retries": 0,
            "total_time": 0.0,
        }

        # 确保失败HTML目录存在
        if self.config.save_failed_html:
            os.makedirs(self.config.failed_html_dir, exist_ok=True)

    def cancel(self):
        """取消爬取操作"""
        self.cancelled = True
        logger.info("Crawler cancellation requested")

    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self.cancelled

    def set_status_callback(
        self,
        callback: Callable[[str, CrawlStatusType, str], None]
    ):
        """设置状态回调函数"""
        self.status_callback = callback

    def _notify_status(self, url: str, status: CrawlStatusType, message: str = ""):
        """通知状态变化"""
        if self.status_callback:
            try:
                self.status_callback(url, status, message)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    async def crawl_urls(
        self,
        urls: List[str],
        max_concurrent: Optional[int] = None
    ) -> List[EnhancedCrawlResult]:
        """
        批量爬取URL

        Args:
            urls: URL列表
            max_concurrent: 最大并发数

        Returns:
            爬取结果列表
        """
        if not urls:
            return []

        self.cancelled = False
        self.results = []
        max_concurrent = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        start_time = time.time()

        async def crawl_with_semaphore(url: str) -> EnhancedCrawlResult:
            if self.cancelled:
                return EnhancedCrawlResult(
                    url=url,
                    title="",
                    content="",
                    html="",
                    status=CrawlStatusType.CANCELLED,
                    error="Operation cancelled"
                )

            async with semaphore:
                return await self.crawl_single(url)

        tasks = [crawl_with_semaphore(url) for url in urls]
        self.results = await asyncio.gather(*tasks, return_exceptions=False)

        self.stats["total_time"] = time.time() - start_time

        # 更新统计
        for result in self.results:
            if result.status == CrawlStatusType.SUCCESS:
                self.stats["successful"] += 1
            elif result.status == CrawlStatusType.CAPTCHA:
                self.stats["captcha_detected"] += 1
            elif result.status == CrawlStatusType.BLOCKED:
                self.stats["blocked"] += 1
            else:
                self.stats["failed"] += 1

        return self.results

    async def crawl_single(self, url: str) -> EnhancedCrawlResult:
        """
        爬取单个URL

        Args:
            url: 目标URL

        Returns:
            爬取结果
        """
        self.stats["total_requests"] += 1
        self._notify_status(url, CrawlStatusType.CRAWLING, "Starting crawl")
        start_time = time.time()
        retry_count = 0
        last_error = None
        proxy_used = None

        async def do_crawl() -> EnhancedCrawlResult:
            nonlocal retry_count, last_error, proxy_used

            if self.cancelled:
                raise asyncio.CancelledError("Operation cancelled")

            # 频率限制
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # 获取代理
            proxy_used = self.config.proxy_config.get_next_proxy()

            async with async_playwright() as p:
                # 配置浏览器选项
                launch_options = {
                    "headless": self.config.headless,
                }

                # 代理配置
                if proxy_used:
                    launch_options["proxy"] = {"server": proxy_used}

                browser = await p.chromium.launch(**launch_options)

                try:
                    result = await self._crawl_page(browser, url, proxy_used)
                    return result

                except Exception as e:
                    retry_count += 1
                    last_error = e
                    self.stats["retries"] += 1

                    # 标记代理失败
                    if proxy_used:
                        self.config.proxy_config.mark_failed(proxy_used)

                    # 转换为适当的异常类型
                    error_msg = str(e).lower()
                    if "timeout" in error_msg:
                        raise RetryTimeoutError(str(e))
                    elif "net::" in error_msg or "connection" in error_msg:
                        raise NetworkError(str(e))
                    else:
                        raise

                finally:
                    await browser.close()

        try:
            result = await self.retry_handler.execute_with_retry(
                do_crawl,
                cancelled_check=self.is_cancelled
            )
            result.retry_count = retry_count
            result.crawl_time = time.time() - start_time

            if result.status == CrawlStatusType.SUCCESS:
                self._notify_status(url, CrawlStatusType.SUCCESS, "Crawl completed")
            else:
                self._notify_status(url, result.status, result.error or "Unknown error")

            return result

        except asyncio.CancelledError:
            return EnhancedCrawlResult(
                url=url,
                title="",
                content="",
                html="",
                status=CrawlStatusType.CANCELLED,
                error="Operation cancelled",
                retry_count=retry_count,
                crawl_time=time.time() - start_time,
                proxy_used=proxy_used
            )

        except CaptchaError as e:
            self._notify_status(url, CrawlStatusType.CAPTCHA, str(e))
            return EnhancedCrawlResult(
                url=url,
                title="",
                content="",
                html="",
                status=CrawlStatusType.CAPTCHA,
                error=str(e),
                captcha_detected=True,
                retry_count=retry_count,
                crawl_time=time.time() - start_time,
                proxy_used=proxy_used
            )

        except BlockedError as e:
            self._notify_status(url, CrawlStatusType.BLOCKED, str(e))
            return EnhancedCrawlResult(
                url=url,
                title="",
                content="",
                html="",
                status=CrawlStatusType.BLOCKED,
                error=str(e),
                retry_count=retry_count,
                crawl_time=time.time() - start_time,
                proxy_used=proxy_used
            )

        except Exception as e:
            error_msg = str(last_error or e)
            self._notify_status(url, CrawlStatusType.FAILED, error_msg)
            return EnhancedCrawlResult(
                url=url,
                title="",
                content="",
                html="",
                status=CrawlStatusType.FAILED,
                error=error_msg,
                retry_count=retry_count,
                crawl_time=time.time() - start_time,
                proxy_used=proxy_used
            )

    async def _crawl_page(
        self,
        browser: Browser,
        url: str,
        proxy_used: Optional[str]
    ) -> EnhancedCrawlResult:
        """实际执行页面爬取"""
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        page.set_default_timeout(self.config.timeout)

        response: Optional[Response] = None
        status_code = None

        try:
            # 导航到页面
            response = await page.goto(url, wait_until="networkidle")
            status_code = response.status if response else None

            # 等待内容加载
            await asyncio.sleep(self.config.delay_between_requests)

            # 获取HTML内容
            html = await page.content()

            # 检测Captcha
            if self.config.detect_captcha:
                headers = {}
                if response:
                    headers = dict(response.headers)

                captcha_result = self.captcha_detector.detect_from_html(
                    html, status_code or 200, headers
                )

                if captcha_result.detected:
                    logger.warning(
                        f"Captcha detected at {url}: {captcha_result.captcha_type.value}"
                    )

                    # 保存失败的HTML
                    if self.config.save_failed_html:
                        await self._save_failed_html(url, html, "captcha")

                    if self.config.pause_on_captcha:
                        raise CaptchaError(
                            f"Captcha detected: {captcha_result.captcha_type.value}"
                        )

                    return EnhancedCrawlResult(
                        url=url,
                        title="",
                        content="",
                        html=html,
                        status=CrawlStatusType.CAPTCHA,
                        status_code=status_code,
                        captcha_detected=True,
                        captcha_type=captcha_result.captcha_type,
                        proxy_used=proxy_used
                    )

                # 检查是否被阻止
                is_blocked, reason = self.captcha_detector.is_blocked_response(
                    html, status_code or 200
                )

                if is_blocked:
                    logger.warning(f"Blocked at {url}: {reason}")

                    if self.config.save_failed_html:
                        await self._save_failed_html(url, html, "blocked")

                    raise BlockedError(f"Access blocked: {reason}")

            # 提取内容
            title = await page.title()
            content = await self._extract_content(page)

            # 保存结果
            result = EnhancedCrawlResult(
                url=url,
                title=title,
                content=content,
                html=html,
                status=CrawlStatusType.SUCCESS,
                status_code=status_code,
                proxy_used=proxy_used
            )

            await self._save_result(result)

            return result

        except (CaptchaError, BlockedError):
            raise

        except Exception as e:
            # 尝试保存失败的HTML
            try:
                if self.config.save_failed_html:
                    html = await page.content()
                    await self._save_failed_html(url, html, "error")
            except Exception:
                pass
            raise

        finally:
            await context.close()

    async def _extract_content(self, page: Page) -> str:
        """从页面提取主要文本内容"""
        content = await page.evaluate("""
            () => {
                // 移除脚本和样式
                const elementsToRemove = document.querySelectorAll(
                    'script, style, nav, footer, header, aside, .ad, .advertisement, .sidebar, .cookie-notice, .popup'
                );
                elementsToRemove.forEach(el => el.remove());

                // 尝试找到主内容区域
                const mainContent = document.querySelector(
                    'main, article, .content, .post, .entry, #content, #main, [role="main"]'
                ) || document.body;

                // 提取文本
                const text = mainContent.innerText || mainContent.textContent;

                // 清理空白
                return text.replace(/\\s+/g, ' ').trim();
            }
        """)

        return content

    async def _save_result(self, result: EnhancedCrawlResult):
        """保存爬取结果到文件"""
        safe_name = "".join(c if c.isalnum() else "_" for c in result.url)[:100]
        timestamp = result.crawled_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_name}.json"

        filepath = os.path.join(settings.crawled_dir, filename)

        data = result.to_dict()
        data["content"] = result.content  # 包含完整内容

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _save_failed_html(self, url: str, html: str, reason: str):
        """保存失败的HTML内容用于调试"""
        safe_name = "".join(c if c.isalnum() else "_" for c in url)[:80]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reason}_{timestamp}_{safe_name}.html"

        filepath = os.path.join(self.config.failed_html_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"<!-- URL: {url} -->\n")
                f.write(f"<!-- Reason: {reason} -->\n")
                f.write(f"<!-- Time: {timestamp} -->\n")
                f.write(html)
        except Exception as e:
            logger.error(f"Failed to save failed HTML: {e}")

    def get_status(self) -> List[CrawlStatus]:
        """获取爬取状态"""
        return [
            CrawlStatus(
                url=r.url,
                status=r.status.value,
                content_length=len(r.content),
                error=r.error
            )
            for r in self.results
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "captcha_detected": 0,
            "blocked": 0,
            "retries": 0,
            "total_time": 0.0,
        }


# 便捷函数
async def crawl_with_retry(
    url: str,
    max_retries: int = 3,
    use_proxy: bool = False
) -> EnhancedCrawlResult:
    """
    便捷函数：带重试的单URL爬取

    Args:
        url: 目标URL
        max_retries: 最大重试次数
        use_proxy: 是否使用代理

    Returns:
        爬取结果
    """
    config = CrawlerConfig()
    config.retry_config.max_retries = max_retries
    config.proxy_config.enabled = use_proxy

    crawler = EnhancedCrawler(config)
    return await crawler.crawl_single(url)


async def batch_crawl(
    urls: List[str],
    max_concurrent: int = 5,
    use_proxy: bool = False,
    cancelled_check: Optional[Callable[[], bool]] = None
) -> List[EnhancedCrawlResult]:
    """
    便捷函数：批量爬取URL

    Args:
        urls: URL列表
        max_concurrent: 最大并发数
        use_proxy: 是否使用代理
        cancelled_check: 取消检查函数

    Returns:
        爬取结果列表
    """
    config = CrawlerConfig(max_concurrent=max_concurrent)
    config.proxy_config.enabled = use_proxy

    crawler = EnhancedCrawler(config)

    if cancelled_check:
        # 定期检查取消状态
        async def check_cancelled():
            while not crawler.cancelled:
                if cancelled_check():
                    crawler.cancel()
                    break
                await asyncio.sleep(0.5)

        check_task = asyncio.create_task(check_cancelled())

        try:
            results = await crawler.crawl_urls(urls, max_concurrent)
        finally:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass

        return results

    return await crawler.crawl_urls(urls, max_concurrent)
