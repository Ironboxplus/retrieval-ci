"""
BaseScraper - 论文爬虫基类

定义所有论文爬虫的通用接口和基础功能。

特性:
- 抽象基类定义统一接口
- 通用重试逻辑（带指数退避）
- 速率限制（令牌桶算法）
- 完善的错误处理
- 进度日志
- 支持 Playwright 和 Chrome MCP 浏览器自动化
- 异步优先设计
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    List, Optional, Dict, Any, AsyncIterator,
    Callable, TypeVar, Generic, Union, Awaitable, Type
)
import asyncio
import time
import random
import traceback

import httpx
from httpx import AsyncClient, Response, HTTPStatusError, TimeoutException

from ..models.paper import (
    Paper, PaperType, CrawlTask, CrawlProgress, TaskStatus, Conference
)
from ..config import Settings, get_settings
from ..utils.logger import get_logger, ProgressLogger


# 爬取结果包装类
@dataclass
class CrawlResult:
    """爬取结果"""
    task: CrawlTask
    papers: List[Paper] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.papers)

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    def summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        return {
            "task_id": self.task.id,
            "conferences": [c.value for c in self.task.target_conferences],
            "years": f"{self.task.year_start}-{self.task.year_end}",
            "total_papers": len(self.papers),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
        }


# 类型变量
T = TypeVar('T')


class ScraperError(Exception):
    """爬虫基础异常"""

    def __init__(self, message: str, source: str = "", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.source = source
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "source": self.source,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class RateLimitError(ScraperError):
    """速率限制异常"""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(ScraperError):
    """认证异常"""
    pass


class NotFoundError(ScraperError):
    """资源不存在异常"""
    pass


class ParseError(ScraperError):
    """解析异常"""
    pass


class NetworkError(ScraperError):
    """网络异常"""
    pass


class BrowserError(ScraperError):
    """浏览器自动化异常"""
    pass


@dataclass
class RateLimiter:
    """速率限制器（令牌桶算法）

    控制请求频率，防止触发目标站点的反爬措施。
    """

    requests_per_second: float = 1.0
    burst_size: int = 10

    # 内部状态
    _tokens: float = field(default=0, init=False)
    _last_update: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.burst_size)

    async def acquire(self, tokens: int = 1) -> float:
        """获取令牌

        Args:
            tokens: 需要的令牌数量

        Returns:
            等待的时间（秒）
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # 补充令牌
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.requests_per_second
            )

            # 计算等待时间
            wait_time = 0.0
            if self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= tokens

            return wait_time


@dataclass
class RetryPolicy:
    """重试策略配置"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    # 需要重试的状态码
    retry_status_codes: List[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # 需要重试的异常类型
    retry_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [
            TimeoutException,
            NetworkError,
            ConnectionError,
            asyncio.TimeoutError,
        ]
    )

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # 添加随机抖动，避免雷群效应
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(
        self,
        attempt: int,
        exception: Optional[Exception] = None,
        status_code: Optional[int] = None,
    ) -> bool:
        """判断是否应该重试

        Args:
            attempt: 当前尝试次数
            exception: 捕获的异常
            status_code: HTTP状态码

        Returns:
            是否重试
        """
        if attempt >= self.max_retries:
            return False

        if status_code and status_code in self.retry_status_codes:
            return True

        if exception:
            for exc_type in self.retry_exceptions:
                if isinstance(exception, exc_type):
                    return True

        return False


class BrowserBackend(Enum):
    """浏览器后端类型"""
    NONE = "none"  # 不使用浏览器
    PLAYWRIGHT = "playwright"
    CHROME_MCP = "chrome_mcp"  # Chrome DevTools MCP


@dataclass
class BrowserContext:
    """浏览器上下文

    抽象浏览器操作，支持 Playwright 和 Chrome MCP。
    """

    backend: BrowserBackend = BrowserBackend.NONE
    page: Optional[Any] = None  # Playwright Page 或 None
    browser: Optional[Any] = None  # Playwright Browser 或 None

    # Playwright 配置
    headless: bool = True
    timeout: int = 30000
    viewport: Dict[str, int] = field(
        default_factory=lambda: {"width": 1920, "height": 1080}
    )
    user_agent: Optional[str] = None

    async def navigate(self, url: str) -> None:
        """导航到URL"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            await self.page.goto(url, wait_until="networkidle")
        elif self.backend == BrowserBackend.CHROME_MCP:
            # Chrome MCP 通过外部工具调用
            raise NotImplementedError("Chrome MCP navigation should be called externally")

    async def get_content(self) -> str:
        """获取页面内容"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            return await self.page.content()
        return ""

    async def wait_for_selector(self, selector: str, timeout: Optional[int] = None) -> Any:
        """等待元素出现"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            return await self.page.wait_for_selector(
                selector,
                timeout=timeout or self.timeout
            )
        return None

    async def click(self, selector: str) -> None:
        """点击元素"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            await self.page.click(selector)

    async def fill(self, selector: str, value: str) -> None:
        """填充表单"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            await self.page.fill(selector, value)

    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """截图"""
        if self.backend == BrowserBackend.PLAYWRIGHT and self.page:
            return await self.page.screenshot(path=path)
        return b""

    async def close(self) -> None:
        """关闭浏览器"""
        if self.backend == BrowserBackend.PLAYWRIGHT:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()


def retry_async(policy: Optional[RetryPolicy] = None):
    """异步重试装饰器

    Args:
        policy: 重试策略，如果为None则使用默认策略
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except HTTPStatusError as e:
                    if not policy.should_retry(attempt, status_code=e.response.status_code):
                        raise
                    last_exception = e
                    delay = policy.get_delay(attempt)

                    # 检查 Retry-After 头
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass

                    await asyncio.sleep(delay)
                except Exception as e:
                    if not policy.should_retry(attempt, exception=e):
                        raise
                    last_exception = e
                    delay = policy.get_delay(attempt)
                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry failed without exception")

        return wrapper
    return decorator


class BaseScraper(ABC):
    """论文爬虫抽象基类

    定义所有论文爬虫必须实现的接口，并提供通用功能实现。

    使用示例:
        class OpenReviewScraper(BaseScraper):
            @property
            def source_name(self) -> str:
                return "openreview"

            @property
            def supported_conferences(self) -> List[str]:
                return ["ICLR", "NeurIPS", "ICML"]

            async def get_paper_list(self, conference, year, paper_type):
                # 实现具体逻辑
                pass
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
        browser_backend: BrowserBackend = BrowserBackend.NONE,
    ):
        """初始化爬虫

        Args:
            config: 配置对象，如果为None则使用默认配置
            rate_limiter: 速率限制器
            retry_policy: 重试策略
            browser_backend: 浏览器后端类型
        """
        self.config = config or get_settings()
        self.rate_limiter = rate_limiter or RateLimiter(
            requests_per_second=1.0 / self.config.download.request_delay,
            burst_size=10,
        )
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=self.config.download.max_retries,
            base_delay=self.config.download.retry_delay,
            exponential_base=self.config.download.retry_backoff,
        )
        self.browser_backend = browser_backend

        self.logger = get_logger(self.__class__.__name__)

        # HTTP客户端（懒加载）
        self._client: Optional[AsyncClient] = None
        self._browser_context: Optional[BrowserContext] = None

        # 爬取统计
        self._stats: Dict[str, int] = {
            "requests": 0,
            "success": 0,
            "failed": 0,
            "retries": 0,
        }

    # ==================== 抽象属性 ====================

    @property
    @abstractmethod
    def source_name(self) -> str:
        """数据源名称

        Returns:
            数据源标识符，如 "openreview", "acl", "arxiv"
        """
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """API基础URL

        Returns:
            API的基础URL
        """
        pass

    @property
    @abstractmethod
    def supported_conferences(self) -> List[str]:
        """支持的会议列表

        Returns:
            会议名称列表，如 ["ICLR", "NeurIPS", "ACL"]
        """
        pass

    # ==================== 抽象方法 ====================

    @abstractmethod
    async def get_paper_list(
        self,
        conference: str,
        year: int,
        paper_type: Optional[PaperType] = None,
        track: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Paper]:
        """获取论文列表

        Args:
            conference: 会议名称（如 "ICLR", "NeurIPS"）
            year: 年份
            paper_type: 论文类型筛选
            track: 论文track筛选
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            论文列表

        Raises:
            ScraperError: 爬取失败时抛出
        """
        pass

    @abstractmethod
    async def get_paper_details(self, paper_id: str) -> Paper:
        """获取论文详情

        Args:
            paper_id: 论文ID（数据源特定的ID）

        Returns:
            包含完整信息的论文对象

        Raises:
            NotFoundError: 论文不存在
            ScraperError: 爬取失败
        """
        pass

    @abstractmethod
    async def get_pdf_url(self, paper: Paper) -> str:
        """获取论文PDF下载链接

        Args:
            paper: 论文对象

        Returns:
            PDF直接下载URL

        Raises:
            NotFoundError: PDF不存在
            ScraperError: 获取失败
        """
        pass

    # ==================== 可选实现的方法 ====================

    async def search(
        self,
        query: str,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100,
    ) -> List[Paper]:
        """搜索论文

        默认实现为不支持搜索，子类可覆盖。

        Args:
            query: 搜索关键词
            conference: 限定会议
            year: 限定年份
            limit: 返回数量限制

        Returns:
            匹配的论文列表
        """
        raise NotImplementedError(f"{self.source_name} does not support search")

    async def get_conference_info(
        self,
        conference: str,
        year: int
    ) -> Dict[str, Any]:
        """获取会议信息

        Args:
            conference: 会议名称
            year: 年份

        Returns:
            会议信息字典
        """
        return {
            "conference": conference,
            "year": year,
            "source": self.source_name,
        }

    def validate_conference(self, conference: str) -> bool:
        """验证会议是否支持

        Args:
            conference: 会议名称

        Returns:
            是否支持该会议
        """
        return conference.upper() in [c.upper() for c in self.supported_conferences]

    # ==================== 通用爬取方法 ====================

    async def crawl(
        self,
        task: CrawlTask,
        progress_callback: Optional[Callable[[CrawlProgress], None]] = None,
    ) -> AsyncIterator[Paper]:
        """执行爬取任务

        通用的爬取流程实现，包括：
        1. 遍历所有目标会议和年份
        2. 获取论文列表
        3. 获取每篇论文的详情
        4. 获取PDF链接
        5. 处理错误和进度

        Args:
            task: 爬取任务
            progress_callback: 进度回调函数

        Yields:
            爬取到的论文对象
        """
        start_time = datetime.now()
        task.start()  # 使用 CrawlTask 的 start() 方法

        conferences_str = ", ".join(c.value for c in task.target_conferences)
        self.logger.info(
            f"Starting crawl task: {task.id} | "
            f"Conferences: {conferences_str} | "
            f"Years: {task.year_start}-{task.year_end}"
        )

        all_papers: List[Paper] = []

        try:
            # 遍历所有会议和年份组合
            for conference in task.target_conferences:
                # 验证会议是否被此爬虫支持
                if not self.validate_conference(conference.value):
                    self.logger.warning(
                        f"Conference '{conference.value}' is not supported by {self.source_name}, skipping"
                    )
                    continue

                for year in task.year_range:
                    self.logger.info(f"Crawling {conference.value} {year}")

                    # 更新进度
                    task.progress.current_conference = conference
                    task.progress.current_year = year

                    # 获取论文列表
                    self._report_progress_v2(
                        task, progress_callback,
                        action="fetching_list",
                        message=f"Fetching paper list for {conference.value} {year}"
                    )

                    try:
                        papers = await self.get_paper_list(
                            conference=conference.value,
                            year=year,
                            paper_type=task.paper_types[0] if task.paper_types else None,
                            limit=task.max_papers_per_venue,
                        )
                    except ScraperError as e:
                        self.logger.error(f"Failed to get paper list for {conference.value} {year}: {e}")
                        task.add_error(f"Failed to get paper list for {conference.value} {year}: {e}")
                        continue

                    task.progress.total_papers += len(papers)
                    self.logger.info(f"Found {len(papers)} papers for {conference.value} {year}")

                    # 应用关键词过滤
                    if task.include_keywords:
                        papers = self._filter_by_keywords(papers, task.include_keywords)
                        self.logger.info(f"After keyword filter: {len(papers)} papers")

                    # 处理每篇论文
                    for idx, paper in enumerate(papers):
                        task.progress.current_paper_id = paper.id

                        try:
                            self._report_progress_v2(
                                task, progress_callback,
                                action="fetching_details",
                                current_paper=paper.title[:50],
                                message=f"Processing {idx + 1}/{len(papers)}: {paper.title[:50]}..."
                            )

                            # 获取详情（如果需要）
                            if not paper.abstract:
                                try:
                                    paper = await self.get_paper_details(paper.id)
                                except NotFoundError:
                                    self.logger.warning(f"Paper not found: {paper.id}")
                                    task.progress.skipped_papers += 1
                                    continue

                            # 获取PDF链接（如果需要）
                            if task.download_pdf and not paper.pdf_url:
                                try:
                                    pdf_url = await self.get_pdf_url(paper)
                                    # Paper 是 Pydantic model，需要创建新实例或使用 model_copy
                                    paper = paper.model_copy(update={"pdf_url": pdf_url})
                                except (NotFoundError, ScraperError) as e:
                                    self.logger.warning(f"Failed to get PDF URL: {e}")

                            task.progress.crawled_papers += 1
                            all_papers.append(paper)

                            yield paper

                        except ScraperError as e:
                            self.logger.error(f"Failed to process paper {paper.id}: {e}")
                            task.progress.failed_papers += 1
                            task.add_error(f"Paper {paper.id}: {e}")

                        except Exception as e:
                            self.logger.exception(f"Unexpected error processing paper {paper.id}")
                            task.progress.failed_papers += 1
                            task.add_error(f"Paper {paper.id}: {type(e).__name__}: {e}")

            task.complete()

        except Exception as e:
            self.logger.exception(f"Crawl task failed: {e}")
            task.fail(str(e))
            raise

        finally:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Crawl task finished: {task.id} | "
                f"Status: {task.status.value} | "
                f"Crawled: {task.progress.crawled_papers} | "
                f"Failed: {task.progress.failed_papers} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    async def crawl_all(self, task: CrawlTask) -> CrawlResult:
        """执行爬取任务并收集所有结果

        Args:
            task: 爬取任务

        Returns:
            爬取结果
        """
        papers = []
        async for paper in self.crawl(task):
            papers.append(paper)

        return CrawlResult(task=task, papers=papers, errors=task.errors)

    # ==================== HTTP请求方法 ====================

    @property
    async def client(self) -> AsyncClient:
        """获取HTTP客户端（懒加载）"""
        if self._client is None or self._client.is_closed:
            self._client = AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.download.connect_timeout,
                    read=self.config.download.read_timeout,
                    write=30.0,
                    pool=30.0,
                ),
                headers=self._get_default_headers(),
                follow_redirects=True,
            )
        return self._client

    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认请求头"""
        headers = {
            "User-Agent": self.config.scraper.user_agent,
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # 添加API密钥（如果有）
        api_key = self.config.api_keys.__dict__.get(f"{self.source_name}_api_key")
        if api_key:
            # 不同API使用不同的头
            if self.source_name == "semantic_scholar":
                headers["x-api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"

        return headers

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Response:
        """发送HTTP请求（带速率限制和重试）

        Args:
            method: HTTP方法
            url: 请求URL
            params: 查询参数
            json: JSON请求体
            headers: 额外的请求头
            timeout: 超时时间

        Returns:
            响应对象

        Raises:
            NetworkError: 网络错误
            RateLimitError: 速率限制
            ScraperError: 其他错误
        """
        # 速率限制
        wait_time = await self.rate_limiter.acquire()
        if wait_time > 0:
            self.logger.debug(f"Rate limited, waited {wait_time:.2f}s")

        self._stats["requests"] += 1

        # 重试逻辑
        last_exception: Optional[Exception] = None

        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                client = await self.client
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                )

                # 检查状态码
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 60.0
                    raise RateLimitError(
                        f"Rate limited by {self.source_name}",
                        retry_after=wait,
                        source=self.source_name,
                    )

                response.raise_for_status()
                self._stats["success"] += 1
                return response

            except HTTPStatusError as e:
                if not self.retry_policy.should_retry(
                    attempt, status_code=e.response.status_code
                ):
                    self._stats["failed"] += 1
                    raise ScraperError(
                        f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                        source=self.source_name,
                        details={"status_code": e.response.status_code},
                    )
                last_exception = e
                self._stats["retries"] += 1

            except (TimeoutException, asyncio.TimeoutError) as e:
                if not self.retry_policy.should_retry(attempt, exception=e):
                    self._stats["failed"] += 1
                    raise NetworkError(
                        f"Request timeout: {url}",
                        source=self.source_name,
                    )
                last_exception = e
                self._stats["retries"] += 1

            except Exception as e:
                if not self.retry_policy.should_retry(attempt, exception=e):
                    self._stats["failed"] += 1
                    raise NetworkError(
                        f"Request failed: {e}",
                        source=self.source_name,
                    )
                last_exception = e
                self._stats["retries"] += 1

            # 等待后重试
            delay = self.retry_policy.get_delay(attempt)
            self.logger.warning(
                f"Request failed (attempt {attempt + 1}/{self.retry_policy.max_retries}), "
                f"retrying in {delay:.1f}s: {last_exception}"
            )
            await asyncio.sleep(delay)

        self._stats["failed"] += 1
        raise NetworkError(
            f"Request failed after {self.retry_policy.max_retries} retries",
            source=self.source_name,
        )

    async def get(self, url: str, **kwargs) -> Response:
        """GET请求"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Response:
        """POST请求"""
        return await self.request("POST", url, **kwargs)

    async def get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """GET请求并解析JSON响应"""
        response = await self.get(url, **kwargs)
        try:
            return response.json()
        except Exception as e:
            raise ParseError(
                f"Failed to parse JSON response: {e}",
                source=self.source_name,
                details={"response_text": response.text[:500]},
            )

    async def post_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """POST请求并解析JSON响应"""
        response = await self.post(url, **kwargs)
        try:
            return response.json()
        except Exception as e:
            raise ParseError(
                f"Failed to parse JSON response: {e}",
                source=self.source_name,
                details={"response_text": response.text[:500]},
            )

    # ==================== 浏览器自动化方法 ====================

    async def get_browser_context(self) -> BrowserContext:
        """获取浏览器上下文

        Returns:
            浏览器上下文对象
        """
        if self._browser_context is None:
            self._browser_context = await self._create_browser_context()
        return self._browser_context

    async def _create_browser_context(self) -> BrowserContext:
        """创建浏览器上下文"""
        if self.browser_backend == BrowserBackend.PLAYWRIGHT:
            return await self._create_playwright_context()
        elif self.browser_backend == BrowserBackend.CHROME_MCP:
            return BrowserContext(backend=BrowserBackend.CHROME_MCP)
        else:
            return BrowserContext(backend=BrowserBackend.NONE)

    async def _create_playwright_context(self) -> BrowserContext:
        """创建Playwright浏览器上下文"""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise BrowserError(
                "Playwright is not installed. Run: pip install playwright && playwright install",
                source=self.source_name,
            )

        playwright = await async_playwright().start()

        browser_config = self.config.browser if hasattr(self.config, 'browser') else None
        headless = browser_config.headless if browser_config else True

        browser = await playwright.chromium.launch(headless=headless)

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=self.config.scraper.user_agent,
        )

        page = await context.new_page()

        return BrowserContext(
            backend=BrowserBackend.PLAYWRIGHT,
            page=page,
            browser=browser,
            headless=headless,
        )

    async def fetch_with_browser(self, url: str) -> str:
        """使用浏览器获取页面内容

        用于处理需要JavaScript渲染的页面。

        Args:
            url: 页面URL

        Returns:
            页面HTML内容
        """
        if self.browser_backend == BrowserBackend.NONE:
            raise BrowserError(
                "Browser backend is not configured",
                source=self.source_name,
            )

        ctx = await self.get_browser_context()
        await ctx.navigate(url)
        return await ctx.get_content()

    # ==================== 辅助方法 ====================

    def _filter_by_keywords(
        self,
        papers: List[Paper],
        keywords: List[str]
    ) -> List[Paper]:
        """根据关键词过滤论文

        Args:
            papers: 论文列表
            keywords: 关键词列表

        Returns:
            匹配的论文列表
        """
        if not keywords:
            return papers

        keywords_lower = [k.lower() for k in keywords]
        filtered = []

        for paper in papers:
            text = f"{paper.title} {paper.abstract or ''}".lower()
            if any(kw in text for kw in keywords_lower):
                filtered.append(paper)

        return filtered

    def _report_progress_v2(
        self,
        task: CrawlTask,
        callback: Optional[Callable[[CrawlProgress], None]],
        action: str,
        current_paper: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """报告进度（适配新版 CrawlTask 模型）

        Args:
            task: 爬取任务
            callback: 回调函数
            action: 当前操作
            current_paper: 当前处理的论文
            message: 进度消息
        """
        if callback is None:
            return

        # 更新 task.progress 的时间戳
        task.progress.last_update = datetime.now()

        try:
            callback(task.progress)
        except Exception as e:
            self.logger.warning(f"Progress callback failed: {e}")

    def _report_progress(
        self,
        task: CrawlTask,
        callback: Optional[Callable[[CrawlProgress], None]],
        action: str,
        current_paper: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """报告进度（兼容旧接口）

        Args:
            task: 爬取任务
            callback: 回调函数
            action: 当前操作
            current_paper: 当前处理的论文
            message: 进度消息
        """
        # 委托给新方法
        self._report_progress_v2(task, callback, action, current_paper, message)

    def get_stats(self) -> Dict[str, int]:
        """获取爬取统计信息"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "requests": 0,
            "success": 0,
            "failed": 0,
            "retries": 0,
        }

    # ==================== 上下文管理 ====================

    async def __aenter__(self) -> "BaseScraper":
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出"""
        await self.close()

    async def close(self) -> None:
        """关闭资源"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

        if self._browser_context:
            await self._browser_context.close()
            self._browser_context = None

        self.logger.debug(f"Scraper closed. Stats: {self._stats}")


# ==================== 同步包装器 ====================

class SyncScraperWrapper:
    """同步爬虫包装器

    为异步爬虫提供同步接口，便于在同步代码中使用。
    """

    def __init__(self, scraper: BaseScraper):
        self._scraper = scraper
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """获取或创建事件循环"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        """运行协程"""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def get_paper_list(self, *args, **kwargs) -> List[Paper]:
        return self._run(self._scraper.get_paper_list(*args, **kwargs))

    def get_paper_details(self, paper_id: str) -> Paper:
        return self._run(self._scraper.get_paper_details(paper_id))

    def get_pdf_url(self, paper: Paper) -> str:
        return self._run(self._scraper.get_pdf_url(paper))

    def search(self, *args, **kwargs) -> List[Paper]:
        return self._run(self._scraper.search(*args, **kwargs))

    def crawl_all(self, task: CrawlTask) -> CrawlResult:
        return self._run(self._scraper.crawl_all(task))

    def close(self) -> None:
        self._run(self._scraper.close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
