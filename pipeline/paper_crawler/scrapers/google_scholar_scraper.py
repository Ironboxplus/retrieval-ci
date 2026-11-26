"""
GoogleScholarScraper - Google Scholar 引用数爬虫

支持多种获取方式:
1. requests + BeautifulSoup (主要方式)
2. Chrome MCP (反爬备用方式)
3. scholarly 库 (备选方式)

功能:
- 根据论文标题获取引用数
- 批量获取引用数
- 标题相似度验证
- 频率限制和重试机制
- 代理配置支持

使用注意:
- 遵守爬虫礼仪，添加合理延迟
- 使用 User-Agent 模拟浏览器
- 处理 CAPTCHA 情况（记录日志，跳过或等待）
"""

import re
import time
import random
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote_plus, urljoin
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .base_scraper import (
    BaseScraper,
    ScraperError,
    RateLimitError,
    NetworkError,
    ParseError,
    BrowserBackend,
    RateLimiter,
    RetryPolicy,
)
from ..models.paper import Paper, PaperType
from ..config import Settings, get_settings
from ..utils.logger import get_logger


class FetchMethod(Enum):
    """获取方式枚举"""
    REQUESTS = "requests"  # requests + BeautifulSoup
    CHROME_MCP = "chrome_mcp"  # Chrome DevTools MCP
    SCHOLARLY = "scholarly"  # scholarly 库


@dataclass
class GoogleScholarConfig:
    """Google Scholar 配置"""
    # 请求延迟范围 (秒)
    min_delay: float = 3.0
    max_delay: float = 8.0

    # 批量请求间隔 (秒)
    batch_delay: float = 10.0

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 5.0

    # 标题相似度阈值
    similarity_threshold: float = 0.5

    # 代理配置
    proxy: Optional[str] = None

    # User-Agent 列表 (轮换使用)
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ])

    # 优先使用的获取方式
    preferred_method: FetchMethod = FetchMethod.REQUESTS

    # 是否启用 CAPTCHA 检测
    captcha_detection: bool = True

    # CAPTCHA 等待时间 (秒)
    captcha_wait_time: float = 60.0


class CaptchaError(ScraperError):
    """CAPTCHA 异常"""
    pass


class GoogleScholarScraper:
    """Google Scholar 引用数爬虫

    支持多种获取方式，自动处理反爬和错误情况。

    示例:
        >>> scraper = GoogleScholarScraper()
        >>> count = scraper.get_citation_count("Attention is All You Need")
        >>> print(f"引用数: {count}")

        >>> # 批量获取
        >>> titles = ["BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3"]
        >>> results = scraper.get_citation_count_batch(titles)
    """

    GOOGLE_SCHOLAR_URL = "https://scholar.google.com"
    SEARCH_URL = "https://scholar.google.com/scholar"

    def __init__(
        self,
        config: Optional[GoogleScholarConfig] = None,
        settings: Optional[Settings] = None,
    ):
        """初始化 Google Scholar 爬虫

        Args:
            config: Google Scholar 配置
            settings: 全局设置
        """
        self.config = config or GoogleScholarConfig()
        self.settings = settings or get_settings()
        self.logger = get_logger(self.__class__.__name__)

        # 创建 requests session
        self._session: Optional[requests.Session] = None

        # 统计信息
        self._stats = {
            "requests": 0,
            "success": 0,
            "failed": 0,
            "captcha": 0,
            "retries": 0,
        }

        # 上次请求时间
        self._last_request_time: float = 0

        # scholarly 库 (懒加载)
        self._scholarly = None

    @property
    def session(self) -> requests.Session:
        """获取 requests session (懒加载)"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            })

            # 配置代理
            if self.config.proxy:
                self._session.proxies = {
                    "http": self.config.proxy,
                    "https": self.config.proxy,
                }

        return self._session

    def _get_random_user_agent(self) -> str:
        """获取随机 User-Agent"""
        return random.choice(self.config.user_agents)

    def _get_random_delay(self) -> float:
        """获取随机延迟时间"""
        return random.uniform(self.config.min_delay, self.config.max_delay)

    def _wait_for_rate_limit(self) -> None:
        """等待以遵守频率限制"""
        if self._last_request_time > 0:
            elapsed = time.time() - self._last_request_time
            delay = self._get_random_delay()
            if elapsed < delay:
                wait_time = delay - elapsed
                self.logger.debug(f"等待 {wait_time:.2f}s 以遵守频率限制")
                time.sleep(wait_time)
        self._last_request_time = time.time()

    @staticmethod
    def normalize_title(title: str) -> str:
        """规范化标题用于比较

        Args:
            title: 原始标题

        Returns:
            规范化后的标题
        """
        # 转小写
        t = title.lower()
        # 移除特殊字符
        t = re.sub(r'[^a-z0-9\s]', '', t)
        # 规范化空格
        t = ' '.join(t.split())
        return t

    @staticmethod
    def title_similarity(title1: str, title2: str) -> float:
        """计算标题相似度（词重叠比例 / Jaccard系数）

        Args:
            title1: 第一个标题
            title2: 第二个标题

        Returns:
            相似度分数 (0.0 - 1.0)
        """
        words1 = set(GoogleScholarScraper.normalize_title(title1).split())
        words2 = set(GoogleScholarScraper.normalize_title(title2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _detect_captcha(self, html_content: str) -> bool:
        """检测页面是否包含 CAPTCHA

        Args:
            html_content: HTML 内容

        Returns:
            是否检测到 CAPTCHA
        """
        if not self.config.captcha_detection:
            return False

        captcha_indicators = [
            "captcha",
            "unusual traffic",
            "automated requests",
            "robot",
            "verify you're a human",
            "recaptcha",
            "not a robot",
            "please show you're not a robot",
        ]

        html_lower = html_content.lower()
        return any(indicator in html_lower for indicator in captcha_indicators)

    def _parse_citation_count(self, html_content: str, target_title: str) -> Tuple[Optional[int], Optional[str]]:
        """从 HTML 解析引用数

        Args:
            html_content: HTML 内容
            target_title: 目标论文标题

        Returns:
            (引用数, 匹配到的标题) 或 (None, None)
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # 查找搜索结果
        results = soup.select('.gs_r.gs_or.gs_scl')

        if not results:
            # 尝试其他选择器
            results = soup.select('.gs_ri')

        best_match = None
        best_similarity = 0.0
        best_citation = None

        for result in results:
            # 提取标题
            title_elem = result.select_one('.gs_rt a, .gs_rt')
            if not title_elem:
                continue

            found_title = title_elem.get_text(strip=True)

            # 计算相似度
            similarity = self.title_similarity(target_title, found_title)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = found_title

                # 提取引用数
                # 格式: "被引用次数：{number}" 或 "Cited by {number}"
                cite_elem = result.select_one('a[href*="cites="]')
                if cite_elem:
                    cite_text = cite_elem.get_text(strip=True)
                    # 提取数字
                    numbers = re.findall(r'\d+', cite_text)
                    if numbers:
                        best_citation = int(numbers[0])
                    else:
                        best_citation = 0
                else:
                    # 没有引用链接，可能是0引用
                    best_citation = 0

        # 检查相似度是否足够
        if best_similarity >= self.config.similarity_threshold:
            self.logger.debug(f"匹配成功: '{target_title}' -> '{best_match}' (相似度: {best_similarity:.2f})")
            return best_citation, best_match
        else:
            if best_match:
                self.logger.debug(f"相似度不足: '{target_title}' -> '{best_match}' (相似度: {best_similarity:.2f})")
            else:
                self.logger.debug(f"未找到匹配结果: '{target_title}'")
            return None, None

    def _fetch_with_requests(self, title: str) -> Optional[int]:
        """使用 requests + BeautifulSoup 获取引用数

        Args:
            title: 论文标题

        Returns:
            引用数，如果失败返回 None
        """
        self._wait_for_rate_limit()

        # 更新 User-Agent
        self.session.headers["User-Agent"] = self._get_random_user_agent()

        # 构建搜索 URL
        params = {
            "q": title,
            "hl": "en",  # 使用英语界面
        }

        self._stats["requests"] += 1

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    self.SEARCH_URL,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    # 速率限制
                    self._stats["retries"] += 1
                    wait_time = self.config.retry_delay * (attempt + 1)
                    self.logger.warning(f"[限流] 等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                html_content = response.text

                # 检测 CAPTCHA
                if self._detect_captcha(html_content):
                    self._stats["captcha"] += 1
                    self.logger.warning(f"[CAPTCHA] 检测到 CAPTCHA，标题: {title[:50]}...")
                    raise CaptchaError("Google Scholar CAPTCHA detected", source="google_scholar")

                # 解析引用数
                citation_count, matched_title = self._parse_citation_count(html_content, title)

                if citation_count is not None:
                    self._stats["success"] += 1
                    return citation_count
                else:
                    # 未找到匹配
                    self._stats["success"] += 1  # 请求成功，只是没找到
                    return 0

            except requests.exceptions.Timeout:
                self._stats["retries"] += 1
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"[超时] 重试 {attempt + 2}/{self.config.max_retries}...")
                    time.sleep(self.config.retry_delay)
                else:
                    self._stats["failed"] += 1
                    self.logger.error(f"[失败] 请求超时: {title[:50]}...")
                    return None

            except requests.exceptions.RequestException as e:
                self._stats["retries"] += 1
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(f"[错误] {e}, 重试 {attempt + 2}/{self.config.max_retries}...")
                    time.sleep(self.config.retry_delay)
                else:
                    self._stats["failed"] += 1
                    self.logger.error(f"[失败] 请求错误: {e}")
                    return None

            except CaptchaError:
                # CAPTCHA 错误，不重试
                self._stats["failed"] += 1
                return None

        self._stats["failed"] += 1
        return None

    def _fetch_with_scholarly(self, title: str) -> Optional[int]:
        """使用 scholarly 库获取引用数

        Args:
            title: 论文标题

        Returns:
            引用数，如果失败返回 None
        """
        # 懒加载 scholarly
        if self._scholarly is None:
            try:
                import scholarly
                self._scholarly = scholarly
                self.logger.info("scholarly 库加载成功")
            except ImportError:
                self.logger.error("scholarly 库未安装，请运行: pip install scholarly")
                return None

        self._wait_for_rate_limit()
        self._stats["requests"] += 1

        try:
            # 搜索论文
            search_query = self._scholarly.search_pubs(title)

            # 获取第一个结果
            try:
                pub = next(search_query)
            except StopIteration:
                self.logger.debug(f"scholarly 未找到结果: {title[:50]}...")
                self._stats["success"] += 1
                return 0

            # 验证标题相似度
            found_title = pub.get("bib", {}).get("title", "")
            similarity = self.title_similarity(title, found_title)

            if similarity >= self.config.similarity_threshold:
                citation_count = pub.get("num_citations", 0)
                self._stats["success"] += 1
                self.logger.debug(f"scholarly 匹配成功: {title[:50]}... -> {citation_count} 引用")
                return citation_count
            else:
                self.logger.debug(f"scholarly 相似度不足: {similarity:.2f}")
                self._stats["success"] += 1
                return 0

        except Exception as e:
            self._stats["failed"] += 1
            self.logger.error(f"scholarly 错误: {e}")
            return None

    def get_citation_count(
        self,
        title: str,
        method: Optional[FetchMethod] = None,
    ) -> int:
        """根据标题获取引用数

        Args:
            title: 论文标题
            method: 获取方式，默认使用配置的优先方式

        Returns:
            引用数，如果失败返回 0
        """
        method = method or self.config.preferred_method

        self.logger.debug(f"获取引用数: {title[:50]}... (方式: {method.value})")

        if method == FetchMethod.REQUESTS:
            result = self._fetch_with_requests(title)
        elif method == FetchMethod.SCHOLARLY:
            result = self._fetch_with_scholarly(title)
        elif method == FetchMethod.CHROME_MCP:
            # Chrome MCP 需要外部调用，这里返回提示
            self.logger.warning("Chrome MCP 方式需要通过 get_citation_count_with_chrome_mcp() 使用")
            result = None
        else:
            result = None

        if result is None:
            # 主要方式失败，尝试备用方式
            if method == FetchMethod.REQUESTS:
                self.logger.info("requests 方式失败，尝试 scholarly...")
                result = self._fetch_with_scholarly(title)

        return result if result is not None else 0

    def get_citation_count_batch(
        self,
        titles: List[str],
        method: Optional[FetchMethod] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, int]:
        """批量获取引用数

        Args:
            titles: 论文标题列表
            method: 获取方式
            progress_callback: 进度回调函数 (current, total, title, count)

        Returns:
            {标题: 引用数} 字典
        """
        results = {}
        total = len(titles)

        self.logger.info(f"开始批量获取 {total} 篇论文的引用数")

        for i, title in enumerate(titles):
            count = self.get_citation_count(title, method)
            results[title] = count

            if progress_callback:
                progress_callback(i + 1, total, title, count)
            else:
                short_title = title[:50] + "..." if len(title) > 50 else title
                self.logger.info(f"[{i + 1}/{total}] {short_title} -> {count} 引用")

            # 批量请求额外延迟
            if i < total - 1:
                time.sleep(self.config.batch_delay)

        self.logger.info(f"批量获取完成，成功: {sum(1 for v in results.values() if v > 0)}/{total}")
        return results

    def get_citation_count_with_chrome_mcp(
        self,
        title: str,
        chrome_tools: Dict[str, callable],
    ) -> Optional[int]:
        """使用 Chrome MCP 获取引用数

        这个方法需要传入 Chrome DevTools MCP 工具函数。
        适用于反爬严重时的备用方式。

        Args:
            title: 论文标题
            chrome_tools: Chrome MCP 工具字典，应包含:
                - navigate_page: 导航页面
                - take_snapshot: 获取页面快照
                - wait_for: 等待元素

        Returns:
            引用数，如果失败返回 None

        使用示例 (在 Claude Code 中):
            # 这个方法设计用于与 Chrome MCP 工具配合使用
            # 在实际使用中，需要手动调用 Chrome MCP 工具
        """
        self.logger.info(f"Chrome MCP 方式获取: {title[:50]}...")

        # 构建搜索 URL
        search_url = f"{self.SEARCH_URL}?q={quote_plus(title)}&hl=en"

        try:
            # 导航到搜索页面
            navigate = chrome_tools.get("navigate_page")
            if navigate:
                navigate(url=search_url)

            # 等待搜索结果加载
            wait_for = chrome_tools.get("wait_for")
            if wait_for:
                wait_for(text="results", timeout=10000)

            # 获取页面快照
            take_snapshot = chrome_tools.get("take_snapshot")
            if take_snapshot:
                snapshot = take_snapshot()

                # 在快照中查找引用信息
                # 格式: "Cited by {number}" 或 "被引用次数：{number}"
                if snapshot:
                    # 解析快照文本查找引用链接
                    cite_pattern = r'Cited by (\d+)'
                    cite_pattern_zh = r'被引用次数：(\d+)'

                    matches = re.findall(cite_pattern, str(snapshot))
                    if not matches:
                        matches = re.findall(cite_pattern_zh, str(snapshot))

                    if matches:
                        # 返回第一个匹配的引用数
                        self._stats["success"] += 1
                        return int(matches[0])

            self._stats["success"] += 1
            return 0

        except Exception as e:
            self._stats["failed"] += 1
            self.logger.error(f"Chrome MCP 错误: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "requests": 0,
            "success": 0,
            "failed": 0,
            "captcha": 0,
            "retries": 0,
        }

    def close(self) -> None:
        """关闭资源"""
        if self._session:
            self._session.close()
            self._session = None
        self.logger.debug(f"GoogleScholarScraper 关闭. 统计: {self._stats}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
def get_citation_count(title: str, **kwargs) -> int:
    """获取单篇论文的 Google Scholar 引用数

    Args:
        title: 论文标题
        **kwargs: GoogleScholarConfig 参数

    Returns:
        引用数
    """
    config = GoogleScholarConfig(**kwargs) if kwargs else None
    with GoogleScholarScraper(config=config) as scraper:
        return scraper.get_citation_count(title)


def get_citation_count_batch(titles: List[str], **kwargs) -> Dict[str, int]:
    """批量获取 Google Scholar 引用数

    Args:
        titles: 论文标题列表
        **kwargs: GoogleScholarConfig 参数

    Returns:
        {标题: 引用数} 字典
    """
    config = GoogleScholarConfig(**kwargs) if kwargs else None
    with GoogleScholarScraper(config=config) as scraper:
        return scraper.get_citation_count_batch(titles)


if __name__ == "__main__":
    # 示例用法
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建爬虫
    scraper = GoogleScholarScraper()

    # 测试单个标题
    test_titles = [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Language Models are Few-Shot Learners",
    ]

    print("=" * 60)
    print("Google Scholar 引用数获取测试")
    print("=" * 60)

    for title in test_titles:
        print(f"\n标题: {title[:60]}...")
        count = scraper.get_citation_count(title)
        print(f"引用数: {count}")
        print("-" * 60)

    # 打印统计
    print("\n统计信息:")
    for key, value in scraper.get_stats().items():
        print(f"  {key}: {value}")

    scraper.close()
