"""
浏览器爬虫 - 使用 browser-use 进行网页内容爬取
"""
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import json
import os
from datetime import datetime

from browser_use import Browser
from playwright.async_api import async_playwright, Page

from config import settings
from app.models.schemas import CrawlStatus


@dataclass
class CrawlResult:
    """爬取结果"""
    url: str
    title: str
    content: str
    html: str
    status: str
    error: Optional[str] = None
    crawled_at: datetime = None

    def __post_init__(self):
        if self.crawled_at is None:
            self.crawled_at = datetime.now()


class BrowserCrawler:
    """使用无头浏览器爬取网页内容"""

    def __init__(self, use_proxy: bool = False):
        self.use_proxy = use_proxy
        self.results: List[CrawlResult] = []
        self.status_callback: Optional[Callable] = None

    async def crawl_urls(self, urls: List[str],
                         max_concurrent: int = None) -> List[CrawlResult]:
        """
        批量爬取URL

        Args:
            urls: URL列表
            max_concurrent: 最大并发数

        Returns:
            爬取结果列表
        """
        if max_concurrent is None:
            max_concurrent = settings.crawler.max_concurrent

        self.results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_single(url)

        tasks = [crawl_with_semaphore(url) for url in urls]
        self.results = await asyncio.gather(*tasks)

        return self.results

    async def crawl_single(self, url: str) -> CrawlResult:
        """
        爬取单个URL

        Args:
            url: 目标URL

        Returns:
            爬取结果
        """
        async with async_playwright() as p:
            # 配置浏览器选项
            launch_options = {
                "headless": settings.crawler.headless,
            }

            # 代理配置（强制走代理，无直连回退）
            if self.use_proxy:
                launch_options["proxy"] = settings.proxy.get_browser_proxy()

            browser = await p.chromium.launch(**launch_options)

            try:
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()

                # 设置超时
                page.set_default_timeout(settings.crawler.timeout)

                # 导航到页面
                await page.goto(url, wait_until="networkidle")

                # 等待内容加载
                await asyncio.sleep(settings.crawler.delay_between_requests)

                # 提取内容
                title = await page.title()
                html = await page.content()
                content = await self._extract_content(page)

                result = CrawlResult(
                    url=url,
                    title=title,
                    content=content,
                    html=html,
                    status="success"
                )

                # 保存到文件
                await self._save_result(result)

                return result

            except Exception as e:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    html="",
                    status="failed",
                    error=str(e)
                )
            finally:
                await browser.close()

    async def _extract_content(self, page: Page) -> str:
        """
        从页面提取主要文本内容

        Args:
            page: Playwright页面对象

        Returns:
            提取的文本内容
        """
        # 使用JavaScript提取主要内容
        content = await page.evaluate("""
            () => {
                // 移除脚本和样式
                const elementsToRemove = document.querySelectorAll('script, style, nav, footer, header, aside, .ad, .advertisement, .sidebar');
                elementsToRemove.forEach(el => el.remove());

                // 尝试找到主内容区域
                const mainContent = document.querySelector('main, article, .content, .post, .entry, #content, #main')
                    || document.body;

                // 提取文本
                const text = mainContent.innerText || mainContent.textContent;

                // 清理空白
                return text.replace(/\\s+/g, ' ').trim();
            }
        """)

        return content

    async def _save_result(self, result: CrawlResult):
        """保存爬取结果到文件"""
        # 生成安全的文件名
        safe_name = "".join(c if c.isalnum() else "_" for c in result.url)[:100]
        timestamp = result.crawled_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_name}.json"

        filepath = os.path.join(settings.crawled_dir, filename)

        data = {
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "status": result.status,
            "error": result.error,
            "crawled_at": result.crawled_at.isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_status(self) -> List[CrawlStatus]:
        """获取爬取状态"""
        return [
            CrawlStatus(
                url=r.url,
                status=r.status,
                content_length=len(r.content),
                error=r.error
            )
            for r in self.results
        ]
