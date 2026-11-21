"""
统一引用数获取模块

支持从多个来源获取论文引用数:
1. OpenAlex API (默认，速度快，无反爬)
2. Google Scholar (更准确，但有反爬限制)
3. 混合模式 (两者都尝试，取较大值)

使用方法:
    from citation_fetcher import CitationFetcher

    fetcher = CitationFetcher(source="openalex")
    count = fetcher.get_citation_count("Attention is All You Need")
"""

import json
import time
import re
import random
import requests
from typing import Optional, Dict, List, Literal
from dataclasses import dataclass, field
from pathlib import Path

# 尝试导入 Google Scholar 爬虫
try:
    from pipeline.paper_crawler.scrapers import GoogleScholarScraper, GoogleScholarConfig
    GOOGLE_SCHOLAR_AVAILABLE = True
except ImportError:
    GOOGLE_SCHOLAR_AVAILABLE = False


@dataclass
class CitationFetcherConfig:
    """引用数获取器配置"""

    # 来源选择: openalex, google_scholar, both
    source: Literal["openalex", "google_scholar", "both"] = "openalex"

    # OpenAlex 配置
    openalex_base_url: str = "https://api.openalex.org/works"
    openalex_delay: float = 0.3
    openalex_email: Optional[str] = None  # 用于 polite pool

    # Google Scholar 配置
    gs_min_delay: float = 3.0
    gs_max_delay: float = 8.0
    gs_proxy: Optional[str] = None

    # 通用配置
    max_retries: int = 3
    similarity_threshold: float = 0.5


class CitationFetcher:
    """统一引用数获取器

    示例:
        >>> fetcher = CitationFetcher(source="openalex")
        >>> count = fetcher.get_citation_count("Attention is All You Need")
        >>> print(count)

        >>> # 批量获取
        >>> results = fetcher.get_citation_count_batch(["Title 1", "Title 2"])
    """

    def __init__(self, config: Optional[CitationFetcherConfig] = None, **kwargs):
        """初始化

        Args:
            config: 配置对象
            **kwargs: 可直接传入 CitationFetcherConfig 参数
        """
        if config is None:
            config = CitationFetcherConfig(**kwargs)
        self.config = config

        # 创建 requests session
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "AcademicResearchTool/1.0 (mailto:research@example.com)",
            "Accept": "application/json",
        })

        # Google Scholar 爬虫 (懒加载)
        self._gs_scraper = None

        # 统计
        self._stats = {
            "openalex_success": 0,
            "openalex_failed": 0,
            "google_scholar_success": 0,
            "google_scholar_failed": 0,
        }

    @property
    def gs_scraper(self) -> Optional["GoogleScholarScraper"]:
        """懒加载 Google Scholar 爬虫"""
        if self._gs_scraper is None and GOOGLE_SCHOLAR_AVAILABLE:
            gs_config = GoogleScholarConfig(
                min_delay=self.config.gs_min_delay,
                max_delay=self.config.gs_max_delay,
                proxy=self.config.gs_proxy,
                similarity_threshold=self.config.similarity_threshold,
            )
            self._gs_scraper = GoogleScholarScraper(config=gs_config)
        return self._gs_scraper

    @staticmethod
    def normalize_title(title: str) -> str:
        """规范化标题用于比较"""
        t = title.lower()
        t = re.sub(r'[^a-z0-9\s]', '', t)
        t = ' '.join(t.split())
        return t

    @staticmethod
    def title_similarity(title1: str, title2: str) -> float:
        """计算标题相似度"""
        words1 = set(CitationFetcher.normalize_title(title1).split())
        words2 = set(CitationFetcher.normalize_title(title2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _get_from_openalex(self, title: str) -> Optional[int]:
        """从 OpenAlex 获取引用数"""
        for attempt in range(self.config.max_retries):
            try:
                # 清理标题
                clean_title = title
                for char in ['$', '\\', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', ',', '.', '+', '*', '&', '%', '#', '@', '/', '-']:
                    clean_title = clean_title.replace(char, ' ')
                clean_title = ' '.join(clean_title.split())[:150]

                params = {
                    "search": clean_title,
                    "per-page": 3,
                    "select": "title,cited_by_count",
                }

                response = self._session.get(
                    self.config.openalex_base_url,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if data.get("results"):
                    best_match = None
                    best_similarity = 0.0

                    for work in data["results"]:
                        work_title = work.get("title", "")
                        similarity = self.title_similarity(title, work_title)

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = work

                    if best_match and best_similarity >= self.config.similarity_threshold:
                        self._stats["openalex_success"] += 1
                        return best_match.get("cited_by_count", 0)

                self._stats["openalex_success"] += 1
                return 0

            except requests.exceptions.Timeout:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2)
                else:
                    self._stats["openalex_failed"] += 1
                    return None
            except requests.exceptions.RequestException:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2)
                else:
                    self._stats["openalex_failed"] += 1
                    return None

        self._stats["openalex_failed"] += 1
        return None

    def _get_from_google_scholar(self, title: str) -> Optional[int]:
        """从 Google Scholar 获取引用数"""
        if not GOOGLE_SCHOLAR_AVAILABLE:
            print("警告: Google Scholar 爬虫不可用")
            return None

        if self.gs_scraper is None:
            return None

        try:
            count = self.gs_scraper.get_citation_count(title)
            if count is not None:
                self._stats["google_scholar_success"] += 1
            else:
                self._stats["google_scholar_failed"] += 1
            return count
        except Exception as e:
            print(f"Google Scholar 错误: {e}")
            self._stats["google_scholar_failed"] += 1
            return None

    def get_citation_count(self, title: str) -> int:
        """获取论文引用数

        根据配置的 source 从不同来源获取引用数。

        Args:
            title: 论文标题

        Returns:
            引用数，如果失败返回 0
        """
        if self.config.source == "openalex":
            result = self._get_from_openalex(title)
            time.sleep(self.config.openalex_delay + random.uniform(0, 0.1))
            return result if result is not None else 0

        elif self.config.source == "google_scholar":
            result = self._get_from_google_scholar(title)
            return result if result is not None else 0

        elif self.config.source == "both":
            # 先尝试 OpenAlex (速度快)
            oa_count = self._get_from_openalex(title)
            time.sleep(self.config.openalex_delay)

            # 再尝试 Google Scholar
            gs_count = self._get_from_google_scholar(title)

            # 取较大值
            if oa_count is None and gs_count is None:
                return 0
            elif oa_count is None:
                return gs_count
            elif gs_count is None:
                return oa_count
            else:
                return max(oa_count, gs_count)

        return 0

    def get_citation_count_batch(
        self,
        titles: List[str],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, int]:
        """批量获取引用数

        Args:
            titles: 论文标题列表
            progress_callback: 进度回调 (current, total, title, count)

        Returns:
            {标题: 引用数} 字典
        """
        results = {}
        total = len(titles)

        for i, title in enumerate(titles):
            count = self.get_citation_count(title)
            results[title] = count

            if progress_callback:
                progress_callback(i + 1, total, title, count)

        return results

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return dict(self._stats)

    def close(self):
        """关闭资源"""
        if self._session:
            self._session.close()
        if self._gs_scraper:
            self._gs_scraper.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
def get_citation_count(
    title: str,
    source: Literal["openalex", "google_scholar", "both"] = "openalex",
) -> int:
    """获取论文引用数

    Args:
        title: 论文标题
        source: 来源选择

    Returns:
        引用数
    """
    with CitationFetcher(source=source) as fetcher:
        return fetcher.get_citation_count(title)


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("统一引用数获取器示例")
    print("=" * 60)

    titles = [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
    ]

    # 使用 OpenAlex
    print("\n[OpenAlex]")
    with CitationFetcher(source="openalex") as fetcher:
        for title in titles:
            count = fetcher.get_citation_count(title)
            print(f"  {title[:50]}...: {count}")
        print(f"  统计: {fetcher.get_stats()}")

    # 如果 Google Scholar 可用，也测试一下
    if GOOGLE_SCHOLAR_AVAILABLE:
        print("\n[Google Scholar]")
        with CitationFetcher(source="google_scholar") as fetcher:
            count = fetcher.get_citation_count(titles[0])
            print(f"  {titles[0][:50]}...: {count}")
            print(f"  统计: {fetcher.get_stats()}")

    print("\n完成!")
