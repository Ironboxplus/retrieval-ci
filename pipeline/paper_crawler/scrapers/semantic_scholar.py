"""
SemanticScholarScraper - Semantic Scholar API 爬虫

通过 Semantic Scholar API 获取论文元数据、引用数、PDF链接等。

API 文档: https://api.semanticscholar.org/
- 免费版: 100 requests / 5 minutes
- 有 API Key: 更高限制

主要端点:
- 论文搜索: GET /graph/v1/paper/search
- 论文详情: GET /graph/v1/paper/{paper_id}
- 批量获取: POST /graph/v1/paper/batch
"""

import asyncio
import time
import os
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

from pipeline.paper_crawler.scrapers.base_scraper import BaseScraper
from pipeline.paper_crawler.models.paper import Paper, Author, PaperMetadata, PaperSource
from pipeline.paper_crawler.utils.logger import get_logger


@dataclass
class RateLimiter:
    """速率限制器

    Semantic Scholar 免费版限制: 100 requests / 5 minutes
    """
    max_requests: int = 100
    time_window: float = 300.0  # 5 minutes in seconds
    _request_times: List[float] = field(default_factory=list)

    async def acquire(self) -> None:
        """获取请求许可，如需要则等待"""
        now = time.time()

        # 清理过期的请求记录
        self._request_times = [
            t for t in self._request_times
            if now - t < self.time_window
        ]

        # 如果达到限制，等待直到最早的请求过期
        if len(self._request_times) >= self.max_requests:
            wait_time = self._request_times[0] + self.time_window - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._request_times = self._request_times[1:]

        self._request_times.append(time.time())


class SemanticScholarScraper(BaseScraper):
    """Semantic Scholar API 爬虫

    功能:
    - 搜索论文（支持按会议/venue搜索）
    - 获取论文详情（标题、摘要、作者、引用数）
    - 获取 PDF 链接
    - 批量获取论文
    - 自动处理速率限制（100 requests / 5 minutes 免费版）

    使用示例（同步）:
        with SemanticScholarScraper() as scraper:
            # 搜索论文
            papers = scraper.search("machine learning", limit=10)

            # 获取论文详情
            paper = scraper.get_paper("649def34f8be52c8b66281af98ae884c09aef38b")

    使用示例（异步）:
        async with SemanticScholarScraper() as scraper:
            # 搜索论文
            papers = await scraper.search_async("machine learning", limit=10)

            # 按会议搜索
            papers = await scraper.search_by_venue("NeurIPS", year=2023, limit=50)

            # 获取论文详情
            paper = await scraper.get_paper_async("649def34f8be52c8b66281af98ae884c09aef38b")

            # 批量获取
            papers = await scraper.get_papers_batch(["paper_id_1", "paper_id_2"])
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    # 论文详情需要的字段
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "authors",
        "authors.name",
        "authors.affiliations",
        "venue",
        "year",
        "citationCount",
        "referenceCount",
        "openAccessPdf",
        "url",
        "externalIds",
        "fieldsOfStudy",
        "publicationTypes",
    ]

    # 搜索用的字段（部分字段在搜索时不可用）
    SEARCH_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "authors",
        "venue",
        "year",
        "citationCount",
        "openAccessPdf",
        "externalIds",
        "fieldsOfStudy",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        max_requests_per_window: int = 95,  # 留一点余量
        time_window: float = 300.0,
    ):
        """初始化 Semantic Scholar 爬虫

        Args:
            api_key: API Key（可选，有则提高速率限制）
            request_delay: 请求间隔（秒）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            max_requests_per_window: 时间窗口内最大请求数
            time_window: 速率限制时间窗口（秒）
        """
        super().__init__(
            request_delay=request_delay,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.rate_limiter = RateLimiter(
            max_requests=max_requests_per_window,
            time_window=time_window,
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = get_logger(self.__class__.__name__)

    @property
    def headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    # ==================== 上下文管理器 ====================

    def __enter__(self) -> "SemanticScholarScraper":
        """同步上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """同步上下文管理器出口"""
        self.close()

    async def __aenter__(self) -> "SemanticScholarScraper":
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close_async()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """确保 aiohttp session 存在"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout,
            )
        return self._session

    async def close_async(self) -> None:
        """关闭异步 session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ==================== 异步请求 ====================

    async def _request_async(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """异步请求，带重试和速率限制

        Args:
            method: HTTP 方法
            url: 请求 URL
            **kwargs: 传递给 aiohttp 的参数

        Returns:
            JSON 响应或 None
        """
        session = await self._ensure_session()

        for attempt in range(self.max_retries):
            try:
                # 等待速率限制
                await self.rate_limiter.acquire()

                async with session.request(method, url, **kwargs) as response:
                    # 处理速率限制响应
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(
                            f"Rate limited, waiting {retry_after}s before retry"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()

                    # 添加请求延迟
                    await asyncio.sleep(self.request_delay)

                    return await response.json()

            except aiohttp.ClientResponseError as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"HTTP {e.status} - {e.message}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.request_delay * (attempt + 1))

            except aiohttp.ClientError as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.request_delay * (attempt + 1))

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.request_delay * (attempt + 1))

        return None

    # ==================== 论文解析 ====================

    def _parse_paper(self, data: Dict[str, Any]) -> Optional[Paper]:
        """解析 API 响应为 Paper 对象

        Args:
            data: API 返回的论文数据

        Returns:
            Paper 对象或 None
        """
        try:
            # 解析作者
            authors = []
            for author_data in data.get("authors", []):
                affiliations = author_data.get("affiliations", [])
                affiliation = affiliations[0] if affiliations else None
                authors.append(Author(
                    name=author_data.get("name", ""),
                    affiliation=affiliation,
                ))

            # 解析 PDF URL
            pdf_url = None
            open_access_pdf = data.get("openAccessPdf")
            if open_access_pdf:
                pdf_url = open_access_pdf.get("url")

            # 解析关键词（从 fieldsOfStudy 获取）
            keywords = data.get("fieldsOfStudy", []) or []

            # 构建元数据
            metadata = PaperMetadata(
                venue=data.get("venue"),
                year=data.get("year"),
                keywords=keywords,
                citations=data.get("citationCount"),
            )

            # 构建 Paper 对象
            paper_id = data.get("paperId", "")
            return Paper(
                id=paper_id,
                title=data.get("title", ""),
                abstract=data.get("abstract", "") or "",
                authors=authors,
                source=PaperSource.SEMANTIC_SCHOLAR,
                url=data.get("url", f"https://www.semanticscholar.org/paper/{paper_id}"),
                pdf_url=pdf_url,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Failed to parse Semantic Scholar paper: {e}")
            return None

    # ==================== 同步接口（实现基类抽象方法） ====================

    def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Paper]:
        """同步搜索论文（实现基类抽象方法）

        Args:
            query: 搜索关键词
            limit: 最大返回数量
            offset: 偏移量

        Returns:
            论文列表
        """
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(self.SEARCH_FIELDS),
        }

        response = self.get(url, params=params, headers=self.headers)
        if response is None:
            return []

        data = response.json()
        papers = []

        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        return papers

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """同步获取单篇论文详情（实现基类抽象方法）

        Args:
            paper_id: 论文 ID（Semantic Scholar paper ID）

        Returns:
            Paper 对象或 None
        """
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            "fields": ",".join(self.PAPER_FIELDS),
        }

        response = self.get(url, params=params, headers=self.headers)
        if response is None:
            return None

        return self._parse_paper(response.json())

    # ==================== 异步接口 ====================

    async def search_async(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        year: Optional[str] = None,
        venue: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False,
    ) -> List[Paper]:
        """异步搜索论文

        Args:
            query: 搜索关键词
            limit: 最大返回数量（API 单次最多返回 100）
            offset: 偏移量（用于分页）
            year: 年份过滤（如 "2023" 或 "2020-2023"）
            venue: 会议/期刊名称
            fields_of_study: 研究领域过滤
            open_access_only: 是否只返回开放获取论文

        Returns:
            论文列表
        """
        url = f"{self.BASE_URL}/paper/search"

        params = {
            "query": query,
            "limit": min(limit, 100),  # API 最大 100
            "offset": offset,
            "fields": ",".join(self.SEARCH_FIELDS),
        }

        # 添加过滤条件
        if year:
            params["year"] = year
        if venue:
            params["venue"] = venue
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""

        self.logger.info(f"Searching papers: query='{query}', limit={limit}, offset={offset}")

        data = await self._request_async("GET", url, params=params)

        if not data:
            self.logger.warning(f"Search returned no results for query: {query}")
            return []

        papers = []
        for paper_data in data.get("data", []):
            paper = self._parse_paper(paper_data)
            if paper:
                papers.append(paper)

        total = data.get("total", 0)
        self.logger.info(f"Found {len(papers)} papers (total available: {total})")

        return papers

    async def search_all_async(
        self,
        query: str,
        limit: Optional[int] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> AsyncGenerator[Paper, None]:
        """异步迭代搜索所有论文（自动分页）

        Args:
            query: 搜索关键词
            limit: 最大返回数量（None 表示不限制）
            batch_size: 每批获取数量
            **kwargs: 传递给 search_async 的其他参数

        Yields:
            Paper 对象
        """
        offset = 0
        count = 0

        while True:
            current_limit = batch_size
            if limit is not None:
                current_limit = min(batch_size, limit - count)
                if current_limit <= 0:
                    break

            papers = await self.search_async(
                query,
                limit=current_limit,
                offset=offset,
                **kwargs,
            )

            if not papers:
                break

            for paper in papers:
                yield paper
                count += 1
                if limit is not None and count >= limit:
                    return

            offset += len(papers)

            # 如果返回的论文数少于请求数，说明已经没有更多了
            if len(papers) < current_limit:
                break

    async def search_by_venue(
        self,
        venue: str,
        year: Optional[int] = None,
        limit: int = 100,
        query: str = "*",
    ) -> List[Paper]:
        """按会议/期刊搜索论文

        Args:
            venue: 会议/期刊名称（如 "NeurIPS", "ACL", "ICML"）
            year: 年份
            limit: 最大返回数量
            query: 搜索关键词（默认 "*" 表示所有）

        Returns:
            论文列表
        """
        year_str = str(year) if year else None

        self.logger.info(f"Searching papers by venue: {venue}, year={year}")

        return await self.search_async(
            query=query,
            limit=limit,
            venue=venue,
            year=year_str,
        )

    async def get_paper_async(self, paper_id: str) -> Optional[Paper]:
        """异步获取单篇论文详情

        Args:
            paper_id: 论文 ID，支持以下格式:
                - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
                - DOI: "DOI:10.1234/example"
                - arXiv ID: "ARXIV:2106.12345"
                - ACL ID: "ACL:P19-1001"
                - PubMed ID: "PMID:12345678"
                - Corpus ID: "CorpusId:12345678"

        Returns:
            Paper 对象或 None
        """
        url = f"{self.BASE_URL}/paper/{paper_id}"

        params = {
            "fields": ",".join(self.PAPER_FIELDS),
        }

        self.logger.info(f"Getting paper details: {paper_id}")

        data = await self._request_async("GET", url, params=params)

        if not data:
            self.logger.warning(f"Paper not found: {paper_id}")
            return None

        paper = self._parse_paper(data)
        if paper:
            self.logger.info(f"Retrieved paper: {paper.title}")
        return paper

    async def get_papers_batch(
        self,
        paper_ids: List[str],
        batch_size: int = 500,  # API 最大 500
    ) -> List[Paper]:
        """批量获取论文详情

        Args:
            paper_ids: 论文 ID 列表
            batch_size: 每批获取数量（API 最大 500）

        Returns:
            论文列表
        """
        url = f"{self.BASE_URL}/paper/batch"

        params = {
            "fields": ",".join(self.PAPER_FIELDS),
        }

        all_papers = []

        # 分批处理
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]

            self.logger.info(
                f"Batch fetching papers: {i + 1}-{i + len(batch_ids)} of {len(paper_ids)}"
            )

            data = await self._request_async(
                "POST",
                url,
                params=params,
                json={"ids": batch_ids},
            )

            if not data:
                self.logger.warning(f"Batch request failed for papers {i + 1}-{i + len(batch_ids)}")
                continue

            for paper_data in data:
                if paper_data is None:
                    continue
                paper = self._parse_paper(paper_data)
                if paper:
                    all_papers.append(paper)

        self.logger.info(f"Batch fetched {len(all_papers)} papers")
        return all_papers

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """获取论文的引用（哪些论文引用了这篇论文）

        Args:
            paper_id: 论文 ID
            limit: 最大返回数量
            offset: 偏移量

        Returns:
            引用论文列表
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"

        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": ",".join([f"citingPaper.{f}" for f in self.SEARCH_FIELDS]),
        }

        self.logger.info(f"Getting citations for paper: {paper_id}")

        data = await self._request_async("GET", url, params=params)

        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper")
            if citing_paper:
                paper = self._parse_paper(citing_paper)
                if paper:
                    papers.append(paper)

        self.logger.info(f"Found {len(papers)} citations")
        return papers

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """获取论文的参考文献

        Args:
            paper_id: 论文 ID
            limit: 最大返回数量
            offset: 偏移量

        Returns:
            参考文献列表
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/references"

        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": ",".join([f"citedPaper.{f}" for f in self.SEARCH_FIELDS]),
        }

        self.logger.info(f"Getting references for paper: {paper_id}")

        data = await self._request_async("GET", url, params=params)

        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            cited_paper = item.get("citedPaper")
            if cited_paper:
                paper = self._parse_paper(cited_paper)
                if paper:
                    papers.append(paper)

        self.logger.info(f"Found {len(papers)} references")
        return papers

    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 10,
    ) -> List[Paper]:
        """获取基于论文的推荐

        Args:
            paper_id: 论文 ID
            limit: 最大返回数量

        Returns:
            推荐论文列表
        """
        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"

        params = {
            "limit": min(limit, 100),
            "fields": ",".join(self.SEARCH_FIELDS),
        }

        self.logger.info(f"Getting recommendations for paper: {paper_id}")

        data = await self._request_async("GET", url, params=params)

        if not data:
            return []

        papers = []
        for paper_data in data.get("recommendedPapers", []):
            paper = self._parse_paper(paper_data)
            if paper:
                papers.append(paper)

        self.logger.info(f"Found {len(papers)} recommendations")
        return papers


# ==================== 便捷函数 ====================

async def search_papers(
    query: str,
    limit: int = 100,
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Paper]:
    """便捷函数：搜索论文

    Args:
        query: 搜索关键词
        limit: 最大返回数量
        api_key: API Key（可选）
        **kwargs: 传递给 search_async 的其他参数

    Returns:
        论文列表
    """
    async with SemanticScholarScraper(api_key=api_key) as scraper:
        return await scraper.search_async(query, limit=limit, **kwargs)


async def get_paper(
    paper_id: str,
    api_key: Optional[str] = None,
) -> Optional[Paper]:
    """便捷函数：获取论文详情

    Args:
        paper_id: 论文 ID
        api_key: API Key（可选）

    Returns:
        Paper 对象或 None
    """
    async with SemanticScholarScraper(api_key=api_key) as scraper:
        return await scraper.get_paper_async(paper_id)


async def search_by_venue(
    venue: str,
    year: Optional[int] = None,
    limit: int = 100,
    api_key: Optional[str] = None,
) -> List[Paper]:
    """便捷函数：按会议搜索论文

    Args:
        venue: 会议/期刊名称
        year: 年份
        limit: 最大返回数量
        api_key: API Key（可选）

    Returns:
        论文列表
    """
    async with SemanticScholarScraper(api_key=api_key) as scraper:
        return await scraper.search_by_venue(venue, year=year, limit=limit)


if __name__ == "__main__":
    # 简单测试
    async def main():
        async with SemanticScholarScraper() as scraper:
            # 测试搜索
            print("=== Search Test ===")
            papers = await scraper.search_async("transformer attention", limit=5)
            for paper in papers:
                print(f"- {paper.title}")
                print(f"  Authors: {', '.join(a.name for a in paper.authors[:3])}")
                print(f"  Citations: {paper.metadata.citations}")
                print(f"  PDF: {paper.pdf_url}")
                print()

            # 测试按会议搜索
            print("=== Venue Search Test ===")
            papers = await scraper.search_by_venue("NeurIPS", year=2023, limit=3)
            for paper in papers:
                print(f"- {paper.title}")
                print(f"  Venue: {paper.metadata.venue}")
                print()

            # 测试获取单篇论文
            if papers:
                print("=== Paper Detail Test ===")
                paper = await scraper.get_paper_async(papers[0].id)
                if paper:
                    print(f"Title: {paper.title}")
                    print(f"Abstract: {paper.abstract[:200]}..." if paper.abstract else "No abstract")

            # 测试批量获取
            if len(papers) >= 2:
                print("\n=== Batch Fetch Test ===")
                paper_ids = [p.id for p in papers[:2]]
                batch_papers = await scraper.get_papers_batch(paper_ids)
                print(f"Batch fetched {len(batch_papers)} papers")

    asyncio.run(main())
