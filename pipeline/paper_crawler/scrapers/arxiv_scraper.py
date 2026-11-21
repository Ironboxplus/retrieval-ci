"""
ArxivScraper - arXiv论文爬虫

从arXiv获取学术论文、预印本。
API文档: https://arxiv.org/help/api
速率限制: 3秒/请求
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import arxiv

from pipeline.paper_crawler.scrapers.base_scraper import (
    BaseScraper,
    ScraperError,
    NotFoundError,
    RateLimiter,
    RetryPolicy,
)
from pipeline.paper_crawler.models.paper import (
    Paper,
    Author,
    Conference,
    PaperType,
)


class ArxivScraper(BaseScraper):
    """arXiv论文爬虫

    使用官方arxiv Python库访问arXiv API。

    Features:
        - 根据arXiv ID获取论文
        - 按关键词、类别搜索论文
        - 下载PDF文件
        - 获取完整元数据

    Example:
        >>> async with ArxivScraper() as scraper:
        ...     paper = await scraper.get_paper_by_arxiv_id("2301.07041")
        ...     print(paper.title)
    """

    # arXiv 类别映射
    CATEGORIES = {
        # Computer Science
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.LG": "Machine Learning",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.RO": "Robotics",
        "cs.SE": "Software Engineering",
        "cs.DS": "Data Structures and Algorithms",
        "cs.IR": "Information Retrieval",
        # Statistics
        "stat.ML": "Machine Learning (Statistics)",
        # Mathematics
        "math.OC": "Optimization and Control",
        # Physics
        "quant-ph": "Quantum Physics",
        # Electrical Engineering
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
    }

    # 类别到Conference的映射
    CATEGORY_TO_CONFERENCE = {
        "cs.LG": Conference.OTHER,
        "cs.CV": Conference.CVPR,
        "cs.CL": Conference.ACL,
        "cs.AI": Conference.AAAI,
    }

    def __init__(
        self,
        page_size: int = 100,
        **kwargs,
    ):
        """初始化ArxivScraper

        Args:
            page_size: 每页结果数量
            **kwargs: 传递给BaseScraper的参数
        """
        # 设置arXiv特定的速率限制(至少3秒间隔)
        if "rate_limiter" not in kwargs:
            kwargs["rate_limiter"] = RateLimiter(
                requests_per_second=1.0 / 3.0,  # 3秒一个请求
                burst_size=1,
            )

        super().__init__(**kwargs)
        self.page_size = page_size
        self._arxiv_client: Optional[arxiv.Client] = None

    # ==================== 抽象属性实现 ====================

    @property
    def source_name(self) -> str:
        return "arxiv"

    @property
    def base_url(self) -> str:
        return "https://arxiv.org"

    @property
    def supported_conferences(self) -> List[str]:
        # arXiv不是会议,但可以映射类别
        return list(self.CATEGORIES.keys())

    # ==================== arxiv客户端 ====================

    @property
    def arxiv_client(self) -> arxiv.Client:
        """懒加载arxiv客户端"""
        if self._arxiv_client is None:
            self._arxiv_client = arxiv.Client(
                page_size=self.page_size,
                delay_seconds=3.0,  # arXiv要求
                num_retries=self.retry_policy.max_retries,
            )
        return self._arxiv_client

    async def close(self) -> None:
        """关闭客户端"""
        await super().close()
        self._arxiv_client = None

    # ==================== ID处理工具 ====================

    @staticmethod
    def normalize_arxiv_id(arxiv_id: str) -> str:
        """标准化arXiv ID

        移除'arxiv:'前缀和版本号后缀。

        Args:
            arxiv_id: arXiv论文ID (如 "arxiv:2301.07041v1", "2301.07041")

        Returns:
            标准化后的ID (如 "2301.07041")
        """
        arxiv_id = arxiv_id.strip()
        if arxiv_id.lower().startswith("arxiv:"):
            arxiv_id = arxiv_id[6:]

        # 移除版本号
        if re.search(r'v\d+$', arxiv_id):
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

        return arxiv_id

    @staticmethod
    def is_valid_arxiv_id(arxiv_id: str) -> bool:
        """验证arXiv ID格式"""
        normalized = ArxivScraper.normalize_arxiv_id(arxiv_id)
        # 新格式: YYMM.NNNNN
        if re.match(r'^\d{4}\.\d{4,5}$', normalized):
            return True
        # 旧格式: category/YYMMNNN
        if re.match(r'^[a-z-]+/\d{7}$', normalized, re.IGNORECASE):
            return True
        return False

    # ==================== 结果转换 ====================

    def _result_to_paper(self, result: arxiv.Result) -> Paper:
        """将arxiv.Result转换为Paper对象"""
        # 提取arXiv ID
        arxiv_id = result.entry_id.split("/")[-1]
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

        # 解析作者
        authors = [Author(name=author.name) for author in result.authors]

        # 解析类别
        categories = result.categories if result.categories else []
        primary_category = result.primary_category or ""

        # 推断会议
        conference = Conference.OTHER
        for cat, conf in self.CATEGORY_TO_CONFERENCE.items():
            if cat in categories or cat == primary_category:
                conference = conf
                break

        # 提取年份
        year = result.published.year if result.published else datetime.now().year

        # 构建Paper对象
        return Paper(
            id=f"arxiv:{arxiv_id}",
            title=result.title.replace("\n", " ").strip(),
            authors=authors,
            abstract=result.summary.replace("\n", " ").strip() if result.summary else None,
            keywords=list(categories),
            year=year,
            conference=conference,
            paper_type=PaperType.UNKNOWN,
            arxiv_id=arxiv_id,
            pdf_url=result.pdf_url,
            paper_page_url=result.entry_id,
            source=self.source_name,
            created_at=result.published or datetime.now(),
        )

    # ==================== 抽象方法实现 ====================

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

        对于arXiv,conference参数被解释为类别代码(如 cs.LG)

        Args:
            conference: arXiv类别代码
            year: 年份
            paper_type: 忽略(arXiv没有论文类型)
            track: 忽略
            limit: 返回数量限制
            offset: 偏移量
        """
        query = f"cat:{conference}"

        # 添加年份过滤
        # arXiv的日期查询格式: submittedDate:[YYYYMMDD TO YYYYMMDD]
        date_from = f"{year}0101"
        date_to = f"{year}1231"
        query += f" AND submittedDate:[{date_from} TO {date_to}]"

        return await self.search(
            query=query,
            limit=limit or 100,
        )

    async def get_paper_details(self, paper_id: str) -> Paper:
        """获取论文详情

        Args:
            paper_id: arXiv论文ID
        """
        paper = await self.get_paper_by_arxiv_id(paper_id)
        if paper is None:
            raise NotFoundError(
                f"Paper not found: {paper_id}",
                source=self.source_name,
            )
        return paper

    async def get_pdf_url(self, paper: Paper) -> str:
        """获取论文PDF下载链接"""
        if paper.pdf_url:
            return str(paper.pdf_url)

        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

        raise NotFoundError(
            f"Cannot determine PDF URL for paper: {paper.id}",
            source=self.source_name,
        )

    async def search(
        self,
        query: str,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100,
    ) -> List[Paper]:
        """搜索arXiv论文

        Args:
            query: 搜索查询字符串，支持arXiv查询语法
            conference: 类别过滤(如 cs.LG)
            year: 年份过滤
            limit: 最大返回数量

        Returns:
            Paper对象列表
        """
        # 构建查询
        full_query = query

        if conference:
            full_query = f"cat:{conference} AND ({full_query})"

        if year:
            date_from = f"{year}0101"
            date_to = f"{year}1231"
            full_query = f"({full_query}) AND submittedDate:[{date_from} TO {date_to}]"

        try:
            search = arxiv.Search(
                query=full_query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            papers = []
            for result in self.arxiv_client.results(search):
                paper = self._result_to_paper(result)
                papers.append(paper)

            self.logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            raise ScraperError(
                f"Search failed: {e}",
                source=self.source_name,
            )

    # ==================== arXiv特定方法 ====================

    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """根据arXiv ID获取单篇论文

        Args:
            arxiv_id: arXiv论文ID

        Returns:
            Paper对象或None
        """
        normalized_id = self.normalize_arxiv_id(arxiv_id)

        if not self.is_valid_arxiv_id(normalized_id):
            self.logger.error(f"Invalid arXiv ID format: {arxiv_id}")
            return None

        try:
            search = arxiv.Search(id_list=[normalized_id])
            results = list(self.arxiv_client.results(search))

            if results:
                paper = self._result_to_paper(results[0])
                self.logger.info(f"Retrieved paper: {paper.title}")
                return paper
            else:
                self.logger.warning(f"Paper not found: {normalized_id}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to get paper {arxiv_id}: {e}")
            return None

    async def get_papers_by_ids(self, arxiv_ids: List[str]) -> List[Paper]:
        """批量获取多篇论文

        Args:
            arxiv_ids: arXiv论文ID列表

        Returns:
            Paper对象列表
        """
        normalized_ids = [
            self.normalize_arxiv_id(pid)
            for pid in arxiv_ids
            if self.is_valid_arxiv_id(self.normalize_arxiv_id(pid))
        ]

        if not normalized_ids:
            return []

        try:
            search = arxiv.Search(id_list=normalized_ids)
            papers = []

            for result in self.arxiv_client.results(search):
                paper = self._result_to_paper(result)
                papers.append(paper)
                self.logger.info(f"Retrieved paper: {paper.title}")

            return papers

        except Exception as e:
            self.logger.error(f"Failed to get papers: {e}")
            return []

    async def search_by_category(
        self,
        category: str,
        limit: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Paper]:
        """按类别搜索论文

        Args:
            category: arXiv类别代码 (如 "cs.LG", "cs.CV")
            limit: 最大返回数量
            date_from: 起始日期 (格式: "YYYYMMDD")
            date_to: 结束日期 (格式: "YYYYMMDD")

        Returns:
            Paper对象列表
        """
        query = f"cat:{category}"

        if date_from or date_to:
            from_date = date_from or "*"
            to_date = date_to or "*"
            query += f" AND submittedDate:[{from_date} TO {to_date}]"

        return await self.search(query, limit=limit)

    async def search_by_author(self, author: str, limit: int = 100) -> List[Paper]:
        """按作者搜索论文"""
        return await self.search(f"au:{author}", limit=limit)

    async def search_by_title(self, title: str, limit: int = 100) -> List[Paper]:
        """按标题搜索论文"""
        return await self.search(f"ti:{title}", limit=limit)

    async def get_latest_papers(
        self,
        category: str = "cs.LG",
        limit: int = 50,
    ) -> List[Paper]:
        """获取指定类别的最新论文"""
        return await self.search_by_category(category=category, limit=limit)

    async def download_pdf(
        self,
        paper: Paper,
        dirpath: str = "./papers",
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """下载论文PDF

        Args:
            paper: Paper对象
            dirpath: 下载目录
            filename: 自定义文件名(不含扩展名)

        Returns:
            下载的PDF文件路径
        """
        pdf_url = await self.get_pdf_url(paper)

        # 创建目录
        dir_path = Path(dirpath)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 确定文件名
        if filename:
            pdf_filename = f"{filename}.pdf"
        else:
            safe_id = (paper.arxiv_id or paper.id).replace("/", "_").replace(":", "_")
            pdf_filename = f"{safe_id}.pdf"

        filepath = dir_path / pdf_filename

        try:
            response = await self.get(pdf_url)
            with open(filepath, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Downloaded PDF: {filepath}")
            return str(filepath.absolute())

        except Exception as e:
            self.logger.error(f"Failed to download PDF for {paper.id}: {e}")
            return None

    async def download_pdf_by_id(
        self,
        arxiv_id: str,
        dirpath: str = "./papers",
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """根据arXiv ID下载PDF"""
        paper = await self.get_paper_by_arxiv_id(arxiv_id)
        if paper is None:
            return None
        return await self.download_pdf(paper, dirpath, filename)

    def build_query(
        self,
        all_fields: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        journal_ref: Optional[str] = None,
    ) -> str:
        """构建arXiv查询字符串

        Args:
            all_fields: 搜索所有字段
            title: 标题关键词
            author: 作者姓名
            abstract: 摘要关键词
            category: 类别代码
            journal_ref: 期刊引用

        Returns:
            arXiv查询字符串
        """
        parts = []

        if all_fields:
            parts.append(f"all:{all_fields}")
        if title:
            parts.append(f"ti:{title}")
        if author:
            parts.append(f"au:{author}")
        if abstract:
            parts.append(f"abs:{abstract}")
        if category:
            parts.append(f"cat:{category}")
        if journal_ref:
            parts.append(f"jr:{journal_ref}")

        return " AND ".join(parts) if parts else ""


# ==================== 便捷函数 ====================

async def get_arxiv_paper(arxiv_id: str) -> Optional[Paper]:
    """快速获取单篇arXiv论文"""
    async with ArxivScraper() as scraper:
        return await scraper.get_paper_by_arxiv_id(arxiv_id)


async def search_arxiv(query: str, limit: int = 10) -> List[Paper]:
    """快速搜索arXiv论文"""
    async with ArxivScraper() as scraper:
        return await scraper.search(query, limit=limit)


async def download_arxiv_pdf(arxiv_id: str, dirpath: str = "./papers") -> Optional[str]:
    """快速下载arXiv论文PDF"""
    async with ArxivScraper() as scraper:
        return await scraper.download_pdf_by_id(arxiv_id, dirpath)
