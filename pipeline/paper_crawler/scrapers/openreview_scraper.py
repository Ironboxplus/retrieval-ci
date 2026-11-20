"""
OpenReviewScraper - OpenReview 论文爬虫

使用 openreview-py 官方 SDK 获取 ICLR、ICML、NeurIPS 等顶会论文。

支持功能:
- 按会议和年份筛选论文
- 获取 Best Paper / Outstanding Paper
- 获取论文完整元数据
- 批量获取并保存

安装依赖:
    pip install openreview-py
"""

from typing import List, Optional, Dict, Any, Generator
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json
import time

try:
    import openreview
    from openreview.api import OpenReviewClient
    OPENREVIEW_AVAILABLE = True
except ImportError:
    OPENREVIEW_AVAILABLE = False
    OpenReviewClient = None  # type: ignore  # Placeholder for type hints

from pipeline.paper_crawler.scrapers.base_scraper import BaseScraper
from pipeline.paper_crawler.models.paper import Paper, Author, PaperType, Conference as PaperConference


class Conference(Enum):
    """支持的会议枚举"""
    ICLR = "ICLR.cc"
    ICML = "ICML.cc"
    NEURIPS = "NeurIPS.cc"


class DecisionType(Enum):
    """论文决议类型"""
    ACCEPT = "accept"
    ORAL = "oral"
    SPOTLIGHT = "spotlight"
    POSTER = "poster"
    REJECT = "reject"


@dataclass
class OpenReviewConfig:
    """OpenReview 配置"""
    baseurl: str = "https://api2.openreview.net"
    username: Optional[str] = None
    password: Optional[str] = None


class OpenReviewScraper(BaseScraper):
    """OpenReview 论文爬虫

    使用 openreview-py SDK 获取 ICLR、ICML、NeurIPS 论文。

    示例:
        >>> scraper = OpenReviewScraper()
        >>> papers = scraper.get_conference_papers("ICLR", 2024, limit=100)
        >>> best_papers = scraper.get_best_papers("ICLR", 2024)
    """

    # 会议 venue 模板
    VENUE_TEMPLATES = {
        "ICLR": "ICLR.cc/{year}/Conference",
        "ICML": "ICML.cc/{year}/Conference",
        "NEURIPS": "NeurIPS.cc/{year}/Conference",
    }

    # 会议支持的年份范围
    SUPPORTED_YEARS = {
        "ICLR": range(2018, 2027),
        "ICML": range(2019, 2027),
        "NEURIPS": range(2019, 2027),
    }

    def __init__(
        self,
        config: Optional[OpenReviewConfig] = None,
        **kwargs,
    ):
        """初始化 OpenReview 爬虫

        Args:
            config: OpenReview 配置
            **kwargs: BaseScraper 参数
        """
        super().__init__(**kwargs)

        if not OPENREVIEW_AVAILABLE:
            raise ImportError(
                "openreview-py is required. Install with: pip install openreview-py"
            )

        self.config = config or OpenReviewConfig()
        self._or_client: Optional[OpenReviewClient] = None
        # For sync close method
        self.request_delay = 1.0

    # ==================== Abstract Property Implementations ====================

    @property
    def source_name(self) -> str:
        """Data source name."""
        return "openreview"

    @property
    def base_url(self) -> str:
        """API base URL."""
        return "https://api2.openreview.net"

    @property
    def supported_conferences(self) -> List[str]:
        """List of supported conferences."""
        return list(self.VENUE_TEMPLATES.keys())

    # ==================== Abstract Method Implementations ====================

    async def get_paper_list(
        self,
        conference: str,
        year: int,
        paper_type: Optional[Any] = None,
        track: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Paper]:
        """Get list of papers for a conference/year."""
        decision = None
        if paper_type:
            if hasattr(paper_type, 'value'):
                decision = paper_type.value
            else:
                decision = str(paper_type)
        return self.get_conference_papers(conference, year, limit=limit, decision=decision)

    async def get_paper_details(self, paper_id: str) -> Paper:
        """Get paper details by ID."""
        paper = self.get_paper(paper_id)
        if paper is None:
            from pipeline.paper_crawler.scrapers.base_scraper import NotFoundError
            raise NotFoundError(f"Paper not found: {paper_id}", source="openreview")
        return paper

    async def get_pdf_url(self, paper: Paper) -> str:
        """Get PDF URL for a paper."""
        if paper.pdf_url:
            return str(paper.pdf_url)
        return f"https://openreview.net/pdf?id={paper.id}"

    @property
    def or_client(self) -> OpenReviewClient:
        """懒加载 OpenReview 客户端"""
        if self._or_client is None:
            try:
                if self.config.username and self.config.password:
                    self._or_client = OpenReviewClient(
                        baseurl=self.config.baseurl,
                        username=self.config.username,
                        password=self.config.password,
                    )
                else:
                    self._or_client = OpenReviewClient(
                        baseurl=self.config.baseurl,
                    )
                self.logger.info(f"Connected to OpenReview API: {self.config.baseurl}")
            except Exception as e:
                self.logger.error(f"Failed to connect to OpenReview: {e}")
                raise
        return self._or_client

    def close(self):
        """关闭客户端连接"""
        super().close()
        self._or_client = None

    def _get_venue(self, conference: str, year: int) -> str:
        """获取会议的 venue 标识符

        Args:
            conference: 会议名称 (ICLR, ICML, NeurIPS)
            year: 年份

        Returns:
            venue 字符串
        """
        conference = conference.upper()
        if conference not in self.VENUE_TEMPLATES:
            raise ValueError(f"Unsupported conference: {conference}")
        return self.VENUE_TEMPLATES[conference].format(year=year)

    def _get_submission_invitation(self, conference: str, year: int) -> str:
        """获取投稿 invitation ID

        不同年份的 invitation 格式可能不同。
        """
        venue = self._get_venue(conference, year)
        # 大部分情况使用 /-/Submission
        return f"{venue}/-/Submission"

    def _get_decision_invitation(self, conference: str, year: int) -> str:
        """获取决议 invitation ID"""
        venue = self._get_venue(conference, year)
        return f"{venue}/-/Decision"

    def _parse_note_to_paper(self, note: Any, conference: str, year: int) -> Optional[Paper]:
        """将 OpenReview Note 转换为 Paper 对象

        Args:
            note: OpenReview Note 对象
            conference: 会议名称
            year: 年份

        Returns:
            Paper 对象或 None
        """
        try:
            content = note.content

            # 提取标题 - 处理不同格式
            title = self._extract_field(content, "title")
            if not title:
                self.logger.warning(f"Note {note.id} has no title, skipping")
                return None

            # 提取摘要
            abstract = self._extract_field(content, "abstract") or ""

            # 提取作者
            authors = self._parse_authors(content)

            # 提取关键词
            keywords = self._extract_list_field(content, "keywords")

            # 构建 PDF URL
            pdf_url = self._get_pdf_url(note)

            # 论文 URL
            paper_url = f"https://openreview.net/forum?id={note.id}"

            # 获取决议/状态
            venue_str = self._extract_field(content, "venue") or f"{conference} {year}"

            # Determine paper type from venue string
            venue_lower = venue_str.lower()
            paper_type = PaperType.UNKNOWN
            if "oral" in venue_lower:
                paper_type = PaperType.ORAL
            elif "spotlight" in venue_lower:
                paper_type = PaperType.SPOTLIGHT
            elif "poster" in venue_lower:
                paper_type = PaperType.POSTER
            elif "outstanding" in venue_lower or "best paper" in venue_lower:
                paper_type = PaperType.OUTSTANDING

            # Map conference string to Conference enum
            conf_map = {
                "ICLR": PaperConference.ICLR,
                "ICML": PaperConference.ICML,
                "NEURIPS": PaperConference.NEURIPS,
            }
            paper_conference = conf_map.get(conference.upper(), PaperConference.OTHER)

            return Paper(
                id=note.id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                conference=paper_conference,
                paper_type=paper_type,
                keywords=keywords,
                pdf_url=pdf_url,
                paper_page_url=paper_url,
                openreview_id=note.id,
                source="openreview",
            )

        except Exception as e:
            self.logger.error(f"Failed to parse note {getattr(note, 'id', 'unknown')}: {e}")
            return None

    def _extract_field(self, content: Dict[str, Any], field_name: str) -> Optional[str]:
        """从内容中提取字段值

        OpenReview 内容字段可能是字符串或字典格式。
        """
        if field_name not in content:
            return None

        value = content[field_name]
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return str(value) if value else None

    def _extract_list_field(self, content: Dict[str, Any], field_name: str) -> List[str]:
        """从内容中提取列表字段"""
        if field_name not in content:
            return []

        value = content[field_name]
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, dict) and "value" in value:
            val = value["value"]
            return [str(v) for v in val] if isinstance(val, list) else []
        return []

    def _parse_authors(self, content: Dict[str, Any]) -> List[Author]:
        """解析作者信息"""
        authors = []

        # 尝试获取作者名称列表
        author_names = self._extract_list_field(content, "authors")

        # 尝试获取作者机构
        affiliations = self._extract_list_field(content, "authorids") or []

        for i, name in enumerate(author_names):
            affiliation = affiliations[i] if i < len(affiliations) else None
            # authorids 通常是邮箱格式，可以提取机构
            if affiliation and "@" in str(affiliation):
                affiliation = None
            authors.append(Author(name=name, affiliation=affiliation))

        return authors

    def _get_pdf_url(self, note: Any) -> Optional[str]:
        """获取 PDF 下载链接"""
        content = note.content

        # 尝试从 content 获取 PDF 字段
        pdf_field = content.get("pdf")
        if pdf_field:
            if isinstance(pdf_field, str):
                if pdf_field.startswith("http"):
                    return pdf_field
                return f"https://openreview.net{pdf_field}"
            if isinstance(pdf_field, dict) and "value" in pdf_field:
                val = pdf_field["value"]
                if val:
                    if val.startswith("http"):
                        return val
                    return f"https://openreview.net{val}"

        # 默认 PDF URL 格式
        return f"https://openreview.net/pdf?id={note.id}"

    def get_conference_papers(
        self,
        conference: str,
        year: int,
        limit: Optional[int] = None,
        decision: Optional[str] = None,
    ) -> List[Paper]:
        """获取指定会议年份的论文

        Args:
            conference: 会议名称 (ICLR, ICML, NeurIPS)
            year: 年份
            limit: 最大返回数量 (None 表示获取全部)
            decision: 筛选决议类型 (oral, spotlight, poster)

        Returns:
            论文列表
        """
        conference = conference.upper()
        if conference not in self.VENUE_TEMPLATES:
            raise ValueError(f"Unsupported conference: {conference}. "
                           f"Supported: {list(self.VENUE_TEMPLATES.keys())}")

        self.logger.info(f"Fetching {conference} {year} papers...")

        try:
            venue = self._get_venue(conference, year)

            # 构建查询参数
            # 尝试获取已接收的论文 (通过 venue content 筛选)
            papers = []

            # 方法1: 使用 get_all_notes 获取所有提交
            invitation = self._get_submission_invitation(conference, year)
            self.logger.debug(f"Using invitation: {invitation}")

            try:
                notes = self.or_client.get_all_notes(
                    invitation=invitation,
                    details="original",
                )
            except Exception as e:
                # 尝试备用 invitation 格式
                self.logger.warning(f"Primary invitation failed: {e}, trying alternatives...")
                notes = self._try_alternative_invitations(conference, year)

            self.logger.info(f"Found {len(notes) if notes else 0} submissions")

            # 解析并筛选论文
            count = 0
            for note in notes or []:
                if limit and count >= limit:
                    break

                # 根据 decision 筛选
                if decision:
                    note_venue = self._extract_field(note.content, "venue") or ""
                    note_venue_lower = note_venue.lower()
                    if decision.lower() not in note_venue_lower:
                        continue

                paper = self._parse_note_to_paper(note, conference, year)
                if paper:
                    papers.append(paper)
                    count += 1

            self.logger.info(f"Parsed {len(papers)} papers successfully")
            return papers

        except Exception as e:
            self.logger.error(f"Failed to fetch {conference} {year} papers: {e}")
            return []

    def _try_alternative_invitations(self, conference: str, year: int) -> List[Any]:
        """尝试备用的 invitation 格式"""
        venue = self._get_venue(conference, year)

        # 不同年份可能有不同的 invitation 格式
        alternative_invitations = [
            f"{venue}/-/Blind_Submission",
            f"{venue}/Paper/-/Submission",
            f"{venue}/-/submission",
            # For older ICLR conferences
            f"ICLR.cc/{year}/Conference/-/Blind_Submission",
            f"ICLR.cc/{year}/Conference/-/Submission",
            # For NeurIPS
            f"NeurIPS.cc/{year}/Conference/-/Submission",
            f"NeurIPS.cc/{year}/Conference/-/Blind_Submission",
            # Try with venue name directly
            f"{conference.upper()}.cc/{year}/Conference/-/Submission",
            f"{conference.upper()}.cc/{year}/Conference/-/Blind_Submission",
        ]

        for inv in alternative_invitations:
            try:
                self.logger.debug(f"Trying invitation: {inv}")
                notes = self.or_client.get_all_notes(invitation=inv)
                if notes:
                    self.logger.info(f"Found {len(notes)} notes with invitation: {inv}")
                    return notes
            except Exception:
                continue

        return []

    def get_accepted_papers(
        self,
        conference: str,
        year: int,
        limit: Optional[int] = None,
    ) -> List[Paper]:
        """获取已接收的论文

        Args:
            conference: 会议名称
            year: 年份
            limit: 最大返回数量

        Returns:
            已接收论文列表
        """
        conference = conference.upper()
        self.logger.info(f"Fetching accepted papers for {conference} {year}...")

        try:
            venue = self._get_venue(conference, year)

            # 使用 venue 字段筛选已接收论文
            # 已接收的论文通常在 content.venue 或 content.venueid 中有会议标识
            all_papers = self.get_conference_papers(conference, year, limit=None)

            accepted_papers = []
            for paper in all_papers:
                # 检查 venue 是否表明已接收
                venue_str = paper.metadata.venue or ""
                venue_lower = venue_str.lower()

                # 排除被拒稿的
                if "reject" in venue_lower or "withdrawn" in venue_lower:
                    continue

                # 检查是否有接收标识
                if any(keyword in venue_lower for keyword in
                       ["accept", "oral", "spotlight", "poster", conference.lower()]):
                    accepted_papers.append(paper)
                    if limit and len(accepted_papers) >= limit:
                        break

            self.logger.info(f"Found {len(accepted_papers)} accepted papers")
            return accepted_papers

        except Exception as e:
            self.logger.error(f"Failed to fetch accepted papers: {e}")
            return []

    def get_best_papers(
        self,
        conference: str,
        year: int,
    ) -> List[Paper]:
        """获取 Best Paper / Outstanding Paper

        Args:
            conference: 会议名称
            year: 年份

        Returns:
            Best Paper 列表
        """
        conference = conference.upper()
        self.logger.info(f"Fetching best papers for {conference} {year}...")

        try:
            venue = self._get_venue(conference, year)

            # Best Paper 通常在 venue 或 award 字段中标注
            all_papers = self.get_conference_papers(conference, year, limit=None)

            best_papers = []
            best_keywords = [
                "best paper", "outstanding", "award", "honorable mention",
                "best student", "notable", "top paper"
            ]

            for paper in all_papers:
                venue_str = paper.metadata.venue or ""
                venue_lower = venue_str.lower()

                # 检查是否有 Best Paper 标识
                if any(keyword in venue_lower for keyword in best_keywords):
                    best_papers.append(paper)
                    self.logger.info(f"Found best paper: {paper.title[:50]}...")

            self.logger.info(f"Found {len(best_papers)} best/outstanding papers")
            return best_papers

        except Exception as e:
            self.logger.error(f"Failed to fetch best papers: {e}")
            return []

    def get_oral_papers(
        self,
        conference: str,
        year: int,
    ) -> List[Paper]:
        """获取 Oral Presentation 论文"""
        return self.get_conference_papers(conference, year, decision="oral")

    def get_spotlight_papers(
        self,
        conference: str,
        year: int,
    ) -> List[Paper]:
        """获取 Spotlight 论文"""
        return self.get_conference_papers(conference, year, decision="spotlight")

    def get_poster_papers(
        self,
        conference: str,
        year: int,
    ) -> List[Paper]:
        """获取 Poster 论文"""
        return self.get_conference_papers(conference, year, decision="poster")

    def search(self, query: str, limit: int = 100) -> List[Paper]:
        """搜索论文

        在 OpenReview 中搜索论文。

        Args:
            query: 搜索关键词
            limit: 最大返回数量

        Returns:
            论文列表
        """
        self.logger.info(f"Searching for: {query}")

        try:
            # OpenReview API v2 支持全文搜索
            notes = self.or_client.get_all_notes(
                content={"title": {"value-regex": f".*{query}.*", "$options": "i"}},
                limit=limit,
            )

            papers = []
            for note in notes or []:
                paper = self._parse_note_to_paper(note, "Unknown", 0)
                if paper:
                    # 尝试从 venue 解析年份
                    venue = self._extract_field(note.content, "venue") or ""
                    for year in range(2018, 2027):
                        if str(year) in venue:
                            paper.metadata.year = year
                            break
                    papers.append(paper)

                if len(papers) >= limit:
                    break

            self.logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            # 回退到逐会议搜索
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> List[Paper]:
        """回退搜索方法 - 逐会议年份搜索"""
        papers = []
        query_lower = query.lower()

        for conf in ["ICLR", "ICML", "NEURIPS"]:
            for year in range(2024, 2018, -1):
                if len(papers) >= limit:
                    break

                try:
                    conf_papers = self.get_conference_papers(conf, year, limit=500)
                    for paper in conf_papers:
                        if len(papers) >= limit:
                            break
                        if query_lower in paper.title.lower() or query_lower in paper.abstract.lower():
                            papers.append(paper)
                except Exception:
                    continue

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """获取单篇论文详情

        Args:
            paper_id: OpenReview 论文 ID (forum ID)

        Returns:
            论文对象或 None
        """
        self.logger.info(f"Fetching paper: {paper_id}")

        try:
            note = self.or_client.get_note(paper_id)
            if note:
                # 尝试从 venue 解析会议和年份
                venue = self._extract_field(note.content, "venue") or ""
                conference = "Unknown"
                year = 0

                for conf in ["ICLR", "ICML", "NeurIPS"]:
                    if conf.lower() in venue.lower():
                        conference = conf
                        break

                for y in range(2018, 2027):
                    if str(y) in venue:
                        year = y
                        break

                return self._parse_note_to_paper(note, conference, year)
            return None

        except Exception as e:
            self.logger.error(f"Failed to get paper {paper_id}: {e}")
            return None

    def iter_conference_papers(
        self,
        conference: str,
        year: int,
        batch_size: int = 100,
    ) -> Generator[Paper, None, None]:
        """迭代获取会议论文

        Args:
            conference: 会议名称
            year: 年份
            batch_size: 每批大小 (用于日志)

        Yields:
            Paper 对象
        """
        conference = conference.upper()
        self.logger.info(f"Iterating {conference} {year} papers...")

        try:
            venue = self._get_venue(conference, year)
            invitation = self._get_submission_invitation(conference, year)

            # get_all_notes 已经处理了分页，逐个 yield
            notes = self.or_client.get_all_notes(invitation=invitation)

            count = 0
            for note in notes or []:
                paper = self._parse_note_to_paper(note, conference, year)
                if paper:
                    yield paper
                    count += 1
                    if count % batch_size == 0:
                        self.logger.info(f"Processed {count} papers...")

            self.logger.info(f"Finished iterating {count} papers")

        except Exception as e:
            self.logger.error(f"Failed to iterate papers: {e}")

    def save_papers(
        self,
        papers: List[Paper],
        output_dir: Path,
        format: str = "json",
    ) -> Path:
        """保存论文列表到文件

        Args:
            papers: 论文列表
            output_dir: 输出目录
            format: 输出格式 (json, jsonl)

        Returns:
            输出文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == "jsonl":
            output_file = output_dir / f"papers_{timestamp}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for paper in papers:
                    f.write(json.dumps(paper.model_dump(mode="json"), ensure_ascii=False) + "\n")
        else:
            output_file = output_dir / f"papers_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    [paper.model_dump(mode="json") for paper in papers],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        self.logger.info(f"Saved {len(papers)} papers to {output_file}")
        return output_file

    def batch_fetch_conferences(
        self,
        conferences: List[str],
        years: List[int],
        output_dir: Optional[Path] = None,
        save_per_conference: bool = True,
    ) -> Dict[str, List[Paper]]:
        """批量获取多个会议年份的论文

        Args:
            conferences: 会议列表
            years: 年份列表
            output_dir: 输出目录 (可选)
            save_per_conference: 是否按会议分别保存

        Returns:
            会议-年份到论文列表的映射
        """
        results = {}

        for conf in conferences:
            for year in years:
                key = f"{conf}_{year}"
                self.logger.info(f"Fetching {key}...")

                try:
                    papers = self.get_accepted_papers(conf, year)
                    results[key] = papers

                    if output_dir and save_per_conference and papers:
                        conf_dir = Path(output_dir) / conf.lower()
                        conf_dir.mkdir(parents=True, exist_ok=True)
                        output_file = conf_dir / f"{year}.json"
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(
                                [p.model_dump(mode="json") for p in papers],
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        self.logger.info(f"Saved {len(papers)} papers to {output_file}")

                    # 请求间隔
                    time.sleep(self.request_delay)

                except Exception as e:
                    self.logger.error(f"Failed to fetch {key}: {e}")
                    results[key] = []

        # 保存汇总
        if output_dir and not save_per_conference:
            all_papers = []
            for papers in results.values():
                all_papers.extend(papers)
            if all_papers:
                self.save_papers(all_papers, Path(output_dir))

        return results

    def get_reviews(self, paper_id: str) -> List[Dict[str, Any]]:
        """获取论文的审稿意见

        Args:
            paper_id: 论文 ID

        Returns:
            审稿意见列表
        """
        self.logger.info(f"Fetching reviews for paper: {paper_id}")

        try:
            # 获取论文的所有回复
            replies = self.or_client.get_all_notes(
                forum=paper_id,
                replyto=paper_id,
            )

            reviews = []
            for reply in replies or []:
                content = reply.content

                # 检查是否是审稿意见
                if any(key in content for key in ["rating", "recommendation", "summary"]):
                    review = {
                        "id": reply.id,
                        "rating": self._extract_field(content, "rating"),
                        "confidence": self._extract_field(content, "confidence"),
                        "summary": self._extract_field(content, "summary"),
                        "strengths": self._extract_field(content, "strengths"),
                        "weaknesses": self._extract_field(content, "weaknesses"),
                        "questions": self._extract_field(content, "questions"),
                    }
                    reviews.append(review)

            self.logger.info(f"Found {len(reviews)} reviews")
            return reviews

        except Exception as e:
            self.logger.error(f"Failed to get reviews for {paper_id}: {e}")
            return []


# 便捷函数
def fetch_iclr_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """获取 ICLR 论文的便捷函数"""
    with OpenReviewScraper() as scraper:
        return scraper.get_accepted_papers("ICLR", year, limit=limit)


def fetch_icml_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """获取 ICML 论文的便捷函数"""
    with OpenReviewScraper() as scraper:
        return scraper.get_accepted_papers("ICML", year, limit=limit)


def fetch_neurips_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """获取 NeurIPS 论文的便捷函数"""
    with OpenReviewScraper() as scraper:
        return scraper.get_accepted_papers("NeurIPS", year, limit=limit)


if __name__ == "__main__":
    # 示例用法
    import logging
    logging.basicConfig(level=logging.INFO)

    scraper = OpenReviewScraper()

    # 获取 ICLR 2024 论文
    print("Fetching ICLR 2024 papers...")
    papers = scraper.get_conference_papers("ICLR", 2024, limit=10)

    for paper in papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(a.name for a in paper.authors[:3])}...")
        print(f"URL: {paper.url}")
        print(f"PDF: {paper.pdf_url}")

    # 获取 Best Papers
    print("\n\nFetching ICLR 2024 Best Papers...")
    best_papers = scraper.get_best_papers("ICLR", 2024)
    for paper in best_papers:
        print(f"- {paper.title}")

    scraper.close()
