"""
AI顶会论文爬取工具 - 主入口脚本

支持的命令:
- crawl: 爬取论文元数据
- download: 下载论文PDF
- convert: 转换PDF为Markdown
- index: 建立索引
- full: 执行完整流程

使用方式:
    python -m paper_crawler.main crawl -c ICLR -y 2024
    python -m paper_crawler.main download -i papers.json -o ./pdfs
    python -m paper_crawler.main full
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import click

from pipeline.paper_crawler.config import Config, Settings, get_settings
from pipeline.paper_crawler.models.paper import (
    Paper,
    Author,
    Conference,
    PaperType,
)
from pipeline.paper_crawler.scrapers import (
    ArxivScraper,
    get_openreview_scraper,
    get_acl_scraper,
    get_neurips_scraper,
    get_semantic_scholar_scraper,
)
from pipeline.paper_crawler.utils.logger import get_logger


# 版本信息
__version__ = "0.1.0"

# 支持的会议列表
SUPPORTED_CONFERENCES = {
    # OpenReview 平台
    "ICLR": {
        "platform": "openreview",
        "venue_template": "ICLR.cc/{year}/Conference",
        "conference_enum": Conference.ICLR,
        "description": "International Conference on Learning Representations",
    },
    "NeurIPS": {
        "platform": "openreview",
        "venue_template": "NeurIPS.cc/{year}/Conference",
        "conference_enum": Conference.NEURIPS,
        "description": "Neural Information Processing Systems",
    },
    # NeurIPS 专用爬虫
    "NeurIPS_PROC": {
        "platform": "neurips",
        "conference_enum": Conference.NEURIPS,
        "description": "NeurIPS Proceedings (official site)",
    },
    # ACL Anthology
    "ACL": {
        "platform": "acl",
        "venue_id": "acl",
        "conference_enum": Conference.ACL,
        "description": "Association for Computational Linguistics",
    },
    "EMNLP": {
        "platform": "acl",
        "venue_id": "emnlp",
        "conference_enum": Conference.EMNLP,
        "description": "Empirical Methods in Natural Language Processing",
    },
    "NAACL": {
        "platform": "acl",
        "venue_id": "naacl",
        "conference_enum": Conference.NAACL,
        "description": "North American Chapter of ACL",
    },
    # CVPR/ICCV/ECCV - 通过 Semantic Scholar
    "CVPR": {
        "platform": "semantic_scholar",
        "venue_query": "CVPR",
        "conference_enum": Conference.CVPR,
        "description": "Computer Vision and Pattern Recognition",
    },
    "ICCV": {
        "platform": "semantic_scholar",
        "venue_query": "ICCV",
        "conference_enum": Conference.ICCV,
        "description": "International Conference on Computer Vision",
    },
    "ECCV": {
        "platform": "semantic_scholar",
        "venue_query": "ECCV",
        "conference_enum": Conference.ECCV,
        "description": "European Conference on Computer Vision",
    },
    # ICML - OpenReview
    "ICML": {
        "platform": "openreview",
        "venue_template": "ICML.cc/{year}/Conference",
        "conference_enum": Conference.ICML,
        "description": "International Conference on Machine Learning",
    },
    # AI 综合会议
    "IJCAI": {
        "platform": "semantic_scholar",
        "venue_query": "IJCAI",
        "conference_enum": Conference.IJCAI,
        "description": "International Joint Conference on AI",
    },
    "AAAI": {
        "platform": "semantic_scholar",
        "venue_query": "AAAI",
        "conference_enum": Conference.AAAI,
        "description": "AAAI Conference on Artificial Intelligence",
    },
}

# 论文类型
PAPER_TYPES = ["best_paper", "oral", "spotlight", "poster", "all"]

# 初始化 logger
logger = get_logger("PaperCrawlerCLI")


def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """将 Paper 对象转换为字典"""
    data = paper.model_dump(mode="json")
    # 转换枚举为字符串
    if "conference" in data and data["conference"]:
        data["conference"] = data["conference"] if isinstance(data["conference"], str) else data["conference"]
    if "paper_type" in data and data["paper_type"]:
        data["paper_type"] = data["paper_type"] if isinstance(data["paper_type"], str) else data["paper_type"]
    return data


def paper_from_dict(data: Dict[str, Any]) -> Paper:
    """从字典创建 Paper 对象"""
    # 处理可能的枚举转换问题
    if "conference" in data and isinstance(data["conference"], str):
        try:
            data["conference"] = Conference(data["conference"])
        except ValueError:
            data["conference"] = Conference.OTHER

    if "paper_type" in data and isinstance(data["paper_type"], str):
        try:
            data["paper_type"] = PaperType(data["paper_type"])
        except ValueError:
            data["paper_type"] = PaperType.UNKNOWN

    return Paper.model_validate(data)


class CrawlerContext:
    """命令行上下文，存储共享状态"""

    def __init__(self):
        self.settings: Settings = get_settings()
        self.config: Config = Config()
        self.verbose: bool = False
        self.debug: bool = False

        # 延迟初始化的爬虫实例
        self._scrapers: Optional[Dict[str, Any]] = None
        self._downloader = None
        self._converter = None

    @property
    def scrapers(self) -> Dict[str, Any]:
        """延迟初始化爬虫"""
        if self._scrapers is None:
            self._scrapers = {}
            try:
                OpenReviewScraper = get_openreview_scraper()
                self._scrapers["openreview"] = OpenReviewScraper()
            except Exception as e:
                logger.warning(f"无法初始化 OpenReview 爬虫: {e}")

            try:
                ACLScraper = get_acl_scraper()
                self._scrapers["acl"] = ACLScraper()
            except Exception as e:
                logger.warning(f"无法初始化 ACL 爬虫: {e}")

            try:
                NeurIPSScraper = get_neurips_scraper()
                self._scrapers["neurips"] = NeurIPSScraper()
            except Exception as e:
                logger.warning(f"无法初始化 NeurIPS 爬虫: {e}")

            try:
                SemanticScholarScraper = get_semantic_scholar_scraper()
                self._scrapers["semantic_scholar"] = SemanticScholarScraper()
            except Exception as e:
                logger.warning(f"无法初始化 Semantic Scholar 爬虫: {e}")

            try:
                self._scrapers["arxiv"] = ArxivScraper()
            except Exception as e:
                logger.warning(f"无法初始化 arXiv 爬虫: {e}")

        return self._scrapers

    @property
    def downloader(self):
        """延迟初始化下载器"""
        if self._downloader is None:
            from pipeline.paper_crawler.downloaders.pdf_downloader import PDFDownloader
            self._downloader = PDFDownloader()
        return self._downloader

    @property
    def converter(self):
        """延迟初始化转换器"""
        if self._converter is None:
            from pipeline.paper_crawler.converters.pdf_to_md import PDFToMarkdownConverter
            self._converter = PDFToMarkdownConverter()
        return self._converter


pass_context = click.make_pass_decorator(CrawlerContext, ensure=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="详细输出模式")
@click.option("--debug", "-d", is_flag=True, help="调试模式")
@click.option("--config", "-C", type=click.Path(exists=True), help="配置文件路径")
@click.version_option(version=__version__, prog_name="paper-crawler")
@click.pass_context
def cli(ctx, verbose: bool, debug: bool, config: Optional[str]):
    """AI顶会论文爬取工具

    支持爬取 ICLR, ICML, NeurIPS, ACL, CVPR 等顶会论文。

    \b
    快速开始:
        paper-crawler crawl -c ICLR -y 2024
        paper-crawler full

    \b
    更多帮助:
        paper-crawler --help
        paper-crawler crawl --help
    """
    ctx.ensure_object(CrawlerContext)
    ctx.obj.verbose = verbose
    ctx.obj.debug = debug

    if verbose:
        click.echo(f"Paper Crawler v{__version__}")
        click.echo(f"Data directory: {ctx.obj.config.data_dir}")


@cli.command()
@click.option(
    "--conference", "-c",
    multiple=True,
    type=click.Choice(list(SUPPORTED_CONFERENCES.keys()), case_sensitive=False),
    help="会议名称 (可多选)"
)
@click.option(
    "--year", "-y",
    multiple=True,
    type=int,
    help="年份 (可多选，如 -y 2024 -y 2023)"
)
@click.option(
    "--paper-type", "-t",
    type=click.Choice(PAPER_TYPES),
    default="all",
    help="论文类型"
)
@click.option(
    "--limit", "-l",
    default=100,
    type=int,
    help="每个会议年份的最大论文数"
)
@click.option(
    "--query", "-q",
    default=None,
    help="搜索关键词 (可选，用于过滤论文)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="输出文件路径 (JSON格式)"
)
@pass_context
def crawl(
    ctx: CrawlerContext,
    conference: tuple,
    year: tuple,
    paper_type: str,
    limit: int,
    query: Optional[str],
    output: Optional[str]
):
    """爬取论文元数据

    \b
    示例:
        # 爬取 ICLR 2024 的论文
        paper-crawler crawl -c ICLR -y 2024

        # 爬取多个会议多年的论文
        paper-crawler crawl -c ICLR -c ICML -y 2024 -y 2023

        # 只爬取 best paper
        paper-crawler crawl -c NeurIPS -y 2024 -t best_paper
    """
    # 默认值
    conferences = list(conference) if conference else ["ICLR"]
    years = list(year) if year else [2024]

    all_papers: List[Paper] = []

    click.echo(f"开始爬取论文...")
    click.echo(f"  会议: {', '.join(conferences)}")
    click.echo(f"  年份: {', '.join(map(str, years))}")
    click.echo(f"  类型: {paper_type}")
    click.echo(f"  限制: {limit}/每会议年份")

    with click.progressbar(
        [(c, y) for c in conferences for y in years],
        label="爬取进度",
        show_pos=True,
    ) as bar:
        for conf, yr in bar:
            papers = crawl_conference(ctx, conf, yr, paper_type, limit, query)
            all_papers.extend(papers)
            if ctx.verbose:
                click.echo(f"\n  {conf} {yr}: 获取 {len(papers)} 篇论文")

    click.echo(f"\n总计获取 {len(all_papers)} 篇论文")

    # 保存结果
    if output:
        output_path = Path(output)
    else:
        output_path = ctx.config.metadata_dir / f"papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    papers_data = [paper_to_dict(paper) for paper in all_papers]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)

    click.echo(f"已保存到: {output_path}")

    # 显示统计
    show_statistics(all_papers)


def crawl_conference(
    ctx: CrawlerContext,
    conference: str,
    year: int,
    paper_type: str,
    limit: int,
    query: Optional[str] = None
) -> List[Paper]:
    """爬取特定会议的论文"""
    # Case-insensitive conference lookup
    conf_info = None
    conf_key = None
    for key, info in SUPPORTED_CONFERENCES.items():
        if key.upper() == conference.upper():
            conf_info = info
            conf_key = key
            break

    if not conf_info:
        logger.warning(f"不支持的会议: {conference}")
        return []

    platform = conf_info["platform"]
    scraper = ctx.scrapers.get(platform)

    if not scraper:
        logger.error(f"找不到爬虫: {platform}")
        return []

    papers = []

    try:
        if platform == "openreview":
            # OpenReviewScraper uses get_conference_papers with decision parameter
            decision = paper_type if paper_type != "all" else None
            papers = scraper.get_conference_papers(
                conference=conference.upper(),
                year=year,
                limit=limit,
                decision=decision
            )

        elif platform == "acl":
            venue_id = conf_info.get("venue_id", conference.lower())
            papers = scraper.get_venue_papers(venue_id, year=year, limit=limit)

        elif platform == "neurips":
            papers = scraper.get_year_papers(year)[:limit]

        elif platform == "semantic_scholar":
            venue_query = conf_info.get("venue_query", conference)
            search_query = f"{venue_query} {year}"
            if query:
                search_query = f"{query} {search_query}"
            papers = scraper.search(search_query, limit=limit)

        else:
            logger.warning(f"未知平台: {platform}")

    except Exception as e:
        logger.error(f"爬取 {conference} {year} 失败: {e}")
        if ctx.debug:
            import traceback
            traceback.print_exc()

    # 过滤论文类型
    if paper_type != "all" and papers:
        papers = filter_by_type(papers, paper_type)

    return papers


def filter_by_type(papers: List[Paper], paper_type: str) -> List[Paper]:
    """根据论文类型过滤"""
    filtered = []

    type_mapping = {
        "best_paper": PaperType.BEST_PAPER,
        "oral": PaperType.ORAL,
        "spotlight": PaperType.SPOTLIGHT,
        "poster": PaperType.POSTER,
    }

    target_type = type_mapping.get(paper_type)

    for paper in papers:
        if target_type:
            if paper.paper_type == target_type:
                filtered.append(paper)
        else:
            filtered.append(paper)

    return filtered if filtered else papers  # 如果过滤后为空，返回原始列表


def show_statistics(papers: List[Paper]):
    """显示论文统计信息"""
    if not papers:
        return

    click.echo("\n统计信息:")

    # 按来源统计
    source_counts: Dict[str, int] = {}
    for paper in papers:
        source = paper.source if paper.source else "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    click.echo("  按来源:")
    for source, count in sorted(source_counts.items()):
        click.echo(f"    {source}: {count}")

    # 有 PDF 链接的数量
    pdf_count = sum(1 for p in papers if p.pdf_url)
    click.echo(f"  有PDF链接: {pdf_count}/{len(papers)}")


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="论文元数据JSON文件"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="PDF输出目录"
)
@click.option(
    "--concurrent", "-n",
    default=5,
    type=int,
    help="并发下载数"
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=True,
    help="跳过已存在的文件"
)
@pass_context
def download(
    ctx: CrawlerContext,
    input: str,
    output: Optional[str],
    concurrent: int,
    skip_existing: bool
):
    """下载论文PDF

    \b
    示例:
        paper-crawler download -i papers.json -o ./pdfs
        paper-crawler download -i papers.json -n 10
    """
    input_path = Path(input)
    output_dir = Path(output) if output else ctx.config.pdf_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"读取论文列表: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    papers = [paper_from_dict(p) for p in papers_data]
    papers_with_pdf = [p for p in papers if p.pdf_url]

    click.echo(f"共 {len(papers)} 篇论文，其中 {len(papers_with_pdf)} 篇有PDF链接")

    if not papers_with_pdf:
        click.echo("没有可下载的论文")
        return

    # 执行下载
    downloaded = 0
    failed = 0
    skipped = 0

    # 存储下载路径
    pdf_paths: Dict[str, str] = {}

    with click.progressbar(
        papers_with_pdf,
        label="下载进度",
        show_pos=True,
    ) as bar:
        for paper in bar:
            try:
                pdf_path = output_dir / f"{paper.id}.pdf"

                if skip_existing and pdf_path.exists():
                    skipped += 1
                    pdf_paths[paper.id] = str(pdf_path)
                    continue

                result = ctx.downloader.download(paper, output_dir=output_dir)
                if result:
                    pdf_paths[paper.id] = str(result)
                    downloaded += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                if ctx.debug:
                    logger.error(f"下载失败 {paper.id}: {e}")

    click.echo(f"\n下载完成:")
    click.echo(f"  成功: {downloaded}")
    click.echo(f"  失败: {failed}")
    click.echo(f"  跳过: {skipped}")

    # 更新元数据并保存
    for p_data in papers_data:
        if p_data.get("id") in pdf_paths:
            p_data["pdf_path"] = pdf_paths[p_data["id"]]

    output_metadata = input_path.parent / f"{input_path.stem}_downloaded.json"
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)

    click.echo(f"已更新元数据: {output_metadata}")


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="论文元数据JSON文件 (需包含pdf_path)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Markdown输出目录"
)
@click.option(
    "--method", "-m",
    type=click.Choice(["marker", "markitdown", "pymupdf", "nougat"]),
    default="marker",
    help="PDF转换方法"
)
@pass_context
def convert(
    ctx: CrawlerContext,
    input: str,
    output: Optional[str],
    method: str
):
    """转换PDF为Markdown

    \b
    示例:
        paper-crawler convert -i papers_downloaded.json
        paper-crawler convert -i papers.json -m marker
    """
    input_path = Path(input)
    output_dir = Path(output) if output else ctx.config.markdown_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"读取论文列表: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    # 过滤有本地PDF的论文
    papers_with_pdf_data = [
        p for p in papers_data
        if p.get("pdf_path") and Path(p.get("pdf_path", "")).exists()
    ]

    click.echo(f"共 {len(papers_data)} 篇论文，其中 {len(papers_with_pdf_data)} 篇有本地PDF")

    if not papers_with_pdf_data:
        click.echo("没有可转换的PDF文件")
        return

    converted = 0
    failed = 0
    md_paths: Dict[str, str] = {}

    # Temporarily override converter output_dir if specified
    if output_dir:
        ctx.converter.output_dir = output_dir

    with click.progressbar(
        papers_with_pdf_data,
        label="转换进度",
        show_pos=True,
    ) as bar:
        for p_data in bar:
            try:
                pdf_path = Path(p_data.get("pdf_path", ""))
                paper_id = p_data.get("id", pdf_path.stem)

                # Get conference and year from data
                conference = p_data.get("conference", "")
                if isinstance(conference, dict):
                    conference = conference.get("value", "")
                year = p_data.get("year")

                # Use convert_single directly with all the info from JSON
                result = ctx.converter.convert_single(
                    pdf_path=pdf_path,
                    conference=conference,
                    year=year,
                    paper_id=paper_id,
                )
                if result:
                    md_paths[paper_id] = str(result)
                    converted += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if ctx.debug:
                    logger.error(f"转换失败 {p_data.get('id', 'unknown')}: {e}")

    click.echo(f"\n转换完成:")
    click.echo(f"  成功: {converted}")
    click.echo(f"  失败: {failed}")

    # 更新元数据
    for p_data in papers_data:
        if p_data.get("id") in md_paths:
            p_data["markdown_path"] = md_paths[p_data["id"]]

    output_metadata = input_path.parent / f"{input_path.stem}_converted.json"
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)

    click.echo(f"已更新元数据: {output_metadata}")


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    help="论文目录或元数据JSON文件"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="索引输出目录"
)
@click.option(
    "--index-type", "-t",
    type=click.Choice(["faiss", "chroma", "qdrant", "simple"]),
    default="simple",
    help="索引类型"
)
@pass_context
def index(
    ctx: CrawlerContext,
    input: Optional[str],
    output: Optional[str],
    index_type: str
):
    """建立论文索引

    \b
    示例:
        paper-crawler index -i papers_converted.json
        paper-crawler index -t faiss
    """
    input_path = Path(input) if input else ctx.config.markdown_dir
    output_dir = Path(output) if output else ctx.config.data_dir / "index"
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"建立索引...")
    click.echo(f"  输入: {input_path}")
    click.echo(f"  输出: {output_dir}")
    click.echo(f"  类型: {index_type}")

    # 收集 Markdown 文件
    if input_path.is_file() and input_path.suffix == ".json":
        # 从元数据读取
        with open(input_path, "r", encoding="utf-8") as f:
            papers_data = json.load(f)
        md_files = [
            Path(p.get("markdown_path", ""))
            for p in papers_data
            if p.get("markdown_path") and Path(p.get("markdown_path", "")).exists()
        ]
    else:
        # 扫描目录
        md_files = list(input_path.glob("**/*.md"))

    click.echo(f"找到 {len(md_files)} 个Markdown文件")

    if not md_files:
        click.echo("没有可索引的文件")
        return

    # 创建简单索引
    if index_type == "simple":
        index_data = []
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                index_data.append({
                    "id": md_file.stem,
                    "path": str(md_file),
                    "title": extract_title(content),
                    "content_preview": content[:500],
                    "content_length": len(content),
                })
            except Exception as e:
                logger.error(f"处理失败 {md_file}: {e}")

        index_file = output_dir / "simple_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        click.echo(f"已创建简单索引: {index_file} ({len(index_data)} 条)")
    else:
        click.echo(f"索引类型 '{index_type}' 需要额外依赖，请确保已安装相关库")
        # TODO: 实现其他索引类型

    click.echo("索引建立完成")


def extract_title(content: str) -> str:
    """从 Markdown 内容提取标题"""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


@cli.command()
@click.option(
    "--conferences", "-c",
    default="ICLR,ICML,NeurIPS,ACL,CVPR",
    help="会议列表 (逗号分隔)"
)
@click.option(
    "--years", "-y",
    default="2024,2023,2022",
    help="年份列表 (逗号分隔)"
)
@click.option(
    "--paper-type", "-t",
    type=click.Choice(PAPER_TYPES),
    default="oral",
    help="论文类型 (建议从 oral 开始，含金量高)"
)
@click.option(
    "--limit", "-l",
    default=50,
    type=int,
    help="每个会议年份的最大论文数"
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="跳过PDF下载"
)
@click.option(
    "--skip-convert",
    is_flag=True,
    help="跳过MD转换"
)
@click.option(
    "--skip-index",
    is_flag=True,
    help="跳过建立索引"
)
@pass_context
def full(
    ctx: CrawlerContext,
    conferences: str,
    years: str,
    paper_type: str,
    limit: int,
    skip_download: bool,
    skip_convert: bool,
    skip_index: bool
):
    """执行完整爬取流程

    \b
    一键完成: 爬取 -> 下载 -> 转换 -> 索引

    \b
    示例:
        # 默认配置 (推荐)
        paper-crawler full

        # 自定义会议和年份
        paper-crawler full -c "ICLR,ICML" -y "2024,2023"

        # 只爬取 best paper (高质量)
        paper-crawler full -t best_paper

        # 大规模爬取
        paper-crawler full -c "ICLR,ICML,NeurIPS,ACL,CVPR" -y "2024,2023,2022,2021,2020" -l 200
    """
    asyncio.run(run_full_pipeline(
        ctx,
        conferences.split(","),
        [int(y.strip()) for y in years.split(",")],
        paper_type,
        limit,
        skip_download,
        skip_convert,
        skip_index
    ))


async def run_full_pipeline(
    ctx: CrawlerContext,
    conferences: List[str],
    years: List[int],
    paper_type: str,
    limit: int,
    skip_download: bool = False,
    skip_convert: bool = False,
    skip_index: bool = False
):
    """运行完整流水线"""
    click.echo("=" * 60)
    click.echo("AI顶会论文爬取工具 - 完整流程")
    click.echo("=" * 60)

    start_time = datetime.now()

    # 步骤1: 爬取论文
    click.echo("\n[1/4] 爬取论文元数据...")
    all_papers: List[Paper] = []

    total_tasks = len(conferences) * len(years)
    current = 0

    for conf in conferences:
        conf = conf.strip().upper()
        if conf not in SUPPORTED_CONFERENCES:
            click.echo(f"  跳过不支持的会议: {conf}")
            continue

        for year in years:
            current += 1
            click.echo(f"  [{current}/{total_tasks}] 爬取 {conf} {year}...")

            try:
                papers = crawl_conference(ctx, conf, year, paper_type, limit)
                all_papers.extend(papers)
                click.echo(f"    获取 {len(papers)} 篇论文")
            except Exception as e:
                click.echo(f"    失败: {e}")

    click.echo(f"\n  总计: {len(all_papers)} 篇论文")

    if not all_papers:
        click.echo("没有获取到论文，退出")
        return

    # 保存元数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = ctx.config.metadata_dir / f"papers_{timestamp}.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    papers_data = [paper_to_dict(p) for p in all_papers]
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)

    click.echo(f"  已保存元数据: {metadata_file}")

    # 步骤2: 下载PDF
    pdf_paths: Dict[str, str] = {}

    if not skip_download:
        click.echo("\n[2/4] 下载PDF文件...")
        papers_with_pdf = [p for p in all_papers if p.pdf_url]
        click.echo(f"  可下载: {len(papers_with_pdf)} 篇")

        downloaded = 0
        for i, paper in enumerate(papers_with_pdf):
            try:
                result = ctx.downloader.download(paper)
                if result:
                    pdf_paths[paper.id] = str(result)
                    downloaded += 1
                if (i + 1) % 10 == 0:
                    click.echo(f"    进度: {i+1}/{len(papers_with_pdf)}")
            except Exception as e:
                if ctx.debug:
                    logger.error(f"下载失败 {paper.id}: {e}")

        click.echo(f"  已下载: {downloaded} 篇")

        # 更新元数据
        for p_data in papers_data:
            if p_data.get("id") in pdf_paths:
                p_data["pdf_path"] = pdf_paths[p_data["id"]]

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)
    else:
        click.echo("\n[2/4] 跳过PDF下载")

    # 步骤3: 转换为Markdown
    md_paths: Dict[str, str] = {}

    if not skip_convert:
        click.echo("\n[3/4] 转换为Markdown...")

        # 重新获取有本地PDF的论文
        papers_with_local_pdf = [
            (p_data, pdf_paths.get(p_data.get("id", "")))
            for p_data in papers_data
            if p_data.get("id") in pdf_paths or (
                p_data.get("pdf_path") and Path(p_data.get("pdf_path", "")).exists()
            )
        ]

        click.echo(f"  可转换: {len(papers_with_local_pdf)} 篇")

        converted = 0
        for i, (p_data, pdf_path) in enumerate(papers_with_local_pdf):
            try:
                paper = paper_from_dict(p_data)
                result = ctx.converter.convert_paper(paper)
                if result:
                    md_paths[paper.id] = str(result)
                    converted += 1
                if (i + 1) % 10 == 0:
                    click.echo(f"    进度: {i+1}/{len(papers_with_local_pdf)}")
            except Exception as e:
                if ctx.debug:
                    logger.error(f"转换失败 {p_data.get('id', 'unknown')}: {e}")

        click.echo(f"  已转换: {converted} 篇")

        # 更新元数据
        for p_data in papers_data:
            if p_data.get("id") in md_paths:
                p_data["markdown_path"] = md_paths[p_data["id"]]

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(papers_data, f, ensure_ascii=False, indent=2, default=str)
    else:
        click.echo("\n[3/4] 跳过MD转换")

    # 步骤4: 建立索引
    if not skip_index:
        click.echo("\n[4/4] 建立索引...")

        md_files = [
            Path(p_data.get("markdown_path", ""))
            for p_data in papers_data
            if p_data.get("markdown_path") and Path(p_data.get("markdown_path", "")).exists()
        ]

        if md_files:
            index_dir = ctx.config.data_dir / "index"
            index_dir.mkdir(parents=True, exist_ok=True)

            index_data = []
            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                    index_data.append({
                        "id": md_file.stem,
                        "path": str(md_file),
                        "title": extract_title(content),
                        "content_length": len(content),
                    })
                except Exception:
                    pass

            index_file = index_dir / f"index_{timestamp}.json"
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

            click.echo(f"  已创建索引: {index_file} ({len(index_data)} 条)")
        else:
            click.echo("  没有可索引的Markdown文件")
    else:
        click.echo("\n[4/4] 跳过建立索引")

    # 完成
    elapsed = datetime.now() - start_time
    click.echo("\n" + "=" * 60)
    click.echo("完成!")
    click.echo(f"  总耗时: {elapsed}")
    click.echo(f"  论文数: {len(all_papers)}")
    click.echo(f"  元数据: {metadata_file}")
    click.echo("=" * 60)


@cli.command()
def list_conferences():
    """列出支持的会议"""
    click.echo("支持的会议列表:")
    click.echo("-" * 60)

    for name, info in SUPPORTED_CONFERENCES.items():
        desc = info.get("description", "")
        platform = info.get("platform", "")
        click.echo(f"  {name:12} - {desc}")
        click.echo(f"               平台: {platform}")

    click.echo("-" * 60)
    click.echo(f"共 {len(SUPPORTED_CONFERENCES)} 个会议")


@cli.command()
@pass_context
def status(ctx: CrawlerContext):
    """显示当前状态"""
    click.echo("Paper Crawler 状态")
    click.echo("-" * 40)

    # 配置信息
    click.echo(f"数据目录: {ctx.config.data_dir}")
    click.echo(f"PDF目录: {ctx.config.pdf_dir}")
    click.echo(f"Markdown目录: {ctx.config.markdown_dir}")
    click.echo(f"元数据目录: {ctx.config.metadata_dir}")

    # 统计信息
    click.echo("\n文件统计:")

    pdf_count = len(list(ctx.config.pdf_dir.glob("**/*.pdf"))) if ctx.config.pdf_dir.exists() else 0
    md_count = len(list(ctx.config.markdown_dir.glob("**/*.md"))) if ctx.config.markdown_dir.exists() else 0
    json_count = len(list(ctx.config.metadata_dir.glob("**/*.json"))) if ctx.config.metadata_dir.exists() else 0

    click.echo(f"  PDF文件: {pdf_count}")
    click.echo(f"  Markdown文件: {md_count}")
    click.echo(f"  元数据文件: {json_count}")


@cli.command()
@click.argument("query")
@click.option(
    "--source", "-s",
    type=click.Choice(["arxiv", "semantic_scholar", "openreview"]),
    default="arxiv",
    help="搜索源"
)
@click.option(
    "--limit", "-l",
    default=10,
    type=int,
    help="最大结果数"
)
@pass_context
def search(ctx: CrawlerContext, query: str, source: str, limit: int):
    """快速搜索论文

    \b
    示例:
        paper-crawler search "transformer attention"
        paper-crawler search "LLM" -s semantic_scholar -l 20
    """
    click.echo(f"搜索: {query}")
    click.echo(f"来源: {source}, 限制: {limit}")
    click.echo("-" * 40)

    scraper = ctx.scrapers.get(source)
    if not scraper:
        click.echo(f"不支持的来源: {source}")
        return

    try:
        papers = scraper.search(query, limit=limit)

        if not papers:
            click.echo("未找到结果")
            return

        for i, paper in enumerate(papers, 1):
            click.echo(f"\n[{i}] {paper.title}")
            if paper.authors:
                authors = ", ".join(str(a) for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += " et al."
                click.echo(f"    作者: {authors}")
            if hasattr(paper, 'paper_page_url') and paper.paper_page_url:
                click.echo(f"    链接: {paper.paper_page_url}")
            if paper.pdf_url:
                click.echo(f"    PDF: {paper.pdf_url}")

    except Exception as e:
        click.echo(f"搜索失败: {e}")
        if ctx.debug:
            import traceback
            traceback.print_exc()


def main():
    """主入口函数"""
    cli()


if __name__ == "__main__":
    main()
