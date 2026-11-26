"""
PaperPipeline - 论文处理工作流

使用 LangGraph 实现论文爬取、下载、转换、索引的自动化流程。

功能特性:
- 检查点(断点续传): 使用 SQLiteSaver 持久化状态
- 错误恢复: 自动重试失败的操作
- 并行处理: 使用 Send 实现并行下载/转换
- 日志记录: 详细的操作日志

工作流程:
1. GetPaperList - 获取论文列表
2. DownloadPDF - 下载论文PDF
3. ConvertToMD - 转换为Markdown
4. BuildIndex - 建立索引
5. SaveMetadata - 保存元数据
"""

from typing import TypedDict, List, Optional, Annotated, Literal, Any, Dict, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from operator import add
from datetime import datetime
import json
import asyncio
import hashlib
import time

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy, Send, interrupt, Command

# Project imports
from pipeline.paper_crawler.models.paper import Paper, CrawlStatus, PaperSource
from pipeline.paper_crawler.scrapers import (
    BaseScraper,
    get_arxiv_scraper,
    get_semantic_scholar_scraper,
    get_openreview_scraper,
    get_acl_scraper,
    get_neurips_scraper,
)
from pipeline.paper_crawler.downloaders.pdf_downloader import PDFDownloader
from pipeline.paper_crawler.converters.pdf_to_md import PDFToMarkdownConverter
from pipeline.paper_crawler.utils.logger import get_logger, ProgressLogger
from pipeline.paper_crawler.config import config, settings

# 延迟获取爬虫类
def _get_scraper_classes():
    """延迟获取爬虫类，避免循环导入"""
    return {
        "arxiv": get_arxiv_scraper(),
        "semantic_scholar": get_semantic_scholar_scraper(),
        "openreview": get_openreview_scraper(),
        "acl": get_acl_scraper(),
        "neurips": get_neurips_scraper(),
    }


# =============================================================================
# 状态定义
# =============================================================================

class PaperState(TypedDict):
    """工作流主状态

    使用 Annotated 配合 add 函数实现列表自动合并。
    """
    # 输入参数
    conference: str                           # 会议名称 (e.g., "ICLR", "NeurIPS")
    year: int                                 # 年份
    source: str                               # 数据源 (arxiv, openreview, etc.)
    query: Optional[str]                      # 搜索关键词 (可选)
    max_papers: int                           # 最大论文数量

    # 处理选项
    download_pdf: bool                        # 是否下载PDF
    convert_to_md: bool                       # 是否转换为Markdown
    build_index: bool                         # 是否建立索引

    # 状态数据
    papers: List[Dict[str, Any]]              # 论文列表 (存储为字典以便序列化)
    downloaded: Annotated[List[str], add]     # 已下载的论文ID
    converted: Annotated[List[str], add]      # 已转换的论文ID
    indexed: Annotated[List[str], add]        # 已索引的论文ID
    errors: Annotated[List[str], add]         # 错误信息

    # 进度追踪
    current_step: str                         # 当前步骤
    progress: float                           # 进度百分比 (0-100)
    start_time: Optional[str]                 # 开始时间
    end_time: Optional[str]                   # 结束时间


class SinglePaperState(TypedDict):
    """单篇论文处理状态 (用于并行处理)"""
    paper: Dict[str, Any]                     # 论文数据
    downloaded: Annotated[List[str], add]     # 下载结果
    converted: Annotated[List[str], add]      # 转换结果
    indexed: Annotated[List[str], add]        # 索引结果
    errors: Annotated[List[str], add]         # 错误信息


# =============================================================================
# 辅助类
# =============================================================================

@dataclass
class PipelineConfig:
    """管道配置"""
    max_concurrent_downloads: int = 5
    max_concurrent_conversions: int = 3
    download_timeout: int = 60
    retry_max_attempts: int = 3
    retry_initial_interval: float = 1.0
    retry_backoff_factor: float = 2.0
    checkpoint_dir: Optional[Path] = None


class PipelineContext:
    """管道上下文 - 存储共享资源"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.logger = get_logger("PaperPipeline")
        self.progress_logger: Optional[ProgressLogger] = None

        # 初始化组件 (延迟加载)
        self._scrapers: Dict[str, BaseScraper] = {}
        self._downloader: Optional[PDFDownloader] = None
        self._converter: Optional[PDFToMarkdownConverter] = None

    @property
    def scrapers(self) -> Dict[str, BaseScraper]:
        if not self._scrapers:
            scraper_classes = _get_scraper_classes()
            self._scrapers = {
                name: cls() for name, cls in scraper_classes.items()
            }
        return self._scrapers

    @property
    def downloader(self) -> PDFDownloader:
        if self._downloader is None:
            self._downloader = PDFDownloader(
                max_concurrent=self.config.max_concurrent_downloads,
                timeout=self.config.download_timeout,
            )
        return self._downloader

    @property
    def converter(self) -> PDFToMarkdownConverter:
        if self._converter is None:
            self._converter = PDFToMarkdownConverter()
        return self._converter


# 全局上下文实例
_context: Optional[PipelineContext] = None


def get_context() -> PipelineContext:
    """获取管道上下文"""
    global _context
    if _context is None:
        _context = PipelineContext()
    return _context


def set_context(ctx: PipelineContext):
    """设置管道上下文"""
    global _context
    _context = ctx


# =============================================================================
# 工具函数
# =============================================================================

def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """将Paper对象转换为可序列化的字典"""
    return paper.to_dict()


def dict_to_paper(data: Dict[str, Any]) -> Paper:
    """从字典恢复Paper对象"""
    return Paper.from_dict(data)


def generate_task_id(conference: str, year: int, source: str) -> str:
    """生成任务ID"""
    content = f"{conference}-{year}-{source}-{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# 节点函数
# =============================================================================

def get_paper_list(state: PaperState) -> dict:
    """获取论文列表节点

    根据会议/年份/数据源获取论文列表。
    """
    ctx = get_context()
    logger = ctx.logger

    conference = state.get("conference", "")
    year = state.get("year", datetime.now().year)
    source = state.get("source", "arxiv")
    query = state.get("query")
    max_papers = state.get("max_papers", 100)

    logger.info(f"[GetPaperList] 开始获取论文列表 | 会议: {conference} | 年份: {year} | 数据源: {source}")

    # 获取对应的爬虫
    scraper = ctx.scrapers.get(source)
    if not scraper:
        error_msg = f"不支持的数据源: {source}"
        logger.error(f"[GetPaperList] {error_msg}")
        return {
            "papers": [],
            "errors": [error_msg],
            "current_step": "get_papers",
            "progress": 10.0,
        }

    try:
        # 构建搜索查询
        search_query = query if query else f"{conference} {year}"

        # 获取论文列表
        papers = scraper.search(search_query, limit=max_papers)

        # 转换为可序列化的字典
        papers_data = [paper_to_dict(p) for p in papers]

        logger.info(f"[GetPaperList] 成功获取 {len(papers_data)} 篇论文")

        # 初始化进度记录器
        ctx.progress_logger = ProgressLogger(
            task_name=f"{conference}-{year}",
            total=len(papers_data),
        )

        return {
            "papers": papers_data,
            "current_step": "get_papers",
            "progress": 20.0,
            "start_time": datetime.now().isoformat(),
        }

    except Exception as e:
        error_msg = f"获取论文列表失败: {str(e)}"
        logger.error(f"[GetPaperList] {error_msg}")
        return {
            "papers": [],
            "errors": [error_msg],
            "current_step": "get_papers",
            "progress": 10.0,
        }


def download_papers(state: PaperState) -> dict:
    """下载论文PDF节点

    批量下载论文PDF文件。
    """
    ctx = get_context()
    logger = ctx.logger

    # 检查是否需要下载
    if not state.get("download_pdf", True):
        logger.info("[DownloadPDF] 跳过下载步骤")
        return {
            "current_step": "download",
            "progress": 50.0,
        }

    papers_data = state.get("papers", [])
    if not papers_data:
        logger.warning("[DownloadPDF] 没有论文需要下载")
        return {
            "current_step": "download",
            "progress": 50.0,
        }

    logger.info(f"[DownloadPDF] 开始下载 {len(papers_data)} 篇论文")

    downloaded_ids = []
    errors = []

    for i, paper_data in enumerate(papers_data):
        try:
            paper = dict_to_paper(paper_data)

            # 下载PDF
            path = ctx.downloader.download(paper)

            if path:
                paper.pdf_path = str(path)
                paper.crawl_status = CrawlStatus.COMPLETED
                downloaded_ids.append(paper.id)

                # 更新papers列表中的数据
                papers_data[i] = paper_to_dict(paper)

                logger.info(f"[DownloadPDF] 已下载: {paper.title[:50]}...")
            else:
                paper.crawl_status = CrawlStatus.FAILED
                papers_data[i] = paper_to_dict(paper)
                errors.append(f"下载失败: {paper.id}")

        except Exception as e:
            error_msg = f"下载异常 [{paper_data.get('id', 'unknown')}]: {str(e)}"
            logger.error(f"[DownloadPDF] {error_msg}")
            errors.append(error_msg)

        # 更新进度
        if ctx.progress_logger:
            ctx.progress_logger.update(completed=1 if downloaded_ids else 0)

    logger.info(f"[DownloadPDF] 下载完成 | 成功: {len(downloaded_ids)} | 失败: {len(errors)}")

    return {
        "papers": papers_data,
        "downloaded": downloaded_ids,
        "errors": errors,
        "current_step": "download",
        "progress": 50.0,
    }


def convert_to_markdown(state: PaperState) -> dict:
    """转换为Markdown节点

    将已下载的PDF转换为Markdown格式。
    """
    ctx = get_context()
    logger = ctx.logger

    # 检查是否需要转换
    if not state.get("convert_to_md", True):
        logger.info("[ConvertToMD] 跳过转换步骤")
        return {
            "current_step": "convert",
            "progress": 70.0,
        }

    papers_data = state.get("papers", [])
    downloaded_ids = set(state.get("downloaded", []))

    if not downloaded_ids:
        logger.warning("[ConvertToMD] 没有已下载的论文需要转换")
        return {
            "current_step": "convert",
            "progress": 70.0,
        }

    logger.info(f"[ConvertToMD] 开始转换 {len(downloaded_ids)} 篇论文")

    converted_ids = []
    errors = []

    for i, paper_data in enumerate(papers_data):
        paper_id = paper_data.get("id", "")

        # 只处理已下载的论文
        if paper_id not in downloaded_ids:
            continue

        try:
            paper = dict_to_paper(paper_data)

            if not paper.pdf_path:
                continue

            # 转换为Markdown
            md_path = ctx.converter.convert_paper(paper)

            if md_path:
                paper.markdown_path = str(md_path)
                converted_ids.append(paper.id)
                papers_data[i] = paper_to_dict(paper)

                logger.info(f"[ConvertToMD] 已转换: {paper.title[:50]}...")
            else:
                errors.append(f"转换失败: {paper.id}")

        except Exception as e:
            error_msg = f"转换异常 [{paper_id}]: {str(e)}"
            logger.error(f"[ConvertToMD] {error_msg}")
            errors.append(error_msg)

    logger.info(f"[ConvertToMD] 转换完成 | 成功: {len(converted_ids)} | 失败: {len(errors)}")

    return {
        "papers": papers_data,
        "converted": converted_ids,
        "errors": errors,
        "current_step": "convert",
        "progress": 70.0,
    }


def build_index(state: PaperState) -> dict:
    """建立索引节点

    为转换后的Markdown建立搜索索引。
    """
    ctx = get_context()
    logger = ctx.logger

    # 检查是否需要建立索引
    if not state.get("build_index", True):
        logger.info("[BuildIndex] 跳过索引步骤")
        return {
            "current_step": "index",
            "progress": 85.0,
        }

    converted_ids = set(state.get("converted", []))

    if not converted_ids:
        logger.warning("[BuildIndex] 没有已转换的论文需要索引")
        return {
            "current_step": "index",
            "progress": 85.0,
        }

    logger.info(f"[BuildIndex] 开始索引 {len(converted_ids)} 篇论文")

    indexed_ids = []
    errors = []
    papers_data = state.get("papers", [])

    # 索引数据存储路径
    index_dir = settings.storage.data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "paper_index.json"

    # 加载现有索引
    existing_index = {}
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                existing_index = json.load(f)
        except Exception:
            pass

    for paper_data in papers_data:
        paper_id = paper_data.get("id", "")

        if paper_id not in converted_ids:
            continue

        try:
            paper = dict_to_paper(paper_data)

            # 构建索引条目
            index_entry = {
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.author_names,
                "year": paper.year,
                "source": paper.source.value if isinstance(paper.source, PaperSource) else paper.source,
                "pdf_path": paper.pdf_path,
                "markdown_path": paper.markdown_path,
                "keywords": paper.metadata.keywords if paper.metadata else [],
                "indexed_at": datetime.now().isoformat(),
            }

            existing_index[paper.id] = index_entry
            indexed_ids.append(paper.id)

            logger.info(f"[BuildIndex] 已索引: {paper.title[:50]}...")

        except Exception as e:
            error_msg = f"索引异常 [{paper_id}]: {str(e)}"
            logger.error(f"[BuildIndex] {error_msg}")
            errors.append(error_msg)

    # 保存索引
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(existing_index, f, ensure_ascii=False, indent=2)
        logger.info(f"[BuildIndex] 索引已保存到: {index_file}")
    except Exception as e:
        error_msg = f"保存索引失败: {str(e)}"
        logger.error(f"[BuildIndex] {error_msg}")
        errors.append(error_msg)

    logger.info(f"[BuildIndex] 索引完成 | 成功: {len(indexed_ids)} | 失败: {len(errors)}")

    return {
        "indexed": indexed_ids,
        "errors": errors,
        "current_step": "index",
        "progress": 85.0,
    }


def save_metadata(state: PaperState) -> dict:
    """保存元数据节点

    将所有论文的元数据保存到文件。
    """
    ctx = get_context()
    logger = ctx.logger

    papers_data = state.get("papers", [])

    if not papers_data:
        logger.warning("[SaveMetadata] 没有论文元数据需要保存")
        return {
            "current_step": "save_metadata",
            "progress": 100.0,
            "end_time": datetime.now().isoformat(),
        }

    logger.info(f"[SaveMetadata] 开始保存 {len(papers_data)} 篇论文的元数据")

    errors = []
    saved_count = 0

    # 确保元数据目录存在
    metadata_dir = settings.storage.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for paper_data in papers_data:
        paper_id = paper_data.get("id", "")

        try:
            # 保存单个论文的元数据
            metadata_path = metadata_dir / f"{paper_id}.json"

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)

            saved_count += 1

        except Exception as e:
            error_msg = f"保存元数据失败 [{paper_id}]: {str(e)}"
            logger.error(f"[SaveMetadata] {error_msg}")
            errors.append(error_msg)

    # 保存汇总信息
    summary = {
        "conference": state.get("conference", ""),
        "year": state.get("year", 0),
        "source": state.get("source", ""),
        "total_papers": len(papers_data),
        "downloaded": len(state.get("downloaded", [])),
        "converted": len(state.get("converted", [])),
        "indexed": len(state.get("indexed", [])),
        "errors": len(state.get("errors", [])),
        "start_time": state.get("start_time"),
        "end_time": datetime.now().isoformat(),
    }

    summary_path = metadata_dir / f"summary_{state.get('conference', 'unknown')}_{state.get('year', 0)}.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        errors.append(f"保存汇总信息失败: {str(e)}")

    logger.info(f"[SaveMetadata] 元数据保存完成 | 成功: {saved_count} | 失败: {len(errors)}")

    # 完成进度记录
    if ctx.progress_logger:
        ctx.progress_logger.finish(f"已保存 {saved_count} 篇论文元数据")

    return {
        "errors": errors,
        "current_step": "save_metadata",
        "progress": 100.0,
        "end_time": datetime.now().isoformat(),
    }


# =============================================================================
# 条件判断函数
# =============================================================================

def should_download(state: PaperState) -> str:
    """判断是否需要下载"""
    papers = state.get("papers", [])
    download_pdf = state.get("download_pdf", True)

    if not papers:
        return "save_metadata"

    if download_pdf:
        return "download"
    else:
        return "save_metadata"


def should_convert(state: PaperState) -> str:
    """判断是否需要转换"""
    downloaded = state.get("downloaded", [])
    convert_to_md = state.get("convert_to_md", True)

    if not downloaded or not convert_to_md:
        return "save_metadata"

    return "convert"


def should_index(state: PaperState) -> str:
    """判断是否需要建立索引"""
    converted = state.get("converted", [])
    build_index_flag = state.get("build_index", True)

    if not converted or not build_index_flag:
        return "save_metadata"

    return "index"


def check_errors(state: PaperState) -> str:
    """检查是否有严重错误需要中止"""
    errors = state.get("errors", [])
    papers = state.get("papers", [])

    # 如果没有获取到任何论文且有错误，中止流程
    if not papers and errors:
        return "end"

    return "continue"


# =============================================================================
# 并行处理节点 (Map-Reduce 模式)
# =============================================================================

def fan_out_downloads(state: PaperState) -> List[Send]:
    """分发下载任务到并行节点"""
    papers_data = state.get("papers", [])

    if not state.get("download_pdf", True) or not papers_data:
        return []

    return [
        Send("download_single", {
            "paper": paper_data,
            "downloaded": [],
            "converted": [],
            "indexed": [],
            "errors": [],
        })
        for paper_data in papers_data
    ]


def download_single(state: SinglePaperState) -> dict:
    """下载单篇论文 (并行执行)"""
    ctx = get_context()
    paper_data = state.get("paper", {})

    try:
        paper = dict_to_paper(paper_data)

        if not paper.pdf_url:
            return {
                "errors": [f"论文无PDF链接: {paper.id}"],
            }

        path = ctx.downloader.download(paper)

        if path:
            paper.pdf_path = str(path)
            return {
                "downloaded": [paper.id],
            }
        else:
            return {
                "errors": [f"下载失败: {paper.id}"],
            }

    except Exception as e:
        return {
            "errors": [f"下载异常 [{paper_data.get('id', 'unknown')}]: {str(e)}"],
        }


def aggregate_downloads(state: PaperState) -> dict:
    """聚合下载结果"""
    ctx = get_context()
    downloaded = state.get("downloaded", [])
    errors = state.get("errors", [])

    ctx.logger.info(f"[Aggregate] 下载完成 | 成功: {len(downloaded)} | 错误: {len(errors)}")

    return {
        "current_step": "download",
        "progress": 50.0,
    }


# =============================================================================
# 重试策略
# =============================================================================

# 下载节点的重试策略
download_retry_policy = RetryPolicy(
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=30.0,
    max_attempts=3,
    jitter=True,
    retry_on=(ConnectionError, TimeoutError),
)

# 转换节点的重试策略
convert_retry_policy = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=1.5,
    max_interval=10.0,
    max_attempts=2,
    jitter=True,
    retry_on=Exception,
)


# =============================================================================
# 构建工作流图
# =============================================================================

def build_paper_pipeline(
    checkpointer=None,
    use_parallel: bool = False,
) -> StateGraph:
    """构建论文处理工作流图

    Args:
        checkpointer: 检查点存储器 (用于断点续传)
        use_parallel: 是否使用并行处理模式

    Returns:
        编译后的工作流图
    """
    # 创建状态图
    workflow = StateGraph(PaperState)

    # 添加节点
    workflow.add_node("get_papers", get_paper_list)
    workflow.add_node("download", download_papers, retry=download_retry_policy)
    workflow.add_node("convert", convert_to_markdown, retry=convert_retry_policy)
    workflow.add_node("index", build_index)
    workflow.add_node("save_metadata", save_metadata)

    # 定义边
    workflow.add_edge(START, "get_papers")

    # 条件边: get_papers -> download 或 save_metadata
    workflow.add_conditional_edges(
        "get_papers",
        should_download,
        {
            "download": "download",
            "save_metadata": "save_metadata",
        }
    )

    # 条件边: download -> convert 或 save_metadata
    workflow.add_conditional_edges(
        "download",
        should_convert,
        {
            "convert": "convert",
            "save_metadata": "save_metadata",
        }
    )

    # 条件边: convert -> index 或 save_metadata
    workflow.add_conditional_edges(
        "convert",
        should_index,
        {
            "index": "index",
            "save_metadata": "save_metadata",
        }
    )

    # index -> save_metadata
    workflow.add_edge("index", "save_metadata")

    # save_metadata -> END
    workflow.add_edge("save_metadata", END)

    # 编译图
    return workflow.compile(checkpointer=checkpointer)


def build_parallel_pipeline(checkpointer=None) -> StateGraph:
    """构建并行处理工作流图

    使用 Send 实现论文的并行下载和转换。

    Args:
        checkpointer: 检查点存储器

    Returns:
        编译后的工作流图
    """
    workflow = StateGraph(PaperState)

    # 添加节点
    workflow.add_node("get_papers", get_paper_list)
    workflow.add_node("download_single", download_single, retry=download_retry_policy)
    workflow.add_node("aggregate", aggregate_downloads)
    workflow.add_node("convert", convert_to_markdown, retry=convert_retry_policy)
    workflow.add_node("index", build_index)
    workflow.add_node("save_metadata", save_metadata)

    # 定义边
    workflow.add_edge(START, "get_papers")

    # 使用条件边实现并行分发
    workflow.add_conditional_edges(
        "get_papers",
        fan_out_downloads,
        ["download_single"],
    )

    # 并行下载后聚合
    workflow.add_edge("download_single", "aggregate")

    # 继续后续流程
    workflow.add_conditional_edges(
        "aggregate",
        should_convert,
        {
            "convert": "convert",
            "save_metadata": "save_metadata",
        }
    )

    workflow.add_conditional_edges(
        "convert",
        should_index,
        {
            "index": "index",
            "save_metadata": "save_metadata",
        }
    )

    workflow.add_edge("index", "save_metadata")
    workflow.add_edge("save_metadata", END)

    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# 主管道类
# =============================================================================

class PaperPipeline:
    """论文处理管道

    完整流程:
    1. 获取论文列表 (GetPaperList)
    2. 下载PDF (DownloadPDF)
    3. 转换为Markdown (ConvertToMD)
    4. 建立索引 (BuildIndex)
    5. 保存元数据 (SaveMetadata)

    特性:
    - 检查点: 支持断点续传
    - 错误恢复: 自动重试失败的操作
    - 并行处理: 可选的并行下载模式
    - 日志记录: 详细的操作日志

    Example:
        >>> pipeline = PaperPipeline()
        >>> result = pipeline.run(
        ...     conference="ICLR",
        ...     year=2024,
        ...     source="openreview",
        ...     max_papers=10,
        ... )
        >>> print(f"Downloaded: {len(result['downloaded'])}")
    """

    def __init__(
        self,
        use_parallel: bool = False,
        use_checkpointing: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        """初始化管道

        Args:
            use_parallel: 是否使用并行处理模式
            use_checkpointing: 是否启用检查点
            checkpoint_dir: 检查点存储目录
        """
        self.use_parallel = use_parallel
        self.use_checkpointing = use_checkpointing
        self.checkpoint_dir = checkpoint_dir or settings.storage.data_dir / "checkpoints"
        self.logger = get_logger("PaperPipeline")

        # 初始化检查点存储
        self.checkpointer = None
        if use_checkpointing:
            self.checkpointer = self._init_checkpointer()

        # 构建工作流
        if use_parallel:
            self.graph = build_parallel_pipeline(self.checkpointer)
        else:
            self.graph = build_paper_pipeline(self.checkpointer)

    def _init_checkpointer(self):
        """初始化检查点存储"""
        try:
            # 尝试使用 SQLite 持久化检查点
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.checkpoint_dir / "checkpoints.db"

            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            return SqliteSaver(conn)

        except ImportError:
            # 退回到内存检查点
            self.logger.warning("SQLite checkpointer not available, using memory saver")
            return MemorySaver()

    def run(
        self,
        conference: str,
        year: int,
        source: str = "arxiv",
        query: Optional[str] = None,
        max_papers: int = 100,
        download_pdf: bool = True,
        convert_to_md: bool = True,
        build_index: bool = True,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> PaperState:
        """运行管道

        Args:
            conference: 会议名称
            year: 年份
            source: 数据源
            query: 搜索关键词 (可选)
            max_papers: 最大论文数量
            download_pdf: 是否下载PDF
            convert_to_md: 是否转换为Markdown
            build_index: 是否建立索引
            thread_id: 线程ID (用于检查点恢复)
            resume: 是否从检查点恢复

        Returns:
            最终状态
        """
        # 生成或使用提供的线程ID
        if thread_id is None:
            thread_id = generate_task_id(conference, year, source)

        self.logger.info(f"开始处理 | 会议: {conference} | 年份: {year} | 线程: {thread_id}")

        # 配置
        config = {"configurable": {"thread_id": thread_id}}

        # 初始状态
        initial_state: PaperState = {
            "conference": conference,
            "year": year,
            "source": source,
            "query": query,
            "max_papers": max_papers,
            "download_pdf": download_pdf,
            "convert_to_md": convert_to_md,
            "build_index": build_index,
            "papers": [],
            "downloaded": [],
            "converted": [],
            "indexed": [],
            "errors": [],
            "current_step": "init",
            "progress": 0.0,
            "start_time": None,
            "end_time": None,
        }

        # 如果需要恢复，检查是否有现有状态
        if resume and self.checkpointer:
            try:
                existing_state = self.graph.get_state(config)
                if existing_state and existing_state.values:
                    self.logger.info(f"从检查点恢复 | 步骤: {existing_state.values.get('current_step')}")
                    initial_state = existing_state.values
            except Exception as e:
                self.logger.warning(f"无法恢复检查点: {e}")

        # 运行工作流
        try:
            result = self.graph.invoke(initial_state, config)

            self.logger.info(
                f"处理完成 | "
                f"论文: {len(result.get('papers', []))} | "
                f"下载: {len(result.get('downloaded', []))} | "
                f"转换: {len(result.get('converted', []))} | "
                f"索引: {len(result.get('indexed', []))} | "
                f"错误: {len(result.get('errors', []))}"
            )

            return result

        except Exception as e:
            self.logger.error(f"管道执行失败: {e}")
            raise

    async def run_async(self, **kwargs) -> PaperState:
        """异步运行管道

        Args:
            **kwargs: 与 run() 相同的参数

        Returns:
            最终状态
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**kwargs))

    def stream(
        self,
        conference: str,
        year: int,
        source: str = "arxiv",
        **kwargs,
    ):
        """流式运行管道 (实时获取中间状态)

        Args:
            conference: 会议名称
            year: 年份
            source: 数据源
            **kwargs: 其他参数

        Yields:
            中间状态
        """
        thread_id = kwargs.pop("thread_id", None) or generate_task_id(conference, year, source)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: PaperState = {
            "conference": conference,
            "year": year,
            "source": source,
            "query": kwargs.get("query"),
            "max_papers": kwargs.get("max_papers", 100),
            "download_pdf": kwargs.get("download_pdf", True),
            "convert_to_md": kwargs.get("convert_to_md", True),
            "build_index": kwargs.get("build_index", True),
            "papers": [],
            "downloaded": [],
            "converted": [],
            "indexed": [],
            "errors": [],
            "current_step": "init",
            "progress": 0.0,
            "start_time": None,
            "end_time": None,
        }

        for event in self.graph.stream(initial_state, config):
            yield event

    def get_state(self, thread_id: str) -> Optional[PaperState]:
        """获取指定线程的状态

        Args:
            thread_id: 线程ID

        Returns:
            状态或None
        """
        if not self.checkpointer:
            return None

        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        return state.values if state else None

    def list_threads(self) -> List[str]:
        """列出所有线程ID

        Returns:
            线程ID列表
        """
        if not self.checkpointer:
            return []

        try:
            # 对于 SQLiteSaver，列出所有线程
            if hasattr(self.checkpointer, "list"):
                return list(self.checkpointer.list())
        except Exception:
            pass

        return []

    def visualize(self, output_path: Optional[Path] = None) -> str:
        """可视化工作流图

        Args:
            output_path: 输出文件路径 (支持 .png, .svg)

        Returns:
            Mermaid 图表字符串
        """
        try:
            mermaid = self.graph.get_graph().draw_mermaid()

            if output_path:
                # 保存为图片需要 graphviz
                try:
                    from langgraph.graph import Graph
                    self.graph.get_graph().draw_png(str(output_path))
                except ImportError:
                    # 保存为 mermaid 文本
                    output_path = output_path.with_suffix(".md")
                    with open(output_path, "w") as f:
                        f.write(f"```mermaid\n{mermaid}\n```")

            return mermaid

        except Exception as e:
            self.logger.warning(f"无法生成可视化: {e}")
            return ""


# =============================================================================
# 便捷函数
# =============================================================================

def run_pipeline(
    conference: str,
    year: int,
    source: str = "arxiv",
    **kwargs,
) -> PaperState:
    """运行论文处理管道的便捷函数

    Args:
        conference: 会议名称
        year: 年份
        source: 数据源
        **kwargs: 其他参数

    Returns:
        最终状态
    """
    pipeline = PaperPipeline()
    return pipeline.run(conference=conference, year=year, source=source, **kwargs)


# 全局工作流实例 (延迟初始化)
_app = None


def get_app() -> StateGraph:
    """获取全局工作流实例"""
    global _app
    if _app is None:
        _app = build_paper_pipeline(checkpointer=MemorySaver())
    return _app


# =============================================================================
# 向后兼容的 app 实例
# =============================================================================

# 使用内存检查点的默认工作流
app = build_paper_pipeline(checkpointer=MemorySaver())


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 示例: 运行论文处理管道

    # 创建管道实例
    pipeline = PaperPipeline(
        use_parallel=False,
        use_checkpointing=True,
    )

    # 打印工作流图
    print("工作流图 (Mermaid):")
    print(pipeline.visualize())
    print()

    # 运行管道
    result = pipeline.run(
        conference="ICLR",
        year=2024,
        source="arxiv",
        query="large language model",
        max_papers=5,
        download_pdf=True,
        convert_to_md=True,
        build_index=True,
    )

    # 打印结果
    print("\n处理结果:")
    print(f"  论文总数: {len(result.get('papers', []))}")
    print(f"  已下载: {len(result.get('downloaded', []))}")
    print(f"  已转换: {len(result.get('converted', []))}")
    print(f"  已索引: {len(result.get('indexed', []))}")
    print(f"  错误数: {len(result.get('errors', []))}")

    if result.get("errors"):
        print("\n错误信息:")
        for err in result["errors"][:5]:
            print(f"  - {err}")
