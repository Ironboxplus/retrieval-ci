"""
Pydantic 数据模型
"""
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime


class FileType(str, Enum):
    """支持的文件类型"""
    BOOKMARK = "bookmark"
    PDF = "pdf"
    TXT = "txt"


class RetrievalMethod(str, Enum):
    """检索方法"""
    VECTOR = "vector"
    HYBRID = "hybrid"
    SPARSE = "sparse"


class BookmarkItem(BaseModel):
    """书签条目"""
    title: str
    url: str
    add_date: Optional[datetime] = None
    folder: Optional[str] = None
    tags: List[str] = []


class Document(BaseModel):
    """文档模型"""
    id: str
    title: str
    source: str  # 来源：文件名或URL
    source_type: FileType
    content: str
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    chunks: List["Chunk"] = []


class Chunk(BaseModel):
    """文本块模型"""
    id: str
    doc_id: str
    content: str
    level: int  # 层级: 1=文档摘要, 2=段落, 3=句子
    start_idx: int
    end_idx: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


# venue_tier of arXiv preprints in `papers`; default search excludes it and
# /api/arxiv/search is scoped to it.
PREPRINT_TIER = "preprint"


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(
        ...,
        description=(
            "检索文本。系统为 BGE-M3 向量检索（dense+sparse 混合 RRF + cross-encoder "
            "精排），对长而具体的语义输入效果最好——建议使用尽可能详细、完整的查询"
            "（完整问题描述 / 方法句 / 概念组合），而非单个关键词。"
        ),
    )
    method: RetrievalMethod = RetrievalMethod.HYBRID
    top_k: int = Field(default=20, ge=1, le=1000, description="返回结果数量，必须大于0")
    filters: Dict[str, Any] = {}
    source_type: Optional[str] = None
    data_source: Optional[str] = None
    venue: Optional[Union[str, List[str]]] = None
    year: Optional[Union[int, List[int]]] = None
    venue_tier: Optional[str] = None
    include_preprints: bool = Field(
        default=False,
        description=(
            "是否混入 arXiv 预印本（venue_tier=preprint）。默认不混入；只检索预印本请用 "
            "/api/arxiv/search。已按 venue_tier / paper_ids / doc_ids / venue=arXiv 限定范围"
            "的请求不受此项影响。"
        ),
    )
    include_withdrawn: bool = Field(
        default=False,
        description=(
            "是否包含已撤稿的 arXiv 预印本。默认排除；按 paper_ids / doc_ids 限定的请求"
            "不受此项影响，仍能找到撤稿论文。命中的 metadata.withdrawn 标记撤稿。"
        ),
    )
    title_only: bool = Field(
        default=False,
        description=(
            "仅匹配论文标题与 L1（标题+摘要）块。聚焦标题/摘要，不被正文里附录等次要"
            "内容带偏大方向；适合论文初步调研、按主题快速圈定候选论文。"
        ),
    )
    dedup_by_paper: bool = Field(
        default=True,
        description=(
            "每篇论文卡片展示几个 chunk。True（默认，保留最高分）：每篇论文只返回最高分"
            "的那个 chunk，防止单篇多段刷屏（W3）、控制返回数据量。False（展开全部匹配）："
            "每篇论文返回其全部命中的 chunk——适合精读单篇、定位论文内的多个相关段落。"
            "注意：无论该开关如何，结果始终按「论文最高分」分组排序，粒度是论文。"
        ),
    )
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "限定检索范围到指定的一个或多个论文（paper_id）。只在这些论文的 chunk 上检索，"
            "其余论文一律不参与。用于「在指定论文内检索」——配合 dedup_by_paper=False + 较大"
            "的 top_k，可拿到单篇论文全部命中片段及其分数，看清论文不同部分与查询的匹配程度。"
            "只对 source_type=paper/review 有效：上传文档和网页没有 paper_id（见"
            " core/storage/papers_payload.py 的 profile 表），要用 doc_ids。"
        ),
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "限定检索范围到指定的一个或多个文档（documents.id）。上传的书/网页没有 paper_id，"
            "doc_id 是它们唯一的主键，所以「在这本书里检索」只能走这里。对论文同样可用。"
            "值一律按字符串下推——Qdrant 的 doc_id KEYWORD 索引只收字符串，整数值的历史点位"
            "不在索引里（papers_payload.py D2）。"
        ),
    )

    @model_validator(mode="after")
    def _reject_conflicting_id_filters(self) -> "SearchRequest":
        """paper_ids + doc_ids 同时给 = 必然空集，明确拒绝而不是静默返回 0 条。

        两者都映射成 payload 相等条件，而 QdrantStore._build_filter 把所有条件放进
        Filter(must=...) 做 AND。一个点不可能既属于 paper_id 又属于另一个文档的
        doc_id（论文的两者指向同一文档时才偶然成立），所以这个组合几乎总是返回空，
        而空结果和「没搜到」无法区分——那正是这次 bug 难查的原因。
        """
        if self.paper_ids and self.doc_ids:
            raise ValueError(
                "paper_ids and doc_ids are mutually exclusive: they become AND-ed "
                "payload conditions, so passing both matches nothing. Pass whichever "
                "id the search result carried."
            )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "语义检索 + 会议/年份过滤",
                    "description": "用完整的方法描述而非关键词；venue/year 收窄到目标会议。",
                    "value": {
                        "query": "retrieval-augmented generation with a learned reranker for long-context QA",
                        "method": "hybrid",
                        "top_k": 20,
                        "venue": "NeurIPS",
                        "year": 2024,
                    },
                },
                {
                    "summary": "在指定论文内检索（展开全部命中片段）",
                    "description": "paper_ids 锁定单篇 + dedup_by_paper=false + 较大 top_k，拿回该论文全部命中段落及分数。",
                    "value": {
                        "query": "how is the position head pretrained",
                        "paper_ids": ["neurips-2024-some-paper"],
                        "dedup_by_paper": False,
                        "top_k": 50,
                    },
                },
            ]
        }
    )


class SearchResult(BaseModel):
    """检索结果"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    doc_title: str
    source: str
    metadata: Dict[str, Any] = {}


class AggregatedResult(BaseModel):
    """按文档聚合的结果"""
    doc_id: str
    doc_title: str
    source: str
    hit_count: int  # 命中的chunk数量
    total_score: float
    avg_score: float
    max_score: float = 0.0  # 最高分数
    chunks: List[SearchResult] = []


class PaperGroup(BaseModel):
    """按论文分组的检索结果（粒度=论文）。

    检索结果以论文为单位呈现：每篇论文是一张卡片，按其命中 chunk 的最高分
    (`max_score`) 排序，卡片内携带该论文命中的 chunk（`chunks`，已按分降序）。
    `dedup_by_paper=True` 时每篇只保留最高分 chunk；False 时携带全部命中片段。
    """
    paper_id: str  # 业务主键；非论文来源（网页等）无 paper_id 时为空串
    doc_id: str
    doc_title: str = ""
    source: str = ""
    max_score: float = 0.0       # 该论文命中 chunk 的最高分 —— 论文排序依据
    chunk_count: int = 0         # 本卡片携带的 chunk 数（= len(chunks)）
    metadata: Dict[str, Any] = {}  # 论文级元数据（取自最高分 chunk：venue/year/authors…）
    chunks: List[SearchResult] = []


class PipelineMetricsResponse(BaseModel):
    """检索管线性能指标"""
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    mode: str = "hybrid"


class VenueAggregation(BaseModel):
    venue: str
    count: int
    avg_score: float


class YearAggregation(BaseModel):
    year: int
    count: int
    avg_score: float


class SearchResponse(BaseModel):
    """检索响应。

    `results` 是命中 chunk 的扁平列表（向后兼容，`total_results == len(results)`）。
    `papers` 是同一批 chunk 按论文分组后的视图（粒度=论文，按论文最高分排序）——
    前端以 `papers` 为主渲染论文卡片。
    """
    query: str
    method: RetrievalMethod
    total_results: int
    results: List[SearchResult] = []
    papers: List[PaperGroup] = []
    aggregated: List[AggregatedResult] = []
    pipeline_metrics: Optional[PipelineMetricsResponse] = None
    venue_distribution: List[VenueAggregation] = []
    year_distribution: List[YearAggregation] = []


class UploadResponse(BaseModel):
    """上传响应"""
    success: bool
    message: str
    file_id: Optional[str] = None
    documents_count: int = 0
    urls_found: int = 0  # 书签中发现的URL数量


class CrawlRequest(BaseModel):
    """爬取请求"""
    urls: List[str]
    use_proxy: bool = False


class CrawlStatus(BaseModel):
    """爬取状态"""
    url: str
    status: str  # pending, crawling, success, failed
    content_length: int = 0
    error: Optional[str] = None


class CrawlResponse(BaseModel):
    """爬取响应"""
    task_id: str
    total_urls: int
    status: List[CrawlStatus] = []


class IndexStats(BaseModel):
    """索引统计"""
    total_documents: int
    total_chunks: int
    chunk_levels: Dict[int, int]  # 各层级chunk数量
    vector_index_size: int
    bm25_index_size: int


# 解决前向引用
Document.model_rebuild()


# ---------------------------------------------------------------------------
# Schedules (app/api/routes/schedules.py). Kept as one block, appended last.
# ---------------------------------------------------------------------------
class ScheduleMetric(BaseModel):
    """One labelled gauge on a schedule card."""
    name: str
    value: Optional[Union[float, int, str]] = None
    unit: str = ""


class ScheduleTaskBrief(BaseModel):
    """The Task a tick started, as of now."""
    id: int
    status: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_json: Optional[Dict[str, Any]] = None


class ScheduleTickResponse(BaseModel):
    id: int
    schedule_id: int
    scheduled_for: Optional[str] = Field(None, description="The slot the trigger was due, UTC ISO-8601")
    fired_at: Optional[str] = None
    trigger: str = Field(description="cron | manual | retry")
    status: str = Field(description="enqueued | skipped | missed | error")
    reason: Optional[str] = None
    task_id: Optional[int] = None
    attempt: int = 0
    task: Optional[ScheduleTaskBrief] = None


class ScheduleRunningTask(BaseModel):
    id: int
    task_type: str
    status: str
    progress: int = 0
    total: int = 0
    started_at: Optional[str] = Field(None, description="RFC 3339 with offset")


class ScheduleResponse(BaseModel):
    id: int
    key: str
    kind: str
    title: str
    enabled: bool
    system: bool
    trigger_json: Dict[str, Any] = Field(description="daily | weekly | interval, with timezone")
    params_json: Dict[str, Any]
    catchup: str = Field(description="latest | skip")
    misfire_grace_s: Optional[int] = Field(None, description="null = the kind's default")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    next_run_time: Optional[str] = Field(None, description="From the engine; null while paused or engine idle")
    next_fire_times: List[str] = Field(default_factory=list, description="Next 5 slots, ISO-8601 with offset")
    last_tick: Optional[ScheduleTickResponse] = None
    running_task: Optional[ScheduleRunningTask] = None
    metrics: List[ScheduleMetric] = Field(default_factory=list)


class ScheduleCreateRequest(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    kind: str
    title: str = Field(min_length=1, max_length=200)
    trigger: Dict[str, Any]
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    catchup: str = "latest"
    misfire_grace_s: Optional[int] = Field(None, ge=0)


class SchedulePatchRequest(BaseModel):
    """Only the fields present in the body change."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    enabled: Optional[bool] = None
    trigger: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    catchup: Optional[str] = None
    misfire_grace_s: Optional[int] = Field(None, ge=0)


class ScheduleKindDefault(BaseModel):
    key: str
    title: str
    trigger: Dict[str, Any]
    params: Dict[str, Any]
    catchup: str
    misfire_grace_s: Optional[int] = None
    enabled: bool


class ScheduleRetryPolicy(BaseModel):
    max_retries: int
    base_delay_s: int
    backoff: float
    max_delay_s: int


class ScheduleKindResponse(BaseModel):
    name: str
    title: str
    params_schema: Dict[str, Any] = Field(description="JSON Schema of the kind's parameters")
    defaults: List[ScheduleKindDefault]
    retry: ScheduleRetryPolicy


class ScheduleFireResponse(BaseModel):
    status: str = Field(description="enqueued | skipped | noop")
    task_id: Optional[int] = None
    reason: Optional[str] = None


class SchedulePreviewRequest(BaseModel):
    trigger: Dict[str, Any]
    count: int = Field(5, ge=1, le=20)


class SchedulePreviewResponse(BaseModel):
    next_fire_times: List[str] = Field(description="ISO-8601 with the trigger's UTC offset")


class ScheduleDeleteResponse(BaseModel):
    id: int
    deleted: bool = True
