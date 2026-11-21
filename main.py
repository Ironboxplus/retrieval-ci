"""
Retrieval System - FastAPI 主入口
"""
import core.wmi_fix  # noqa: F401 — must be first (Python 3.13 WMI hang bypass)

import os
import uuid
import asyncio
import importlib.util
import sys
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

# Bypass HTTP_PROXY (used for HuggingFace etc.) for localhost services — qdrant
# on 127.0.0.1:6333 would otherwise be routed through the user's proxy and
# silently corrupt the body (observed 2026-05-16: "Format error in JSON body
# column 5317" → search returned 0 candidates).
_no_proxy_extra = "localhost,127.0.0.1,::1"
for var in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(var, "")
    parts = {p.strip() for p in existing.split(",") if p.strip()}
    parts.update(_no_proxy_extra.split(","))
    os.environ[var] = ",".join(sorted(parts))

from functools import partial

from fastapi import (
    FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks, Request,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.compression import CompressedRequestMiddleware, CompressedResponseMiddleware
from pydantic import BaseModel

from core.logging import setup_logging, get_logger
from config import settings

setup_logging()
logger = get_logger(__name__)
from app.models.schemas import (
    FileType, RetrievalMethod,
    SearchRequest, SearchResponse, SearchResult, AggregatedResult,
    PipelineMetricsResponse, VenueAggregation, YearAggregation,
    UploadResponse, CrawlRequest, CrawlResponse, IndexStats
)
from app.api.identity import resolve_owner
from app.services.match_doc.jobs import MATCH_PAPER_JOB
from app.services.match_doc.schemas import (
    MatchDocJobReceipt,
    MatchDocJobStatus,
    MatchDocRequest,
    MatchDocResponse,
)
from pipeline.parsers import BookmarkParser, PDFParser, TXTParser
from core.indexing.embedders.bge_m3_core import InputTooLong
from core.retrieval.pipeline import SearchPipeline, SearchRequest as PipelineSearchRequest
from core.storage.qdrant_store import QdrantRetrievalUpstreamError
from app.services.search_grouping import arxiv_scope, build_qdrant_filters, group_into_papers, attach_chunk_positions
from core.db.database import init_db
from app.api.routes import register_routes
from app.services.index_service import get_index_service


def _load_module_from_path(module_name: str, module_path: Path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_paper_api_routes = _load_module_from_path(
    "_retrival_paper_api_routes",
    Path(__file__).resolve().parent / "pipeline" / "paper_crawler" / "integration" / "api_routes.py",
)
mount_paper_routes = _paper_api_routes.mount_paper_routes
create_runtime_paper_indexer = _paper_api_routes.create_runtime_paper_indexer


# 全局组件
search_pipeline: Optional[SearchPipeline] = None
match_paper_job_runner = None

paper_indexer = create_runtime_paper_indexer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global search_pipeline, match_paper_job_runner

    # 启动时初始化
    logger.info("Initializing database...")
    init_db()

    # Realign journal_fts rowids with entry ids before serving: the first journal
    # request would otherwise pay for the rebuild. A failure here is not swallowed
    # for good - every journal write re-runs the guard and raises until it succeeds.
    try:
        from app.services.journal import get_journal_service
        get_journal_service().repository.ensure_fts_aligned()
    except Exception as exc:
        logger.error("journal_fts alignment failed at startup: %s", exc)

    logger.info("Initializing components...")

    # The embedding service must be the one this build's fingerprint names. A service that serves other
    # weights or a shorter limit raises here and stops the start-up: it is deliberately outside the
    # try below, which turns any failure into "no search pipeline". One that stays silent for
    # SERVICE_WAIT_S is logged, and the backend runs while embedding calls raise until it is up.
    from core.indexing.embedders import require_embedding_service
    require_embedding_service()

    logger.info("Loading BGE-M3 search pipeline...")
    try:
        # W46: get_embedder() warms the process-wide BGE-M3 cache so all
        # downstream callers (IndexService variants, SemanticSectionChunker
        # _BgeM3LlamaAdapter, SearchPipeline) share one hot-in-GPU instance.
        from core.indexing.embedders import get_embedder
        from core.storage.corpus_store import build_search_store
        embedder = get_embedder()
        # papers / arxiv / reviews routing; serves the single `papers` collection
        # until the corpus layout file switches to the split (no restart needed).
        store = build_search_store()

        reranker = None
        try:
            from core.retrieval.reranker import get_shared_reranker
            reranker = get_shared_reranker()
            reranker._load_model()
            logger.info("Reranker loaded: bge-reranker-v2-m3")
        except Exception as re_err:
            logger.warning(f"Reranker not available ({re_err}), running without rerank")

        search_pipeline = SearchPipeline(embedder=embedder, store=store, reranker=reranker)
        logger.info(f"Search pipeline ready: {type(embedder).__name__}, dim={embedder.dimension}, sparse={embedder.supports_sparse}")
    except Exception as e:
        logger.error(f"Search pipeline init failed: {e}")
        search_pipeline = None

    logger.info("Components initialized.")

    if search_pipeline is not None:
        try:
            from app.services.match_doc.jobs import get_match_paper_job_runner

            match_paper_job_runner = get_match_paper_job_runner(search_pipeline)
            resumed_matches = match_paper_job_runner.resume_active()
            if resumed_matches:
                logger.info(
                    "match_paper.resume relaunched %d task(s): %s",
                    len(resumed_matches),
                    resumed_matches,
                )
        except Exception as exc:
            match_paper_job_runner = None
            logger.warning("match-paper job runner unavailable: %s", exc)

    # W58 diag: background thread logs private commit every 60s to catch leaks.
    # 2026-11-20: the old value was psutil uss, which on Linux counts resident pages only and
    # hid a 13 GB swapped-out rise; core.memory_watch logs VmData/VmSwap/VmRSS and captures
    # thread stacks plus a tracemalloc window when VmData bursts.
    import torch
    from core.memory_watch import start_memory_watch

    def _gpu_fields() -> str:
        gpu_a = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        gpu_r = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        return "gpu_alloc=%.2fGB gpu_rsv=%.2fGB" % (gpu_a, gpu_r)

    start_memory_watch(logger, extra=_gpu_fields)

    # Resume tasks left "running" by a prior (killed) process — W61: ingest
    # pipelines, upload pipelines (extract→index) and medical OA downloads, each
    # through its registered runner. The medical runner probes the atomically
    # checkpointed manifest and resumes only on the designated .88 executor; a
    # Windows/.111 process fails closed and never writes PDFs. Guarded per
    # runner: one resume failure must never block startup or the others.
    from app.services.pipeline.runner_registry import all_runners
    for task_type, get_task_runner in all_runners().items():
        try:
            resumed = get_task_runner().resume_running()
            if resumed:
                logger.info(f"{task_type}.resume relaunched {len(resumed)} task(s): {resumed}")
        except Exception as e:
            logger.warning(f"{task_type}.resume skipped: {e}")

    # Recurring jobs (public-code update checks, arXiv delta, ...). After the
    # runners resumed, so a schedule firing at startup sees their running tasks.
    scheduler_service = None
    if settings.scheduler.enabled:
        try:
            from app.services.scheduler.service import get_scheduler_service
            scheduler_service = get_scheduler_service()
            if scheduler_service.start():
                logger.info("scheduler service started")
        except Exception as e:
            logger.warning(f"scheduler skipped: {e}")

    yield

    # 关闭时清理
    if scheduler_service is not None:
        scheduler_service.shutdown()
    try:
        from app.services.public_code import get_public_code_job_service
        get_public_code_job_service().shutdown(wait=False)
    except Exception as e:
        logger.warning(f"public-code job shutdown skipped: {e}")
    logger.info("Shutting down...")


app = FastAPI(
    title="Retrieval System",
    description="学术论文检索系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# zstd/gzip 传输压缩。index_code 的正文实测 19.3 MiB → zstd-3 压到 2.2 MiB（9.1x，54ms），
# 省的是 LAN overlay 和公网隧道的传输时间，不是内存（正文解压后仍要完整解析）。
# **完全按请求协商，已发布的旧客户端不受影响**：没有 Content-Encoding 的正文原样透传，
# 响应只会用客户端自己在 Accept-Encoding 里列出的编码。细节见 app/api/compression.py。
# add_middleware 后加的在外层，所以这两个要在 CORS 之后加：请求体解码发生在路由之前，
# 响应压缩包住路由。
app.add_middleware(CompressedResponseMiddleware, minimum_size=1024)
app.add_middleware(CompressedRequestMiddleware)

# 内网算力 token 闸（付费三层架构）：仅当设置环境变量 INTERNAL_TOKEN 时启用。
# 设计——**局域网直连依然免认证，便于内网使用**：私有非环回 IP（含 10.100.100.x lab
# 覆盖网）的客户端无需令牌直接放行；公网经反代隧道到达时 peer 是环回(127.x)，不算 LAN
# → 必须带 BWH 网关注入的 X-Internal-Token（网关只认令牌不认用户）。未设置 INTERNAL_TOKEN
# 时完全 no-op，保持现有直连行为不变。判定逻辑在 core.internal_gate（已单测）。
_INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "").strip()
_INTERNAL_TOKEN_EXEMPT = frozenset({"/health", "/healthz"})
_INTERNAL_TRUST_LAN = os.environ.get("INTERNAL_TOKEN_TRUST_LAN", "1") != "0"

if _INTERNAL_TOKEN:
    from fastapi.responses import JSONResponse
    from core.internal_gate import gate_allows

    @app.middleware("http")
    async def _internal_token_gate(request, call_next):
        if gate_allows(
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
            token_header=request.headers.get("X-Internal-Token"),
            expected_token=_INTERNAL_TOKEN,
            exempt=_INTERNAL_TOKEN_EXEMPT,
            trust_lan=_INTERNAL_TRUST_LAN,
        ):
            return await call_next(request)
        return JSONResponse({"detail": "internal token required"}, status_code=401)

    logger.info("internal-token gate ENABLED (trust_lan=%s) — LAN 直连免令牌，公网经网关须带 X-Internal-Token",
                _INTERNAL_TRUST_LAN)

# 静态文件（旧前端已弃用，新前端在 frontend/ 由 Vite 提供）
static_dir = os.path.join(settings.base_dir, "static")

# 注册API路由模块
register_routes(app)
mount_paper_routes(app, indexer=paper_indexer)

from app.api.routes.pipeline import router as pipeline_router, batch_index_router
app.include_router(pipeline_router)
app.include_router(batch_index_router)

# Prometheus /metrics (W128) — MUST be wired before the SPA catch-all below
# (`GET /{spa_path:path}`), else the catch-all shadows it and returns SPA HTML
# (W114). Compute-side metrics only; per-user audit lives in the gateway → Loki.
from core.observability import setup_metrics, make_log_relay_router
setup_metrics(app)

# Per-user audit relay (W128 Phase 2): the BWH gateway ships one access-log line
# per authenticated request inward over the EXISTING internal tunnel; we forward
# it into the 4090's local Loki. Keeps BWH stateless (it only emits) and the audit
# next to Prometheus/Grafana. Best-effort — a Loki hiccup never errors the relay.
_loki_push_url = os.environ.get("LOKI_PUSH_URL", "http://127.0.0.1:3100")
app.include_router(make_log_relay_router(
    loki_url=_loki_push_url, labels={"job": "gateway-access", "host": "bwh"}))


# ==================== 会议注册表 ====================

_conferences_cache: Optional[dict] = None

@app.get("/api/conferences")
def get_conferences():
    global _conferences_cache
    if _conferences_cache is None:
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry, load_categories
        registry = load_conference_registry()
        categories = load_categories()
        confs = []
        for entry in registry:
            confs.append({
                "title": entry.get("title", entry["name"]),
                "description": entry.get("description", ""),
                "sub": entry.get("sub", ""),
                "rank": entry.get("rank", {}),
                "dblp": entry.get("dblp", ""),
                "venue_tier": entry["venue_tier"],
                "years": entry.get("years", []),
                "downloadable": entry.get("downloadable", True),
            })
        _conferences_cache = {"categories": categories, "conferences": confs}
    return _conferences_cache


# ==================== 页面路由 ====================
# 新前端在 frontend/ 由 Vite dev server (5173) 或 build 后 nginx 提供
# :8000 仅提供 API


# ==================== 文件上传 ====================

@app.post("/api/upload", response_model=UploadResponse)
def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传文件

    - file_type: bookmark / pdf / txt

    注意：使用统一的数据路径，同时写入SQLite和ChromaDB/Whoosh
    """
    from core.db.database import get_db_session
    from core.db.models import Document as DBDocument, SourceType, DataSource, DocumentStatus
    from app.services.index_service import get_index_service

    try:
        file_type_enum = FileType(file_type)
    except ValueError:
        raise HTTPException(400, f"Unsupported file type: {file_type}")

    # 保存文件
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(settings.upload_dir, f"{file_id}{file_ext}")

    content = file.file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 解析文件并统一写入数据库+索引（singleton 复用 BGE-M3 模型实例）
    docs_count = 0
    urls_found = 0
    index_service = get_index_service()

    # 映射FileType到SourceType
    source_type_map = {
        FileType.BOOKMARK: SourceType.upload,
        FileType.PDF: SourceType.upload,
        FileType.TXT: SourceType.upload,
    }

    with get_db_session() as db:
        if file_type_enum == FileType.BOOKMARK:
            parser = BookmarkParser()
            bookmarks = parser.parse(content.decode("utf-8", errors="ignore"))
            urls_found = len(bookmarks)

            # 将书签信息作为文档索引 - 使用统一路径
            for bm in bookmarks:
                # 1. 创建SQLite文档记录
                doc = DBDocument(
                    title=bm.title,
                    source=bm.url,
                    source_type=source_type_map.get(file_type_enum, SourceType.upload),
                    data_source=DataSource.user,
                    file_path=save_path,
                    url=bm.url,
                    status=DocumentStatus.pending,
                    metadata_json={"folder": bm.folder, "tags": bm.tags}
                )
                db.add(doc)
                db.flush()  # 获取ID

                # 2. 建立索引(写入ChromaDB + Whoosh + SQLite chunks)
                doc_content = f"{bm.title}\n{bm.url}\n{bm.folder or ''}"
                result = index_service.index_document(
                    doc_id=doc.id,
                    content=doc_content,
                    title=bm.title,
                    source=bm.url,
                    metadata={"folder": bm.folder, "tags": bm.tags},
                    db=db,
                    source_type="upload",
                    data_source="user"
                )

                if result["success"]:
                    doc.status = DocumentStatus.indexed
                else:
                    doc.status = DocumentStatus.failed

                docs_count += 1

        elif file_type_enum == FileType.PDF:
            parser = PDFParser()
            text, metadata = parser.parse(save_path)

            # 1. 创建SQLite文档记录
            doc = DBDocument(
                title=file.filename,
                source=file.filename,
                source_type=source_type_map.get(file_type_enum, SourceType.upload),
                data_source=DataSource.user,
                file_path=save_path,
                status=DocumentStatus.pending,
                metadata_json=metadata
            )
            db.add(doc)
            db.flush()

            # 2. 建立索引
            result = index_service.index_document(
                doc_id=doc.id,
                content=text,
                title=file.filename,
                source=file.filename,
                metadata=metadata,
                db=db,
                source_type="upload",
                data_source="user"
            )

            if result["success"]:
                doc.status = DocumentStatus.indexed
            else:
                doc.status = DocumentStatus.failed

            docs_count = 1

        elif file_type_enum == FileType.TXT:
            parser = TXTParser()
            text, metadata = parser.parse(save_path)

            # 1. 创建SQLite文档记录
            doc = DBDocument(
                title=file.filename,
                source=file.filename,
                source_type=source_type_map.get(file_type_enum, SourceType.upload),
                data_source=DataSource.user,
                file_path=save_path,
                status=DocumentStatus.pending,
                metadata_json=metadata
            )
            db.add(doc)
            db.flush()

            # 2. 建立索引
            result = index_service.index_document(
                doc_id=doc.id,
                content=text,
                title=file.filename,
                source=file.filename,
                metadata=metadata,
                db=db,
                source_type="upload",
                data_source="user"
            )

            if result["success"]:
                doc.status = DocumentStatus.indexed
            else:
                doc.status = DocumentStatus.failed

            docs_count = 1

    return UploadResponse(
        success=True,
        message=f"File uploaded and indexed successfully",
        file_id=file_id,
        documents_count=docs_count,
        urls_found=urls_found
    )


# ==================== 爬取 ====================

@app.post("/api/crawl", response_model=CrawlResponse)
def start_crawl(request: CrawlRequest):
    """启动网页爬取任务 — 委托给 EnhancedCrawler 路径。"""
    from app.tasks.crawl_tasks import enqueue_crawl_urls

    task_id = enqueue_crawl_urls(
        request.urls,
        use_proxy=request.use_proxy,
        auto_index=True,
        max_concurrent=3,
    )
    return CrawlResponse(
        task_id=task_id,
        total_urls=len(request.urls),
        status=[],
    )


# ==================== 检索 ====================
# Qdrant payload filter building (incl. year normalization + paper_ids) now lives
# in app/services/search_grouping.build_qdrant_filters — shared by both handlers.


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """执行检索（统一走 SearchPipeline：BGE-M3 dense+sparse 混合 RRF + cross-encoder 精排）。

    - method: vector（纯语义）/ hybrid（默认，dense+sparse RRF）/ sparse（纯词法）
    - source_type: 按来源筛选 (webpage|paper|upload)
    - data_source: 按数据来源筛选 (system|user)

    大模型调用建议：本系统是向量检索，查询文本越详细、具体，效果越好（用完整问题/方法
    描述，而非单个关键词）。两个可选开关可组合使用：
    - title_only=True：只匹配标题与 L1 块，聚焦标题/摘要、不被附录等次要内容带偏，用于
      论文初步调研、圈定候选论文。
    - dedup_by_paper=False（原始 chunk 模式）：同篇不只取最高分 chunk，返回原始 top-k
      段落，用于小范围精搜、定位论文内具体段落。
    建议对同一检索文本组合不同选项分别调用（先 title_only 找范围，再用原始 chunk 模式
    精搜目标），合并结果以获得最全面的内容。
    """
    import traceback as _tb
    if not search_pipeline:
        raise HTTPException(503, "Search pipeline not initialized")

    if request.method == RetrievalMethod.VECTOR:
        mode = "dense"
    elif request.method == RetrievalMethod.SPARSE:
        mode = "sparse"
    else:
        mode = "hybrid"

    # C2/W87: push venue/year/venue_tier/paper_ids down to the Qdrant payload
    # filter so retrieve is filtered server-side (shared builder = single source
    # of truth with /api/search/stream). paper_ids restricts to specific papers.
    qdrant_filters = build_qdrant_filters(request)
    pipe_req = PipelineSearchRequest(
        query=request.query,
        top_k=request.top_k,
        mode=mode,
        source_type=request.source_type,
        level=1 if request.title_only else None,
        filters=qdrant_filters,
        dedup_by_paper=request.dedup_by_paper,
        include_preprints=request.include_preprints,
        include_withdrawn=request.include_withdrawn,
    )
    # Offload the whole blocking path (embed + Qdrant + rerank + SQL enrich).
    # Running search_pipeline.search inline starved the asyncio event loop under
    # concurrent MCP search_papers calls (2026-07-25 hang forensics: one of two
    # parallel tools never completed while the other returned in ~7s; direct
    # /api/search for the same query is ~1s when the loop is free). Stream route
    # already uses stream_blocking; sync route must match that contract.
    try:
        pipe_resp = await asyncio.to_thread(search_pipeline.search, pipe_req)
    except QdrantRetrievalUpstreamError:
        raise HTTPException(503, "Search retrieval upstream unavailable; retry later")
    except InputTooLong:
        raise  # app.api.errors maps it to a 422 that states the token count
    except Exception as e:
        logger.error(f"Search pipeline failed: {e}")
        _tb.print_exc()
        raise HTTPException(500, f"Search failed: {e}")

    results = pipe_resp.results[:request.top_k]

    # Enrichment delegated to single helper (also used by /api/search/stream).
    # Architecture: Qdrant payload is base (identity + filter fields); SQL JOIN
    # overrides with paper-specific fields where present. None values from SQL
    # are dropped to avoid wiping Qdrant values.
    enriched = await asyncio.to_thread(_enrich_results_sync, results, request)
    # W87: in-paper search → stamp document positions for the 按正文顺序 view.
    enriched = await asyncio.to_thread(_attach_paper_chunk_positions, enriched, request)

    # No application-layer dedup. Data integrity is enforced at the DB layer:
    #   - SQLite documents.content_hash UNIQUE (prevents same-doc re-ingest)
    #   - SQLite papers UNIQUE(title, venue, year)
    #   - Qdrant point_id is deterministic from chunk_id (re-ingest overwrites)
    #   - scripts/cleanup_qdrant_orphans.py purges historic ghost points

    # venue/year/venue_tier are pushed down to Qdrant retrieve filter (see
    # qdrant_filters above) — no application-layer post-filter needed.
    # title_only is similarly handled at retrieve stage via pipe_req.level=1.

    # W87: group the flat chunk hits into paper cards (granularity = paper,
    # ranked by each paper's best chunk score). dedup_by_paper now only governs
    # chunks-per-card (1 = keep best, all = every matched chunk); the pipeline
    # already fetched the matching volume, so grouping is a pure presentation
    # transform. `results` stays flat for backward compat (total_results==len).
    papers = group_into_papers(enriched, keep_best_only=request.dedup_by_paper)

    # Distributions are paper-granular (one vote per paper, via its best chunk)
    # so multi-chunk papers don't over-count. Shared helper, single source of
    # truth with /api/search/stream.
    venue_dist, year_dist = _distributions([p.chunks[0] for p in papers if p.chunks])

    # Pipeline metrics
    import dataclasses
    metrics_resp = PipelineMetricsResponse(**dataclasses.asdict(pipe_resp.metrics))

    return SearchResponse(
        query=request.query,
        method=request.method,
        total_results=len(enriched),
        results=enriched,
        papers=papers,
        aggregated=[],
        pipeline_metrics=metrics_resp,
        venue_distribution=venue_dist,
        year_distribution=year_dist,
    )


def _doc_id_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _enrich_results_sync(results, request: SearchRequest):
    """Shared enrichment for both /api/search and /api/search/stream.

    SQL JOIN provides paper-specific fields (venue_tier, track_type, impact_level,
    citation_count, authors, title) that aren't always in Qdrant payload.
    Qdrant payload provides identity (paper_id, doc_id, chunk_id) and
    filter-pushdown fields (venue, year). The dict merge below uses Qdrant payload
    as base + SQL as override, with None values dropped from SQL so they don't
    wipe valid Qdrant values.
    """
    doc_ids = [d for d in (_doc_id_int(r.doc_id) for r in results) if d is not None]
    paper_info: dict[int, dict] = {}
    paper_titles: dict[int, str] = {}
    doc_info: dict[int, dict] = {}
    doc_titles: dict[int, str] = {}
    if doc_ids:
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper, Document as DBDocument
        with get_db_session() as db:
            for p in db.query(DBPaper).filter(DBPaper.doc_id.in_(doc_ids)).all():
                fields = {
                    "title": p.title,
                    "venue": p.venue,
                    "year": p.year,
                    "authors": p.authors_json,
                    "venue_tier": p.venue_tier,
                    "track_type": p.track_type,
                    "impact_level": p.impact_level,
                    "citation_count": p.citation_count,
                    "paper_id": p.paper_id,  # business ID — single source of truth
                    # SQL is the authority: the Qdrant `withdrawn` flag can lag a
                    # withdrawal/reinstatement. A bool, so False overrides a stale True.
                    "withdrawn": bool(p.withdrawn),
                    "source_version": p.source_version,
                }
                paper_info[p.doc_id] = {k: v for k, v in fields.items() if v is not None}
                if p.title:
                    paper_titles[p.doc_id] = p.title

            # W43: webpage / upload doc_ids aren't in the papers table. Pull
            # Document rows for any doc_id not already covered by paper_info so
            # search cards still render title/url/source_type for non-paper
            # results (previously metadata stayed None → UI card collapsed).
            missing = [d for d in doc_ids if d not in paper_info]
            if missing:
                for doc in db.query(DBDocument).filter(DBDocument.id.in_(missing)).all():
                    fields = {
                        "title": doc.title,
                        "source": doc.source,
                        "url": doc.url,
                        "source_type": doc.source_type.value if doc.source_type else None,
                        "data_source": doc.data_source.value if doc.data_source else None,
                    }
                    doc_info[doc.id] = {k: v for k, v in fields.items() if v is not None}
                    if doc.title:
                        doc_titles[doc.id] = doc.title

    enriched = [
        SearchResult(
            doc_id=r.doc_id, chunk_id=r.chunk_id, content=r.content,
            score=r.score,
            doc_title=(
                paper_titles.get(_doc_id_int(r.doc_id))
                or doc_titles.get(_doc_id_int(r.doc_id))
                or r.doc_title
            ),
            source=r.source,
            metadata={
                **(r.metadata or {}),
                # Document fields first, Paper fields second so Paper wins for
                # paper docs (richer schema). Webpages get Document fallback.
                **doc_info.get(_doc_id_int(r.doc_id), {}),
                **paper_info.get(_doc_id_int(r.doc_id), {}),
            },
        )
        for r in results
    ]

    return enriched


def _distributions(enriched):
    from collections import defaultdict
    venue_stats: dict[str, list[float]] = defaultdict(list)
    year_stats: dict[int, list[float]] = defaultdict(list)
    for r in enriched:
        v = (r.metadata or {}).get("venue")
        y = (r.metadata or {}).get("year")
        if v:
            venue_stats[v].append(r.score)
        if y:
            year_stats[int(y)].append(r.score)
    venue_dist = [
        VenueAggregation(venue=v, count=len(s), avg_score=round(sum(s) / len(s), 4))
        for v, s in sorted(venue_stats.items(), key=lambda x: -len(x[1]))
    ]
    year_dist = [
        YearAggregation(year=y, count=len(s), avg_score=round(sum(s) / len(s), 4))
        for y, s in sorted(year_stats.items())
    ]
    return venue_dist, year_dist


def _attach_paper_chunk_positions(results, request: SearchRequest):
    """W87: for a single-paper in-paper search, stamp each result with its SQL
    chunk_index so the UI can offer a document-order view. The Qdrant payload
    omits positional fields (8-field whitelist), but `content` is whitelisted and
    matches the SQL chunk text exactly — so we map content → chunk_index here.

    No-op unless exactly one document is requested (the in-paper case); a general
    search would otherwise pay a full-paper SQL scan per result paper for data the
    grouped view doesn't use. Accepts either id: `paper_ids` resolves through the
    Paper row to its doc_id, `doc_ids` (uploads/webpages, which have no Paper row
    at all) is already the doc_id.
    """
    paper_ids = getattr(request, "paper_ids", None)
    doc_ids = getattr(request, "doc_ids", None)
    requested = paper_ids or doc_ids
    if not requested or len(requested) != 1 or not results:
        return results
    try:
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper, Chunk as DBChunk
        with get_db_session() as db:
            if paper_ids:
                paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_ids[0]).first()
                if paper is None:
                    return results
                doc_id = paper.doc_id
            else:
                doc_id = int(doc_ids[0])
            rows = (
                db.query(DBChunk.content, DBChunk.chunk_index)
                .filter(DBChunk.doc_id == doc_id)
                .all()
            )
        content_to_index: dict[str, int] = {}
        for content, idx in rows:
            if content is not None and content not in content_to_index:
                content_to_index[content] = idx
        return attach_chunk_positions(results, content_to_index)
    except Exception:
        logger.exception("attach chunk positions failed (id=%s)", requested[0])
        return results


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """NDJSON streaming search: yields per-stage events as embed/retrieve/rerank/enrich
    each complete, so the UI can show real-time progress instead of one terminal blob.

    Stream contract — one JSON object per line:
      {"stage":"embed","embed_ms":N}
      {"stage":"retrieve","retrieve_ms":N,"candidates":N,"mode":"hybrid"}
      {"stage":"rerank","rerank_ms":N,"candidates_before":N,"candidates_after":N}
      {"stage":"enrich","enriched_count":N}
      {"stage":"complete","response": <SearchResponse>}

    大模型调用建议（同 /api/search）：向量检索，查询越详细越好；title_only=True 聚焦
    标题/L1（初步调研、不被附录带偏），dedup_by_paper=False 为原始 chunk 模式（同篇多段、
    小范围精搜）。建议对同一文本组合不同选项分别调用并合并结果。
    """
    import json as _json
    from core.retrieval.stream_runner import stream_blocking

    if not search_pipeline:
        raise HTTPException(503, "Search pipeline not initialized")

    if request.method == RetrievalMethod.VECTOR:
        mode = "dense"
    elif request.method == RetrievalMethod.SPARSE:
        mode = "sparse"
    else:
        mode = "hybrid"

    qdrant_filters = build_qdrant_filters(request)
    pipe_req = PipelineSearchRequest(
        query=request.query, top_k=request.top_k, mode=mode,
        source_type=request.source_type,
        level=1 if request.title_only else None,
        filters=qdrant_filters,
        dedup_by_paper=request.dedup_by_paper,
        include_preprints=request.include_preprints,
        include_withdrawn=request.include_withdrawn,
    )

    async def streamer():
        # W60: route the blocking pipeline through the bounded, leak-safe
        # stream runner. A single process-wide executor caps concurrent
        # CUDA-touching threads (MAX_STREAM_WORKERS); abandoning the stream
        # (client disconnect) signals the producer to stop instead of leaking
        # a thread blocked on a full queue.
        raw_results = None
        raw_metrics: dict = {}
        async for ev in stream_blocking(lambda: search_pipeline.search_stream(pipe_req)):
            if ev.get("stage") == "results":
                raw_results = ev.get("results") or []
                raw_metrics = ev.get("metrics") or {}
                continue
            # Forward embed/retrieve/rerank/error events directly
            yield _json.dumps(ev, ensure_ascii=False) + "\n"

        if raw_results is None:
            yield _json.dumps({"stage": "complete", "response": None}, ensure_ascii=False) + "\n"
            return

        enriched = await asyncio.to_thread(_enrich_results_sync, raw_results, request)
        # W87: in-paper search → stamp document positions for the 按正文顺序 view.
        enriched = await asyncio.to_thread(_attach_paper_chunk_positions, enriched, request)
        # W87: paper-grouped view (granularity = paper, ranked by best chunk).
        papers = group_into_papers(enriched, keep_best_only=request.dedup_by_paper)
        venue_dist, year_dist = _distributions([p.chunks[0] for p in papers if p.chunks])
        yield _json.dumps({"stage": "enrich", "enriched_count": len(enriched)},
                          ensure_ascii=False) + "\n"

        response = SearchResponse(
            query=request.query, method=request.method,
            total_results=len(enriched), results=enriched, papers=papers,
            aggregated=[],
            pipeline_metrics=PipelineMetricsResponse(**raw_metrics),
            venue_distribution=venue_dist, year_distribution=year_dist,
        )
        yield _json.dumps(
            {"stage": "complete", "response": response.model_dump()},
            ensure_ascii=False, default=str,
        ) + "\n"

    return StreamingResponse(streamer(), media_type="application/x-ndjson")


# ==================== arXiv 预印本检索 ====================
# 预印本与会议论文同在 `papers`（venue=arXiv、venue_tier=preprint），默认检索排除它们
# （core/retrieval/pipeline.retrieve_filters）。这两个路由是同一条检索管线，范围固定为
# venue_tier=preprint；venue / venue_tier 入参被覆盖，其余参数语义与 /api/search 相同。

@app.post("/api/arxiv/search", response_model=SearchResponse)
async def search_arxiv(request: SearchRequest):
    """只检索 arXiv 预印本（venue_tier=preprint）。参数与 /api/search 相同。"""
    return await search(arxiv_scope(request))


@app.post("/api/arxiv/search/stream")
async def search_arxiv_stream(request: SearchRequest):
    """/api/arxiv/search 的 NDJSON 流式版本，事件格式同 /api/search/stream。"""
    return await search_stream(arxiv_scope(request))


# ==================== 文档匹配（功能2：doc → 最相近论文）====================


@app.post("/api/match-doc", response_model=MatchDocResponse)
async def match_doc(request: MatchDocRequest):
    """给一篇查询文档，找语料库里内容最相近的论文（doc → paper 匹配，功能2）。

    三种输入二选一：
    - input_type=text：直接粘贴文本（text 字段）；
    - input_type=pdf：先 POST /api/upload 拿 task_id，extract 完成后用 upload_task_id；
    - input_type=paper_id：已入库论文，找它的内容邻居（自动排除自身）。

    打分（agg）：coverage（默认，整篇重合度高者胜）/ maxsim（单段最强命中胜）。
    查询文档切成多块，每块独立走混合检索（BGE-M3 dense+sparse + 精排），再按论文聚合。
    """
    from app.services.match_doc.service import get_match_doc_service

    if not search_pipeline:
        raise HTTPException(503, "Search pipeline not initialized")
    try:
        service = get_match_doc_service(search_pipeline)
        # match() batches all query chunks through exact BGE/Qdrant and heterogeneous
        # reranking, but remains synchronous CPU/GPU work and must stay off-loop.
        # 留在 event loop 上会让一次 match 把整个 backend 挂起几十秒（2026-09-06 实测：
        # match 卡住期间并发 /api/search 120s 不返回）。与 /api/search 同样 offload。
        return await asyncio.to_thread(service.match, request)
    except InputTooLong:
        raise  # a ValueError, but not a 400: app.api.errors maps it to 422 with the token count
    except ValueError as e:  # 输入校验错（缺字段 / 空文本 / MD 未就绪）
        raise HTTPException(400, str(e))
    except RuntimeError as e:  # 依赖未就绪
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.error(f"match_doc failed: {e}")
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, f"match_doc failed: {e}")


@app.post(
    "/api/match-doc/jobs",
    status_code=202,
    response_model=MatchDocJobReceipt,
)
async def submit_match_paper_job(
    request: MatchDocRequest,
    http_request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Queue an exact paper match in the existing durable Tasks system."""
    if search_pipeline is None or match_paper_job_runner is None:
        raise HTTPException(503, "Search pipeline not initialized")
    if request.input_type != "paper_id":
        raise HTTPException(400, "Durable match-paper jobs require input_type=paper_id")
    owner = resolve_owner(http_request, user_id)
    try:
        return await asyncio.to_thread(
            partial(match_paper_job_runner.start, request, owner_user_id=owner),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _owned_match_job(task_id: int, owner: str) -> dict:
    """Load one match job that `owner` submitted, or 404.

    404 rather than 403 on an owner mismatch: task ids are sequential, so a 403
    would confirm that someone else's job exists at that id.
    """
    if match_paper_job_runner is None:
        raise HTTPException(503, "Search pipeline not initialized")
    task = match_paper_job_runner.store.get_task(task_id)
    if (
        task is None
        or task.get("task_type") != MATCH_PAPER_JOB
        or task.get("owner_user_id") != owner
    ):
        raise HTTPException(404, f"No match job {task_id}")
    return task


def _match_job_status(task: dict) -> MatchDocJobStatus:
    raw = task.get("result")
    return MatchDocJobStatus(
        task_id=int(task["id"]),
        status=str(task.get("status") or "unknown"),
        progress=int(task.get("progress") or 0),
        total=int(task.get("total") or 0),
        error=task.get("error"),
        # Only a completed job publishes a result; publish_result_if_running is
        # what guarantees a cancelled/timed-out job never carries one.
        result=(
            MatchDocResponse.model_validate(raw)
            if task.get("status") == "completed" and isinstance(raw, dict)
            else None
        ),
    )


@app.get("/api/match-doc/jobs/{task_id}", response_model=MatchDocJobStatus)
async def get_match_paper_job(
    task_id: int,
    http_request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Poll one durable match job. Scoped to the caller who submitted it."""
    owner = resolve_owner(http_request, user_id)
    task = await asyncio.to_thread(_owned_match_job, task_id, owner)
    return _match_job_status(task)


@app.post("/api/match-doc/jobs/{task_id}/cancel", response_model=MatchDocJobStatus)
async def cancel_match_paper_job(
    task_id: int,
    http_request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Request cancellation. Never publishes a partial result."""
    owner = resolve_owner(http_request, user_id)

    def _cancel() -> dict:
        _owned_match_job(task_id, owner)
        match_paper_job_runner.cancel(task_id)
        return _owned_match_job(task_id, owner)

    return _match_job_status(await asyncio.to_thread(_cancel))


# ==================== 嵌入向量（共享 BGE-M3 给外部知识库）====================
# 把进程内已暖的 BGE-M3 singleton 暴露成 HTTP，让外部知识库（MCP 代码库 / ACE
# playbook）复用同一个模型，而不是各自再 load 一份 ~6.4GB checkpoint。dense 默认
# 返回；sparse 可选（仅 BGE-M3 支持）。调用方自行分批（建议 ≤64/请求）。


class EmbedRequest(BaseModel):
    texts: List[str]
    include_sparse: bool = False


class EmbedResponse(BaseModel):
    model: str
    dim: int
    dense: List[List[float]]
    sparse: Optional[List[dict]] = None


@app.post("/api/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """用进程内共享的 BGE-M3 嵌入文本（dense [+ 可选 sparse]）。

    设计：外部代码知识库与 ACE playbook 通过本端点复用 backend 的暖模型（W46
    singleton），避免重复加载。空输入直接返回空向量列表，不触模型。
    """
    name = settings.embedding.model_name
    dim = settings.embedding.dimension
    if not request.texts:
        return EmbedResponse(model=name, dim=dim, dense=[], sparse=None)

    from core.indexing.embedders import embed_texts, get_embedder
    embedder = get_embedder()
    result = await asyncio.to_thread(embed_texts, request.texts)
    dense = [v.tolist() for v in result.dense]
    sparse = result.sparse if (request.include_sparse and embedder.supports_sparse) else None
    return EmbedResponse(model=name, dim=embedder.dimension, dense=dense, sparse=sparse)


# ==================== 文档管理 ====================
# 注意: 文档管理路由已移至 api/routes/documents.py
# 不要在这里重复定义，否则会覆盖数据库版本的路由


# ==================== 统计 ====================

def _compute_stats():
    from sqlalchemy import func, text
    from core.db.database import get_db_session
    from core.db.models import Document as DBDocument, Chunk as DBChunk, Paper as DBPaper

    with get_db_session() as db:
        total_documents = db.query(func.count(DBDocument.id)).scalar() or 0
        total_papers = db.query(func.count(DBPaper.paper_id)).scalar() or 0
        # COUNT(*) is the true row count; MAX(id) overshoots after delete cascades
        # (PK gaps). SQLite has internal row-count cache making COUNT fast on PK.
        total_chunks = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0

    from core.storage.corpus_store import paper_store
    try:
        qdrant_count = paper_store().count()
    except Exception:
        qdrant_count = 0

    coverage_ratio = (qdrant_count / total_chunks) if total_chunks else 0.0

    return {
        "total_documents": total_documents,
        "total_papers": total_papers,
        "total_chunks": total_chunks,
        "vector_index_size": qdrant_count,
        "coverage_ratio": round(coverage_ratio, 4),
        "embedding_model": settings.embedding.model_name,
        "embedding_dimension": settings.embedding.dimension,
        "supports_sparse": settings.embedding.supports_sparse,
    }


def _get_stats_sync():
    """COUNT(*) over the chunks table takes seconds: served from the shared stale-while-revalidate cache."""
    from core.cache import TOTAL_COUNT_TTL_SECONDS, stats_cache

    return stats_cache.get("api.stats", _compute_stats, ttl=TOTAL_COUNT_TTL_SECONDS)


@app.get("/api/stats")
async def get_stats():
    """获取索引统计信息"""
    import asyncio
    return await asyncio.to_thread(_get_stats_sync)


# ==================== 配置 ====================

@app.get("/api/config")
def get_config():
    """获取当前配置"""
    return {
        "proxy": {
            "http": settings.proxy.http_proxy
        },
        "embedding_model": settings.embedding.model_name,
        "chunk_sizes": settings.chunk.chunk_sizes,
        "retrieval": {
            "initial_top_k": settings.retrieval.initial_top_k,
            "final_top_n": settings.retrieval.final_top_n,
            "bm25_weight": settings.retrieval.bm25_weight,
            "vector_weight": settings.retrieval.vector_weight
        }
    }


@app.post("/api/config/proxy")
def update_proxy(http_proxy: str):
    """更新代理 URL（代理强制启用，不提供关闭开关）"""
    settings.proxy.http_proxy = http_proxy
    settings.proxy.https_proxy = http_proxy
    return {"success": True}


# ==================== 前端静态资源（按监听端口选择 SPA bundle）====================
# 部署态：同一个 FastAPI app 同时监听 internal/public 两个端口，因此不能只 mount
# frontend/dist。公共端口必须选择 VITE_PUBLIC_MODE=1 的独立构建；若缺失则返回 503，
# 绝不能把内部管理台作为降级结果泄露出去。所有 API 路由已经先注册，下面的 GET
# handlers 只服务静态文件和 SPA fallback。
from starlette.responses import FileResponse
from core.frontend_assets import FrontendAssetRoots, FrontendAssetsUnavailable
from app.api.identity import INTERNAL_API_PORT, PUBLIC_API_PORT

_frontend_dist = os.path.join(settings.base_dir, "frontend", "dist")
_public_frontend_dist = os.path.join(settings.base_dir, "frontend", "dist-public-code")
_frontend_assets = FrontendAssetRoots(
    internal_dir=Path(_frontend_dist),
    public_dir=Path(_public_frontend_dist),
    internal_port=INTERNAL_API_PORT,
    public_port=PUBLIC_API_PORT,
)


def _frontend_port(request: Request) -> int | None:
    server = request.scope.get("server")
    if not isinstance(server, (tuple, list)) or len(server) < 2:
        return None
    try:
        return int(server[1])
    except (TypeError, ValueError):
        return None


# Vite fingerprints every file under assets/, so those never change under a
# name; index.html is the one pointer that does. Without explicit headers a
# browser heuristically reused a stale index.html after a deploy and kept
# running the previous bundle.
_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
_SPA_CACHE_CONTROL = "no-cache"


def _frontend_unavailable(exc: FrontendAssetsUnavailable) -> HTTPException:
    logger.error("frontend bundle unavailable: %s", exc)
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/assets/{asset_path:path}", include_in_schema=False)
async def serve_frontend_asset(asset_path: str, request: Request):
    """Serve an asset from the bundle selected by the listener's local port."""
    try:
        asset = _frontend_assets.asset_file_for(port=_frontend_port(request), path=asset_path)
    except FrontendAssetsUnavailable as exc:
        raise _frontend_unavailable(exc) from exc
    if asset is None:
        raise HTTPException(status_code=404, detail="frontend asset not found")
    return FileResponse(asset, headers={"Cache-Control": _ASSET_CACHE_CONTROL})


@app.get("/assets", include_in_schema=False)
async def reject_frontend_asset_directory():
    """Keep a bare asset directory out of the SPA fallback and return a real 404."""
    raise HTTPException(status_code=404, detail="frontend asset not found")


@app.get("/{spa_path:path}", include_in_schema=False)
async def serve_spa(spa_path: str, request: Request):
    """Serve a selected bundle's static file or its SPA index fallback."""
    try:
        return FileResponse(_frontend_assets.spa_file_for(
            port=_frontend_port(request), path=spa_path),
            headers={"Cache-Control": _SPA_CACHE_CONTROL})
    except FrontendAssetsUnavailable as exc:
        raise _frontend_unavailable(exc) from exc


if __name__ == "__main__":
    import socket
    import uvicorn

    # Two doors on ONE app served by a SINGLE uvicorn.Server (via two pre-bound sockets):
    #   * INTERNAL 0.0.0.0:<INTERNAL_API_PORT> — trusted lab overlay
    #     (private-LAN peer = internal-admin).
    #   * PUBLIC 127.0.0.1:<PUBLIC_API_PORT> — loopback-only; the reverse xray tunnel lands
    #     here. app.api.identity refuses the internal-admin shortcut on this port — it reads
    #     the per-connection LOCAL port from scope["server"], so tunnelled callers must carry
    #     a gateway identity. See identity.LISTEN_ADDRESSES.
    # One Server (not two Servers on one app): the FastAPI lifespan — GPU model load,
    # resume_running() task relaunch (W67), the mem watchdog — runs exactly ONCE, and
    # uvicorn's own SIGTERM handling stops the single server cleanly (systemctl restart).
    from app.api.identity import LISTEN_ADDRESSES

    def _listen(host: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(2048)
        return s

    _sockets = [_listen(host, port) for host, port in LISTEN_ADDRESSES]
    # Connections that never send a request are closed (2026-10-07 fd exhaustion);
    # see app/api/idle_connections.py.
    from app.api.idle_connections import http_protocol_class

    uvicorn.Server(uvicorn.Config(app, http=http_protocol_class())).run(sockets=_sockets)
# reload

