"""
Paper API Routes - 论文检索 API 路由

提供与现有 retrival 系统 API 兼容的论文检索端点。
可以直接挂载到现有的 FastAPI 应用。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from core.storage.papers_payload import project_papers_payload


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


_INTEGRATION_DIR = Path(__file__).resolve().parent
_INDEX_INTEGRATION = _load_module_from_path(
    "_retrival_paper_index_integration",
    _INTEGRATION_DIR / "index_integration.py",
)

PaperIndexer = _INDEX_INTEGRATION.PaperIndexer
PaperIndexConfig = _INDEX_INTEGRATION.PaperIndexConfig
PaperSearchResult = _INDEX_INTEGRATION.PaperSearchResult
create_runtime_paper_indexer = _INDEX_INTEGRATION.create_runtime_paper_indexer
_get_paper_models = _INDEX_INTEGRATION._get_paper_models

DEFAULT_PAPER_API_PREFIX = "/api/papers"


def _get_paper_model_module():
    _get_paper_models()
    return _load_module_from_path(
        "_retrival_paper_models_runtime",
        _INTEGRATION_DIR.parent / "models" / "paper.py",
    )


def _coerce_enum(enum_cls, value, fallback):
    if value in enum_cls._value2member_map_:
        return enum_cls(value)
    return fallback


def _normalize_author_names(authors: Optional[List[Any]]) -> List[str]:
    normalized: List[str] = []
    for author in authors or []:
        if author is None:
            continue
        if isinstance(author, str):
            author_name = author.strip()
        elif isinstance(author, dict):
            author_name = str(author.get("name") or "").strip()
        else:
            author_name = str(getattr(author, "name", author) or "").strip()
        if author_name:
            normalized.append(author_name)
    return normalized


def _build_paper_point_payload(chunk_row: Dict[str, Any], paper_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build one `papers` Qdrant point payload for the single-paper reindex path.

    Routes through project_papers_payload (core/storage/papers_payload.py) —
    the collection's only executable schema. Do not hand-roll a key list
    here again: that is exactly how this function's previous inline dict
    carried a stale "9-field" comment that had drifted from
    local_ingest.PAYLOAD_KEEP_KEYS's 8 for months before the 2026-09-01
    audit caught it (docs/papers_payload_contract_plan.md 1.4). No
    `data_source` — D3: it is meaningless for a paper and its presence here
    is exactly what leaked onto ~1,327 production paper/review points.
    """
    raw = {
        "doc_id": chunk_row["doc_id"],
        "paper_id": paper_meta.get("paper_id"),
        "source_type": "paper",
        "level": chunk_row["level"],
        "venue": paper_meta.get("venue"),
        "year": paper_meta.get("year"),
        "venue_tier": paper_meta.get("venue_tier"),
        "content": chunk_row["content"],
    }
    return project_papers_payload(raw)


def _build_paper_from_request(request: "PaperIndexRequest"):
    Paper, PaperSource, PaperType, Conference = _get_paper_models()
    return Paper(
        id=request.paper_id,
        title=request.title,
        abstract=request.abstract,
        authors=_normalize_author_names(request.authors),
        year=request.year if request.year is not None else 2000,
        conference=_coerce_enum(Conference, request.conference, Conference.OTHER),
        source=_coerce_enum(PaperSource, request.source, PaperSource.OTHER),
        paper_page_url=request.url,
        pdf_url=request.pdf_url,
        paper_type=_coerce_enum(PaperType, request.paper_type, PaperType.UNKNOWN),
        keywords=list(request.keywords or []),
    )


# ==================== Request/Response Models ====================


class PaperSearchRequest(BaseModel):
    """论文搜索请求"""

    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=20, ge=1, le=100, description="返回数量")
    method: str = Field(default="hybrid", description="检索方法: bm25/vector/hybrid")
    aggregate_by_paper: bool = Field(default=True, description="是否按论文聚合")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件")


class PaperSearchResultResponse(BaseModel):
    """论文搜索结果响应项"""

    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    source: str
    venue: Optional[str]
    year: Optional[int]
    paper_type: str
    score: float
    hit_count: int
    pdf_url: Optional[str]
    url: Optional[str]
    matched_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class PaperSearchResponse(BaseModel):
    """论文搜索响应"""

    query: str
    method: str
    total_results: int
    papers: List[PaperSearchResultResponse]


class PaperIndexRequest(BaseModel):
    """论文索引请求"""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    source: str = "arxiv"
    url: str
    pdf_url: Optional[str] = None
    markdown_path: Optional[str] = None
    content: Optional[str] = None
    conference: str = "Other"
    venue: Optional[str] = None
    year: Optional[int] = None
    paper_type: str = "unknown"
    keywords: List[str] = Field(default_factory=list)


class PaperIndexResponse(BaseModel):
    """论文索引响应"""

    success: bool
    paper_id: str
    doc_id: Optional[str] = None
    message: str


class BatchIndexRequest(BaseModel):
    """批量索引请求"""

    papers: List[PaperIndexRequest]
    markdown_dir: Optional[str] = None


class BatchIndexResponse(BaseModel):
    """批量索引响应"""

    success: bool
    total: int
    indexed: int
    failed: int
    results: Dict[str, Optional[str]]


class PaperBatchDownloadRequest(BaseModel):
    """Download up to ten papers as one ZIP archive."""

    paper_ids: List[str] = Field(min_length=1, max_length=10)
    artifacts: Literal["pdf", "markdown", "both"] = "pdf"


@dataclass(frozen=True)
class PaperBatchArchive:
    path: Path
    filename: str
    file_count: int
    error_count: int


@dataclass(frozen=True)
class PaperBatchSource:
    paper_id: str
    artifact: str
    path: Path
    filename: str
    size: int
    compression: int


@dataclass(frozen=True)
class ResolvedPaperArtifact:
    path: Path
    title: str | None


@dataclass(frozen=True)
class PaperArtifactPolicy:
    stored_path: str | None
    roots: Sequence[str]
    suffix: str
    compression: int
    missing_path_detail: str
    missing_file_detail: str


class PaperBatchTooLarge(ValueError):
    pass


_PAPER_BATCH_MAX_SOURCE_BYTES = 1024 * 1024 * 1024
_PAPER_BATCH_BUILD_SLOTS = threading.BoundedSemaphore(value=2)

# Index coverage changes only when an ingest lands, so a short TTL is plenty and
# keeps a cold 2.67s scan off the search page's critical path.
_COVERAGE_TTL_SECONDS = 300.0
_coverage_cache: tuple[float, Dict[str, Any]] | None = None
_coverage_lock = threading.Lock()


def _compute_index_coverage() -> Dict[str, Any]:
    """One GROUP BY (venue, year); venue totals and year spans derived in memory.

    Blocking SQL — call it from a worker thread, never the event loop. Papers
    with no venue or no year are counted in `unclassified` rather than dropped,
    so the card totals always reconcile against /api/papers/stats. Production
    currently has none of either (measured 2026-09-21), but a future non-CCF
    ingest would otherwise silently shrink the numbers.
    """
    from sqlalchemy import func
    from core.db.database import get_db_session
    from core.db.models import Paper as DBPaper

    with get_db_session() as db:
        rows = (
            db.query(DBPaper.venue, DBPaper.year, func.count(DBPaper.paper_id))
            .group_by(DBPaper.venue, DBPaper.year)
            .all()
        )

    venues: Dict[str, Dict[str, Any]] = {}
    unclassified = 0
    for venue, year, count in rows:
        if not venue or year is None:
            unclassified += int(count)
            continue
        entry = venues.setdefault(venue, {"venue": venue, "count": 0, "years": {}})
        entry["count"] += int(count)
        entry["years"][int(year)] = entry["years"].get(int(year), 0) + int(count)

    out: List[Dict[str, Any]] = []
    for entry in venues.values():
        years = entry["years"]
        ordered = sorted(years)
        out.append({
            "venue": entry["venue"],
            "count": entry["count"],
            "min_year": ordered[0],
            "max_year": ordered[-1],
            # Contiguous span check: a gap means a year was never crawled, which
            # the card should be able to show rather than implying full coverage
            # between the endpoints.
            "missing_years": [y for y in range(ordered[0], ordered[-1] + 1)
                              if y not in years],
            "years": [{"year": y, "count": years[y]} for y in ordered],
        })
    out.sort(key=lambda v: v["count"], reverse=True)

    return {
        "venues": out,
        "total_papers": sum(v["count"] for v in out) + unclassified,
        "unclassified": unclassified,
        "venue_count": len(out),
        "min_year": min((v["min_year"] for v in out), default=None),
        "max_year": max((v["max_year"] for v in out), default=None),
    }


def _index_coverage_cached() -> Dict[str, Any]:
    """TTL-cached wrapper. The lock keeps a cold start from running N scans."""
    global _coverage_cache
    now = time.monotonic()
    cached = _coverage_cache
    if cached is not None and now - cached[0] < _COVERAGE_TTL_SECONDS:
        return cached[1]
    with _coverage_lock:
        cached = _coverage_cache
        if cached is not None and time.monotonic() - cached[0] < _COVERAGE_TTL_SECONDS:
            return cached[1]
        payload = _compute_index_coverage()
        _coverage_cache = (time.monotonic(), payload)
        return payload


def _cleanup_paper_batch_archive(
    path: Path,
    slot: threading.BoundedSemaphore,
) -> None:
    """Delete one generated archive before making its capacity slot reusable."""
    try:
        path.unlink(missing_ok=True)
    finally:
        slot.release()


class PaperStatsResponse(BaseModel):
    """论文索引统计响应"""

    total_papers: int
    total_documents: int
    total_chunks: int
    chunk_levels: Dict[int, int]
    vector_index_size: int
    bm25_index_size: int
    collection_name: str


def _paper_archive_stem(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._") or "paper"


def _unique_archive_name(
    paper_id: str,
    suffix: str,
    used_names: set[str],
) -> str:
    stem = _paper_archive_stem(paper_id)
    candidate = f"{stem}{suffix}"
    if candidate in used_names:
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:8]
        candidate = f"{stem}-{digest}{suffix}"
    used_names.add(candidate)
    return candidate


def _paper_artifact_policy(
    paper,
    artifact: Literal["pdf", "markdown"],
    settings,
) -> PaperArtifactPolicy:
    if artifact == "pdf":
        return PaperArtifactPolicy(
            stored_path=paper.pdf_path,
            roots=settings.paper_pdf_roots,
            suffix=".pdf",
            compression=zipfile.ZIP_STORED,
            missing_path_detail="PDF not found",
            missing_file_detail="PDF file not found on disk",
        )
    return PaperArtifactPolicy(
        stored_path=paper.txt_path,
        roots=settings.paper_markdown_roots,
        suffix=".md",
        compression=zipfile.ZIP_DEFLATED,
        missing_path_detail="Markdown file path not set",
        missing_file_detail="Markdown file not found on disk",
    )


def _resolve_paper_artifact_path(
    paper,
    artifact: Literal["pdf", "markdown"],
    settings,
) -> tuple[PaperArtifactPolicy, Path | None]:
    from core.storage.artifact_resolver import resolve_paper_artifact
    from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component

    policy = _paper_artifact_policy(paper, artifact, settings)
    resolved = resolve_paper_artifact(
        policy.stored_path,
        paper.venue,
        paper.year,
        policy.roots,
        fallback_basename=(
            f"{sanitize_file_component(paper.paper_id)}{policy.suffix}"
        ),
    )
    return policy, Path(resolved) if resolved else None


def _create_paper_batch_archive(request: PaperBatchDownloadRequest) -> PaperBatchArchive:
    """Resolve requested paper artifacts and build a temporary ZIP plus manifest."""
    from config import settings
    from core.db.database import get_db_session
    from core.db.models import Paper as DBPaper

    paper_ids = list(dict.fromkeys(paper_id.strip() for paper_id in request.paper_ids))
    if any(not paper_id for paper_id in paper_ids):
        raise ValueError("paper_ids cannot contain blank values")
    artifact_kinds = (
        ("pdf", "markdown") if request.artifacts == "both" else (request.artifacts,)
    )

    with get_db_session() as db:
        papers = db.query(DBPaper).filter(DBPaper.paper_id.in_(paper_ids)).all()
        papers_by_id = {paper.paper_id: paper for paper in papers}

    errors: list[dict[str, str]] = []
    used_names: set[str] = set()
    sources: list[PaperBatchSource] = []
    total_source_bytes = 0
    for paper_id in paper_ids:
        paper = papers_by_id.get(paper_id)
        if paper is None:
            errors.append({"paper_id": paper_id, "error": "paper not found"})
            continue
        for artifact_kind in artifact_kinds:
            policy, resolved_path = _resolve_paper_artifact_path(
                paper, artifact_kind, settings)
            if not resolved_path:
                errors.append({
                    "paper_id": paper_id,
                    "artifact": artifact_kind,
                    "error": "artifact not found",
                })
                continue
            source_path = Path(resolved_path)
            source_size = source_path.stat().st_size
            total_source_bytes += source_size
            if total_source_bytes > _PAPER_BATCH_MAX_SOURCE_BYTES:
                raise PaperBatchTooLarge(
                    f"Requested source artifacts total {total_source_bytes} bytes, which "
                    f"exceeds the {_PAPER_BATCH_MAX_SOURCE_BYTES}-byte batch limit."
                )
            sources.append(PaperBatchSource(
                paper_id=paper_id,
                artifact=artifact_kind,
                path=resolved_path,
                filename=_unique_archive_name(paper_id, policy.suffix, used_names),
                size=source_size,
                compression=policy.compression,
            ))

    descriptor, archive_name = tempfile.mkstemp(
        prefix="retrieval-paper-batch-", suffix=".zip")
    os.close(descriptor)
    archive_path = Path(archive_name)
    files = [{
        "paper_id": source.paper_id,
        "artifact": source.artifact,
        "filename": source.filename,
        "size": source.size,
    } for source in sources]
    try:
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            for source in sources:
                archive.write(
                    source.path,
                    arcname=source.filename,
                    compress_type=source.compression,
                )
            manifest = {
                "paper_ids": paper_ids,
                "artifacts": request.artifacts,
                "files": files,
                "errors": errors,
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise

    return PaperBatchArchive(
        path=archive_path,
        filename=f"papers-{len(paper_ids)}-{request.artifacts}.zip",
        file_count=len(files),
        error_count=len(errors),
    )


def _resolve_single_paper_artifact(
    paper_id: str,
    artifact: Literal["pdf", "markdown"],
) -> ResolvedPaperArtifact:
    from config import settings
    from core.db.database import get_db_session
    from core.db.models import Paper as DBPaper

    with get_db_session() as db:
        paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_id).first()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        policy, resolved_path = _resolve_paper_artifact_path(
            paper, artifact, settings)
        if not resolved_path:
            detail = (policy.missing_file_detail if policy.stored_path
                      else policy.missing_path_detail)
            raise HTTPException(status_code=404, detail=detail)
        return ResolvedPaperArtifact(path=resolved_path, title=paper.title)


# ==================== Router ====================


def create_paper_router(
    indexer: Optional[PaperIndexer] = None,
    prefix: str = DEFAULT_PAPER_API_PREFIX,
) -> APIRouter:
    """创建论文 API 路由

    Args:
        indexer: 可选的 PaperIndexer 实例。如果为 None，将创建新实例
        prefix: 路由前缀

    Returns:
        FastAPI APIRouter
    """
    router = APIRouter(prefix=prefix, tags=["papers"])

    # 使用闭包保存 indexer 实例
    _indexer: Optional[PaperIndexer] = indexer

    def get_indexer() -> PaperIndexer:
        nonlocal _indexer
        if _indexer is None:
            _indexer = create_runtime_paper_indexer()
        return _indexer

    @router.post("/search", response_model=PaperSearchResponse)
    async def search_papers(request: PaperSearchRequest):
        """搜索论文

        使用混合检索方法搜索已索引的论文。
        支持按论文聚合结果和多种过滤条件。
        """
        import asyncio
        idx = get_indexer()

        results = await asyncio.to_thread(
            idx.search_papers,
            query=request.query,
            top_k=request.top_k,
            method=request.method,
            filters=request.filters,
            aggregate_by_paper=request.aggregate_by_paper,
        )

        # 转换为响应格式
        papers = []
        for r in results:
            papers.append(
                PaperSearchResultResponse(
                    paper_id=r.paper_id,
                    title=r.title,
                    authors=r.authors,
                    abstract=r.abstract,
                    source=r.source,
                    venue=r.conference,
                    year=r.year,
                    paper_type=r.paper_type,
                    score=r.score,
                    hit_count=r.hit_count,
                    pdf_url=r.pdf_url,
                    url=r.paper_page_url,
                    matched_chunks=[
                        {
                            "chunk_id": c.chunk_id,
                            "content": c.content[:500],
                            "score": c.score,
                        }
                        for c in r.matched_chunks[:3]
                    ],
                )
            )

        return PaperSearchResponse(
            query=request.query,
            method=request.method,
            total_results=len(results),
            papers=papers,
        )

    @router.get("/search")
    async def search_papers_get(
        query: str = Query(..., description="搜索查询"),
        top_k: int = Query(default=20, ge=1, le=100),
        method: str = Query(default="hybrid"),
        year: Optional[int] = Query(default=None, description="按年份过滤"),
        source: Optional[str] = Query(default=None, description="按来源过滤"),
    ):
        """GET 方式搜索论文（便捷接口）"""
        import asyncio
        idx = get_indexer()

        filters = {}
        if year:
            filters["year"] = year
        if source:
            filters["source"] = source

        results = await asyncio.to_thread(
            idx.search_papers,
            query=query,
            top_k=top_k,
            method=method,
            filters=filters if filters else None,
        )

        return {
            "query": query,
            "total": len(results),
            "papers": [r.to_dict() for r in results],
        }

    @router.post("/index", response_model=PaperIndexResponse)
    async def index_paper(request: PaperIndexRequest):
        """索引单篇论文"""
        idx = get_indexer()

        try:
            paper = _build_paper_from_request(request)

            # 索引
            markdown_path = Path(request.markdown_path) if request.markdown_path else None
            doc_id = await idx.index_paper(
                paper=paper,
                markdown_path=markdown_path,
                content=request.content,
            )

            return PaperIndexResponse(
                success=True,
                paper_id=request.paper_id,
                doc_id=doc_id,
                message="Paper indexed successfully",
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Indexing failed: {str(e)}") from e

    @router.post("/index/batch", response_model=BatchIndexResponse)
    async def index_papers_batch(request: BatchIndexRequest):
        """批量索引论文"""
        idx = get_indexer()

        try:
            papers = []
            contents = {}

            for req in request.papers:
                paper = _build_paper_from_request(req)
                papers.append(paper)

                if req.content:
                    contents[req.paper_id] = req.content

            markdown_dir = Path(request.markdown_dir) if request.markdown_dir else None

            results = await idx.index_batch(
                papers=papers,
                markdown_dir=markdown_dir,
                contents=contents if contents else None,
            )

            indexed = len([v for v in results.values() if v])
            failed = len(papers) - indexed

            return BatchIndexResponse(
                success=failed == 0,
                total=len(papers),
                indexed=indexed,
                failed=failed,
                results=results,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Batch indexing failed: {str(e)}") from e

    @router.get("/tags")
    async def get_tags():
        """获取所有可用标签（会议/年份/tier/track/impact）用于筛选"""
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper
        from sqlalchemy import func, distinct

        with get_db_session() as db:
            venues = [r[0] for r in db.query(distinct(DBPaper.venue)).filter(DBPaper.venue != None).all()]
            years = sorted([r[0] for r in db.query(distinct(DBPaper.year)).filter(DBPaper.year != None).all()])
            tiers = [r[0] for r in db.query(distinct(DBPaper.venue_tier)).filter(DBPaper.venue_tier != None).all()]
            tracks = [r[0] for r in db.query(distinct(DBPaper.track_type)).filter(DBPaper.track_type != None).all()]
            impacts = [r[0] for r in db.query(distinct(DBPaper.impact_level)).filter(DBPaper.impact_level != None).all()]
            total = db.query(func.count(DBPaper.paper_id)).scalar() or 0

            venue_counts = {v: c for v, c in db.query(DBPaper.venue, func.count(DBPaper.paper_id)).group_by(DBPaper.venue).all()}
            year_counts = {y: c for y, c in db.query(DBPaper.year, func.count(DBPaper.paper_id)).group_by(DBPaper.year).all()}

        return {
            "total_papers": total,
            "venues": sorted(venues),
            "years": years,
            "venue_tiers": sorted(tiers),
            "track_types": sorted(tracks),
            "impact_levels": sorted(impacts),
            "venue_counts": venue_counts,
            "year_counts": dict(sorted(year_counts.items())),
        }

    @router.get("/browse")
    async def browse_papers(
        venue: Optional[List[str]] = Query(None, description="Filter by venue(s) — repeat (?venue=A&venue=B) or comma-separated (?venue=A,B)"),
        year: Optional[List[str]] = Query(None, description="Filter by year(s) — repeat or comma-separated; values must be int-castable"),
        venue_tier: Optional[str] = Query(None, description="Filter by tier (e.g. CCF-A)"),
        track_type: Optional[str] = Query(None, description="Filter by track (e.g. oral, poster, main)"),
        impact_level: Optional[str] = Query(None, description="Filter by impact (e.g. best_paper)"),
        q: Optional[str] = Query(None, description="Keyword — matched against title + abstract (case-insensitive substring)"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=1000, description="Results per page"),
    ):
        """按标签组合 + 关键词浏览论文（SQL 全集合筛选，非向量检索）"""
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper
        from sqlalchemy import or_

        # Accept both `venue=A&venue=B` and `venue=A,B` (single-string CSV) forms.
        def _split_csv(values, cast):
            if not values:
                return None
            out: List[Any] = []
            for v in values:
                if isinstance(v, str) and "," in v:
                    out.extend(cast(p.strip()) for p in v.split(",") if p.strip())
                else:
                    out.append(cast(v))
            return out or None

        venue_list = _split_csv(venue, str)
        year_list = _split_csv(year, int)

        with get_db_session() as db:
            qy = db.query(DBPaper)
            if venue_list:
                qy = qy.filter(DBPaper.venue.in_(venue_list))
            if year_list:
                qy = qy.filter(DBPaper.year.in_(year_list))
            if venue_tier:
                qy = qy.filter(DBPaper.venue_tier == venue_tier)
            if track_type:
                qy = qy.filter(DBPaper.track_type == track_type)
            if impact_level:
                qy = qy.filter(DBPaper.impact_level == impact_level)
            if q and q.strip():
                kw = f"%{q.strip()}%"
                qy = qy.filter(or_(DBPaper.title.ilike(kw), DBPaper.abstract.ilike(kw)))

            total = qy.count()
            papers = qy.order_by(DBPaper.year.desc(), DBPaper.venue, DBPaper.title)\
                      .offset((page - 1) * page_size)\
                      .limit(page_size)\
                      .all()

            results = []
            for p in papers:
                results.append({
                    "paper_id": p.paper_id,
                    "doc_id": p.doc_id,
                    "title": p.title,
                    "authors": p.authors_json or [],
                    "abstract": (p.abstract or "")[:300],
                    "venue": p.venue,
                    "year": p.year,
                    "venue_tier": p.venue_tier,
                    "track_type": p.track_type,
                    "impact_level": p.impact_level,
                    "venue_source": p.venue_source,
                    "citation_count": p.citation_count or 0,
                    "pdf_path": p.pdf_path,
                    "txt_path": p.txt_path,
                })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "results": results,
        }

    @router.get("/coverage")
    async def get_index_coverage():
        """每个 venue 索引到了哪些年份，以及各年多少篇。

        `/api/papers/stats` 只有全局总数，答不了「NeurIPS 索引到哪年了」。这里一次
        `GROUP BY venue, year` 取回全部 81 个 (venue, year) 单元格（实测 0.12s，
        18 个 venue / 2021-2026），venue 级的 count 与 min/max year 在内存里派生，
        不再多打一次 SQL。

        结果缓存 `_COVERAGE_TTL_SECONDS`：这份数据只在 ingest 之后变，而首次冷查询
        要 2.67s（页缓存未热），不该让每次打开检索页都付这个钱。
        """
        import asyncio
        return await asyncio.to_thread(_index_coverage_cached)

    @router.get("/stats", response_model=PaperStatsResponse)
    async def get_stats():
        """获取论文索引统计信息

        Stage C 等大型 ingest 期间 vector store count 可能 60s 阻塞 (W8)。
        `_cached_vector_count` 内部 2s probe timeout + 30s 缓存 + stale 兜底，
        endpoint 整体在 to_thread 中跑避免阻塞 FastAPI event loop。
        """
        import asyncio
        idx = get_indexer()
        stats = await asyncio.to_thread(idx.get_stats)
        return PaperStatsResponse(**stats)

    @router.get("/", response_model=List[str])
    async def list_indexed_papers():
        """列出所有已索引的论文 ID"""
        idx = get_indexer()
        return idx.get_indexed_paper_ids()

    @router.post(
        "/download-batch",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP archive containing the requested artifacts.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"},
                    },
                },
                "headers": {
                    "Accept-Ranges": {
                        "schema": {"type": "string"},
                    },
                    "X-Retrieval-Artifact-Count": {
                        "schema": {"type": "integer"},
                    },
                    "X-Retrieval-Error-Count": {
                        "schema": {"type": "integer"},
                    },
                },
            },
            413: {"description": "Requested source artifacts exceed 1 GiB."},
            416: {
                "description": "Range is unsupported for generated archives.",
                "headers": {
                    "Accept-Ranges": {"schema": {"type": "string"}},
                },
            },
            429: {
                "description": "All archive build/stream slots are busy.",
                "headers": {
                    "Retry-After": {"schema": {"type": "integer"}},
                },
            },
        },
    )
    async def download_papers_batch(
        request: PaperBatchDownloadRequest,
        http_request: Request,
    ):
        """Build one ZIP containing the requested original paper artifacts."""
        if http_request.headers.get("range"):
            raise HTTPException(
                status_code=416,
                detail="Range requests are not supported for generated batch archives.",
                headers={"Accept-Ranges": "none"},
            )
        slot = _PAPER_BATCH_BUILD_SLOTS
        if not slot.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Paper batch archive capacity is busy; retry later.",
                headers={"Retry-After": "2"},
            )
        try:
            try:
                bundle = await asyncio.to_thread(_create_paper_batch_archive, request)
            except PaperBatchTooLarge as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BaseException:
            slot.release()
            raise
        return FileResponse(
            path=bundle.path,
            media_type="application/zip",
            filename=bundle.filename,
            headers={
                "Accept-Ranges": "none",
                "X-Retrieval-Artifact-Count": str(bundle.file_count),
                "X-Retrieval-Error-Count": str(bundle.error_count),
            },
            background=BackgroundTask(_cleanup_paper_batch_archive, bundle.path, slot),
        )

    @router.get("/{paper_id}/pdf")
    async def get_paper_pdf(paper_id: str):
        """Serve paper PDF for inline preview (paper_id is the PK string)"""
        artifact = _resolve_single_paper_artifact(paper_id, "pdf")
        return FileResponse(
            path=artifact.path,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )

    @router.get(
        "/{paper_id}/markdown",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Original extracted Markdown file.",
                "content": {
                    "text/markdown": {
                        "schema": {"type": "string", "format": "binary"},
                    },
                },
            },
            404: {"description": "Paper or Markdown artifact not found."},
        },
    )
    async def get_paper_markdown_file(paper_id: str):
        """Stream the original extracted Markdown file with HTTP Range support."""
        artifact = _resolve_single_paper_artifact(paper_id, "markdown")
        return FileResponse(
            path=artifact.path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{_paper_archive_stem(paper_id)}.md",
        )

    @router.get("/{paper_id}/bibtex")
    async def get_paper_bibtex(paper_id: str):
        """Generate BibTeX citation for a paper."""
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper
        import json as _json

        with get_db_session() as db:
            paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_id).first()
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")

            authors_raw = paper.authors_json
            if isinstance(authors_raw, str):
                try:
                    authors_list = _json.loads(authors_raw)
                except Exception:
                    authors_list = [a.strip() for a in authors_raw.split("|") if a.strip()]
            elif isinstance(authors_raw, list):
                authors_list = authors_raw
            else:
                authors_list = []

            author_str = " and ".join(authors_list) if authors_list else "Unknown"
            title = (paper.title or "").replace("{", "\\{").replace("}", "\\}")
            venue = paper.venue or "Unknown"
            year = paper.year or ""
            doi = paper.doi or ""

            bib_key = paper_id.replace("/", "_").replace(":", "_")
            lines = [f"@inproceedings{{{bib_key},"]
            lines.append(f"  title = {{{title}}},")
            lines.append(f"  author = {{{author_str}}},")
            lines.append(f"  booktitle = {{Proceedings of {venue} {year}}},")
            lines.append(f"  year = {{{year}}},")
            if doi:
                lines.append(f"  doi = {{{doi}}},")
            lines.append("}")
            bibtex = "\n".join(lines)

            return {"bibtex": bibtex}

    @router.get("/{paper_id}/fulltext")
    async def get_paper_fulltext(paper_id: str):
        """Return the full markdown text of a paper."""
        artifact = _resolve_single_paper_artifact(paper_id, "markdown")
        content = await asyncio.to_thread(
            artifact.path.read_text,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "paper_id": paper_id,
            "title": artifact.title,
            "content": content,
            "size": len(content),
        }

    @router.get("/{paper_id}/index-status")
    async def get_paper_index_status(paper_id: str):
        """返回论文索引状态：SQL chunks 数 + Qdrant 向量数 + 是否已索引"""
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper, Chunk as DBChunk
        from core.storage.qdrant_store import QdrantStore
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        with get_db_session() as db:
            paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_id).first()
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")

            sql_chunks_count = (
                db.query(DBChunk).filter(DBChunk.doc_id == paper.doc_id).count()
            )

        qdrant_points_count = 0
        try:
            qs = QdrantStore(collection_name="papers")
            # W24: doc_id payload is split between str (Stage A-E ~6.68M) and
            # int (Stage F ~2.49M). Use paper_id (always str, canonical) so the
            # filter is invariant across batches.
            qdrant_points_count = qs.client.count(
                qs.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper.paper_id))]
                ),
                exact=True,
            ).count
        except Exception as e:  # noqa: BLE001
            print(f"[index-status] Qdrant count failed for paper_id={paper.paper_id}: {e}", flush=True)

        return {
            "paper_id": paper_id,
            "doc_id": paper.doc_id,
            "sql_chunks_count": sql_chunks_count,
            "qdrant_points_count": qdrant_points_count,
            "indexed": qdrant_points_count > 0,
        }

    @router.post("/{paper_id}/index")
    async def index_paper_by_id(paper_id: str):
        """按论文 ID 触发向量索引（仅对已切 chunks 但缺 Qdrant 向量的情况）。

        Safe mode: 不动 SQLite papers/documents/chunks，只读 chunks 表 + 写 Qdrant。
        - 已索引（Qdrant 有 points）→ 409 Conflict
        - SQL chunks 缺失 → 400（用户应该先走 markdown ingestion）
        - SQL chunks 有 + Qdrant 无 → 触发批量 embed + Qdrant upsert
        """
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper, Chunk as DBChunk
        from core.storage.qdrant_store import QdrantStore
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        with get_db_session() as db:
            paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_id).first()
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            doc_id = paper.doc_id
            sql_chunks_count = (
                db.query(DBChunk).filter(DBChunk.doc_id == doc_id).count()
            )

        if sql_chunks_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Paper has no SQL chunks; ingest markdown first via /api/papers/index (batch)",
            )

        qs = QdrantStore(collection_name="papers")
        existing = qs.client.count(
            qs.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))]
            ),
            exact=True,
        ).count
        if existing > 0:
            return {
                "paper_id": paper_id,
                "doc_id": doc_id,
                "indexed": True,
                "status": "already_indexed",
                "qdrant_points_count": existing,
                "sql_chunks_count": sql_chunks_count,
            }

        # Safe path: read chunks from SQL, embed, upsert to Qdrant. No DB writes.
        from core.storage.qdrant_store import (
            _chunk_id_to_int,
            generate_chunk_id,
            sparse_dict_to_qdrant,
        )
        from qdrant_client.models import PointStruct, SparseVector

        idx = get_indexer()
        embedder = idx.new_embedder
        try:
            with get_db_session() as db:
                chunk_rows = db.query(DBChunk).filter(DBChunk.doc_id == doc_id).all()
                # Snapshot all SQL state needed for payload BEFORE session closes
                chunks_snapshot = [
                    {
                        "doc_id": c.doc_id, "content": c.content, "level": c.level,
                        "start_idx": c.start_idx, "end_idx": c.end_idx,
                    }
                    for c in chunk_rows
                ]
                paper_meta = {
                    "paper_id": paper.paper_id,
                    "venue": paper.venue,
                    "year": paper.year,
                    "venue_tier": paper.venue_tier,
                }
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Read SQL chunks failed: {e}") from e

        texts = [c["content"] for c in chunks_snapshot]
        try:
            result = embedder.embed_texts(texts)
            chunks_embedded = len(result.dense)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Embed failed: {e}") from e

        # Build PointStructs matching the same point_id + payload schema as
        # local_ingest.py (parquet path). Single source of truth: SQL Paper.
        sparse_list = result.sparse or [None] * len(chunks_snapshot)
        points: list[PointStruct] = []
        for i, c in enumerate(chunks_snapshot):
            chunk_id = generate_chunk_id(
                doc_id=c["doc_id"], content=c["content"],
                level=c["level"], start_idx=c["start_idx"],
            )
            payload = _build_paper_point_payload(c, paper_meta)

            sparse_d = sparse_list[i]
            if sparse_d:
                sparse_v = sparse_dict_to_qdrant(sparse_d)
            else:
                sparse_v = SparseVector(indices=[], values=[])

            points.append(PointStruct(
                id=_chunk_id_to_int(chunk_id),
                vector={
                    "dense": result.dense[i].tolist(),
                    "sparse": sparse_v,
                },
                payload=payload,
            ))

        # Batched upsert with wait=True — every batch ACKed by Qdrant before next.
        # This is what D3 got wrong (wait=False → 56% silent drop).
        BATCH = 200
        upserted = 0
        try:
            for s in range(0, len(points), BATCH):
                qs.client.upsert(
                    qs.collection_name,
                    points=points[s:s + BATCH],
                    wait=True,
                )
                upserted += min(BATCH, len(points) - s)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Qdrant upsert failed after {upserted}/{len(points)}: {e}",
            ) from e

        return {
            "paper_id": paper_id,
            "doc_id": doc_id,
            "indexed": True,
            "status": "indexed",
            "chunks_embedded": chunks_embedded,
            "chunks_upserted": upserted,
            "qdrant_points_count": upserted,
            "sql_chunks_count": sql_chunks_count,
        }

    @router.get("/{paper_id}")
    async def get_paper(paper_id: str):
        """获取论文详情（paper_id 是 PK string，直接精确查找）"""
        from core.db.database import get_db_session
        from core.db.models import Paper as DBPaper, Document as DBDocument, Chunk as DBChunk

        with get_db_session() as db:
            paper = db.query(DBPaper).filter(DBPaper.paper_id == paper_id).first()
            if paper is None:
                idx = get_indexer()
                result = idx.get_paper_by_id(paper_id)
                if result:
                    return result
                raise HTTPException(status_code=404, detail="Paper not found")

            document = db.query(DBDocument).filter(DBDocument.id == paper.doc_id).first()
            chunks = (
                db.query(DBChunk).filter(DBChunk.doc_id == paper.doc_id)
                .order_by(DBChunk.chunk_index.asc()).limit(50).all()
            )

            return {
                "paper_id": paper_id,
                "doc_id": paper.doc_id,
                "title": paper.title,
                "source": document.source if document else None,
                "chunk_count": len(chunks),
                "metadata": {
                    "authors": paper.authors_json or [],
                    "venue": paper.venue,
                    "year": paper.year,
                    "abstract": paper.abstract or "",
                    "venue_tier": paper.venue_tier,
                    "track_type": paper.track_type,
                    "impact_level": paper.impact_level,
                    "citation_count": paper.citation_count or 0,
                    "venue_source": paper.venue_source,
                    "pdf_path": paper.pdf_path,
                    "txt_path": paper.txt_path,
                },
                "chunks": [
                    {
                        "id": c.id,
                        "content": c.content,
                        "level": c.level,
                        "chunk_index": c.chunk_index,
                    }
                    for c in chunks
                ],
            }

    @router.delete("/{paper_id}")
    async def delete_paper(paper_id: str):
        """删除论文索引"""
        idx = get_indexer()
        success = idx.delete_paper(paper_id)
        if not success:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {"success": True, "paper_id": paper_id}

    return router


# ==================== Integration Helper ====================


def mount_paper_routes(app, indexer: Optional[PaperIndexer] = None):
    """将论文路由挂载到现有的 FastAPI 应用

    Args:
        app: FastAPI 应用实例
        indexer: 可选的 PaperIndexer 实例

    Example:
        from fastapi import FastAPI
        from pipeline.paper_crawler.integration.api_routes import mount_paper_routes

        app = FastAPI()
        mount_paper_routes(app)
    """
    router = create_paper_router(indexer=indexer)
    app.include_router(router)
