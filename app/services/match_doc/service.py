"""编排层（功能2）：adapt → 逐查询块检索 → 论文级聚合 → MatchDocResponse。

检索通过注入的 `search_fn(query, *, top_k, filters) -> list[SearchResult]` 进来，
生产环境绑到进程内已暖的 `SearchPipeline`（复用 BGE-M3 + Qdrant + bge-reranker，
dedup_by_paper=False 让聚合器自己按论文分组）。注入式 → 离线可测、对检索业务零侵入。

paper_id 输入会把查询论文自身从结果里排除（找邻居不该匹配自己）。
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Callable, Optional

from app.models.schemas import SearchResult
from app.services.match_doc.adapters import (
    LoadPaperChunks,
    LoadUploadMd,
    build_query_chunks,
)
from app.services.match_doc.aggregator import aggregate_matches
from app.services.match_doc.schemas import MatchDocRequest, MatchDocResponse
from core.storage.qdrant_store import QdrantRetrievalCancelledError

_logger = logging.getLogger(__name__)

SearchFn = Callable[..., list[SearchResult]]


class MatchDocService:
    """doc → 最相近论文匹配的编排器。"""

    def __init__(self, *, search_fn: SearchFn, chunker, batch_search_fn=None,
                 load_paper_chunks: Optional[LoadPaperChunks] = None,
                 load_upload_md: Optional[LoadUploadMd] = None,
                 load_paper_meta: Optional[Callable[[list], dict]] = None):
        self._search_fn = search_fn
        self._batch_search_fn = batch_search_fn
        self._chunker = chunker
        self._load_paper_chunks = load_paper_chunks
        self._load_upload_md = load_upload_md
        # 标题不在 Qdrant 8 字段白名单里 → L2 chunk 命中无 title。可选 SQL 回填
        # title/venue/year（按 paper_id 批量），让结果卡片显示真实标题。
        self._load_paper_meta = load_paper_meta

    def match(
        self,
        req: MatchDocRequest,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> MatchDocResponse:
        t0 = time.perf_counter()
        if should_cancel is not None and should_cancel():
            raise QdrantRetrievalCancelledError("match cancelled before input adaptation")
        query_chunks = build_query_chunks(
            req, chunker=self._chunker,
            load_paper_chunks=self._load_paper_chunks,
            load_upload_md=self._load_upload_md,
        )
        if progress_callback is not None:
            progress_callback({
                "phase": "prepared",
                "completed": 0,
                "total": len(query_chunks),
            })
        if not query_chunks:
            _logger.info("match_doc.match empty_query input_type=%s", req.input_type)
            return MatchDocResponse(
                input_type=req.input_type, query_chunks=[], papers=[],
                metrics={"query_chunks": 0, "total_ms": 0.0},
            )

        filters = deepcopy(req.filters or {})
        exclude = {req.paper_id} if (req.input_type == "paper_id" and req.paper_id) else None
        if exclude:
            filters.setdefault("__exclude__", {}).setdefault("paper_id", []).extend(exclude)
        if self._batch_search_fn:
            per_chunk_results, batch_metrics = self._batch_search_fn(
                [qc.text for qc in query_chunks],
                top_k=req.per_chunk_k,
                filters=filters,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )
        else:
            per_chunk_results = []
            for qc in query_chunks:
                if should_cancel is not None and should_cancel():
                    raise QdrantRetrievalCancelledError(
                        "match cancelled during serial retrieval fallback",
                    )
                per_chunk_results.append(
                    self._search_fn(
                        qc.text,
                        top_k=req.per_chunk_k,
                        filters=filters,
                    ) or [],
                )
            batch_metrics = {}
        if len(per_chunk_results) != len(query_chunks):
            raise RuntimeError("search result group count did not match query chunks")
        t_retrieve = time.perf_counter()
        if should_cancel is not None and should_cancel():
            raise QdrantRetrievalCancelledError("match cancelled before aggregation")
        if progress_callback is not None:
            progress_callback({
                "phase": "aggregate",
                "completed": len(query_chunks),
                "total": len(query_chunks),
            })

        papers = aggregate_matches(
            per_chunk_results, agg=req.agg, sim_threshold=req.sim_threshold,
            max_papers=req.top_k, exclude_paper_ids=exclude,
        )
        self._enrich_titles(papers)
        t_end = time.perf_counter()

        metrics: dict[str, Any] = {
            "query_chunks": len(query_chunks),
            "candidates": sum(len(r) for r in per_chunk_results),
            "hybrid_candidates": batch_metrics.pop("hybrid_candidates", sum(len(r) for r in per_chunk_results)),
            "papers": len(papers),
            "retrieve_ms": round((t_retrieve - t0) * 1000, 1),
            "aggregate_ms": round((t_end - t_retrieve) * 1000, 1),
            "total_ms": round((t_end - t0) * 1000, 1),
            "agg": req.agg,
            **batch_metrics,
        }
        _logger.info(
            "match_doc.match.done input_type=%s query_chunks=%d candidates=%d papers=%d total_ms=%.0f",
            req.input_type, len(query_chunks), metrics["candidates"], len(papers),
            metrics["total_ms"],
        )
        return MatchDocResponse(
            input_type=req.input_type, query_chunks=query_chunks,
            papers=papers, metrics=metrics,
        )

    def _enrich_titles(self, papers: list) -> None:
        """SQL 回填 title（及缺失的 venue/year）—— title 不在 Qdrant 白名单。原地改。"""
        if not self._load_paper_meta or not papers:
            return
        ids = [p.paper_id for p in papers if p.paper_id]
        if not ids:
            return
        try:
            meta = self._load_paper_meta(ids) or {}
        except Exception as exc:  # 回填失败不该让整个匹配挂掉
            _logger.warning("match_doc.enrich_titles.fail err=%s", exc)
            return
        for p in papers:
            m = meta.get(p.paper_id)
            if not m:
                continue
            if not p.title and m.get("title"):
                p.title = m["title"]
            if not p.venue and m.get("venue"):
                p.venue = m["venue"]
            if p.year is None and m.get("year") is not None:
                p.year = m["year"]


# ---- 生产工厂（lazy，按需绑定进程内 SearchPipeline + DB）----------------

def get_match_doc_service(search_pipeline=None) -> MatchDocService:
    """绑定真实依赖：进程内 SearchPipeline、语料同款 chunker、SQL 本体块 loader、
    上传 extract MD loader。在请求处理时调用（此时单例已暖）。

    `search_pipeline` 由调用方（路由）显式传入 —— 路由持有正确的进程内单例。
    **不要 `import main` 去取**：backend 以 `python main.py` 启动时运行模块是
    `__main__`，`import main` 会加载第二份全新副本（其 search_pipeline=None，
    lifespan 没跑过）→ 误判未就绪。兜底才从运行模块 (`__main__`/`main`) 解析。
    """
    from core.indexing.chunker import create_chunker
    from core.retrieval.pipeline import SearchRequest as PipelineSearchRequest

    pipeline = search_pipeline
    if pipeline is None:
        import sys
        running = sys.modules.get("__main__") or sys.modules.get("main")
        pipeline = getattr(running, "search_pipeline", None)
    if pipeline is None:
        raise RuntimeError("search_pipeline 未初始化（backend 未就绪）")

    def search_fn(query: str, *, top_k: int, filters: dict) -> list[SearchResult]:
        resp = pipeline.search(PipelineSearchRequest(
            query=query, top_k=top_k, mode="hybrid",
            filters=filters or {}, dedup_by_paper=False,
        ))
        return resp.results

    def batch_search_fn(
        queries: list[str],
        *,
        top_k: int,
        filters: dict,
        progress_callback=None,
        should_cancel=None,
    ):
        request_batch = [PipelineSearchRequest(
            query=q, top_k=top_k, mode="hybrid", filters=filters or {}, dedup_by_paper=False)
            for q in queries]
        if not hasattr(pipeline, "search_many"):
            return [pipeline.search(item).results for item in request_batch], {}
        responses = pipeline.search_many(
            request_batch,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        if len(responses) != len(request_batch):
            raise RuntimeError("search batch response count did not match query chunks")
        metrics = [getattr(r, "metrics", None) for r in responses]
        return [r.results for r in responses], {
            "embedding_batches": metrics[-1].embedding_batches if metrics and metrics[-1] else 0,
            "qdrant_batches": metrics[-1].qdrant_batches if metrics and metrics[-1] else 0,
            "qdrant_fallback_splits": metrics[-1].qdrant_fallback_splits if metrics and metrics[-1] else 0,
            "rerank_model_calls": metrics[-1].rerank_model_calls if metrics and metrics[-1] else 0,
            "rerank_batches": metrics[-1].rerank_batches if metrics and metrics[-1] else 0,
            "rerank_pairs": metrics[-1].rerank_pairs if metrics and metrics[-1] else 0,
            "hybrid_candidates": sum(m.candidates_before_rerank for m in metrics if m),
            "embed_ms": metrics[-1].embed_ms if metrics and metrics[-1] else 0.0,
            "qdrant_ms": metrics[-1].retrieve_ms if metrics and metrics[-1] else 0.0,
            "rerank_ms": metrics[-1].rerank_ms if metrics and metrics[-1] else 0.0,
        }

    return MatchDocService(
        # 查询文档切块用 SectionAwareChunker（按标题/大小切，**无 40 字符下限**）。
        # 不用 section_l1 —— 它是入库专用（SemanticSectionChunker），丢弃 <40 字符的段
        # → 短查询文本（如 "retrieval augmented generation" 30 字符）被切成 0 段（W103）。
        search_fn=search_fn,
        batch_search_fn=batch_search_fn,
        chunker=create_chunker(),
        load_paper_chunks=_load_paper_chunks_sql,
        load_upload_md=_load_upload_md_sql,
        load_paper_meta=_load_paper_meta_sql,
    )


def _load_paper_meta_sql(paper_ids: list) -> dict:
    """SQL：批量 paper_id → {title, venue, year}（结果卡片标题回填）。"""
    from core.db import get_db_session
    from core.db.models import Paper

    ids = [str(p) for p in paper_ids if p]
    if not ids:
        return {}
    with get_db_session() as session:
        rows = (
            session.query(Paper.paper_id, Paper.title, Paper.venue, Paper.year)
            .filter(Paper.paper_id.in_(ids))
            .all()
        )
        return {pid: {"title": title, "venue": venue, "year": year}
                for (pid, title, venue, year) in rows}


def _load_paper_chunks_sql(paper_id: str) -> list[tuple[str, Optional[int]]]:
    """SQL：paper_id → doc_id → 本体块（level=2 优先，回退全部），按 chunk_index 有序。"""
    from core.db import get_db_session
    from core.db.models import Chunk, Paper

    with get_db_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        if paper is None or paper.doc_id is None:
            return []
        rows = (
            session.query(Chunk.content, Chunk.level)
            .filter(Chunk.doc_id == paper.doc_id, Chunk.level == 2)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        if not rows:  # 回退：无 level=2 则取全部块
            rows = (
                session.query(Chunk.content, Chunk.level)
                .filter(Chunk.doc_id == paper.doc_id)
                .order_by(Chunk.chunk_index.asc())
                .all()
            )
        return [(content, level) for (content, level) in rows]


def _load_upload_md_sql(task_id: str) -> str:
    """上传 PDF extract 出的 MD 全文（由 upload_task_id 定位 extract 产物）。"""
    from app.services.match_doc.upload_md import load_upload_md_for_task

    return load_upload_md_for_task(task_id)
