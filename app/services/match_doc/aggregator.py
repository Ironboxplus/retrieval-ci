"""纯聚合核心：多查询块的候选 chunk → 论文级排名（功能2）。

保持 DB / Qdrant / pipeline import 干净（仿 search_grouping.py），单测秒级、确定性。

输入 `per_chunk_results[i]` 是「查询文档第 i 块」的候选 chunk 列表（SearchResult，
score 降序无所谓——内部按 paper 取最佳）。每个 SearchResult 的 `metadata.paper_id`
是论文主键，venue/year 也在 metadata，title 在 doc_title。

打分（每篇论文）：
  matched[q] = 该论文在查询块 q 上的最佳 (sim, 命中chunk文本)   # 同块多命中取最大
  coverage   = 命中的查询块数
  coverage 模式：score = Σ_q matched_sim / 查询块总数   # 覆盖整篇 → 高分
  maxsim 模式 ：score = max_q matched_sim               # 单段最强 → 高分
"""
from __future__ import annotations

from typing import Iterable, Optional

from app.models.schemas import SearchResult
from app.services.match_doc.schemas import MatchedSpan, PaperMatch


def _paper_key(r: SearchResult) -> str:
    """论文分组键：业务 paper_id 优先，回退 doc_id（非论文来源也能分组）。"""
    pid = (r.metadata or {}).get("paper_id") if r.metadata else None
    if pid:
        return str(pid)
    return str(r.doc_id or r.chunk_id or "")


def aggregate_matches(
    per_chunk_results: list[list[SearchResult]],
    *,
    agg: str = "coverage",
    sim_threshold: float = 0.0,
    max_papers: Optional[int] = None,
    exclude_paper_ids: Optional[Iterable[str]] = None,
) -> list[PaperMatch]:
    """折叠逐块候选为论文级排名。详见模块 docstring。

    Args:
        per_chunk_results: 每个查询块的候选 chunk 列表。
        agg: "coverage"（默认，整篇重合）| "maxsim"（单段最强）。
        sim_threshold: 低于此 score 的命中忽略。
        max_papers: 截断返回论文数。
        exclude_paper_ids: 排除这些 paper_id（如查询论文自身）。

    Returns:
        PaperMatch 列表，score 降序、coverage 降序 tie-break。
    """
    num_chunks = len(per_chunk_results)
    if num_chunks == 0:
        return []
    excluded = {str(p) for p in (exclude_paper_ids or ())}

    # key -> {best per query chunk idx: (sim, paper_chunk), + 论文级元数据 seed}
    papers: dict[str, dict] = {}
    for q_idx, results in enumerate(per_chunk_results):
        for r in results or []:
            if r.score < sim_threshold:
                continue
            key = _paper_key(r)
            if not key or key in excluded:
                continue
            g = papers.get(key)
            if g is None:
                meta = dict(r.metadata or {})
                g = {
                    "paper_id": str(meta.get("paper_id") or ""),
                    "doc_id": str(r.doc_id or ""),
                    "title": r.doc_title or meta.get("title", "") or "",
                    "venue": meta.get("venue", "") or "",
                    "year": meta.get("year"),
                    "metadata": meta,
                    "best": {},  # q_idx -> (sim, paper_chunk)
                }
                papers[key] = g
            prev = g["best"].get(q_idx)
            if prev is None or r.score > prev[0]:
                g["best"][q_idx] = (r.score, r.content)

    out: list[PaperMatch] = []
    for g in papers.values():
        best = g["best"]
        if not best:
            continue
        sims = [s for (s, _c) in best.values()]
        coverage = len(best)
        if agg == "maxsim":
            score = max(sims)
        else:  # coverage
            score = sum(sims) / num_chunks
        matched = [
            MatchedSpan(query_chunk_idx=q, paper_chunk=c, sim=s)
            for q, (s, c) in sorted(best.items())
        ]
        out.append(
            PaperMatch(
                paper_id=g["paper_id"],
                doc_id=g["doc_id"],
                title=g["title"],
                venue=g["venue"],
                year=g["year"],
                score=score,
                coverage=coverage,
                metadata=g["metadata"],
                matched=matched,
            )
        )

    out.sort(key=lambda p: (p.score, p.coverage), reverse=True)
    if max_papers is not None:
        out = out[:max_papers]
    return out
