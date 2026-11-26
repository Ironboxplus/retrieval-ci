"""功能2 聚合器单测 —— 纯函数，DB/Qdrant-free，离线秒级。

聚合器把「每个查询块各自的候选 chunk 列表」折叠成论文级排名。
核心契约：
  - coverage 模式：score = Σ(每个命中查询块的最佳相似度) / 查询块总数
      → 覆盖你整篇文档的论文胜出，只命中一段的高相似度论文反而靠后。
  - maxsim 模式：score = 所有命中里的最大相似度 → 单段最强命中胜出。
  - 同一查询块内同一论文多次命中只取最佳。
  - sim_threshold 过滤弱命中；exclude_paper_ids 排除查询论文自身。
"""
from __future__ import annotations

from app.models.schemas import SearchResult
from app.services.match_doc.aggregator import aggregate_matches


def _sr(paper_id: str, content: str, score: float, *, doc_id: str = "",
        title: str = "", venue: str = "", year: int | None = None) -> SearchResult:
    """构造一个检索结果 chunk（paper_id/venue/year 落在 metadata，对齐 pipeline 输出）。"""
    meta: dict = {"paper_id": paper_id}
    if venue:
        meta["venue"] = venue
    if year is not None:
        meta["year"] = year
    return SearchResult(
        doc_id=doc_id or f"d-{paper_id}",
        chunk_id=f"c-{paper_id}-{content[:6]}",
        content=content,
        score=score,
        doc_title=title or f"title-{paper_id}",
        source="",
        metadata=meta,
    )


def test_empty_input_returns_empty():
    assert aggregate_matches([]) == []
    assert aggregate_matches([[], []]) == []


def test_coverage_rewards_covering_whole_doc():
    # 2 个查询块。A 两块都命中(0.8,0.6)；B 只命中第0块但分高(0.95)。
    per_chunk = [
        [_sr("A", "a0", 0.8), _sr("B", "b0", 0.95)],
        [_sr("A", "a1", 0.6)],
    ]
    out = aggregate_matches(per_chunk, agg="coverage")
    assert [p.paper_id for p in out] == ["A", "B"]  # A 覆盖全文 → 胜
    a = out[0]
    assert a.coverage == 2
    assert abs(a.score - (0.8 + 0.6) / 2) < 1e-9  # =0.7
    b = out[1]
    assert b.coverage == 1
    assert abs(b.score - 0.95 / 2) < 1e-9  # =0.475，被未覆盖的那块稀释


def test_maxsim_picks_best_single_passage():
    per_chunk = [
        [_sr("A", "a0", 0.8), _sr("B", "b0", 0.95)],
        [_sr("A", "a1", 0.6)],
    ]
    out = aggregate_matches(per_chunk, agg="maxsim")
    assert [p.paper_id for p in out] == ["B", "A"]  # B 单段最强 → 胜
    assert abs(out[0].score - 0.95) < 1e-9
    assert abs(out[1].score - 0.8) < 1e-9


def test_best_per_query_chunk_dedup():
    # A 在第0块出现两次(0.9,0.5)，只应取 0.9。
    per_chunk = [[_sr("A", "hi", 0.9), _sr("A", "lo", 0.5)]]
    out = aggregate_matches(per_chunk, agg="coverage")
    assert len(out) == 1
    assert out[0].coverage == 1
    assert abs(out[0].score - 0.9) < 1e-9
    assert len(out[0].matched) == 1
    assert out[0].matched[0].sim == 0.9
    assert out[0].matched[0].paper_chunk == "hi"


def test_sim_threshold_filters_weak_matches():
    per_chunk = [[_sr("A", "a0", 0.9)], [_sr("A", "a1", 0.2), _sr("B", "b1", 0.85)]]
    out = aggregate_matches(per_chunk, agg="coverage", sim_threshold=0.5)
    # A 的第二块 0.2 被过滤 → A coverage=1; B coverage=1
    by = {p.paper_id: p for p in out}
    assert by["A"].coverage == 1
    assert abs(by["A"].score - 0.9 / 2) < 1e-9
    assert by["B"].coverage == 1


def test_exclude_paper_ids_drops_query_paper():
    per_chunk = [[_sr("SELF", "x", 0.99), _sr("OTHER", "y", 0.7)]]
    out = aggregate_matches(per_chunk, agg="maxsim", exclude_paper_ids={"SELF"})
    assert [p.paper_id for p in out] == ["OTHER"]


def test_matched_spans_and_metadata_carried():
    per_chunk = [
        [_sr("A", "alpha", 0.8, venue="KDD", year=2024, title="Paper A")],
        [_sr("A", "beta", 0.6, venue="KDD", year=2024, title="Paper A")],
    ]
    out = aggregate_matches(per_chunk, agg="coverage")
    a = out[0]
    assert a.title == "Paper A"
    assert a.venue == "KDD"
    assert a.year == 2024
    assert {m.query_chunk_idx for m in a.matched} == {0, 1}
    assert {m.paper_chunk for m in a.matched} == {"alpha", "beta"}


def test_max_papers_cap_and_tiebreak_by_coverage():
    per_chunk = [
        [_sr("A", "a0", 0.5), _sr("B", "b0", 0.5)],
        [_sr("A", "a1", 0.5)],
    ]
    # A: cov2 score=0.5 ; B: cov1 score=0.25. cap=1 → only A.
    out = aggregate_matches(per_chunk, agg="coverage", max_papers=1)
    assert [p.paper_id for p in out] == ["A"]
