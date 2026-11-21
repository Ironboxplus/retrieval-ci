"""功能2 adapters + service 单测 —— 注入 chunker/loaders/search_fn，DB/Qdrant-free 离线秒级。

adapters：三种输入（text / pdf-upload / paper_id）→ 统一 QueryChunk 列表。
service ：adapt → 逐块检索（注入 search_fn）→ 聚合 → MatchDocResponse；paper_id 输入自动排除自身。
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.models.schemas import SearchResult
from app.services.match_doc.adapters import build_query_chunks, chunks_from_text
from app.services.match_doc.schemas import MatchDocRequest
from app.services.match_doc.service import MatchDocService


# ---- fakes -------------------------------------------------------------

@dataclass
class _FakeChunk:
    content: str
    level: int = 2


class _LineChunker:
    """按非空行切块的假 chunker（对齐 Chunker.chunk_document 签名）。"""

    def chunk_document(self, doc_id, text, content_type=None, **_):
        return [_FakeChunk(content=ln.strip(), level=2)
                for ln in (text or "").splitlines() if ln.strip()]


def _sr(paper_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(
        doc_id=f"d-{paper_id}", chunk_id=f"c-{paper_id}-{content[:6]}",
        content=content, score=score, doc_title=f"title-{paper_id}",
        source="", metadata={"paper_id": paper_id},
    )


# ---- adapters ----------------------------------------------------------

def test_chunks_from_text_splits_and_indexes():
    qcs = chunks_from_text("line one\nline two\n\nline three",
                           chunker=_LineChunker(), max_chunks=40)
    assert [q.idx for q in qcs] == [0, 1, 2]
    assert [q.text for q in qcs] == ["line one", "line two", "line three"]


def test_chunks_from_text_truncates_to_max():
    text = "\n".join(f"l{i}" for i in range(100))
    qcs = chunks_from_text(text, chunker=_LineChunker(), max_chunks=5)
    assert len(qcs) == 5
    assert [q.idx for q in qcs] == [0, 1, 2, 3, 4]


def test_chunks_from_text_empty_returns_empty():
    assert chunks_from_text("", chunker=_LineChunker(), max_chunks=40) == []
    assert chunks_from_text("   \n  ", chunker=_LineChunker(), max_chunks=40) == []


def test_build_query_chunks_text_input():
    req = MatchDocRequest(input_type="text", text="alpha\nbeta", max_query_chunks=40)
    qcs = build_query_chunks(req, chunker=_LineChunker())
    assert [q.text for q in qcs] == ["alpha", "beta"]


def test_build_query_chunks_paper_input_uses_loader():
    req = MatchDocRequest(input_type="paper_id", paper_id="aaai-2022-1", max_query_chunks=40)
    loaded = {"aaai-2022-1": [("body chunk a", 2), ("body chunk b", 2)]}
    qcs = build_query_chunks(req, chunker=_LineChunker(),
                             load_paper_chunks=lambda pid: loaded[pid])
    assert [q.text for q in qcs] == ["body chunk a", "body chunk b"]
    assert [q.idx for q in qcs] == [0, 1]


def test_build_query_chunks_pdf_input_chunks_md():
    req = MatchDocRequest(input_type="pdf", upload_task_id="t-9", max_query_chunks=40)
    qcs = build_query_chunks(req, chunker=_LineChunker(),
                             load_upload_md=lambda tid: "p one\np two")
    assert [q.text for q in qcs] == ["p one", "p two"]


def test_build_query_chunks_missing_field_raises():
    with pytest.raises(ValueError):
        build_query_chunks(MatchDocRequest(input_type="text"), chunker=_LineChunker())
    with pytest.raises(ValueError):
        build_query_chunks(MatchDocRequest(input_type="paper_id"), chunker=_LineChunker())


# ---- service -----------------------------------------------------------

def _service(search_map, **kw):
    """search_fn 按查询块文本返回预设候选；记录调用以验证检索参数。"""
    calls = []

    def search_fn(query, *, top_k, filters):
        calls.append({"query": query, "top_k": top_k, "filters": filters})
        return search_map.get(query, [])

    svc = MatchDocService(search_fn=search_fn, chunker=_LineChunker(), **kw)
    return svc, calls


def test_service_matches_text_and_aggregates_coverage():
    search_map = {
        "q0": [_sr("A", "a0", 0.8), _sr("B", "b0", 0.95)],
        "q1": [_sr("A", "a1", 0.6)],
    }
    svc, calls = _service(search_map)
    req = MatchDocRequest(input_type="text", text="q0\nq1", agg="coverage", per_chunk_k=50)
    resp = svc.match(req)
    assert resp.input_type == "text"
    assert [q.text for q in resp.query_chunks] == ["q0", "q1"]
    assert [p.paper_id for p in resp.papers] == ["A", "B"]  # A 覆盖全文胜
    assert resp.papers[0].coverage == 2
    # per_chunk_k 透传为 top_k
    assert all(c["top_k"] == 50 for c in calls)
    assert resp.metrics["query_chunks"] == 2


def test_service_paper_input_excludes_self():
    search_map = {"selfbody": [_sr("SELF", "x", 0.99), _sr("OTHER", "y", 0.7)]}
    svc, _ = _service(
        search_map,
        load_paper_chunks=lambda pid: [("selfbody", 2)],
    )
    req = MatchDocRequest(input_type="paper_id", paper_id="SELF", agg="maxsim")
    resp = svc.match(req)
    assert [p.paper_id for p in resp.papers] == ["OTHER"]  # 自身被排除


def test_service_top_k_caps_returned_papers():
    search_map = {
        "q0": [_sr("A", "a0", 0.5), _sr("B", "b0", 0.5), _sr("C", "c0", 0.5)],
        "q1": [_sr("A", "a1", 0.5)],
    }
    svc, _ = _service(search_map)
    req = MatchDocRequest(input_type="text", text="q0\nq1", agg="coverage", top_k=1)
    resp = svc.match(req)
    assert [p.paper_id for p in resp.papers] == ["A"]  # cov2 胜，cap=1


def test_service_whitespace_text_raises():
    """纯空白 = 没给内容 → 校验错（非 graceful empty）。"""
    svc, _ = _service({})
    with pytest.raises(ValueError):
        svc.match(MatchDocRequest(input_type="text", text="   "))


def test_service_unusable_text_returns_empty_papers():
    """有文本但 chunker 切不出可用块（全噪声）→ graceful 空结果，不报错。"""
    class _EmptyChunker:
        def chunk_document(self, doc_id, text, content_type=None, **_):
            return []

    def search_fn(q, *, top_k, filters):
        return []

    svc = MatchDocService(search_fn=search_fn, chunker=_EmptyChunker())
    resp = svc.match(MatchDocRequest(input_type="text", text="......"))
    assert resp.papers == []
    assert resp.query_chunks == []
    assert resp.metrics["query_chunks"] == 0


def test_factory_uses_passed_pipeline_not_reimport():
    """get_match_doc_service(pipeline) 必须用传入的 pipeline —— 不能 import main 取
    （backend 跑 python main.py 时运行模块是 __main__，import main 取到 None 副本）。"""
    from app.services.match_doc.service import get_match_doc_service

    class _FakePipelineResp:
        def __init__(self, results):
            self.results = results

    class _FakePipeline:
        def __init__(self):
            self.calls = []

        def search(self, req):
            self.calls.append(req)
            return _FakePipelineResp([_sr("A", "hit", 0.9)])

    pipe = _FakePipeline()
    svc = get_match_doc_service(pipe)
    # 真实 section_l1 chunker 丢弃 <40 字符的块 → 用足够长的查询文本。
    long_text = (
        "We propose a retrieval augmented generation method that improves the "
        "factual accuracy of large language models by retrieving relevant "
        "passages from an external corpus and conditioning generation on them."
    )
    resp = svc.match(MatchDocRequest(input_type="text", text=long_text, per_chunk_k=7))
    assert len(pipe.calls) >= 1  # 走了传入的 pipeline
    assert pipe.calls[0].top_k == 7 and pipe.calls[0].dedup_by_paper is False
    assert [p.paper_id for p in resp.papers] == ["A"]


def test_factory_chunker_keeps_short_query_text():
    """W103：生产工厂的 chunker 不能丢短文本。"retrieval augmented generation"(30字)
    用 section_l1 会被切成 0 段 → 空结果；必须用 SectionAwareChunker(无 40 字符下限)。"""
    from app.services.match_doc.service import get_match_doc_service

    class _FakePipeline:
        def search(self, req):
            class R:
                results = [_sr("A", "hit", 0.9)]
            return R()

    svc = get_match_doc_service(_FakePipeline())
    for short in ["retrieval augmented generation", "RAG for LLMs", "graph neural network"]:
        resp = svc.match(MatchDocRequest(input_type="text", text=short))
        assert len(resp.query_chunks) >= 1, f"短文本被切成 0 段: {short!r}"
        assert resp.papers, f"短文本无结果: {short!r}"


def test_service_enriches_empty_titles_from_meta_loader():
    """title 不在 Qdrant 白名单 → chunk 命中无标题；load_paper_meta 回填。"""
    search_map = {"q0": [_sr("A", "a0", 0.9)]}

    def search_fn(query, *, top_k, filters):
        return search_map.get(query, [])

    svc = MatchDocService(
        search_fn=search_fn, chunker=_LineChunker(),
        load_paper_meta=lambda ids: {"A": {"title": "Real Title A", "venue": "KDD", "year": 2024}},
    )
    # _sr 给的 title 是 "title-A"（doc_title），但 venue/year 为空 → 回填补 venue/year
    resp = svc.match(MatchDocRequest(input_type="text", text="q0"))
    p = resp.papers[0]
    assert p.venue == "KDD" and p.year == 2024  # 空字段被回填


def test_service_filters_threshold_passed_through():
    search_map = {"q0": [_sr("A", "a0", 0.2), _sr("B", "b0", 0.9)]}
    svc, calls = _service(search_map)
    req = MatchDocRequest(input_type="text", text="q0", sim_threshold=0.5,
                          filters={"venue": "KDD"})
    resp = svc.match(req)
    assert [p.paper_id for p in resp.papers] == ["B"]  # A 0.2 < 0.5 过滤
    assert calls[0]["filters"] == {"venue": "KDD"}
