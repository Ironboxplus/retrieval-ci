"""TDD: paper-grouped search results + paper_ids filter (W87).

The retrieval UX is paper-granular: results are grouped by paper, papers are
ranked by their single best (max) chunk score, and each paper card carries the
chunks that matched. The `dedup_by_paper` toggle only governs how many chunks a
card shows (1 = keep best, all = show every matched chunk) — it never changes
the paper-level grouping or ordering.

This locks two pure, DB-free helpers (so they're fast + deterministic):
  - search_grouping.group_into_papers(results, keep_best_only, max_papers)
  - search_grouping.build_qdrant_filters(request)  ← now also honors paper_ids

Plus the schema surface: SearchRequest.paper_ids, SearchResponse.papers,
and the PaperGroup model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sr(chunk_id: str, paper_id: str | None, doc_id: str, score: float, content: str = "body"):
    """Build a SearchResult the way the pipeline emits it (paper_id in metadata)."""
    from app.models.schemas import SearchResult

    meta = {"venue": "NeurIPS", "year": 2024}
    if paper_id is not None:
        meta["paper_id"] = paper_id
    return SearchResult(
        doc_id=doc_id, chunk_id=chunk_id, content=content, score=score,
        doc_title=f"Title-{paper_id or doc_id}", source="src", metadata=meta,
    )


# Two papers; paper A contributes two chunks, ranked score-desc the way the
# pipeline hands them over after rerank: A(0.9) > B(0.8) > A(0.7).
RESULTS = [
    _sr("cA1", "A", "1", 0.9),
    _sr("cB1", "B", "2", 0.8),
    _sr("cA2", "A", "1", 0.7),
]


class TestGroupIntoPapers(unittest.TestCase):
    def test_groups_chunks_by_paper(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=False)
        self.assertEqual([p.paper_id for p in papers], ["A", "B"])
        self.assertEqual(len(papers), 2)

    def test_paper_carries_its_matched_chunks_score_desc(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=False)
        a = papers[0]
        self.assertEqual([c.chunk_id for c in a.chunks], ["cA1", "cA2"])
        self.assertEqual([c.score for c in a.chunks], [0.9, 0.7])
        self.assertEqual(a.chunk_count, 2)

    def test_papers_ranked_by_max_chunk_score(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=False)
        self.assertEqual(papers[0].paper_id, "A")  # 0.9
        self.assertEqual(papers[0].max_score, 0.9)
        self.assertEqual(papers[1].paper_id, "B")  # 0.8
        self.assertEqual(papers[1].max_score, 0.8)

    def test_keep_best_only_collapses_to_single_chunk(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=True)
        a = papers[0]
        self.assertEqual([c.chunk_id for c in a.chunks], ["cA1"])  # only the best
        self.assertEqual(a.chunk_count, 1)
        self.assertEqual(a.max_score, 0.9)  # max still reflects the group

    def test_max_papers_caps_paper_count(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=False, max_papers=1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "A")

    def test_falls_back_to_doc_id_when_no_paper_id(self):
        from app.services.search_grouping import group_into_papers

        results = [_sr("w1", None, "55", 0.6)]
        papers = group_into_papers(results, keep_best_only=False)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].doc_id, "55")
        # No business paper_id → empty string, never crashes
        self.assertEqual(papers[0].paper_id, "")

    def test_paper_metadata_taken_from_best_chunk(self):
        from app.services.search_grouping import group_into_papers

        papers = group_into_papers(RESULTS, keep_best_only=False)
        self.assertEqual(papers[0].metadata.get("venue"), "NeurIPS")
        self.assertEqual(papers[0].doc_title, "Title-A")


class TestBuildQdrantFilters(unittest.TestCase):
    def test_paper_ids_become_paper_id_filter(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        f = build_qdrant_filters(SearchRequest(query="x", paper_ids=["a", "b"]))
        self.assertEqual(f.get("paper_id"), ["a", "b"])

    def test_venue_year_tier_datasource_preserved(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        f = build_qdrant_filters(SearchRequest(
            query="x", venue="NeurIPS", year=2024,
            venue_tier="CCF-A", data_source="system",
        ))
        self.assertEqual(f["venue"], "NeurIPS")
        self.assertEqual(f["year"], 2024)
        self.assertEqual(f["venue_tier"], "CCF-A")
        self.assertEqual(f["data_source"], "system")

    def test_year_list_passes_through(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        f = build_qdrant_filters(SearchRequest(query="x", year=[2023, 2024]))
        self.assertEqual(f["year"], [2023, 2024])

    def test_empty_request_has_no_filters(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        self.assertEqual(build_qdrant_filters(SearchRequest(query="x")), {})

    def test_empty_year_list_skipped(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        f = build_qdrant_filters(SearchRequest(query="x", year=[]))
        self.assertNotIn("year", f)

    def test_empty_paper_ids_skipped(self):
        from app.models.schemas import SearchRequest
        from app.services.search_grouping import build_qdrant_filters

        f = build_qdrant_filters(SearchRequest(query="x", paper_ids=[]))
        self.assertNotIn("paper_id", f)


class TestAttachChunkPositions(unittest.TestCase):
    """In-paper search needs document order, but the Qdrant payload omits
    positional fields (8-field whitelist). attach_chunk_positions maps each
    result to its SQL chunk_index by exact content match (content IS in the
    whitelist, identical to SQL), so the UI can offer a 按正文顺序 view.
    """

    def test_attaches_chunk_index_by_content(self):
        from app.services.search_grouping import attach_chunk_positions

        body = _sr("b", "A", "1", 0.5, content="Body paragraph")
        abstract = _sr("a", "A", "1", 0.4, content="Abstract text")
        out = attach_chunk_positions([body, abstract], {"Abstract text": 0, "Body paragraph": 7})
        self.assertEqual(out[0].metadata["chunk_index"], 7)   # body
        self.assertEqual(out[1].metadata["chunk_index"], 0)   # abstract

    def test_unmatched_content_left_without_index(self):
        from app.services.search_grouping import attach_chunk_positions

        r = _sr("x", "A", "1", 0.5, content="not in map")
        out = attach_chunk_positions([r], {"something else": 3})
        self.assertNotIn("chunk_index", out[0].metadata)


class TestSchemaSurface(unittest.TestCase):
    def test_search_request_paper_ids_defaults_none(self):
        from app.models.schemas import SearchRequest

        self.assertIsNone(SearchRequest(query="x").paper_ids)

    def test_search_request_paper_ids_settable(self):
        from app.models.schemas import SearchRequest

        self.assertEqual(SearchRequest(query="x", paper_ids=["p1"]).paper_ids, ["p1"])

    def test_search_response_papers_defaults_empty(self):
        from app.models.schemas import SearchResponse

        resp = SearchResponse(query="x", method="hybrid", total_results=0)
        self.assertEqual(resp.papers, [])

    def test_paper_group_model_shape(self):
        from app.models.schemas import PaperGroup, SearchResult

        pg = PaperGroup(
            paper_id="A", doc_id="1", doc_title="T", source="s",
            max_score=0.9, chunk_count=1,
            metadata={"venue": "NeurIPS"},
            chunks=[SearchResult(doc_id="1", chunk_id="c", content="x",
                                 score=0.9, doc_title="T", source="s")],
        )
        self.assertEqual(pg.paper_id, "A")
        self.assertEqual(pg.max_score, 0.9)
        self.assertEqual(len(pg.chunks), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
