"""TDD: dedup_by_paper toggle on SearchPipeline (chunk mode).

Contract:
- Default (dedup_by_paper=True): collapse multi-chunk hits from one paper to
  its single best-scoring chunk — the W3 anti-flooding behavior, unchanged.
- dedup_by_paper=False ("chunk mode"): return the raw top-k chunks ranked by
  score, allowing multiple chunks from the SAME paper. No aggregation, no
  per-paper collapse — the frontend then labels which paper + which chunk.

Both the sync (search) and streaming (search_stream) paths must honor the flag,
and the HTTP-layer SearchRequest must expose it (default True = current behavior).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Two papers; paper A contributes two chunks. Pre-sorted score-desc the way
# Qdrant would return them: A(0.9) > B(0.8) > A(0.7). With reranker=None the
# pipeline preserves this order, so it doubles as the "ranked by chunk score"
# fixture.
RAW = [
    {"id": "cA1", "content": "a1", "score": 0.9, "metadata": {"paper_id": "A", "doc_id": "1"}},
    {"id": "cB1", "content": "b1", "score": 0.8, "metadata": {"paper_id": "B", "doc_id": "2"}},
    {"id": "cA2", "content": "a2", "score": 0.7, "metadata": {"paper_id": "A", "doc_id": "1"}},
]


def _make_pipeline(raw):
    from core.retrieval.pipeline import SearchPipeline

    embedder = MagicMock()
    embed_result = MagicMock()
    embed_result.dense = np.array([[0.1] * 1024], dtype=np.float32)
    embed_result.sparse = [{"42": 0.5}]
    embedder.embed_texts.return_value = embed_result

    store = MagicMock()
    store.hybrid_search.return_value = raw
    store.sparse_search.return_value = raw
    store.search.return_value = raw

    return SearchPipeline(embedder=embedder, store=store, reranker=None)


class TestDedupByPaperToggle(unittest.TestCase):
    def test_default_dedups_to_one_chunk_per_paper(self):
        """Regression guard: omitting the flag keeps the W3 keep-best behavior."""
        from core.retrieval.pipeline import SearchRequest

        resp = _make_pipeline(RAW).search(SearchRequest(query="x", mode="hybrid"))
        paper_ids = [r.metadata.get("paper_id") for r in resp.results]
        self.assertEqual(paper_ids, ["A", "B"])  # A's 2nd chunk dropped
        self.assertEqual(len(resp.results), 2)

    def test_dedup_off_returns_all_chunks_ranked_by_score(self):
        from core.retrieval.pipeline import SearchRequest

        resp = _make_pipeline(RAW).search(
            SearchRequest(query="x", mode="hybrid", dedup_by_paper=False)
        )
        chunk_ids = [r.chunk_id for r in resp.results]
        self.assertEqual(chunk_ids, ["cA1", "cB1", "cA2"])  # all 3, same-paper kept
        scores = [r.score for r in resp.results]
        self.assertEqual(scores, sorted(scores, reverse=True))  # ranked by chunk score

    def test_dedup_off_keeps_two_chunks_from_same_paper(self):
        from core.retrieval.pipeline import SearchRequest

        resp = _make_pipeline(RAW).search(
            SearchRequest(query="x", mode="hybrid", dedup_by_paper=False)
        )
        a_chunks = [r for r in resp.results if r.metadata.get("paper_id") == "A"]
        self.assertEqual(len(a_chunks), 2)


class TestDedupToggleStream(unittest.TestCase):
    def test_stream_dedup_off_returns_all_chunks(self):
        from core.retrieval.pipeline import SearchRequest

        events = list(
            _make_pipeline(RAW).search_stream(
                SearchRequest(query="x", mode="hybrid", dedup_by_paper=False)
            )
        )
        final = [e for e in events if e.get("stage") == "results"][-1]
        chunk_ids = [r.chunk_id for r in final["results"]]
        self.assertEqual(chunk_ids, ["cA1", "cB1", "cA2"])

    def test_stream_default_still_dedups(self):
        from core.retrieval.pipeline import SearchRequest

        events = list(
            _make_pipeline(RAW).search_stream(SearchRequest(query="x", mode="hybrid"))
        )
        final = [e for e in events if e.get("stage") == "results"][-1]
        self.assertEqual(len(final["results"]), 2)


class TestApiSchemaHasDedupField(unittest.TestCase):
    def test_api_search_request_dedup_by_paper_defaults_true(self):
        from app.models.schemas import SearchRequest as ApiReq

        self.assertTrue(ApiReq(query="x").dedup_by_paper)

    def test_api_search_request_dedup_by_paper_settable_false(self):
        from app.models.schemas import SearchRequest as ApiReq

        self.assertFalse(ApiReq(query="x", dedup_by_paper=False).dedup_by_paper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
