"""
TDD tests for search system enhancement (SE1-SE3).
Tests written BEFORE implementation.
"""
import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSearchPipelineMetrics(unittest.TestCase):
    """SE1/SE3: SearchPipeline must return stage-level metrics."""

    def test_search_response_has_metrics(self):
        from core.retrieval.pipeline import SearchResponse
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SearchResponse)}
        self.assertIn('metrics', fields)

    def test_pipeline_metrics_has_stages(self):
        from core.retrieval.pipeline import PipelineMetrics
        import dataclasses
        fields = {f.name for f in dataclasses.fields(PipelineMetrics)}
        self.assertIn('embed_ms', fields)
        self.assertIn('retrieve_ms', fields)
        self.assertIn('rerank_ms', fields)
        self.assertIn('total_ms', fields)
        self.assertIn('candidates_before_rerank', fields)
        self.assertIn('candidates_after_rerank', fields)
        self.assertIn('mode', fields)

    def test_metrics_serializable(self):
        from core.retrieval.pipeline import PipelineMetrics
        import dataclasses
        m = PipelineMetrics(
            embed_ms=12.5, retrieve_ms=45.3, rerank_ms=80.1,
            total_ms=140.0, candidates_before_rerank=100,
            candidates_after_rerank=20, mode="hybrid",
        )
        d = dataclasses.asdict(m)
        self.assertEqual(d['embed_ms'], 12.5)
        self.assertEqual(d['candidates_before_rerank'], 100)


class TestSearchFiltering(unittest.TestCase):
    """SE2: /api/search must support venue/year/venue_tier filters."""

    def test_search_request_schema_has_filters(self):
        from app.models.schemas import SearchRequest
        fields = SearchRequest.model_fields
        self.assertIn('venue', fields)
        self.assertIn('year', fields)

    def test_main_search_applies_venue_filter(self):
        """venue/year/venue_tier are pushed down to the Qdrant retrieve filter via the
        shared build_qdrant_filters(request) helper (no application-layer post-filter)."""
        with open('main.py', 'r', encoding='utf-8') as f:
            source = f.read()
        self.assertIn('build_qdrant_filters(request)', source)
        from app.services.search_grouping import build_qdrant_filters
        from app.models.schemas import SearchRequest
        filters = build_qdrant_filters(SearchRequest(query="x", venue="NeurIPS"))
        self.assertEqual(filters.get('venue'), 'NeurIPS')


class TestSearchResponseFormat(unittest.TestCase):
    """SE3: API response includes pipeline_metrics."""

    def test_search_response_schema_has_metrics(self):
        from app.models.schemas import SearchResponse
        fields = SearchResponse.model_fields
        self.assertIn('pipeline_metrics', fields)


class TestSparseSearch(unittest.TestCase):
    """Sparse-only search mode."""

    def test_retrieval_method_has_sparse(self):
        from app.models.schemas import RetrievalMethod
        self.assertIn('sparse', [e.value for e in RetrievalMethod])

    def test_pipeline_handles_sparse_mode(self):
        import inspect
        from core.retrieval.pipeline import SearchPipeline
        source = inspect.getsource(SearchPipeline.search)
        self.assertIn('sparse', source)

    def test_qdrant_store_has_sparse_search(self):
        from core.storage.qdrant_store import QdrantStore
        self.assertTrue(hasattr(QdrantStore, 'sparse_search'))


if __name__ == "__main__":
    unittest.main()
