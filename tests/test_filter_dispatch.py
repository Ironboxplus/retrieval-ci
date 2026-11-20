"""TDD: filter dispatch contract for SearchPipeline + QdrantStore.

Locks the unified filter_dict contract after E1/E2 simplification:
- SearchPipeline merges source_type/level/free-form filters into a single
  payload-equality dict passed as `filter_dict=` to the store.
- QdrantStore._build_filter treats source_type/level/data_source as sugar
  over filter_dict and produces a Filter with one FieldCondition per key.

Also locks the C1 contract: title_only=True in /api/search routes to
pipe_req.level=1 (no application-layer post-filter on titles).
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestQdrantBuildFilterMerged(unittest.TestCase):
    """_build_filter must funnel level/source_type/data_source through filter_dict."""

    def setUp(self):
        from core.storage.qdrant_store import QdrantStore
        # Construct without touching the client by patching __init__
        self.store = QdrantStore.__new__(QdrantStore)
        self.store.collection_name = "papers"

    def test_no_kwargs_returns_none(self):
        self.assertIsNone(self.store._build_filter())

    def test_empty_filter_dict_returns_none(self):
        self.assertIsNone(self.store._build_filter(filter_dict={}))

    def test_single_kwarg_one_condition(self):
        f = self.store._build_filter(source_type="paper")
        self.assertIsNotNone(f)
        self.assertEqual(len(f.must), 1)
        cond = f.must[0]
        self.assertEqual(cond.key, "source_type")
        self.assertEqual(cond.match.value, "paper")

    def test_multiple_named_kwargs_become_conditions(self):
        f = self.store._build_filter(level=1, source_type="paper",
                                     data_source="system")
        self.assertEqual(len(f.must), 3)
        keys = sorted(c.key for c in f.must)
        self.assertEqual(keys, ["data_source", "level", "source_type"])

    def test_filter_dict_and_kwargs_merge(self):
        f = self.store._build_filter(filter_dict={"venue": "ICML"},
                                     level=1, source_type="paper")
        keys = sorted(c.key for c in f.must)
        self.assertEqual(keys, ["level", "source_type", "venue"])
        venue_cond = next(c for c in f.must if c.key == "venue")
        self.assertEqual(venue_cond.match.value, "ICML")

    def test_exclude_builds_must_not(self):
        # __exclude__ reserved key -> Filter.must_not (positive must left empty).
        f = self.store._build_filter(filter_dict={"__exclude__": {"source_type": ["review"]}})
        self.assertIsNotNone(f)
        self.assertIsNone(f.must)
        self.assertEqual(len(f.must_not), 1)
        cond = f.must_not[0]
        self.assertEqual(cond.key, "source_type")
        self.assertEqual(list(cond.match.any), ["review"])

    def test_exclude_coexists_with_must(self):
        f = self.store._build_filter(
            filter_dict={"venue": "ICLR", "__exclude__": {"source_type": ["review"]}})
        self.assertEqual([c.key for c in f.must], ["venue"])
        self.assertEqual([c.key for c in f.must_not], ["source_type"])

    def test_kwarg_overrides_filter_dict_same_key(self):
        # If caller passes both filter_dict={"source_type": "x"} and source_type="y",
        # the named kwarg takes precedence (last-write-wins via dict merge).
        f = self.store._build_filter(filter_dict={"source_type": "old"},
                                     source_type="new")
        st_conds = [c for c in f.must if c.key == "source_type"]
        self.assertEqual(len(st_conds), 1)
        self.assertEqual(st_conds[0].match.value, "new")


class TestPipelineFilterDispatch(unittest.TestCase):
    """SearchPipeline must merge all filter sources into one dict via filter_dict=."""

    def _make_pipeline(self):
        from core.retrieval.pipeline import SearchPipeline

        # Mock embedder returns dense + sparse for one query
        embedder = MagicMock()
        embed_result = MagicMock()
        embed_result.dense = np.array([[0.1] * 1024], dtype=np.float32)
        embed_result.sparse = [{"42": 0.5}]
        embedder.embed_texts.return_value = embed_result

        store = MagicMock()
        store.hybrid_search.return_value = []
        store.sparse_search.return_value = []
        store.search.return_value = []

        return SearchPipeline(embedder=embedder, store=store, reranker=None), store

    def test_hybrid_passes_filter_dict_only(self):
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        pipe.search(SearchRequest(query="x", mode="hybrid",
                                  source_type="paper", level=1,
                                  filters={"venue": "ICML"}))
        store.hybrid_search.assert_called_once()
        kwargs = store.hybrid_search.call_args.kwargs
        # E1 contract: ONLY filter_dict kwarg (no source_type=/level=)
        self.assertIn("filter_dict", kwargs)
        self.assertNotIn("source_type", kwargs)
        self.assertNotIn("level", kwargs)
        fd = kwargs["filter_dict"]
        self.assertEqual(fd.get("source_type"), "paper")
        self.assertEqual(fd.get("level"), 1)
        self.assertEqual(fd.get("venue"), "ICML")

    def test_sparse_passes_filter_dict_only(self):
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        pipe.search(SearchRequest(query="x", mode="sparse", level=1))
        store.sparse_search.assert_called_once()
        kwargs = store.sparse_search.call_args.kwargs
        self.assertIn("filter_dict", kwargs)
        self.assertEqual(kwargs["filter_dict"].get("level"), 1)

    def test_dense_search_passes_filter_dict_only(self):
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        # Force dense path by providing no sparse via embedder
        pipe._embedder.embed_texts.return_value.sparse = None
        pipe.search(SearchRequest(query="x", mode="hybrid", source_type="webpage"))
        store.search.assert_called_once()
        # dense store.search uses positional + kwargs; check kwargs has filter_dict
        kwargs = store.search.call_args.kwargs
        self.assertIn("filter_dict", kwargs)
        self.assertEqual(kwargs["filter_dict"].get("source_type"), "webpage")

    def test_default_excludes_reviews(self):
        # No source_type + include_reviews=False (default): the only filter is a
        # must_not excluding review chunks (they share the papers collection).
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        pipe.search(SearchRequest(query="x", mode="hybrid"))
        fd = store.hybrid_search.call_args.kwargs["filter_dict"]
        self.assertEqual(fd, {"__exclude__": {"source_type": ["review"], "venue_tier": ["preprint"], "withdrawn": True}})

    def test_include_reviews_drops_the_exclusion(self):
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        pipe.search(SearchRequest(query="x", mode="hybrid", include_reviews=True))
        fd = store.hybrid_search.call_args.kwargs["filter_dict"]
        self.assertEqual(fd, {"__exclude__": {"venue_tier": ["preprint"], "withdrawn": True}})

    def test_source_type_review_is_review_only(self):
        # Explicit source_type=review -> positive match, no exclusion.
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        pipe.search(SearchRequest(query="x", mode="hybrid", source_type="review"))
        fd = store.hybrid_search.call_args.kwargs["filter_dict"]
        self.assertEqual(fd.get("source_type"), "review")
        # No review exclusion; the preprint one stays (source_type does not scope tiers).
        self.assertNotIn("source_type", fd.get("__exclude__", {}))

    def test_stream_path_also_excludes_reviews(self):
        # dual-endpoint sync: search_stream must apply the same default exclusion.
        from core.retrieval.pipeline import SearchRequest
        pipe, store = self._make_pipeline()
        list(pipe.search_stream(SearchRequest(query="x", mode="hybrid")))
        fd = store.hybrid_search.call_args.kwargs["filter_dict"]
        self.assertEqual(fd, {"__exclude__": {"source_type": ["review"], "venue_tier": ["preprint"], "withdrawn": True}})


class TestTitleOnlyRoutesToLevelOne(unittest.TestCase):
    """C1 contract: title_only=True at /api/search → pipe_req.level=1."""

    def test_schemas_title_only_field_default_false(self):
        from app.models.schemas import SearchRequest
        req = SearchRequest(query="x")
        self.assertFalse(req.title_only)

    def test_pipeline_request_has_level_field(self):
        from core.retrieval.pipeline import SearchRequest as PipeReq
        req = PipeReq(query="x", level=1)
        self.assertEqual(req.level, 1)

    def test_pipeline_level_filter_routes_into_filter_dict(self):
        from core.retrieval.pipeline import SearchPipeline, SearchRequest as PipeReq

        embedder = MagicMock()
        embed_result = MagicMock()
        embed_result.dense = np.array([[0.1] * 1024], dtype=np.float32)
        embed_result.sparse = [{"1": 1.0}]
        embedder.embed_texts.return_value = embed_result

        store = MagicMock()
        store.hybrid_search.return_value = []

        pipe = SearchPipeline(embedder=embedder, store=store, reranker=None)
        pipe.search(PipeReq(query="any", mode="hybrid", level=1))

        kwargs = store.hybrid_search.call_args.kwargs
        self.assertEqual(kwargs["filter_dict"].get("level"), 1)


class TestWithdrawnExclusion(unittest.TestCase):
    """include_withdrawn rides the include_preprints path: retrieve_filters adds the
    must_not, QdrantStore._build_filter turns the bool into a MatchValue."""

    def _store(self):
        from core.storage.qdrant_store import QdrantStore
        store = QdrantStore.__new__(QdrantStore)
        store.collection_name = "arxiv"
        return store

    def test_scalar_exclude_is_match_value_and_bool_stays_bool(self):
        f = self._store()._build_filter(filter_dict={"__exclude__": {"withdrawn": True}})
        self.assertEqual(len(f.must_not), 1)
        cond = f.must_not[0]
        self.assertEqual(cond.key, "withdrawn")
        self.assertIs(cond.match.value, True)
        self.assertFalse(hasattr(cond.match, "any"))

    def test_list_exclude_stays_match_any(self):
        f = self._store()._build_filter(
            filter_dict={"__exclude__": {"venue_tier": ["preprint"], "withdrawn": True}})
        by_key = {c.key: c.match for c in f.must_not}
        self.assertEqual(list(by_key["venue_tier"].any), ["preprint"])
        self.assertIs(by_key["withdrawn"].value, True)

    def test_match_any_cannot_carry_a_bool(self):
        # Why a bool must not be wrapped in a list: qdrant_client rejects it.
        from qdrant_client.http import models as qm
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            qm.MatchAny(any=[True])

    def test_default_excludes_withdrawn(self):
        from core.retrieval.pipeline import SearchRequest, retrieve_filters
        self.assertIs(retrieve_filters(SearchRequest(query="x"))["__exclude__"]["withdrawn"], True)

    def test_include_withdrawn_drops_the_exclusion(self):
        from core.retrieval.pipeline import SearchRequest, retrieve_filters
        item = retrieve_filters(SearchRequest(query="x", include_withdrawn=True))
        self.assertNotIn("withdrawn", item["__exclude__"])

    def test_id_scoped_requests_are_not_narrowed(self):
        from core.retrieval.pipeline import SearchRequest, retrieve_filters
        for key in ("paper_id", "doc_id"):
            item = retrieve_filters(SearchRequest(query="x", filters={key: ["a"]}))
            self.assertNotIn("withdrawn", item["__exclude__"], key)

    def test_venue_scope_does_not_lift_it(self):
        # arxiv_scope sets venue_tier=preprint; withdrawn papers must still be dropped.
        from core.retrieval.pipeline import SearchRequest, retrieve_filters
        item = retrieve_filters(SearchRequest(query="x", filters={"venue_tier": "preprint"}))
        self.assertIs(item["__exclude__"]["withdrawn"], True)

    def test_http_request_defaults_and_carries_the_flag(self):
        from app.models.schemas import SearchRequest
        self.assertFalse(SearchRequest(query="x").include_withdrawn)
        self.assertTrue(SearchRequest(query="x", include_withdrawn=True).include_withdrawn)


class TestRetrievalMethodEnumSlim(unittest.TestCase):
    """A4 contract: only VECTOR/HYBRID/SPARSE remain (BM25/ENHANCED gone)."""

    def test_enum_values(self):
        from app.models.schemas import RetrievalMethod
        values = {m.value for m in RetrievalMethod}
        self.assertEqual(values, {"vector", "hybrid", "sparse"})

    def test_legacy_enum_names_removed(self):
        from app.models.schemas import RetrievalMethod
        self.assertFalse(hasattr(RetrievalMethod, "BM25"))
        self.assertFalse(hasattr(RetrievalMethod, "ENHANCED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
