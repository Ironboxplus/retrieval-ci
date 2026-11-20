"""TDD: SearchRequest.year accepts list of years for multi-year filtering.

Frontend redesign of the search filter panel requires users to pick multiple
years at once. The Qdrant payload filter already supports list values via
MatchAny (see core/storage/qdrant_store._build_filter list branch), so the only
plumbing change needed is widening the SearchRequest schema. Single-int values
must keep working for backwards compatibility (existing API callers).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.schemas import SearchRequest


class TestSearchRequestYearList(unittest.TestCase):
    def test_year_accepts_single_int_back_compat(self):
        req = SearchRequest(query="x", year=2024)
        self.assertEqual(req.year, 2024)

    def test_year_accepts_list_of_ints(self):
        req = SearchRequest(query="x", year=[2022, 2023, 2024])
        self.assertEqual(req.year, [2022, 2023, 2024])

    def test_year_accepts_empty_list(self):
        # Empty list is a valid value; the route layer decides whether to push
        # it into qdrant_filters (it should skip empty lists).
        req = SearchRequest(query="x", year=[])
        self.assertEqual(req.year, [])

    def test_year_none_default(self):
        req = SearchRequest(query="x")
        self.assertIsNone(req.year)


class TestQdrantFilterYearList(unittest.TestCase):
    """Filter layer already routes list → MatchAny; lock that for year too."""

    def setUp(self):
        from core.storage.qdrant_store import QdrantStore
        self.store = QdrantStore.__new__(QdrantStore)
        self.store.collection_name = "papers"

    def test_year_list_produces_match_any(self):
        from qdrant_client.http.models import MatchAny
        f = self.store._build_filter(filter_dict={"year": [2022, 2023, 2024]})
        self.assertIsNotNone(f)
        self.assertEqual(len(f.must), 1)
        cond = f.must[0]
        self.assertEqual(cond.key, "year")
        self.assertIsInstance(cond.match, MatchAny)
        self.assertEqual(list(cond.match.any), [2022, 2023, 2024])

    def test_year_single_int_still_match_value(self):
        from qdrant_client.http.models import MatchValue
        f = self.store._build_filter(filter_dict={"year": 2024})
        self.assertEqual(len(f.must), 1)
        cond = f.must[0]
        self.assertEqual(cond.key, "year")
        self.assertIsInstance(cond.match, MatchValue)
        self.assertEqual(cond.match.value, 2024)


class TestRouteSkipsEmptyYearList(unittest.TestCase):
    """An empty year list must NOT be pushed as a payload filter — MatchAny([])
    matches nothing and would silently zero out results. The /api/search and
    /api/search/stream routes both build qdrant_filters via the shared
    build_qdrant_filters helper (W87); this pins the skip-empty-list contract on
    its year normalizer.
    """

    def test_helper_skips_empty_year_list(self):
        from app.services.search_grouping import normalize_year_filter
        self.assertIsNone(normalize_year_filter(None))
        self.assertIsNone(normalize_year_filter([]))
        self.assertEqual(normalize_year_filter(2024), 2024)
        self.assertEqual(normalize_year_filter([2022, 2023]), [2022, 2023])


if __name__ == "__main__":
    unittest.main()
