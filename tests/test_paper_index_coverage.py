"""TDD: `/papers/coverage` — per-venue indexed year span and counts.

Backs the Search page's coverage cards. `/api/papers/stats` only returns global
totals, so it cannot answer "which conference is indexed through which year".

Seeds papers under a venue name unique to this run, so the assertions hold no
matter what else is in the live SQLite DB — the same isolation trick
test_papers_browse_filter_keyword.py uses. Real SQLite, no fakes.
"""

from __future__ import annotations

import unittest
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.db import get_db_session
from core.db.models import DataSource, Document, DocumentStatus, Paper, SourceType
from pipeline.paper_crawler.integration import api_routes
from pipeline.paper_crawler.integration.api_routes import create_paper_router


SEED_PREFIX = "test-coverage-"


def _seed_paper(db, paper_id: str, *, venue: str, year: int) -> None:
    doc = Document(
        title=paper_id,
        source=f"https://example.com/{paper_id}",
        source_type=SourceType.paper,
        data_source=DataSource.system,
        status=DocumentStatus.indexed,
    )
    db.add(doc)
    db.flush()
    db.add(
        Paper(
            paper_id=paper_id,
            doc_id=doc.id,
            title=paper_id,
            year=year,
            venue=venue,
            venue_tier="CCF-A",
            track_type="main",
        )
    )


class PaperIndexCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suffix = uuid.uuid4().hex[:8]
        # A venue nobody else uses, so the assertions are independent of corpus size.
        cls.venue = f"TESTVENUE-{cls.suffix}"
        # 2020 deliberately absent: the gap must surface as missing_years rather
        # than being smoothed over by a bare min..max span.
        cls.years = [2019, 2019, 2021, 2022]
        cls.ids = [f"{SEED_PREFIX}{cls.suffix}-{i}" for i in range(len(cls.years))]
        with get_db_session() as db:
            for paper_id, year in zip(cls.ids, cls.years):
                _seed_paper(db, paper_id, venue=cls.venue, year=year)
            db.commit()

        app = FastAPI()
        app.include_router(create_paper_router(prefix="/api/papers"))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        with get_db_session() as db:
            doc_ids = [
                doc_id for (doc_id,) in
                db.query(Paper.doc_id).filter(Paper.paper_id.in_(cls.ids)).all()
            ]
            # Bulk-delete papers first. An ORM db.delete(paper) only flushes at
            # commit, by which point the documents bulk DELETE has already
            # cascaded them away — SQLAlchemy then warns it matched 0 rows.
            db.query(Paper).filter(Paper.paper_id.in_(cls.ids)).delete(
                synchronize_session=False)
            db.query(Document).filter(Document.id.in_(doc_ids)).delete(
                synchronize_session=False)
        api_routes._coverage_cache = None

    def setUp(self) -> None:
        # The endpoint caches for 5 minutes; a stale entry from an earlier test
        # would answer before this run's seed rows landed.
        api_routes._coverage_cache = None

    def _seeded_venue(self) -> dict:
        response = self.client.get("/api/papers/coverage")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        match = [v for v in body["venues"] if v["venue"] == self.venue]
        self.assertEqual(len(match), 1, f"seeded venue missing from {body['venue_count']} venues")
        return match[0]

    def test_year_span_and_count(self) -> None:
        venue = self._seeded_venue()
        self.assertEqual(venue["count"], len(self.years))
        self.assertEqual(venue["min_year"], 2019)
        self.assertEqual(venue["max_year"], 2022)

    def test_gap_year_is_reported_not_implied(self) -> None:
        """A venue indexed 2019,2021,2022 must not read as "2019-2022 complete"."""
        self.assertEqual(self._seeded_venue()["missing_years"], [2020])

    def test_per_year_counts(self) -> None:
        by_year = {row["year"]: row["count"] for row in self._seeded_venue()["years"]}
        self.assertEqual(by_year, {2019: 2, 2021: 1, 2022: 1})

    def test_totals_reconcile_with_venue_rows(self) -> None:
        """total_papers must equal the sum of the cards plus unclassified.

        The cards are the only thing the user sees; if the sum silently differed
        from the corpus total, a venue dropped by a NULL venue/year would be
        invisible.
        """
        body = self.client.get("/api/papers/coverage").json()
        self.assertEqual(
            body["total_papers"],
            sum(v["count"] for v in body["venues"]) + body["unclassified"],
        )

    def test_venues_sorted_by_count_desc(self) -> None:
        counts = [v["count"] for v in self.client.get("/api/papers/coverage").json()["venues"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_cache_serves_a_repeat_call(self) -> None:
        """Second call inside the TTL returns the identical cached object."""
        api_routes._coverage_cache = None
        first = api_routes._index_coverage_cached()
        second = api_routes._index_coverage_cached()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
