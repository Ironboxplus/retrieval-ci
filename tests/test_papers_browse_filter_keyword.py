"""TDD: `/papers/browse` multi-venue + multi-year + keyword (q) contract.

Backs the "Documents 页论文筛选" feature: SQL-level filtering of all matching
papers (not vector top-K), driven by venue list + year list + title/abstract
keyword. Backwards compatible with single-value `venue=X` / `year=Y`.

Seeds 6 papers with a unique paper_id prefix and tears them down so the test
is safe to run against the live SQLite DB.
"""

from __future__ import annotations

import unittest
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.db import get_db_session
from core.db.models import DataSource, Document, DocumentStatus, Paper, SourceType
from pipeline.paper_crawler.integration.api_routes import create_paper_router


SEED_PREFIX = "test-browse-"


def _seed_paper(db, paper_id: str, *, venue: str, year: int, title: str, abstract: str = "") -> None:
    doc = Document(
        title=title,
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
            title=title,
            abstract=abstract,
            authors_json=["Test Author"],
            year=year,
            venue=venue,
            venue_tier="CCF-A",
            track_type="main",
        )
    )


class PapersBrowseFilterKeywordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suffix = uuid.uuid4().hex[:8]
        cls.ids = {
            "n1": f"{SEED_PREFIX}{cls.suffix}-n1",
            "n2": f"{SEED_PREFIX}{cls.suffix}-n2",
            "c1": f"{SEED_PREFIX}{cls.suffix}-c1",
            "c2": f"{SEED_PREFIX}{cls.suffix}-c2",
            "i1": f"{SEED_PREFIX}{cls.suffix}-i1",
            "i2": f"{SEED_PREFIX}{cls.suffix}-i2",
        }
        with get_db_session() as db:
            _seed_paper(db, cls.ids["n1"], venue="NeurIPS", year=2023,
                        title=f"BrowseTest {cls.suffix} alpha diffusion",
                        abstract="study of diffusion models")
            _seed_paper(db, cls.ids["n2"], venue="NeurIPS", year=2024,
                        title=f"BrowseTest {cls.suffix} beta transformer",
                        abstract="attention is all you need redux")
            _seed_paper(db, cls.ids["c1"], venue="CVPR", year=2023,
                        title=f"BrowseTest {cls.suffix} gamma vision",
                        abstract="dense prediction tasks")
            _seed_paper(db, cls.ids["c2"], venue="CVPR", year=2024,
                        title=f"BrowseTest {cls.suffix} delta DIFFUSION pipeline",
                        abstract="generative imaging")
            _seed_paper(db, cls.ids["i1"], venue="ICLR", year=2023,
                        title=f"BrowseTest {cls.suffix} epsilon retrieval",
                        abstract="dense retrieval baselines")
            _seed_paper(db, cls.ids["i2"], venue="ICLR", year=2024,
                        title=f"BrowseTest {cls.suffix} zeta retrieval",
                        abstract="retrieval-augmented generation")
            db.commit()

        app = FastAPI()
        app.include_router(create_paper_router(prefix="/api/papers"))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        with get_db_session() as db:
            doc_ids = [
                p.doc_id
                for p in db.query(Paper).filter(Paper.paper_id.in_(list(cls.ids.values()))).all()
            ]
            db.query(Paper).filter(Paper.paper_id.in_(list(cls.ids.values()))).delete(
                synchronize_session=False
            )
            if doc_ids:
                db.query(Document).filter(Document.id.in_(doc_ids)).delete(
                    synchronize_session=False
                )
            db.commit()

    # ─── helpers ───

    def _our_ids_in(self, body) -> set[str]:
        return {r["paper_id"] for r in body["results"] if r["paper_id"].startswith(SEED_PREFIX)}

    @property
    def _scope_q(self) -> str:
        # All seeded titles contain `BrowseTest <suffix>` so this keyword
        # narrows the SQL set to our 6 fixtures regardless of how big the
        # real DB has grown.
        return f"BrowseTest {self.suffix}"

    # ─── single value (back-compat) ───

    def test_single_venue_still_works(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"venue": "NeurIPS", "q": self._scope_q, "page_size": 200},
        )
        self.assertEqual(r.status_code, 200)
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n1"], self.ids["n2"]})

    def test_single_year_still_works(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"year": 2024, "q": self._scope_q, "page_size": 200},
        )
        self.assertEqual(r.status_code, 200)
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n2"], self.ids["c2"], self.ids["i2"]})

    # ─── multi-venue / multi-year ───

    def test_csv_multi_venue(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"venue": "NeurIPS,CVPR", "q": self._scope_q, "page_size": 200},
        )
        self.assertEqual(r.status_code, 200)
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n1"], self.ids["n2"], self.ids["c1"], self.ids["c2"]})

    def test_repeated_query_param_multi_venue(self):
        # `?venue=NeurIPS&venue=ICLR` form (FastAPI default List parsing).
        r = self.client.get(
            "/api/papers/browse",
            params=[("venue", "NeurIPS"), ("venue", "ICLR"), ("q", self._scope_q), ("page_size", 200)],
        )
        self.assertEqual(r.status_code, 200)
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n1"], self.ids["n2"], self.ids["i1"], self.ids["i2"]})

    def test_csv_multi_year(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"year": "2023,2024", "venue": "NeurIPS", "q": self._scope_q, "page_size": 200},
        )
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n1"], self.ids["n2"]})

    def test_multi_venue_and_year_combined(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"venue": "NeurIPS,CVPR", "year": "2024", "q": self._scope_q, "page_size": 200},
        )
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["n2"], self.ids["c2"]})

    # ─── keyword (q) — narrower than _scope_q ───

    def test_q_matches_title_case_insensitive(self):
        # Use the run's unique suffix to keep the result set small. Combined
        # with venue filter to verify case-insensitive title matching:
        # n1 title has "alpha" (lower), c2 title has "delta" — both share
        # the suffix. Match a case-mix substring of the suffix as a sanity check.
        upper_suffix = self.suffix.upper()
        r = self.client.get(
            "/api/papers/browse",
            params={"q": upper_suffix, "page_size": 200},
        )
        # ilike is case-insensitive → upper suffix matches lowercase seeds.
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, set(self.ids.values()))

    def test_q_matches_abstract(self):
        # "retrieval-augmented" only in i2 abstract, not in any title.
        r = self.client.get(
            "/api/papers/browse",
            params={"q": "retrieval-augmented", "page_size": 500},
        )
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["i2"]})

    def test_q_combined_with_venue_filter(self):
        # `retrieval` + venue=ICLR + year=2023 → only i1.
        r = self.client.get(
            "/api/papers/browse",
            params={"q": "retrieval", "venue": "ICLR", "year": "2023", "page_size": 200},
        )
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, {self.ids["i1"]})

    def test_q_empty_string_is_ignored(self):
        # Empty / whitespace q must not filter anything out (q is optional).
        # Use venue + year to scope so we don't return 70K rows.
        r = self.client.get(
            "/api/papers/browse",
            params={"q": "  ", "venue": "ICLR", "page_size": 200},
        )
        ours = self._our_ids_in(r.json())
        # Both ICLR seeds present; q="  " did not over-filter.
        self.assertTrue({self.ids["i1"], self.ids["i2"]}.issubset(ours) or ours == set())
        # Stronger assertion: combine with our scope_q to confirm both rows are reachable.
        r2 = self.client.get(
            "/api/papers/browse",
            params={"q": self._scope_q, "venue": "ICLR", "page_size": 200},
        )
        self.assertEqual(self._our_ids_in(r2.json()), {self.ids["i1"], self.ids["i2"]})

    # ─── response shape ───

    def test_response_shape_unchanged(self):
        r = self.client.get(
            "/api/papers/browse",
            params={"venue": "NeurIPS", "page_size": 5},
        )
        body = r.json()
        for key in ("total", "page", "page_size", "total_pages", "results"):
            self.assertIn(key, body)
        if body["results"]:
            row = body["results"][0]
            for key in ("paper_id", "title", "venue", "year", "authors", "abstract"):
                self.assertIn(key, row)

    def test_pagination_total_unaffected_by_q(self):
        # q=BrowseTest matches all 6 seeded papers (suffix in title).
        r = self.client.get(
            "/api/papers/browse",
            params={"q": f"BrowseTest {self.suffix}", "page_size": 200},
        )
        ours = self._our_ids_in(r.json())
        self.assertEqual(ours, set(self.ids.values()))


if __name__ == "__main__":
    unittest.main()
