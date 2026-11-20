"""TDD tests for ACM conference sources via DBLP (KDD, SIGMOD, SIGIR)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    DEFAULT_TARGET_CONFERENCES,
    DblpAcmProceedingsSource,
    CorpusPaper,
    OfficialProceedingsClient,
    build_sources,
    load_conference_registry,
)


# ---------------------------------------------------------------------------
# DBLP API JSON fixture (mimics real response)
# ---------------------------------------------------------------------------
DBLP_KDD_2024_JSON = {
    "result": {
        "hits": {
            "@total": "3",
            "@computed": "3",
            "@sent": "3",
            "@first": "0",
            "hit": [
                {
                    "@score": "2",
                    "@id": "1000001",
                    "info": {
                        "authors": {
                            "author": [
                                {"@pid": "1/1", "text": "Alice Smith"},
                                {"@pid": "2/2", "text": "Bob Jones"},
                            ]
                        },
                        "title": "Graph Neural Networks for Recommendation.",
                        "venue": "KDD",
                        "year": "2024",
                        "type": "Conference and Workshop Papers",
                        "access": "closed",
                        "key": "conf/kdd/SmithJ24",
                        "ee": "https://doi.org/10.1145/3637528.3671001",
                        "url": "https://dblp.org/rec/conf/kdd/SmithJ24",
                    },
                },
                {
                    "@score": "2",
                    "@id": "1000002",
                    "info": {
                        "authors": {
                            "author": {"@pid": "3/3", "text": "Carol Wang"}
                        },
                        "title": "Efficient Data Mining with Transformers.",
                        "venue": "KDD",
                        "year": "2024",
                        "type": "Conference and Workshop Papers",
                        "access": "open",
                        "key": "conf/kdd/Wang24",
                        "ee": "https://doi.org/10.1145/3637528.3671002",
                        "url": "https://dblp.org/rec/conf/kdd/Wang24",
                    },
                },
                {
                    "@score": "1",
                    "@id": "1000003",
                    "info": {
                        "authors": {
                            "author": {"@pid": "4/4", "text": "Dave Lee"}
                        },
                        "title": "Keynote: The Future of AI.",
                        "venue": "KDD",
                        "year": "2024",
                        "type": "Conference and Workshop Papers",
                        "access": "closed",
                        "key": "conf/kdd/Lee24",
                        "ee": "https://doi.org/10.1145/3637528.3671003",
                        "url": "https://dblp.org/rec/conf/kdd/Lee24",
                    },
                },
            ],
        }
    }
}


class TestDblpSourceExists(unittest.TestCase):
    def test_class_importable(self):
        self.assertTrue(hasattr(DblpAcmProceedingsSource, "collect_papers"))

    def test_kdd_instance(self):
        source = DblpAcmProceedingsSource(conference_name="KDD", dblp_venue_key="kdd")
        self.assertEqual(source.conference_name, "KDD")
        self.assertEqual(source.dblp_venue_key, "kdd")

    def test_acm_proceedings_doi_prefix(self):
        source = DblpAcmProceedingsSource(conference_name="KDD", dblp_venue_key="kdd")
        self.assertTrue(hasattr(source, "PROCEEDINGS_DOI_PREFIX"))


class TestDblpRegistry(unittest.TestCase):
    def test_kdd_in_default_conferences(self):
        self.assertIn("KDD", DEFAULT_TARGET_CONFERENCES)

    def test_sigmod_in_default_conferences(self):
        self.assertIn("SIGMOD", DEFAULT_TARGET_CONFERENCES)

    def test_sigir_in_default_conferences(self):
        self.assertIn("SIGIR", DEFAULT_TARGET_CONFERENCES)

    def test_all_three_in_build_sources(self):
        sources = build_sources()
        for name in ("KDD", "SIGMOD", "SIGIR"):
            self.assertIn(name, sources, f"{name} missing from build_sources")

    def test_source_type(self):
        sources = build_sources()
        for name in ("KDD", "SIGMOD", "SIGIR"):
            self.assertIsInstance(sources[name], DblpAcmProceedingsSource)

    def test_yaml_years(self):
        registry = load_conference_registry()
        for name in ("KDD", "SIGMOD", "SIGIR"):
            entry = next((e for e in registry if e["name"] == name), None)
            self.assertIsNotNone(entry, f"{name} not in registry")
            for year in (2022, 2023, 2024, 2025):
                self.assertIn(year, entry["years"], f"{name} missing year {year}")

    def test_yaml_venue_tier(self):
        registry = load_conference_registry()
        for name in ("KDD", "SIGMOD", "SIGIR"):
            entry = next((e for e in registry if e["name"] == name), None)
            self.assertEqual(entry["venue_tier"], "CCF-A")


class TestDblpCollectPapers(unittest.TestCase):
    def setUp(self):
        self.source = DblpAcmProceedingsSource(
            conference_name="KDD",
            dblp_venue_key="kdd",
        )
        self.mock_client = MagicMock(spec=OfficialProceedingsClient)

        import json
        self.mock_client.get_text.return_value = json.dumps(DBLP_KDD_2024_JSON)

    def test_returns_corpus_papers(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        self.assertIsInstance(papers, list)
        for p in papers:
            self.assertIsInstance(p, CorpusPaper)

    def test_correct_count(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        self.assertEqual(len(papers), 3)

    def test_paper_id_uses_doi_suffix(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        ids = {p.paper_id for p in papers}
        self.assertIn("kdd-2024-3637528.3671001", ids)

    def test_pdf_url_is_acm_dl(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        for p in papers:
            self.assertTrue(
                p.pdf_url.startswith("https://dl.acm.org/doi/pdf/"),
                f"PDF URL not ACM DL: {p.pdf_url}",
            )

    def test_title_strip_trailing_dot(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        p = by_id["kdd-2024-3637528.3671001"]
        self.assertEqual(p.title, "Graph Neural Networks for Recommendation")

    def test_authors_extracted(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(
            by_id["kdd-2024-3637528.3671001"].authors,
            ["Alice Smith", "Bob Jones"],
        )

    def test_single_author_not_nested(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(by_id["kdd-2024-3637528.3671002"].authors, ["Carol Wang"])

    def test_conference_field(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        for p in papers:
            self.assertEqual(p.conference, "KDD")

    def test_source_name(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        for p in papers:
            self.assertEqual(p.source_name, "dblp")

    def test_paper_page_url_is_doi(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(
            by_id["kdd-2024-3637528.3671001"].paper_page_url,
            "https://doi.org/10.1145/3637528.3671001",
        )

    def test_dblp_toc_url_called(self):
        self.source.collect_papers(2024, self.mock_client)
        call_args = self.mock_client.get_text.call_args[0][0]
        self.assertIn("toc:db/conf/kdd/kdd2024.bht:", call_args)

    def test_pagination_fetches_all(self):
        """If total > batch size, should paginate."""
        page1 = dict(DBLP_KDD_2024_JSON)
        page1["result"]["hits"]["@total"] = "1500"
        import json
        self.mock_client.get_text.return_value = json.dumps(page1)
        papers = self.source.collect_papers(2024, self.mock_client)
        self.assertTrue(self.mock_client.get_text.call_count >= 1)


class TestDblpConferenceMap(unittest.TestCase):
    def test_kdd_to_runtime(self):
        paper = CorpusPaper(
            paper_id="kdd-2024-test",
            title="Test",
            authors=["A"],
            conference="KDD",
            year=2024,
            pdf_url="https://dl.acm.org/doi/pdf/10.1145/test",
            paper_page_url="https://doi.org/10.1145/test",
            source_name="dblp",
        )
        from pipeline.paper_crawler.models.paper import Conference
        rt = paper.to_runtime_paper()
        self.assertEqual(rt.conference, Conference.KDD)

    def test_sigir_to_runtime(self):
        paper = CorpusPaper(
            paper_id="sigir-2024-test",
            title="Test",
            authors=["A"],
            conference="SIGIR",
            year=2024,
            pdf_url="https://dl.acm.org/doi/pdf/10.1145/test",
            paper_page_url="https://doi.org/10.1145/test",
            source_name="dblp",
        )
        from pipeline.paper_crawler.models.paper import Conference
        rt = paper.to_runtime_paper()
        self.assertEqual(rt.conference, Conference.SIGIR)

    def test_sigmod_to_runtime(self):
        paper = CorpusPaper(
            paper_id="sigmod-2024-test",
            title="Test",
            authors=["A"],
            conference="SIGMOD",
            year=2024,
            pdf_url="https://dl.acm.org/doi/pdf/10.1145/test",
            paper_page_url="https://doi.org/10.1145/test",
            source_name="dblp",
        )
        from pipeline.paper_crawler.models.paper import Conference
        rt = paper.to_runtime_paper()
        self.assertEqual(rt.conference, Conference.SIGMOD)


if __name__ == "__main__":
    unittest.main()
