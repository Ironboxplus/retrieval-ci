"""TDD tests for EMNLP proceedings source (ACL Anthology)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    DEFAULT_TARGET_CONFERENCES,
    EmnlpMainConferenceSource,
    CorpusPaper,
    OfficialProceedingsClient,
    build_sources,
    load_conference_registry,
)


# ---------------------------------------------------------------------------
# Minimal ACL Anthology HTML fixture (same structure as ACL, EMNLP volumes)
# ---------------------------------------------------------------------------
EMNLP_2024_HTML = """
<html><body>
<div class="d-sm-flex align-items-stretch mb-3">
  <span class="d-block">
    <strong><a class="align-middle" href="/2024.emnlp-main.42/">
      Attention Is All You Need (Again)
    </a></strong>
    <a href="/people/j/jane-doe/">Jane Doe</a>
    <a href="/people/j/john-smith/">John Smith</a>
  </span>
  <a href="https://aclanthology.org/2024.emnlp-main.42.pdf" aria-label="Open PDF">pdf</a>
</div>

<div class="d-sm-flex align-items-stretch mb-3">
  <span class="d-block">
    <strong><a class="align-middle" href="/2024.emnlp-main.100/">
      Efficient Transformers: A Survey
    </a></strong>
    <a href="/people/a/alice-wang/">Alice Wang</a>
  </span>
  <a href="https://aclanthology.org/2024.emnlp-main.100.pdf" aria-label="Open PDF">pdf</a>
</div>

<!-- article_number == 0 should be skipped (volume preface) -->
<div class="d-sm-flex align-items-stretch mb-3">
  <span class="d-block">
    <strong><a class="align-middle" href="/2024.emnlp-main.0/">
      Preface
    </a></strong>
  </span>
  <a href="https://aclanthology.org/2024.emnlp-main.0.pdf" aria-label="Open PDF">pdf</a>
</div>

<!-- entry without PDF link should be skipped -->
<div class="d-sm-flex align-items-stretch mb-3">
  <span class="d-block">
    <strong><a class="align-middle" href="/2024.emnlp-main.999/">
      Ghost Paper
    </a></strong>
  </span>
</div>
</body></html>
"""


class TestEmnlpSourceExists(unittest.TestCase):
    """Source class exists and is importable."""

    def test_class_importable(self):
        self.assertTrue(hasattr(EmnlpMainConferenceSource, "collect_papers"))

    def test_conference_name(self):
        source = EmnlpMainConferenceSource()
        self.assertEqual(source.conference_name, "EMNLP")

    def test_volume_suffix_is_main_only(self):
        source = EmnlpMainConferenceSource()
        self.assertEqual(source.MAIN_VOLUME_SUFFIXES, ("main",))


class TestEmnlpRegistry(unittest.TestCase):
    """EMNLP registered in conferences.yaml + source map."""

    def test_emnlp_in_default_conferences(self):
        self.assertIn("EMNLP", DEFAULT_TARGET_CONFERENCES)

    def test_emnlp_in_build_sources(self):
        sources = build_sources()
        self.assertIn("EMNLP", sources)

    def test_emnlp_source_type(self):
        sources = build_sources()
        self.assertIsInstance(sources["EMNLP"], EmnlpMainConferenceSource)

    def test_emnlp_yaml_years(self):
        registry = load_conference_registry()
        emnlp_entry = next((e for e in registry if e["name"] == "EMNLP"), None)
        self.assertIsNotNone(emnlp_entry)
        for year in (2022, 2023, 2024, 2025):
            self.assertIn(year, emnlp_entry["years"])

    def test_emnlp_yaml_venue_tier(self):
        registry = load_conference_registry()
        emnlp_entry = next((e for e in registry if e["name"] == "EMNLP"), None)
        self.assertEqual(emnlp_entry["venue_tier"], "CCF-B")  # EMNLP is CCF-B (corrected prior sprint)


class TestEmnlpCollectPapers(unittest.TestCase):
    """collect_papers parses ACL Anthology HTML correctly."""

    def setUp(self):
        self.source = EmnlpMainConferenceSource()
        self.mock_client = MagicMock(spec=OfficialProceedingsClient)

        from bs4 import BeautifulSoup
        self.mock_client.get_soup.return_value = BeautifulSoup(EMNLP_2024_HTML, "html.parser")

    def test_returns_list_of_corpus_papers(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        self.assertIsInstance(papers, list)
        for p in papers:
            self.assertIsInstance(p, CorpusPaper)

    def test_correct_count_skips_preface_and_no_pdf(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        self.assertEqual(len(papers), 2)

    def test_paper_id_format(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        ids = {p.paper_id for p in papers}
        self.assertIn("emnlp-2024-main-42", ids)
        self.assertIn("emnlp-2024-main-100", ids)

    def test_pdf_url_format(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(
            by_id["emnlp-2024-main-42"].pdf_url,
            "https://aclanthology.org/2024.emnlp-main.42.pdf",
        )

    def test_paper_page_url(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(
            by_id["emnlp-2024-main-42"].paper_page_url,
            "https://aclanthology.org/2024.emnlp-main.42/",
        )

    def test_title_extracted(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(by_id["emnlp-2024-main-42"].title, "Attention Is All You Need (Again)")

    def test_authors_extracted(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        by_id = {p.paper_id: p for p in papers}
        self.assertEqual(by_id["emnlp-2024-main-42"].authors, ["Jane Doe", "John Smith"])

    def test_conference_field(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        for p in papers:
            self.assertEqual(p.conference, "EMNLP")

    def test_source_name(self):
        papers = self.source.collect_papers(2024, self.mock_client)
        for p in papers:
            self.assertEqual(p.source_name, "acl_anthology")

    def test_volume_url_called(self):
        self.source.collect_papers(2024, self.mock_client)
        self.mock_client.get_soup.assert_called_once_with(
            "https://aclanthology.org/volumes/2024.emnlp-main/"
        )


class TestEmnlpConferenceMap(unittest.TestCase):
    """CorpusPaper.to_runtime_paper works for EMNLP."""

    def test_to_runtime_paper_emnlp(self):
        paper = CorpusPaper(
            paper_id="emnlp-2024-main-42",
            title="Test",
            authors=["Author A"],
            conference="EMNLP",
            year=2024,
            pdf_url="https://aclanthology.org/2024.emnlp-main.42.pdf",
            paper_page_url="https://aclanthology.org/2024.emnlp-main.42/",
            source_name="acl_anthology",
        )
        runtime = paper.to_runtime_paper()
        from pipeline.paper_crawler.models.paper import Conference
        self.assertEqual(runtime.conference, Conference.EMNLP)


if __name__ == "__main__":
    unittest.main()
