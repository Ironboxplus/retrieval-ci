"""TDD for the generalized UsenixProceedingsSource (FAST / USENIX ATC / Security).

Uses a mocked technical-sessions HTML fixture (no network). Verifies:
  - deterministic PDF URL pattern  system/files/{slug}{YY}-{paper}.pdf
  - paper_id namespacing per venue
  - non-paper (keynote) entries dropped by sub-type taxonomy
  - UsenixSecuritySource still emits legacy uss- ids + sec{YY}- (<=2022) prefix
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    OfficialProceedingsClient,
    UsenixProceedingsSource,
    UsenixSecuritySource,
)


def _index_html(url_slug: str, yy: str) -> str:
    """Minimal USENIX technical-sessions page: one real paper + one keynote."""
    return f"""
    <html><body>
      <article class="node-paper">
        <h2><a href="/conference/{url_slug}{yy}/presentation/yang-zhe">A Real Research Paper</a></h2>
        <div class="field-name-field-paper-people-text">Zhe Yang, Alice Liu <em>Tsinghua University</em></div>
      </article>
      <article class="node-paper">
        <h2><a href="/conference/{url_slug}{yy}/presentation/wu-keynote">Opening Keynote</a></h2>
        <div class="field-name-field-paper-sub-type"><div class="field-item">Keynote</div></div>
        <div class="field-name-field-paper-people-text">Famous Person <em>Big Co</em></div>
      </article>
    </body></html>
    """


def _collect(source, url_slug, yy, year):
    client = MagicMock(spec=OfficialProceedingsClient)
    client.get_soup.return_value = BeautifulSoup(_index_html(url_slug, yy), "html.parser")
    return source.collect_papers(year, client)


class TestUsenixProceedingsSource(unittest.TestCase):
    def test_fast_parses_real_paper(self):
        src = UsenixProceedingsSource(conference_name="FAST", url_slug="fast")
        papers = _collect(src, "fast", "24", 2024)
        self.assertEqual(len(papers), 1)  # keynote filtered out
        p = papers[0]
        self.assertEqual(p.paper_id, "fast-2024-yang-zhe")
        self.assertEqual(p.conference, "FAST")
        self.assertEqual(p.title, "A Real Research Paper")
        self.assertEqual(p.pdf_url, "https://www.usenix.org/system/files/fast24-yang-zhe.pdf")
        self.assertIn("Zhe Yang", p.authors)
        self.assertIn("Alice Liu", p.authors)

    def test_atc_pdf_pattern_and_id(self):
        src = UsenixProceedingsSource(conference_name="USENIX ATC", url_slug="atc")
        papers = _collect(src, "atc", "24", 2024)
        self.assertEqual(papers[0].paper_id, "atc-2024-yang-zhe")
        self.assertEqual(papers[0].pdf_url, "https://www.usenix.org/system/files/atc24-yang-zhe.pdf")

    def test_keynote_filtered(self):
        src = UsenixProceedingsSource(conference_name="FAST", url_slug="fast")
        ids = {p.paper_id for p in _collect(src, "fast", "24", 2024)}
        self.assertNotIn("fast-2024-wu-keynote", ids)

    def test_supports_year_window(self):
        src = UsenixProceedingsSource(conference_name="FAST", url_slug="fast")
        self.assertTrue(src.supports_year(2022))
        self.assertTrue(src.supports_year(2025))

    def test_source_name_default(self):
        src = UsenixProceedingsSource(conference_name="FAST", url_slug="fast")
        papers = _collect(src, "fast", "24", 2024)
        self.assertEqual(papers[0].source_name, "usenix_fast_techsessions")


class TestUsenixSecurityBackwardCompat(unittest.TestCase):
    def test_legacy_paper_id_and_modern_prefix(self):
        src = UsenixSecuritySource()
        p = _collect(src, "usenixsecurity", "24", 2024)[0]
        self.assertEqual(p.paper_id, "uss-2024-yang-zhe")
        self.assertEqual(p.conference, "USENIX Security")
        self.assertEqual(p.pdf_url, "https://www.usenix.org/system/files/usenixsecurity24-yang-zhe.pdf")
        self.assertEqual(p.source_name, "usenix_security_techsessions")

    def test_pre_2023_sec_prefix(self):
        src = UsenixSecuritySource()
        p = _collect(src, "usenixsecurity", "22", 2022)[0]
        self.assertEqual(p.pdf_url, "https://www.usenix.org/system/files/sec22-yang-zhe.pdf")

    def test_is_subclass(self):
        self.assertTrue(issubclass(UsenixSecuritySource, UsenixProceedingsSource))


if __name__ == "__main__":
    unittest.main()
