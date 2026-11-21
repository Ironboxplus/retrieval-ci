"""TDD for NdssSymposiumSource — NDSS main-symposium open-access crawler.

NDSS is fully open access at ndss-symposium.org. The authoritative MAIN list is
the per-year ``/ndss{year}/accepted-papers/`` index; workshops (USEC / WOSOC /
MADWeb / ...) and posters live on separate pages and are therefore structurally
excluded.  Each entry links to ``/ndss-paper/{slug}/`` whose page carries the
camera-ready PDF ``/wp-content/uploads/{year}-{cycleid}-paper.pdf``.

Main-only guarantees asserted here (the user requirement — every paper must be a
main-conference paper, never a demo/poster/slides/keynote):
  - only ``/ndss-paper/`` links are taken (home/nav links ignored)
  - the camera-ready ``*-paper.pdf`` is chosen; ``*-slides.pdf`` / ``*-poster.pdf``
    are rejected and never leak into pdf_url or alternate_pdf_urls
  - paper_id derives from the stable cycle id (``ndss-2026-s1950``)
  - authors are parsed off the affiliation-parenthesised block
  - a main paper with no ``-paper.pdf`` is flagged, not given a slides URL
  - runtime track is always ``main_conference``
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    OfficialProceedingsClient,
    NdssSymposiumSource,
)

INDEX = """
<html><body>
  <a href="https://www.ndss-symposium.org/ndss-paper/cold-framework/">CoLD: A Real Paper</a>
  <a href="https://www.ndss-symposium.org/ndss-paper/jailbreak-causal/">A Causal Perspective</a>
  <a href="https://www.ndss-symposium.org/ndss-paper/onlyslides-talk/">Slides Only Entry</a>
  <a href="https://www.ndss-symposium.org/ndss2026/">NDSS 2026 home</a>
  <a href="https://www.ndss-symposium.org/ndss-paper/cold-framework/">CoLD dup link</a>
</body></html>
"""

PAGE_COLD = """
<html><body>
  <h1>CoLD: Collaborative Label Denoising Framework</h1>
  <div class="paper-data">Shuo Yang (The University of Hong Kong, Hong Kong SAR, China), Xinran Zheng (University College London, London, United Kingdom)</div>
  <div class="entry-content">
    <p>Shuo Yang (The University of Hong Kong, Hong Kong SAR, China), Xinran Zheng (University College London, London, United Kingdom)</p>
    <p>Label noise presents a significant challenge in network intrusion detection, leading to erroneous classifications and decreased detection accuracy across systems.</p>
  </div>
  <a href="https://www.ndss-symposium.org/wp-content/uploads/2026-s1950-paper.pdf">Paper (PDF)</a>
  <a href="https://www.ndss-symposium.org/wp-content/uploads/s1950-yang-slides.pdf">Slides (PDF)</a>
</body></html>
"""

PAGE_JAIL = """
<html><body>
  <h1>A Causal Perspective for Enhancing Jailbreak Attack and Defense</h1>
  <div class="paper-data">Wei Chen (MIT, USA)</div>
  <div class="entry-content">
    <p>Wei Chen (MIT, USA)</p>
    <p>We study jailbreak attacks and defenses through a causal lens, deriving a framework that explains transferability across many target models in practice.</p>
  </div>
  <a href="https://www.ndss-symposium.org/wp-content/uploads/2026-f0042-paper.pdf">Paper (PDF)</a>
</body></html>
"""

# A main paper whose camera-ready PDF is not yet posted (only slides present).
PAGE_SLIDES_ONLY = """
<html><body>
  <h1>Slides Only Entry</h1>
  <div class="paper-data">Pat Doe (Some University, Country)</div>
  <div class="entry-content"><p>Pat Doe (Some University, Country)</p>
    <p>This camera ready paper has not been posted yet so only slides are available right now.</p></div>
  <a href="https://www.ndss-symposium.org/wp-content/uploads/s0001-doe-slides.pdf">Slides (PDF)</a>
</body></html>
"""


def _client():
    c = MagicMock(spec=OfficialProceedingsClient)

    def soup(url):
        if url.endswith("/accepted-papers/"):
            html = INDEX
        elif "cold-framework" in url:
            html = PAGE_COLD
        elif "jailbreak-causal" in url:
            html = PAGE_JAIL
        elif "onlyslides-talk" in url:
            html = PAGE_SLIDES_ONLY
        else:
            html = "<html></html>"
        return BeautifulSoup(html, "html.parser")

    c.get_soup.side_effect = soup
    return c


class TestNdssSymposiumSource(unittest.TestCase):
    def setUp(self):
        self.src = NdssSymposiumSource()
        self.papers = self.src.collect_papers(2026, _client())
        self.by_id = {p.paper_id: p for p in self.papers}

    def test_collects_three_distinct_papers_dedup_home_ignored(self):
        # 3 distinct /ndss-paper/ slugs; the /ndss2026/ home link is not a paper;
        # the duplicate CoLD link collapses to one.
        self.assertEqual(len(self.papers), 3)

    def test_paper_id_from_cycle_id(self):
        self.assertIn("ndss-2026-s1950", self.by_id)
        self.assertIn("ndss-2026-f0042", self.by_id)

    def test_camera_ready_pdf_chosen_not_slides(self):
        p = self.by_id["ndss-2026-s1950"]
        self.assertEqual(
            p.pdf_url,
            "https://www.ndss-symposium.org/wp-content/uploads/2026-s1950-paper.pdf",
        )
        self.assertNotIn("slides", p.pdf_url)

    def test_slides_never_leak_into_alternates(self):
        p = self.by_id["ndss-2026-s1950"]
        self.assertTrue(all("slides" not in u for u in p.alternate_pdf_urls))

    def test_authors_parsed_without_affiliation(self):
        p = self.by_id["ndss-2026-s1950"]
        self.assertEqual(p.authors, ["Shuo Yang", "Xinran Zheng"])

    def test_abstract_captured_not_author_line(self):
        p = self.by_id["ndss-2026-s1950"]
        self.assertIn("Label noise", p.abstract)
        self.assertNotIn("Shuo Yang", p.abstract)

    def test_conference_and_source_name(self):
        p = self.by_id["ndss-2026-s1950"]
        self.assertEqual(p.conference, "NDSS")
        self.assertEqual(p.source_name, "ndss_accepted_papers")

    def test_runtime_track_is_main_conference(self):
        rp = self.by_id["ndss-2026-s1950"].to_runtime_paper()
        self.assertEqual(rp.metadata.track, "main_conference")

    def test_slides_only_paper_flagged_not_given_slides_url(self):
        # The slides-only main paper is still emitted (it IS a main paper) but its
        # pdf_url must be empty + flagged, never the slides URL.
        slides_only = [p for p in self.papers if p.title == "Slides Only Entry"]
        self.assertEqual(len(slides_only), 1)
        p = slides_only[0]
        self.assertEqual(p.pdf_url, "")
        self.assertEqual(p.pdf_resolution_status, "missing_camera_ready")
        self.assertTrue(all("slides" not in (u or "") for u in p.alternate_pdf_urls))

    def test_supports_year_window(self):
        self.assertTrue(self.src.supports_year(2024))
        self.assertTrue(self.src.supports_year(2026))
        self.assertFalse(self.src.supports_year(2015))


if __name__ == "__main__":
    unittest.main()
