"""TDD for the venue track classifier.

Maps a DBLP section heading (+ optional title for flat venues) to a `track_type`
label and an `is_main` flag. Cases are taken verbatim from real DBLP TOCs audited
2026-06-03 (SIGIR/KDD/PPoPP/DAC/ISCA). The classifier only recognises the closed
vocabulary of NON-research tracks; everything unrecognised defaults to research
(keep-by-default, never silently drop -- see project memory W89 lesson).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.track_classifier import (  # noqa: E402
    classify_track,
    TrackVerdict,
    MAIN_TRACKS,
)


class TestMainTracks(unittest.TestCase):
    def test_research_session_names_are_main(self):
        for sec in ["ML Workloads", "Session: Neural IR", "Session 4A: LLMs",
                    "Synchronization and Concurrency Control I", "Graph Processing",
                    "Session 2A: Best Paper Nominees"]:
            with self.subTest(sec=sec):
                v = classify_track(sec)
                self.assertEqual(v.track_type, "research")
                self.assertTrue(v.is_main)

    def test_short_papers_are_main(self):
        for sec in ["Session: Short Research Papers", "Short Papers", "Short Paper Track"]:
            with self.subTest(sec=sec):
                v = classify_track(sec)
                self.assertEqual(v.track_type, "short")
                self.assertTrue(v.is_main)

    def test_industry_tracks_are_main(self):
        for sec in ["Session: SIRIP: LLMs 1", "Session 9A: Industry Track",
                    "Applied Data Papers", "Industrial Track"]:
            with self.subTest(sec=sec):
                v = classify_track(sec)
                self.assertEqual(v.track_type, "industry")
                self.assertTrue(v.is_main)


class TestNonMainTracks(unittest.TestCase):
    CASES = [
        ("Workshops", "workshop"),
        ("Workshop Summaries", "workshop"),
        ("Hands-On Tutorials", "tutorial"),
        ("Lecture-Style Tutorials", "tutorial"),
        ("Tutorials", "tutorial"),
        ("Doctoral Consortium", "doctoral"),
        ("Keynote Talks", "keynote"),
        ("Keynote", "keynote"),
        ("Applied Data Invited Talks", "invited"),
        ("Session: SIRIP: Panel", "panel"),
        ("Demo Papers", "demo"),
        ("Demonstrations", "demo"),
        ("POSTER SESSION: Posters", "poster"),
    ]

    def test_non_main_sections(self):
        for sec, tt in self.CASES:
            with self.subTest(sec=sec):
                v = classify_track(sec)
                self.assertEqual(v.track_type, tt, f"{sec!r} -> {v.track_type!r}, want {tt!r}")
                self.assertFalse(v.is_main)


class TestFlatVenueTitlePrefix(unittest.TestCase):
    """Flat venues (DAC/EuroSys) have no section headers -> fall back to title prefix.
    Prefix match REQUIRES a trailing colon to avoid false hits like 'Posterior...'."""

    def test_dac_style_prefixes(self):
        cases = [
            ("POSTER: RadiK: Scalable Radix Top-K Selection on GPUs", "poster"),
            ("Late Breaking Results: A Fast Synthesis Flow", "late-breaking"),
            ("Invited: Trends in Chiplet Design", "invited"),
        ]
        for title, tt in cases:
            with self.subTest(title=title):
                v = classify_track("", title)         # no section -> title drives it
                self.assertEqual(v.track_type, tt)
                self.assertFalse(v.is_main)

    def test_no_colon_no_false_positive(self):
        # A genuine research title that merely starts with these letters must stay research.
        for title in ["Posterior Probability Matters: Doubly-Adaptive Calibration",
                      "Demonstrating Quantum Advantage via Sampling",
                      "Invited Reviewers Considered Harmful"]:
            with self.subTest(title=title):
                v = classify_track("", title)
                self.assertEqual(v.track_type, "research")
                self.assertTrue(v.is_main)

    def test_clean_flat_section_defaults_research(self):
        # EuroSys flat TOC: no section, normal title -> research.
        v = classify_track("", "Jade: A High-throughput Concurrent Copying Garbage Collector")
        self.assertEqual(v.track_type, "research")
        self.assertTrue(v.is_main)


class TestRegressionGuards(unittest.TestCase):
    def test_ctr_ads_session_is_research_not_industry(self):
        # 'Ads' (advertising research) must NOT be mislabelled industry via a bare \bads\b.
        v = classify_track("Session: CTR, Ads and Click Models")
        self.assertEqual(v.track_type, "research")
        self.assertTrue(v.is_main)

    def test_section_takes_precedence_over_title(self):
        # If the section says demo, a normal-looking title doesn't rescue it.
        v = classify_track("Demo Papers", "A Scalable Vector Database")
        self.assertEqual(v.track_type, "demo")
        self.assertFalse(v.is_main)

    def test_main_tracks_constant(self):
        self.assertEqual(MAIN_TRACKS, frozenset({"research", "short", "industry"}))

    def test_verdict_is_main_derived_from_track(self):
        self.assertTrue(TrackVerdict("research").is_main)
        self.assertTrue(TrackVerdict("short").is_main)
        self.assertTrue(TrackVerdict("industry").is_main)
        self.assertFalse(TrackVerdict("workshop").is_main)
        self.assertFalse(TrackVerdict("poster").is_main)


class TestParseDblpTocTracks(unittest.TestCase):
    """parse_dblp_toc_tracks() maps DOI -> track from a DBLP HTML TOC."""

    SECTIONED = """
    <div id="main">
      <header class="h2"><h2 id="a">Research Papers</h2></header>
      <ul class="publ-list">
        <li class="entry inproceedings"><cite><span class="title">A Great Result.</span></cite>
          <nav class="publ"><a href="https://doi.org/10.1145/1111.2222">doi</a></nav></li>
      </ul>
      <header class="h2"><h2 id="b">Demo Papers</h2></header>
      <ul class="publ-list">
        <li class="entry inproceedings"><cite><span class="title">Demonstration of Foo.</span></cite>
          <nav class="publ"><a href="https://doi.org/10.1145/1111.3333">doi</a></nav></li>
      </ul>
      <header class="h2"><h2 id="c">Workshops</h2></header>
      <ul class="publ-list">
        <li class="entry inproceedings"><cite><span class="title">Report on the X Workshop.</span></cite>
          <nav class="publ"><a href="https://doi.org/10.1145/1111.4444">doi</a></nav></li>
        <li class="entry editor"><cite><span class="title">Front Matter.</span></cite></li>
      </ul>
    </div>
    """

    def test_sectioned_doi_to_track(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import parse_dblp_toc_tracks
        doi_track, manifest = parse_dblp_toc_tracks(self.SECTIONED)
        self.assertEqual(doi_track["10.1145/1111.2222"], "research")
        self.assertEqual(doi_track["10.1145/1111.3333"], "demo")
        self.assertEqual(doi_track["10.1145/1111.4444"], "workshop")
        # editor / front-matter entry has no DOI link and isn't an inproceedings -> skipped
        self.assertEqual(len(doi_track), 3)
        self.assertEqual(manifest["Demo Papers"], {"demo": 1})
        self.assertEqual(manifest["Workshops"], {"workshop": 1})

    def test_flat_toc_uses_title_prefix(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import parse_dblp_toc_tracks
        flat = """
        <div id="main"><ul class="publ-list">
          <li class="entry inproceedings"><cite><span class="title">Late Breaking Results: A Fast Flow.</span></cite>
            <nav><a href="https://doi.org/10.1145/9.1">doi</a></nav></li>
          <li class="entry inproceedings"><cite><span class="title">A Real Architecture Paper.</span></cite>
            <nav><a href="https://doi.org/10.1145/9.2">doi</a></nav></li>
        </ul></div>
        """
        doi_track, _ = parse_dblp_toc_tracks(flat)
        self.assertEqual(doi_track["10.1145/9.1"], "late-breaking")
        self.assertEqual(doi_track["10.1145/9.2"], "research")

    def test_page_count_downgrades_short_main_entry_to_poster(self):
        # W112: an UNSECTIONED TOC (e.g. PPoPP 2022) defaults every entry to
        # 'research'. A 2-page entry is structurally a poster/extended-abstract,
        # not a main paper. The page-count guard reclassifies <=4pp main entries
        # as poster so filter_main_track_papers drops them.
        from pipeline.paper_crawler.ccf_a_main_corpus import parse_dblp_toc_tracks
        flat = """
        <div id="main"><ul class="publ-list">
          <li class="entry inproceedings"><cite><span class="title">A Full Research Paper.</span>
            <span itemprop="pagination">1-16</span></cite>
            <nav><a href="https://doi.org/10.1145/7.1">doi</a></nav></li>
          <li class="entry inproceedings"><cite><span class="title">Towards Some Parallel Idea.</span>
            <span itemprop="pagination">431-432</span></cite>
            <nav><a href="https://doi.org/10.1145/7.2">doi</a></nav></li>
          <li class="entry inproceedings"><cite><span class="title">A Three Page Abstract.</span>
            <span itemprop="pagination">445-447</span></cite>
            <nav><a href="https://doi.org/10.1145/7.3">doi</a></nav></li>
        </ul></div>
        """
        doi_track, _ = parse_dblp_toc_tracks(flat)
        self.assertEqual(doi_track["10.1145/7.1"], "research")   # 16pp full paper kept
        self.assertEqual(doi_track["10.1145/7.2"], "poster")     # 2pp downgraded
        self.assertEqual(doi_track["10.1145/7.3"], "poster")     # 3pp downgraded

    def test_page_count_guard_keeps_full_papers_and_ignores_nonmain(self):
        # >=5pp main entries are untouched; an already-non-main short entry stays
        # non-main (guard only acts on main-classified rows, never up-classifies).
        from pipeline.paper_crawler.ccf_a_main_corpus import parse_dblp_toc_tracks
        html = """
        <div id="main">
          <header class="h2"><h2 id="a">Research Papers</h2></header>
          <ul class="publ-list">
            <li class="entry inproceedings"><cite><span class="title">Edge Case Five Pager.</span>
              <span itemprop="pagination">10-14</span></cite>
              <nav><a href="https://doi.org/10.1145/6.1">doi</a></nav></li>
          </ul>
          <header class="h2"><h2 id="b">Posters</h2></header>
          <ul class="publ-list">
            <li class="entry inproceedings"><cite><span class="title">A Poster.</span>
              <span itemprop="pagination">99-100</span></cite>
              <nav><a href="https://doi.org/10.1145/6.2">doi</a></nav></li>
          </ul>
        </div>
        """
        doi_track, _ = parse_dblp_toc_tracks(html)
        self.assertEqual(doi_track["10.1145/6.1"], "research")   # 5pp kept (>4)
        self.assertEqual(doi_track["10.1145/6.2"], "poster")     # section already poster


class TestFilterMainTrackPapers(unittest.TestCase):
    """filter_main_track_papers() is the STRICT pre-download gate for section-blind
    ACM venues: keep ONLY confirmed-main DOIs, drop non-main AND unclassifiable."""

    @staticmethod
    def _paper(doi: str):
        from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper
        return CorpusPaper(
            paper_id=f"ppopp-2024-{doi.split('/')[-1]}",
            title="t", authors=[], conference="PPoPP", year=2024,
            pdf_url=f"https://dl.acm.org/doi/pdf/{doi}",
            paper_page_url=f"https://doi.org/{doi}", source_name="dblp",
        )

    def test_keeps_only_main_drops_nonmain_and_unclassified(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import filter_main_track_papers
        papers = [self._paper(f"10.1145/1.{i}") for i in range(1, 6)]
        doi_to_track = {
            "10.1145/1.1": "research",   # keep
            "10.1145/1.2": "short",      # keep
            "10.1145/1.3": "poster",     # drop
            "10.1145/1.4": "workshop",   # drop
            # 10.1145/1.5 absent → unclassified → drop
        }
        kept, dropped = filter_main_track_papers(papers, doi_to_track)
        kept_dois = {p.pdf_url.split("/pdf/")[-1] for p in kept}
        self.assertEqual(kept_dois, {"10.1145/1.1", "10.1145/1.2"})
        self.assertEqual(dropped, {"poster": 1, "workshop": 1, "unclassified": 1})

    def test_industry_track_is_main(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import filter_main_track_papers
        kept, dropped = filter_main_track_papers(
            [self._paper("10.1145/2.1")], {"10.1145/2.1": "industry"})
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, {})

    def test_doi_match_is_case_insensitive(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import filter_main_track_papers
        # collect_papers may emit a mixed-case DOI in pdf_url; the map is lower-cased.
        kept, _ = filter_main_track_papers(
            [self._paper("10.1145/AbC.DeF")], {"10.1145/abc.def": "research"})
        self.assertEqual(len(kept), 1)

    def test_empty_track_map_drops_all(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import filter_main_track_papers
        kept, dropped = filter_main_track_papers(
            [self._paper("10.1145/3.1"), self._paper("10.1145/3.2")], {})
        self.assertEqual(kept, [])
        self.assertEqual(dropped, {"unclassified": 2})


if __name__ == "__main__":
    unittest.main()
