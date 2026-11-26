"""TDD tests for 2025 conference source configuration."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    AaaiTechnicalTracksSource,
    AclMainConferenceSource,
    CvfProceedingsSource,
    IclrOpenReviewSource,
    IcmlProceedingsSource,
    IjcaiProceedingsSource,
    NeuripsProceedingsSource,
    DEFAULT_TARGET_CONFERENCES,
    DEFAULT_TARGET_YEARS,
    ICML_VOLUME_BY_YEAR,
    OPENREVIEW_COOKIE_FILE_ENV,
    build_sources,
    extract_inline_text,
    http_retry_delay_seconds,
    load_openreview_cookies_from_env,
)


class TestTargetYearsConfig(unittest.TestCase):
    def test_2025_in_default_target_years(self):
        self.assertIn(2025, DEFAULT_TARGET_YEARS)

    def test_target_years_sorted(self):
        self.assertEqual(list(DEFAULT_TARGET_YEARS), sorted(DEFAULT_TARGET_YEARS))


class TestAcl2026MainSource(unittest.TestCase):
    def test_registry_enables_acl_2026(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry

        acl = next(row for row in load_conference_registry() if row["name"] == "ACL")
        self.assertIn(2026, acl["years"])

    def test_fixed_case_markup_does_not_split_words(self):
        """ACL wraps protected capitals inside a word in inline spans.

        ``get_text(" ")`` turns ``OctoTools`` into ``O cto T ools``.  The
        production collector must preserve inline word boundaries while still
        collapsing layout whitespace between actual words.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            """
            <a class="align-middle" href="/2026.acl-long.1/">
              <span class="acl-fixed-case">O</span>cto<span class="acl-fixed-case">T</span>ools:
              A <em>Multi-Agent</em> Framework
            </a>
            """,
            "html.parser",
        )

        self.assertEqual(
            extract_inline_text(soup.select_one("a.align-middle")),
            "OctoTools: A Multi-Agent Framework",
        )


class TestIcmlVolumeMapping(unittest.TestCase):
    def test_2025_volume_exists(self):
        self.assertIn(2025, ICML_VOLUME_BY_YEAR)

    def test_2025_volume_is_v267(self):
        self.assertEqual(ICML_VOLUME_BY_YEAR[2025], "v267")

    def test_icml_supports_2025(self):
        source = IcmlProceedingsSource()
        self.assertTrue(source.supports_year(2025))

    def test_icml_supports_2026_via_openreview(self):
        source = IcmlProceedingsSource()
        self.assertTrue(source.supports_year(2026))
        self.assertIn(2026, source.VIRTUAL_SCHEDULE_URL_BY_YEAR)

    def test_icml_2026_collects_virtual_schedule_records(self):
        source = IcmlProceedingsSource()

        class FakeClient:
            def __init__(self):
                self.urls = []

            def get_soup(self, url):
                from bs4 import BeautifulSoup

                self.urls.append(url)
                if url.endswith("/papers.html"):
                    return BeautifulSoup(
                        """
                        <ul>
                          <li><a href="/virtual/2026/poster/1">Joint-Embedding Predictive Learning</a></li>
                          <li><a href="/virtual/2026/poster/2">Duplicate Session Listing</a></li>
                        </ul>
                        """,
                        "html.parser",
                    )
                return BeautifulSoup(
                    """
                    <div class="showDetailBtn" data-event-id="1">
                      <div class="maincard narrower poster" id="maincard_1">
                        <div class="maincardBody">Joint-Embedding Predictive Learning</div>
                        <div class="maincardFooter">A. Author ⋅ B. Writer</div>
                        <a href="https://openreview.net/forum?id=BZfkxSasd3">OpenReview</a>
                      </div>
                    </div>
                    <div class="showDetailBtn" data-event-id="2">
                      <div class="maincard narrower poster" id="maincard_2">
                        <div class="maincardBody">Duplicate Session Listing</div>
                        <div class="maincardFooter">Ignored Author</div>
                        <a href="https://openreview.net/forum?id=BZfkxSasd3">OpenReview</a>
                      </div>
                    </div>
                    """,
                    "html.parser",
                )

        fake_client = FakeClient()
        papers = source.collect_papers(2026, fake_client)

        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.paper_id, "icml-2026-BZfkxSasd3")
        self.assertEqual(paper.conference, "ICML")
        self.assertEqual(paper.year, 2026)
        self.assertEqual(paper.pdf_url, "https://openreview.net/pdf?id=BZfkxSasd3")
        self.assertEqual(paper.paper_page_url, "https://openreview.net/forum?id=BZfkxSasd3")
        self.assertEqual(paper.source_name, "icml_virtual_schedule")
        self.assertEqual(paper.authors, ["A. Author", "B. Writer"])
        self.assertEqual(paper.abstract, "")
        self.assertEqual(
            fake_client.urls,
            [
                "https://icml.cc/Conferences/2026/Schedule",
                "https://icml.cc/virtual/2026/papers.html",
            ],
        )

    def test_icml_2026_virtual_schedule_keeps_only_papers_page_posters(self):
        source = IcmlProceedingsSource()

        class FakeClient:
            def __init__(self):
                self.urls = []

            def get_soup(self, url):
                from bs4 import BeautifulSoup

                self.urls.append(url)
                if url.endswith("/papers.html"):
                    return BeautifulSoup(
                        """
                        <ul>
                          <li><a href="/virtual/2026/poster/111">Main Conference Paper</a></li>
                        </ul>
                        """,
                        "html.parser",
                    )
                return BeautifulSoup(
                    """
                    <div class="showDetailBtn" data-event-id="111">
                      <div class="maincard narrower poster" id="maincard_111">
                        <div class="maincardBody">Main Conference Paper</div>
                        <div class="maincardFooter">A. Main \\u22c5 B. Main</div>
                        <a href="https://openreview.net/forum?id=MAIN123">OpenReview</a>
                      </div>
                    </div>
                    <div class="showDetailBtn" data-event-id="222">
                      <div class="maincard narrower workshop" id="maincard_222">
                        <div class="maincardBody">Workshop Paper</div>
                        <div class="maincardFooter">A. Workshop</div>
                        <a href="https://openreview.net/forum?id=WORKSHOP456">OpenReview</a>
                      </div>
                    </div>
                    """,
                    "html.parser",
                )

        fake_client = FakeClient()
        papers = source.collect_papers(2026, fake_client)

        self.assertEqual([paper.paper_id for paper in papers], ["icml-2026-MAIN123"])
        self.assertEqual(
            fake_client.urls,
            [
                "https://icml.cc/Conferences/2026/Schedule",
                "https://icml.cc/virtual/2026/papers.html",
            ],
        )

    def test_icml_openreview_challenge_error_mentions_cookie_file(self):
        source = IcmlProceedingsSource()

        class FakeClient:
            def get_text(self, url):
                return json.dumps({"name": "ChallengeRequiredError", "message": "challenge"})

        with self.assertRaisesRegex(RuntimeError, OPENREVIEW_COOKIE_FILE_ENV):
            source._paginate_openreview_notes(source.OPENREVIEW_YEAR_CONFIG[2026], FakeClient())


class TestOpenReviewCookieFile(unittest.TestCase):
    def test_load_cdp_export_cookies_from_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / "cookies.json"
            cookie_path.write_text(
                json.dumps(
                    [
                        {
                            "domain": ".openreview.net",
                            "name": "openreview.clearanceToken",
                            "value": "secret-clearance",
                            "path": "/",
                        },
                        {
                            "domain": "example.com",
                            "name": "ignored",
                            "value": "not-openreview",
                            "path": "/",
                        },
                    ]
                    ,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {OPENREVIEW_COOKIE_FILE_ENV: str(cookie_path)}):
                cookies = load_openreview_cookies_from_env()

        self.assertIsNotNone(cookies)
        jar_items = {(cookie.domain, cookie.name, cookie.value) for cookie in cookies.jar}
        self.assertIn((".openreview.net", "openreview.clearanceToken", "secret-clearance"), jar_items)
        self.assertNotIn(("example.com", "ignored", "not-openreview"), jar_items)


class TestIcmlDownloadHelper(unittest.TestCase):
    def test_load_missing_papers_does_not_inspect_existing_pdf_content(self):
        from scripts.collect import download_icml2026_cdp as helper

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "paper_manifest.json"
            pdf_root = root / "pdf"
            existing_pdf = pdf_root / "icml" / "2026" / "icml-2026-EXISTS.pdf"
            existing_pdf.parent.mkdir(parents=True)
            existing_pdf.write_bytes(b"already renamed artifact; content is not read on resume\n")
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "paper_id": "icml-2026-EXISTS",
                            "title": "Already Downloaded",
                            "authors": [],
                            "conference": "ICML",
                            "year": 2026,
                            "pdf_url": "https://openreview.net/pdf?id=EXISTS",
                            "paper_page_url": "https://openreview.net/forum?id=EXISTS",
                            "source_name": "icml_virtual_schedule",
                        },
                        {
                            "paper_id": "icml-2026-MISSING",
                            "title": "Needs Download",
                            "authors": [],
                            "conference": "ICML",
                            "year": 2026,
                            "pdf_url": "https://openreview.net/pdf?id=MISSING",
                            "paper_page_url": "https://openreview.net/forum?id=MISSING",
                            "source_name": "icml_virtual_schedule",
                        },
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            builder = type("FakeBuilder", (), {"manifest_path": manifest_path, "pdf_root": pdf_root})()

            with mock.patch(
                "pipeline.paper_crawler.runtime_files.PdfReader",
                side_effect=AssertionError("strict PDF validation should not run here"),
            ):
                missing = helper.load_missing_papers(builder, limit=0)

        self.assertEqual([paper.paper_id for paper in missing], ["icml-2026-MISSING"])

    def test_refresh_usable_clearance_retries_cdp_timeout(self):
        from scripts.collect import download_icml2026_cdp as helper

        with mock.patch.object(
            helper,
            "refresh_clearance",
            side_effect=[helper.websocket.WebSocketTimeoutException("timed out"), 75.0],
        ) as refresh, mock.patch.object(helper.time, "sleep") as sleep:
            seconds_left = helper.refresh_usable_clearance(
                "http://127.0.0.1:9222",
                "FORUM",
                safety_seconds=30.0,
                min_interval_seconds=10.0,
                settle_seconds=6.0,
                retry_seconds=5.0,
            )

        self.assertEqual(seconds_left, 75.0)
        self.assertEqual(refresh.call_count, 2)
        sleep.assert_called_once_with(5.0)


class TestHttpRetryDelay(unittest.TestCase):
    def test_429_without_retry_after_cools_down(self):
        import httpx

        request = httpx.Request("GET", "https://openreview.net/pdf?id=x")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("too many", request=request, response=response)

        self.assertEqual(http_retry_delay_seconds(exc, attempt=1), 60.0)

    def test_429_retry_after_respects_larger_delay(self):
        import httpx

        request = httpx.Request("GET", "https://openreview.net/pdf?id=x")
        response = httpx.Response(429, headers={"retry-after": "120"}, request=request)
        exc = httpx.HTTPStatusError("too many", request=request, response=response)

        self.assertEqual(http_retry_delay_seconds(exc, attempt=1), 120.0)


class TestAaaiScanWindows(unittest.TestCase):
    def test_2025_scan_window_exists(self):
        source = AaaiTechnicalTracksSource()
        self.assertIn(2025, source.issue_scan_windows)

    def test_2025_scan_window_covers_known_ids(self):
        source = AaaiTechnicalTracksSource()
        window = source.issue_scan_windows[2025]
        self.assertIn(624, window)
        self.assertIn(652, window)

    def test_2025_scan_window_reasonable_size(self):
        source = AaaiTechnicalTracksSource()
        window = list(source.issue_scan_windows[2025])
        self.assertGreater(len(window), 20)
        self.assertLess(len(window), 50)


class TestCvfIccv2025(unittest.TestCase):
    def test_iccv_supports_2025(self):
        source = CvfProceedingsSource("ICCV")
        self.assertTrue(source.supports_year(2025))

    def test_iccv_rejects_2024(self):
        source = CvfProceedingsSource("ICCV")
        self.assertFalse(source.supports_year(2024))

    def test_cvpr_supports_2025(self):
        source = CvfProceedingsSource("CVPR")
        self.assertTrue(source.supports_year(2025))


class TestIclrOpenReviewSource(unittest.TestCase):
    def test_iclr_in_default_conferences(self):
        self.assertIn("ICLR", DEFAULT_TARGET_CONFERENCES)

    def test_iclr_in_build_sources(self):
        sources = build_sources()
        self.assertIn("ICLR", sources)
        self.assertIsInstance(sources["ICLR"], IclrOpenReviewSource)

    def test_supports_2022_to_2025(self):
        source = IclrOpenReviewSource()
        for year in [2022, 2023, 2024, 2025]:
            with self.subTest(year=year):
                self.assertTrue(source.supports_year(year))

    def test_rejects_2021(self):
        source = IclrOpenReviewSource()
        self.assertFalse(source.supports_year(2021))

    def test_year_config_has_all_supported_years(self):
        source = IclrOpenReviewSource()
        for year in [2022, 2023, 2024, 2025]:
            self.assertIn(year, source.YEAR_CONFIG)

    def test_v1_years_have_accepted_venues(self):
        source = IclrOpenReviewSource()
        for year in [2022, 2023]:
            config = source.YEAR_CONFIG[year]
            self.assertEqual(config["api"], "v1")
            self.assertIn("accepted_venues", config)
            self.assertGreater(len(config["accepted_venues"]), 0)

    def test_v2_years_have_venueid(self):
        source = IclrOpenReviewSource()
        for year in [2024, 2025]:
            config = source.YEAR_CONFIG[year]
            self.assertEqual(config["api"], "v2")
            self.assertIn("venueid", config)

    def test_extract_field_v2_format(self):
        content = {"title": {"value": "Test Paper"}, "authors": {"value": ["A", "B"]}}
        self.assertEqual(IclrOpenReviewSource._extract_field(content, "title"), "Test Paper")
        self.assertEqual(IclrOpenReviewSource._extract_field(content, "authors", []), ["A", "B"])

    def test_extract_field_v1_format(self):
        content = {"title": "Test Paper", "authors": ["A", "B"]}
        self.assertEqual(IclrOpenReviewSource._extract_field(content, "title"), "Test Paper")
        self.assertEqual(IclrOpenReviewSource._extract_field(content, "authors", []), ["A", "B"])

    def test_extract_field_missing_key(self):
        self.assertEqual(IclrOpenReviewSource._extract_field({}, "title"), "")
        self.assertEqual(IclrOpenReviewSource._extract_field({}, "authors", []), [])


# conf/conferences.yaml lists ECCV with years [2022, 2024, 2026] ("Even years only").
EVEN_YEARS_ONLY = {"ECCV"}


class TestAllSourcesSupport2025(unittest.TestCase):
    def test_all_eight_conferences_support_2025(self):
        sources = build_sources()
        for name, source in sources.items():
            with self.subTest(conference=name):
                if name in EVEN_YEARS_ONLY:
                    self.assertFalse(
                        source.supports_year(2025),
                        f"{name} is held in even years only and has no 2025 edition",
                    )
                    continue
                self.assertTrue(
                    source.supports_year(2025),
                    f"{name} must support year 2025",
                )

    def test_core_conferences_registered(self):
        # The original 12 AI/DB venues remain registered. The registry has since grown
        # (SC/CG/MX/DS expansions) — test_conference_registry.py owns the authoritative
        # full count; here we only assert the core set survives future additions.
        sources = build_sources()
        core = {"AAAI", "ACL", "CVPR", "EMNLP", "ICCV", "ICLR", "ICML", "IJCAI", "KDD", "NeurIPS", "SIGIR", "SIGMOD"}
        missing = core - set(sources.keys())
        self.assertEqual(missing, set(), f"core venues missing from registry: {missing}")
        self.assertGreaterEqual(len(sources), 18)


if __name__ == "__main__":
    unittest.main()
