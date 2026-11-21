"""Real-network acceptance tests for the 2026 KDD, SIGIR, and EMNLP sources.

These tests deliberately use the configured local proxy and the live official
conference pages.  They do not replace HTTP, browser, filesystem, or source
objects with mocks.  The browser download is opt-in because it launches a real
Chrome process and may spend several minutes obtaining a fresh Cloudflare
session.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import os
from pathlib import Path
import shutil
import socket

import pytest
from pypdf import PdfReader

from pipeline.medical_corpus.cf_bypass import (
    minimal_backend_available,
    turnstile_available,
)
from pipeline.download_control import ChainedProxyPool
from pipeline.paper_crawler import ccf_a_main_corpus as corpus
from scripts.collect.collect_and_download_acm import (
    CONFERENCE_CONFIG,
    download_acm_pdfs,
    select_acm_manifest_records,
)
from scripts.collect.collect_and_download_emnlp import (
    collect_emnlp_2026_openreview_notes,
)
from scripts.collect.collect_2026_to_manifest import (
    announced_prior_year_replay,
    maybe_resolve_open_access,
)


PROXY_URL = "http://127.0.0.1:10808"
FIRST_HOP_PROXY_URL = "socks5://127.0.0.1:10808"
KDD_TRACK_COUNTS = {
    "research": 785,
    "applied_data_science": 186,
    "datasets_and_benchmarks": 182,
    "ai_for_sciences": 238,
    "blue_sky_ideas": 17,
}
SIGIR_TRACK_COUNTS = {
    "research": 233,
    "short": 149,
    "industry": 129,
    "resource": 61,
    "reproducibility": 28,
    "low_resource": 14,
    "perspective": 12,
}


def _require_proxy() -> None:
    with socket.create_connection(("127.0.0.1", 10808), timeout=2):
        return


def _runtime_pool_path() -> Path:
    configured = os.environ.get("RETRIVAL_PROXY_POOL_FILE")
    return Path(configured) if configured else Path.home() / "Desktop" / "a.txt"


def _runtime_xray_path() -> Path:
    configured = os.environ.get("RETRIVAL_XRAY_EXECUTABLE")
    if configured:
        return Path(configured)
    discovered = shutil.which("xray")
    if discovered:
        return Path(discovered)
    return Path(r"C:\Arc\zz_v2rayN-With-Core-SelfContained\bin\xray\xray.exe")


@pytest.fixture(scope="module")
def official_client():
    _require_proxy()
    client = corpus.OfficialProceedingsClient(proxy_url=PROXY_URL, timeout_seconds=90)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="module")
def kdd_2026_papers(official_client):
    source_cls = getattr(corpus, "KddOfficialProceedingsSource", None)
    assert source_cls is not None, "KDD 2026 must use its official two-cycle paper list"
    return source_cls().collect_papers(2026, official_client)


@pytest.fixture(scope="module")
def emnlp_2026_candidates(official_client):
    source = corpus.EmnlpMainConferenceSource()
    assert hasattr(source, "collect_announced_papers")
    return source.collect_announced_papers(2026, official_client)


@pytest.fixture(scope="module")
def sigir_2026_papers(official_client):
    source_cls = getattr(corpus, "SigirOfficialProgramSource", None)
    assert source_cls is not None, "SIGIR 2026 must use the official final program"
    return source_cls().collect_papers(2026, official_client)


def test_registry_enables_both_2026_sources():
    by_name = {row["name"]: row for row in corpus.load_conference_registry()}
    assert 2026 in by_name["KDD"]["years"]
    assert 2026 in by_name["EMNLP"]["years"]
    assert isinstance(
        by_name["KDD"]["source"],
        getattr(corpus, "KddOfficialProceedingsSource", object),
    )


def test_kdd_2026_live_official_list_is_complete(kdd_2026_papers):
    # The official page currently renders 1,415 rows, including seven exact
    # duplicate DOI rows. Canonical paper_id/DOI identity yields 1,408 papers.
    assert len(kdd_2026_papers) == 1408
    assert len({paper.paper_id for paper in kdd_2026_papers}) == 1408
    assert Counter(paper.venue_track for paper in kdd_2026_papers) == KDD_TRACK_COUNTS
    assert all(paper.conference == "KDD" and paper.year == 2026 for paper in kdd_2026_papers)
    assert all(paper.source_name == "kdd_official_program" for paper in kdd_2026_papers)
    assert all(
        paper.pdf_url.startswith("https://dl.acm.org/doi/pdf/10.1145/")
        for paper in kdd_2026_papers
    )
    assert all(
        paper.paper_page_url.startswith("https://doi.org/10.1145/")
        for paper in kdd_2026_papers
    )


def test_kdd_acm_cli_uses_official_2026_source():
    source_cls = getattr(corpus, "KddOfficialProceedingsSource", None)
    assert source_cls is not None
    assert CONFERENCE_CONFIG["KDD"]["source_class"] is source_cls


def test_sigir_2026_registry_and_acm_cli_use_the_official_program_source():
    source_cls = getattr(corpus, "SigirOfficialProgramSource", None)
    assert source_cls is not None
    by_name = {row["name"]: row for row in corpus.load_conference_registry()}
    assert 2026 in by_name["SIGIR"]["years"]
    assert isinstance(by_name["SIGIR"]["source"], source_cls)
    assert CONFERENCE_CONFIG["SIGIR"]["source_class"] is source_cls


def test_sigir_2026_live_final_program_is_main_conference_only(sigir_2026_papers):
    assert len(sigir_2026_papers) == 626
    assert len({paper.paper_id for paper in sigir_2026_papers}) == 626
    assert Counter(paper.venue_track for paper in sigir_2026_papers) == SIGIR_TRACK_COUNTS
    assert all(paper.conference == "SIGIR" and paper.year == 2026 for paper in sigir_2026_papers)
    assert all(paper.source_name == "sigir_official_program" for paper in sigir_2026_papers)
    assert all(paper.title and paper.authors for paper in sigir_2026_papers)
    # The final program itself omits abstracts for exactly these two Full papers.
    # Preserve that source truth instead of fabricating text during collection.
    assert {
        paper.title
        for paper in sigir_2026_papers
        if not paper.abstract
    } == {
        "Minimal-Perturbation Counterfactuals through Guided Denoising Diffusion for Recommender Systems Explanation",
        "MCP-Focus: Leveraging Function-Oriented Document Enhancement for MCP Server Retrieval",
    }
    assert all(
        paper.pdf_url.startswith("https://dl.acm.org/doi/pdf/10.1145/3805712.")
        for paper in sigir_2026_papers
    )
    assert all(
        paper.venue_track not in {"demo", "doctoral", "poster", "workshop", "tois"}
        for paper in sigir_2026_papers
    )


def test_kdd_manifest_collection_keeps_acm_pdf_urls_by_default(
    kdd_2026_papers,
    official_client,
):
    source = corpus.KddOfficialProceedingsSource()
    before = [paper.pdf_url for paper in kdd_2026_papers]
    resolution = maybe_resolve_open_access(
        source,
        kdd_2026_papers,
        official_client,
        enabled=False,
    )
    assert resolution is None
    assert [paper.pdf_url for paper in kdd_2026_papers] == before
    assert all(url.startswith("https://dl.acm.org/doi/pdf/") for url in before)


def test_kdd_download_manifest_selector_is_limited_to_2026():
    builder = corpus.MainConferenceCorpusBuilder(proxy_url=PROXY_URL)
    records = corpus.load_json_with_recovery(builder.manifest_path, default_value=[])
    selected = select_acm_manifest_records(records, {"KDD"}, year=2026)
    assert len(selected) == 1408
    assert all(row.get("conference") == "KDD" and row.get("year") == 2026 for row in selected)


def test_emnlp_2026_live_announced_page_is_rejected_as_a_2025_replay(
    emnlp_2026_candidates,
):
    assert len(emnlp_2026_candidates) == 1809
    assert len({row.candidate_id for row in emnlp_2026_candidates}) == 1809
    builder = corpus.MainConferenceCorpusBuilder(proxy_url=PROXY_URL)
    manifest = corpus.load_json_with_recovery(builder.manifest_path, default_value=[])
    replay = announced_prior_year_replay(
        emnlp_2026_candidates,
        manifest,
        conference="EMNLP",
        year=2026,
    )
    assert replay["prior_year_count"] == 1809
    assert replay["overlap_count"] >= 1780
    assert replay["overlap_ratio"] >= 0.98


def test_emnlp_2026_openreview_main_decision_mapping_is_not_public_yet(
    emnlp_2026_candidates,
):
    pool_path = _runtime_pool_path()
    xray_path = _runtime_xray_path()
    if not pool_path.is_file():
        pytest.skip(f"real proxy pool is not present: {pool_path}")
    if not xray_path.is_file():
        pytest.skip(f"real Xray executable is not present: {xray_path}")

    pool = ChainedProxyPool.from_file(
        pool_path,
        first_hop_proxy=FIRST_HOP_PROXY_URL,
    )
    notes = asyncio.run(
        collect_emnlp_2026_openreview_notes(
            chained_proxy_pool=pool,
            xray_executable=xray_path,
            challenge_retries=6,
        )
    )

    # On 2026-09-14 the public venue config exposes this exact decision label,
    # but the corresponding notes query is still empty.  Once OpenReview makes
    # it public this sentinel intentionally fails and the downloader can switch
    # from source-waiting to exact accepted-main collection.
    assert notes == []


def test_selected_turnstile_fork_has_a_real_browser_backend():
    assert turnstile_available()
    assert minimal_backend_available()


def test_camoufox_runtime_can_access_a_closed_shadow_root():
    async def run_browser() -> str:
        from camoufox.async_api import AsyncCamoufox

        async with AsyncCamoufox(
            headless=True,
            config={"forceScopeAccess": True},
        ) as browser:
            page = await browser.new_page()
            await page.set_content(
                """
                <div id="host"></div>
                <script>
                  const host = document.querySelector('#host');
                  document.documentElement.dataset.shadowUnl = typeof host.shadowRootUnl;
                  const root = host.attachShadow({mode: 'closed'});
                  root.innerHTML = '<button id="inside">reachable</button>';
                </script>
                """
            )
            inside = await asyncio.wait_for(
                page.evaluate(
                    "document.querySelector('#host').shadowRootUnl"
                    ".querySelector('#inside').textContent"
                ),
                timeout=5,
            )
            page_visible = await page.get_attribute("html", "data-shadow-unl")
            window_utils = await page.evaluate("typeof window.windowUtils")
            juggler_mouse = await page.evaluate(
                "typeof window.windowUtils?.jugglerSendMouseEvent"
            )
            return inside, page_visible, window_utils, juggler_mouse

    assert asyncio.run(run_browser()) == (
        "reachable",
        "undefined",
        "object",
        "function",
    )


def test_kdd_2026_real_acm_download_uses_turnstile_solver(kdd_2026_papers, tmp_path: Path):
    pool_path = _runtime_pool_path()
    xray_path = _runtime_xray_path()
    if not pool_path.is_file():
        pytest.skip(f"real proxy pool is not present: {pool_path}")
    if not xray_path.is_file():
        pytest.skip(f"real Xray executable is not present: {xray_path}")

    paper = kdd_2026_papers[0]
    publisher_url = paper.pdf_url
    assert publisher_url.startswith("https://dl.acm.org/doi/pdf/")
    pool = ChainedProxyPool.from_file(
        pool_path,
        first_hop_proxy=FIRST_HOP_PROXY_URL,
    )

    result = asyncio.run(
        download_acm_pdfs(
            [paper],
            tmp_path / "pdf",
            PROXY_URL,
            batch_size=1,
            delay_between=0,
            limit=1,
            challenge_retries=6,
            require_solver=True,
            chained_proxy_pool=pool,
            xray_executable=xray_path,
        )
    )
    downloaded = Path(result[0].pdf_path or "")
    assert downloaded.is_file()
    assert downloaded.read_bytes()[:5] == b"%PDF-"
    assert len(PdfReader(str(downloaded)).pages) > 0
    assert result[0].downloaded_from_url == publisher_url
    assert result[0].download_error is None


def test_sigir_2026_real_acm_download_uses_two_hop_proxy(
    sigir_2026_papers,
    tmp_path: Path,
):
    pool_path = _runtime_pool_path()
    xray_path = _runtime_xray_path()
    if not pool_path.is_file():
        pytest.skip(f"real proxy pool is not present: {pool_path}")
    if not xray_path.is_file():
        pytest.skip(f"real Xray executable is not present: {xray_path}")

    paper = sigir_2026_papers[0]
    publisher_url = paper.pdf_url
    assert publisher_url.startswith("https://dl.acm.org/doi/pdf/10.1145/3805712.")
    pool = ChainedProxyPool.from_file(
        pool_path,
        first_hop_proxy=FIRST_HOP_PROXY_URL,
    )

    result = asyncio.run(
        download_acm_pdfs(
            [paper],
            tmp_path / "pdf",
            PROXY_URL,
            batch_size=1,
            delay_between=0,
            limit=1,
            challenge_retries=6,
            require_solver=True,
            chained_proxy_pool=pool,
            xray_executable=xray_path,
        )
    )
    downloaded = Path(result[0].pdf_path or "")
    assert downloaded.is_file()
    assert downloaded.read_bytes()[:5] == b"%PDF-"
    assert len(PdfReader(str(downloaded)).pages) > 0
    assert result[0].downloaded_from_url == publisher_url
    assert result[0].download_error is None
