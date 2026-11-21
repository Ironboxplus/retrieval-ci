"""TDD for robotics corpus preparation (ICRA/CoRL/RSS, 2022–2026 full scope)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.robotics_corpus.models import RoboticsPaperRecord, RoboticsVenue, sanitize_id_component
from pipeline.robotics_corpus.pdf_fetch import download_pdf, plan_pdf_fetch
from pipeline.robotics_corpus.proxy import RoboticsProxyConfig
from pipeline.robotics_corpus.sources import (
    CORL_PMLR_VOLUME_BY_YEAR,
    DEFAULT_TARGET_YEARS,
    DEFAULT_VENUES,
    extract_ieee_arnumber,
    rss_slug_for_year,
    collect_corl_year,
    collect_rss_year,
    collect_icra_year,
)
from pipeline.robotics_corpus.prepare import PrepareConfig, collect_manifest, run_download_pass


def test_default_scope_is_full_2022_2026_all_three_venues():
    """User requirement: bulk window is 2022–2026 for ICRA+CoRL+RSS, not a sample year."""
    assert DEFAULT_TARGET_YEARS == (2022, 2023, 2024, 2025, 2026)
    assert {v.value for v in DEFAULT_VENUES} == {"ICRA", "CoRL", "RSS"}
    cfg = PrepareConfig()
    assert tuple(cfg.years) == DEFAULT_TARGET_YEARS
    assert [v.value for v in cfg.resolved_venues()] == ["ICRA", "CoRL", "RSS"]
    assert cfg.sample_limit_per_venue_year == 0
    assert cfg.download_limit == 0


def test_rss_slug_maps_2022_to_2025():
    assert rss_slug_for_year(2022) == "rss18"
    assert rss_slug_for_year(2023) == "rss19"
    assert rss_slug_for_year(2024) == "rss20"
    assert rss_slug_for_year(2025) == "rss21"
    assert rss_slug_for_year(2026) == "rss22"


def test_corl_pmlr_volumes_cover_2022_2025():
    for year in (2022, 2023, 2024, 2025):
        assert year in CORL_PMLR_VOLUME_BY_YEAR


def test_ieee_arnumber_from_doi():
    assert extract_ieee_arnumber("https://doi.org/10.1109/ICRA57147.2024.10611355") == "10611355"
    assert extract_ieee_arnumber("10.1109/ICRA57147.2024.10611355") == "10611355"
    assert extract_ieee_arnumber("https://ieeexplore.ieee.org/document/10611355/") == "10611355"
    assert extract_ieee_arnumber("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10611355") == "10611355"


def test_sanitize_id():
    assert sanitize_id_component("a/b c") == "a_b_c"


def test_plan_pdf_fetch_dry(tmp_path: Path):
    rec = RoboticsPaperRecord(
        paper_id="rss-2024-p001",
        title="Stein Variational Ergodic Search",
        venue=RoboticsVenue.RSS,
        year=2024,
        pdf_url="https://www.roboticsproceedings.org/rss20/p001.pdf",
    )
    plan = plan_pdf_fetch(rec, pdf_root=tmp_path / "pdf")
    assert plan.action == "would_download"
    assert plan.dest_path.as_posix().endswith("rss/2024/rss-2024-p001.pdf")


def test_download_pdf_dry_run_no_network(tmp_path: Path):
    rec = RoboticsPaperRecord(
        paper_id="corl-2024-x",
        title="t",
        venue=RoboticsVenue.CORL,
        year=2024,
        pdf_url="https://proceedings.mlr.press/v270/x.pdf",
    )
    with patch("pipeline.robotics_corpus.pdf_fetch.httpx.Client") as client_cls:
        out = download_pdf(
            rec,
            pdf_root=tmp_path / "pdf",
            proxy=RoboticsProxyConfig(),
            execute=False,
        )
        client_cls.assert_not_called()
    assert out.download_status == "planned"


def test_download_pdf_execute_success(tmp_path: Path):
    rec = RoboticsPaperRecord(
        paper_id="rss-2024-p001",
        title="t",
        venue=RoboticsVenue.RSS,
        year=2024,
        pdf_url="https://www.roboticsproceedings.org/rss20/p001.pdf",
    )
    pdf_bytes = b"%PDF-1.4\n" + b"0" * 2000

    class FakeResp:
        status_code = 200
        content = pdf_bytes
        headers = {"content-type": "application/pdf"}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return FakeResp()

    with patch("pipeline.robotics_corpus.pdf_fetch.httpx.Client", FakeClient):
        out = download_pdf(
            rec,
            pdf_root=tmp_path / "pdf",
            proxy=RoboticsProxyConfig(),
            execute=True,
            allow_cf_bypass=False,
        )
    assert out.download_status == "downloaded"
    assert out.pdf_path and Path(out.pdf_path).exists()
    assert Path(out.pdf_path).read_bytes().startswith(b"%PDF")


def test_collect_rss_parses_html():
    html = """
    <html><body>
    <a href="p001.html">Paper One</a><a href="p001.pdf"></a>
    <a href="p002.html">Paper Two</a><a href="p002.pdf"></a>
    <a href="../index.html">home</a>
    </body></html>
    """
    client = MagicMock()
    from bs4 import BeautifulSoup
    client.get_soup.return_value = BeautifulSoup(html, "html.parser")
    papers = collect_rss_year(2024, client)
    assert len(papers) == 2
    assert papers[0].venue == RoboticsVenue.RSS
    assert papers[0].year == 2024
    assert papers[0].pdf_url.endswith("p001.pdf") or papers[1].pdf_url.endswith("p001.pdf")
    assert all(p.paper_id.startswith("rss-2024-") for p in papers)


def test_collect_corl_parses_pmlr_html():
    html = """
    <html><body>
    <div class="paper">
      <p class="title">Equivariant Diffusion Policy</p>
      <span class="authors">A Author, B Author</span>
      <a href="https://proceedings.mlr.press/v270/wang25a.html">abs</a>
      <a href="https://raw.githubusercontent.com/mlresearch/v270/main/assets/wang25a/wang25a.pdf">Download PDF</a>
      <a href="https://openreview.net/forum?id=abc123">OpenReview</a>
    </div>
    </body></html>
    """
    client = MagicMock()
    from bs4 import BeautifulSoup
    client.get_soup.return_value = BeautifulSoup(html, "html.parser")
    papers = collect_corl_year(2024, client)
    assert len(papers) == 1
    p = papers[0]
    assert p.venue == RoboticsVenue.CORL
    assert p.year == 2024
    assert p.pdf_url.endswith(".pdf")
    assert any("openreview.net/pdf" in u for u in p.alternate_pdf_urls)


def test_collect_icra_prefers_html_then_json_fallback():
    payload = {
        "result": {
            "hits": {
                "@total": "1",
                "hit": [{
                    "info": {
                        "title": "TinyMPC: Model-Predictive Control.",
                        "key": "conf/icra/Smith24",
                        "ee": "https://doi.org/10.1109/ICRA57147.2024.10610987",
                        "authors": {"author": [{"text": "Jane Smith"}]},
                    }
                }],
            }
        }
    }
    client = MagicMock()
    client.get_soup.side_effect = RuntimeError("html unavailable")
    client.get_json.return_value = payload
    papers = collect_icra_year(2024, client)
    assert len(papers) == 1
    p = papers[0]
    assert p.venue == RoboticsVenue.ICRA
    assert p.year == 2024
    assert p.doi and "10.1109" in p.doi
    assert p.pdf_url and ("stampPDF" in p.pdf_url or "arxiv" in p.pdf_url)
    assert p.raw.get("arnumber") == "10610987"


def test_collect_icra_html_toc():
    html = """
    <html><body>
    <ul class="publ-list">
      <li class="entry article" id="x">
        <span class="title">TinyMPC: Model-Predictive Control.</span>
        <span itemprop="author"><span itemprop="name">Jane Smith</span></span>
        <a href="https://doi.org/10.1109/ICRA57147.2024.10610987">ee</a>
        <a href="https://dblp.org/rec/conf/icra/Smith24">view</a>
      </li>
    </ul>
    </body></html>
    """
    client = MagicMock()
    from bs4 import BeautifulSoup
    from pipeline.robotics_corpus.sources import collect_icra_year_from_html
    client.get_soup.return_value = BeautifulSoup(html, "html.parser")
    papers = collect_icra_year_from_html(2024, client)
    assert len(papers) == 1
    assert papers[0].raw.get("arnumber") == "10610987"


def test_collect_manifest_full_scope_mocked(tmp_path: Path):
    """collect walks every venue×year cell; missing cell is skipped, not fatal."""
    def fake_collect(venue, year, client, *, max_papers: int = 0):
        if venue == RoboticsVenue.RSS and year == 2026:
            raise RuntimeError("RSS 2026 not published")
        return [
            RoboticsPaperRecord(
                paper_id=f"{venue.value.lower()}-{year}-p1",
                title=f"{venue.value} {year} paper",
                venue=venue,
                year=year,
                pdf_url=f"https://example.com/{venue.value}/{year}/p1.pdf",
            )
        ]

    cfg = PrepareConfig(
        data_root=tmp_path,
        years=(2025, 2026),
        venues=(RoboticsVenue.RSS, RoboticsVenue.CORL),
        proxy=RoboticsProxyConfig(),
        min_interval_s=0.0,
    )
    with patch("pipeline.robotics_corpus.prepare.RoboticsHttpClient") as client_cls, \
         patch("pipeline.robotics_corpus.prepare.collect_venue_year", side_effect=fake_collect):
        client_cls.return_value.__enter__.return_value = MagicMock()
        client_cls.return_value.__exit__.return_value = False
        summary = collect_manifest(cfg)

    assert summary["records_written"] == 3  # RSS-2025, CoRL-2025, CoRL-2026
    assert any(c["year"] == 2026 and c["venue"] == "RSS" for c in summary["skipped_cells"])
    assert cfg.manifest_path.exists()
    lines = cfg.manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["download_status"] == "planned"


def test_download_pass_limit_zero_means_all(tmp_path: Path):
    records = [
        RoboticsPaperRecord(
            paper_id=f"rss-2024-p{i:03d}",
            title=f"t{i}",
            venue=RoboticsVenue.RSS,
            year=2024,
            pdf_url=f"https://example.com/p{i}.pdf",
            download_status="planned",
        )
        for i in range(3)
    ]
    manifest = tmp_path / "manifest" / "robotics_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(json.dumps(r.to_dict()) for r in records) + "\n",
        encoding="utf-8",
    )
    cfg = PrepareConfig(
        data_root=tmp_path,
        execute_download=False,
        download_limit=0,
        proxy=RoboticsProxyConfig(),
    )
    summary = run_download_pass(cfg)
    assert summary["attempted"] == 3
    assert summary["planned"] == 3
