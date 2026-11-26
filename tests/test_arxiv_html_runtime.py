"""Real-network acceptance for the arXiv CS HTML harvester.

Nothing is faked: OAI-PMH pages come from arXiv through the local first hop,
HTML comes from export.arxiv.org through the operator's two-hop pool, state
lives in a real SQLite file. Network tests run only where the runtime pool
file and first hop exist.
"""
from __future__ import annotations

import asyncio
import gzip
import os
import socket
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp_socks import ProxyConnector

from pipeline.download_control import ChainedProxyPool
from pipeline.paper_crawler.arxiv_html import (
    OAI_ENDPOINT,
    ArxivRawRecord,
    HarvestStore,
    download_pending,
    fetch_one,
    html_path,
    parse_list_records,
    pdf_path,
)

FIRST_HOP = "socks5://127.0.0.1:10808"


def _pool_path() -> Path:
    configured = os.environ.get("RETRIVAL_PROXY_POOL_FILE")
    return Path(configured) if configured else Path.home() / "Desktop" / "a.txt"


def _first_hop_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10808), timeout=2):
            return True
    except OSError:
        return False


needs_network = pytest.mark.skipif(not _first_hop_up(), reason="local first hop 10808 not running")
needs_pool = pytest.mark.skipif(not _pool_path().is_file(), reason="runtime proxy pool file absent")


def _record(arxiv_id: str, version: str, date: str) -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp=date[:10], categories="cs.CL cs.AI", title="t", authors="a",
        abstract="x", comments=None, license=None, latest_version=version,
        latest_version_date=date, first_version_date="2024-01-02T00:00:00+00:00",
        latest_size="100kb", content_version=version, content_version_date=date, withdrawn=False,
    )


def test_new_version_requeues_a_fetched_paper():
    with tempfile.TemporaryDirectory() as tmp:
        store = HarvestStore(Path(tmp) / "s.sqlite3")
        try:
            store.upsert([_record("2401.00001", "v1", "2024-01-02T00:00:00+00:00"),
                          _record("2311.00001", "v1", "2023-11-02T00:00:00+00:00")])
            assert [r[0] for r in store.pending(max_attempts=3)] == ["2401.00001"]
            store.mark("2401.00001", "ok", html_version="v1", raw_bytes=10, gz_bytes=5)
            assert store.pending(max_attempts=3) == []
            store.upsert([_record("2401.00001", "v1", "2024-01-02T00:00:00+00:00")])
            assert store.pending(max_attempts=3) == []
            store.upsert([_record("2401.00001", "v2", "2024-03-02T00:00:00+00:00")])
            assert store.pending(max_attempts=3) == [("2401.00001", "v2")]
        finally:
            store.close()


def test_html_path_layout():
    root = Path("r")
    assert html_path(root, "2609.08019") == root / "html" / "2609" / "2609.08019.html.gz"
    assert html_path(root, "cs/0101001") == root / "html" / "cs" / "0101001.html.gz"


@needs_network
def test_real_oai_page_parses_cs_records():
    async def fetch() -> bytes:
        async with aiohttp.ClientSession(
            connector=ProxyConnector.from_url(FIRST_HOP),
            timeout=aiohttp.ClientTimeout(total=240),
            trust_env=False,
        ) as session:
            url = f"{OAI_ENDPOINT}?verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-09-01&until=2026-09-01"
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()

    records, _token, error = parse_list_records(asyncio.run(fetch()))
    assert error is None
    assert len(records) > 50
    for r in records:
        assert r.arxiv_id and r.latest_version.startswith("v")
        assert any(c.startswith("cs.") for c in r.categories.split())
        assert r.first_version_date <= r.latest_version_date


@needs_network
@needs_pool
def test_real_two_hop_html_download_and_missing_html():
    async def run(root: Path):
        pool = ChainedProxyPool.from_file(_pool_path(), first_hop_proxy=FIRST_HOP)
        report = await pool.probe_pdf(
            "https://export.arxiv.org/pdf/2609.08019", required_healthy=3,
            max_candidates=64, concurrency=12,
        )
        assert report.healthy_count > 0
        ok = await fetch_one(pool, root, "2609.08019", "v1", max_exits=12)
        missing = await fetch_one(pool, root, "2406.12345", "v1", max_exits=12)
        return ok, missing

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (status, fields), (missing_status, _) = asyncio.run(run(root))
        assert status == "ok"
        assert fields["html_version"] == "v1"
        body = gzip.decompress(html_path(root, "2609.08019").read_bytes())
        assert len(body) == fields["raw_bytes"] and b"ltx_document" in body
        assert b"URL Extraction from Scholarly Documents" in body
        assert missing_status == "no_html"
        assert not html_path(root, "2406.12345").exists()
        assert not list(root.rglob("*.part"))


@needs_network
@needs_pool
def test_download_reprobes_a_pool_that_lost_its_verified_exits():
    """An unprobed pool (0 verified exits, below concurrency) is re-probed
    by the running download, and the papers still land. ``follow`` keeps the
    run alive until the re-probe reports; its log line ends the enumeration."""
    ids = ["2609.08019", "2406.12345"]
    lines: list[str] = []

    async def run(store: HarvestStore, root: Path):
        pool = ChainedProxyPool.from_file(_pool_path(), first_hop_proxy=FIRST_HOP)
        assert pool.healthy_count == 0

        def log(line: str) -> None:
            lines.append(line)
            if line.startswith("pool re-probe"):
                store.set_state("oai_done", "test")

        return await download_pending(
            store, pool, root, concurrency=2, log=log, follow=True, follow_poll_seconds=0.5,
            probe_url="https://export.arxiv.org/pdf/2609.08019", reprobe_check_seconds=0.01,
        ), pool.healthy_count

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = HarvestStore(root / "s.sqlite3")
        store.upsert([_record(i, "v1", "2024-06-18T00:00:00+00:00") for i in ids])
        counts, healthy = asyncio.run(run(store, root))
        store.close()
        assert counts == {"ok": 1, "no_html": 1, "error": 0}
        reprobes = [line for line in lines if line.startswith("pool re-probe: healthy 0 -> ")]
        assert reprobes, lines
        assert healthy > 0


@needs_network
@needs_pool
def test_pdf_fallback_for_papers_without_html():
    """A 404 on HTML fetches the PDF in the same worker; a paper already
    marked no_html is queued for its PDF; a withdrawn paper's PDF 404 is
    recorded as missing, not an error."""
    async def run(store: HarvestStore, root: Path):
        pool = ChainedProxyPool.from_file(_pool_path(), first_hop_proxy=FIRST_HOP)
        await pool.probe_pdf("https://export.arxiv.org/pdf/2609.08019", required_healthy=3,
                             max_candidates=64, concurrency=12)
        return await download_pending(store, pool, root, concurrency=2, pdf_fallback=True, log=lambda _m: None)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = HarvestStore(root / "s.sqlite3")
        store.upsert([_record("2406.12345", "v1", "2024-06-18T00:00:00+00:00"),
                      _record("2606.01316", "v2", "2026-06-10T00:00:00+00:00")])
        store.mark("2606.01316", "no_html", error="HTTP 404")  # withdrawn by its authors
        counts = asyncio.run(run(store, root))
        rows = dict((r[0], r[1:]) for r in store.db.execute(
            "SELECT arxiv_id, status, pdf_status, pdf_bytes, pdf_attempts FROM papers"))
        assert store.pdf_pending(max_attempts=3) == []
        store.close()
        assert counts == {"ok": 0, "no_html": 1, "error": 0, "pdf_ok": 1, "pdf_missing": 1, "pdf_error": 0}
        assert rows["2406.12345"][:2] == ("no_html", "ok")
        pdf = pdf_path(root, "2406.12345")
        assert pdf.read_bytes()[:5] == b"%PDF-" and pdf.stat().st_size == rows["2406.12345"][2]
        assert rows["2606.01316"] == ("no_html", "missing", None, 1)
        assert not pdf_path(root, "2606.01316").exists()
        assert not list(root.rglob("*.part"))
