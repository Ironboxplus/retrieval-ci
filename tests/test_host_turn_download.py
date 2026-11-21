"""Multi-venue downloads take turns between hosts and back off refusing hosts.

The user's anti-bot rule is "no two consecutive downloads from the same venue";
TPAMI and IJCV both fall back to arxiv.org, so the queue alternates download
HOSTS and, among hosts, venues. The end-to-end test drives the real
``download_papers`` through a real local HTTP forward proxy that records every
request and refuses one host once with HTTP 429.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pypdf import PdfWriter

from pipeline.download_control import HostTurnQueue, interleave_by_key
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    MainConferenceCorpusBuilder,
    download_host,
)
from pipeline.paper_crawler.runtime_files import is_valid_pdf_artifact


def manifest_rows(spec: dict[tuple[str, str], int]) -> list[dict]:
    """``{(venue, host): count}`` -> manifest records, venue blocks in order."""
    rows = []
    for (venue, host), count in spec.items():
        for i in range(count):
            paper_id = f"{venue.lower()}-2024-{host.split('.')[0]}-{i}"
            rows.append(CorpusPaper(
                paper_id=paper_id, title=f"{venue} paper {i}", authors=["A"], conference=venue, year=2024,
                pdf_url=f"http://{host}/{paper_id}.pdf", paper_page_url=f"http://{host}/{paper_id}",
                source_name="test",
            ).to_manifest_record())
    return rows


def load_papers(tmp_path: Path, rows: list[dict]) -> list[CorpusPaper]:
    path = tmp_path / "paper_manifest.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return [CorpusPaper.from_manifest_record(r) for r in json.loads(path.read_text(encoding="utf-8"))]


def assert_no_avoidable_repeat(sequence: list[str], totals: dict[str, int]) -> None:
    """``sequence[i] == sequence[i-1]`` only when no other key still had work."""
    remaining = dict(totals)
    previous = None
    for value in sequence:
        if value == previous:
            others = sum(n for key, n in remaining.items() if key != value)
            assert others == 0, f"{value} repeated while {others} items of other keys were pending"
        remaining[value] -= 1
        previous = value


def test_host_turns_never_repeat_a_host_or_venue_while_another_has_work(tmp_path):
    spec = {
        ("ECCV", "www.ecva.net"): 80, ("TPAMI", "arxiv.org"): 25, ("IJCV", "arxiv.org"): 9,
        ("IJCV", "link.springer.com"): 6, ("JMLR", "jmlr.org"): 12, ("VLDB", "www.vldb.org"): 10,
    }
    rows = manifest_rows(spec)
    random.Random(7).shuffle(rows)
    papers = load_papers(tmp_path, rows)
    queue = HostTurnQueue(interleave_by_key(papers, key=lambda p: p.conference),
                          host_of=download_host, venue_of=lambda p: p.conference)
    emitted = []
    while True:
        paper, wait = queue.next()
        if paper is None:
            assert wait == 0.0
            break
        emitted.append(paper)
    assert len(emitted) == len(papers)
    host_totals: dict[str, int] = {}
    for (_, host), n in spec.items():
        host_totals[host] = host_totals.get(host, 0) + n
    assert_no_avoidable_repeat([download_host(p) for p in emitted], host_totals)
    arxiv_venues = [p.conference for p in emitted if download_host(p) == "arxiv.org"]
    assert arxiv_venues[:18] == ["TPAMI", "IJCV"] * 9  # one host, two venues: they alternate
    # ECCV dominates; its unavoidable single-host tail is as short as possible.
    tail = len(emitted) - max(i for i, p in enumerate(emitted) if download_host(p) != "www.ecva.net") - 1
    assert tail == 80 - (len(papers) - 80)


def test_blocked_host_cools_down_escalates_and_is_abandoned():
    now = [0.0]
    items = [("a.test", i) for i in range(3)] + [("b.test", i) for i in range(2)]
    queue = HostTurnQueue(items, host_of=lambda item: item[0], clock=lambda: now[0],
                          base_cooldown_seconds=10.0, max_strikes=2)
    first, _ = queue.next()
    assert first[0] == "a.test"  # most remaining
    queue.block("a.test", retry_after_seconds=30.0)
    assert queue.requeue(first)
    second, _ = queue.next()
    assert second[0] == "b.test"
    third, _ = queue.next()
    assert third[0] == "b.test"  # a.test cools down, b.test is the only ready host
    assert queue.next() == (None, 30.0)  # Retry-After 30 beats the 10 s base
    now[0] = 31.0
    retried, _ = queue.next()
    assert retried == first
    queue.block("a.test")  # second consecutive refusal reaches max_strikes
    assert queue.abandoned_hosts == {"a.test"}
    assert sorted(queue.abandoned) == [("a.test", 1), ("a.test", 2)]
    assert queue.requeue(retried) is False and retried in queue.abandoned
    assert queue.next() == (None, 0.0)


# ------------------------------------------------------------ end-to-end proxy
def tiny_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class RecordingProxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, refuse_once: set[str], challenge_once: set[str] = frozenset()):
        super().__init__(("127.0.0.1", 0), ProxyHandler)
        self.requests: list[str] = []
        self.refuse_once = set(refuse_once)
        self.challenge_once = set(challenge_once)
        self.pdf = tiny_pdf()
        self.pdf_by_host: dict[str, bytes] = {}
        # url -> (status, content type, body, extra headers): answered before anything else
        self.routes: dict[str, tuple[int, str, bytes, dict[str, str]]] = {}
        self.lock = threading.Lock()


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — http.server API
        server: RecordingProxy = self.server  # type: ignore[assignment]
        host = urlsplit(self.path).hostname or ""
        with server.lock:
            server.requests.append(self.path)
            refuse = host in server.refuse_once
            server.refuse_once.discard(host)
            challenge = host in server.challenge_once
            server.challenge_once.discard(host)
            route = server.routes.get(self.path)
        if route is not None:
            status, content_type, body, headers = route
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
            return
        if challenge:  # what Springer's edge answered through the proxy on 2026-11-10
            body = b"<!DOCTYPE html><html><head><title>Client Challenge</title></head><body></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if refuse:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = server.pdf_by_host.get(host, server.pdf)
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def proxy():
    server = RecordingProxy(refuse_once={"blocked.test"}, challenge_once={"springer.test"})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_download_papers_with_host_turns_alternates_hosts_and_retries_a_refusal(tmp_path, proxy):
    spec = {
        ("JMLR", "jmlr.test"): 3, ("VLDB", "vldb.test"): 3, ("TPAMI", "arxiv.test"): 2,
        ("IJCV", "arxiv.test"): 2, ("IJCV", "springer.test"): 1, ("ECCV", "blocked.test"): 1,
    }
    papers = load_papers(tmp_path, manifest_rows(spec))
    builder = MainConferenceCorpusBuilder(
        corpus_root=tmp_path / "corpus", proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}")
    on_disk = builder.pdf_root / "vldb" / "2024" / "vldb-2024-vldb-0.pdf"
    on_disk.parent.mkdir(parents=True)
    on_disk.write_bytes(tiny_pdf())  # already downloaded: no request, no turn
    result = asyncio.run(builder.download_papers(
        papers, max_concurrent=4, max_bytes_per_second=0, host_turns=True, host_cooldown_seconds=0.5,
    ))

    assert all(p.pdf_path and is_valid_pdf_artifact(Path(p.pdf_path)) for p in result)
    assert all(p.download_error is None for p in result)
    assert not any(url.endswith("/vldb-2024-vldb-0.pdf") for url in proxy.requests)
    hosts = [urlsplit(url).hostname for url in proxy.requests]
    assert hosts.count("blocked.test") == 2  # refused once, retried after the cooldown, not hot
    assert hosts.count("springer.test") == 2  # a 200 challenge page is a refusal too
    by_url_venue = {p.pdf_url: p.conference for p in papers}
    venues = [by_url_venue[url] for url in proxy.requests]
    assert all(a != b for a, b in zip(hosts, hosts[1:])), hosts
    assert all(a != b for a, b in zip(venues, venues[1:])), venues
    stored = json.loads((tmp_path / "corpus" / "manifests" / "paper_manifest.json").read_text(encoding="utf-8"))
    assert len(stored) == len(papers)


def test_every_paper_of_an_abandoned_host_leaves_a_failure_line(tmp_path, proxy, caplog):
    """The cfscan runner learns which papers were tried and not obtained from the "Failed to download <id>:
    <reason>" lines (its attempt records let .88 ingest a slice whose missing papers were all tried). Papers
    still queued when their host was abandoned were never requested and used to leave no line, so 19 papers
    of the 2026-11-16 refill looked never tried."""
    rows = manifest_rows({("TPAMI", "refusing.test"): 6, ("JMLR", "jmlr.test"): 2})
    papers = load_papers(tmp_path, rows)
    for paper in papers:
        if download_host(paper) == "refusing.test":
            proxy.routes[paper.pdf_url] = (429, "text/plain", b"", {"Retry-After": "0"})
    builder = MainConferenceCorpusBuilder(
        corpus_root=tmp_path / "corpus", proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}")
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(builder.download_papers(
            papers, max_concurrent=1, max_bytes_per_second=0, host_turns=True, host_cooldown_seconds=0.05,
        ))

    refused = [p for p in result if download_host(p) == "refusing.test"]
    requested = {url for url in proxy.requests if urlsplit(url).hostname == "refusing.test"}
    assert all(p.pdf_path is None for p in refused)
    assert len(requested) < len(refused)  # the host was abandoned before every paper was asked for
    failed = {}
    for record in caplog.records:
        m = re.match(r"Failed to download (\S+): (.*)", record.getMessage())
        if m:
            failed.setdefault(m.group(1), m.group(2))
    assert {p.paper_id for p in refused} <= failed.keys()
    never_asked = [p for p in refused if p.pdf_url not in requested]
    assert never_asked and all("deferred" in failed[p.paper_id] and "refusing.test" in failed[p.paper_id]
                               for p in never_asked)
    assert all(p.pdf_path for p in result if download_host(p) == "jmlr.test")


def test_acm_browser_queue_alternates_ccs_and_sosp_and_skips_front_matter(tmp_path):
    from scripts.collect.collect_and_download_acm import select_pending_downloads

    def acm(venue: str, year: int, doi: str) -> CorpusPaper:
        return CorpusPaper(
            paper_id=f"{venue.lower()}-{year}-{doi.rpartition('/')[2]}", title=doi, authors=["A"],
            conference=venue, year=year, pdf_url=f"https://dl.acm.org/doi/pdf/{doi}",
            paper_page_url=f"https://doi.org/{doi}", source_name="crossref", doi=doi,
        )

    papers = [acm("CCS", 2024, "10.1145/3658644")]  # proceedings volume: front matter
    papers += [acm("CCS", 2024, f"10.1145/3658644.36700{i:02d}") for i in range(10)]
    papers += [acm("SOSP", 2024, f"10.1145/3694715.36952{i:02d}") for i in range(4)]
    on_disk = tmp_path / "ccs" / "2024" / f"{papers[1].paper_id}.pdf"
    on_disk.parent.mkdir(parents=True)
    on_disk.write_bytes(tiny_pdf())

    pending, front_matter = select_pending_downloads(papers, tmp_path)

    assert front_matter == 1
    assert papers[1].pdf_path == str(on_disk) and all(p is not papers[1] for p, _ in pending)
    venues = [paper.conference for paper, _ in pending]
    assert len(venues) == 13
    assert_no_avoidable_repeat(venues, {"CCS": 9, "SOSP": 4})
    assert venues[:8] == ["CCS", "SOSP"] * 4
