"""The paper downloader presents a browser TLS fingerprint and a matching User-Agent.

Springer's edge answers OpenSSL's ClientHello with a "Client Challenge" page whatever the
IP or headers (measured 2026-11-15); the same request through a curl-impersonate
handshake gets the PDF. These tests drive the real ``download_papers`` through a real
local forward proxy that records the request headers, the CONNECT line and the raw
ClientHello bytes it receives, so no part of the HTTP layer is replaced.
"""
from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from pipeline.paper_crawler import ccf_a_main_corpus
from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, MainConferenceCorpusBuilder
from pipeline.paper_crawler.runtime_files import is_valid_pdf_artifact
from tests.test_host_turn_download import tiny_pdf

TLS_HANDSHAKE_RECORD = 0x16
ALPN_H2 = b"\x02h2"  # a browser offers h2; httpx's OpenSSL context offers http/1.1 only


class ProbeProxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ProbeHandler)
        self.pdf = tiny_pdf()
        self.requests: list[tuple[str, dict[str, str]]] = []  # (request target, headers) of plain GETs
        self.connects: list[str] = []  # CONNECT authorities
        self.client_hellos: list[bytes] = []  # first TLS record sent through each tunnel
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class ProbeHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — http.server API
        server: ProbeProxy = self.server  # type: ignore[assignment]
        with server.lock:
            server.requests.append((self.path, dict(self.headers.items())))
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(server.pdf)))
        self.end_headers()
        self.wfile.write(server.pdf)

    def do_CONNECT(self):  # noqa: N802 — http.server API
        server: ProbeProxy = self.server  # type: ignore[assignment]
        self.send_response(200, "Connection established")
        self.end_headers()
        self.connection.settimeout(5)
        hello = self.read_tls_record()
        with server.lock:
            server.connects.append(self.path)
            server.client_hellos.append(hello)
        # The tunnel ends here: the point is what the client said, not a served page.
        self.close_connection = True

    def read_tls_record(self) -> bytes:
        data = b""
        while len(data) < 5:
            chunk = self.connection.recv(65536)
            if not chunk:
                return data
            data += chunk
        assert data[0] == TLS_HANDSHAKE_RECORD, f"not a TLS handshake: {data[:5]!r}"
        wanted = 5 + int.from_bytes(data[3:5], "big")
        while len(data) < wanted:
            chunk = self.connection.recv(65536)
            if not chunk:
                break
            data += chunk
        return data

    def log_message(self, *args):
        pass


@pytest.fixture()
def probe():
    server = ProbeProxy()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def one_paper(url: str) -> CorpusPaper:
    return CorpusPaper(
        paper_id="ijcv-2021-probe", title="Probe", authors=["A"], conference="IJCV", year=2021,
        pdf_url=url, paper_page_url=url, source_name="test",
    )


def run_download(tmp_path: Path, probe: ProbeProxy, url: str) -> CorpusPaper:
    builder = MainConferenceCorpusBuilder(
        corpus_root=tmp_path / "corpus", proxy_url=probe.url, download_retry_attempts=1)
    (paper,) = asyncio.run(builder.download_papers(
        [one_paper(url)], max_concurrent=1, max_bytes_per_second=0))
    return paper


def old_client_request(probe: ProbeProxy, url: str) -> None:
    """The client ``download_papers`` built before this change: OpenSSL transport, client-level proxy."""
    async def fetch() -> None:
        async with httpx.AsyncClient(
            proxy=probe.url, timeout=10, follow_redirects=True, trust_env=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*;q=0.8",
                "Accept-Encoding": "identity",
            },
        ) as client:
            await client.get(url)

    try:
        asyncio.run(fetch())
    except httpx.HTTPError:
        pass  # an https target ends at the recording proxy's closed tunnel


def test_pdf_request_carries_the_chrome152_user_agent_and_download_headers(tmp_path, probe):
    paper = run_download(tmp_path, probe, "http://pdf.test/ijcv-2021-probe.pdf")

    assert paper.download_error is None
    assert paper.pdf_path and is_valid_pdf_artifact(Path(paper.pdf_path))
    ((target, headers),) = probe.requests
    assert target == "http://pdf.test/ijcv-2021-probe.pdf"  # absolute-form: a forward proxy request
    assert "Chrome/152" in headers["User-Agent"]
    assert headers["Accept"] == "application/pdf,*/*;q=0.8"
    assert headers["Accept-Encoding"] == "identity"


def test_https_target_is_tunnelled_with_connect_and_a_browser_client_hello(tmp_path, probe):
    run_download(tmp_path, probe, "https://tls.test/ijcv-2021-probe.pdf")

    assert probe.connects == ["tls.test:443"]
    (hello,) = probe.client_hellos
    assert b"tls.test" in hello  # SNI
    assert ALPN_H2 in hello


def test_user_agent_and_impersonation_target_name_the_same_chrome_version():
    target = ccf_a_main_corpus.DOWNLOAD_IMPERSONATE
    assert target.startswith("chrome")
    assert f"Chrome/{target.removeprefix('chrome')}." in ccf_a_main_corpus.DOWNLOAD_USER_AGENT


def test_the_pre_change_client_fails_both_browser_checks(probe):
    # Red proof: the checks above can fail. The old client sends Chrome/124 and an
    # OpenSSL ClientHello without h2.
    old_client_request(probe, "http://pdf.test/old.pdf")
    old_client_request(probe, "https://tls.test/old.pdf")

    ((_, headers),) = probe.requests
    assert "Chrome/152" not in headers["User-Agent"]
    assert probe.connects == ["tls.test:443"]
    (hello,) = probe.client_hellos
    assert ALPN_H2 not in hello
