"""Every tier-1 request goes through the explicit proxy, and nothing goes direct.

The client refuses to exist without an explicit proxy, ignores ambient
HTTP(S)_PROXY, and the live test routes each new source's listing (and the
Semantic Scholar / arXiv lookups) through a real local CONNECT proxy that
counts tunnels and chains them to the configured first hop (127.0.0.1:10808).
"""
from __future__ import annotations

import asyncio
import select
import socket
import threading
from socketserver import StreamRequestHandler, ThreadingTCPServer

import pytest

from pipeline.paper_crawler import ccf_a_main_corpus as corpus
from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    OfficialProceedingsClient,
    build_sources,
)

UPSTREAM = ("127.0.0.1", 10808)  # the deployment's HTTP proxy (first hop)


@pytest.mark.parametrize("proxy", ["", None, "   ", "127.0.0.1:10808", "ftp://127.0.0.1:10808"])
def test_the_listing_client_refuses_to_run_without_an_explicit_proxy(proxy):
    with pytest.raises(RuntimeError, match="explicit"):
        OfficialProceedingsClient(proxy_url=proxy)


def test_the_pdf_downloader_refuses_to_run_without_an_explicit_proxy(tmp_path):
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path, proxy_url="")
    with pytest.raises(RuntimeError, match="explicit"):
        asyncio.run(builder.download_papers([]))


def test_the_listing_client_ignores_ambient_proxy_variables(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "*")
    client = OfficialProceedingsClient(proxy_url="http://127.0.0.1:10808")
    try:
        assert client.client._trust_env is False
    finally:
        client.close()


class CountingConnectProxy(ThreadingTCPServer):
    """A real HTTP CONNECT proxy on localhost that records each tunnel's target
    and chains it through the upstream proxy, so it never connects out directly."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _ConnectHandler)
        self.tunnels: list[str] = []
        self.refused: list[str] = []
        self.lock = threading.Lock()


class _ConnectHandler(StreamRequestHandler):
    def handle(self):
        server: CountingConnectProxy = self.server  # type: ignore[assignment]
        request_line = self.rfile.readline().decode("latin-1").strip()
        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
            pass
        method, _, rest = request_line.partition(" ")
        target = rest.split(" ", 1)[0]
        if method != "CONNECT":
            with server.lock:
                server.refused.append(request_line)
            self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
            return
        with server.lock:
            server.tunnels.append(target.rsplit(":", 1)[0])
        upstream = socket.create_connection(UPSTREAM, timeout=30)
        upstream.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            chunk = upstream.recv(4096)
            if not chunk:
                break
            reply += chunk
        self.connection.sendall(reply)
        if not reply.split(b" ", 2)[1:2] == [b"200"]:
            upstream.close()
            return
        sockets = [self.connection, upstream]
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], 60)
                if not readable:
                    break
                for sock in readable:
                    data = sock.recv(65536)
                    if not data:
                        return
                    (upstream if sock is self.connection else self.connection).sendall(data)
        finally:
            upstream.close()


@pytest.fixture()
def counting_proxy():
    server = CountingConnectProxy()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.network
def test_every_new_source_and_lookup_goes_through_the_configured_proxy(counting_proxy, monkeypatch):
    # An ambient proxy that does not exist: any request honouring it would fail.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    client = OfficialProceedingsClient(proxy_url=f"http://127.0.0.1:{counting_proxy.server_address[1]}")
    sources = build_sources()
    listings = {
        ("ECCV", 2022): "www.ecva.net",
        ("ECCV", 2026): "eccv.ecva.net",
        ("JMLR", 2026): "jmlr.org",
        ("VLDB", 2026): "www.vldb.org",
        ("OSDI", 2026): "www.usenix.org",
        ("IEEE S&P", 2024): "www.computer.org",
        ("SOSP", 2025): "api.crossref.org",
    }
    try:
        for (venue, year), host in listings.items():
            before = len(counting_proxy.tunnels)
            papers = sources[venue].collect_papers(year, client)
            assert papers, (venue, year)
            assert host in counting_proxy.tunnels[before:], (venue, counting_proxy.tunnels[before:])
        paper = corpus.CorpusPaper(paper_id="x", title="Deep Audio-Visual Speech Recognition", authors=[],
                                   conference="TPAMI", year=2022, pdf_url="", paper_page_url="",
                                   source_name="crossref", doi="10.1109/TPAMI.2018.2889052")
        assert corpus.resolve_open_access_pdf_urls([paper], client)["matched"] == 1
        assert "1809.02108" in corpus.fetch_arxiv_records(["1809.02108"], client)
    finally:
        client.close()
    assert {"api.semanticscholar.org", "export.arxiv.org"} <= set(counting_proxy.tunnels)
    assert set(counting_proxy.tunnels) <= set(listings.values()) | {"api.semanticscholar.org", "export.arxiv.org"}
    assert counting_proxy.refused == []
