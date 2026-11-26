"""arXiv daily-update harvest: withdrawal parsing, content versions, watermark, zero-exit pool.

Nothing is faked. SQLite is a real file under ``tmp_path``. The OAI-PMH
endpoint and the HTML origin are real ``aiohttp.web`` servers on 127.0.0.1, and
the first hop is a real (minimal) SOCKS5 server speaking RFC 1928, so the
traffic crosses the same ``aiohttp_socks`` code path as production. The OAI
page is a real ``ListRecords`` response captured from arXiv (three records: a
withdrawn paper, a withdrawn-then-revived one, a plain three-version one).
"""
from __future__ import annotations

import asyncio
import inspect
import socket
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp_socks import ProxyConnector

import pipeline.paper_crawler.arxiv_html as arxiv_html
from pipeline.download_control import (
    ChainedProxyDownloadError,
    ChainedProxyPool,
    ChainedTargetHTTPError,
)
from pipeline.paper_crawler.arxiv_html import (
    ADDED_COLUMNS,
    ArxivRawRecord,
    HarvestStore,
    download_pending,
    fetch_one,
    harvest_oai,
    html_path,
    iter_status_lines,
    parse_list_records,
    retry_after_seconds,
)

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_oai" / "list_records_withdrawn_and_revived.xml"


def _page() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- A1


def test_fixture_records_carry_withdrawal_and_content_version():
    records, token, error = parse_list_records(_page().encode("utf-8"))
    assert (token, error) == (None, None)
    by_id = {r.arxiv_id: r for r in records}
    assert list(by_id) == ["2608.14288", "2609.10457", "2107.07420"]

    # v2 is 0kb: withdrawn, but v1 (12kb) is the last version with content.
    w = by_id["2608.14288"]
    assert (w.latest_version, w.latest_size, w.withdrawn) == ("v2", "0kb", True)
    assert (w.content_version, w.content_version_date) == ("v1", "2026-08-14T13:21:05+00:00")

    # v2 is 0kb, v3 revives it: not withdrawn, content moves to v3. Its comments
    # still say "withdrawn" - the sizes, not the comments, decide.
    r = by_id["2609.10457"]
    assert "withdrawn" in (r.comments or "")
    assert (r.latest_version, r.latest_size, r.withdrawn) == ("v3", "27108kb", False)
    assert (r.content_version, r.content_version_date) == ("v3", "2026-09-18T06:51:37+00:00")

    p = by_id["2107.07420"]
    assert (p.latest_version, p.latest_size, p.withdrawn) == ("v3", "1589kb", False)
    assert (p.content_version, p.content_version_date) == ("v3", "2024-10-11T19:34:17+00:00")
    assert p.first_version_date == "2021-07-15T16:05:48+00:00"


def _single_record_page(versions: list[tuple[str, str, str | None]]) -> bytes:
    version_xml = "".join(
        f'<version version="{v}"><date>{d}</date>' + (f"<size>{s}</size>" if s is not None else "") + "</version>"
        for v, d, s in versions
    )
    return (
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords><record>'
        "<header><identifier>oai:arXiv.org:2601.00001</identifier><datestamp>2026-01-05</datestamp></header>"
        '<metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>2601.00001</id>'
        f"{version_xml}<title>t</title><authors>a</authors><categories>cs.LG</categories>"
        "<abstract>x</abstract></arXivRaw></metadata></record></ListRecords></OAI-PMH>"
    ).encode("utf-8")


def test_size_rules_missing_case_whitespace_and_all_empty():
    d1, d2 = "Mon, 05 Jan 2026 10:00:00 GMT", "Tue, 06 Jan 2026 10:00:00 GMT"
    # No <size>: cannot tell, so not withdrawn and the newest version is the content.
    [rec], _, _ = parse_list_records(_single_record_page([("v1", d1, "9kb"), ("v2", d2, None)]))
    assert (rec.withdrawn, rec.latest_size, rec.content_version) == (False, None, "v2")
    # Case and surrounding whitespace do not matter.
    [rec], _, _ = parse_list_records(_single_record_page([("v1", d1, "9kb"), ("v2", d2, " 0KB ")]))
    assert (rec.withdrawn, rec.content_version) == (True, "v1")
    # Every version empty (rare): the first one is the content version.
    [rec], _, _ = parse_list_records(_single_record_page([("v1", d1, "0kb"), ("v2", d2, "0kb")]))
    assert (rec.withdrawn, rec.content_version, rec.content_version_date) == (
        True, "v1", "2026-01-05T10:00:00+00:00")


# --------------------------------------------------------------------------- A2


def _rec(arxiv_id: str, *, latest: str, content: str, withdrawn: bool = False,
         content_date: datetime | None = None, latest_date: datetime | None = None,
         datestamp: str = "2026-09-21") -> ArxivRawRecord:
    content_date = content_date or datetime(2026, 9, 1, tzinfo=timezone.utc)
    latest_date = latest_date or content_date
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp=datestamp, categories="cs.LG", title="t", authors="a",
        abstract="x", comments=None, license=None, latest_version=latest,
        latest_version_date=latest_date.isoformat(),
        first_version_date=datetime(2026, 8, 1, tzinfo=timezone.utc).isoformat(),
        latest_size="0kb" if withdrawn else "10kb", content_version=content,
        content_version_date=content_date.isoformat(), withdrawn=withdrawn,
    )


def _row(store: HarvestStore, arxiv_id: str) -> dict:
    cur = store.db.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,))
    return dict(zip([c[0] for c in cur.description], cur.fetchone()))


@pytest.fixture()
def store(tmp_path):
    s = HarvestStore(tmp_path / "h.sqlite3")
    yield s
    s.close()


def test_content_version_drives_requeue_and_withdrawal_does_not(store):
    pid = "2601.00001"
    store.upsert([_rec(pid, latest="v1", content="v1")])
    assert store.pending(max_attempts=3) == [(pid, "v1")]
    store.mark(pid, "ok", html_version="v1", raw_bytes=1, gz_bytes=1)
    assert store.pending(max_attempts=3) == []

    store.upsert([_rec(pid, latest="v2", content="v2")])
    assert _row(store, pid)["status"] == "pending"
    assert store.pending(max_attempts=3) == [(pid, "v2")]
    store.mark(pid, "ok", html_version="v2", raw_bytes=1, gz_bytes=1)

    # v3 is a withdrawal: newest version empty, content stays v2, HTML stays valid.
    store.upsert([_rec(pid, latest="v3", content="v2", withdrawn=True)])
    row = _row(store, pid)
    assert (row["status"], row["attempts"], row["withdrawn"]) == ("ok", 1, 1)  # attempts: reset by the v2 requeue, +1 for its fetch
    assert (row["latest_version"], row["latest_size"], row["content_version"]) == ("v3", "0kb", "v2")
    assert store.pending(max_attempts=3) == []

    # Re-reading the same withdrawal record changes nothing either.
    store.upsert([_rec(pid, latest="v3", content="v2", withdrawn=True)])
    assert _row(store, pid)["status"] == "ok" and store.pending(max_attempts=3) == []

    # v4 revives it: new content version, queued again, flag cleared.
    store.upsert([_rec(pid, latest="v4", content="v4")])
    row = _row(store, pid)
    assert (row["status"], row["attempts"], row["withdrawn"]) == ("pending", 0, 0)
    assert store.pending(max_attempts=3) == [(pid, "v4")]


def test_withdrawn_before_first_fetch_downloads_the_content_version(store):
    store.upsert([_rec("2601.00002", latest="v2", content="v1", withdrawn=True)])
    assert store.pending(max_attempts=3) == [("2601.00002", "v1")]


def test_no_html_is_retried_only_inside_the_window(store):
    now = datetime.now(timezone.utc)
    store.upsert([
        _rec("2601.00010", latest="v1", content="v1", content_date=now - timedelta(days=1)),
        _rec("2601.00011", latest="v1", content="v1", content_date=now - timedelta(days=10)),
    ])
    for pid in ("2601.00010", "2601.00011"):
        store.mark(pid, "no_html", error="HTTP 404")
    assert store.pending(max_attempts=3) == [("2601.00010", "v1")]
    assert store.pending(max_attempts=3, no_html_retry_days=0) == []
    assert [p for p, _ in store.pending(max_attempts=3, no_html_retry_days=30)] == [
        "2601.00010", "2601.00011"]
    summary = store.summary()
    assert summary["no_html_retryable"] == 1
    assert store.summary(no_html_retry_days=30)["no_html_retryable"] == 2


def test_stale_html_version_is_pending_again(store):
    pid = "2601.00020"
    store.upsert([_rec(pid, latest="v2", content="v2")])
    store.mark(pid, "ok", html_version="v1", raw_bytes=1, gz_bytes=1)  # an older version's page
    assert store.pending(max_attempts=3) == [(pid, "v2")]
    store.mark(pid, "ok", html_version="v2", raw_bytes=1, gz_bytes=1)
    assert store.pending(max_attempts=3) == []


def test_rows_from_before_content_version_read_as_latest_version(tmp_path):
    """An old database: opening it adds the columns, and a row with NULL
    content_version behaves as if content_version == latest_version."""
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE papers (
            arxiv_id TEXT PRIMARY KEY, datestamp TEXT NOT NULL, categories TEXT NOT NULL,
            primary_category TEXT NOT NULL, title TEXT NOT NULL, authors TEXT NOT NULL,
            abstract TEXT NOT NULL, comments TEXT, license TEXT, latest_version TEXT NOT NULL,
            latest_version_date TEXT NOT NULL, first_version_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            html_version TEXT, raw_bytes INTEGER, gz_bytes INTEGER, fetched_at TEXT, error TEXT);
        INSERT INTO papers(arxiv_id, datestamp, categories, primary_category, title, authors, abstract,
                           latest_version, latest_version_date, first_version_date, status, html_version)
        VALUES ('2601.00030', '2026-01-05', 'cs.LG', 'cs.LG', 't', 'a', 'x', 'v2',
                '2026-01-05T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 'ok', 'v2');
        """
    )
    db.commit()
    db.close()
    store = HarvestStore(path)
    try:
        assert {r[1] for r in store.db.execute("PRAGMA table_info(papers)")} >= set(ADDED_COLUMNS)
        row = _row(store, "2601.00030")
        assert (row["content_version"], row["withdrawn"]) == (None, 0)
        assert store.pending(max_attempts=3) == []  # html v2 == COALESCE(NULL, latest v2)
        # The same version coming back from OAI does not requeue it.
        store.upsert([_rec("2601.00030", latest="v2", content="v2")])
        assert _row(store, "2601.00030")["status"] == "ok"
        # A newer one does.
        store.upsert([_rec("2601.00030", latest="v3", content="v3")])
        assert store.pending(max_attempts=3) == [("2601.00030", "v3")]
    finally:
        store.close()


def test_summary_counts_withdrawn_and_status_lines(store):
    store.upsert([_rec("2601.00040", latest="v2", content="v1", withdrawn=True),
                  _rec("2601.00041", latest="v1", content="v1")])
    summary = store.summary()
    assert summary["withdrawn"] == 1
    assert any(line.startswith("withdrawn: 1") for line in iter_status_lines(summary))


# --------------------------------------------------------------------------- A3


def test_retry_after_accepts_delay_seconds_and_http_date():
    # RFC 9110 section 10.2.3: Retry-After = HTTP-date / delay-seconds.
    now = datetime(2015, 10, 21, 7, 27, 30, tzinfo=timezone.utc)
    assert retry_after_seconds("120", now=now) == 120.0
    assert retry_after_seconds(" 0 ", now=now) == 0.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT", now=now) == 30.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:00:00 GMT", now=now) == 0.0  # already past
    assert retry_after_seconds("soon", default=17.0, now=now) == 17.0
    assert retry_after_seconds(None, default=17.0, now=now) == 17.0
    assert retry_after_seconds("", default=17.0, now=now) == 17.0


class FakeOai:
    """A real HTTP OAI-PMH endpoint on 127.0.0.1 that records every request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, str]] = []
        self.responses: list = []  # per-request overrides: callables (request) -> web.Response
        self.after_watermark = "2026-09-23"
        self.on_request = None

    def _records_page(self, token: str | None, datestamp: str) -> str:
        page = _page().replace("2026-09-21", datestamp)
        if token:
            page = page.replace("</ListRecords>", f"<resumptionToken>{token}</resumptionToken></ListRecords>")
        return page

    async def handle(self, request: web.Request) -> web.Response:
        query = dict(request.query)
        self.requests.append(query)
        if self.on_request:
            self.on_request(query)
        if self.responses:
            return self.responses.pop(0)(request)
        if query.get("resumptionToken") == "tok1":
            return web.Response(text=self._records_page(None, self.after_watermark),
                                content_type="text/xml")
        if query.get("from", "") >= self.after_watermark:
            return web.Response(
                text='<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
                     '<error code="noRecordsMatch">no matching records</error></OAI-PMH>',
                content_type="text/xml")
        return web.Response(text=self._records_page("tok1", "2026-09-21"), content_type="text/xml")


async def _serve(handlers: dict[str, object]):
    """Start a real aiohttp server on a free loopback port; returns (runner, base url)."""
    app = web.Application()
    for path, handler in handlers.items():
        app.router.add_get(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - the bound ephemeral port
    return runner, f"http://127.0.0.1:{port}"


def _walk(store: HarvestStore, base: str, **kw):
    return harvest_oai(store, first_hop_proxy=None, endpoint=f"{base}/oai", page_delay_s=0,
                       retry_floor_s=0.01, log=lambda _m: None, **kw)


def test_watermark_advances_and_next_walk_starts_from_it(store):
    oai = FakeOai()
    seen_done: list[str | None] = []
    oai.on_request = lambda query: seen_done.append(store.get_state("oai_done"))

    async def scenario():
        runner, base = await _serve({"/oai": oai.handle})
        try:
            store.set_state("oai_done", "stale-from-a-previous-walk")
            first = await _walk(store, base, from_date="2026-01-01")
            state_after_first = (store.get_state("oai_from"), store.get_state("oai_done"),
                                 store.get_state("oai_token"))
            second = await _walk(store, base, from_date="2026-01-01")
            return first, state_after_first, second
        finally:
            await runner.cleanup()

    first, state_after_first, second = asyncio.run(scenario())

    # Walk 1: no watermark -> from_date; page 2 via the resumption token.
    assert first == {"pages": 2, "records": 6}
    assert oai.requests[0] == {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "cs",
                               "from": "2026-01-01"}
    assert oai.requests[1] == {"verb": "ListRecords", "resumptionToken": "tok1"}
    # oai_done was cleared as the walk began (the stale marker was gone by the first request)...
    assert seen_done[0] is None
    # ...and the watermark is the newest datestamp, set together with a fresh oai_done.
    oai_from, oai_done, oai_token = state_after_first
    assert oai_from == "2026-09-23" and oai_done and oai_token is None
    assert store.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3

    # Walk 2 starts at the watermark (inclusive day, OAI-PMH 2.0 section 2.7.1);
    # the server has nothing newer: noRecordsMatch is a finished walk.
    assert oai.requests[2]["from"] == "2026-09-23"
    assert second == {"pages": 1, "records": 0}
    assert store.get_state("oai_from") == "2026-09-23"
    assert store.get_state("oai_done")


def test_without_watermark_the_walk_starts_at_the_newest_stored_datestamp(store):
    store.upsert([_rec("2601.00001", latest="v1", content="v1", datestamp="2026-09-22")])
    oai = FakeOai()

    async def scenario():
        runner, base = await _serve({"/oai": oai.handle})
        try:
            return await _walk(store, base, from_date="2023-12-01")
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
    assert oai.requests[0]["from"] == "2026-09-22"  # not from_date


def test_max_pages_leaves_the_watermark_and_the_token_resumes(store):
    oai = FakeOai()

    async def scenario():
        runner, base = await _serve({"/oai": oai.handle})
        try:
            cut = await _walk(store, base, from_date="2026-01-01", max_pages=1)
            mid = (store.get_state("oai_from"), store.get_state("oai_done"), store.get_state("oai_token"))
            rest = await _walk(store, base, from_date="2026-01-01")
            return cut, mid, rest
        finally:
            await runner.cleanup()

    cut, mid, rest = asyncio.run(scenario())
    assert cut == {"pages": 1, "records": 3}
    assert mid == (None, None, "tok1")  # no watermark, not done, token kept
    assert oai.requests[1] == {"verb": "ListRecords", "resumptionToken": "tok1"}  # resumed, not restarted
    assert rest == {"pages": 1, "records": 3}
    assert store.get_state("oai_from") == "2026-09-23" and store.get_state("oai_token") is None


def test_503_retry_after_in_both_forms_is_honoured(store):
    oai = FakeOai()
    http_date = (datetime.now(timezone.utc) + timedelta(seconds=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    oai.responses = [
        lambda r: web.Response(status=503, headers={"Retry-After": "0"}),
        lambda r: web.Response(status=503, headers={"Retry-After": http_date}),
    ]

    async def scenario():
        runner, base = await _serve({"/oai": oai.handle})
        try:
            started = time.monotonic()
            result = await _walk(store, base, from_date="2026-01-01")
            return result, time.monotonic() - started
        finally:
            await runner.cleanup()

    result, elapsed = asyncio.run(scenario())
    assert result["records"] == 6
    assert len(oai.requests) == 4  # two 503s, then the two pages
    assert elapsed < 10


# --------------------------------------------------------------------------- A4


async def _socks5_server(connects: list[tuple[str, int]]):
    """A minimal RFC 1928 SOCKS5 server (no auth, CONNECT) that records targets."""
    async def pipe(reader, writer):
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def handle(reader, writer):
        try:
            _ver, nmethods = await reader.readexactly(2)
            await reader.readexactly(nmethods)
            writer.write(b"\x05\x00")
            _ver, _cmd, _rsv, atyp = await reader.readexactly(4)
            if atyp == 1:
                host = socket.inet_ntoa(await reader.readexactly(4))
            elif atyp == 3:
                host = (await reader.readexactly((await reader.readexactly(1))[0])).decode()
            else:
                raise ValueError(f"unsupported address type {atyp}")
            port = int.from_bytes(await reader.readexactly(2), "big")
            connects.append((host, port))
            up_reader, up_writer = await asyncio.open_connection(host, port)
            writer.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, f"socks5://127.0.0.1:{server.sockets[0].getsockname()[1]}"


def test_exitless_pool_is_one_first_hop_route():
    async def scenario():
        pool = ChainedProxyPool("socks5://127.0.0.1:1", [])
        state = await pool._acquire(set())
        connector = pool._connector(state.url)
        try:
            return (pool.configured_count, state.url, state.ordinal, type(connector), connector.limit)
        finally:
            await connector.close()

    configured, url, ordinal, kind, limit = asyncio.run(scenario())
    assert (configured, url, ordinal, limit) == (0, None, 0, 1)
    assert kind is ProxyConnector  # first hop only, not a chain


def test_acquire_waits_out_a_cooldown_instead_of_failing():
    async def scenario():
        pool = ChainedProxyPool("socks5://127.0.0.1:1", [])
        state = pool._states[0]
        state.cooldown_until = time.monotonic() + 0.3
        started = time.monotonic()
        leased = await pool._acquire(set())
        waited = time.monotonic() - started
        # Everything excluded is a real "nothing to lease", not a wait.
        started = time.monotonic()
        nothing = await pool._acquire({state.ordinal})
        return leased is state, waited, nothing, time.monotonic() - started

    leased_same, waited, nothing, instant = asyncio.run(scenario())
    assert leased_same and 0.25 <= waited < 2.0
    assert nothing is None and instant < 0.1


def test_pool_with_exits_waits_for_the_earliest_cooldown_and_keeps_its_order():
    async def scenario():
        pool = ChainedProxyPool("socks5://127.0.0.1:1", ["socks5://u:p@127.0.0.1:2", "socks5://u:p@127.0.0.1:3"])
        slow, soon = pool._states
        slow.cooldown_until = time.monotonic() + 30
        soon.cooldown_until = time.monotonic() + 0.2
        started = time.monotonic()
        first = await pool._acquire(set())
        waited = time.monotonic() - started
        # With both free, the unverified/least-used ordering still applies.
        slow.cooldown_until = soon.cooldown_until = 0.0
        soon.verified = True
        second = await pool._acquire(set())
        return first is soon, waited, second is soon, pool.configured_count

    first_is_soon, waited, second_is_verified, configured = asyncio.run(scenario())
    assert first_is_soon and 0.15 <= waited < 2.0
    assert second_is_verified and configured == 2


def test_pool_file_with_no_exits_is_still_an_error(tmp_path):
    empty = tmp_path / "pool.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no permitted exits"):
        ChainedProxyPool.from_file(empty, first_hop_proxy="socks5://127.0.0.1:1")


class FakeHtmlOrigin:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.fail_first = 0
        self.pages: dict[str, str] = {}

    async def handle(self, request: web.Request) -> web.Response:
        path = request.path
        self.paths.append(path)
        if self.fail_first > 0:
            self.fail_first -= 1
            return web.Response(status=500)
        if path not in self.pages:
            return web.Response(status=404)
        return web.Response(text=self.pages[path], content_type="text/html")


def _html(body: str) -> str:
    return f'<html><body><article class="ltx_document">{body}</article></body></html>'


def test_exitless_download_is_pinned_goes_via_first_hop_and_retries_in_place(tmp_path, monkeypatch):
    origin = FakeHtmlOrigin()
    origin.pages["/html/2601.00001v2"] = _html("<p>2601.00001v2 body</p>")
    origin.fail_first = 1  # the one route fails once, is cooled down, then succeeds
    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/html/{tail:.*}": origin.handle})
        monkeypatch.setattr(arxiv_html, "HTML_BASE", base + "/html/")
        try:
            pool = ChainedProxyPool(first_hop, [])
            ok = await fetch_one(pool, tmp_path, "2601.00001", "v2", max_exits=3)
            gone = await fetch_one(pool, tmp_path, "2601.00002", "v1", max_exits=3)
            return ok, gone, base
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    started = time.monotonic()
    (ok_status, ok_fields), (gone_status, gone_fields), base = asyncio.run(scenario())
    assert time.monotonic() - started >= 1.5  # cooldown after the 500 was waited out, not skipped
    # Requested the content version; the first (500) and the retry hit the same URL.
    assert origin.paths[:2] == ["/html/2601.00001v2", "/html/2601.00001v2"]
    assert ok_status == "ok" and ok_fields["html_version"] == "v2"
    assert html_path(tmp_path, "2601.00001").exists()
    assert (gone_status, gone_fields) == ("no_html", {"error": "HTTP 404"})
    assert not list(tmp_path.rglob("*.part"))
    # Every request crossed the first hop; nothing went direct.
    port = int(base.rsplit(":", 1)[1])
    assert len(connects) == len(origin.paths) and {c for c in connects} == {("127.0.0.1", port)}


def test_a_trickling_body_is_cut_off_by_the_total_timeout(tmp_path, monkeypatch):
    """The origin writes one byte every 0.3 s for ~6 s: every read is well inside
    sock_read, so only the total timeout can end the request. A timeout of 1 s must
    return "error" in about a second and leave no partial file."""
    async def trickle(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        for _ in range(20):
            await response.write(b"x")
            await asyncio.sleep(0.3)
        return response

    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/html/{tail:.*}": trickle})
        monkeypatch.setattr(arxiv_html, "HTML_BASE", base + "/html/")
        try:
            pool = ChainedProxyPool(first_hop, [])
            started = time.monotonic()
            result = await fetch_one(pool, tmp_path, "2601.00060", "v1", max_exits=1, timeout_seconds=1.0)
            return result, time.monotonic() - started
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    (status, fields), elapsed = asyncio.run(scenario())
    assert status == "error" and "TimeoutError" in fields["error"]
    assert 0.9 <= elapsed < 2.5  # one attempt, ended by the 1 s total (not by the origin finishing at ~6 s)
    assert not html_path(tmp_path, "2601.00060").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_download_pending_passes_html_timeout_down_and_defaults_to_a_minute(tmp_path, monkeypatch):
    assert arxiv_html.HTML_TIMEOUT_S == 60.0
    for fn, name in ((fetch_one, "timeout_seconds"), (download_pending, "html_timeout_s")):
        assert inspect.signature(fn).parameters[name].default == 60.0

    async def trickle(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        for _ in range(20):
            await response.write(b"x")
            await asyncio.sleep(0.3)
        return response

    store = HarvestStore(tmp_path / "h.sqlite3")
    store.upsert([_rec("2601.00061", latest="v1", content="v1")])
    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/html/{tail:.*}": trickle})
        monkeypatch.setattr(arxiv_html, "HTML_BASE", base + "/html/")
        try:
            started = time.monotonic()
            counts = await download_pending(store, ChainedProxyPool(first_hop, []), tmp_path, concurrency=1,
                                            max_exits=1, html_timeout_s=1.0, log=lambda _m: None)
            return counts, time.monotonic() - started
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    try:
        counts, elapsed = asyncio.run(scenario())
        assert counts == {"ok": 0, "no_html": 0, "error": 1}
        assert elapsed < 3.0
        assert "TimeoutError" in _row(store, "2601.00061")["error"]
    finally:
        store.close()


def test_html_version_falls_back_to_the_requested_version(tmp_path, monkeypatch):
    origin = FakeHtmlOrigin()
    origin.pages["/html/2601.00003v4"] = _html("<p>a page that never names its own version</p>")
    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/html/{tail:.*}": origin.handle})
        monkeypatch.setattr(arxiv_html, "HTML_BASE", base + "/html/")
        try:
            return await fetch_one(ChainedProxyPool(first_hop, []), tmp_path, "2601.00003", "v4", max_exits=1)
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    status, fields = asyncio.run(scenario())
    assert status == "ok" and fields["html_version"] == "v4"


def test_download_pending_requests_each_papers_content_version(tmp_path, monkeypatch):
    """A withdrawn paper is fetched at its last content version, a revived one at
    its newest, and ok marks come back with that version as html_version."""
    store = HarvestStore(tmp_path / "h.sqlite3")
    store.upsert([
        _rec("2601.00050", latest="v2", content="v1", withdrawn=True),
        _rec("2601.00051", latest="v3", content="v3"),
    ])
    origin = FakeHtmlOrigin()
    origin.pages["/html/2601.00050v1"] = _html("<p>2601.00050v1</p>")
    origin.pages["/html/2601.00051v3"] = _html("<p>2601.00051v3</p>")
    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/html/{tail:.*}": origin.handle})
        monkeypatch.setattr(arxiv_html, "HTML_BASE", base + "/html/")
        try:
            return await download_pending(store, ChainedProxyPool(first_hop, []), tmp_path, concurrency=2,
                                          log=lambda _m: None)
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    try:
        counts = asyncio.run(scenario())
        assert counts == {"ok": 2, "no_html": 0, "error": 0}
        assert sorted(origin.paths) == ["/html/2601.00050v1", "/html/2601.00051v3"]
        assert _row(store, "2601.00050")["html_version"] == "v1"
        assert _row(store, "2601.00051")["html_version"] == "v3"
        assert store.pending(max_attempts=3) == []
    finally:
        store.close()


def test_exitless_pool_failure_reports_attempts(tmp_path):
    """Both attempts fail at connect time (closed first hop); the error says how many were made."""
    pool = ChainedProxyPool("socks5://127.0.0.1:1", [])
    dest = tmp_path / "x.part"

    async def scenario():
        # Cooldown between attempts is 2 s (2**1); the assertion below only needs the count.
        with pytest.raises(ChainedProxyDownloadError) as info:
            await pool.download_to_path("http://127.0.0.1:9/x", dest, max_attempts=2,
                                        validator=lambda p: True, timeout_seconds=3.0)
        return str(info.value)

    message = asyncio.run(scenario())
    assert "after 2 attempt(s) over 0 distinct exit(s)" in message


def test_target_404_is_terminal_on_the_exitless_route(tmp_path):
    origin = FakeHtmlOrigin()
    connects: list[tuple[str, int]] = []

    async def scenario():
        socks, first_hop = await _socks5_server(connects)
        runner, base = await _serve({"/{tail:.*}": origin.handle})
        try:
            pool = ChainedProxyPool(first_hop, [])
            with pytest.raises(ChainedTargetHTTPError) as info:
                await pool.download_to_path(f"{base}/missing", tmp_path / "m.part", max_attempts=3,
                                            validator=lambda p: True)
            return info.value.status_code
        finally:
            await runner.cleanup()
            socks.close()
            await socks.wait_closed()

    assert asyncio.run(scenario()) == 404
    assert origin.paths == ["/missing"]  # one request: a 404 is the answer, not a reason to retry
