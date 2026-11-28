"""One reading of Retry-After and one download backoff curve for every HTTP client.

Retry-After is ``delay-seconds`` or an ``HTTP-date`` (RFC 9110 section 10.2.3).
Each client used to parse it its own way, and the ones that only knew seconds
dropped a date-form answer from the server. Responses are real ``httpx`` /
``requests`` / ``aiohttp`` objects; the HTTP clients under test talk to a local
scripted server through their explicit-proxy path.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import httpx
import pytest
import requests
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from pipeline import http_retry
from pipeline.http_retry import exponential_backoff_seconds, retry_after_from, retry_after_seconds
from pipeline.medical_corpus import europe_pmc
from pipeline.medical_corpus.proxy import MedicalProxyConfig
from pipeline.medical_corpus.shield import classify_http_shield, retry_delay_seconds
from pipeline.paper_crawler.ccf_a_main_corpus import http_retry_delay_seconds
from pipeline.robotics_corpus.models import RoboticsPaperRecord
from pipeline.robotics_corpus.pdf_fetch import download_pdf as robotics_download_pdf
from pipeline.robotics_corpus.proxy import RoboticsProxyConfig
from scripts.reviews import harvest_openreview

NOW = datetime(2015, 10, 21, 7, 27, 30, tzinfo=timezone.utc)
HTTP_DATE_IN_30S = "Wed, 21 Oct 2015 07:28:00 GMT"
HTTP_DATE_PAST = "Wed, 21 Oct 2015 07:00:00 GMT"


def _http_date_in(seconds: float) -> str:
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds), usegmt=True)


def _httpx_response(retry_after: str | None, status: int = 429) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(status, headers=headers, request=httpx.Request("GET", "https://example.test/x"))


# --- retry_after_from: every input shape, every source type ---------------------------------

@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("120", 120.0),
        (" 0 ", 0.0),
        (HTTP_DATE_IN_30S, 30.0),
        (HTTP_DATE_PAST, 0.0),
        ("soon", 17.0),
        ("", 17.0),
        (None, 17.0),
    ],
)
def test_retry_after_from_httpx_response(header: str | None, expected: float) -> None:
    assert retry_after_from(_httpx_response(header), default=17.0, now=NOW) == expected


def test_retry_after_from_status_error_reads_its_response() -> None:
    response = _httpx_response(HTTP_DATE_IN_30S)
    error = httpx.HTTPStatusError("too many", request=response.request, response=response)
    assert retry_after_from(error, default=None, now=NOW) == 30.0


def test_retry_after_from_aiohttp_response_error() -> None:
    error = aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(
            url=URL("https://example.test/x"), method="GET",
            headers=CIMultiDictProxy(CIMultiDict()), real_url=URL("https://example.test/x")),
        history=(), status=503, headers=CIMultiDict({"Retry-After": "45"}))
    assert retry_after_from(error, default=None) == 45.0


def test_retry_after_from_requests_response() -> None:
    response = requests.Response()
    response.status_code = 429
    response.headers["retry-after"] = "9"
    assert retry_after_from(response, default=None) == 9.0


def test_default_none_means_the_server_gave_no_usable_answer() -> None:
    assert retry_after_from(_httpx_response(None), default=None) is None
    assert retry_after_from(_httpx_response("garbage"), default=None) is None
    assert retry_after_from(RuntimeError("no response attached"), default=None) is None


def test_retry_after_seconds_is_exported_unchanged() -> None:
    assert retry_after_seconds("120", now=NOW) == 120.0
    assert retry_after_seconds(HTTP_DATE_PAST, now=NOW) == 0.0
    assert http_retry.retry_after_seconds is retry_after_seconds


def test_download_backoff_curve_doubles_from_one_and_a_half_and_stops_at_thirty() -> None:
    assert [exponential_backoff_seconds(attempt) for attempt in range(1, 8)] == [
        1.5, 3.0, 6.0, 12.0, 24.0, 30.0, 30.0]


# --- shield: Retry-After reaches the delay, not only the message ----------------------------

def test_rate_limit_delay_honours_the_server_retry_after() -> None:
    decision = classify_http_shield(status_code=429, headers={"Retry-After": "45"})
    assert decision.retry_after_s == 45.0
    assert retry_delay_seconds(decision, 1) == 45.0  # the attempt-1 curve alone is 30 s
    assert retry_delay_seconds(decision, 4) == 120.0  # the curve already exceeds the hint


def test_503_http_date_retry_after_reaches_the_delay() -> None:
    decision = classify_http_shield(status_code=503, headers={"Retry-After": _http_date_in(200)})
    assert decision.retry_after_s is not None and 190 < decision.retry_after_s <= 200
    assert 190 < retry_delay_seconds(decision, 1) <= 200


def test_an_absurd_retry_after_is_capped() -> None:
    decision = classify_http_shield(status_code=429, headers={"Retry-After": "86400"})
    assert retry_delay_seconds(decision, 1) == 600.0


def test_no_retry_after_keeps_the_attempt_curve() -> None:
    decision = classify_http_shield(status_code=429)
    assert decision.retry_after_s is None
    assert retry_delay_seconds(decision, 2) == 60.0


# --- clients -------------------------------------------------------------------------------

def _scripted_proxy(replies: list[tuple[int, dict[str, str], bytes]]):
    """A plain-HTTP forward proxy answering every request from ``replies`` in order
    (the last reply repeats). Clients reach it exactly as they reach the real proxy."""
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append(self.path)
            status, headers, body = replies[min(len(seen) - 1, len(replies) - 1)]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, seen


@pytest.fixture()
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def _europe_pmc_client(server) -> europe_pmc.EuropePMCClient:
    proxy_url = f"http://127.0.0.1:{server.server_port}"
    return europe_pmc.EuropePMCClient(
        proxy=MedicalProxyConfig(http_proxy=proxy_url, https_proxy=proxy_url),
        min_interval_s=0.0, max_retries=4)


def test_europe_pmc_waits_the_servers_retry_after(monkeypatch, sleeps) -> None:
    monkeypatch.setattr(europe_pmc, "EUROPE_PMC_SEARCH", "http://europepmc.invalid/search")
    payload = json.dumps({"hitCount": 0, "resultList": {"result": []}}).encode()
    server, seen = _scripted_proxy([
        (429, {"Retry-After": "45", "Content-Type": "text/plain"}, b"slow down"),
        (200, {"Content-Type": "application/json"}, payload),
    ])
    try:
        with _europe_pmc_client(server) as client:
            page = client.search_page("heart")
    finally:
        server.shutdown()
    assert page.hit_count == 0
    assert sleeps == [45.0]
    assert len(seen) == 2


def test_europe_pmc_backoff_does_not_sleep_after_the_last_attempt(monkeypatch, sleeps) -> None:
    monkeypatch.setattr(europe_pmc, "EUROPE_PMC_SEARCH", "http://europepmc.invalid/search")
    server, seen = _scripted_proxy([(200, {"Content-Type": "application/json"}, b"not json")])
    try:
        with _europe_pmc_client(server) as client:
            with pytest.raises(RuntimeError, match="EuropePMC search failed after retries"):
                client.search_page("heart")
    finally:
        server.shutdown()
    assert len(seen) == 4
    assert sleeps == [1.5, 3.0, 6.0]


def test_robotics_pdf_download_backoff_curve(sleeps, tmp_path: Path) -> None:
    server, _seen = _scripted_proxy([(200, {"Content-Type": "text/plain"}, b"x")])
    proxy_url = f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()  # nothing listens any more: every attempt is a real connect error
    record = RoboticsPaperRecord.from_dict({
        "paper_id": "rb-1", "title": "t", "venue": "ICRA", "year": 2026,
        "pdf_url": "http://robotics.invalid/p.pdf"})
    out = robotics_download_pdf(
        record, pdf_root=tmp_path, execute=True, allow_cf_bypass=False, max_retries=3,
        proxy=RoboticsProxyConfig(http_proxy=proxy_url, https_proxy=proxy_url))
    assert out.download_status == "failed"
    assert sleeps == [1.5, 3.0, 6.0]


# --- Retry-After in the date form, per client ------------------------------------------------

def test_ccf_429_with_http_date_retry_after_waits_the_date() -> None:
    response = _httpx_response(_http_date_in(300))
    error = httpx.HTTPStatusError("too many", request=response.request, response=response)
    assert 290 < http_retry_delay_seconds(error, attempt=1) <= 300


def test_ccf_429_without_a_usable_retry_after_keeps_the_sixty_second_floor() -> None:
    for header in (None, "garbage"):
        response = _httpx_response(header)
        error = httpx.HTTPStatusError("too many", request=response.request, response=response)
        assert http_retry_delay_seconds(error, attempt=1) == 60.0


def test_openreview_429_with_http_date_retry_after_waits_the_date(monkeypatch) -> None:
    harvester = harvest_openreview.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    limited = requests.Response()
    limited.status_code = 429
    limited._content = b"{}"
    limited.headers["Retry-After"] = _http_date_in(300)
    ok = requests.Response()
    ok.status_code = 200
    ok._content = json.dumps({"notes": []}).encode()
    replies = iter([limited, ok])
    recorded: list[float] = []
    monkeypatch.setattr(harvester.session, "get", lambda *a, **k: next(replies))
    monkeypatch.setattr(harvest_openreview.time, "sleep", recorded.append)
    assert harvester._get({}) == {"notes": []}
    assert 290 < recorded[0] <= 300
