"""Online-first reranking (docs/rerank_online_first_plan.md): a batch goes to the hosted reranker
first and to the local vLLM service when that is refused, slow or failing.

Both "services" are real threaded HTTP servers on 127.0.0.1 that speak the ``/rerank`` shape; the
client under test is the real ``VllmReranker`` over real httpx. Nothing here calls the real hosted
service and the keys used are fake literals.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.retrieval.reranker import (
    BACKEND_LOCAL, BACKEND_MIXED, BACKEND_ONLINE, OnlineRerankConfig, RerankCancelledError, VllmReranker,
)

FAKE_KEY = "test-key-not-real"
ONLINE_BASE = 0.50     # online score of document i is ONLINE_BASE + i / 1000
LOCAL_BASE = 0.10      # local score of document i is LOCAL_BASE + i / 1000
SERVER_SLEEP_S = 1.5   # longer than the client's read timeout in these tests


class FakeRerankServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, base_score: float, reported_tokens: int | None = 7):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.base_score = base_score
        self.reported_tokens = reported_tokens
        self.mode = "ok"                      # ok | 429 | 500 | sleep | reset | garbage
        self.script: deque[str] = deque()     # per-request modes consumed before ``mode``
        self.requests: list[dict] = []        # {"path", "headers", "body"} of every request
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address):  # a deliberately dropped connection is not news
        pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()

    @property
    def count(self) -> int:
        with self.lock:
            return len(self.requests)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status: int, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 -- http.server API
        server: FakeRerankServer = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with server.lock:
            server.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": body})
            mode = server.script.popleft() if server.script else server.mode
        if mode == "429":
            return self._send(429, b'{"message": "rate limited"}')
        if mode == "500":
            return self._send(500, b'{"message": "boom"}')
        if mode == "sleep":
            time.sleep(SERVER_SLEEP_S)
            return
        if mode == "reset":
            self.close_connection = True
            self.connection.shutdown(2)
            return
        if mode == "garbage":
            return self._send(200, b'{"unexpected": true}')
        docs = body["documents"]
        results = [{"index": i, "relevance_score": server.base_score + i / 1000} for i in range(len(docs))]
        payload = {"results": results}
        if server.reported_tokens is not None:
            payload["meta"] = {"tokens": {"input_tokens": server.reported_tokens}}
        self._send(200, json.dumps(payload).encode())


@pytest.fixture()
def online():
    with FakeRerankServer(ONLINE_BASE) as server:
        yield server


@pytest.fixture()
def local():
    with FakeRerankServer(LOCAL_BASE, reported_tokens=None) as server:
        yield server


def _config(online_server: FakeRerankServer, **overrides) -> OnlineRerankConfig:
    values = dict(
        base_url=online_server.url + "/v1", model="BAAI/bge-reranker-v2-m3", api_key=FAKE_KEY,
        rpm=1000, tpm=80_000, max_wait_s=0.3, connect_timeout_s=0.5, timeout_s=0.5,
        failures_to_open=3, cooldown_s=0.6,
    )
    values.update(overrides)
    return OnlineRerankConfig(**values)


def _reranker(local_server, online_server, *, batch_size=32, **overrides) -> VllmReranker:
    return VllmReranker(base_url=local_server.url, online=_config(online_server, **overrides),
                        batch_size=batch_size, in_flight=1)


def _expected(base: float, count: int) -> list[float]:
    return [pytest.approx(base + i / 1000) for i in range(count)]


TEXTS = ["alpha passage", "beta passage", "gamma passage"]


# ------------------------------------------------------------------------------ 1. healthy online
def test_a_healthy_online_service_scores_and_the_local_service_is_not_touched(online, local):
    reranker = _reranker(local, online)

    scores = reranker.score_pairs("a query", TEXTS)

    assert scores == _expected(ONLINE_BASE, 3)
    assert local.count == 0 and online.count == 1
    assert reranker.last_backend() == BACKEND_ONLINE == "online"
    assert reranker.online_failures == 0
    assert online.requests[0]["path"] == "/v1/rerank"
    assert reranker._bucket.settled_tokens == 7       # settled to the provider's reported usage


def test_rerank_orders_by_the_online_scores(online, local):
    from app.models.schemas import SearchResult

    results = [SearchResult(doc_id=f"d{i}", chunk_id=f"c{i}", content=t, score=0.0,
                            doc_title="", source="", metadata={}) for i, t in enumerate(TEXTS)]

    top = _reranker(local, online).rerank("a query", results, top_k=2)

    assert [r.doc_id for r in top] == ["d2", "d1"]
    assert top[0].metadata["reranker_score"] == pytest.approx(ONLINE_BASE + 0.002)


# --------------------------------------------------------------------- 2. each failure goes local
@pytest.mark.parametrize("mode", ["429", "500", "sleep", "reset", "garbage"])
def test_an_online_failure_scores_that_batch_on_the_local_service_and_counts_it(online, local, mode):
    online.mode = mode
    reranker = _reranker(local, online)

    started = time.perf_counter()
    scores = reranker.score_pairs("a query", TEXTS)
    elapsed = time.perf_counter() - started

    assert scores == _expected(LOCAL_BASE, 3)
    assert online.count == 1 and local.count == 1
    assert reranker.last_backend() == BACKEND_LOCAL
    assert reranker.online_failures == 1
    assert elapsed < SERVER_SLEEP_S                   # the 0.5 s client timeout cut the sleeping server short


def test_a_partly_failing_search_is_mixed(online, local):
    online.script.extend(["ok", "500"])
    reranker = _reranker(local, online, batch_size=2)

    scores = reranker.score_pairs("a query", ["a", "b", "c", "d"])

    assert scores == [pytest.approx(ONLINE_BASE), pytest.approx(ONLINE_BASE + 0.001),
                      pytest.approx(LOCAL_BASE), pytest.approx(LOCAL_BASE + 0.001)]
    assert reranker.last_backend() == BACKEND_MIXED == "mixed"


# --------------------------------------------------------------------------------- 3. the circuit
def test_three_failures_open_the_circuit_and_a_success_after_the_cooldown_closes_it(online, local):
    online.mode = "500"
    reranker = _reranker(local, online)
    for _ in range(3):
        reranker.score_pairs("a query", TEXTS)
    assert online.count == 3 and reranker.online_failures == 3

    scores = reranker.score_pairs("a query", TEXTS)          # open: online is not touched
    assert scores == _expected(LOCAL_BASE, 3)
    assert online.count == 3 and local.count == 4

    online.mode = "ok"
    time.sleep(0.8)                                          # past the 0.6 s cooldown
    scores = reranker.score_pairs("a query", TEXTS)          # half-open trial succeeds
    assert scores == _expected(ONLINE_BASE, 3)
    assert online.count == 4 and local.count == 4
    assert reranker.online_failures == 0 and reranker.last_backend() == BACKEND_ONLINE

    reranker.score_pairs("a query", TEXTS)                   # closed again
    assert online.count == 5


def test_a_failed_trial_after_the_cooldown_reopens_the_circuit(online, local):
    online.mode = "500"
    reranker = _reranker(local, online)
    for _ in range(3):
        reranker.score_pairs("a query", TEXTS)
    time.sleep(0.8)

    reranker.score_pairs("a query", TEXTS)                   # the one trial, fails
    assert online.count == 4
    reranker.score_pairs("a query", TEXTS)                   # open again: no new online request
    assert online.count == 4
    assert local.count == 5


def test_a_non_service_error_online_falls_back_counts_a_failure_and_keeps_the_trial_alive(online, local):
    # A closed httpx client raises RuntimeError from post(), not an httpx.HTTPError.
    reranker = _reranker(local, online)
    reranker._online_client.close()

    scores = reranker.score_pairs("a query", TEXTS)

    assert scores == _expected(LOCAL_BASE, 3)
    assert online.count == 0 and local.count == 1
    assert reranker.last_backend() == BACKEND_LOCAL
    assert reranker.online_failures == 1

    for _ in range(2):
        reranker.score_pairs("a query", TEXTS)
    assert reranker.online_failures == 3                     # open
    time.sleep(0.8)                                          # past the cooldown

    import httpx
    reranker._online_client = httpx.Client(trust_env=False)  # the online side is healthy again
    scores = reranker.score_pairs("a query", TEXTS)          # the half-open trial must still be let through
    assert scores == _expected(ONLINE_BASE, 3)
    assert online.count == 1 and reranker.online_failures == 0
    assert reranker.last_backend() == BACKEND_ONLINE


# ---------------------------------------------------------------------------------- 4. no key
def test_without_a_key_the_online_service_is_never_called(online, local):
    reranker = VllmReranker(base_url=local.url, online=None, batch_size=32)

    scores = reranker.score_pairs("a query", TEXTS)

    assert scores == _expected(LOCAL_BASE, 3)
    assert online.count == 0 and local.count == 1
    assert not reranker.online_enabled and reranker.last_backend() == BACKEND_LOCAL


def test_default_settings_without_a_key_mean_local_only(monkeypatch, local):
    from config import Settings
    import core.retrieval.reranker as module

    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER__ONLINE_API_KEY", raising=False)
    monkeypatch.setattr(module, "settings", Settings(_env_file=None))

    reranker = VllmReranker(base_url=local.url)

    assert not reranker.online_enabled
    assert reranker.score_pairs("a query", TEXTS) == _expected(LOCAL_BASE, 3)


def test_the_siliconflow_env_key_reaches_the_client_and_is_masked(monkeypatch, online, local):
    from config import Settings
    import core.retrieval.reranker as module

    monkeypatch.setenv("SILICONFLOW_API_KEY", FAKE_KEY)
    monkeypatch.delenv("RERANKER__ONLINE_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER__ONLINE_URL", online.url + "/v1")
    loaded = Settings(_env_file=None)
    monkeypatch.setattr(module, "settings", loaded)

    assert loaded.reranker.online_api_key.get_secret_value() == FAKE_KEY
    for text in (repr(loaded), str(loaded), repr(loaded.reranker), loaded.model_dump_json(),
                 json.dumps(loaded.reranker.model_dump(mode="json"))):
        assert FAKE_KEY not in text

    reranker = VllmReranker(base_url=local.url)
    assert reranker.online_enabled
    assert FAKE_KEY not in repr(reranker._online) and FAKE_KEY not in repr(reranker._online_cfg)
    assert reranker.score_pairs("a query", TEXTS) == _expected(ONLINE_BASE, 3)
    assert online.requests[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"


def test_the_explicit_reranker_key_wins_over_the_siliconflow_one(monkeypatch):
    from config import Settings

    monkeypatch.setenv("SILICONFLOW_API_KEY", "from-siliconflow")
    monkeypatch.setenv("RERANKER__ONLINE_API_KEY", "from-reranker")

    assert Settings(_env_file=None).reranker.online_api_key.get_secret_value() == "from-reranker"


def test_building_settings_does_not_leak_a_key_into_the_shared_default(monkeypatch):
    from config import Settings

    monkeypatch.delenv("RERANKER__ONLINE_API_KEY", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", FAKE_KEY)
    assert Settings(_env_file=None).reranker.online_api_key.get_secret_value() == FAKE_KEY
    monkeypatch.delenv("SILICONFLOW_API_KEY")
    assert Settings(_env_file=None).reranker.online_api_key.get_secret_value() == ""


def test_online_defaults_match_the_plan():
    from config import RerankerSettings

    cfg = RerankerSettings()
    assert (cfg.online_url, cfg.online_model) == ("https://api.siliconflow.cn/v1", "BAAI/bge-reranker-v2-m3")
    assert (cfg.online_rpm, cfg.online_tpm, cfg.online_max_wait_s) == (1000, 80000, 1.0)
    assert (cfg.online_connect_timeout_s, cfg.online_timeout_s) == (1.5, 3.0)
    assert (cfg.online_failures_to_open, cfg.online_cooldown_s) == (3, 300.0)


# ------------------------------------------------------------------------------------- 5. bucket
def test_a_request_the_bucket_cannot_hold_goes_local_at_once_and_is_not_a_failure(online, local):
    # 1 token a second: the first request (empty bucket) goes, charging far more than a minute
    # of budget, so every later one would have to wait minutes -- far past max_wait_s.
    reranker = _reranker(local, online, tpm=60, max_wait_s=0.3)
    long_texts = ["x" * 300, "y" * 300]
    assert reranker.score_pairs("a query", long_texts) == _expected(ONLINE_BASE, 2)
    assert online.count == 1

    for _ in range(5):
        started = time.perf_counter()
        scores = reranker.score_pairs("a query", long_texts)
        assert time.perf_counter() - started < 0.3 + 1.0
        assert scores == _expected(LOCAL_BASE, 2)

    assert online.count == 1 and local.count == 5
    assert reranker.online_failures == 0                    # a refusal for budget is not a failure
    assert reranker.last_backend() == BACKEND_LOCAL


# ----------------------------------------------------------------------------- 7. what goes over
def test_each_request_carries_only_the_fields_its_service_understands(online, local):
    online.mode = "ok"
    reranker = _reranker(local, online)
    reranker.score_pairs("a query", TEXTS)                  # online
    online.mode = "500"
    reranker.score_pairs("a query", TEXTS)                  # fails online, then local

    sent = online.requests[0]
    assert set(sent["body"]) == {"model", "query", "documents", "return_documents"}
    assert sent["body"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert sent["body"]["query"] == "a query" and sent["body"]["documents"] == TEXTS
    assert sent["body"]["return_documents"] is False
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    from core.retrieval.reranker import RERANKER_SERVED_MODEL
    received = local.requests[0]
    assert received["path"] == "/rerank"
    assert received["body"] == {
        "model": RERANKER_SERVED_MODEL, "query": "a query", "documents": TEXTS,
        "truncate_prompt_tokens": reranker._max_length, "priority": 0,
    }
    assert not any(name.lower() == "authorization" for name in received["headers"])


# ------------------------------------------------------------------------------- cancel / empty
def test_a_cancel_stops_before_any_request_to_either_service(online, local):
    reranker = _reranker(local, online)

    with pytest.raises(RerankCancelledError):
        reranker.score_pairs("a query", ["a"] * 5, should_cancel=lambda: True)

    assert online.count == 0 and local.count == 0


def test_an_empty_input_sends_nothing(online, local):
    assert _reranker(local, online).score_pairs("a query", []) == []
    assert online.count == 0 and local.count == 0
