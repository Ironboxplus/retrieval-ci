"""Prometheus instrumentation helper (W128 observability).

`core.observability.setup_metrics(app)` attaches the prometheus-fastapi
instrumentator to a FastAPI app and exposes a `/metrics` endpoint in Prometheus
exposition format. Tested in isolation on a dummy app (not the heavy main.py
backend) so it runs without models/Qdrant.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from core.observability import (build_loki_payload, make_log_relay_router,
                                setup_metrics)


def _app():
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    # Isolated registry so full-suite ordering can't pollute these assertions with
    # a handler="/metrics" label leaked from another test's app on the global one.
    setup_metrics(app, registry=CollectorRegistry())
    return app


def test_metrics_endpoint_exposed_in_prometheus_format():
    client = TestClient(_app())
    # drive one request so the instrumentator records something
    assert client.get("/ping").status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "# HELP" in body and "# TYPE" in body
    # default request metric family must be present
    assert "http_request" in body


def test_metrics_endpoint_excluded_from_its_own_counters():
    """/metrics scrapes shouldn't inflate the request counters for handlers."""
    client = TestClient(_app())
    client.get("/ping")
    body = client.get("/metrics").text
    # the /ping handler is tracked; the /metrics handler is excluded
    assert "/ping" in body
    assert 'handler="/metrics"' not in body


def test_setup_metrics_is_idempotent_safe_custom_endpoint():
    """A custom endpoint path is honored."""
    app = FastAPI()

    @app.get("/x")
    def x():
        return {"ok": True}

    setup_metrics(app, endpoint="/internal/metrics")
    client = TestClient(app)
    client.get("/x")
    assert client.get("/internal/metrics").status_code == 200
    assert client.get("/metrics").status_code == 404


# --- W128 Phase 2: gateway access-log → Loki relay (rides the existing tunnel) ---

def test_build_loki_payload_structure():
    """One pre-serialized line → Loki push-API shape: stream labels + [ns, line]."""
    payload = build_loki_payload('{"username":"m1110"}',
                                 labels={"job": "gateway-access", "host": "bwh"},
                                 ts_ns=1234567890)
    assert payload == {"streams": [{
        "stream": {"job": "gateway-access", "host": "bwh"},
        "values": [["1234567890", '{"username":"m1110"}']],
    }]}


def _relay_app(monkeypatch, *, ok=True):
    captured = {}

    async def fake_push(line, *, loki_url, labels, timeout=2.0):
        captured["line"] = line
        captured["labels"] = labels
        captured["loki_url"] = loki_url
        return ok

    monkeypatch.setattr("core.observability.push_to_loki", fake_push)
    app = FastAPI()
    app.include_router(make_log_relay_router(
        loki_url="http://127.0.0.1:3100",
        labels={"job": "gateway-access", "host": "bwh"}))
    return TestClient(app), captured


def test_relay_route_pushes_serialized_record(monkeypatch):
    client, captured = _relay_app(monkeypatch, ok=True)
    rec = {"evt": "access", "username": "m1110", "path": "/api/search", "status": 200}
    r = client.post("/internal/ingest-log", json=rec)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # the line shipped to Loki is the record re-serialized as compact JSON
    assert json.loads(captured["line"]) == rec
    assert captured["labels"] == {"job": "gateway-access", "host": "bwh"}


def test_relay_route_best_effort_when_loki_down(monkeypatch):
    """A Loki failure must NOT error the relay — the stateless gateway never retries."""
    client, _ = _relay_app(monkeypatch, ok=False)
    r = client.post("/internal/ingest-log", json={"evt": "access", "username": "x"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


class TestLokiFailuresDoNotFloodTheProductionLog:
    """Measured on `.88` 2026-09-02: 9,367 error-shaped lines since 08-31, and
    essentially all of them were this one best-effort push logging a full
    traceback per attempt while Loki was unreachable. The push failing is fine
    and by design; burying every real production error under it is not — it is
    what made a one-off `/api/search` IncompleteRead unfindable in the log.

    These use a real refused connection (127.0.0.1:1), not a stand-in: the
    behaviour under test is how a genuine httpx failure gets reported.
    """

    UNREACHABLE = "http://127.0.0.1:1"

    @pytest.mark.asyncio
    async def test_repeated_identical_failures_are_reported_once(self, caplog):
        from core import observability

        observability._warned_loki_failures.clear()
        with caplog.at_level("WARNING", logger="core.observability"):
            for _ in range(5):
                ok = await observability.push_to_loki(
                    "line", loki_url=self.UNREACHABLE, labels={}, timeout=0.5)
                assert ok is False, "the push must still report failure to its caller"

        assert len(caplog.records) == 1, (
            f"5 identical failures must warn once, got {len(caplog.records)}")

    @pytest.mark.asyncio
    async def test_the_warning_carries_no_traceback_but_names_the_cause(self, caplog):
        from core import observability

        observability._warned_loki_failures.clear()
        with caplog.at_level("WARNING", logger="core.observability"):
            await observability.push_to_loki(
                "line", loki_url=self.UNREACHABLE, labels={}, timeout=0.5)

        record = caplog.records[0]
        assert record.exc_info is None, "a best-effort hop must not dump a stack"
        message = record.getMessage()
        # Which httpx error an unreachable port raises is platform-dependent
        # (ConnectError on Linux, ConnectTimeout on Windows), so assert that the
        # type is named rather than which type it is.
        assert "Connect" in message, f"the cause must survive: {message}"
        assert self.UNREACHABLE in message, f"the target must survive: {message}"

    @pytest.mark.asyncio
    async def test_a_different_failure_is_still_reported(self, caplog):
        """Deduplication must not silence a NEW problem."""
        from core import observability

        observability._warned_loki_failures.clear()
        with caplog.at_level("WARNING", logger="core.observability"):
            await observability.push_to_loki(
                "line", loki_url=self.UNREACHABLE, labels={}, timeout=0.5)
            await observability.push_to_loki(
                "line", loki_url="http://127.0.0.1:2", labels={}, timeout=0.5)

        assert len(caplog.records) == 2, "a new target is a new failure signature"
