"""Helpers for the tests that talk to real vLLM services.

The services are started by whoever owns the GPU window (WSL on the dev box, the user units on
.88); the tests find them through environment variables and skip with that reason when they are not
set, they never start a model themselves:

    RETRIEVAL_TEST_VLLM_EMBED_URL   e.g. http://127.0.0.1:18200
    RETRIEVAL_TEST_VLLM_RERANK_URL  e.g. http://127.0.0.1:18201

``RecordingProxy`` stands in front of a real service and forwards every request unchanged, keeping
(method, path, JSON body) of each one: the way to assert what really went over the wire.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

EMBED_ENV = "RETRIEVAL_TEST_VLLM_EMBED_URL"
RERANK_ENV = "RETRIEVAL_TEST_VLLM_RERANK_URL"


def service_url(env_name: str) -> str:
    url = os.environ.get(env_name, "").strip()
    if not url:
        pytest.skip(f"{env_name} is not set: this test needs a running vLLM service (see tests/vllm_support.py)")
    try:
        httpx.get(url.rstrip("/") + "/v1/models", timeout=10.0, trust_env=False).raise_for_status()
    except Exception as exc:
        pytest.fail(f"{env_name}={url} does not answer /v1/models: {exc!r}")
    return url.rstrip("/")


class RecordingProxy:
    """A loopback reverse proxy to ``upstream`` that records each request it forwards."""

    def __init__(self, upstream: str):
        self.upstream = upstream.rstrip("/")
        self.requests: list[tuple[str, str, object]] = []
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _forward(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                with proxy._lock:
                    proxy.requests.append((self.command, self.path, body))
                response = httpx.request(
                    self.command, proxy.upstream + self.path, content=raw or None, timeout=900.0, trust_env=False,
                    headers={"content-type": self.headers.get("Content-Type", "application/json")})
                self.send_response(response.status_code)
                self.send_header("Content-Type", response.headers.get("content-type", "application/json"))
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            do_GET = do_POST = _forward

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "RecordingProxy":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def posts(self, path: str) -> list[object]:
        with self._lock:
            return [body for method, p, body in self.requests if method == "POST" and p == path]
