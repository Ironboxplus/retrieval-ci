"""The MCP client compresses large uploads — against a real server, over a real socket.

`index_code` is the only request in this system that is ever large: the body for
this repository alone is 19.3 MiB of JSON, and zstd -3 takes it to 2.2 MiB in
54 ms. Everything else is a query and stays well under the threshold.

The hard requirement is that this cannot depend on both ends upgrading together.
Two directions have to keep working:

  * old client -> new backend: no Content-Encoding, body passed through untouched
    (tests/test_http_compression.py);
  * new client -> old backend: a backend with no decompressor rejects the body
    before the handler ever runs, so the client resends it uncompressed and stops
    compressing for the rest of the session. Covered here, against a server built
    without the middleware.

A real uvicorn server on a loopback port, not an ASGI shortcut: the header and
body handling under test is exactly what a transport would otherwise elide.
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request

zstandard = pytest.importorskip("zstandard")

from app.api.compression import (  # noqa: E402
    CompressedRequestMiddleware,
    CompressedResponseMiddleware,
)


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _WireRecorder:
    """Outermost ASGI middleware: records each request exactly as it arrived."""

    def __init__(self, app, seen: list[dict]):
        self.app = app
        self.seen = seen

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            self.seen.append({
                "method": scope["method"],
                "content_encoding": headers.get("content-encoding"),
                "declared_length": headers.get("content-length"),
                "files": None,
            })
        await self.app(scope, receive, send)


class _Server:
    """A real uvicorn server in a thread, recording what each request looked like."""

    def __init__(self, *, with_compression: bool):
        self.seen: list[dict] = []
        app = FastAPI()

        def record(payload: dict, request: Request) -> dict:
            self.seen[-1]["files"] = len(payload.get("files", []))
            return {"files": len(payload.get("files", []))}

        @app.post("/api/echo")
        async def echo_post(payload: dict, request: Request):  # noqa: ANN201
            return record(payload, request)

        @app.put("/api/echo")
        async def echo_put(payload: dict, request: Request):  # noqa: ANN201
            return record(payload, request)

        if with_compression:
            app.add_middleware(CompressedResponseMiddleware, minimum_size=500)
            app.add_middleware(CompressedRequestMiddleware)

        # Outermost, so it sees the bytes as they arrived. The decompressing
        # middleware deliberately strips Content-Encoding/Content-Length from the
        # scope before the route runs, so a handler cannot report what was on the
        # wire — which is exactly what these tests are about.
        app.add_middleware(_WireRecorder, seen=self.seen)

        self.port = _free_port()
        self._server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self):
        self._thread.start()
        deadline = time.time() + 15
        while not self._server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self._server.started, "uvicorn did not come up"
        return self

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture()
def plugin():
    """The real MCP plugin module, pointed at a throwaway server per test."""
    import mcp_plugin.retrieval_mcp as rm

    saved = (rm.BASE_URL, rm._CLIENT, rm._SERVER_ACCEPTS_ZSTD)
    yield rm
    if rm._CLIENT is not None:
        with contextlib.suppress(Exception):
            import asyncio
            asyncio.run(rm._CLIENT.aclose())
    rm.BASE_URL, rm._CLIENT, rm._SERVER_ACCEPTS_ZSTD = saved


def _aim(rm, server: _Server) -> None:
    rm.BASE_URL = server.url
    rm._CLIENT = None
    rm._SERVER_ACCEPTS_ZSTD = None


def _big_payload(files: int = 4000) -> dict:
    return {"project": "/repo", "files": [
        {"rel_path": f"src/module_{i}.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 9, "label": f"fn_{i}",
                     "content": "qdrant upsert vector wait render button",
                     "code": f"def fn_{i}(): return {i}"}]}
        for i in range(files)]}


class TestUploadsToACapableBackend:

    @pytest.mark.asyncio
    async def test_a_large_body_goes_up_compressed_and_arrives_intact(self, plugin):
        payload = _big_payload()
        raw = len(json.dumps(payload).encode())
        assert raw > plugin._ZSTD_MIN_BYTES, "fixture is below the compression threshold"

        with _Server(with_compression=True) as server:
            _aim(plugin, server)
            out = await plugin._post("/api/echo", payload)

        assert out == {"files": 4000}
        seen = server.seen[-1]
        assert seen["content_encoding"] == "zstd"
        assert seen["files"] == 4000, "the server decoded a different payload"
        # The point of the exercise: far fewer bytes on the wire.
        assert int(seen["declared_length"]) < raw / 3, (
            f"sent {seen['declared_length']} bytes for a {raw}-byte body")

    @pytest.mark.asyncio
    async def test_a_small_body_is_not_worth_compressing(self, plugin):
        with _Server(with_compression=True) as server:
            _aim(plugin, server)
            out = await plugin._post("/api/echo", {"files": ["a.py"], "query": "qdrant"})

        assert out == {"files": 1}
        assert server.seen[-1]["content_encoding"] is None

    @pytest.mark.asyncio
    async def test_put_is_compressed_too_not_only_post(self, plugin):
        """`index_dir` incremental refreshes go out as PUT and are just as large."""
        with _Server(with_compression=True) as server:
            _aim(plugin, server)
            out = await plugin._put("/api/echo", _big_payload())

        assert out == {"files": 4000}
        assert server.seen[-1]["method"] == "PUT"
        assert server.seen[-1]["content_encoding"] == "zstd"


class TestUploadsToAnOlderBackend:
    """The already-deployed backend has no decompressor. It must still work."""

    @pytest.mark.asyncio
    async def test_the_body_is_resent_uncompressed_and_the_call_succeeds(self, plugin):
        payload = _big_payload()

        with _Server(with_compression=False) as server:
            _aim(plugin, server)
            out = await plugin._post("/api/echo", payload)

            assert out == {"files": 4000}, "the fallback did not deliver the payload"
            # One rejected compressed attempt, then the plain body that worked. The
            # rejected one never reached the handler, so nothing ran twice.
            assert [s["content_encoding"] for s in server.seen] == ["zstd", None], server.seen
            assert server.seen[0]["files"] is None, "the handler ran on the zstd body"
            assert server.seen[1]["files"] == 4000

            # And the client must not pay that cost twice in a session.
            await plugin._post("/api/echo", payload)
            assert [s["content_encoding"] for s in server.seen] == ["zstd", None, None], \
                server.seen
            assert plugin._SERVER_ACCEPTS_ZSTD is False

    @pytest.mark.asyncio
    async def test_a_capable_backend_advertises_itself_so_no_attempt_is_wasted(self, plugin):
        """RFC 7694: `Accept-Encoding` as a RESPONSE header means "I accept these in
        requests". One ordinary small call teaches the client, before any big upload."""
        with _Server(with_compression=True) as server:
            _aim(plugin, server)
            await plugin._post("/api/echo", {"files": []})
            assert plugin._SERVER_ACCEPTS_ZSTD is True

            await plugin._post("/api/echo", _big_payload())

        assert [s["content_encoding"] for s in server.seen] == [None, "zstd"]
