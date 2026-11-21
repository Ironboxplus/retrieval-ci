"""zstd on the wire, in both directions.

Measured on a real ``index_code`` payload for this repository (650 files,
8,848 chunks, 19.3 MiB of JSON):

    gzip -6    2,714,299 B    7.5x    447 ms
    zstd -1    2,600,437 B    7.8x     46 ms
    zstd -3    2,233,493 B    9.1x     54 ms
    zstd -9    1,882,886 B   10.8x    226 ms

zstd -3 is the operating point: better ratio than gzip -6 at an eighth of the
CPU. What it buys is transfer time over the LAN overlay and the public tunnel.

What it does NOT buy is memory. A compressed body still has to be decompressed
before it can be parsed, so the decompressed size is what the server holds
either way — and an unbounded decompressor turns a small upload into a large
allocation, which is the shape of the 2026-09-04 OOM. Hence the streaming
decompressor here: it feeds the app incrementally instead of materialising a
second full copy of the expanded body on top of the one Starlette keeps.
"""
from __future__ import annotations

import gzip
import json

import httpx
import pytest
from fastapi import FastAPI

zstandard = pytest.importorskip("zstandard")

from app.api.compression import (  # noqa: E402
    CompressedRequestMiddleware,
    CompressedResponseMiddleware,
)


def _app(*, min_response_bytes: int = 500) -> FastAPI:
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict):  # noqa: ANN201
        return {"files": len(payload.get("files", [])),
                "total_chars": sum(len(f) for f in payload.get("files", []))}

    @app.get("/big")
    async def big():  # noqa: ANN201
        return {"rows": ["a highly repetitive row of text" for _ in range(400)]}

    @app.get("/small")
    async def small():  # noqa: ANN201
        return {"ok": True}

    app.add_middleware(CompressedResponseMiddleware, minimum_size=min_response_bytes)
    app.add_middleware(CompressedRequestMiddleware)
    return app


async def _request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://kb") as client:
        return await client.request(method, path, **kwargs)


async def _raw_asgi_get(app: FastAPI, path: str,
                        headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    """Call the ASGI app the way uvicorn does, to see the bytes actually sent."""
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
             "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1234),
             "server": ("kb", 80),
             "headers": [(k.lower().encode(), v.encode()) for k, v in
                         {"host": "kb", **headers}.items()]}
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (start["status"],
            {k.decode().lower(): v.decode() for k, v in start["headers"]},
            body)


PAYLOAD = {"files": ["def some_repetitive_function(): return 1" for _ in range(200)]}
EXPECTED = {"files": 200, "total_chars": 200 * len("def some_repetitive_function(): return 1")}


class TestUploadingAZstdBody:

    @pytest.mark.asyncio
    async def test_a_zstd_request_body_is_decompressed_and_parsed(self):
        raw = json.dumps(PAYLOAD).encode()
        packed = zstandard.ZstdCompressor(level=3).compress(raw)
        assert len(packed) < len(raw) / 3, "fixture is not actually compressible"

        r = await _request(_app(), "POST", "/echo", content=packed,
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "zstd"})

        assert r.status_code == 200, r.text
        assert r.json() == EXPECTED

    @pytest.mark.asyncio
    async def test_a_gzip_request_body_also_works(self):
        """Already-published clients have gzip long before they have zstd."""
        raw = json.dumps(PAYLOAD).encode()
        r = await _request(_app(), "POST", "/echo", content=gzip.compress(raw),
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "gzip"})
        assert r.status_code == 200, r.text
        assert r.json() == EXPECTED

    @pytest.mark.asyncio
    async def test_an_uncompressed_body_is_untouched(self):
        """The 0.4.8 clients in the field send plain JSON and must keep working."""
        r = await _request(_app(), "POST", "/echo", json=PAYLOAD)
        assert r.status_code == 200, r.text
        assert r.json() == EXPECTED

    @pytest.mark.asyncio
    async def test_a_body_split_across_many_frames_reassembles(self):
        """The decompressor is fed incrementally, so frame boundaries must not
        land inside a zstd block and lose data."""
        raw = json.dumps(PAYLOAD).encode()
        packed = zstandard.ZstdCompressor(level=3).compress(raw)

        async def stream():
            for i in range(0, len(packed), 7):   # deliberately awkward size
                yield packed[i:i + 7]

        r = await _request(_app(), "POST", "/echo", content=stream(),
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "zstd"})
        assert r.status_code == 200, r.text
        assert r.json() == EXPECTED

    @pytest.mark.asyncio
    async def test_a_corrupt_compressed_body_is_a_400_not_a_500(self):
        """A truncated upload is a client problem and must read as one."""
        r = await _request(_app(), "POST", "/echo", content=b"not-valid-zstd-at-all",
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "zstd"})
        assert r.status_code == 400, r.text
        assert "zstd" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_a_truncated_body_names_the_upload_not_the_schema(self):
        """Half of a 19 MiB upload decodes into valid-looking partial JSON.

        Reported as a schema error it sends the client looking at its own payload;
        the frame is what ended early, and the message has to say so.
        """
        raw = json.dumps(PAYLOAD).encode()
        packed = zstandard.ZstdCompressor(level=3).compress(raw)

        r = await _request(_app(), "POST", "/echo", content=packed[:len(packed) // 2],
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "zstd"})

        assert r.status_code == 400, r.text
        detail = r.json()["detail"].lower()
        assert "truncated" in detail or "middle of a frame" in detail, detail

    @pytest.mark.asyncio
    async def test_an_unknown_encoding_is_refused_rather_than_guessed(self):
        r = await _request(_app(), "POST", "/echo", content=b"\x00\x01",
                           headers={"Content-Type": "application/json",
                                    "Content-Encoding": "br"})
        assert r.status_code == 415, r.text


class TestAlreadyPublishedClientsAreUnaffected:
    """The 0.4.8 and earlier MCP clients are in the field and are not upgrading
    on this deploy's schedule. Compression is negotiated per request, so they must
    keep working byte-identically without knowing this middleware exists."""

    @pytest.mark.asyncio
    async def test_a_0_4_8_client_request_round_trips_unchanged(self):
        """Plain JSON up, and the Accept-Encoding httpx sends without `zstandard`."""
        r = await _request(_app(), "POST", "/echo", json=PAYLOAD,
                           headers={"Accept-Encoding": "gzip, deflate"})

        assert r.status_code == 200, r.text
        assert r.json() == EXPECTED
        # Small reply, so not even gzip applies: identical bytes to before.
        assert "content-encoding" not in r.headers

    @pytest.mark.asyncio
    async def test_a_0_4_8_client_reads_a_large_reply_it_never_asked_to_compress(self):
        status, headers, raw = await _raw_asgi_get(
            _app(), "/big", {"accept-encoding": "gzip, deflate"})

        assert status == 200
        # gzip is offered, so gzip is what it gets — never zstd it cannot decode.
        assert headers["content-encoding"] == "gzip"
        assert len(json.loads(gzip.decompress(raw))["rows"]) == 400

    @pytest.mark.asyncio
    async def test_a_client_sending_no_accept_encoding_at_all_gets_plain_bytes(self):
        status, headers, raw = await _raw_asgi_get(_app(), "/big", {})

        assert status == 200
        assert "content-encoding" not in headers
        assert len(json.loads(raw)["rows"]) == 400


class TestTheRealIndexRouteAcceptsACompressedUpload:
    """The middlewares in isolation prove the protocol; this proves the pipeline.

    A zstd body has to survive the real router, the real ``CodeIndexRequest``
    validation, the real service, and land in a real Qdrant — which is the only
    thing that shows the decoded bytes were the exact bytes the client meant.
    """

    @pytest.fixture()
    def client(self):
        pytest.importorskip("qdrant_client")
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        from app.api.routes import code_kb
        from app.services.code_kb.jobs import CodeIndexJobQueue
        from app.services.code_kb.service import CodeKBService
        from app.services.code_kb.vector_store import CodeVectorStore
        from config import settings
        from core.storage.qdrant_store import QdrantStore

        # Loopback and hardcoded, never from QDRANT_URL: a deploy host's config must
        # not be able to aim this at the production collection.
        url, collection = "http://127.0.0.1:6333", "code_chunks_compression_test"
        try:
            qc = QdrantClient(url=url, timeout=30)
            qc.get_collections()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no local Qdrant on {url}: {exc}")

        dim = 8
        try:
            qc.delete_collection(collection)
        except Exception:  # noqa: BLE001 - absent is the expected case
            pass
        params = qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
        qc.create_collection(
            collection_name=collection,
            vectors_config={"dense": params} if settings.embedding.supports_sparse else params)

        inner = QdrantStore.__new__(QdrantStore)
        inner.collection_name = collection
        inner.client = qc
        store = CodeVectorStore(collection_name=collection, store=inner)
        store.ensure_search_payload_indexes()

        def embed(texts):
            # Deterministic bag-of-words, so a corrupted body cannot accidentally
            # produce a vector that still matches the query.
            vocab = ["qdrant", "upsert", "vector", "wait", "render", "button", "react", "def"]
            out = []
            for text in texts:
                row = [float(text.lower().count(w)) for w in vocab]
                out.append(row if any(row) else [1.0] * dim)
            return out

        class _Catalog:
            scope_aliases: dict = {}

            def equivalent_project_scopes(self, *, owner_user_id, project, kind="code"):
                return [project]

            def __getattr__(self, name):
                return lambda **kwargs: None

        svc = CodeKBService(embed_fn=embed, vector_store=store)
        catalog = _Catalog()
        queue = CodeIndexJobQueue(service=svc, catalog_service=catalog)
        code_kb.get_code_kb_service = lambda: svc
        code_kb.get_code_index_queue = lambda: queue
        code_kb.get_existing_code_index_queue = lambda: queue
        code_kb.get_index_catalog_service = lambda: catalog

        api = FastAPI()
        api.include_router(code_kb.router, prefix="/api/code")
        api.add_middleware(CompressedResponseMiddleware, minimum_size=500)
        api.add_middleware(CompressedRequestMiddleware)
        from fastapi.testclient import TestClient
        try:
            yield TestClient(api)
        finally:
            queue.shutdown(timeout=5)
            try:
                qc.delete_collection(collection)
            except Exception:  # noqa: BLE001
                pass

    def test_a_zstd_index_upload_is_indexed_and_then_searchable(self, client):
        import time

        body = {"project": "/compressed-repo", "host": "h1", "root_dir": "/compressed-repo",
                "files": [{"rel_path": f"src/m{i}.py", "language": "python",
                           "chunks": [{"start_line": 1, "end_line": 9,
                                       "label": f"upsert_batch_{i}",
                                       "content": "qdrant upsert vector wait",
                                       "code": f"def upsert_{i}(): ..."}]}
                          for i in range(40)]}
        raw = json.dumps(body).encode()
        packed = zstandard.ZstdCompressor(level=3).compress(raw)
        assert len(packed) < len(raw) / 3, "fixture is not actually compressible"

        headers = {"X-Retrival-User-Id": "compress-owner",
                   "Content-Type": "application/json", "Content-Encoding": "zstd"}
        r = client.post("/api/code/index", content=packed, headers=headers)
        assert r.status_code == 202, r.text

        job_id = r.json()["job_id"]
        for _ in range(100):
            state = client.get(f"/api/code/jobs/{job_id}",
                               headers={"X-Retrival-User-Id": "compress-owner"}).json()
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert state["status"] == "completed", state
        assert state["indexed"] == 40, state

        hits = client.post("/api/code/search",
                           json={"query": "qdrant upsert vector", "project": "/compressed-repo",
                                 "top_k": 5},
                           headers={"X-Retrival-User-Id": "compress-owner"}).json()
        assert hits["hits"], hits
        assert hits["hits"][0]["path"].startswith("src/m")


class TestDownloadingAZstdBody:

    @pytest.mark.asyncio
    async def test_a_zstd_capable_client_gets_a_zstd_response(self):
        r = await _request(_app(), "GET", "/big", headers={"Accept-Encoding": "zstd"})
        assert r.status_code == 200
        # httpx decodes it for us; the header proves it arrived compressed.
        assert r.headers["content-encoding"] == "zstd"
        assert len(r.json()["rows"]) == 400

    @pytest.mark.asyncio
    async def test_zstd_is_preferred_over_gzip_when_both_are_offered(self):
        r = await _request(_app(), "GET", "/big",
                           headers={"Accept-Encoding": "gzip, deflate, zstd"})
        assert r.headers["content-encoding"] == "zstd"

    @pytest.mark.asyncio
    async def test_a_gzip_only_client_still_gets_gzip(self):
        r = await _request(_app(), "GET", "/big", headers={"Accept-Encoding": "gzip"})
        assert r.headers["content-encoding"] == "gzip"
        assert len(r.json()["rows"]) == 400

    @pytest.mark.asyncio
    async def test_a_client_that_accepts_nothing_gets_plain_bytes(self):
        r = await _request(_app(), "GET", "/big", headers={"Accept-Encoding": "identity"})
        assert "content-encoding" not in r.headers
        assert len(r.json()["rows"]) == 400

    @pytest.mark.asyncio
    async def test_a_small_response_is_not_worth_compressing(self):
        """Compressing 15 bytes costs CPU and a header and saves nothing."""
        r = await _request(_app(), "GET", "/small", headers={"Accept-Encoding": "zstd"})
        assert "content-encoding" not in r.headers
        assert r.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_the_declared_length_matches_the_compressed_bytes(self):
        """A stale Content-Length from the uncompressed body truncates the read.

        httpx decodes zstd transparently, so it cannot see this — the check has to
        be against the bytes that actually go out. Driving the ASGI callable is not
        a stand-in for the server; it IS the interface uvicorn calls.
        """
        status, headers, raw = await _raw_asgi_get(
            _app(), "/big", {"accept-encoding": "zstd"})

        assert status == 200
        assert headers["content-encoding"] == "zstd"
        declared = headers.get("content-length")
        assert declared is not None, "response must declare its compressed length"
        assert int(declared) == len(raw), (
            f"Content-Length says {declared} but {len(raw)} bytes were sent")
        assert len(json.loads(zstandard.ZstdDecompressor().decompress(raw))["rows"]) == 400
