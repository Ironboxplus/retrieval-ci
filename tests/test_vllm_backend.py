"""The backend's side of the vLLM switch: settings, the foreground gate, the shared embedder's
plumbing and the reranker client's own logic. Nothing here needs a running service; what does
(the wire format, the real scores) is in test_vllm_backend_services.py.

The gate tests run in real time: the 10 second rule is about wall-clock behaviour.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.indexing.embedders import BaseEmbedder, EmbeddingResult
from tests.embedding_model_support import bge_m3_snapshot


@pytest.fixture(scope="module")
def tokenizer_env():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


# ------------------------------------------------------------------------------ settings
def test_service_urls_default_to_the_loopback_ports_of_the_plan():
    from config import EmbeddingSettings, RerankerSettings

    assert EmbeddingSettings().service_url == "http://127.0.0.1:18200"
    assert RerankerSettings().service_url == "http://127.0.0.1:18201"


def test_service_urls_come_from_the_nested_environment_variables(monkeypatch):
    from config import Settings

    monkeypatch.setenv("EMBEDDING__SERVICE_URL", "http://embed.example:1")
    monkeypatch.setenv("RERANKER__SERVICE_URL", "http://rerank.example:2")

    settings = Settings()

    assert settings.embedding.service_url == "http://embed.example:1"
    assert settings.reranker.service_url == "http://rerank.example:2"


def test_the_startup_wait_for_the_embedding_service_is_a_setting_with_a_default_of_three_minutes(monkeypatch):
    from config import EmbeddingSettings, Settings

    assert EmbeddingSettings().service_wait_s == 180.0
    monkeypatch.setenv("EMBEDDING__SERVICE_WAIT_S", "7.5")
    assert Settings().embedding.service_wait_s == 7.5


def test_require_embedding_service_waits_for_the_configured_time_by_default(embedding_service_url, monkeypatch):
    from config import settings
    from core.indexing.embedders import require_embedding_service

    embedding_service_url("http://127.0.0.1:9")
    monkeypatch.setattr(settings.embedding, "service_wait_s", 1.2)
    started = time.monotonic()
    assert require_embedding_service() is False
    assert 1.1 <= time.monotonic() - started < 6.0


# ------------------------------------------------------------------ the foreground gate (real time)
def test_the_quiet_period_is_ten_seconds():
    from core.indexing.embedders import BACKGROUND_QUIET_S, SERVICE_PRIORITY

    assert BACKGROUND_QUIET_S == 10.0
    assert SERVICE_PRIORITY == {"interactive": 0, "background": 5}


def test_background_runs_at_once_when_no_interactive_call_ever_happened():
    from core.indexing.embedders import ForegroundGate

    gate = ForegroundGate()
    started = time.monotonic()
    with gate.acquire("background"):
        pass
    assert time.monotonic() - started < 0.5


def test_an_interactive_call_resets_the_ten_second_gate_and_background_two_seconds_later_waits_the_rest():
    from core.indexing.embedders import ForegroundGate

    gate = ForegroundGate()                                   # the production quiet period
    with gate.acquire("interactive"):
        pass
    time.sleep(2.0)

    started = time.monotonic()
    with gate.acquire("background"):
        waited = time.monotonic() - started

    assert 7.5 <= waited <= 8.6, f"waited {waited:.2f}s, expected about 8s"

    with gate.acquire("interactive"):                         # a second interactive call resets it again
        pass
    started = time.monotonic()
    with gate.acquire("background"):
        again = time.monotonic() - started
    assert 9.5 <= again <= 10.6, f"waited {again:.2f}s, expected about 10s"


def test_background_waits_for_a_running_interactive_call_and_then_the_quiet_period():
    from core.indexing.embedders import ForegroundGate

    gate = ForegroundGate(quiet_s=0.6)
    inside, release = threading.Event(), threading.Event()
    released_at: list[float] = []

    def interactive():
        with gate.acquire("interactive"):
            inside.set()
            release.wait(10)
            released_at.append(time.monotonic())

    thread = threading.Thread(target=interactive)
    thread.start()
    assert inside.wait(5)
    waited: list[float] = []

    def background():
        started = time.monotonic()
        with gate.acquire("background"):
            waited.append(time.monotonic() - started)
            waited.append(time.monotonic())

    worker = threading.Thread(target=background)
    worker.start()
    time.sleep(0.8)
    assert not waited, "background started while an interactive call was running"
    release.set()
    thread.join(5)
    worker.join(5)

    assert waited and waited[1] - released_at[0] >= 0.55      # started only after the quiet period
    assert waited[1] - released_at[0] < 1.5


def test_interactive_calls_never_wait():
    from core.indexing.embedders import ForegroundGate

    gate = ForegroundGate()
    with gate.acquire("interactive"):
        pass
    started = time.monotonic()
    with gate.acquire("interactive"):
        pass
    assert time.monotonic() - started < 0.5


def test_an_unknown_priority_is_refused():
    from core.indexing.embedders import ForegroundGate

    with pytest.raises(ValueError, match="priority"):
        with ForegroundGate().acquire("urgent"):                # type: ignore[arg-type]
            pass


# ---------------------------------------------------------- the shared embedder and its callers
class RecordingEmbedder(BaseEmbedder):
    """A real BaseEmbedder implementation that records the calls it gets (the service is elsewhere)."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    dimension = 1024
    supports_sparse = True

    def embed_texts(self, texts, *, priority="interactive"):
        self.calls.append((len(texts), priority))
        dense = np.tile(np.eye(1, 1024, dtype=np.float32), (len(texts), 1))
        return EmbeddingResult(dense=dense, sparse=[{"7": 0.5} for _ in texts])


@pytest.fixture()
def registered_embedder():
    from config import settings
    from core.indexing import embedders

    embedder = RecordingEmbedder()
    embedders._SHARED_EMBEDDERS[settings.embedding.model_name] = embedder
    yield embedder
    embedders.reset_embedder_cache()


def test_shared_encode_is_one_background_call_of_the_shared_embedder(registered_embedder):
    from app.services.pipeline.stages import _shared_encode

    dense, sparse = _shared_encode([f"text {i}" for i in range(100)])

    assert registered_embedder.calls == [(100, "background")]        # no pieces, no token budget
    assert dense.shape == (100, 1024) and len(sparse) == 100


def test_embed_in_pieces_and_embed_chunks_make_one_call_not_eight_text_pieces(registered_embedder):
    from app.models.schemas import Chunk
    from core.indexing.embedders import embed_in_pieces, embed_texts

    embed_in_pieces(registered_embedder, ["a"] * 50, "background")
    embed_texts(["b"] * 30, priority="interactive")
    chunks = [Chunk(id=str(i), doc_id="d", content=f"c{i}", level=2, start_idx=0, end_idx=2, metadata={})
              for i in range(20)]
    registered_embedder.embed_chunks(chunks)

    assert registered_embedder.calls == [(50, "background"), (30, "interactive"), (20, "background")]
    assert all(len(c.embedding) == 1024 for c in chunks)


def test_the_embedder_is_built_from_the_service_url_without_contacting_it(monkeypatch):
    from core.indexing.embedders import VllmBgeM3Embedder, create_embedder

    monkeypatch.setenv("EMBEDDING__SERVICE_URL", "http://127.0.0.1:1")       # nothing listens
    from config import Settings
    monkeypatch.setattr("config.settings", Settings())

    embedder = create_embedder()

    assert isinstance(embedder, VllmBgeM3Embedder)
    assert (embedder.dimension, embedder.supports_sparse) == (1024, True)
    assert embedder._base_url == "http://127.0.0.1:1"


def test_an_over_long_text_raises_before_any_request_leaves(tokenizer_env):
    from core.indexing.embedders import VllmBgeM3Embedder
    from core.indexing.embedders.bge_m3_core import InputTooLong
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    from tests.embedding_model_support import text_of_tokens

    embedder = VllmBgeM3Embedder(base_url="http://127.0.0.1:1")              # a request would fail to connect
    too_long = text_of_tokens(load_tokenizer(get_profile()), get_profile().max_seq_len + 1)

    with pytest.raises(InputTooLong):
        embedder.embed_texts(["fine", too_long], priority="interactive")


def test_a_model_the_service_does_not_serve_reaches_the_retired_embedder():
    from core.indexing.embedders import create_embedder
    from core.inference_retired import RetiredInferencePath

    with pytest.raises(RetiredInferencePath):
        create_embedder("sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------------------------------- the reranker client
def test_the_reranker_client_validates_its_settings_and_needs_no_model_load():
    from config import settings
    from core.retrieval.reranker import RERANK_REQUEST_PAIRS, VllmReranker

    reranker = VllmReranker()
    assert reranker._base_url == settings.reranker.service_url.rstrip("/")
    assert reranker._batch_size == RERANK_REQUEST_PAIRS == 32
    assert reranker._max_length == settings.reranker.max_length
    assert reranker._load_model() is reranker                                  # main.py's startup hook
    assert reranker.score_pairs("q", []) == [] and reranker.rerank("q", []) == []
    for kwargs in ({"batch_size": 0}, {"in_flight": 0}, {"max_length": 10}):
        with pytest.raises(ValueError):
            VllmReranker(**kwargs)


def test_a_cancel_that_is_already_set_stops_before_the_first_request():
    from core.retrieval.reranker import RerankCancelledError, VllmReranker

    reranker = VllmReranker(base_url="http://127.0.0.1:1")                     # a request would fail to connect

    with pytest.raises(RerankCancelledError, match="0/5 pairs"):
        reranker.score_pairs("q", ["a"] * 5, should_cancel=lambda: True)


def test_an_unreachable_service_is_a_reranker_service_error_not_a_hang():
    from core.retrieval.reranker import RerankerServiceError, VllmReranker

    reranker = VllmReranker(base_url="http://127.0.0.1:1", timeout_s=5.0)

    with pytest.raises(RerankerServiceError, match="unreachable"):
        reranker.score_pairs("q", ["a", "b"])


def test_a_long_query_is_clipped_to_the_query_token_limit(tokenizer_env):
    from config import RERANK_QUERY_TOKENS
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    from core.indexing.token_budget import token_offsets
    from core.retrieval.reranker import VllmReranker
    from tests.embedding_model_support import text_of_tokens

    profile = get_profile()
    long_query = text_of_tokens(load_tokenizer(profile), 3000)
    short_query = "how does the reranker clip a query"

    clipped = VllmReranker._clip_query(long_query)

    assert long_query.startswith(clipped)
    assert len(token_offsets(clipped, profile)) == RERANK_QUERY_TOKENS
    assert VllmReranker._clip_query(short_query) == short_query


def test_the_default_reranker_factory_returns_the_service_client():
    from core.retrieval.reranker import VllmReranker, create_reranker, get_shared_reranker

    assert isinstance(create_reranker(), VllmReranker)
    assert isinstance(get_shared_reranker(), VllmReranker)


# ------------------------------------------------------------- the startup check of the embedding service
class ModelsServer:
    """A real HTTP server answering ``GET /v1/models`` the way vLLM does (id, root, max_model_len).

    Only the shape of that one answer is imitated; ``vllm_client.check_service`` and the embedder talk to it
    over a real socket. The real service's own answer is checked in test_vllm_backend_services.py.
    """

    def __init__(self, root: str, max_model_len: int = 8192, port: int = 0):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        body = json.dumps({"data": [{"id": "bge-m3", "root": root, "max_model_len": max_model_len}]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "ModelsServer":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def embedding_service_url(monkeypatch):
    """Point the process-wide embedder at ``url`` for one test."""
    from config import settings
    from core.indexing import embedders

    def point(url: str) -> None:
        monkeypatch.setattr(settings.embedding, "service_url", url)
        embedders.reset_embedder_cache()

    yield point
    embedders.reset_embedder_cache()


def test_a_service_serving_the_profile_revision_passes_the_startup_check(embedding_service_url):
    from core.indexing.embedders import require_embedding_service
    from core.indexing.embedding_profile import get_profile

    server = ModelsServer(f"/models/BAAI/bge-m3-{get_profile().revision}").start()
    try:
        embedding_service_url(server.url)
        assert require_embedding_service(wait_s=0.0) is True
    finally:
        server.stop()


@pytest.mark.parametrize("root, max_len, needle", [
    ("/models/BAAI/bge-m3", 8192, "revision"),                                    # the 88 path before the symlink
    ("/models/BAAI/bge-m3-0000000000000000000000000000000000000000", 8192, "revision"),
    (None, 512, "max_model_len"),                                                   # right weights, FlagEmbedding's 512
])
def test_a_wrong_service_stops_the_startup_check_at_once_instead_of_waiting(
        embedding_service_url, root, max_len, needle):
    from core.indexing.embedders import ServiceError, require_embedding_service
    from core.indexing.embedding_profile import get_profile

    server = ModelsServer(root or f"/models/BAAI/bge-m3-{get_profile().revision}", max_model_len=max_len).start()
    try:
        embedding_service_url(server.url)
        started = time.monotonic()
        with pytest.raises(ServiceError, match=needle):
            require_embedding_service(wait_s=30.0)
        assert time.monotonic() - started < 5.0          # a mismatch is final; only "not answering" is waited for
    finally:
        server.stop()


def test_an_unreachable_service_is_waited_for_then_reported_without_raising(embedding_service_url, caplog):
    from core.indexing.embedders import require_embedding_service

    embedding_service_url("http://127.0.0.1:9")                           # port 9 refuses connections
    started = time.monotonic()
    with caplog.at_level("ERROR", logger="core.indexing.embedders"):
        assert require_embedding_service(wait_s=1.5) is False
    assert 1.4 <= time.monotonic() - started < 6.0
    assert "127.0.0.1:9" in caplog.text and "not answering" in caplog.text     # reported, not silent


def test_a_service_that_comes_up_during_the_wait_is_accepted(embedding_service_url):
    import socket

    from core.indexing.embedders import require_embedding_service
    from core.indexing.embedding_profile import get_profile

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ModelsServer(f"/models/BAAI/bge-m3-{get_profile().revision}", port=port)
    embedding_service_url(server.url)
    threading.Timer(1.5, server.start).start()                            # the unit is still loading its model
    try:
        started = time.monotonic()
        assert require_embedding_service(wait_s=20.0) is True
        assert 1.0 <= time.monotonic() - started < 10.0
    finally:
        server.stop()


def test_the_backend_lifespan_runs_the_service_check_outside_any_catch_all():
    """A mismatch must stop the start-up: the call may not sit inside a ``try`` that swallows Exception
    (the search pipeline block does, and would turn a wrong service into ``search_pipeline = None``)."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    lifespan = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    parents = {child: node for node in ast.walk(lifespan) for child in ast.iter_child_nodes(node)}
    calls = [n for n in ast.walk(lifespan) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "require_embedding_service"]
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        parent = parents[node]
        if isinstance(parent, ast.Try) and node in parent.body:
            for handler in parent.handlers:
                caught = handler.type
                assert caught is not None and getattr(caught, "id", "") not in ("Exception", "BaseException"), \
                    "the service check is inside a catch-all try"
        node = parent
