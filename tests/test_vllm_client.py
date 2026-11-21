"""vllm_client against the real BGE-M3 vLLM service, the real tokenizer and stored FlagEmbedding vectors.

The service is the one scripts/remote_embed/bge_m3_service.sh starts (EMBEDDING__SERVICE_URL, default
http://127.0.0.1:18200); the tests skip when it is not running and never stand in for it. The one
extra process is a recording HTTP proxy in front of the real service, so a test can read the
request bodies the client really sent.
"""
from __future__ import annotations

import http.server
import json
import threading
import urllib.request

import numpy as np
import pytest

from core.indexing.embedders import vllm_client
from core.indexing.embedders.bge_m3_core import InputTooLong
from core.indexing.embedding_profile import get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, real_text, text_of_tokens


@pytest.fixture(scope="module")
def service():
    url = vllm_client.service_url()
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/health", timeout=3)
    except OSError:
        pytest.skip(f"BGE-M3 vLLM service is not running at {url}")
    return url


@pytest.fixture(scope="module")
def tokenizer():
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", bge_m3_snapshot())
    yield load_tokenizer(get_profile())
    mp.undo()


def texts_of(tokenizer, count: int = 48) -> list[str]:
    """Real text of mixed length (16..1500 tokens), half of it ending in whitespace (the case where
    vLLM's own tokenizer and the production one disagree)."""
    out = []
    for i in range(count):
        text = real_text(tokenizer, 16 + (i * 397) % 1500)
        out.append(text + ("\n" if i % 2 else ""))
    return out


def test_the_service_serves_the_profile(service):
    entry = vllm_client.check_service(get_profile(), service)
    assert get_profile().revision in entry["root"] and entry["max_model_len"] == 8192


def flagembedding_reference() -> list[dict]:
    """48 real code chunks (513-2873 tokens, half ending in whitespace) with the vectors the production
    FlagEmbedding fp16 worker stored for them (see the fixture's "source"). In-process FlagEmbedding is
    retired, so the stored vectors are the reference."""
    import base64
    import gzip
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "bge_m3_flagembedding_reference.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["embed_fp"] == get_profile().fingerprint
    return [{**row, "dense": np.frombuffer(base64.b64decode(row["dense_f16_b64"]), dtype="<f2").astype(np.float32),
             "sparse": {int(k): v for k, v in row["sparse"].items()}} for row in doc["rows"]]


def test_vectors_match_the_stored_flagembedding_fp16_vectors(service, tokenizer):
    reference = flagembedding_reference()
    texts = [row["text"] for row in reference]
    assert [len(ids) for ids in tokenizer(texts)["input_ids"]] == [row["tokens"] for row in reference]
    dense, sparse = vllm_client.encode(texts, get_profile(), base_url=service, tokenizer=tokenizer)
    ref_dense = np.stack([row["dense"] for row in reference])
    cos = np.sum(dense * ref_dense, axis=1) / np.linalg.norm(dense, axis=1) / np.linalg.norm(ref_dense, axis=1)
    assert dense.dtype == np.float32 and dense.shape == (len(texts), 1024)
    # The service runs with cuBLAS fp16 accumulation (user's choice 2026-11-14). Measured over 25,000 real code
    # chunks against FlagEmbedding fp16: dense cos mean 0.999956, min 0.998087 (PLAN-2026-11-14-vllm-embed-host.md §5).
    assert cos.min() >= 0.998 and cos.mean() >= 0.9999, (cos.min(), cos.mean())

    def top20(d):
        return {k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:20]}

    overlap = [len(top20(a) & top20(row["sparse"])) / min(20, len(row["sparse"])) for a, row in zip(sparse, reference)]
    # Same 25,000 chunks: mean 0.9937, worst row 0.90. The mean bound keeps its margin; a per-row bound at the
    # measured worst would be flaky.
    assert np.mean(overlap) >= 0.9, overlap
    assert all(isinstance(k, int) and v > 0 and k not in (0, 1, 2, 3) for s in sparse for k, v in s.items())


def test_results_do_not_depend_on_how_the_call_is_cut_into_requests(service, tokenizer):
    profile = get_profile()
    texts = texts_of(tokenizer, 24)
    one, _ = vllm_client.encode(texts, profile, base_url=service, tokenizer=tokenizer,
                                rows_per_request=1, max_in_flight=1)
    many, _ = vllm_client.encode(texts, profile, base_url=service, tokenizer=tokenizer,
                                 rows_per_request=5, max_in_flight=3)
    # FP16 accumulation is always on: batch composition moves a row by up to the plan's gate (mean/min).
    cos = np.sum(one * many, axis=1)
    assert cos.mean() >= 0.9999 and cos.min() >= 0.998


def test_an_over_long_text_raises_before_any_request_is_sent(tokenizer):
    # Port 9 refuses connections: had the client sent anything, the error would be a ServiceError.
    over = text_of_tokens(tokenizer, 8193)
    with pytest.raises(InputTooLong) as raised:
        vllm_client.encode(["fine", over], get_profile(), base_url="http://127.0.0.1:9", tokenizer=tokenizer)
    assert raised.value.index == 1 and raised.value.tokens == 8193


def test_the_limit_is_inclusive(service, tokenizer):
    dense, _ = vllm_client.encode([text_of_tokens(tokenizer, 8192)], get_profile(), base_url=service,
                                  tokenizer=tokenizer)
    assert dense.shape == (1, 1024)


def test_empty_input_sends_nothing():
    dense, sparse = vllm_client.encode([], get_profile(), base_url="http://127.0.0.1:9")
    assert dense.shape == (0, 1024) and sparse == []


def test_an_unreachable_service_is_a_service_error(tokenizer):
    with pytest.raises(vllm_client.ServiceError):
        vllm_client.check_service(get_profile(), "http://127.0.0.1:9")
    with pytest.raises(vllm_client.ServiceError):
        vllm_client.encode(["text"], get_profile(), base_url="http://127.0.0.1:9", tokenizer=tokenizer)


class _Recorder(http.server.BaseHTTPRequestHandler):
    upstream = ""
    bodies: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).bodies.append(json.loads(body))
        request = urllib.request.Request(self.upstream + self.path, data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=300) as response:
            payload = response.read()
            self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_requests_carry_token_ids_and_a_top_level_priority(service, tokenizer):
    _Recorder.upstream, _Recorder.bodies = service, []
    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        texts = ["def f(x):\n    return x\n", "alpha beta", "gamma"]
        vllm_client.encode(texts, get_profile(), base_url=f"http://127.0.0.1:{proxy.server_port}",
                           tokenizer=tokenizer, priority=7, rows_per_request=2)
    finally:
        proxy.shutdown()
    # Requests are in flight together, so they may arrive in either order.
    assert sorted(len(b["data"]["input"]) for b in _Recorder.bodies) == [1, 2]
    first = next(b for b in _Recorder.bodies if len(b["data"]["input"]) == 2)
    assert first["priority"] == 7 and "priority" not in first["data"]
    assert first["data"]["input"][0] == tokenizer(texts[0])["input_ids"]     # the production ids, trailing "▁" kept
    assert first["data"]["embed_task"] == "dense&sparse" and first["model"] == vllm_client.SERVED_MODEL_NAME
