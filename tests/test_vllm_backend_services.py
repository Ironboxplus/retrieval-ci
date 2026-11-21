"""The backend's clients against real vLLM services (BGE-M3 embedding, bge-reranker-v2-m3).

Needs RETRIEVAL_TEST_VLLM_EMBED_URL / RETRIEVAL_TEST_VLLM_RERANK_URL (tests/vllm_support.py);
skipped, with that reason, when they are not set. A recording proxy in front of the real service
shows what really went over the wire.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tests.embedding_model_support import bge_m3_snapshot, real_text, text_of_tokens
from tests.vllm_support import EMBED_ENV, RERANK_ENV, RecordingProxy, service_url


@pytest.fixture(autouse=True)
def tokenizer_env(monkeypatch):
    monkeypatch.setenv("BGE_M3_PATH", bge_m3_snapshot())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture()
def embed_url():
    return service_url(EMBED_ENV)


@pytest.fixture()
def rerank_url():
    return service_url(RERANK_ENV)


def _tokenizer():
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    return load_tokenizer(get_profile())


# ---------------------------------------------------------------------------------- embedding
def test_the_embedder_sends_priority_at_the_top_level_and_returns_unit_vectors(embed_url):
    from core.indexing.embedders import VllmBgeM3Embedder

    texts = [real_text(_tokenizer(), n) for n in (40, 300, 900)]
    with RecordingProxy(embed_url) as proxy:
        embedder = VllmBgeM3Embedder(base_url=proxy.url)
        interactive = embedder.embed_texts(texts, priority="interactive")
        background = embedder.embed_texts(texts, priority="background")

    bodies = proxy.posts("/pooling")
    assert [b["priority"] for b in bodies] == [0, 5]                       # top level, not inside "data"
    assert all("priority" not in b["data"] for b in bodies)
    assert all(b["data"]["input"] and isinstance(b["data"]["input"][0][0], int) for b in bodies)   # token ids
    for result in (interactive, background):
        assert result.dense.shape == (3, 1024) and result.dense.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(result.dense, axis=1), 1.0, atol=1e-2)
        assert len(result.sparse) == 3 and all(isinstance(k, str) for w in result.sparse for k in w)
    assert float(np.sum(interactive.dense * background.dense, axis=1).min()) >= 0.9999


def test_the_startup_check_accepts_the_real_service(embed_url, monkeypatch):
    """What main.py's lifespan runs: the service's directory names the profile's revision and its
    max_model_len is the profile's (on .88 the directory is the revision-named symlink)."""
    from config import settings
    from core.indexing import embedders
    from core.indexing.embedders import require_embedding_service

    monkeypatch.setattr(settings.embedding, "service_url", embed_url)
    embedders.reset_embedder_cache()
    try:
        assert require_embedding_service(wait_s=0.0) is True
    finally:
        embedders.reset_embedder_cache()


def test_an_over_long_text_never_reaches_the_service(embed_url):
    from core.indexing.embedders import VllmBgeM3Embedder
    from core.indexing.embedders.bge_m3_core import InputTooLong
    from core.indexing.embedding_profile import get_profile

    too_long = text_of_tokens(_tokenizer(), get_profile().max_seq_len + 1)
    with RecordingProxy(embed_url) as proxy:
        with pytest.raises(InputTooLong):
            VllmBgeM3Embedder(base_url=proxy.url).embed_texts(["fine", too_long], priority="interactive")

    assert proxy.posts("/pooling") == []


def test_a_background_call_two_seconds_after_an_interactive_one_waits_about_eight_seconds(embed_url):
    from core.indexing.embedders import ForegroundGate, VllmBgeM3Embedder
    import core.indexing.embedders as embedders

    embedder = VllmBgeM3Embedder(base_url=embed_url)
    original = embedders._FOREGROUND_GATE
    embedders._FOREGROUND_GATE = ForegroundGate()                           # a fresh gate: no earlier call counts
    try:
        embedder.embed_texts(["a business request"], priority="interactive")
        time.sleep(2.0)
        started = time.monotonic()
        embedder.embed_texts(["background indexing"], priority="background")
        elapsed = time.monotonic() - started
    finally:
        embedders._FOREGROUND_GATE = original

    assert 7.5 <= elapsed <= 9.5, f"background call took {elapsed:.2f}s (8s wait plus the encode)"


# ------------------------------------------------------------------------------------ reranker
def _passages():
    tokenizer = _tokenizer()
    return [
        "The reranker scores each (query, passage) pair with a cross-encoder.",
        "A completely unrelated passage about cooking pasta with tomatoes and basil.",
        real_text(tokenizer, 600),
        "Qdrant stores dense and sparse vectors for every chunk of the code index.",
    ]


def test_scores_are_probabilities_in_input_order_and_independent_of_the_batch_split(rerank_url):
    from core.retrieval.reranker import VllmReranker

    query = "how does a cross-encoder rerank passages"
    texts = _passages() * 3                                                  # 12 pairs
    together = VllmReranker(base_url=rerank_url, batch_size=32).score_pairs(query, texts)
    split = VllmReranker(base_url=rerank_url, batch_size=5, in_flight=2).score_pairs(query, texts)

    assert len(together) == 12 and all(0.0 <= s <= 1.0 for s in together)
    assert together[0] > together[1]                                         # the on-topic passage wins
    np.testing.assert_allclose(split, together, atol=2e-3)                   # order kept across requests
    np.testing.assert_allclose(together[:4], together[4:8], atol=2e-3)


def test_the_request_carries_top_level_truncation_and_priority_and_the_pair_cut_not_refused(rerank_url):
    from config import RERANK_QUERY_TOKENS, settings
    from core.retrieval.reranker import RERANK_PRIORITY, RERANKER_SERVED_MODEL, VllmReranker

    tokenizer = _tokenizer()
    long_query = text_of_tokens(tokenizer, 2000)
    long_passage = text_of_tokens(tokenizer, 9000)
    with RecordingProxy(rerank_url) as proxy:
        scores = VllmReranker(base_url=proxy.url).score_pairs(long_query, [long_passage, "short"])

    (body,) = proxy.posts("/rerank")
    assert body["model"] == RERANKER_SERVED_MODEL and body["priority"] == RERANK_PRIORITY
    assert body["truncate_prompt_tokens"] == settings.reranker.max_length
    assert len(body["documents"]) == 2 and long_query.startswith(body["query"])
    assert len(tokenizer(body["query"], add_special_tokens=False)["input_ids"]) <= RERANK_QUERY_TOKENS + 2
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)         # cut, not an HTTP 400


def test_cancel_between_requests_raises_and_stops_sending(rerank_url):
    from core.retrieval.reranker import RerankCancelledError, VllmReranker

    texts = _passages() * 8                                                  # 32 pairs, batch 4 -> 8 requests
    seen: list[int] = []

    with RecordingProxy(rerank_url) as proxy:
        reranker = VllmReranker(base_url=proxy.url, batch_size=4, in_flight=1)

        def cancel_after_two_requests() -> bool:
            seen.append(len(proxy.posts("/rerank")))
            return len(proxy.posts("/rerank")) >= 2

        with pytest.raises(RerankCancelledError, match="pairs"):
            reranker.score_pairs("cross-encoder", texts, should_cancel=cancel_after_two_requests)

    assert len(proxy.posts("/rerank")) == 2 and seen[0] == 0


def test_rerank_orders_results_by_score_and_keeps_their_fields(rerank_url):
    from app.models.schemas import SearchResult
    from core.retrieval.reranker import VllmReranker

    results = [SearchResult(doc_id=str(i), chunk_id=f"c{i}", content=text, score=0.1, doc_title=f"t{i}",
                            source="s", metadata={"k": i}) for i, text in enumerate(_passages())]

    top = VllmReranker(base_url=rerank_url).rerank("how does a cross-encoder rerank passages", results, top_k=2)

    assert len(top) == 2 and top[0].score >= top[1].score
    assert top[0].doc_id == "0"
    assert top[0].metadata["reranker_score"] == top[0].score and top[0].metadata["k"] == 0


def _stored_flagreranker_reference() -> list[dict]:
    """200 real candidate sets (100 over code, 100 over prose, 20 candidates each) with the scores the torch
    FlagReranker gave them; see the fixture's "source". In-process reranking is retired, so these are the reference."""
    import gzip
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "rerank_flagreranker_reference.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["sets"]


def test_order_and_scores_match_the_stored_flagreranker_reference(rerank_url):
    """Plan section 3.4: the service reorders real candidate sets the way the torch reranker did."""
    from core.retrieval.reranker import VllmReranker

    sets = _stored_flagreranker_reference()
    reranker = VllmReranker(base_url=rerank_url)

    def order(scores):
        return list(np.argsort(-np.asarray(scores), kind="stable"))

    top1 = top10 = full = 0
    sigmoid_diff, logit_diff = [], []
    for item in sets:
        ours = np.asarray(reranker.score_pairs(item["query"], item["texts"]))
        reference = np.asarray(item["flag_scores"])
        top1 += order(ours)[0] == order(reference)[0]
        top10 += order(ours)[:10] == order(reference)[:10]
        full += order(ours) == order(reference)
        sigmoid_diff.append(float(np.abs(ours - reference).max()))
        clip = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))     # noqa: E731
        logit_diff.append(np.abs(clip(ours) - clip(reference)))
    n = len(sets)
    mean_logit = float(np.concatenate(logit_diff).mean())
    print(f"rerank parity over {n} sets: top-1 {top1}/{n}, top-10 order {top10}/{n}, full order {full}/{n}, "
          f"max |sigmoid diff| {max(sigmoid_diff):.5f}, mean |logit diff| {mean_logit:.4f}")

    assert top1 >= 0.99 * n                      # the best candidate is the same one
    assert top10 >= 0.90 * n                     # the plan's floor for exact top-10 order
    assert max(sigmoid_diff) <= 0.01 and mean_logit <= 0.005     # research: mean logit diff 0.0027
