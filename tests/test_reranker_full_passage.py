"""The reranker scores a whole chunk, not the first ~480 tokens FlagReranker defaults to.

Real bge-reranker-v2-m3 behind the vLLM reranking service (tests/vllm_support.py; skipped without
RETRIEVAL_TEST_VLLM_RERANK_URL). The evidence for the query sits at the END of a ~1,900-token
passage, past a 512-token pair limit: the same passage cut to its first 480 tokens must score low,
the whole one high.
"""
from __future__ import annotations

import pytest

from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens
from tests.vllm_support import RERANK_ENV, service_url

QUERY = "What is the capital city of the fictional country Zorblax?"
EVIDENCE = " The capital city of Zorblax is Quenmar, a port on the northern coast."


@pytest.fixture(scope="module")
def passages():
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", bge_m3_snapshot())
    try:
        tokenizer = load_tokenizer(get_profile())
        filler = text_of_tokens(tokenizer, 1850)
    finally:
        mp.undo()
    return filler + EVIDENCE, filler


def test_the_default_reranker_reads_the_whole_chunk(passages):
    from core.retrieval.reranker import VllmReranker

    whole, without_evidence = passages
    reranker = VllmReranker(base_url=service_url(RERANK_ENV))

    # What the old 512-token pair limit saw: the filler only, the evidence was past the cut.
    cut_score = reranker.score_pairs(QUERY, [without_evidence])[0]
    new_score = reranker.score_pairs(QUERY, [whole])[0]

    assert cut_score < 0.2, f"fixture error: the filler alone should not answer the query ({cut_score:.3f})"
    assert new_score > 0.8, f"the evidence at token ~1900 was not scored ({new_score:.3f})"
