"""The shared embedder must not cut text: FlagEmbedding's silent 512-token default was the P0.

The embedder is a client of a real BGE-M3 vLLM service (tests/vllm_support.py; skipped without
RETRIEVAL_TEST_VLLM_EMBED_URL) and the tokenizer is the real one. A text that is cut to its first
512 tokens must embed differently from the whole text: if the client or the service truncated,
the two vectors would be the same.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.embedding_model_support import bge_m3_snapshot, real_text, text_of_tokens
from tests.vllm_support import EMBED_ENV, service_url

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    yield load_tokenizer(get_profile())
    mp.undo()


@pytest.fixture(scope="module")
def embedder(tokenizer):
    from core.indexing.embedders import VllmBgeM3Embedder
    return VllmBgeM3Embedder(base_url=service_url(EMBED_ENV))


def test_a_3000_token_text_is_embedded_in_full(embedder, tokenizer):
    text = real_text(tokenizer, 3000)
    cut = real_text(tokenizer, 512)
    assert len(tokenizer(text)["input_ids"]) > 2500 and text.startswith(cut)

    full, cut_vector, again = (embedder.embed_texts([t]).dense[0] for t in (text, cut, text))

    assert float(full @ again) >= 0.9999     # the same input gives the same vector
    assert float(full @ cut_vector) < 0.999  # red if anything between here and the model keeps 512 tokens


def test_the_8192_token_limit_is_inclusive_and_8193_raises(embedder, tokenizer):
    from core.indexing.embedders.bge_m3_core import InputTooLong

    at_limit = text_of_tokens(tokenizer, 8192)
    over = text_of_tokens(tokenizer, 8193)

    result = embedder.embed_texts([at_limit])
    assert result.dense.shape == (1, 1024)
    with pytest.raises(InputTooLong) as raised:
        embedder.embed_texts(["short", over])
    assert raised.value.index == 1 and raised.value.tokens == 8193


def test_encoded_vectors_are_finite_unit_length_float32(embedder):
    result = embedder.embed_texts(["alpha beta", "gamma delta epsilon"])
    assert result.dense.dtype == np.float32 and np.isfinite(result.dense).all()
    np.testing.assert_allclose(np.linalg.norm(result.dense, axis=1), 1.0, atol=1e-3)
    assert all(isinstance(key, str) for weights in result.sparse for key in weights)


def test_a_non_unit_or_non_finite_dense_vector_is_refused_before_it_is_returned():
    from core.indexing.embedders.bge_m3_core import encode
    from core.indexing.embedding_profile import get_profile

    class Model:
        def __init__(self, dense):
            self._dense = dense

        def encode(self, texts, **_):
            return {"dense_vecs": np.asarray(self._dense, dtype=np.float32),
                    "lexical_weights": [{} for _ in texts]}

    profile = get_profile()
    unit = [1.0] + [0.0] * (profile.dimension - 1)
    assert encode(Model([unit]), ["ok"], profile)[0].shape == (1, profile.dimension)
    with pytest.raises(ValueError, match="unit"):
        encode(Model([[0.5] + [0.0] * (profile.dimension - 1)]), ["scaled"], profile)
    with pytest.raises(ValueError, match="finite"):
        encode(Model([[float("nan")] + [0.0] * (profile.dimension - 1)]), ["nan"], profile)


def _mixed_real_texts(count: int) -> list[str]:
    """Slices of the repository's own files: Chinese and English prose, Python, markdown, tables."""
    import random

    rng = random.Random(20261114)
    sources = [path.read_text(encoding="utf-8", errors="ignore")
               for pattern in ("*.md", "core/**/*.py", "app/**/*.py", "tests/*.py", "conf/*.yaml")
               for path in sorted(REPO.glob(pattern))[:40]]
    sources = [s for s in sources if len(s) > 400]
    texts = []
    while len(texts) < count:
        source = rng.choice(sources)
        length = rng.choice([20, 80, 300, 900, 2500])
        start = rng.randrange(0, max(1, len(source) - length))
        if source[start:start + length].strip():
            texts.append(source[start:start + length])
    return texts


def test_fp16_unit_norm_guard_never_trips_on_500_real_mixed_texts(embedder):
    from core.indexing.embedders.bge_m3_core import NORM_TOLERANCE

    texts = _mixed_real_texts(500)
    deviations = []
    for start in range(0, len(texts), 50):
        dense = embedder.embed_texts(texts[start:start + 50]).dense
        deviations.append(float(np.abs(np.linalg.norm(dense, axis=1) - 1.0).max()))
    print(f"largest fp16 norm error over 500 texts: {max(deviations):.2e}")
    assert max(deviations) < NORM_TOLERANCE / 2      # a wide margin, not a borderline pass
    assert NORM_TOLERANCE >= 1e-2
