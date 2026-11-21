"""The module-level ``embed_texts`` is one call of the shared embedder, whatever the priority.

It used to cut background work into 8-text pieces so a waiting query could take the in-process
model between them. The model is a vLLM service now (priority is a request field, the client
batches), so nothing is split here; the ten-second background wait is tested in
tests/test_vllm_backend.py.
"""
import numpy as np
import pytest

from core.indexing import embedders


@pytest.mark.parametrize("sparse_enabled", [True, False])
def test_background_embedding_is_one_call_and_keeps_outputs_aligned(monkeypatch, sparse_enabled):
    calls = []

    class Model:
        def embed_texts(self, texts, *, priority):
            calls.append((list(texts), priority))
            return embedders.EmbeddingResult(
                dense=np.asarray([[int(t), -int(t)] for t in texts], dtype=np.float32),
                sparse=[{t: float(t)} for t in texts] if sparse_enabled else None,
            )

    monkeypatch.setattr(embedders, "get_embedder", lambda: Model())
    texts = [str(i) for i in range(19)]
    result = embedders.embed_texts(texts, priority="background")

    assert calls == [(texts, "background")]
    np.testing.assert_array_equal(result.dense, [[i, -i] for i in range(19)])
    assert result.dense.dtype == np.float32
    assert result.sparse == ([{t: float(t)} for t in texts] if sparse_enabled else None)


def test_interactive_embedding_is_one_shared_embedder_call(monkeypatch):
    calls = []
    expected = embedders.EmbeddingResult(np.zeros((19, 2)), None)

    class Model:
        def embed_texts(self, texts, *, priority):
            calls.append((len(texts), priority))
            return expected

    monkeypatch.setattr(embedders, "get_embedder", lambda: Model())
    assert embedders.embed_texts(["query"] * 19) is expected
    assert calls == [(19, "interactive")]


def test_background_failure_propagates_without_a_partial_result(monkeypatch):
    failure = RuntimeError("service failed")

    class Model:
        def embed_texts(self, texts, *, priority):
            raise failure

    monkeypatch.setattr(embedders, "get_embedder", lambda: Model())
    with pytest.raises(RuntimeError) as exc:
        embedders.embed_texts(["chunk"] * 19, priority="background")
    assert exc.value is failure
