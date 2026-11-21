"""Embedder abstraction layer and the vLLM-backed BGE-M3 embedder.

The in-process embedders (BgeM3Embedder, SentenceTransformerEmbedder) are retired and tested in
tests/test_retired_inference_paths.py; the ten-second background wait and the request shape are in
tests/test_vllm_backend.py and tests/test_vllm_backend_services.py. This file keeps the abstraction
contract and the embedder against a real service.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.vllm_support import EMBED_ENV, service_url


def test_public_embed_texts_function_uses_the_process_shared_embedder(monkeypatch):
    import numpy as np
    from core.indexing import embedders

    class SharedEmbedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts, *, priority):
            self.calls.append((texts, priority))
            return embedders.EmbeddingResult(
                dense=np.asarray([[1.0, 2.0]], dtype=np.float32),
                sparse=[{"7": 0.5}],
            )

    shared = SharedEmbedder()
    monkeypatch.setattr(embedders, "get_embedder", lambda: shared)

    result = embedders.embed_texts(["kernel"], priority="background")

    assert shared.calls == [(["kernel"], "background")]
    assert result.dense.tolist() == [[1.0, 2.0]]
    assert result.sparse == [{"7": 0.5}]


class TestEmbeddingResult(unittest.TestCase):
    def test_import(self):
        from core.indexing.embedders import EmbeddingResult
        self.assertTrue(callable(EmbeddingResult))

    def test_dense_only(self):
        import numpy as np
        from core.indexing.embedders import EmbeddingResult
        r = EmbeddingResult(dense=np.zeros((2, 1024)), sparse=None)
        self.assertEqual(r.dense.shape, (2, 1024))
        self.assertIsNone(r.sparse)

    def test_dense_and_sparse(self):
        import numpy as np
        from core.indexing.embedders import EmbeddingResult
        sparse = [{"hello": 0.5, "world": 0.3}, {"test": 0.9}]
        r = EmbeddingResult(dense=np.zeros((2, 1024)), sparse=sparse)
        self.assertEqual(len(r.sparse), 2)


class TestBaseEmbedder(unittest.TestCase):
    def test_is_abstract(self):
        from core.indexing.embedders import BaseEmbedder
        with self.assertRaises(TypeError):
            BaseEmbedder()

    def test_has_required_interface(self):
        from core.indexing.embedders import BaseEmbedder
        self.assertTrue(hasattr(BaseEmbedder, "embed_texts"))
        self.assertTrue(hasattr(BaseEmbedder, "dimension"))
        self.assertTrue(hasattr(BaseEmbedder, "supports_sparse"))


class TestCreateEmbedder(unittest.TestCase):
    def test_factory_function_exists(self):
        from core.indexing.embedders import create_embedder
        self.assertTrue(callable(create_embedder))

    def test_create_bge_m3_is_the_service_client(self):
        from core.indexing.embedders import BaseEmbedder, VllmBgeM3Embedder, create_embedder
        embedder = create_embedder("BAAI/bge-m3")
        self.assertIsInstance(embedder, BaseEmbedder)
        self.assertIsInstance(embedder, VllmBgeM3Embedder)
        self.assertEqual(embedder.dimension, 1024)
        self.assertTrue(embedder.supports_sparse)

    def test_a_model_that_is_not_served_reaches_the_retired_embedder(self):
        from core.indexing.embedders import create_embedder
        from core.inference_retired import RetiredInferencePath
        with self.assertRaises(RetiredInferencePath):
            create_embedder("sentence-transformers/all-MiniLM-L6-v2")


def test_embedder_against_a_real_service_returns_dense_and_sparse():
    from core.indexing.embedders import EmbeddingResult, VllmBgeM3Embedder

    embedder = VllmBgeM3Embedder(base_url=service_url(EMBED_ENV))
    result = embedder.embed_texts(["attention mechanism in transformers"], priority="interactive")

    assert isinstance(result, EmbeddingResult)
    assert result.dense.shape == (1, embedder.dimension) == (1, 1024)
    assert embedder.supports_sparse
    assert isinstance(result.sparse[0], dict) and len(result.sparse[0]) > 0


if __name__ == "__main__":
    unittest.main()
