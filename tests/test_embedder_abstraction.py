"""TDD tests for embedder abstraction layer and BGE-M3 implementation."""
import sys
import threading
import time
import unittest
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_interactive_embedding_runs_before_next_background_batch():
    """Once an interactive query is waiting, a background indexer must yield the
    shared model after its current batch instead of immediately taking another batch."""
    from core.indexing.embedders import embedding_slot

    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def first_background():
        with embedding_slot("background"):
            order.append("background-1")
            first_started.set()
            assert release_first.wait(3)

    def interactive():
        with embedding_slot("interactive"):
            order.append("interactive")

    def second_background():
        with embedding_slot("background"):
            order.append("background-2")

    first = threading.Thread(target=first_background)
    query = threading.Thread(target=interactive)
    second = threading.Thread(target=second_background)
    first.start()
    assert first_started.wait(1)
    query.start()
    time.sleep(0.05)
    second.start()
    release_first.set()
    for thread in (first, query, second):
        thread.join(timeout=3)

    assert order == ["background-1", "interactive", "background-2"]


def test_background_embedding_eventually_runs_under_continuous_interactive_waiters():
    from core.indexing.embedders import _EmbeddingScheduler

    scheduler = _EmbeddingScheduler(max_interactive_burst=2)
    release_active = threading.Event()
    active_started = threading.Event()
    order: list[str] = []

    def active_interactive():
        with scheduler.acquire("interactive"):
            active_started.set()
            assert release_active.wait(3)

    def background():
        with scheduler.acquire("background"):
            order.append("background")

    def interactive(index):
        with scheduler.acquire("interactive"):
            order.append(f"interactive-{index}")

    active = threading.Thread(target=active_interactive)
    active.start()
    assert active_started.wait(1)
    background_thread = threading.Thread(target=background)
    interactive_threads = [
        threading.Thread(target=interactive, args=(index,)) for index in range(3)
    ]
    background_thread.start()
    for thread in interactive_threads:
        thread.start()
    time.sleep(0.05)
    release_active.set()
    for thread in [active, background_thread, *interactive_threads]:
        thread.join(timeout=3)

    assert order.index("background") == 2


def test_bge_m3_cuda_ordinal_cold_load_defers_a_cache_trim(monkeypatch):
    """Constructor + warmup are covered by the allocator activity gate too."""
    from core import gpu_memory
    from core.indexing.embedders import bge_m3 as bge_m3_module

    release = threading.Event()
    constructed = threading.Event()
    created_kwargs = []

    class FakeBgeModel:
        def __init__(self, *_args, **kwargs):
            created_kwargs.append(kwargs)
            constructed.set()
            assert release.wait(timeout=1)

        def encode(self, _texts, **_kwargs):
            return {
                "dense_vecs": [[0.0] * 1024],
                "lexical_weights": [{}],
            }

    monkeypatch.setattr(bge_m3_module, "BGEM3FlagModel", FakeBgeModel)
    cache_trimmed = threading.Event()
    monkeypatch.setattr(
        bge_m3_module,
        "request_cuda_cache_trim_if_idle",
        lambda _device: cache_trimmed.set() or True,
    )

    created = []
    loader = threading.Thread(
        target=lambda: created.append(bge_m3_module.BgeM3Embedder(device="cuda:0")),
    )
    loader.start()
    assert constructed.wait(timeout=1)

    # Use the same global gate rather than a fake local lock: a trim request
    # must queue while the constructor/warmup is in progress.
    assert gpu_memory._GPU_ALLOCATOR_COORDINATOR.request_trim(cache_trimmed.set) is False
    assert not cache_trimmed.is_set()

    release.set()
    loader.join(timeout=1)
    assert not loader.is_alive()
    assert len(created) == 1
    assert created_kwargs[0]["use_bf16"] is True
    assert cache_trimmed.wait(timeout=1)


def test_create_embedder_applies_the_process_cap_before_model_construction(monkeypatch):
    from config import settings
    from core.indexing import embedders
    from core.indexing.embedders import bge_m3 as bge_m3_module

    events: list[tuple[str, object]] = []

    class FakeBgeModel:
        def __init__(self, *_args, **_kwargs):
            events.append(("model", None))

        def encode(self, _texts, **_kwargs):
            return {"dense_vecs": [[0.0] * 1024], "lexical_weights": [{}]}

    monkeypatch.setattr(settings.reranker, "cuda_memory_fraction", 0.25)
    monkeypatch.setattr(
        embedders,
        "configure_cuda_memory_fraction",
        lambda device, fraction: events.append(("cap", (device, fraction))) or True,
    )
    monkeypatch.setattr(bge_m3_module, "BGEM3FlagModel", FakeBgeModel)
    monkeypatch.setattr(bge_m3_module, "trim_cuda_cache_if_idle", lambda _device: False)

    embedders.create_embedder("BAAI/bge-m3", device="cuda:0")

    assert events[0] == ("cap", ("cuda:0", 0.25))
    assert events[1] == ("model", None)


def test_create_embedder_does_not_apply_a_cuda_cap_to_an_explicit_cpu_probe(monkeypatch):
    from config import settings
    from core.indexing import embedders
    from core.indexing.embedders import bge_m3 as bge_m3_module

    class FakeBgeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, _texts, **_kwargs):
            return {"dense_vecs": [[0.0] * 1024], "lexical_weights": [{}]}

    monkeypatch.setattr(settings.reranker, "cuda_memory_fraction", 0.25)
    monkeypatch.setattr(
        embedders,
        "configure_cuda_memory_fraction",
        lambda *_args: (_ for _ in ()).throw(AssertionError("CPU must not configure CUDA")),
    )
    monkeypatch.setattr(bge_m3_module, "BGEM3FlagModel", FakeBgeModel)

    embedders.create_embedder("BAAI/bge-m3", device="cpu")


def test_bge_m3_embedding_releases_idle_cuda_cache_after_encode(monkeypatch):
    """Embedding-only requests must not leave a cache high-water mark behind."""
    from core.indexing.embedders import bge_m3 as bge_m3_module

    events: list[object] = []

    class FakeBgeModel:
        def encode(self, _texts, **_kwargs):
            events.append("encode")
            return {"dense_vecs": [[1.0] * 1024], "lexical_weights": [{7: 0.5}]}

    embedder = object.__new__(bge_m3_module.BgeM3Embedder)
    embedder._device = "cuda:0"
    embedder._model = FakeBgeModel()
    monkeypatch.setattr(
        bge_m3_module,
        "request_cuda_cache_trim_if_idle",
        lambda device: events.append(("trim", device)) or True,
    )

    result = embedder.embed_texts(["allocator cache"])

    assert result.dense.shape == (1, 1024)
    assert result.sparse == [{"7": 0.5}]
    assert events == ["encode", ("trim", "cuda:0")]


def test_sentence_transformer_embedding_releases_idle_cuda_cache_after_encode(monkeypatch):
    """The dense-only implementation has the same steady-memory contract."""
    from core.indexing.embedders import sentence_transformer as sentence_module

    events: list[object] = []

    class FakeSentenceModel:
        def encode(self, _texts, **_kwargs):
            events.append("encode")
            return [[1.0, 2.0]]

    embedder = object.__new__(sentence_module.SentenceTransformerEmbedder)
    embedder._device = "cuda:0"
    embedder._model = FakeSentenceModel()
    monkeypatch.setattr(
        sentence_module,
        "request_cuda_cache_trim_if_idle",
        lambda device: events.append(("trim", device)) or True,
    )

    result = embedder.embed_texts(["allocator cache"])

    assert result.dense.tolist() == [[1.0, 2.0]]
    assert result.sparse is None
    assert events == ["encode", ("trim", "cuda:0")]


def test_embedding_preserves_the_original_encode_error_when_maintenance_fails(monkeypatch):
    """A failed cache release must neither hide nor replace a model failure."""
    from core import gpu_memory
    from core.indexing.embedders import bge_m3 as bge_m3_module

    class FailingBgeModel:
        def encode(self, _texts, **_kwargs):
            raise ValueError("encode failed")

    embedder = object.__new__(bge_m3_module.BgeM3Embedder)
    embedder._device = "cuda:0"
    embedder._model = FailingBgeModel()
    maintenance_attempted: list[str] = []

    def failing_trim(_device):
        maintenance_attempted.append("trim")
        raise RuntimeError("maintenance failed")

    monkeypatch.setattr(
        gpu_memory,
        "trim_cuda_cache_if_idle",
        failing_trim,
    )

    try:
        embedder.embed_texts(["allocator cache"])
    except ValueError as exc:
        assert str(exc) == "encode failed"
    else:
        raise AssertionError("the original encode error must be preserved")
    assert maintenance_attempted == ["trim"]


def test_embedding_result_survives_a_cache_maintenance_failure(monkeypatch):
    """Post-encode cleanup is best effort and cannot turn a 200 into a 500."""
    from core import gpu_memory
    from core.indexing.embedders import bge_m3 as bge_m3_module

    class FakeBgeModel:
        def encode(self, _texts, **_kwargs):
            return {"dense_vecs": [[1.0] * 1024], "lexical_weights": [{}]}

    embedder = object.__new__(bge_m3_module.BgeM3Embedder)
    embedder._device = "cuda:0"
    embedder._model = FakeBgeModel()
    monkeypatch.setattr(
        gpu_memory,
        "trim_cuda_cache_if_idle",
        lambda _device: (_ for _ in ()).throw(RuntimeError("maintenance failed")),
    )

    result = embedder.embed_texts(["allocator cache"])

    assert result.dense.shape == (1, 1024)
    assert result.sparse == [{}]


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

    def test_create_sentence_transformer(self):
        from core.indexing.embedders import create_embedder, BaseEmbedder
        embedder = create_embedder("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(embedder, BaseEmbedder)
        self.assertEqual(embedder.dimension, 384)
        self.assertFalse(embedder.supports_sparse)

    def test_create_bge_m3(self):
        from core.indexing.embedders import create_embedder, BaseEmbedder
        embedder = create_embedder("BAAI/bge-m3")
        self.assertIsInstance(embedder, BaseEmbedder)
        self.assertEqual(embedder.dimension, 1024)
        self.assertTrue(embedder.supports_sparse)


class TestSentenceTransformerEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.indexing.embedders.sentence_transformer import SentenceTransformerEmbedder
        cls.embedder = SentenceTransformerEmbedder()

    def test_embed_returns_result(self):
        from core.indexing.embedders import EmbeddingResult
        result = self.embedder.embed_texts(["hello world"])
        self.assertIsInstance(result, EmbeddingResult)

    def test_dense_shape(self):
        result = self.embedder.embed_texts(["hello", "world"])
        self.assertEqual(result.dense.shape[0], 2)
        self.assertEqual(result.dense.shape[1], self.embedder.dimension)

    def test_sparse_is_none(self):
        result = self.embedder.embed_texts(["test"])
        self.assertIsNone(result.sparse)


class TestBgeM3Embedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from core.indexing.embedders.bge_m3 import BgeM3Embedder
            cls.embedder = BgeM3Embedder()
            cls.available = True
        except Exception:
            cls.available = False

    def test_model_loads(self):
        if not self.available:
            self.skipTest("BGE-M3 model not available locally")
        self.assertEqual(self.embedder.dimension, 1024)

    def test_embed_returns_dense_and_sparse(self):
        if not self.available:
            self.skipTest("BGE-M3 model not available locally")
        from core.indexing.embedders import EmbeddingResult
        result = self.embedder.embed_texts(["attention mechanism in transformers"])
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.dense.shape[1], 1024)
        self.assertIsNotNone(result.sparse)
        self.assertIsInstance(result.sparse[0], dict)
        self.assertGreater(len(result.sparse[0]), 0)

    def test_supports_sparse(self):
        if not self.available:
            self.skipTest("BGE-M3 model not available locally")
        self.assertTrue(self.embedder.supports_sparse)


if __name__ == "__main__":
    unittest.main()
