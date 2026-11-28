"""TDD tests for reranker module."""
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from unittest.mock import patch, MagicMock


class TestRerankerImport:
    def test_import_base_reranker(self):
        from core.retrieval.reranker import BaseReranker
        assert BaseReranker is not None

    def test_import_bge_reranker(self):
        from core.retrieval.reranker import BgeReranker
        assert BgeReranker is not None

    def test_import_factory(self):
        from core.retrieval.reranker import create_reranker
        assert callable(create_reranker)

    def test_import_shared_accessor(self):
        from core.retrieval.reranker import get_shared_reranker
        assert callable(get_shared_reranker)


class TestBaseReranker:
    def test_rerank_not_implemented(self):
        from core.retrieval.reranker import BaseReranker
        with pytest.raises(NotImplementedError):
            BaseReranker().rerank("query", [])

    def test_score_pairs_not_implemented(self):
        from core.retrieval.reranker import BaseReranker
        with pytest.raises(NotImplementedError):
            BaseReranker().score_pairs("query", ["a"])


class TestBgeRerankerInterface:
    def test_default_release_batch_is_flagembedding_compatible_128(self):
        from core.retrieval.reranker import BgeReranker

        assert BgeReranker(device="cpu")._batch_size == 128

    def test_cuda_ordinal_keeps_fp16_enabled(self):
        from core.retrieval.reranker import BgeReranker

        assert BgeReranker(device="cuda:0")._use_fp16 is True

    @pytest.mark.parametrize("batch_size", [0, -1, 129])
    def test_invalid_direct_batch_size_is_rejected(self, batch_size):
        from core.retrieval.reranker import BgeReranker

        with pytest.raises(ValueError, match="batch_size"):
            BgeReranker(device="cpu", batch_size=batch_size)

    def test_negative_direct_idle_cache_budget_is_rejected(self):
        from core.retrieval.reranker import BgeReranker

        with pytest.raises(ValueError, match="idle_cache_limit_mib"):
            BgeReranker(device="cpu", idle_cache_limit_mib=-1)

    def test_init_does_not_load_model(self):
        from core.retrieval.reranker import BgeReranker
        reranker = BgeReranker()
        assert reranker._model is None

    def test_rerank_empty_returns_empty(self):
        from core.retrieval.reranker import BgeReranker
        reranker = BgeReranker()
        assert reranker.rerank("query", []) == []

    def test_rerank_calls_compute_score(self):
        from core.retrieval.reranker import BgeReranker
        from app.models.schemas import SearchResult

        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.9, 0.3, 0.7]

        reranker = BgeReranker()
        reranker._model = mock_model

        results = [
            SearchResult(doc_id="1", chunk_id="c1", content="text A", score=0.5,
                        doc_title="A", source="s", metadata={}),
            SearchResult(doc_id="2", chunk_id="c2", content="text B", score=0.6,
                        doc_title="B", source="s", metadata={}),
            SearchResult(doc_id="3", chunk_id="c3", content="text C", score=0.4,
                        doc_title="C", source="s", metadata={}),
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].score == 0.9
        assert reranked[0].doc_id == "1"
        assert reranked[1].score == 0.7
        assert reranked[1].doc_id == "3"

    def test_score_pairs_returns_float_scores_in_order(self):
        from core.retrieval.reranker import BgeReranker
        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.9, 0.1]
        reranker = BgeReranker()
        reranker._model = mock_model
        scores = reranker.score_pairs("q", ["text A", "text B"])
        assert scores == [0.9, 0.1]
        # one cross-encoder call, query paired with each text in order
        pairs = mock_model.compute_score.call_args[0][0]
        assert pairs == [["q", "text A"], ["q", "text B"]]

    def test_score_pairs_empty_returns_empty(self):
        from core.retrieval.reranker import BgeReranker
        assert BgeReranker().score_pairs("q", []) == []

    def test_load_model_uses_explicit_configured_batch_size(self, monkeypatch):
        from core.retrieval.reranker import BgeReranker

        created = []

        class FakeModel:
            def __init__(self, *_args, **kwargs):
                created.append(kwargs)

            def compute_score(self, pairs, normalize=True):
                return [1.0] * len(pairs)

        monkeypatch.setitem(
            sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=FakeModel))
        reranker = BgeReranker(device="cpu", batch_size=32)

        reranker.score_pairs("q", ["text"])

        assert created == [{"use_fp16": False, "devices": ["cpu"], "batch_size": 32}]

    def test_cuda_cold_load_applies_the_process_cap_before_reranker_construction(self, monkeypatch):
        from config import settings
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        events = []

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                events.append("model")

            def compute_score(self, pairs, normalize=True):
                return [1.0] * len(pairs)

        monkeypatch.setattr(settings.reranker, "cuda_memory_fraction", 0.25)
        monkeypatch.setattr(
            reranker_module,
            "configure_cuda_memory_fraction",
            lambda device, fraction: events.append(("cap", device, fraction)) or True,
        )
        monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=FakeModel))

        BgeReranker(device="cuda:0")._load_model()

        assert events[0] == ("cap", "cuda:0", 0.25)
        assert events[1] == "model"

    def test_cuda_ordinal_cold_load_defers_a_cache_trim(self, monkeypatch):
        """Warmup/model transfer is CUDA work and cannot race a scheduled trim."""
        from core import gpu_memory
        from core.retrieval.reranker import BgeReranker

        release = threading.Event()
        constructed = threading.Event()
        cache_trimmed = threading.Event()

        class FakeModel:
            def __init__(self, *_args, **kwargs):
                assert kwargs["use_fp16"] is True
                constructed.set()
                assert release.wait(timeout=1)

            def compute_score(self, pairs, normalize=True):
                return [1.0] * len(pairs)

        monkeypatch.setitem(
            sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=FakeModel))
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    empty_cache=cache_trimmed.set,
                    memory_stats=lambda _device: {},
                    memory_allocated=lambda _device: 0,
                    memory_reserved=lambda _device: 0,
                ),
            ),
        )

        reranker = BgeReranker(device="cuda:0")
        loader = threading.Thread(target=reranker._load_model)
        loader.start()
        assert constructed.wait(timeout=1)
        assert gpu_memory.trim_cuda_cache_and_capture_if_idle("cuda:0") == (False, None)
        assert not cache_trimmed.is_set()

        release.set()
        loader.join(timeout=1)
        assert not loader.is_alive()
        assert cache_trimmed.wait(timeout=1)

    def test_score_pairs_logs_post_trim_allocator_snapshot(self, monkeypatch, caplog):
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.return_value = [0.9]
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=0)
        reranker._model = model
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=200,
                                  inactive_split_bytes=10, num_ooms=0,
                                  num_alloc_retries=0),
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=200,
                                  inactive_split_bytes=10, num_ooms=1,
                                  num_alloc_retries=1),
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=100,
                                  inactive_split_bytes=0, num_ooms=1,
                                  num_alloc_retries=1),
        ])
        trimmed: list[bool] = []
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (trimmed.append(True) or True, next(snapshots)),
        )

        with caplog.at_level("INFO", logger="core.retrieval.reranker"):
            assert reranker.score_pairs("q", ["text"]) == [0.9]
        assert trimmed == [True]
        assert "reserved_post_trim_mib=0.0" in caplog.text
        assert reranker.last_cuda_telemetry is not None
        assert reranker.last_cuda_telemetry.trim_executed is True
        assert reranker.last_cuda_telemetry.post_trim is not None

    def test_score_pairs_trims_idle_cache_only_over_the_configured_budget(self, monkeypatch):
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.return_value = [0.9]
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=512)
        reranker._model = model
        mib = 1024 * 1024
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100 * mib, reserved_bytes=200 * mib,
                                  inactive_split_bytes=10 * mib, num_ooms=0,
                                  num_alloc_retries=0),
            CudaAllocatorSnapshot(allocated_bytes=100 * mib, reserved_bytes=700 * mib,
                                  inactive_split_bytes=10 * mib, num_ooms=0,
                                  num_alloc_retries=0),
            CudaAllocatorSnapshot(allocated_bytes=100 * mib, reserved_bytes=100 * mib,
                                  inactive_split_bytes=0, num_ooms=0,
                                  num_alloc_retries=0),
        ])
        trimmed: list[bool] = []
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (trimmed.append(True) or True, next(snapshots)),
        )

        reranker.score_pairs("q", ["text"])

        assert trimmed == [True]

    def test_score_pairs_trims_any_unused_cache_with_zero_budget(self, monkeypatch):
        """Zero is an aggressive budget, not a switch that disables maintenance."""
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.return_value = [0.9]
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=0)
        reranker._model = model
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=200,
                                  inactive_split_bytes=10, num_ooms=3,
                                  num_alloc_retries=4),
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=400,
                                  inactive_split_bytes=10, num_ooms=3,
                                  num_alloc_retries=4),
        ])
        trimmed: list[bool] = []
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (trimmed.append(True) or True, None),
        )

        reranker.score_pairs("q", ["text"])

        assert trimmed == [True]

    def test_score_pairs_keeps_small_idle_cache_without_oom_or_retry(self, monkeypatch):
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.return_value = [0.9]
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=512)
        reranker._model = model
        mib = 1024 * 1024
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100 * mib, reserved_bytes=200 * mib,
                                  inactive_split_bytes=10 * mib, num_ooms=3,
                                  num_alloc_retries=4),
            CudaAllocatorSnapshot(allocated_bytes=100 * mib, reserved_bytes=400 * mib,
                                  inactive_split_bytes=10 * mib, num_ooms=3,
                                  num_alloc_retries=4),
        ])
        trimmed: list[bool] = []
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (trimmed.append(True) or True, None),
        )

        reranker.score_pairs("q", ["text"])

        assert trimmed == []

    def test_score_pairs_keeps_scores_when_maintenance_raises(self, monkeypatch):
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.return_value = [0.9]
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=0)
        reranker._model = model
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=200,
                                  inactive_split_bytes=0, num_ooms=0,
                                  num_alloc_retries=0),
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=300,
                                  inactive_split_bytes=0, num_ooms=0,
                                  num_alloc_retries=0),
        ])
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (_ for _ in ()).throw(RuntimeError("maintenance failed")),
        )

        assert reranker.score_pairs("q", ["text"]) == [0.9]

    def test_failed_score_still_attempts_cache_maintenance_then_reraises(self, monkeypatch):
        from core.gpu_memory import CudaAllocatorSnapshot
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BgeReranker

        model = MagicMock()
        model.compute_score.side_effect = ValueError("model failed")
        reranker = BgeReranker(device="cuda", idle_cache_limit_mib=0)
        reranker._model = model
        snapshots = iter([
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=200,
                                  inactive_split_bytes=0, num_ooms=0,
                                  num_alloc_retries=0),
            CudaAllocatorSnapshot(allocated_bytes=100, reserved_bytes=300,
                                  inactive_split_bytes=0, num_ooms=0,
                                  num_alloc_retries=0),
        ])
        trimmed: list[bool] = []
        monkeypatch.setattr(reranker_module, "capture_cuda_allocator_snapshot",
                            lambda _device: next(snapshots))
        monkeypatch.setattr(
            reranker_module,
            "trim_cuda_cache_and_capture_if_idle",
            lambda _device: (trimmed.append(True) or True, None),
        )

        with pytest.raises(ValueError, match="model failed"):
            reranker.score_pairs("q", ["text"])
        assert trimmed == [True]

    def test_concurrent_first_use_loads_model_once(self, monkeypatch):
        from core.retrieval.reranker import BgeReranker

        created = []

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                time.sleep(0.05)
                created.append(self)

            def compute_score(self, pairs, normalize=True):
                return [1.0] * len(pairs)

        monkeypatch.setitem(
            sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=FakeModel))
        reranker = BgeReranker(device="cpu")
        threads = [
            threading.Thread(target=reranker.score_pairs, args=("q", ["text"]))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert len(created) == 1

    def test_rerank_adds_reranker_score_metadata(self):
        from core.retrieval.reranker import BgeReranker
        from app.models.schemas import SearchResult

        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.85]

        reranker = BgeReranker()
        reranker._model = mock_model

        results = [
            SearchResult(doc_id="1", chunk_id="c1", content="text", score=0.5,
                        doc_title="T", source="s", metadata={"method": "hybrid"}),
        ]
        reranked = reranker.rerank("q", results)
        assert reranked[0].metadata["reranker_score"] == 0.85
        assert reranked[0].metadata["method"] == "hybrid"


class TestCreateReranker:
    def test_factory_returns_bge_reranker(self):
        from core.retrieval.reranker import create_reranker, BgeReranker
        r = create_reranker()
        assert isinstance(r, BgeReranker)

    def test_factory_custom_model(self):
        from core.retrieval.reranker import create_reranker
        r = create_reranker("custom/model")
        assert r._model_name == "custom/model"

    def test_shared_accessor_reuses_one_process_instance(self, monkeypatch):
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BaseReranker

        class StubReranker(BaseReranker):
            def rerank(self, query, results, top_k=10):
                return results[:top_k]

        created = []

        def fake_create(model_name=None):
            instance = StubReranker()
            created.append(instance)
            return instance

        monkeypatch.setattr(reranker_module, "_SHARED_RERANKER", None, raising=False)
        monkeypatch.setattr(reranker_module, "create_reranker", fake_create)

        first = reranker_module.get_shared_reranker()
        second = reranker_module.get_shared_reranker()

        assert first is second
        assert created == [first]

    def test_shared_accessor_is_thread_safe(self, monkeypatch):
        from core.retrieval import reranker as reranker_module
        from core.retrieval.reranker import BaseReranker

        class StubReranker(BaseReranker):
            pass

        created = []

        def fake_create(model_name=None):
            time.sleep(0.05)
            instance = StubReranker()
            created.append(instance)
            return instance

        monkeypatch.setattr(reranker_module, "_SHARED_RERANKER", None, raising=False)
        monkeypatch.setattr(reranker_module, "create_reranker", fake_create)
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(reranker_module.get_shared_reranker()))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert len(created) == 1
        assert results and all(instance is created[0] for instance in results)
