"""Reranker module: the abstract contract, the factory and the process-wide accessor.

The reranker is a client of the vLLM reranking service (VllmReranker); the torch BgeReranker is
retired (tests/test_retired_inference_paths.py). The client's own behavior is tested in
tests/test_vllm_backend.py (no service needed) and tests/test_vllm_backend_services.py (a real
service: scores, order, truncation, cancel).
"""
import threading
import time

import pytest


class TestRerankerImport:
    def test_import_base_reranker(self):
        from core.retrieval.reranker import BaseReranker
        assert BaseReranker is not None

    def test_import_service_client(self):
        from core.retrieval.reranker import VllmReranker
        assert VllmReranker is not None

    def test_import_retired_torch_reranker(self):
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


class TestVllmRerankerInterface:
    def test_default_request_size_is_32_pairs(self):
        from core.retrieval.reranker import VllmReranker

        assert VllmReranker()._batch_size == 32

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_invalid_batch_size_is_rejected(self, batch_size):
        from core.retrieval.reranker import VllmReranker

        with pytest.raises(ValueError, match="batch_size"):
            VllmReranker(batch_size=batch_size)

    def test_init_makes_no_request(self):
        from core.retrieval.reranker import VllmReranker

        VllmReranker(base_url="http://127.0.0.1:1")          # nothing listens there: construction must not connect

    def test_rerank_empty_returns_empty(self):
        from core.retrieval.reranker import VllmReranker

        assert VllmReranker(base_url="http://127.0.0.1:1").rerank("query", []) == []

    def test_score_pairs_empty_returns_empty(self):
        from core.retrieval.reranker import VllmReranker

        assert VllmReranker(base_url="http://127.0.0.1:1").score_pairs("q", []) == []


class TestCreateReranker:
    def test_factory_returns_the_service_client(self):
        from core.retrieval.reranker import VllmReranker, create_reranker
        r = create_reranker()
        assert isinstance(r, VllmReranker)

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
