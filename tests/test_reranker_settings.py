"""Release boundaries for reranker-only nested configuration."""
from __future__ import annotations


def test_generic_batch_size_does_not_change_reranker(monkeypatch):
    """Only RERANKER__BATCH_SIZE may change cross-encoder batching."""
    from config import Settings

    monkeypatch.setenv("BATCH_SIZE", "17")
    assert Settings().reranker.batch_size == 128


def test_nested_reranker_batch_size_is_honoured(monkeypatch):
    from config import Settings

    monkeypatch.setenv("RERANKER__BATCH_SIZE", "128")
    assert Settings().reranker.batch_size == 128


def test_reranker_cuda_memory_fraction_is_opt_in(monkeypatch):
    from config import Settings

    assert Settings().reranker.cuda_memory_fraction is None
    monkeypatch.setenv("RERANKER__CUDA_MEMORY_FRACTION", "0.35")
    assert Settings().reranker.cuda_memory_fraction == 0.35
