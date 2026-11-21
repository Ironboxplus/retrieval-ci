"""Core Qdrant filters for active version pairs in shared collections."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from core.storage.qdrant_scope_filters import build_scope_pair_filter
from core.storage.qdrant_store import QdrantStore


def test_scope_pair_filter_preserves_source_revision_pairs_and_common_filters():
    query_filter = build_scope_pair_filter(
        (
            {"source_id": "ffmpeg", "revision": "ffmpeg-commit"},
            {"source_id": "pytorch", "revision": "pytorch-commit"},
        ),
        payload_filters={"domain": ["machine-learning", "multimedia"], "language": "c"},
    )

    language = next(condition for condition in query_filter.must if condition.key == "language")
    assert language.match.value == "c"
    assert query_filter.must_not is None
    assert len(query_filter.should) == 2
    assert [(condition.key, condition.match.value) for condition in query_filter.should[0].must] == [
        ("revision", "ffmpeg-commit"),
        ("source_id", "ffmpeg"),
    ]
    assert [(condition.key, condition.match.value) for condition in query_filter.should[1].must] == [
        ("revision", "pytorch-commit"),
        ("source_id", "pytorch"),
    ]
    domain = next(condition for condition in query_filter.must if condition.key == "domain")
    assert list(domain.match.any) == ["machine-learning", "multimedia"]


def test_scope_pair_filter_rejects_empty_or_incomplete_pairs():
    with pytest.raises(ValueError, match="at least one active scope"):
        build_scope_pair_filter(())
    with pytest.raises(ValueError, match="same fields"):
        build_scope_pair_filter(({"source_id": "ffmpeg"}, {"source_id": "pytorch", "revision": "x"}))


def test_qdrant_store_uses_an_explicit_scope_filter_without_rebuilding_it():
    query_filter = build_scope_pair_filter(
        ({"source_id": "ffmpeg", "revision": "commit"},),
    )

    class Client:
        received_filter = None

        def query_points(self, *_args, **kwargs):
            self.received_filter = kwargs["query_filter"]
            return SimpleNamespace(
                points=[SimpleNamespace(id=1, score=0.9, payload={"content": "match"})]
            )

    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "public_code_chunks"
    store.client = Client()

    results = store.search([0.1], top_k=1, query_filter=query_filter)

    assert results[0]["content"] == "match"
    assert store.client.received_filter is query_filter
