"""Contract for Journal's explicit isolated-Qdrant collection override."""
from __future__ import annotations

from app.services.journal.vector_store import JournalVectorStore


class _Boundary:
    """No client methods are touched during construction."""


def test_journal_vector_store_uses_explicit_isolated_collection(monkeypatch):
    monkeypatch.setenv("JOURNAL_QDRANT_COLLECTION", "journal_bulk_blackbox_4e3d")
    boundary = _Boundary()
    store = JournalVectorStore(store=boundary)  # type: ignore[arg-type]

    assert store.collection_name == "journal_bulk_blackbox_4e3d"
    assert store.store is boundary


def test_journal_vector_store_preserves_production_default(monkeypatch):
    monkeypatch.delenv("JOURNAL_QDRANT_COLLECTION", raising=False)

    assert JournalVectorStore(store=_Boundary()).collection_name == "journal_chunks"  # type: ignore[arg-type]
