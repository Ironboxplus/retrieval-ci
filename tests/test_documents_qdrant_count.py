"""Unit coverage for the Qdrant count used by document index-status."""


def test_qdrant_paper_document_count_uses_canonical_paper_id_once():
    """Paper rows must avoid the historical mixed-type ``doc_id`` payload."""
    from app.api.routes.documents import _count_qdrant_points_for_document

    class CountClient:
        def __init__(self):
            self.values = []

        def count(self, collection_name, *, count_filter, exact):
            self.values.append(count_filter.must[0].match.value)
            assert collection_name == "papers"
            assert exact is True
            return type("Count", (), {"count": 17})()

    store = type("Store", (), {"client": CountClient(), "collection_name": "papers"})()
    assert _count_qdrant_points_for_document(
        store,
        doc_id=102394,
        paper_id="iclr-2024-wg8NPfeMF9",
    ) == 17
    assert store.client.values == ["iclr-2024-wg8NPfeMF9"]


def test_qdrant_nonpaper_document_count_uses_normalized_doc_id_once():
    """Non-paper documents retain the normal string ``doc_id`` lookup path."""
    from app.api.routes.documents import _count_qdrant_points_for_document

    class CountClient:
        def __init__(self):
            self.conditions = []

        def count(self, collection_name, *, count_filter, exact):
            condition = count_filter.must[0]
            self.conditions.append((condition.key, condition.match.value, exact))
            return type("Count", (), {"count": 3})()

    store = type("Store", (), {"client": CountClient(), "collection_name": "papers"})()
    assert _count_qdrant_points_for_document(store, doc_id=102394, paper_id=None) == 3
    assert store.client.conditions == [("doc_id", "102394", True)]
