"""TDD tests for SQLite FTS5 full-text store replacing Whoosh."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import Chunk


def _make_chunks(doc_id: str, texts: list[str], level: int = 2) -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}_c{i}", doc_id=doc_id, content=t, level=level,
              start_idx=0, end_idx=len(t), metadata={})
        for i, t in enumerate(texts)
    ]


class TestFtsStoreImport(unittest.TestCase):
    def test_can_import(self):
        from core.storage.fts_store import FtsStore
        self.assertTrue(callable(FtsStore))


class TestFtsStoreBasic(unittest.TestCase):
    def setUp(self):
        from core.storage.fts_store import FtsStore
        self.db_path = Path(tempfile.mktemp(suffix=".db"))
        self.store = FtsStore(db_path=str(self.db_path))

    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_add_and_search(self):
        chunks = _make_chunks("doc1", [
            "Vision transformers have revolutionized image classification",
            "Convolutional neural networks remain competitive for detection",
        ])
        self.store.add_document("doc1", chunks, title="Test Paper")
        results = self.store.search("vision transformer", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertIn("vision", results[0]["content"].lower())

    def test_search_returns_score(self):
        chunks = _make_chunks("doc1", ["reinforcement learning for robotics"])
        self.store.add_document("doc1", chunks)
        results = self.store.search("reinforcement learning", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertIn("score", results[0])
        self.assertIsInstance(results[0]["score"], float)

    def test_search_no_results(self):
        chunks = _make_chunks("doc1", ["deep learning for vision"])
        self.store.add_document("doc1", chunks)
        results = self.store.search("quantum computing entanglement", top_k=5)
        self.assertEqual(len(results), 0)

    def test_delete_document(self):
        chunks = _make_chunks("doc1", ["attention mechanism in transformers"])
        self.store.add_document("doc1", chunks)
        self.store.delete_document("doc1")
        results = self.store.search("attention", top_k=5)
        self.assertEqual(len(results), 0)

    def test_upsert_replaces(self):
        chunks1 = _make_chunks("doc1", ["old content about graphs"])
        self.store.add_document("doc1", chunks1)
        chunks2 = _make_chunks("doc1", ["new content about transformers"])
        self.store.add_document("doc1", chunks2)
        results = self.store.search("transformers", top_k=5)
        self.assertGreater(len(results), 0)
        old = self.store.search("graphs", top_k=5)
        self.assertEqual(len(old), 0)


class TestFtsStoreMultiDoc(unittest.TestCase):
    def setUp(self):
        from core.storage.fts_store import FtsStore
        self.db_path = Path(tempfile.mktemp(suffix=".db"))
        self.store = FtsStore(db_path=str(self.db_path))

    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_multiple_docs(self):
        self.store.add_document("doc1", _make_chunks("doc1", ["graph neural networks"]))
        self.store.add_document("doc2", _make_chunks("doc2", ["language model pretraining"]))
        r1 = self.store.search("graph neural", top_k=5)
        r2 = self.store.search("language model", top_k=5)
        self.assertGreater(len(r1), 0)
        self.assertGreater(len(r2), 0)
        self.assertEqual(r1[0]["doc_id"], "doc1")
        self.assertEqual(r2[0]["doc_id"], "doc2")

    def test_count(self):
        self.store.add_document("d1", _make_chunks("d1", ["a", "b"]))
        self.store.add_document("d2", _make_chunks("d2", ["c"]))
        self.assertEqual(self.store.count(), 3)


class TestFtsStoreResultFields(unittest.TestCase):
    """Search results must include all fields needed by BM25Retriever."""

    def setUp(self):
        from core.storage.fts_store import FtsStore
        self.db_path = Path(tempfile.mktemp(suffix=".db"))
        self.store = FtsStore(db_path=str(self.db_path))
        chunks = _make_chunks("doc1", ["attention mechanism in transformers"], level=2)
        self.store.add_document("doc1", chunks, title="Test Paper",
                                source_type="paper", data_source="system")

    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_result_has_title(self):
        results = self.store.search("attention", top_k=5)
        self.assertIn("title", results[0])
        self.assertEqual(results[0]["title"], "Test Paper")

    def test_result_has_level(self):
        results = self.store.search("attention", top_k=5)
        self.assertIn("level", results[0])
        self.assertEqual(results[0]["level"], 2)

    def test_result_has_source_type(self):
        results = self.store.search("attention", top_k=5)
        self.assertIn("source_type", results[0])
        self.assertEqual(results[0]["source_type"], "paper")

    def test_result_has_data_source(self):
        results = self.store.search("attention", top_k=5)
        self.assertIn("data_source", results[0])
        self.assertEqual(results[0]["data_source"], "system")


class TestFtsStoreFiltering(unittest.TestCase):
    def setUp(self):
        from core.storage.fts_store import FtsStore
        self.db_path = Path(tempfile.mktemp(suffix=".db"))
        self.store = FtsStore(db_path=str(self.db_path))
        self.store.add_document("doc1",
            _make_chunks("doc1", ["transformers for vision"], level=2),
            source_type="paper")
        self.store.add_document("doc2",
            _make_chunks("doc2", ["transformers for NLP"], level=3),
            source_type="webpage")

    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_filter_by_source_type(self):
        results = self.store.search("transformers", top_k=10, source_type="paper")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc1")

    def test_filter_by_level(self):
        results = self.store.search("transformers", top_k=10, level=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc2")


if __name__ == "__main__":
    unittest.main()
