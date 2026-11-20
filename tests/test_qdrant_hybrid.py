"""TDD tests for Qdrant hybrid search (dense + sparse named vectors)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


class TestQdrantStoreHybridConfig(unittest.TestCase):
    """Test that QdrantStore can be configured for hybrid (dense + sparse)."""

    def test_supports_sparse_property(self):
        from core.storage.qdrant_store import QdrantStore
        store = QdrantStore.__new__(QdrantStore)
        self.assertTrue(hasattr(store, "supports_sparse"))

    def test_add_chunks_accepts_sparse_vectors(self):
        from core.storage.qdrant_store import QdrantStore
        import inspect
        sig = inspect.signature(QdrantStore.add_chunks)
        self.assertIn("sparse_vectors", sig.parameters)


class TestSparseVectorConversion(unittest.TestCase):
    """Test conversion of sparse dicts to Qdrant SparseVector format."""

    def test_convert_sparse_dict(self):
        from core.storage.qdrant_store import sparse_dict_to_qdrant
        sparse = {"123": 0.5, "456": 0.3, "789": 0.8}
        result = sparse_dict_to_qdrant(sparse)
        self.assertEqual(len(result.indices), 3)
        self.assertEqual(len(result.values), 3)

    def test_empty_sparse_dict(self):
        from core.storage.qdrant_store import sparse_dict_to_qdrant
        result = sparse_dict_to_qdrant({})
        self.assertEqual(len(result.indices), 0)

    def test_sparse_values_are_float(self):
        from core.storage.qdrant_store import sparse_dict_to_qdrant
        result = sparse_dict_to_qdrant({"1": 0.5})
        self.assertIsInstance(result.values[0], float)


class TestSearchPipelineInterface(unittest.TestCase):
    """Test the SearchPipeline orchestration layer."""

    def test_import_pipeline(self):
        from core.retrieval.pipeline import SearchPipeline
        self.assertTrue(callable(SearchPipeline))

    def test_pipeline_has_search_method(self):
        from core.retrieval.pipeline import SearchPipeline
        self.assertTrue(hasattr(SearchPipeline, "search"))


if __name__ == "__main__":
    unittest.main()
