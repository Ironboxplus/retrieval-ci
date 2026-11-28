"""TDD: Verify all legacy code is cleaned up.

Asserts that:
1. main.py doesn't initialize old retrievers (saves ~6.6GB disk I/O on startup)
2. No production code imports obsolete modules
3. Obsolete storage/retriever files are deleted
4. /api/search routes ALL methods through SearchPipeline
"""
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestMainNoOldRetrievers(unittest.TestCase):
    """main.py should not initialize old retrievers that load duplicate models."""

    def _read_main(self):
        return (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    def test_no_vector_retriever_init(self):
        src = self._read_main()
        self.assertNotIn("VectorRetriever()", src)

    def test_no_enhanced_retriever_init(self):
        src = self._read_main()
        self.assertNotIn("EnhancedRetriever()", src)

    def test_no_hybrid_retriever_init(self):
        src = self._read_main()
        self.assertNotIn("HybridRetriever(", src)

    def test_no_old_retriever_imports(self):
        src = self._read_main()
        self.assertNotIn("VectorRetriever", src)
        self.assertNotIn("HybridRetriever", src)
        self.assertNotIn("EnhancedRetriever", src)

    def test_search_all_methods_use_pipeline(self):
        """All /api/search methods should go through search_pipeline."""
        import ast
        src = self._read_main()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = getattr(node, 'attr', '')
                self.assertNotIn(name, ['vector_retriever', 'hybrid_retriever', 'enhanced_retriever'],
                                 f"main.py still references old retriever: {name}")

    def test_no_multi_method_endpoint(self):
        """/api/search/multi-method used old retrievers — should be removed."""
        src = self._read_main()
        self.assertNotIn("multi-method", src)


class TestObsoleteFilesDeleted(unittest.TestCase):
    """Obsolete storage/retriever files should be deleted."""

    def test_chroma_store_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "storage" / "chroma_store.py").exists(),
            "core/storage/chroma_store.py should be deleted")

    def test_whoosh_store_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "storage" / "whoosh_store.py").exists(),
            "core/storage/whoosh_store.py should be deleted")

    def test_old_embedder_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "indexing" / "embedder.py").exists(),
            "core/indexing/embedder.py should be deleted")

    def test_index_manager_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "indexing" / "index_manager.py").exists(),
            "core/indexing/index_manager.py should be deleted")

    def test_vector_retriever_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "retrieval" / "vector_retriever.py").exists(),
            "core/retrieval/vector_retriever.py should be deleted")

    def test_hybrid_retriever_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "retrieval" / "hybrid_retriever.py").exists(),
            "core/retrieval/hybrid_retriever.py should be deleted")

    def test_enhanced_retriever_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "core" / "retrieval" / "enhanced_retriever.py").exists(),
            "core/retrieval/enhanced_retriever.py should be deleted")


class TestNoProductionImportsOfObsolete(unittest.TestCase):
    """No production code should import deleted modules."""

    def test_index_service_no_old_embedder(self):
        src = (PROJECT_ROOT / "app" / "services" / "index_service.py").read_text(encoding="utf-8")
        self.assertNotIn("from core.indexing.embedder import", src)

    def test_retrieval_init_no_old_exports(self):
        init_path = PROJECT_ROOT / "core" / "retrieval" / "__init__.py"
        if init_path.exists():
            src = init_path.read_text(encoding="utf-8")
            self.assertNotIn("VectorRetriever", src)
            self.assertNotIn("HybridRetriever", src)
            self.assertNotIn("EnhancedRetriever", src)


class TestObsoleteScriptsCleanedOrDeleted(unittest.TestCase):
    """Old scripts referencing ChromaStore/WhooshStore/old Embedder should be removed."""

    def test_rebuild_chroma_index_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "scripts" / "rebuild_chroma_index.py").exists(),
            "scripts/rebuild_chroma_index.py references deleted ChromaStore")

    def test_obsolete_tests_deleted(self):
        self.assertFalse(
            (PROJECT_ROOT / "tests" / "test_filter_fix.py").exists(),
            "tests/test_filter_fix.py references deleted ChromaStore/WhooshStore")


if __name__ == "__main__":
    unittest.main()
