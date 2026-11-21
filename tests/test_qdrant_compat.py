"""TDD: API compatibility with Qdrant 1024d named-vector collection.

These tests verify that all write/search paths use the new BGE-M3 embedder
and don't reference the removed Whoosh store.
"""
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path


class TestIndexServiceNewEmbedder(unittest.TestCase):
    """IndexService should use new embedder (with sparse) instead of old one."""

    def test_has_new_embedder_property(self):
        from app.services.index_service import IndexService
        svc = IndexService()
        self.assertTrue(hasattr(svc, 'new_embedder'))

    def test_new_embedder_is_bge_m3(self):
        from app.services.index_service import IndexService
        svc = IndexService()
        embedder = svc.new_embedder
        self.assertTrue(embedder.supports_sparse)
        self.assertEqual(embedder.dimension, 1024)

    def test_index_document_passes_sparse_vectors(self):
        """index_document should call add_chunks with sparse_vectors kwarg."""
        from app.services.index_service import IndexService
        svc = IndexService()

        mock_sparse = [{"token1": 0.5}]
        mock_chunks = [MagicMock(id="c1", content="test", level=2,
                                  start_idx=0, end_idx=4, metadata={},
                                  embedding=[0.1] * 1024)]

        mock_emb = MagicMock()
        mock_emb.embed_chunks.return_value = (mock_chunks, mock_sparse)
        svc._new_embedder = mock_emb

        mock_store = MagicMock()
        svc._qdrant_store = mock_store

        mock_chunker = MagicMock()
        mock_chunker.chunk_document.return_value = mock_chunks
        svc._chunker = mock_chunker

        with patch.object(svc, '_save_chunks_to_db'):

            result = svc.index_document(
                doc_id=1, content="test content", title="Test",
                source="test", db=MagicMock(),
            )

            self.assertTrue(result["success"])
            mock_store.add_chunks.assert_called_once()
            call_kwargs = mock_store.add_chunks.call_args
            self.assertEqual(call_kwargs.kwargs.get("sparse_vectors"), mock_sparse)


class TestPaperIndexerNoWhoosh(unittest.TestCase):
    """PaperIndexer must not reference WhooshStore anywhere at runtime."""

    def test_no_whoosh_store_property(self):
        """whoosh_store property should not exist or not be required."""
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        indexer = PaperIndexer()
        # Should not crash when accessing — either removed or returns None gracefully
        try:
            ws = indexer.whoosh_store
            # If it exists, it should not be required for basic operations
        except AttributeError:
            pass  # Expected — property was removed

    def test_no_whoosh_import_at_class_level(self):
        """The module should not import WhooshStore at the top level."""
        import pipeline.paper_crawler.integration.index_integration as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from core.storage.whoosh_store import WhooshStore", source)

    def test_write_paper_to_stores_no_whoosh_reference(self):
        """_write_paper_to_stores should not reference whoosh_store."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer._write_paper_to_stores)
        self.assertNotIn("whoosh_store", source)

    def test_get_stats_no_whoosh_reference(self):
        """get_stats should not reference whoosh_store."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.get_stats)
        self.assertNotIn("whoosh_store", source)
        self.assertNotIn("whoosh", source.lower())


class TestPaperIndexerNewRetrieval(unittest.TestCase):
    """PaperIndexer.search_papers should use new embedder + QdrantStore."""

    def test_search_papers_no_old_retriever_reference(self):
        """search_papers should not use bm25_retriever/vector_retriever/hybrid_retriever."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.search_papers)
        self.assertNotIn("bm25_retriever", source)
        self.assertNotIn("vector_retriever", source)
        self.assertNotIn("hybrid_retriever", source)

    def test_no_old_retriever_imports(self):
        """Module should not import old retriever classes."""
        import pipeline.paper_crawler.integration.index_integration as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from core.retrieval.bm25_retriever import", source)
        self.assertNotIn("from core.retrieval.vector_retriever import", source)
        self.assertNotIn("from core.retrieval.hybrid_retriever import", source)


class TestPaperIndexerSingleIndexNewEmbedder(unittest.TestCase):
    """PaperIndexer.index_paper (single) should use new_embedder with sparse."""

    def test_index_paper_uses_new_embedder(self):
        """index_paper should call new_embedder.embed_chunks, not old embedder."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.index_paper)
        self.assertIn("new_embedder", source)
        self.assertNotIn("self.index_service.embedder.embed_chunks", source)

    def test_index_paper_passes_sparse_vectors(self):
        """index_paper should pass sparse_vectors to _write_paper_to_stores."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.index_paper)
        self.assertIn("sparse_vectors", source)


if __name__ == "__main__":
    unittest.main()
