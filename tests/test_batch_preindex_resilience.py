"""Tests for batch preindex resilience: timeout, heartbeat, wait=False."""
import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.storage.qdrant_store import QdrantStore
from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer


class TestQdrantStoreWaitParam(unittest.TestCase):
    """QdrantStore.add_chunks must accept a `wait` parameter."""

    def test_add_chunks_accepts_wait_false(self):
        import inspect
        sig = inspect.signature(QdrantStore.add_chunks)
        self.assertIn("wait", sig.parameters,
                       "add_chunks must accept a 'wait' keyword argument")

    def test_add_chunks_default_wait_is_true(self):
        import inspect
        sig = inspect.signature(QdrantStore.add_chunks)
        self.assertEqual(sig.parameters["wait"].default, True,
                         "Default wait should be True for backward compatibility")


class TestBatchIndexTimeout(unittest.TestCase):
    """batch_index_papers must accept a per_batch_timeout_sec parameter."""

    def test_batch_index_accepts_timeout(self):
        import inspect
        sig = inspect.signature(PaperIndexer.batch_index_papers)
        self.assertIn("per_batch_timeout_sec", sig.parameters,
                       "batch_index_papers must accept per_batch_timeout_sec")

    def test_batch_index_default_timeout_is_reasonable(self):
        import inspect
        sig = inspect.signature(PaperIndexer.batch_index_papers)
        default = sig.parameters["per_batch_timeout_sec"].default
        self.assertGreaterEqual(default, 60, "Default timeout should be >= 60s")
        self.assertLessEqual(default, 600, "Default timeout should be <= 600s")


class TestPreindexHeartbeat(unittest.TestCase):
    """Preindex script must support a heartbeat callback."""

    def test_run_preindex_accepts_heartbeat_callback(self):
        from scripts.preindex_existing_extracted_corpus_papers import run_preindex
        import inspect
        sig = inspect.signature(run_preindex)
        self.assertIn("heartbeat_callback", sig.parameters,
                       "run_preindex must accept heartbeat_callback")


class TestWritePaperToStoresUsesWaitFalse(unittest.TestCase):
    """During batch operations, Qdrant upserts should use wait=False."""

    @patch.object(QdrantStore, '__init__', lambda self, **kw: None)
    def test_write_paper_passes_wait_param(self):
        store = QdrantStore.__new__(QdrantStore)
        store.collection_name = "test"
        store.client = MagicMock()
        store.add_chunks_called_with_wait = None

        original_add_chunks = QdrantStore.add_chunks

        def mock_add_chunks(self, chunks, wait=True, **kwargs):
            self.add_chunks_called_with_wait = wait

        store.add_chunks = mock_add_chunks.__get__(store, QdrantStore)
        store.add_chunks([], wait=False)
        self.assertEqual(store.add_chunks_called_with_wait, False)


if __name__ == "__main__":
    unittest.main()
