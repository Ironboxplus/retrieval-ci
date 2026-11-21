"""
TDD tests for issue fixes (S4, F3, F4, F5, I1, I3, Paper detail 404, Tasks Drawer).
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPaperDetailEndpoint(unittest.TestCase):
    """Paper detail: browse generates paper_id as {venue}-{year}-{db_id},
    GET /api/papers/{paper_id} must parse and query DB directly."""

    def test_paper_id_parsing_extracts_db_id(self):
        paper_id = "aaai-2025-37775"
        parts = paper_id.rsplit("-", 1)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "aaai-2025")
        self.assertEqual(int(parts[1]), 37775)

    def test_paper_id_parsing_single_segment(self):
        paper_id = "12345"
        parts = paper_id.rsplit("-", 1)
        self.assertEqual(len(parts), 1)

    def test_paper_id_parsing_complex_venue(self):
        paper_id = "neurips-2024-50123"
        parts = paper_id.rsplit("-", 1)
        db_id = int(parts[1])
        self.assertEqual(db_id, 50123)


class TestPaperSearchFix(unittest.TestCase):
    """S4: search_papers must call embed_texts() not embed()."""

    def test_embedder_has_embed_texts(self):
        from core.indexing.embedders import create_embedder, BaseEmbedder
        self.assertTrue(hasattr(BaseEmbedder, 'embed_texts'))
        self.assertFalse(hasattr(BaseEmbedder, 'embed'))

    def test_search_papers_uses_embed_texts(self):
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.search_papers)
        self.assertIn('embed_texts', source)
        self.assertNotIn('.embed(', source)

    def test_search_papers_is_sync(self):
        """search_papers must be sync (wrapped with asyncio.to_thread at route)."""
        import asyncio
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        self.assertFalse(asyncio.iscoroutinefunction(PaperIndexer.search_papers))

    def test_route_uses_to_thread(self):
        """Route must wrap sync search_papers in asyncio.to_thread."""
        with open("pipeline/paper_crawler/integration/api_routes.py", "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn('asyncio.to_thread', source)

    def test_dense_vector_indexing(self):
        """embed_texts returns 2D dense, search needs 1D — must index [0]."""
        import inspect
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        source = inspect.getsource(PaperIndexer.search_papers)
        self.assertIn('dense[0]', source)


class TestPointsToSearchResults(unittest.TestCase):
    """_points_to_search_results must handle dict format from QdrantStore."""

    def test_handles_dict_format(self):
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        import inspect
        source = inspect.getsource(PaperIndexer._points_to_search_results)
        self.assertIn('isinstance(point, dict)', source)

    def test_reads_metadata_key_not_payload(self):
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer
        import inspect
        source = inspect.getsource(PaperIndexer._points_to_search_results)
        self.assertIn('"metadata"', source)

    def test_normalizes_numeric_qdrant_doc_id_to_search_contract_string(self):
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer

        indexer = PaperIndexer.__new__(PaperIndexer)
        results = indexer._points_to_search_results([
            {
                "id": "point-1",
                "content": "retrieval content",
                "score": 0.9,
                "metadata": {"doc_id": 74536, "title": "Paper"},
            }
        ])

        self.assertEqual(results[0].doc_id, "74536")


class TestPaperSearchRuntimeAdapters(unittest.TestCase):
    def test_get_search_route_offloads_synchronous_indexer(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from pipeline.paper_crawler.integration.api_routes import create_paper_router

        class SyncIndexer:
            def search_papers(self, **_kwargs):
                return []

        app = FastAPI()
        app.include_router(create_paper_router(indexer=SyncIndexer()))
        response = TestClient(app).get(
            "/api/papers/search", params={"query": "retrieval", "top_k": 3}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["papers"], [])

    def test_paper_indexers_reuse_process_shared_embedder(self):
        from unittest.mock import patch
        from pipeline.paper_crawler.integration.index_integration import PaperIndexer

        shared = object()
        first = PaperIndexer.__new__(PaperIndexer)
        first._new_embedder = None
        second = PaperIndexer.__new__(PaperIndexer)
        second._new_embedder = None

        with patch(
            "pipeline.paper_crawler.integration.index_integration._get_shared_embedder",
            return_value=shared,
        ) as getter:
            self.assertIs(first.new_embedder, shared)
            self.assertIs(second.new_embedder, shared)
        self.assertEqual(getter.call_count, 2)


class TestLoggingSystem(unittest.TestCase):
    """I1: Structured logging system."""

    def test_logging_module_exists(self):
        from core.logging import setup_logging, get_logger
        self.assertTrue(callable(setup_logging))
        self.assertTrue(callable(get_logger))

    def test_get_logger_returns_logger(self):
        import logging
        from core.logging import get_logger
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")

    def test_setup_logging_idempotent(self):
        from core.logging import setup_logging
        setup_logging()
        setup_logging()  # should not raise

    def test_main_uses_logger_not_print(self):
        with open("main.py", "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('print(') and not stripped.startswith('#'):
                if 'import' not in stripped:
                    self.fail(f"main.py line {i} still uses print(): {stripped}")


class TestOldFrontendRemoved(unittest.TestCase):
    """I3: Old static frontend on :8000 should be removed."""

    def test_no_static_mount_in_main(self):
        with open("main.py", "r", encoding="utf-8") as f:
            source = f.read()
        # The OLD static frontend (/static + static/index.html) must be gone. The
        # current SPA legitimately serves its build via StaticFiles at /assets, so
        # only the old /static mount is banned — not StaticFiles wholesale.
        self.assertNotIn('app.mount("/static"', source)

    def test_no_home_route_serving_html(self):
        with open("main.py", "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn('FileResponse(index_path)', source)
        self.assertNotIn('static/index.html', source)


class TestPDFPreviewEndpoint(unittest.TestCase):
    """F4: PDF preview endpoint exists in api_routes."""

    def test_pdf_route_defined(self):
        with open(
            "pipeline/paper_crawler/integration/api_routes.py", "r", encoding="utf-8"
        ) as f:
            source = f.read()
        self.assertIn('/{paper_id}/pdf', source)
        self.assertIn('Content-Disposition', source)
        self.assertIn('inline', source)

    def test_pdf_route_before_detail_route(self):
        with open(
            "pipeline/paper_crawler/integration/api_routes.py", "r", encoding="utf-8"
        ) as f:
            source = f.read()
        pdf_pos = source.find('/{paper_id}/pdf')
        detail_pos = source.find('/{paper_id}"')
        self.assertGreater(detail_pos, pdf_pos,
                           "PDF route must be before detail route to avoid path conflicts")


class TestRefreshButtons(unittest.TestCase):
    """F3: All table pages should have refresh buttons.

    F1-d (frontend refactor) deleted Pipeline.tsx and Crawl.tsx — data pipeline
    controls and the crawl-submission page are gone; crawl records now surface
    inline on the `crawl_urls` task card (F1-f), so the old crawl-page refresh
    assertion (and the whole obsolete `TestPipelinePageRedesign` class above)
    were removed rather than pointed at files that no longer exist.
    """

    def _check_page(self, path, query_key):
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        # antd pages use ReloadOutlined; pages moved to shadcn/ui use lucide's RefreshCw.
        self.assertTrue('ReloadOutlined' in source or 'RefreshCw' in source,
                        f"{path} missing a refresh icon (ReloadOutlined or RefreshCw)")
        self.assertIn('useQueryClient', source, f"{path} missing useQueryClient")
        self.assertIn('invalidateQueries', source, f"{path} missing invalidateQueries")
        self.assertIn(query_key, source, f"{path} missing query key {query_key}")

    def test_papers_has_refresh(self):
        self._check_page("frontend/src/pages/PaperList.tsx", "papers")

    def test_tasks_has_refresh(self):
        self._check_page("frontend/src/pages/Tasks.tsx", "tasks")

    def test_documents_has_refresh(self):
        self._check_page("frontend/src/pages/Documents.tsx", "documents")


class TestDrawerDeprecation(unittest.TestCase):
    """Documents Drawer should use size not width (antd v6).

    F1-f (frontend refactor) replaced Tasks.tsx's task-detail Drawer with an
    in-place MorphGrid expansion (click a compact card, it grows into the
    detail view without a slide-out panel) — there is no Drawer left in
    Tasks.tsx to check the `size` prop on, so `test_tasks_drawer_uses_size`
    was removed rather than pointed at a component that no longer exists.
    """

    def test_documents_drawer_uses_size(self):
        with open("frontend/src/pages/Documents.tsx", "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn('width={600}', source)
        self.assertIn('size="large"', source)


if __name__ == "__main__":
    unittest.main()
