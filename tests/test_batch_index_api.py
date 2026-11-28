"""Tests for POST /api/papers/batch-index endpoint and PipelineService.start_batch_index."""
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path


def _record(paper_id, conference, year, md_path):
    """A minimal manifest record satisfying CorpusPaper's required fields."""
    return {
        "paper_id": paper_id, "title": paper_id, "authors": ["A"],
        "conference": conference, "year": year,
        "pdf_url": "http://x/y.pdf", "paper_page_url": "http://x/y",
        "source_name": "test", "markdown_path": md_path,
    }


class TestPipelineServiceBatchIndex(unittest.TestCase):
    """Test PipelineService.start_batch_index against a small synthetic manifest.

    These exercise the filter/count logic deterministically; they must NOT scan the
    real 95k-record production manifest (that rebuilds 95k dataclasses + stats every
    call and stalls the suite for minutes).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        # Two NeurIPS-2024 papers whose markdown exists (eligible) + one AAAI-2023
        # whose markdown is absent (matched but not eligible).
        md1 = root / "a.md"; md1.write_text("# a", encoding="utf-8")
        md2 = root / "b.md"; md2.write_text("# b", encoding="utf-8")
        records = [
            _record("neurips-2024-1", "NeurIPS", 2024, str(md1)),
            _record("neurips-2024-2", "NeurIPS", 2024, str(md2)),
            _record("aaai-2023-1", "AAAI", 2023, str(root / "missing.md")),
        ]
        self.manifest = root / "paper_manifest.json"
        self.manifest.write_text(json.dumps(records), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _svc(self):
        from app.services.pipeline_service import PipelineService
        svc = PipelineService()
        svc._manifest_path = self.manifest
        return svc

    def test_has_batch_index_method(self):
        from app.services.pipeline_service import PipelineService
        svc = PipelineService()
        self.assertTrue(hasattr(svc, "start_batch_index"))

    def test_missing_manifest_returns_error(self):
        from app.services.pipeline_service import PipelineService
        svc = PipelineService()
        svc._manifest_path = Path("/nonexistent/manifest.json")
        result = svc.start_batch_index()
        self.assertEqual(result["status"], "error")
        self.assertIn("Manifest not found", result["error"])

    @patch("app.services.pipeline_service.PipelineService._run_reindex")
    def test_dry_run_returns_counts(self, mock_reindex):
        result = self._svc().start_batch_index(dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["total_matched"], 3)
        self.assertEqual(result["eligible"], 2)  # missing.md excluded
        mock_reindex.assert_not_called()

    @patch("app.services.pipeline_service.PipelineService._run_reindex")
    def test_venue_filter_narrows_results(self, mock_reindex):
        svc = self._svc()
        all_result = svc.start_batch_index(dry_run=True)
        venue_result = svc.start_batch_index(venue="NeurIPS", dry_run=True)
        self.assertEqual(all_result["total_matched"], 3)
        self.assertEqual(venue_result["total_matched"], 2)
        self.assertLess(venue_result["total_matched"], all_result["total_matched"])

    @patch("app.services.pipeline_service.PipelineService._run_reindex")
    def test_year_filter(self, mock_reindex):
        result = self._svc().start_batch_index(year=2024, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["total_matched"], 2)

    @patch("app.services.pipeline_service.PipelineService._run_reindex")
    def test_start_returns_task_id(self, mock_reindex):
        from app.services.pipeline_service import _active_tasks
        result = self._svc().start_batch_index(venue="NeurIPS", year=2024, batch_size=10)
        if result.get("status") == "conflict":
            self.skipTest("Another task running")
        self.assertEqual(result["status"], "started")
        self.assertIn("task_id", result)
        self.assertIn("eligible", result)
        # Cleanup
        tid = result["task_id"]
        if tid in _active_tasks:
            _active_tasks[tid].status = "completed"


class TestBatchIndexRouterImport(unittest.TestCase):
    """Test that batch_index_router is importable and has the endpoint."""

    def test_import_batch_index_router(self):
        from app.api.routes.pipeline import batch_index_router
        self.assertIsNotNone(batch_index_router)

    def test_batch_index_route_registered(self):
        from app.api.routes.pipeline import batch_index_router
        paths = [r.path for r in batch_index_router.routes]
        self.assertIn("/api/papers/batch-index", paths)


class TestBatchIndexRequestModel(unittest.TestCase):
    """Test BatchIndexRequest model defaults and validation."""

    def test_default_values(self):
        from app.api.routes.pipeline import BatchIndexRequest
        req = BatchIndexRequest()
        self.assertIsNone(req.venue)
        self.assertIsNone(req.year)
        self.assertIsNone(req.paper_ids)
        self.assertEqual(req.batch_size, 50)
        self.assertFalse(req.force)
        self.assertFalse(req.dry_run)

    def test_with_filters(self):
        from app.api.routes.pipeline import BatchIndexRequest
        req = BatchIndexRequest(venue="AAAI", year=2024, batch_size=100, dry_run=True)
        self.assertEqual(req.venue, "AAAI")
        self.assertEqual(req.year, 2024)
        self.assertEqual(req.batch_size, 100)
        self.assertTrue(req.dry_run)


class TestSectionChunkerDedup(unittest.TestCase):
    """Test B3 fix: SectionAwareChunker deduplicates identical chunks."""

    def test_duplicate_sections_deduplicated(self):
        from core.indexing.chunker import SectionAwareChunker
        chunker = SectionAwareChunker()
        text = "## Section A\nHello world content.\n\n## Section B\nHello world content.\n"
        chunks = chunker.chunk_document("test-doc", text)
        contents = [c.content for c in chunks]
        unique_normalized = set()
        import re
        for c in contents:
            unique_normalized.add(re.sub(r'\s+', ' ', c))
        self.assertEqual(len(chunks), len(unique_normalized),
                         "SectionAwareChunker should deduplicate identical chunk content")

    def test_different_sections_kept(self):
        from core.indexing.chunker import SectionAwareChunker
        chunker = SectionAwareChunker()
        text = "## Introduction\nFirst section.\n\n## Methods\nSecond section.\n\n## Results\nThird section.\n"
        chunks = chunker.chunk_document("test-doc", text)
        self.assertEqual(len(chunks), 3)

    def test_empty_doc_returns_empty(self):
        from core.indexing.chunker import SectionAwareChunker
        chunker = SectionAwareChunker()
        chunks = chunker.chunk_document("test-doc", "")
        self.assertEqual(len(chunks), 0)


class TestExtractedTextLength(unittest.TestCase):
    """Test B1 fix: _extracted_text_length strips HTML tags."""

    def test_plain_text(self):
        from app.services.crawl_service import _extracted_text_length
        self.assertEqual(_extracted_text_length("hello world"), 11)

    def test_html_tags_stripped(self):
        from app.services.crawl_service import _extracted_text_length
        html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
        length = _extracted_text_length(html)
        self.assertLess(length, len(html))
        self.assertGreater(length, 0)

    def test_empty_content(self):
        from app.services.crawl_service import _extracted_text_length
        self.assertEqual(_extracted_text_length(""), 0)
        self.assertEqual(_extracted_text_length(None), 0)


if __name__ == "__main__":
    unittest.main()
