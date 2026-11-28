"""TDD red tests for IndexService + PaperIndexer chunker wiring.

The new contract: callers may pick the chunking strategy via
``chunker_type`` (forwarded to ``create_chunker``). Defaults preserve the
legacy multi-granularity behaviour for live API ingest paths so existing
6.68M historical chunks stay coherent. New paths (ICLR / MinerU corpus
ingest) opt into ``markdown`` → SectionAwareChunker.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestIndexServiceChunkerType(unittest.TestCase):
    def test_default_chunker_is_text_chunker(self):
        """Default preserves legacy multi-granularity behaviour."""
        from app.services.index_service import IndexService
        from core.indexing.chunker import TextChunker

        svc = IndexService()
        self.assertIsInstance(svc.chunker, TextChunker)

    def test_explicit_markdown_chunker_is_section_aware(self):
        from app.services.index_service import IndexService
        from core.indexing.chunker import SectionAwareChunker

        svc = IndexService(chunker_type="markdown")
        self.assertIsInstance(svc.chunker, SectionAwareChunker)

    def test_explicit_text_chunker_is_section_aware(self):
        """Headerless / plain text uploads route through SectionAware with fallback."""
        from app.services.index_service import IndexService
        from core.indexing.chunker import SectionAwareChunker

        svc = IndexService(chunker_type="text")
        self.assertIsInstance(svc.chunker, SectionAwareChunker)

    def test_explicit_html_chunker_is_section_aware(self):
        from app.services.index_service import IndexService
        from core.indexing.chunker import SectionAwareChunker

        svc = IndexService(chunker_type="html")
        self.assertIsInstance(svc.chunker, SectionAwareChunker)


class TestPaperIndexConfigChunker(unittest.TestCase):
    def test_paper_index_config_accepts_chunker_type(self):
        from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig

        cfg = PaperIndexConfig(chunker_type="markdown")
        self.assertEqual(cfg.chunker_type, "markdown")

    def test_paper_indexer_propagates_chunker_type(self):
        from pipeline.paper_crawler.integration.index_integration import (
            PaperIndexConfig,
            PaperIndexer,
        )
        from core.indexing.chunker import SectionAwareChunker

        indexer = PaperIndexer(PaperIndexConfig(chunker_type="markdown"))
        self.assertIsInstance(indexer.index_service.chunker, SectionAwareChunker)

    def test_paper_indexer_default_is_multigranularity(self):
        """Default PaperIndexConfig keeps legacy chunker for back-compat with live API path."""
        from pipeline.paper_crawler.integration.index_integration import (
            PaperIndexConfig,
            PaperIndexer,
        )
        from core.indexing.chunker import TextChunker

        indexer = PaperIndexer(PaperIndexConfig())
        self.assertIsInstance(indexer.index_service.chunker, TextChunker)


if __name__ == "__main__":
    unittest.main()
