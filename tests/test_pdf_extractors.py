"""TDD red tests for the Strategy-pattern PDF→Markdown extractor.

The new architecture replaces the 882-line marker-based monolith with:

  - ``PdfExtractor`` ABC (single ``extract`` method)
  - ``ExtractionResult`` value object (status + markdown + metadata)
  - ``MinerUExtractor`` (primary, GPU)
  - ``PyMuPDFExtractor`` (lightweight fallback, no formulas/tables)
  - ``create_extractor(strategy)`` factory

These tests verify interface contract + factory + fallback strategy + the
fact that ``PDFToMarkdownConverter`` legacy alias still works for the
paper_pipeline.py call site that uses ``convert_paper(paper)``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestExtractorBaseContract(unittest.TestCase):
    def test_pdf_extractor_is_abc(self):
        from pipeline.paper_crawler.converters.extractors import PdfExtractor

        with self.assertRaises(TypeError):
            PdfExtractor()  # abstract instantiation must fail

    def test_extraction_result_has_required_fields(self):
        from pipeline.paper_crawler.converters.extractors import (
            ExtractionResult,
            ExtractionStatus,
        )

        result = ExtractionResult(
            pdf_path=Path("foo.pdf"),
            status=ExtractionStatus.SUCCESS,
            strategy="mineru",
            markdown="# Hello",
            elapsed_ms=12.3,
        )
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertEqual(result.strategy, "mineru")
        self.assertEqual(result.markdown, "# Hello")
        self.assertIsNone(result.error)

    def test_extraction_status_enum(self):
        from pipeline.paper_crawler.converters.extractors import ExtractionStatus

        self.assertEqual(
            {s.value for s in ExtractionStatus},
            {"success", "failed", "fallback", "skipped"},
        )


class TestPyMuPDFExtractor(unittest.TestCase):
    """The CPU fallback must always be importable (project dep)."""

    def test_pymupdf_extractor_loads(self):
        from pipeline.paper_crawler.converters.extractors import PyMuPDFExtractor

        extractor = PyMuPDFExtractor()
        self.assertEqual(extractor.strategy_name, "pymupdf")

    def test_pymupdf_extractor_reports_missing_pdf(self):
        from pipeline.paper_crawler.converters.extractors import (
            ExtractionStatus,
            PyMuPDFExtractor,
        )

        extractor = PyMuPDFExtractor()
        result = extractor.extract(Path("does_not_exist.pdf"))
        self.assertEqual(result.status, ExtractionStatus.FAILED)
        self.assertIn("not found", (result.error or "").lower())


class TestMinerUExtractor(unittest.TestCase):
    """MinerU optional locally; importing the class must not require the dep."""

    def test_mineru_class_importable_without_dep(self):
        from pipeline.paper_crawler.converters.extractors import MinerUExtractor

        extractor = MinerUExtractor()  # construction must not load mineru
        self.assertEqual(extractor.strategy_name, "mineru")

    def test_mineru_reports_dep_missing_when_not_installed(self):
        """When mineru not installed locally, extract() returns FAILED with clear hint."""
        from pipeline.paper_crawler.converters.extractors import (
            ExtractionStatus,
            MinerUExtractor,
        )

        try:
            import mineru  # noqa: F401
            self.skipTest("mineru is installed, skipping missing-dep test")
        except ImportError:
            pass

        extractor = MinerUExtractor()
        result = extractor.extract(Path("does_not_exist.pdf"))
        self.assertEqual(result.status, ExtractionStatus.FAILED)
        self.assertIn("mineru", (result.error or "").lower())


class TestExtractorFactory(unittest.TestCase):
    def test_factory_default_is_mineru(self):
        from pipeline.paper_crawler.converters.extractors import (
            MinerUExtractor,
            create_extractor,
        )

        self.assertIsInstance(create_extractor(), MinerUExtractor)

    def test_factory_can_request_pymupdf(self):
        from pipeline.paper_crawler.converters.extractors import (
            PyMuPDFExtractor,
            create_extractor,
        )

        self.assertIsInstance(create_extractor("pymupdf"), PyMuPDFExtractor)

    def test_factory_unknown_strategy_raises(self):
        from pipeline.paper_crawler.converters.extractors import create_extractor

        with self.assertRaises(ValueError):
            create_extractor("bogus")


class TestPdfToMarkdownFacade(unittest.TestCase):
    """``PdfToMarkdown`` wraps the chosen extractor and writes to disk."""

    def test_facade_construction(self):
        from pipeline.paper_crawler.converters.pdf_to_md import PdfToMarkdown

        facade = PdfToMarkdown(strategy="pymupdf")
        self.assertEqual(facade.strategy, "pymupdf")

    def test_legacy_alias_exists(self):
        """paper_pipeline.py imports ``PDFToMarkdownConverter`` — must keep working."""
        from pipeline.paper_crawler.converters.pdf_to_md import (
            PdfToMarkdown,
            PDFToMarkdownConverter,
        )

        self.assertTrue(issubclass(PDFToMarkdownConverter, PdfToMarkdown))


class TestExtractorFallbackChain(unittest.TestCase):
    """A pipeline can chain extractors — primary fails → try fallback."""

    def test_fallback_chain_uses_secondary_on_primary_failure(self):
        from pipeline.paper_crawler.converters.extractors import (
            ExtractionResult,
            ExtractionStatus,
            FallbackExtractor,
            PdfExtractor,
        )

        class AlwaysFails(PdfExtractor):
            strategy_name = "always_fails"

            def extract(self, pdf_path, **kwargs):
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.FAILED,
                    strategy=self.strategy_name,
                    error="boom",
                )

        class AlwaysSucceeds(PdfExtractor):
            strategy_name = "always_ok"

            def extract(self, pdf_path, **kwargs):
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.SUCCESS,
                    strategy=self.strategy_name,
                    markdown="# ok",
                )

        chain = FallbackExtractor(primary=AlwaysFails(), fallback=AlwaysSucceeds())
        result = chain.extract(Path("foo.pdf"))
        self.assertEqual(result.status, ExtractionStatus.FALLBACK)
        self.assertEqual(result.strategy, "always_ok")
        self.assertEqual(result.markdown, "# ok")

    def test_fallback_chain_primary_success_skips_fallback(self):
        from pipeline.paper_crawler.converters.extractors import (
            ExtractionResult,
            ExtractionStatus,
            FallbackExtractor,
            PdfExtractor,
        )

        class AlwaysSucceeds(PdfExtractor):
            strategy_name = "always_ok"

            def extract(self, pdf_path, **kwargs):
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.SUCCESS,
                    strategy=self.strategy_name,
                    markdown="# ok",
                )

        class WouldFail(PdfExtractor):
            strategy_name = "would_fail"

            def extract(self, pdf_path, **kwargs):
                raise AssertionError("must not be called when primary succeeds")

        chain = FallbackExtractor(primary=AlwaysSucceeds(), fallback=WouldFail())
        result = chain.extract(Path("foo.pdf"))
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertEqual(result.strategy, "always_ok")


if __name__ == "__main__":
    unittest.main()
