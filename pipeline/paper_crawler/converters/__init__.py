"""Converters: PDF → Markdown (and other future formats).

The strategy interface + concrete extractors live in
:mod:`pipeline.paper_crawler.converters.extractors`. The facade in
:mod:`pipeline.paper_crawler.converters.pdf_to_md` is what the crawler
workflow uses.
"""
from .extractors import (
    ExtractionResult,
    ExtractionStatus,
    FallbackExtractor,
    MinerUExtractor,
    PdfExtractor,
    PyMuPDFExtractor,
    create_default_chain,
    create_extractor,
)
from .pdf_to_md import PdfToMarkdown, PDFToMarkdownConverter

__all__ = [
    "PdfToMarkdown",
    "PDFToMarkdownConverter",
    "PdfExtractor",
    "ExtractionResult",
    "ExtractionStatus",
    "FallbackExtractor",
    "MinerUExtractor",
    "PyMuPDFExtractor",
    "create_extractor",
    "create_default_chain",
]
