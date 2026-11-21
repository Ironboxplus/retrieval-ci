"""
Paper crawler package with lazy top-level imports.

Keeping package import side effects small makes standalone scripts and tooling
more robust, especially in environments where optional crawler settings or SDKs
may be unavailable.
"""

from __future__ import annotations

import importlib
from typing import Any


__version__ = "0.1.0"
__author__ = "Paper Crawler Team"


_LAZY_EXPORTS = {
    "Config": ("paper_crawler.config", "Config"),
    "Paper": ("paper_crawler.models.paper", "Paper"),
    "BaseScraper": ("paper_crawler.scrapers", "BaseScraper"),
    "ArxivScraper": ("paper_crawler.scrapers", "ArxivScraper"),
    "OpenReviewScraper": ("paper_crawler.scrapers", "OpenReviewScraper"),
    "ACLScraper": ("paper_crawler.scrapers", "ACLScraper"),
    "NeurIPSScraper": ("paper_crawler.scrapers", "NeurIPSScraper"),
    "SemanticScholarScraper": ("paper_crawler.scrapers", "SemanticScholarScraper"),
    "get_openreview_scraper": ("paper_crawler.scrapers", "get_openreview_scraper"),
    "get_acl_scraper": ("paper_crawler.scrapers", "get_acl_scraper"),
    "get_neurips_scraper": ("paper_crawler.scrapers", "get_neurips_scraper"),
    "get_semantic_scholar_scraper": ("paper_crawler.scrapers", "get_semantic_scholar_scraper"),
    "get_arxiv_scraper": ("paper_crawler.scrapers", "get_arxiv_scraper"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value

    if name == "get_pdf_downloader":
        from pipeline.paper_crawler.downloaders.pdf_downloader import PDFDownloader

        globals()[name] = PDFDownloader
        return PDFDownloader

    if name == "get_pdf_converter":
        from pipeline.paper_crawler.converters.pdf_to_md import PDFToMarkdownConverter

        globals()[name] = PDFToMarkdownConverter
        return PDFToMarkdownConverter

    if name == "get_paper_pipeline":
        from pipeline.paper_crawler.workflows.paper_pipeline import PaperPipeline

        globals()[name] = PaperPipeline
        return PaperPipeline

    raise AttributeError(f"module 'paper_crawler' has no attribute {name!r}")


__all__ = [
    "Config",
    "Paper",
    "BaseScraper",
    "ArxivScraper",
    "OpenReviewScraper",
    "ACLScraper",
    "NeurIPSScraper",
    "SemanticScholarScraper",
    "get_openreview_scraper",
    "get_acl_scraper",
    "get_neurips_scraper",
    "get_semantic_scholar_scraper",
    "get_arxiv_scraper",
    "get_pdf_downloader",
    "get_pdf_converter",
    "get_paper_pipeline",
]
