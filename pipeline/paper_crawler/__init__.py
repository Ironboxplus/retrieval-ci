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
    "Paper": ("pipeline.paper_crawler.models.paper", "Paper"),
    "get_pdf_converter": (
        "pipeline.paper_crawler.converters.pdf_to_md",
        "PDFToMarkdownConverter",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'paper_crawler' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
