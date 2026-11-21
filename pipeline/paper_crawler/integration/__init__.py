"""
Integration package with lazy exports.

The rebuild/indexing path should not import FastAPI route wiring unless it is
explicitly requested by the caller.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "PaperIndexer": ("paper_crawler.integration.index_integration", "PaperIndexer"),
    "PaperSearchResult": ("paper_crawler.integration.index_integration", "PaperSearchResult"),
    "PaperIndexConfig": ("paper_crawler.integration.index_integration", "PaperIndexConfig"),
    "index_papers_from_pipeline": ("paper_crawler.integration.index_integration", "index_papers_from_pipeline"),
    "create_paper_indexer": ("paper_crawler.integration.index_integration", "create_paper_indexer"),
    "create_paper_router": ("paper_crawler.integration.api_routes", "create_paper_router"),
    "mount_paper_routes": ("paper_crawler.integration.api_routes", "mount_paper_routes"),
    "PaperSearchRequest": ("paper_crawler.integration.api_routes", "PaperSearchRequest"),
    "PaperSearchResponse": ("paper_crawler.integration.api_routes", "PaperSearchResponse"),
    "PaperIndexRequest": ("paper_crawler.integration.api_routes", "PaperIndexRequest"),
    "PaperIndexResponse": ("paper_crawler.integration.api_routes", "PaperIndexResponse"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'paper_crawler.integration' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
