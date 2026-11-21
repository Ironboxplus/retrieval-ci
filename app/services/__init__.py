"""
Service package with lazy top-level exports.

Keeping imports lazy prevents optional crawl/browser dependencies from breaking
unrelated workflows such as document indexing or paper rebuilds.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "DocumentService": ("services.document_service", "DocumentService"),
    "IndexService": ("services.index_service", "IndexService"),
    "TaskService": ("services.task_service", "TaskService"),
    "CrawlService": ("services.crawl_service", "CrawlService"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'services' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
