"""
Indexing package with lazy exports.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "TextChunker": ("indexing.chunker", "TextChunker"),
    "Embedder": ("indexing.embedder", "Embedder"),
    "IndexManager": ("indexing.index_manager", "IndexManager"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'indexing' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
