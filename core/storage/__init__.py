"""
Storage package with lazy exports.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "ChromaStore": ("storage.qdrant_store", "QdrantStore"),
    "QdrantStore": ("storage.qdrant_store", "QdrantStore"),
    "WhooshStore": ("storage.whoosh_store", "WhooshStore"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'storage' has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
