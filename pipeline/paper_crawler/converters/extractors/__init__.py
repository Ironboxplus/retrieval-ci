"""PDF → Markdown extractor strategies.

Public surface:

    from pipeline.paper_crawler.converters.extractors import (
        PdfExtractor, ExtractionResult, ExtractionStatus,
        MarkerExtractor, MinerUExtractor, PyMuPDFExtractor, FallbackExtractor,
        create_extractor,
    )

Use :func:`create_extractor` to obtain a concrete extractor by name. To
chain a primary + fallback strategy together, wrap them in a
:class:`FallbackExtractor`.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    ExtractionResult,
    ExtractionStatus,
    FallbackExtractor,
    PdfExtractor,
)
from .marker import MarkerExtractor
from .mineru import MinerUExtractor
from .pymupdf import PyMuPDFExtractor

# Registry of strategy name → class. Adding a new extractor only requires
# importing it here and adding one row — no caller branching needed.
_REGISTRY: dict[str, type[PdfExtractor]] = {
    "marker": MarkerExtractor,
    "mineru": MinerUExtractor,
    "pymupdf": PyMuPDFExtractor,
}


def create_extractor(strategy: str = "mineru", **kwargs) -> PdfExtractor:
    """Return a fresh extractor for ``strategy``.

    Raises ``ValueError`` for unknown names — failing loud here is
    intentional, since misspelling a strategy silently picking a default
    has caused us before in the embedder factory.
    """
    key = (strategy or "").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"unknown extractor strategy {strategy!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return cls(**kwargs)


def create_default_chain(
    primary_kwargs: Optional[dict] = None,
    fallback_kwargs: Optional[dict] = None,
) -> FallbackExtractor:
    """Common case: MinerU primary, PyMuPDF fallback."""
    return FallbackExtractor(
        primary=MinerUExtractor(**(primary_kwargs or {})),
        fallback=PyMuPDFExtractor(**(fallback_kwargs or {})),
    )


__all__ = [
    "PdfExtractor",
    "ExtractionResult",
    "ExtractionStatus",
    "FallbackExtractor",
    "MarkerExtractor",
    "MinerUExtractor",
    "PyMuPDFExtractor",
    "create_extractor",
    "create_default_chain",
]
