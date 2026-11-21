"""PDF → Markdown extraction interface (Strategy pattern).

A ``PdfExtractor`` knows how to turn one PDF into a Markdown string. The
concrete subclasses (``MinerUExtractor`` for high-quality GPU extraction,
``PyMuPDFExtractor`` as a CPU fallback) all return a uniform
``ExtractionResult`` so the calling pipeline doesn't branch on which
library was used.

The ``FallbackExtractor`` is a small composite: it tries the primary
strategy and, on FAILED, delegates to a secondary. This keeps the
batch-conversion driver clean — it talks to *one* extractor object even
when retry behaviour is involved.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"  # primary failed, secondary succeeded
    SKIPPED = "skipped"    # output already exists


@dataclass
class ExtractionResult:
    """Uniform return value across all extractors.

    Attributes mirror what the legacy ``ConversionResult`` carried so we
    don't break downstream reporting code, but the field names are simpler
    and the value object is now hashable-by-field (frozen dataclass would
    be ideal — left mutable to keep the legacy update-after-the-fact code
    paths working until they migrate over).
    """

    pdf_path: Path
    status: ExtractionStatus
    strategy: str
    markdown: Optional[str] = None
    markdown_path: Optional[Path] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    page_count: int = 0
    has_equations: bool = False
    has_tables: bool = False
    has_images: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {ExtractionStatus.SUCCESS, ExtractionStatus.FALLBACK}


class PdfExtractor(ABC):
    """Concrete extractors implement ``extract`` and set ``strategy_name``."""

    strategy_name: str = "abstract"

    @abstractmethod
    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        """Run extraction on ``pdf_path`` and return a result.

        Implementations must:
          * never raise on a missing or broken PDF — return ExtractionResult
            with FAILED status + error message instead
          * populate ``elapsed_ms`` on success and on failure
          * fill ``markdown`` (string) when status == SUCCESS / FALLBACK
        """

    def _start(self) -> float:
        return time.monotonic()

    def _elapsed_ms(self, t0: float) -> float:
        return (time.monotonic() - t0) * 1000

    def _validate_pdf(self, pdf_path: Path, t0: float) -> Optional[ExtractionResult]:
        """Common pre-flight: file exists + suffix is .pdf. Returns FAILED on miss."""
        if not pdf_path.exists():
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"PDF file not found: {pdf_path}",
                elapsed_ms=self._elapsed_ms(t0),
            )
        if pdf_path.suffix.lower() != ".pdf":
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"Not a PDF file: {pdf_path}",
                elapsed_ms=self._elapsed_ms(t0),
            )
        return None


class FallbackExtractor(PdfExtractor):
    """Composite: try primary, fall back to secondary on FAILED."""

    strategy_name = "fallback_chain"

    def __init__(self, primary: PdfExtractor, fallback: PdfExtractor):
        self.primary = primary
        self.fallback = fallback

    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        primary_result = self.primary.extract(pdf_path, **kwargs)
        if primary_result.succeeded:
            _logger.debug(
                "extractor.primary.ok strategy=%s pdf=%s elapsed_ms=%.1f",
                self.primary.strategy_name, pdf_path.name, primary_result.elapsed_ms,
            )
            return primary_result

        _logger.warning(
            "extractor.primary.fail strategy=%s pdf=%s err=%s -- trying fallback=%s",
            self.primary.strategy_name, pdf_path.name, primary_result.error,
            self.fallback.strategy_name,
        )
        fallback_result = self.fallback.extract(pdf_path, **kwargs)
        if fallback_result.succeeded:
            # Re-label so the caller can tell the primary path didn't work.
            fallback_result.status = ExtractionStatus.FALLBACK
            _logger.info(
                "extractor.fallback.ok strategy=%s pdf=%s elapsed_ms=%.1f",
                self.fallback.strategy_name, pdf_path.name, fallback_result.elapsed_ms,
            )
        else:
            _logger.error(
                "extractor.fallback.fail strategy=%s pdf=%s err=%s",
                self.fallback.strategy_name, pdf_path.name, fallback_result.error,
            )
        return fallback_result


__all__ = [
    "ExtractionResult",
    "ExtractionStatus",
    "PdfExtractor",
    "FallbackExtractor",
]
