"""PyMuPDF-based PDF→Markdown extractor — CPU fallback.

This is the always-available extractor. It produces minimally-structured
text (no formula/table fidelity) but never needs a GPU or large model. Use
it as the secondary strategy behind ``MinerUExtractor`` so that a single
broken PDF can't take down the pipeline.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import ExtractionResult, ExtractionStatus, PdfExtractor

_logger = logging.getLogger(__name__)


def _import_pymupdf():
    """Both ``pymupdf`` (new) and ``fitz`` (legacy) ship the same API."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz  # type: ignore
        return fitz


class PyMuPDFExtractor(PdfExtractor):
    strategy_name = "pymupdf"

    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        t0 = self._start()
        pre_flight = self._validate_pdf(pdf_path, t0)
        if pre_flight is not None:
            return pre_flight

        try:
            pymupdf = _import_pymupdf()
        except ImportError as exc:
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"pymupdf not installed: {exc}",
                elapsed_ms=self._elapsed_ms(t0),
            )

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as exc:
            _logger.exception("pymupdf.open.fail pdf=%s", pdf_path.name)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"failed to open pdf: {exc}",
                elapsed_ms=self._elapsed_ms(t0),
            )

        try:
            parts = []
            image_count = 0
            page_count = len(doc)

            for page_num, page in enumerate(doc, start=1):
                if page_num > 1:
                    parts.append(f"\n\n<!-- Page {page_num} -->\n\n")

                text = page.get_text("text") or ""
                lines = []
                for raw in text.split("\n"):
                    stripped = raw.strip()
                    if not stripped:
                        lines.append("")
                    elif len(stripped) < 100 and stripped.isupper():
                        lines.append(f"## {stripped.title()}")
                    else:
                        lines.append(stripped)
                parts.append("\n".join(lines))

                image_count += len(page.get_images() or [])

            markdown = "\n".join(parts)
            markdown = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", markdown)
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)

            elapsed = self._elapsed_ms(t0)
            _logger.info(
                "pymupdf.extract.ok pdf=%s pages=%d images=%d md_len=%d elapsed_ms=%.1f",
                pdf_path.name, page_count, image_count, len(markdown), elapsed,
            )
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.SUCCESS,
                strategy=self.strategy_name,
                markdown=markdown,
                elapsed_ms=elapsed,
                page_count=page_count,
                has_images=image_count > 0,
            )
        finally:
            doc.close()


__all__ = ["PyMuPDFExtractor"]
