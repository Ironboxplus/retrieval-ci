"""PdfToMarkdown facade.

Thin wrapper around the Strategy extractors in ``.extractors`` package.
This is the file callers (paper_pipeline.py, main.py) keep importing —
the heavy logic now lives in the extractor strategies, so this file stays
small and easy to read.

The legacy class name ``PDFToMarkdownConverter`` still exists as a
subclass alias so existing call sites (and any pickled config) keep
working without churn. Once the active corpus has been re-extracted via
MinerU we can deprecate the alias.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .extractors import (
    ExtractionResult,
    ExtractionStatus,
    PdfExtractor,
    create_default_chain,
    create_extractor,
)

_logger = logging.getLogger(__name__)


class PdfToMarkdown:
    """High-level façade for one-paper-at-a-time PDF→MD conversion.

    The class is a thin wrapper around a :class:`PdfExtractor` strategy:
    construction picks the strategy, ``convert_paper`` and
    ``convert_single`` orchestrate disk IO + metadata header. The legacy
    crawler workflow uses ``convert_paper(paper_obj)``; standalone scripts
    use ``convert_single(pdf_path, …)``.
    """

    def __init__(
        self,
        strategy: str = "mineru",
        *,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
        with_fallback: bool = True,
        extractor: Optional[PdfExtractor] = None,
    ):
        """
        Args:
            strategy: name passed to :func:`create_extractor` (ignored if
                ``extractor`` is given). Default ``"mineru"`` matches the
                project-wide policy.
            output_dir: where to write ``.md`` files. None = no on-disk
                write, caller uses the returned ExtractionResult directly.
            overwrite: re-extract even if the target ``.md`` exists.
            with_fallback: when constructing from ``strategy``, wrap the
                primary in :class:`FallbackExtractor` with PyMuPDF as the
                fallback so corrupted PDFs don't crash a batch run.
            extractor: pre-built extractor instance (lets tests inject mocks).
        """
        self.strategy = strategy
        self.output_dir = Path(output_dir) if output_dir else None
        self.overwrite = overwrite
        if extractor is not None:
            self._extractor = extractor
        elif with_fallback and strategy != "pymupdf":
            self._extractor = create_default_chain(
                primary_kwargs={},
                fallback_kwargs={},
            )
        else:
            self._extractor = create_extractor(strategy)

    @property
    def extractor(self) -> PdfExtractor:
        return self._extractor

    def convert_single(
        self,
        pdf_path: Path,
        *,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
        add_metadata: bool = True,
    ) -> ExtractionResult:
        """Extract one PDF and optionally write the markdown to ``output_dir``."""
        pdf_path = Path(pdf_path)
        out_path = self._target_path(pdf_path, conference, year, paper_id)
        if out_path is not None and out_path.exists() and not self.overwrite:
            _logger.info("pdf_to_md.skip exists pdf=%s out=%s", pdf_path.name, out_path)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.SKIPPED,
                strategy=self._extractor.strategy_name,
                markdown_path=out_path,
            )

        result = self._extractor.extract(pdf_path)
        if not result.succeeded or result.markdown is None:
            return result

        if add_metadata:
            result.markdown = self._prepend_frontmatter(result)

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.markdown, encoding="utf-8")
            result.markdown_path = out_path
            _logger.info("pdf_to_md.write pdf=%s out=%s", pdf_path.name, out_path)

        return result

    def convert_paper(self, paper, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Adapter for the crawler workflow which deals in Paper objects."""
        pdf_path = getattr(paper, "pdf_path", None)
        if not pdf_path:
            _logger.warning("pdf_to_md.paper.no_pdf paper_id=%s", getattr(paper, "id", "?"))
            return None

        prior_output = self.output_dir
        try:
            if output_dir is not None:
                self.output_dir = Path(output_dir)

            result = self.convert_single(
                pdf_path=Path(pdf_path),
                conference=self._conference_value(paper),
                year=self._year_value(paper),
                paper_id=getattr(paper, "id", None),
            )
            return result.markdown_path if result.succeeded else None
        finally:
            self.output_dir = prior_output

    def _target_path(
        self,
        pdf_path: Path,
        conference: Optional[str],
        year: Optional[int],
        paper_id: Optional[str],
    ) -> Optional[Path]:
        if self.output_dir is None:
            return None

        stem = paper_id or pdf_path.stem
        if conference and year:
            return self.output_dir / conference / str(year) / f"{stem}.md"
        return self.output_dir / f"{stem}.md"

    @staticmethod
    def _prepend_frontmatter(result: ExtractionResult) -> str:
        head = (
            "---\n"
            f"source_pdf: {result.pdf_path.name}\n"
            f"extraction_strategy: {result.strategy}\n"
            f"extraction_time: {datetime.now().isoformat()}\n"
            f"page_count: {result.page_count}\n"
            f"has_equations: {result.has_equations}\n"
            f"has_tables: {result.has_tables}\n"
            "---\n\n"
        )
        return head + (result.markdown or "")

    @staticmethod
    def _conference_value(paper: Any) -> Optional[str]:
        conference = getattr(paper, "conference", None)
        if conference is None:
            return None
        return conference.value if hasattr(conference, "value") else str(conference)

    @staticmethod
    def _year_value(paper: Any) -> Optional[int]:
        year = getattr(paper, "year", None)
        if year:
            try:
                return int(year)
            except (TypeError, ValueError):
                return None
        meta = getattr(paper, "metadata", None)
        return int(meta.year) if meta and getattr(meta, "year", None) else None


class PDFToMarkdownConverter(PdfToMarkdown):
    """Legacy alias kept for callers that still import the old class name.

    The marker-based monolith lived in ``scripts/archive/old_converters/``
    after the Stage J refactor; new code should prefer :class:`PdfToMarkdown`.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        use_marker: bool = False,
        strategy: str = "mineru",
    ):
        if use_marker:
            _logger.warning(
                "PDFToMarkdownConverter(use_marker=True) is no-op — marker was "
                "archived in Stage J; using strategy=%s instead", strategy,
            )
        super().__init__(strategy=strategy, output_dir=output_dir)


__all__ = ["PdfToMarkdown", "PDFToMarkdownConverter"]
