"""Marker-based PDF→Markdown extractor. RETIRED: constructing it raises ``RetiredInferencePath``.

Marker runs torch / surya models in the calling process; every model is a vLLM service now (or, for
PDF→Markdown, the registered PaddleOCR-VL worker: PIPELINE_LOCAL_EXTRACTOR). User decision
2026-11-14: "Marker 报错，Paddle 保留". The class stays in the tree so old code fails by name instead of
by an ImportError; ``create_extractor("marker")`` and any ``FallbackExtractor(primary=MarkerExtractor())``
fail at construction, before a fallback could quietly take over.

Marker (vikparuchuri/marker) is fast on academic born-digital PDFs:
~5-10s/PDF on a 2080Ti vs MinerU's 60-100s, while still producing
section-headed markdown that :class:`SectionAwareChunker` can split on.

The model dictionary load is the expensive step (~20-30s) so we cache
both ``model_dict`` and the constructed ``PdfConverter`` at class scope
— a single process that converts thousands of PDFs only pays the
initialization cost once.

Import is deferred to first call. Machines without ``marker-pdf``
installed get a FAILED ``ExtractionResult`` with a helpful message and
fall through to whatever fallback strategy is configured.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from core.inference_retired import EXTRACTOR_INSTEAD, RetiredInferencePath

from .base import ExtractionResult, ExtractionStatus, PdfExtractor

_logger = logging.getLogger(__name__)


class MarkerExtractor(PdfExtractor):
    strategy_name = "marker"

    # Process-wide cache. PdfConverter holds GPU/CPU torch models — loading
    # them per-PDF would dominate runtime. The lock guards the first init
    # against concurrent calls from threaded batch drivers.
    _converter = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        output_format: str = "markdown",
        paginate_output: bool = False,
        use_llm: bool = False,
        extra_config: Optional[dict] = None,
    ):
        raise RetiredInferencePath("MarkerExtractor (in-process torch / surya Marker)", EXTRACTOR_INSTEAD)
        self.output_format = output_format
        self.paginate_output = paginate_output
        self.use_llm = use_llm
        self.extra_config = extra_config or {}

    def _build_config(self) -> dict:
        cfg = {
            "output_format": self.output_format,
            "paginate_output": self.paginate_output,
        }
        if self.use_llm:
            cfg["use_llm"] = True
        cfg.update(self.extra_config)
        return cfg

    def _get_converter(self):
        """Lazy-init the marker PdfConverter, caching at class scope.

        Returns the converter on success, or raises ``ImportError`` /
        ``RuntimeError`` if marker isn't available or fails to load.
        """
        if MarkerExtractor._converter is not None:
            return MarkerExtractor._converter

        with MarkerExtractor._init_lock:
            if MarkerExtractor._converter is not None:
                return MarkerExtractor._converter

            from marker.converters.pdf import PdfConverter  # type: ignore
            from marker.models import create_model_dict  # type: ignore
            from marker.config.parser import ConfigParser  # type: ignore

            _logger.info("marker.init.start (first call; loading models)")
            parser = ConfigParser(self._build_config())
            model_dict = create_model_dict()
            converter = PdfConverter(
                config=parser.generate_config_dict(),
                artifact_dict=model_dict,
                processor_list=parser.get_processors(),
                renderer=parser.get_renderer(),
            )
            MarkerExtractor._converter = converter
            _logger.info("marker.init.ok converter cached")
            return converter

    @classmethod
    def reset_cache(cls) -> None:
        """Drop the cached converter — used by tests."""
        with cls._init_lock:
            cls._converter = None

    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        t0 = self._start()
        pre_flight = self._validate_pdf(pdf_path, t0)
        if pre_flight is not None:
            return pre_flight

        try:
            converter = self._get_converter()
        except ImportError as exc:
            msg = (
                f"marker-pdf not installed ({exc}). "
                "Install with `pip install marker-pdf`."
            )
            _logger.warning("marker.unavailable pdf=%s err=%s", pdf_path.name, msg)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=msg,
                elapsed_ms=self._elapsed_ms(t0),
            )
        except Exception as exc:
            _logger.exception("marker.init.fail pdf=%s", pdf_path.name)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"marker init failed: {exc}",
                elapsed_ms=self._elapsed_ms(t0),
            )

        try:
            from marker.output import text_from_rendered  # type: ignore

            rendered = converter(str(pdf_path))
            md_text, metadata, images = text_from_rendered(rendered)
        except Exception as exc:
            _logger.exception("marker.convert.fail pdf=%s", pdf_path.name)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=f"marker conversion failed: {exc}",
                elapsed_ms=self._elapsed_ms(t0),
            )

        if not (md_text and md_text.strip()):
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error="marker produced empty markdown",
                elapsed_ms=self._elapsed_ms(t0),
            )

        page_count = 0
        if isinstance(metadata, dict):
            page_count = int(metadata.get("page_count") or metadata.get("pages") or 0)

        elapsed = self._elapsed_ms(t0)
        _logger.info(
            "marker.extract.ok pdf=%s md_len=%d images=%d elapsed_ms=%.1f",
            pdf_path.name, len(md_text), len(images or {}), elapsed,
        )
        return ExtractionResult(
            pdf_path=pdf_path,
            status=ExtractionStatus.SUCCESS,
            strategy=self.strategy_name,
            markdown=md_text,
            elapsed_ms=elapsed,
            page_count=page_count,
            has_images=bool(images),
            extra={"image_count": len(images or {})},
        )


__all__ = ["MarkerExtractor"]
