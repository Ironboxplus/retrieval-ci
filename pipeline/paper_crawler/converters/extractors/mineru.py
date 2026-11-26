"""MinerU-based PDF→Markdown extractor (primary strategy).

Wraps the ``mineru`` CLI via :mod:`subprocess` instead of importing the
library directly. Rationale (see [[feedback-library-first]]): MinerU's
Python API surface has shifted across releases (1.x / 2.x / 3.x), but
the CLI ``mineru -p <pdf> -o <dir>`` has been stable since 1.0.
Subprocess is also the deployment shape on kss — keeping local and
remote paths identical avoids drift.

When the CLI is absent (machines without MinerU installed), ``extract``
returns a FAILED ``ExtractionResult`` with a helpful hint so callers can
fall through to :class:`PyMuPDFExtractor`.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from .base import ExtractionResult, ExtractionStatus, PdfExtractor

_logger = logging.getLogger(__name__)


class MinerUExtractor(PdfExtractor):
    strategy_name = "mineru"

    # Per-PDF wall-clock cap; MinerU on a 24GB GPU averages ~30s, larger
    # papers go up to ~3 min. The bulk kss worker uses its own timeout.
    DEFAULT_TIMEOUT_S = 600

    def __init__(
        self,
        cmd: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        method: Optional[str] = None,
        lang: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ):
        self.cmd = cmd or "mineru"
        self.device = device
        self.backend = backend
        self.method = method
        self.lang = lang
        self.timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S

    def _resolve_cmd(self) -> Optional[str]:
        """Return the absolute path to the mineru executable or None."""
        located = shutil.which(self.cmd)
        if located:
            return located
        # Fall back to the venv-bundled mineru.exe sibling of the active
        # python interpreter (Windows-friendly).
        bin_dir = Path(sys.executable).resolve().parent
        for name in (self.cmd, f"{self.cmd}.exe"):
            candidate = bin_dir / name
            if candidate.exists():
                return str(candidate)
        return None

    def _build_argv(self, pdf: Path, out_dir: Path) -> list[str]:
        argv = [self._resolve_cmd() or self.cmd, "-p", str(pdf), "-o", str(out_dir)]
        if self.method:
            argv.extend(["-m", self.method])
        if self.backend:
            argv.extend(["-b", self.backend])
        if self.lang:
            argv.extend(["-l", self.lang])
        if self.device:
            argv.extend(["-d", self.device])
        return argv

    @staticmethod
    def _locate_markdown(out_dir: Path, pdf: Path) -> Optional[Path]:
        """MinerU writes ``<out_dir>/<pdf_stem>/<backend>/<pdf_stem>.md``;
        layout differs slightly across backends, so glob for any .md under
        the per-PDF output folder."""
        stem_dir = out_dir / pdf.stem
        if not stem_dir.exists():
            # Some backends write directly under out_dir.
            stem_dir = out_dir
        candidates = sorted(stem_dir.rglob("*.md"))
        if not candidates:
            return None
        # Prefer the file matching the pdf stem; otherwise the first match.
        for md in candidates:
            if md.stem == pdf.stem:
                return md
        return candidates[0]

    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        t0 = self._start()
        pre_flight = self._validate_pdf(pdf_path, t0)
        if pre_flight is not None:
            return pre_flight

        if self._resolve_cmd() is None:
            msg = (
                f"mineru CLI not found in PATH (looked for {self.cmd!r}). "
                "Install with `pip install mineru` or run on the kss "
                "bulk-extraction host."
            )
            _logger.warning("mineru.unavailable pdf=%s err=%s", pdf_path.name, msg)
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error=msg,
                elapsed_ms=self._elapsed_ms(t0),
            )

        with TemporaryDirectory(prefix="mineru_") as td:
            out_dir = Path(td)
            argv = self._build_argv(pdf_path, out_dir)
            _logger.debug("mineru.invoke pdf=%s argv=%s", pdf_path.name, argv)
            try:
                proc = subprocess.run(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.FAILED,
                    strategy=self.strategy_name,
                    error=f"mineru timeout after {self.timeout_s}s",
                    elapsed_ms=self._elapsed_ms(t0),
                )
            except Exception as exc:
                _logger.exception("mineru.invoke.fail pdf=%s", pdf_path.name)
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.FAILED,
                    strategy=self.strategy_name,
                    error=f"mineru invocation failed: {exc}",
                    elapsed_ms=self._elapsed_ms(t0),
                )

            if proc.returncode != 0:
                tail = (proc.stdout.decode("utf-8", "replace") if proc.stdout else "")[-500:]
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.FAILED,
                    strategy=self.strategy_name,
                    error=f"mineru exit={proc.returncode}; tail={tail!r}",
                    elapsed_ms=self._elapsed_ms(t0),
                )

            md_path = self._locate_markdown(out_dir, pdf_path)
            if md_path is None:
                return ExtractionResult(
                    pdf_path=pdf_path,
                    status=ExtractionStatus.FAILED,
                    strategy=self.strategy_name,
                    error="mineru produced no .md file",
                    elapsed_ms=self._elapsed_ms(t0),
                )
            markdown = md_path.read_text(encoding="utf-8", errors="replace")

        if not markdown.strip():
            return ExtractionResult(
                pdf_path=pdf_path,
                status=ExtractionStatus.FAILED,
                strategy=self.strategy_name,
                error="mineru produced empty markdown",
                elapsed_ms=self._elapsed_ms(t0),
            )

        elapsed = self._elapsed_ms(t0)
        _logger.info(
            "mineru.extract.ok pdf=%s md_len=%d elapsed_ms=%.1f",
            pdf_path.name, len(markdown), elapsed,
        )
        return ExtractionResult(
            pdf_path=pdf_path,
            status=ExtractionStatus.SUCCESS,
            strategy=self.strategy_name,
            markdown=markdown,
            elapsed_ms=elapsed,
        )


__all__ = ["MinerUExtractor"]
