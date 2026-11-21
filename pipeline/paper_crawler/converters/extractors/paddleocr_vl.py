"""PaddleOCR-VL-1.6 PDF→Markdown extractor with a page-by-page check against the PDF.

The whole PaddleOCR-VL pipeline runs (layout detection, the VLM served by vLLM, table
merging, title levelling); this class only feeds it PDFs and refuses its output when a
page does not stand up to the PDF's own text layer (see ``converters/verify``). A page
that fails is converted once more with a repetition penalty; a paper with a page that
still fails gets no Markdown at all, only a report saying which page and why. Nothing
falls back to the text layer: the text layer is a witness, never the body.

Needs ``paddleocr[doc-parser]`` and a running vLLM server (scripts/remote_pvl/pvl_worker.py
starts one), so the import is lazy and ``paddleocr`` is only needed by ``extract_many``.
"""
from __future__ import annotations

import gc
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import ExtractionResult, ExtractionStatus, PdfExtractor
from ..verify.checks import PageVerdict, check_page, letters
from ..verify.katex import KatexRenderer, KatexUnavailable, shared_renderer
from ..verify.mdtext import clean_table_html, parse_page
from ..verify.peer import peer_witness
from ..verify.report import PaperReport
from ..verify.textlayer import PageRef, page_reference

_logger = logging.getLogger(__name__)

# Decoding that loops or drops a block is usually not repeated under a repetition penalty;
# calibrated on pages the first pass failed (see CALIBRATION in verify/checks.py).
RETRY_SAMPLING = {"repetition_penalty": 1.1}
# restructure_pages only relevels titles and merges tables across pages: it may not add or
# drop text, so the merged document's letters stay within this band of the pages' own.
ASSEMBLY_LETTER_RATIO = (0.97, 1.10)

_IMAGE_BLOCK = re.compile(r"<div[^>]*>\s*<img\b[^>]*>\s*</div>|<img\b[^>]*>|!\[[^\]]*\]\([^)]*\)", re.IGNORECASE)


def _plain_markdown(result) -> str:
    """The Markdown of a page or merged-document result without the HTML wrappers of
    ``pretty=True`` (centred ``<div style=...>`` captions, table styles), which would end up in
    every chunk and embedding. ``_to_markdown`` is the MarkdownMixin interface of
    PaddleOCR's result classes; ``.markdown`` is it with the defaults."""
    return result._to_markdown(pretty=False)["markdown_texts"]


# A paragraph the layout cuts in two - at a column or a page break, or around a figure that is not in the Markdown -
# comes back as two text blocks (restructure_pages joins tables and titles across pages, never text).
JOIN_MIN_LETTERS = 40  # a line shorter than this (an affiliation, a heading in a text block) is no paragraph to join
_SENTENCE_END = ".?!:;"
_CLOSING = ")]}\"'”’"
# blocks the Markdown does not show, and images, which are taken out of it afterwards
_NOT_SHOWN = {"image", "chart", "seal"}


def _continues(first: str, second: str) -> bool:
    """``second`` carries on the sentence ``first`` leaves open: a paragraph of some length that ends with no sentence
    punctuation, then one that starts with a lowercase letter and is not an address (the next line of a byline)."""
    first, second = first.rstrip(), second.lstrip()
    head = second.split(None, 1)[0] if second else ""
    if not second[:1].islower() or any(mark in head for mark in "@/") or head.startswith("www"):
        return False
    return len(letters(first)) >= JOIN_MIN_LETTERS and first.rstrip(_CLOSING)[-1:] not in _SENTENCE_END


def _join_split_paragraphs(result) -> int:
    """Join the text blocks of the merged document that split one paragraph (see ``_continues``); returns how many
    were joined. Only blocks that follow each other in the Markdown are candidates: the ones it does not show
    (header, footer, page number, image) are looked through, anything else - a formula, a table, a title - ends a
    paragraph. The text stays the same, so the letter count of the assembly check is unaffected."""
    ignored = set(result["model_settings"].get("markdown_ignore_labels", ())) | _NOT_SHOWN
    kept, previous, joined = [], None, 0
    for block in result["parsing_res_list"]:
        if block.label in ignored:
            kept.append(block)
            continue
        if previous is not None and block.label == previous.label == "text" and _continues(previous.content, block.content):
            hyphen = previous.content.rstrip().endswith("-") and previous.content.rstrip()[-2:-1].isalpha()
            previous.content = previous.content.rstrip() + ("" if hyphen else " ") + block.content.lstrip()
            joined += 1
            continue
        kept.append(block)
        previous = block
    result["parsing_res_list"] = kept
    return joined


def _assembly_ratio(pages_markdown, merged_text: str) -> float:
    """The letters of the merged document over those of the pages it was merged from. The merged text is written without
    its images (``_IMAGE_BLOCK``), so the pages' images are left out of their count too: each is a line of letters of its
    own ("![](imgs/img_in_image_box_635_522_1116_729.jpg)" is 20), a few percent of a page of small figures."""
    pages_letters = sum(len(letters(_IMAGE_BLOCK.sub("", page))) for page in pages_markdown)
    return len(letters(merged_text)) / pages_letters if pages_letters else 0.0


_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def _tidy_tables(data: dict, markdown: str) -> str:
    """The page's table cells with a literal backslash-n made a line break and a bare ``\\pm`` made ±, in the blocks (what
    the checks read, and what the merged document is written from) and in the page's Markdown; returns the Markdown."""
    for block in data["parsing_res_list"]:
        if block["block_label"] == "table" and block.get("block_content"):
            block["block_content"] = clean_table_html(block["block_content"])
    return _TABLE.sub(lambda match: clean_table_html(match.group()), markdown)


KEPT_PAGES_SCHEMA = 1


def kept_pages(pdf: Path, report: PaperReport, pages: Dict[int, dict], markdown_pages: Dict[int, str],
               count: int) -> Optional[dict]:
    """What a refused paper's pages were read as, to judge again later instead of converting the paper again
    (``Batch.adopt``): every page's result and Markdown and the model that wrote it. None unless every page has
    a result. A page that still fails keeps the first model's reading (the retry and the second model replace a
    page only when it passes)."""
    if count < 1 or any(n not in pages for n in range(count)):
        return None
    engines = {}
    for n in range(count):
        final = report.final(n) if n in report.attempts else {}
        engines[str(n)] = final.get("engine", "paddleocr_vl") if final.get("status") != "fail" else "paddleocr_vl"
    # a JSON round trip: the dicts are the pipeline's own and restructure_pages later turns them into objects
    return json.loads(json.dumps({
        "schema": KEPT_PAGES_SCHEMA, "paper": pdf.stem, "page_count": count, "engines": engines,
        "pages": {str(n): pages[n] for n in range(count)},
        "markdown": {str(n): markdown_pages.get(n, "") for n in range(count)},
    }, ensure_ascii=False))


def _refused_blocks(verdict: PageVerdict, data: dict) -> Optional[list]:
    if verdict.status != "fail":
        return None
    return [{"label": b["block_label"], "bbox": b["block_bbox"], "content": b["block_content"]}
            for b in data["parsing_res_list"]]


@dataclass
class Batch:
    """Papers on their way through the extractor: ``convert`` fills it, ``assemble`` turns it into results."""
    pdfs: List[Path]
    t0: float
    results: Dict[Path, ExtractionResult] = field(default_factory=dict)
    reports: Dict[Path, PaperReport] = field(default_factory=dict)
    pages: Dict[Path, Dict[int, dict]] = field(default_factory=dict)
    markdown_pages: Dict[Path, Dict[int, str]] = field(default_factory=dict)
    expected: Dict[Path, int] = field(default_factory=dict)
    runnable: List[Path] = field(default_factory=list)
    todo: List[Path] = field(default_factory=list)  # converted, not yet assembled
    adopted: List[Path] = field(default_factory=list)  # judged from kept pages, not converted again

    def absorb(self, other: "Batch", replacing: List[Path]) -> None:
        """Take ``other`` (a second go at ``replacing``) into this batch in place of the first."""
        for pdf in replacing:
            self.results.pop(pdf, None)
        self.results.update(other.results)
        self.reports.update(other.reports)
        self.pages.update(other.pages)
        self.markdown_pages.update(other.markdown_pages)
        self.expected.update(other.expected)
        self.runnable += [pdf for pdf in other.runnable if pdf not in self.runnable]
        self.todo += other.todo
        self.adopted += [pdf for pdf in other.adopted if pdf not in self.adopted]

    @property
    def unresolved(self) -> List[tuple[Path, int]]:
        return [(pdf, n) for pdf in self.todo if not self.reports[pdf].error for n in self.reports[pdf].failed_pages]


class PaddleOCRVLExtractor(PdfExtractor):
    strategy_name = "paddleocr_vl"

    def __init__(
        self,
        server_url: str,
        *,
        max_concurrency: int = 16,
        katex: Optional[KatexRenderer] = None,
        ocr: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        on_progress: Optional[Callable[[], None]] = None,
        retry_sampling: Optional[dict] = None,
        engine: Optional[dict] = None,
        page_dump: Optional[Path] = None,
        pipeline_config: Optional[dict] = None,
        formula_reader: Optional[Any] = None,
        fallback: Optional[Any] = None,
        keep_pages: Optional[Callable[[Path, dict], None]] = None,
    ):
        self.server_url = server_url
        self.max_concurrency = max_concurrency
        self.katex = katex or shared_renderer()
        self.ocr = ocr  # verify.ocr.OcrReference: the witness for pages without a text layer
        self.formula_reader = formula_reader  # verify.formulanet.FormulaReader: second reader of display formulas
        self.on_progress = on_progress or (lambda: None)
        self.retry_sampling = dict(RETRY_SAMPLING if retry_sampling is None else retry_sampling)
        self.engine = engine or {}
        self.page_dump = page_dump  # directory for the model's raw page results (calibrating the checks offline)
        self.pipeline_config = pipeline_config  # PaddleX pipeline overrides (layout batch size, CPU workers...)
        self.fallback = fallback  # a reader of pages the first model got wrong (extract_many uses it; a worker may defer it)
        # Called with a refused paper's pages (``kept_pages``): a refusal is a verdict of the checks, and checks get
        # fixed; the pages are kept so the paper is judged again from them rather than converted again (ECCV 2022
        # and 2024, 2026-11-19: 1,643 papers refused by a checker that read /ActualText, and none of their pages kept).
        self.keep_pages = keep_pages
        self._pipeline = pipeline
        self._broken_layers: set[Path] = set()  # PDFs whose maths font a second reader proved the text layer cannot spell

    # ------------------------------------------------------------------ public
    def extract(self, pdf_path: Path, **kwargs: Any) -> ExtractionResult:
        return self.extract_many([pdf_path])[0]

    def extract_many(self, pdfs: List[Path]) -> List[ExtractionResult]:
        """Convert ``pdfs`` as one stream through the pipeline; one result per PDF, in order.

        A paper the stream broke on comes back FAILED with ``extra["retryable"]`` set so the
        caller can restart the server and offer it again."""
        batch = self.convert(pdfs)
        if self.fallback is not None:
            self.redo_failed_pages([batch], self.fallback)
        results = self.assemble(batch)
        return [results[pdf] for pdf in pdfs]

    def convert(self, pdfs: List[Path], kept: Optional[Dict[Path, dict]] = None) -> "Batch":
        """The first pass and the retry of ``pdfs``; ``Batch.unresolved`` lists the pages that
        still fail, for the second model (``redo_failed_pages``), before ``assemble``. A paper in ``kept``
        (``kept_pages`` of an earlier refusal) is not converted: its pages are judged again with the checks
        as they are now, and only those that still fail go on to the retry and the second model."""
        kept = kept or {}
        batch = Batch(pdfs=list(pdfs), t0=self._start())
        for pdf in pdfs:
            early = self._validate_pdf(pdf, batch.t0)
            if early is None:
                early = self._count_pages(pdf, batch.t0)
            if early is not None:
                batch.results[pdf] = early
                continue
            batch.runnable.append(pdf)
            batch.reports[pdf] = PaperReport(pdf.stem, str(pdf), engine=self.engine)
            batch.pages[pdf], batch.markdown_pages[pdf] = {}, {}
            if pdf in kept:
                self._adopt(batch, pdf, kept[pdf])

        try:
            self._stream([pdf for pdf in batch.runnable if pdf not in batch.adopted],
                         batch.reports, batch.pages, batch.markdown_pages, batch.expected)
        except KatexUnavailable:
            raise  # a broken checker is not a paper's fault and must not read as "pages failed"
        except Exception as exc:  # the server or the pipeline died under the stream
            _logger.exception("paddleocr_vl.stream_failed")
            for pdf in batch.runnable:
                if len(batch.pages[pdf]) < batch.expected.get(pdf, 1):
                    batch.reports[pdf].error = f"pipeline stream failed: {type(exc).__name__}: {str(exc)[:300]}"
                    batch.results[pdf] = self._failed(pdf, batch.reports[pdf], batch.t0, retryable=True)

        batch.todo = [pdf for pdf in batch.runnable if pdf not in batch.results]
        for pdf in self._retry_failed_pages(batch.todo, batch.reports, batch.pages, batch.markdown_pages):
            # the pipeline died under the retry (an out-of-memory 'cv' worker): the paper is not at fault
            batch.results[pdf] = self._failed(pdf, batch.reports[pdf], batch.t0, retryable=True)
        batch.todo = [pdf for pdf in batch.todo if pdf not in batch.results]
        self.recheck_failed_pages([batch])
        return batch

    def release_pipeline(self, server_url: Optional[str] = None) -> None:
        """Drop the PaddleOCR-VL pipeline (the next page loads a fresh one, pointing at ``server_url``
        if given): the layout model and what Paddle cached for it go back to the card."""
        self._pipeline = None
        if server_url:
            self.server_url = server_url
        gc.collect()

    def redo_failed_pages(self, batches: List["Batch"], reader: Any) -> None:
        """Pages that still fail go to the second model (``reader.redo(pdf, page numbers)`` returns
        page results in the same shape the pipeline's are) and through the same checks, the first model's page
        being the second reading of their display formulas; a page that now passes replaces the first model's."""
        for batch in batches:
            for pdf in batch.todo:
                report = batch.reports[pdf]
                numbers = report.failed_pages
                if not numbers or report.error:
                    continue
                try:
                    redone = reader.redo(pdf, numbers)
                except Exception as exc:
                    _logger.exception("paddleocr_vl.fallback_failed pdf=%s", pdf.name)
                    report.error = f"fallback model failed: {type(exc).__name__}: {str(exc)[:300]}"
                    continue
                for number, (data, markdown) in redone.items():
                    markdown = _tidy_tables(data, markdown)
                    verdict = self._verdict(pdf, number, data, peer=batch.pages[pdf].get(number))
                    report.record(verdict, {"engine": reader.name}, _refused_blocks(verdict, data), engine=reader.name)
                    if verdict.status != "fail":
                        batch.pages[pdf][number], batch.markdown_pages[pdf][number] = data, markdown
                    self.on_progress()

    def assemble(self, batch: "Batch", skip=(), only=None) -> Dict[Path, ExtractionResult]:
        """The results of the papers of ``batch`` that are not in ``skip`` (waiting for the second
        model) or, when ``only`` is given, only of those."""
        for pdf in batch.todo:
            if pdf in skip or (only is not None and pdf not in only) or pdf in batch.results:
                continue
            report = batch.reports[pdf]
            if self.keep_pages is not None and report.status == "failed":
                # before _assemble: on its way it turns the pages' dicts into the pipeline's objects
                pages = kept_pages(pdf, report, batch.pages[pdf], batch.markdown_pages[pdf], batch.expected.get(pdf, 0))
                if pages is not None:
                    try:
                        self.keep_pages(pdf, pages)
                    except Exception:  # keeping them is a saving for later; the verdict stands without it
                        _logger.exception("paddleocr_vl.keep_pages_failed pdf=%s", pdf.name)
            try:
                batch.results[pdf] = self._assemble(pdf, batch.reports[pdf], batch.pages[pdf], batch.markdown_pages[pdf],
                                                    batch.expected[pdf], batch.t0)
            except Exception as exc:  # one paper's merge must not take the other papers of the batch with it
                _logger.exception("paddleocr_vl.assemble_failed pdf=%s", pdf.name)
                batch.reports[pdf].error = f"merging the pages failed: {type(exc).__name__}: {str(exc)[:300]}"
                batch.results[pdf] = self._failed(pdf, batch.reports[pdf], batch.t0)
        return {pdf: batch.results[pdf] for pdf in batch.pdfs
                if pdf in batch.results and pdf not in skip and (only is None or pdf in only)}

    # --------------------------------------------------------------- internals
    def _load_pipeline(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            # Chart recognition is off: it would write numbers the model made up, which
            # the PDF's text layer cannot contradict.
            self._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server", vl_rec_server_url=self.server_url,
                vl_rec_max_concurrency=self.max_concurrency, use_chart_recognition=False, use_queues=True,
                paddlex_config=self._pipeline_config_tree(),
            )
        return self._pipeline

    def _pipeline_config_tree(self) -> Optional[dict]:
        """PaddleOCR takes ``paddlex_config`` as the whole pipeline tree, so the overrides are merged
        into the stock PaddleOCR-VL-1.6 one here."""
        if not self.pipeline_config:
            return None
        from paddlex.inference import load_pipeline_config

        def merge(base: dict, extra: dict) -> dict:
            return {**base, **{k: merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
                               for k, v in extra.items()}}

        return merge(load_pipeline_config("PaddleOCR-VL-1.6"), self.pipeline_config)

    def _count_pages(self, pdf: Path, t0: float) -> Optional[ExtractionResult]:
        import pymupdf

        try:
            with pymupdf.open(pdf) as document:
                if document.needs_pass or document.page_count < 1:
                    raise ValueError("encrypted PDF" if document.needs_pass else "PDF has no pages")
        except Exception as exc:
            return ExtractionResult(pdf_path=pdf, status=ExtractionStatus.FAILED, strategy=self.strategy_name,
                                    error=f"unreadable PDF: {exc}", elapsed_ms=self._elapsed_ms(t0))
        return None

    def _stream(self, runnable, reports, pages, markdown_pages, expected) -> None:
        if not runnable:
            return
        by_path = {str(pdf): pdf for pdf in runnable}
        for result in self._load_pipeline().predict_iter([str(pdf) for pdf in runnable]):
            data = result.json["res"]
            pdf = by_path[data["input_path"]]
            expected[pdf] = data["page_count"]
            number = data["page_index"]
            pages[pdf][number] = data
            markdown_pages[pdf][number] = _tidy_tables(data, _plain_markdown(result))
            del result  # page images are not needed past this point
            if self.page_dump:
                self._dump(pdf, number, data)
            verdict = self._verdict(pdf, number, data)
            reports[pdf].record(verdict, blocks=_refused_blocks(verdict, data))
            self.on_progress()

    def _adopt(self, batch: "Batch", pdf: Path, kept: dict) -> None:
        """The pages of ``kept`` stand in for the first pass of ``pdf``, each judged by the checks as they are now."""
        count = int(kept["page_count"])
        batch.expected[pdf] = count
        batch.adopted.append(pdf)
        for number in range(count):
            data, markdown = kept["pages"][str(number)], kept["markdown"][str(number)]
            engine = kept.get("engines", {}).get(str(number), "paddleocr_vl")
            batch.pages[pdf][number], batch.markdown_pages[pdf][number] = data, markdown
            verdict = self._verdict(pdf, number, data)
            batch.reports[pdf].record(verdict, {"kept": True}, _refused_blocks(verdict, data), engine=engine)
            self.on_progress()

    def _dump(self, pdf: Path, number: int, data: dict) -> None:
        self.page_dump.mkdir(parents=True, exist_ok=True)
        (self.page_dump / f"{pdf.stem}.p{number}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _verdict(self, pdf: Path, number: int, data: dict, peer: Optional[dict] = None) -> PageVerdict:
        """``peer`` is the other engine's page for the same PDF page: with no formula reader, its display formulas are
        the second reading (verify/peer.py)."""
        ref: PageRef = page_reference(pdf, number)
        if not ref.usable and self.ocr is not None:
            ref = self.ocr.page_reference(pdf, number)
        witness = None
        if ref.source == "textlayer":
            if self.formula_reader is not None:
                witness = self._witness(pdf, number)
            elif peer is not None:
                witness = peer_witness(peer, ref.width)
        verdict = check_page(parse_page(data, ref.width / data["width"], number), ref, self.katex, witness,
                             math_layer_broken=pdf in self._broken_layers)
        if verdict.metrics.get("formulas_witnessed"):
            self._broken_layers.add(pdf)
        return verdict

    def recheck_failed_pages(self, batches: List["Batch"]) -> None:
        """The pages that still fail are judged again, now with the second reader of formulas
        (``formula_reader``) and with what it found in the PDF's other pages: a maths font it proved
        the text layer cannot spell may have failed an earlier page on an inline formula. A worker
        that keeps the card for the model server judges them with no reader first (the reader runs
        on the card, once the server is gone) and calls this when it has one."""
        if self.formula_reader is None:
            return
        for batch in batches:
            for pdf in batch.todo:
                report = batch.reports[pdf]
                for number in report.failed_pages:
                    data = batch.pages[pdf].get(number)
                    if data is None:
                        continue
                    verdict = self._verdict(pdf, number, data)
                    if verdict.status != "fail":
                        report.record(verdict, {"formula_witness": True, "math_layer_broken": pdf in self._broken_layers})
                        self.on_progress()

    def _witness(self, pdf: Path, number: int) -> Callable[[tuple], Optional[str]]:
        """The second reader of a display formula's box. When it cannot read (the model is not
        downloaded, a CUDA error) the formula stays judged by the text layer alone, which
        refuses it: the failure is logged, and never turns a page's verdict into an exception."""
        def read(box):
            try:
                return self.formula_reader.latex(pdf, number, box)
            except Exception:
                _logger.exception("paddleocr_vl.formula_reader_failed pdf=%s page=%d", pdf.name, number + 1)
                return None
        return read

    def _retry_failed_pages(self, todo, reports, pages, markdown_pages) -> List[Path]:
        """One more conversion, with the retry sampling, of every page that failed its check. Returns the
        papers the pipeline broke on while doing it (their reports say why)."""
        failed = [(pdf, n) for pdf in todo for n in reports[pdf].failed_pages]
        if not failed:
            return []
        import pymupdf

        with tempfile.TemporaryDirectory(prefix="pvl_retry_") as scratch:
            singles: Dict[str, tuple[Path, int]] = {}
            for index, (pdf, number) in enumerate(failed):
                single = Path(scratch) / f"{index:05d}.pdf"
                with pymupdf.open(pdf) as source, pymupdf.open() as target:
                    target.insert_pdf(source, from_page=number, to_page=number)
                    target.save(single)
                singles[str(single)] = (pdf, number)
            try:
                for result in self._load_pipeline().predict_iter(list(singles), **self.retry_sampling):
                    data = result.json["res"]
                    pdf, number = singles[data["input_path"]]
                    data["page_index"] = number
                    markdown = _tidy_tables(data, _plain_markdown(result))
                    del result
                    verdict = self._verdict(pdf, number, data)
                    reports[pdf].record(verdict, self.retry_sampling, _refused_blocks(verdict, data))
                    if verdict.status != "fail":
                        pages[pdf][number], markdown_pages[pdf][number] = data, markdown
                    self.on_progress()
            except Exception as exc:
                _logger.exception("paddleocr_vl.retry_failed")
                for pdf, _ in failed:
                    reports[pdf].error = reports[pdf].error or f"page retry failed: {type(exc).__name__}: {str(exc)[:300]}"
                return list(dict.fromkeys(pdf for pdf, _ in failed))
        return []

    def _assemble(self, pdf: Path, report: PaperReport, pages: Dict[int, dict], markdown_pages: Dict[int, str],
                  count: int, t0: float) -> ExtractionResult:
        report.seconds = self._elapsed_ms(t0) / 1000
        missing = [n for n in range(count) if n not in pages]
        if missing:
            report.error = f"no result for page(s) {[n + 1 for n in missing[:10]]}"
        if report.status == "failed":
            return self._failed(pdf, report, t0)
        ordered = [pages[n] for n in range(count)]
        # restructure_pages concatenates into the FIRST page's dict, and the result reads its "input_path" when it
        # writes Markdown: a first page the second model wrote (blocks_to_page) or the retry converted has none
        ordered[0].setdefault("input_path", str(pdf))
        # restructure_pages replaces each page's block dicts with block objects in place
        has_images = any(block["block_label"] in ("image", "chart") for data in ordered
                         for block in data["parsing_res_list"])
        merged = self._load_pipeline().restructure_pages(
            [{"res": data} for data in ordered], merge_tables=True, relevel_titles=True, concatenate_pages=True)
        joined = _join_split_paragraphs(merged[0])
        if joined:
            _logger.info("paddleocr_vl.paragraphs_joined pdf=%s joined=%d", pdf.name, joined)
        text = _IMAGE_BLOCK.sub("", _plain_markdown(merged[0])).strip() + "\n"
        ratio = _assembly_ratio((markdown_pages[n] for n in range(count)), text)
        if not ASSEMBLY_LETTER_RATIO[0] <= ratio <= ASSEMBLY_LETTER_RATIO[1]:
            report.error = f"merging the pages changed the text: {ratio:.3f} of the pages' letters remain"
            return self._failed(pdf, report, t0)
        return ExtractionResult(
            pdf_path=pdf, status=ExtractionStatus.SUCCESS, strategy=self.strategy_name, markdown=text,
            elapsed_ms=self._elapsed_ms(t0), page_count=count,
            has_equations="$" in text, has_tables="<table" in text,
            has_images=has_images,
            extra={"report": report.to_json()},
        )

    def _failed(self, pdf: Path, report: PaperReport, t0: float, *, retryable: bool = False) -> ExtractionResult:
        report.seconds = self._elapsed_ms(t0) / 1000
        return ExtractionResult(
            pdf_path=pdf, status=ExtractionStatus.FAILED, strategy=self.strategy_name,
            error="; ".join(report.failure_reasons())[:2000], elapsed_ms=self._elapsed_ms(t0),
            extra={"report": report.to_json(), "retryable": retryable},
        )
