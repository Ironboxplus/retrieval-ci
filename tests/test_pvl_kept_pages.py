"""A refused paper keeps its pages, and the next run judges them again instead of converting the paper again.

ECCV 2022 and 2024 (tasks 1325 and 2499, 2026-11-18): the checks refused 1,643 papers on a reference that read the
PDFs' /ActualText (tests/test_verify_textlayer_actualtext.py). The quality report keeps the model's blocks only for
the pages that failed, so once the checks were fixed every one of those papers had to go through the model again,
about a day of the card. A refusal is the checks' verdict and checks get fixed: what the model read must survive it.

The extractor runs for real with its real checks and KaTeX; only the PaddleOCR-VL pipeline (it needs the card) is
scripted, as in tests/test_pvl_worker_flow.py. The worker part runs the real ``main()`` through the harness.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.test_pvl_worker_flow import kinds, run_worker

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pvl"


@pytest.fixture(scope="module")
def katex():
    from pipeline.paper_crawler.converters.verify.katex import shared_renderer

    return shared_renderer()


def _page(data: dict):
    class Page:
        json = {"res": data}

        def _to_markdown(self, pretty=False):
            return {"markdown_texts": ""}

    return Page()


class ScriptedPipeline:
    """Answers every predict_iter with ``data`` (a page the checks refuse stays refused) and records the calls."""

    def __init__(self, data: dict):
        self.data = data
        self.calls: list[tuple[list, dict]] = []

    def predict_iter(self, paths, **kwargs):
        self.calls.append((list(paths), kwargs))
        for path in paths:
            yield _page({**json.loads(json.dumps(self.data)), "input_path": path})


def _unrescued():
    pdf = FIXTURES / "unrescued.pdf"
    data = json.loads((FIXTURES / "unrescued.page0.pvl.json").read_text(encoding="utf-8"))
    data.update(input_path=str(pdf), page_count=1, page_index=0)
    return pdf, data


def _extractor(katex, pipeline, **kwargs):
    from pipeline.paper_crawler.converters.extractors.paddleocr_vl import PaddleOCRVLExtractor

    return PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex, pipeline=pipeline, **kwargs)


def test_a_refused_paper_hands_every_page_it_was_read_as_to_keep_pages(katex):
    pdf, data = _unrescued()
    kept = {}
    extractor = _extractor(katex, ScriptedPipeline(data), keep_pages=lambda p, pages: kept.setdefault(p, pages))
    result = extractor.assemble(extractor.convert([pdf]))[pdf]
    assert not result.succeeded
    assert set(kept) == {pdf}
    pages = kept[pdf]
    assert pages["page_count"] == 1 and set(pages["pages"]) == {"0"} and set(pages["markdown"]) == {"0"}
    assert pages["pages"]["0"]["parsing_res_list"] == json.loads(json.dumps(data["parsing_res_list"]))
    json.dumps(pages)  # what the worker writes to disk


def test_a_kept_paper_the_checks_now_pass_is_not_converted_again(katex):
    # eccv-2024-05783 p7 was refused by the /ActualText reference; the fixed checks pass what the model read
    record = json.loads((FIXTURES / "eccv2024" / "eccv-2024-05783_p7.json").read_text(encoding="utf-8"))
    pdf = FIXTURES / "eccv2024" / "eccv-2024-05783_p7.pdf"
    kept = {"schema": 1, "paper": pdf.stem, "page_count": 1, "engines": {"0": "paddleocr_vl"},
            "pages": {"0": record["result"]}, "markdown": {"0": ""}}
    pipeline = ScriptedPipeline({})
    batch = _extractor(katex, pipeline).convert([pdf], {pdf: kept})
    assert pipeline.calls == []  # not a page went to the model
    assert batch.todo == [pdf] and batch.reports[pdf].failed_pages == [] and pdf not in batch.results
    assert batch.reports[pdf].final(0)["sampling"] == {"kept": True}


def test_a_kept_page_that_still_fails_goes_alone_to_the_retry(katex):
    pdf, data = _unrescued()
    kept = {"schema": 1, "paper": pdf.stem, "page_count": 1, "engines": {"0": "paddleocr_vl"},
            "pages": {"0": data}, "markdown": {"0": ""}}
    pipeline = ScriptedPipeline(data)
    extractor = _extractor(katex, pipeline)
    batch = extractor.convert([pdf], {pdf: kept})
    # no first pass of the paper: one call, the retry of the page that still fails, with the retry's sampling
    assert len(pipeline.calls) == 1
    paths, kwargs = pipeline.calls[0]
    assert len(paths) == 1 and Path(paths[0]).name != pdf.name and kwargs == extractor.retry_sampling
    assert batch.reports[pdf].failed_pages == [0]


def test_the_worker_keeps_a_refused_papers_pages_and_offers_them_on_the_next_run(tmp_path):
    done, events, _, written = run_worker(tmp_path, 4, {"refused": ["p01"]})
    assert written == ["p00", "p02", "p03"], done.stderr[-2000:]
    path = tmp_path / "md" / "_pages" / "p01.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        assert json.load(fh) == {"schema": 1, "paper": "p01", "page_count": 5}

    done, events, _, written = run_worker(tmp_path, 4, {})
    second = kinds(events, "convert")[-1]
    assert second["papers"] == ["p01"] and second["kept"] == {"p01": {"schema": 1, "paper": "p01", "page_count": 5}}
    assert written == ["p00", "p01", "p02", "p03"], done.stderr[-2000:]
    assert not path.exists()  # written: nothing left to judge again
