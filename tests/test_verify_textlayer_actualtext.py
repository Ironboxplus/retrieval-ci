"""The verifier's reference is the glyphs printed on the page, not a PDF's /ActualText replacement for them.

The ECCV 2024 template attaches each formula's LaTeX source to its glyphs as /ActualText. MuPDF reports that
replacement by default, one character per glyph and the rest on the last one, so the layer of eccv-2024-05783 page 7
reads ``\\m a thca l {L} _ {r = \\| \\hat {I}_t ...`` where the page prints ``Lrec = ∥ˆIt − It∥1``. Every such page was
refused for "omission" (page text ``...mathcallreclambdasumvgg...`` missing from the Markdown) and "formula_symbol":
task 2499 refused 742 of the first 1200 ECCV 2024 papers, task 1325 (ECCV 2022 run) 901 of 1258.

Every page here is a real page of an ECCV 2024 PDF cut out with its fonts (tests/fixtures/pvl/eccv2024/*.pdf) with the
blocks PaddleOCR-VL-1.6 returned for it on .88 (the first attempt kept in the quality report, at 2 pixels per point).
Needs node with KaTeX (``npm ci`` in tools/katex_check), like tests/test_verify_textlayer_math_fonts.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from pipeline.paper_crawler.converters.verify import checks, mdtext, textlayer
from pipeline.paper_crawler.converters.verify.katex import KatexRenderer

ECCV = Path(__file__).resolve().parent / "fixtures" / "pvl" / "eccv2024"
PAGES = ["eccv-2024-05783_p7", "eccv-2024-05774_p6", "eccv-2024-05794_p5"]


@pytest.fixture(scope="module")
def katex():
    renderer = KatexRenderer()
    yield renderer
    renderer.close()


def verdict(katex, stem: str) -> checks.PageVerdict:
    record = json.loads((ECCV / f"{stem}.json").read_text(encoding="utf-8"))
    ref, data = textlayer.page_reference(ECCV / f"{stem}.pdf", 0), record["result"]
    return checks.check_page(mdtext.parse_page(data, ref.width / data["width"], 0), ref, katex)


@pytest.mark.parametrize("stem", PAGES)
def test_the_reference_has_no_latex_source_of_actualtext(stem):
    with pymupdf.open(ECCV / f"{stem}.pdf") as document:
        page = document[0]
        # the fixture really carries it: MuPDF's default extraction has the LaTeX source
        assert any("\\" in word[4] for word in page.get_text("words"))
        words = textlayer.page_words(page)
    assert not any("\\" in word[4] for word in words), [w[4] for w in words if "\\" in w[4]][:10]


def test_the_reference_is_the_printed_formula():
    ref = textlayer.page_reference(ECCV / "eccv-2024-05783_p7.pdf", 0)
    assert "lrec" in ref.compact and "mathcal" not in ref.compact


@pytest.mark.parametrize("stem", PAGES)
def test_a_page_the_model_read_right_is_not_refused_for_the_actualtext(katex, stem):
    v = verdict(katex, stem)
    assert not any(r.startswith(("omission", "formula_symbol")) for r in v.reasons), v.reasons
