"""Errors inside a reference list are recorded and do not refuse a page (user rule, 2026-11-15).

Real verifier, real pages: ``plain.pdf`` page 1 (VLDB 2022) with the model's real answer for it, one block relabelled
``reference_content`` (the label PaddleOCR-VL gives a reference list; MinerU's ``ref_text`` is mapped to it) and a fault
injected into that block. The same fault in a ``text`` block must still refuse the page: the body's checks are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.paper_crawler.converters.verify import checks, mdtext, textlayer
from pipeline.paper_crawler.converters.verify.katex import KatexRenderer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pvl"
FIGURES = "pp. 62062–62076."  # the page range the model invents in a dense reference list: not in the text layer
FOREIGN = " ".join(
    "Zebra crossings painted along the harbour wall remind sailors that migrating herons "
    "never negotiate fares with lighthouse keepers during stormy marmalade seasons".split())
# eight words of three letters: one gram of 24 letters, which the page does not have, written 23 times
LOOP = " ".join(["the and for are but not you all"] * 23)


@pytest.fixture(scope="module")
def katex():
    renderer = KatexRenderer()
    yield renderer
    renderer.close()


def page_data() -> dict:
    return json.loads((FIXTURES / "plain.page1.json").read_text(encoding="utf-8"))


def text_blocks(data: dict) -> list[dict]:
    blocks = [b for b in data["parsing_res_list"] if b["block_label"] == "text" and b["block_content"]]
    return sorted(blocks, key=lambda b: len(b["block_content"]), reverse=True)


def verdict(katex, data: dict) -> checks.PageVerdict:
    ref = textlayer.page_reference(FIXTURES / "plain.pdf", 1)
    return checks.check_page(mdtext.parse_page(data, ref.width / data["width"], 1), ref, katex)


def with_fault(label: str, fault: str, *, body_fault: str = "") -> dict:
    """The page with ``fault`` appended to its longest block, labelled ``label``; ``body_fault`` goes to the second
    longest block, which stays a ``text`` block."""
    data = page_data()
    first, second = text_blocks(data)[:2]
    first["block_label"] = label
    first["block_content"] += " " + fault
    if body_fault:
        second["block_content"] += " " + body_fault
    return data


def test_the_page_without_a_fault_passes_with_its_longest_block_a_reference_block(katex):
    v = verdict(katex, with_fault("reference_content", ""))
    assert v.status == "pass" and not v.reasons and not v.soft, (v.reasons, v.soft)
    assert "reference_fabrication_min_coverage" in v.metrics


def test_a_page_range_the_page_does_not_have_inside_a_reference_block_is_recorded(katex):
    ref = textlayer.page_reference(FIXTURES / "plain.pdf", 1)
    assert "62062" not in "".join(w.text for w in ref.words)
    v = verdict(katex, with_fault("reference_content", FIGURES))
    assert v.status == "pass" and not v.reasons, v.reasons
    assert any(s.startswith("reference list: number: not on the page") and "62062" in s for s in v.soft), v.soft
    assert v.metrics["numbers_unknown"] == 0 and v.metrics["reference_numbers_unknown"] >= 2


def test_the_same_figures_in_a_text_block_refuse_the_page(katex):
    v = verdict(katex, with_fault("text", FIGURES))
    assert v.status == "fail" and any(r.startswith("number: not on the page") and "62062" in r for r in v.reasons), v.reasons
    assert not any(s.startswith("reference list:") for s in v.soft)
    # the one thing that differs is the label of the block
    assert verdict(katex, with_fault("reference_content", FIGURES)).status == "pass"


def test_a_figure_in_a_body_block_and_in_a_reference_block_refuses_the_page(katex):
    v = verdict(katex, with_fault("reference_content", FIGURES, body_fault="See pp. 62062."))
    assert v.status == "fail" and any(r.startswith("number: not on the page") and "62062" in r for r in v.reasons), v.reasons
    assert v.metrics["numbers_unknown"] >= 1
    # without the body's copy only the reference list has it: recorded
    assert verdict(katex, with_fault("reference_content", FIGURES)).status == "pass"


def test_a_sentence_the_page_does_not_have_inside_a_reference_block_is_recorded(katex):
    v = verdict(katex, with_fault("reference_content", FOREIGN))
    assert v.status == "pass" and not v.reasons, v.reasons
    assert any(s.startswith("reference list: fabrication") for s in v.soft), v.soft
    assert v.metrics["reference_fabrication_min_coverage"] < checks.FABRICATION_MIN_COVERAGE
    assert v.metrics["fabrication_min_coverage"] >= checks.FABRICATION_MIN_COVERAGE
    refused = verdict(katex, with_fault("text", FOREIGN))
    assert refused.status == "fail" and any(r.startswith("fabrication") for r in refused.reasons), refused.reasons


def test_a_gram_repeated_inside_a_reference_block_is_recorded(katex):
    v = verdict(katex, with_fault("reference_content", LOOP))
    assert v.status == "pass" and not v.reasons, v.reasons
    assert any(s.startswith("reference list: repetition") and "appears 2" in s for s in v.soft), v.soft
    assert v.metrics["repeated_ngrams"] == 0 and v.metrics["reference_repeated_ngrams"] >= 1
    refused = verdict(katex, with_fault("text", LOOP))
    assert refused.status == "fail" and any(r.startswith("repetition") for r in refused.reasons), refused.reasons
