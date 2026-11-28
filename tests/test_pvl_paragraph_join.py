"""A paragraph the layout cuts in two comes back as two text blocks; restructure_pages joins tables and titles across
pages, never text. task 1302 (VLDB 2022): 1,008 paragraphs of the 279 written Markdown files end mid-sentence and
the next one starts in lowercase, in 263 papers; 568 of them at a page break, 60 at a column break, 262 inside a column
(around a figure the Markdown does not keep). The blocks below are real: refused pages keep theirs in the quality
report, and the page break is the paragraphs of vldb-2022-p72-li.md as the pipeline wrote them.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.paper_crawler.converters.extractors import paddleocr_vl
from pipeline.paper_crawler.converters.extractors.paddleocr_vl import PaddleOCRVLExtractor, _join_split_paragraphs
from pipeline.paper_crawler.converters.verify.checks import PageVerdict
from pipeline.paper_crawler.converters.verify.report import PaperReport

CASES = json.loads((Path(__file__).parent / "fixtures" / "pvl" / "split_paragraphs.json").read_text(encoding="utf-8"))
IGNORED = ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"]


def merged_result(blocks):
    return {"model_settings": {"markdown_ignore_labels": IGNORED},
            "parsing_res_list": [SimpleNamespace(label=b["label"], content=b["content"]) for b in blocks]}


def joined(blocks):
    result = merged_result(blocks)
    return _join_split_paragraphs(result), [(b.label, b.content) for b in result["parsing_res_list"]]


def text(content):
    return {"label": "text", "content": content}


FIRST = "Single-Model follows existing solution that leverages one estimator to learn the distribution of each table and joins of"
SECOND = "multiple tables in the schema. We first generate a full-join table."


def test_a_paragraph_split_by_a_page_break_is_one_paragraph_again():
    count, out = joined(CASES["page_break_in_the_written_markdown"]["blocks"])
    assert count == 1 and len(out) == 3
    assert "joins of multiple tables in the schema." in out[1][1] and "\n" not in out[1][1]
    assert out[0][1].endswith("Single-Model and Multi-models.")  # the paragraph before keeps its own end


def test_a_paragraph_split_at_a_column_break_inside_a_real_page_is_joined():
    case = CASES["column_break_in_a_page"]
    count, out = joined(case["blocks"])
    assert count == 1 and len(out) == len(case["blocks"]) - 1
    assert any("the size of the resulting library" in content for _, content in out)


def test_a_real_page_with_formulas_around_the_split_joins_only_the_split():
    case = CASES["column_break_next_to_formulas"]
    count, out = joined(case["blocks"])
    assert count == 1
    assert [b for b in case["blocks"] if b["label"] != "text"] == [
        {"label": label, "content": content} for label, content in out if label != "text"]  # nothing else is touched


def test_the_lines_of_a_byline_are_not_one_paragraph():
    # "Burnaby, BC, Canada" then "j.pei@duke.edu": a lowercase start that is an address, not a continuation
    case = CASES["byline_lines_are_not_one_paragraph"]
    count, out = joined(case["blocks"])
    assert count == 0 and len(out) == len(case["blocks"])


def test_page_furniture_and_images_are_looked_through_a_formula_or_a_title_is_not():
    for between, expected in (("number", 1), ("header", 1), ("footer", 1), ("image", 1),
                              ("display_formula", 0), ("paragraph_title", 0), ("table", 0), ("figure_title", 0)):
        count, out = joined([text(FIRST), {"label": between, "content": "x"}, text(SECOND)])
        assert count == expected, between
        assert len(out) == (2 if expected else 3), between


@pytest.mark.parametrize("first,second,expected", [
    (FIRST, SECOND, True),
    (FIRST.replace("joins of", "joins of (see [5])"), SECOND, True),            # no sentence end after the brackets
    (FIRST.replace("joins of", "joins."), SECOND, False),                       # a finished sentence
    (FIRST.replace("joins of", "as follows:"), SECOND, False),
    (FIRST, SECOND.capitalize(), False),                                        # the next paragraph starts a sentence
    ("Burnaby, BC, Canada", "j.pei@duke.edu", False),                           # too short to be a paragraph
    (FIRST, "j.pei@duke.edu and more", False),                                  # an address
    (FIRST, "https://example.org/a is the repository", False),
    (FIRST, "\\( x \\) is the cost", False),                                    # a formula opens it: not a lowercase word
])
def test_what_counts_as_a_paragraph_that_goes_on(first, second, expected):
    assert paddleocr_vl._continues(first, second) is expected


def test_a_word_broken_at_the_hyphen_is_joined_without_a_space():
    count, out = joined([text(FIRST.replace("joins of", "multi-")), text("level indexes are built.")])
    assert count == 1 and out[0][1].endswith("multi-level indexes are built.")


def test_only_text_blocks_are_joined():
    for label in ("figure_title", "reference_content", "algorithm", "paragraph_title"):
        count, _ = joined([{"label": label, "content": FIRST}, {"label": label, "content": SECOND}])
        assert count == 0, label


def test_three_pieces_of_one_paragraph_come_back_as_one():
    count, out = joined([text(FIRST), text("multiple tables in the schema, which are joined along the keys of both"),
                         text("sides before anything is trained.")])
    assert count == 2 and len(out) == 1
    assert out[0][1].endswith("keys of both sides before anything is trained.")


class FakePipeline:
    """What restructure_pages gives back, with the one property the extractor relies on: the text of the pages'
    blocks in one document. paddlex is not installed where the tests run."""

    def restructure_pages(self, pages, **kwargs):
        blocks = [SimpleNamespace(label=b["block_label"], content=b["block_content"])
                  for page in pages for b in page["res"]["parsing_res_list"]]
        settings = pages[0]["res"].get("model_settings", {})

        class Merged(dict):
            def _to_markdown(self, pretty=False):
                shown = [b.content for b in self["parsing_res_list"] if b.label not in self["model_settings"]["markdown_ignore_labels"]]
                return {"markdown_texts": "\n\n".join(shown)}

        return [Merged(parsing_res_list=blocks, model_settings=settings)]


def test_the_assembled_paper_has_the_split_paragraph_whole():
    case = CASES["column_break_in_a_page"]
    blocks = [{"block_label": b["label"], "block_bbox": [0, 0, 1, 1], "block_content": b["content"]} for b in case["blocks"]]
    cut = next(i for i, b in enumerate(case["blocks"]) if b["content"].startswith("the resulting library"))
    pages = {0: {"parsing_res_list": blocks[:cut], "model_settings": {"markdown_ignore_labels": IGNORED}},
             1: {"parsing_res_list": blocks[cut:], "model_settings": {"markdown_ignore_labels": IGNORED}}}
    markdown_pages = {n: "\n\n".join(b["block_content"] for b in page["parsing_res_list"]) for n, page in pages.items()}
    pdf = Path("p3535.pdf")
    extractor = PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=object(), pipeline=FakePipeline())
    report = PaperReport(pdf.stem, str(pdf))
    for number in pages:
        report.record(PageVerdict(number, "pass"))  # both pages stood up to their checks
    result = extractor._assemble(pdf, report, pages, markdown_pages, 2, time.perf_counter())
    assert result.markdown and "the size of the resulting library" in result.markdown
    assert "size of\n\nthe resulting" not in result.markdown
