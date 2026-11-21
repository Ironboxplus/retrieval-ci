"""Verification of PaddleOCR-VL Markdown against the PDF's own text layer.

Real data throughout: the PDFs under tests/fixtures/pvl are pages cut from VLDB 2022
papers, and the ``*.pageN.json`` / ``*.pageN.md`` files are what PaddleOCR-VL-1.6 (vLLM,
full pipeline) actually returned for them on the 4070 SUPER. Some of those real outputs
are wrong - the model wrote a tau for a T, a malformed LaTeX array, ``v_10`` for ``v_{10}``
- and the verifier must say so; the ``.retry`` outputs are the same pages converted again
with a repetition penalty, which is what the extractor does. Faults that the model did not
happen to make are injected into a passing page.

Needs node with KaTeX (``npm ci`` in tools/katex_check): the checker is the one the frontend renders with.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.paper_crawler.converters.extractors import paddleocr_vl
from pipeline.paper_crawler.converters.verify import checks, mdtext, normalize, peer, textlayer
from pipeline.paper_crawler.converters.verify.katex import KatexRenderer, KatexUnavailable
from pipeline.paper_crawler.converters.verify.report import PaperReport

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pvl"


@pytest.fixture(scope="module")
def katex():
    renderer = KatexRenderer()
    yield renderer
    renderer.close()


def load(stem: str, page: int, variant: str = "") -> dict:
    return json.loads((FIXTURES / f"{stem}.page{page}{variant}.json").read_text(encoding="utf-8"))


def reference(stem: str, page: int) -> textlayer.PageRef:
    if stem == "scan":
        ocr = json.loads((FIXTURES / "scan.page0.ocr.json").read_text(encoding="utf-8"))
        return textlayer.PageRef.from_lines(
            0, ocr["width"], ocr["height"], [(line["text"], tuple(line["bbox"])) for line in ocr["lines"]], "ocr")
    return textlayer.page_reference(FIXTURES / f"{stem}.pdf", page)


def verdict(katex, stem: str, page: int, data: dict, ref=None) -> checks.PageVerdict:
    ref = ref or reference(stem, page)
    return checks.check_page(mdtext.parse_page(data, ref.width / data["width"], page), ref, katex)


def codes(v: checks.PageVerdict) -> set[str]:
    return {reason.split(":")[0] for reason in v.reasons}


def blocks(data: dict, label: str = "text"):
    return [b for b in data["parsing_res_list"] if b["block_label"] == label and b["block_content"]]


def longest(data: dict) -> dict:
    return max(blocks(data), key=lambda b: len(b["block_content"]))


# ------------------------------------------------------------ real output passes
@pytest.mark.parametrize("stem,page,variant", [
    ("plain", 0, ""), ("plain", 1, ""), ("plain", 2, ""),
    ("table", 0, ""), ("table", 1, ""), ("table", 2, ""),
    ("formula", 1, ".retry"), ("formula", 2, ".retry"),
    ("scan", 0, ""),
])
def test_real_pvl_page_output_passes(katex, stem, page, variant):
    v = verdict(katex, stem, page, load(stem, page, variant))
    assert v.status == "pass", v.reasons


def test_scanned_page_is_checked_against_ocr_and_never_against_formula_symbols(katex):
    v = verdict(katex, "scan", 0, load("scan", 0))
    assert v.metrics["reference"] == "ocr"
    assert v.metrics["formulas"] >= 1 and v.metrics["formulas_unverifiable"] == v.metrics["formulas"]


def test_page_without_text_layer_and_without_ocr_is_unverifiable(katex):
    ref = textlayer.page_reference(FIXTURES / "scan.pdf", 0)
    assert not ref.usable and ref.source == "none" and "0 words" in ref.reason
    v = verdict(katex, "scan", 0, load("scan", 0), ref=ref)
    assert v.status == "unverifiable" and v.reasons[0].startswith("no_reference")


def test_page_without_reference_records_a_formula_katex_rejects(katex):
    data = load("scan", 0)
    block = next(b for b in blocks(data) if "$" in b["block_content"])
    block["block_content"] = block["block_content"].replace("$", " $\\frac{a}{b$ ", 1)
    ref = textlayer.page_reference(FIXTURES / "scan.pdf", 0)
    v = verdict(katex, "scan", 0, data, ref=ref)
    assert v.status == "unverifiable" and any(s.startswith("formula_render") for s in v.soft)


@pytest.mark.parametrize("stem,pages", [("plain", 3), ("table", 3), ("formula", 3)])
def test_the_text_layer_turned_into_markdown_passes(katex, stem, pages):
    import pymupdf

    with pymupdf.open(FIXTURES / f"{stem}.pdf") as document:
        for number in range(pages):
            page = document[number]
            items = [{"block_id": i, "group_id": i, "block_label": "text", "block_bbox": list(b[:4]),
                      "block_content": b[4]} for i, b in enumerate(page.get_text("blocks")) if b[6] == 0]
            data = {"page_index": number, "width": page.rect.width, "height": page.rect.height,
                    "model_settings": {"markdown_ignore_labels": []}, "parsing_res_list": items}
            v = verdict(katex, stem, number, data)
            assert v.status == "pass", (stem, number, v.reasons)


# ----------------------------------------------- look-alikes pass, soft issues are recorded, real errors are refused
def test_a_tau_where_the_pdf_has_a_T_is_the_same_symbol_to_a_reader(katex):
    # the model wrote tau_{KECC}, the PDF's font gives the layer a T: a look-alike, accepted silently
    for variant in ("", ".retry"):
        v = verdict(katex, "formula", 0, load("formula", 0, variant))
        assert v.status == "pass" and not v.reasons and not v.soft, v


def test_a_latex_array_katex_cannot_parse_is_recorded_and_the_page_stands(katex):
    v = verdict(katex, "formula", 1, load("formula", 1))
    assert v.status == "pass" and not v.reasons
    assert any(s.startswith("formula_render") and "\\end" in s for s in v.soft), v.soft


def test_v_10_that_renders_as_v_1_then_0_is_refused(katex):
    v = verdict(katex, "formula", 2, load("formula", 2))
    assert v.status == "fail" and codes(v) == {"number"} and "'0'" in v.reasons[0]


# ------------------------------------------------------------ injected faults
FOREIGN = " ".join(
    "Zebra crossings painted along the harbour wall remind sailors that migrating herons "
    "never negotiate fares with lighthouse keepers during stormy marmalade seasons".split())


def inject(stem: str, page: int, variant: str, edit):
    data = load(stem, page, variant)
    assert edit(data), "the injection found nothing to change"
    return data


def test_a_sentence_from_elsewhere_is_fabrication(katex):
    def edit(data):
        block = longest(data)
        cut = block["block_content"].find(". ") + 2
        block["block_content"] = block["block_content"][:cut] + FOREIGN + ". " + block["block_content"][cut:]
        return True

    v = verdict(katex, "plain", 1, inject("plain", 1, "", edit))
    assert v.status == "fail" and "fabrication" in codes(v)


def test_a_changed_digit_is_a_number_failure(katex):
    ref = reference("plain", 2)
    known = set(re.findall(r"\d+", " ".join(w.text for w in ref.words)))

    def edit(data):
        for block in blocks(data):
            for match in re.finditer(r"\b\d{3}\b", block["block_content"]):
                new = str(int(match.group()) + 1)
                if new not in known:
                    block["block_content"] = block["block_content"][:match.start()] + new + block["block_content"][match.end():]
                    return True
        return False

    v = verdict(katex, "plain", 2, inject("plain", 2, "", edit))
    assert v.status == "fail" and "number" in codes(v)


def test_a_paragraph_copied_five_times_is_repetition(katex):
    def edit(data):
        block = longest(data)
        block["block_content"] = (block["block_content"] + "\n\n") * 5
        return True

    v = verdict(katex, "plain", 0, inject("plain", 0, "", edit))
    assert v.status == "fail" and "repetition" in codes(v)


def test_a_dropped_paragraph_is_omission(katex):
    def edit(data):
        longest(data)["block_content"] = ""
        return True

    v = verdict(katex, "plain", 2, inject("plain", 2, "", edit))
    assert v.status == "fail" and "omission" in codes(v)


def test_a_formula_missing_a_brace_is_a_recorded_render_failure(katex):
    def edit(data):
        for block in blocks(data):
            match = re.search(r"\$[^$]*\\mathrm\{[^$}]*\}[^$]*\$", block["block_content"])
            if match:
                span = match.group()
                block["block_content"] = block["block_content"].replace(span, span.replace("}", "", 1), 1)
                return True
        return False

    v = verdict(katex, "formula", 2, inject("formula", 2, ".retry", edit))
    assert v.status == "pass" and any(s.startswith("formula_render") for s in v.soft)


def test_alpha_swapped_for_beta_is_a_recorded_symbol_difference_when_the_page_has_no_beta(katex):
    ref = reference("formula", 1)
    assert not any("β" in w.text for w in ref.words)

    def edit(data):
        changed = False
        for block in blocks(data):
            if "\\delta" in block["block_content"]:
                block["block_content"] = block["block_content"].replace("\\delta", "\\beta")
                changed = True
        return changed

    v = verdict(katex, "formula", 1, inject("formula", 1, ".retry", edit))
    # one symbol that differs: on record, not a reason to refuse the paper (and a beta is not a look-alike of a b)
    assert v.status == "pass" and "'beta'" in " ".join(v.soft), (v.reasons, v.soft)


def test_a_changed_table_cell_is_refused(katex):
    def edit(data):
        for block in data["parsing_res_list"]:
            if block["block_label"] == "table":
                match = re.search(r"<td>(\d+)\.(\d)", block["block_content"])
                if match:
                    new = f"<td>{match.group(1)}.{(int(match.group(2)) + 3) % 10}"
                    block["block_content"] = block["block_content"].replace(match.group(), new, 1)
                    return True
        return False

    v = verdict(katex, "table", 1, inject("table", 1, "", edit))
    assert v.status == "fail" and codes(v) & {"table", "number"}


# ------------------------------------------------------------- the pieces
def test_group_members_share_one_segment_with_the_union_box():
    data = load("formula", 0)
    page = mdtext.parse_page(data, 1.0)
    members = [b for b in data["parsing_res_list"] if b["group_id"] == 4]
    assert len(members) == 2 and sum(1 for b in members if not b["block_content"]) == 1
    union = (min(b["block_bbox"][0] for b in members), min(b["block_bbox"][1] for b in members),
             max(b["block_bbox"][2] for b in members), max(b["block_bbox"][3] for b in members))
    assert any(segment.bbox == union for segment in page.prose)


def test_formulas_are_found_with_and_without_padding_and_in_all_four_delimiters():
    box = (0.0, 0.0, 1.0, 1.0)
    parts = mdtext.split_formulas(r"a $x_1$ b  $ y $ c $$z$$ d \(u\) e \[v\]", box)
    formulas = [(p.tex, p.display) for p in parts if not isinstance(p, str)]
    assert formulas == [("x_1", False), ("y", False), ("z", True), ("u", False), ("v", True)]


def _text_page(content: str):
    return mdtext.parse_page({"width": 1, "height": 1, "parsing_res_list": [
        {"block_id": 0, "block_label": "text", "block_bbox": [0, 0, 1, 1], "block_content": content}]}, 1.0)


def test_a_dollar_that_never_closes_is_counted():
    page = _text_page("so $x \\le y and more")
    assert page.stray_dollars == 1 and not page.formulas


def test_prices_and_units_are_text_not_formula_delimiters():
    page = _text_page("more than $1,000 in the account, at $2.50 each, per budget in ($/month), "
                      "and $2^{19}$ models; the real $x$ is $5")
    assert page.stray_dollars == 0
    assert [f.tex for f in page.formulas] == ["2^{19}", "x"]


def test_ignored_labels_become_skipped_regions_and_tables_keep_their_cells():
    data = load("table", 0)
    page = mdtext.parse_page(data, 1.0)
    labels = {b["block_label"] for b in data["parsing_res_list"]}
    assert "number" in labels and len(page.skip_boxes) >= 1
    assert len(page.tables) == 2 and "Mahinda" in "".join(p for cell in page.tables[0].cells for p in cell if isinstance(p, str))


def test_page_markdown_files_hold_what_the_verified_blocks_hold():
    for stem, page in (("plain", 1), ("table", 1), ("formula", 1)):
        data, md = load(stem, page, ".retry" if stem == "formula" else ""), (
            FIXTURES / f"{stem}.page{page}{'.retry' if stem == 'formula' else ''}.md").read_text(encoding="utf-8")
        flat = checks.letters(md)
        for block in blocks(data):
            assert checks.letters(mdtext.prose_text(block["block_content"]))[:200] in flat


def test_hyphenated_line_ends_are_joined_in_the_reference():
    import pymupdf

    with pymupdf.open(FIXTURES / "plain.pdf") as document:
        raw = document[1].get_text("words")
    hyphenated = sum(1 for w in raw if len(w[4]) > 1 and w[4].endswith("-"))
    ref = textlayer.page_reference(FIXTURES / "plain.pdf", 1)
    kept = sum(1 for w in ref.words if w.text.endswith("-"))
    assert hyphenated > kept >= 0 and ref.usable


def test_a_text_layer_full_of_unmappable_glyphs_is_not_a_reference():
    words = [(0, i, 1, i + 1, "(cid:%d)abc" % i, 0, i, 0) for i in range(60)]
    ref = textlayer.reference_from_words(0, 100, 100, words)
    assert not ref.usable and "unmappable" in ref.reason


def test_normalize_text_is_olmocrs():
    assert normalize.normalize_text("**Bold** and __also__ <br/>it’s – “x” µ") == 'Bold and also it\'s - "x" μ'
    assert normalize.comparable("Ｆｏｏ  𝑘-ECCs\n") == "foo𝑘-eccs".replace("𝑘", "k")


# ---------------------------------------------------------------- katex
def test_katex_reports_errors_and_text_and_survives_a_dead_process():
    with KatexRenderer() as renderer:
        good = renderer.render(r"\alpha_i + \sum_{k=1}^{n} x_k^2 \leq \mathcal{O}(\delta(G))")
        assert good.ok and "α" in good.chars and "∑" in good.chars and "δ" in good.chars
        bad = renderer.render(r"\frac{a}{b")
        assert not bad.ok and "KaTeX parse error" in bad.error
        renderer._process.kill()
        assert renderer.render("x^2").ok


def test_katex_without_node_modules_fails_loudly(tmp_path):
    with pytest.raises(KatexUnavailable, match="npm ci"):
        KatexRenderer(check_dir=tmp_path).render("x")


# --------------------------------------------------------- report and gate
def test_the_report_is_what_the_extract_gate_reads(katex, tmp_path):
    from app.services.pipeline import stages
    from app.services.pipeline.context import StageContext

    report = PaperReport("paper-a", "/pdf/paper-a.pdf")
    report.record(verdict(katex, "formula", 2, load("formula", 2)))  # v_10 rendered as v_1 then 0: a wrong number
    report.record(verdict(katex, "formula", 2, load("formula", 2)), {"repetition_penalty": 1.1})
    report.record(verdict(katex, "plain", 1, load("plain", 1)))
    data = report.to_json()
    assert data["status"] == "failed" and data["page_counts"] == {"pass": 1, "fail": 1, "unverifiable": 0}
    assert data["pages"][1]["retried"] and data["pages"][1]["attempts"][1]["sampling"] == {"repetition_penalty": 1.1}
    assert data["failure_reasons"][0].startswith("page 3: number")

    ctx = StageContext(venue="CVPR", year=2026, store=None, task_id=0, corpus_root=tmp_path)
    report.write(ctx.local_md_dir / "_quality" / "paper-a.json")
    assert not list((ctx.local_md_dir / "_quality").glob("*.part"))
    summary = stages._quality_failure_summary(ctx)
    assert "1 paper(s) failed verification: paper-a: page 3: number" in summary
    (ctx.local_md_dir / "paper-a.md").write_text("# done later", encoding="utf-8")
    assert stages._quality_failure_summary(ctx) == ""  # a later run that converted it clears the failure


def test_soft_issues_are_recorded_per_page_and_the_paper_still_stands(katex, tmp_path):
    from app.services.pipeline import stages
    from app.services.pipeline.context import StageContext

    report = PaperReport("paper-b", "/pdf/paper-b.pdf")
    report.record(verdict(katex, "formula", 0, load("formula", 0)))  # a tau for a T: a look-alike, nothing to record
    report.record(verdict(katex, "formula", 1, load("formula", 1)))  # a LaTeX array KaTeX cannot parse
    data = report.to_json()
    assert data["status"] == "pass_with_issues" and data["soft_issues"] == 1 and data["failure_reasons"] == []
    assert data["pages"][0]["soft"] == [] and data["pages"][1]["soft"][0].startswith("formula_render")
    assert data["page_counts"]["fail"] == 0

    ctx = StageContext(venue="CVPR", year=2026, store=None, task_id=0, corpus_root=tmp_path)
    report.write(ctx.local_md_dir / "_quality" / "paper-b.json")
    assert stages._quality_failure_summary(ctx) == ""  # nothing refused: the gate counts the Markdown as extracted


def test_a_paper_with_no_pages_is_failed():
    report = PaperReport("p", "p.pdf")
    assert report.status == "failed" and report.failure_reasons() == ["no page was converted"]


# ------------------------------------------------------------- the extractor
def test_the_extractor_is_registered_and_refuses_unreadable_pdfs_without_a_server(tmp_path):
    from pipeline.paper_crawler.converters.extractors import create_extractor
    from pipeline.paper_crawler.converters.extractors.base import ExtractionStatus

    extractor = create_extractor("paddleocr_vl", server_url="http://127.0.0.1:9/v1")
    assert extractor.strategy_name == "paddleocr_vl"
    garbage = tmp_path / "broken.pdf"
    garbage.write_bytes(b"%PDF-1.4 this is not a pdf")
    results = extractor.extract_many([tmp_path / "missing.pdf", garbage])
    assert [r.status for r in results] == [ExtractionStatus.FAILED] * 2
    assert "not found" in results[0].error and "unreadable PDF" in results[1].error
    assert not any(r.markdown for r in results)


# ------------------------------------------------- calibration on 30 real VLDB papers
# tests/fixtures/pvl/calibration_cases.json holds, per case, the text layer words of one PDF
# page and what PaddleOCR-VL-1.6 returned for it. The first group are pages the model got
# right and an earlier version of the checks refused for a reason in the PDF's text layer
# (a Greek letter spelled out, a footnote mark glued to a figure, a ligature dropped...); the
# second group are pages the model got wrong, which must stay refused.
CASES = json.loads((FIXTURES / "calibration_cases.json").read_text(encoding="utf-8"))
RIGHT = [
    "greek_spelled_by_the_layer", "footnote_mark_glued_to_a_version", "doi_broken_after_a_dot",
    "hyphen_inside_a_cpu_name", "script_glued_to_its_base_digits", "accents_and_symbol_font_letters",
    "sum_and_join_as_stray_letters", "prices_and_units_in_prose", "header_wrapped_over_two_lines",
    "ligature_dropped_by_the_layer", "leading_zero_not_printed", "math_alphanumerics_without_the_top_bit",
    "listing_drawn_as_outlines", "sentence_repeated_in_a_proof",
    # round 2 (final30 first-pass pages): LaTeX lines with no dollar signs, prose cut into pieces by inline
    # formulas, a formula that opens with a number, a price with a unit
    "pseudocode_latex_without_delimiters", "prose_between_inline_formulas", "formula_opening_with_a_number",
    "formula_opening_with_a_number_and_no_latex", "price_with_a_unit",
    # round 3 (MinerU's page): underbraces the layer spells "{z" and "|{z}", a sum it spells "õ"
    "underbrace_glyphs_as_stray_letters",
]
WRONG = {  # the model's real errors: a figure that is not the page's, text that is not there, a loop
    "digit_dropped_from_1110": ({"number"}, "'110'"),
    "invented_throughput": ({"number"}, "'120'"),
    "decoding_loop": ({"length", "repetition"}, "appears 224 times"),
}
# what a reader cannot tell apart: the PDF prints a v, the model wrote a nu; the layer holds an empty set where the
# model wrote a zero. Accepted, and the figure written another way is the only thing on record.
HOMOGLYPHS = {
    "nu_where_the_pdf_has_a_v": None,
    "tau_where_the_pdf_has_a_T": None,
    "empty_set_read_as_zero": "number: written differently from the page: ['0']",
}
# small differences, recorded (one symbol of a formula, a formula KaTeX cannot parse, a cell a little short)
SOFT = {
    "u_for_v": ("formula_symbol", "['u']"),
    "z_0_e_0_not_on_the_page": ("formula_symbol", "['z']"),
    "latex_katex_cannot_parse": ("formula_render", "Double subscript"),
    "delta_for_epsilon_in_a_table": ("table", "words in the table region"),
}


def case_verdict(katex, name: str, witness=None) -> checks.PageVerdict:
    case = CASES[name]
    number = case["page"] - 1
    ref = textlayer.reference_from_words(number, case["width"], case["height"], case["words"])
    data = case["result"]
    return checks.check_page(mdtext.parse_page(data, ref.width / data["width"], number), ref, katex, witness)


def recorded_reader(name: str):
    """The second reader of a case's display formulas: what PP-FormulaNet_plus-L read in each box
    when the case was recorded (on the PDF's own page image, 200 dpi)."""
    readings = CASES[name]["readings"]

    def read(box):
        for item in readings:
            if all(abs(a - b) <= 1.0 for a, b in zip(item["box"], box)):
                return item["latex"]
        raise AssertionError(f"{name}: no reading recorded for the box {box}")
    return read


@pytest.mark.parametrize("name", RIGHT)
def test_a_page_the_model_got_right_is_not_refused_for_the_layers_quirks(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass", v.reasons


@pytest.mark.parametrize("name", sorted(WRONG))
def test_a_page_the_model_got_wrong_stays_refused(katex, name):
    expected_codes, excerpt = WRONG[name]
    v = case_verdict(katex, name)
    assert v.status == "fail" and codes(v) == expected_codes, v.reasons
    assert excerpt in " ".join(v.reasons)


@pytest.mark.parametrize("name", sorted(HOMOGLYPHS))
def test_a_symbol_that_looks_like_the_layers_is_accepted(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons, v.reasons
    assert not [s for s in v.soft if s.startswith("formula_symbol")], v.soft
    if HOMOGLYPHS[name]:
        assert HOMOGLYPHS[name] in v.soft


@pytest.mark.parametrize("name", sorted(SOFT))
def test_a_small_difference_is_recorded_and_the_page_stands(katex, name):
    kind, excerpt = SOFT[name]
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons, v.reasons
    assert any(s.startswith(kind) and excerpt in s for s in v.soft), v.soft


# round 5 (task 1302, VLDB 2022, 2026-11-13): pages of papers the worker refused, read against their PDFs. The text
# layer and the Markdown are the real ones (blocks kept in the _quality reports); each case is named for what the
# reader sees. The first group is the checker's mistake or a look-alike and says nothing; the second is a small
# difference that stays on record; the third are real errors that must still refuse.
SILENT = [
    "triangle_for_delta",                  # the layer's font holds Delta as the white triangle U+25B3
    "open_e_for_epsilon",                  # ... epsilon as the open e U+025B
    "figure_pair_set_apart_by_a_formula",  # \(Y_{[901,1400]}\): KaTeX sets the comma apart, the layer has "901,1400"
    "decimal_power_glued_to_its_base",     # 10^{4.5} is "104.5" in the layer
    "power_of_ten_in_table_cells",         # 10^{13} inside a table cell is "1013" there
    "thousands_comma_set_apart",           # 1,000 in a formula
    "doi_wrapped_in_its_digits",           # 10.14778/27329 77.2732986 over two lines
    "date_with_a_comma_before_the_year",   # "May 02, 2022" is the page's "02,2022"
    "stray_glyph_in_a_table_region",       # one word "hhhh" in the region: no recall to speak of
]
RECORDED = {
    "letter_o_written_as_zero": "number: written differently from the page: ['0']",       # (O-AL) as (0-AL)
    "dash_o2_written_dash_02": "number: written differently from the page: ['02']",       # -O2 as -02
    "empty_set_in_pseudocode_read_as_zero": "number: written differently from the page: ['0']",
    "j_where_the_page_has_i": "formula_symbol",                                           # W_j' for W_i': one symbol
    "lone_misread_header_cell": "table: cell 'enlargedrddit'",                           # "rddit" for "reddit"
    "multiplication_sign_for_x_in_a_table": None,                                        # "2x" for "2×" is no issue at all
}
STILL_REFUSED = {
    "digit_where_the_page_has_a_letter": ({"number"}, "['0']"),    # mu^0 for the page's mu^q
    "sql_text_the_page_does_not_have": ({"fabrication", "table", "formula_symbol"}, "reatingsql"),
    "rn_read_as_m_in_sql": ({"repetition"}, "'post date as post year count from denom'"),  # denorm_so for denom_so
}


@pytest.mark.parametrize("name", SILENT)
def test_a_page_the_checker_wrongly_refused_passes_without_a_trace(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons and not v.soft, (v.reasons, v.soft)


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_a_small_difference_on_a_real_page_is_recorded_not_refused(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons, v.reasons
    if RECORDED[name]:
        assert any(s.startswith(RECORDED[name]) for s in v.soft), v.soft
    else:
        assert not [s for s in v.soft if s.startswith("table")], v.soft


@pytest.mark.parametrize("name", sorted(STILL_REFUSED))
def test_a_real_error_on_a_real_page_is_still_refused(katex, name):
    expected, excerpt = STILL_REFUSED[name]
    v = case_verdict(katex, name)
    assert v.status == "fail" and codes(v) == expected and excerpt in " ".join(v.reasons), v.reasons


def test_look_alikes_come_from_unicodes_table_and_the_pairs_of_mathematics_it_lacks():
    from pipeline.paper_crawler.converters.verify.homoglyphs import lookalikes

    assert "v" in lookalikes("nu") and "u" in lookalikes("upsilon") and "o" in lookalikes("omicron")
    assert "△" in lookalikes("delta") and "ɛ" in lookalikes("epsilon") and "t" in lookalikes("tau")
    assert {"w"} <= lookalikes("omega") and {"k"} <= lookalikes("kappa") and {"x"} <= lookalikes("chi")
    assert {"1", "|"} <= lookalikes("l") and "nu" in lookalikes("v")  # and from the other side
    assert "b" not in lookalikes("beta") and "p" not in lookalikes("q")  # a beta is not a b


def test_a_figure_written_another_way_is_the_same_figure_a_different_one_is_not():
    assert checks._variant_key("1,000") == checks._variant_key("1000") == "1000"
    assert checks._variant_key("02") == checks._variant_key("2") == "2"
    assert checks._variant_key("110") != checks._variant_key("1110")
    assert checks._variant_key("0.5") == "0.5" and checks._variant_key("0") == "0"


def test_a_block_with_no_text_in_the_layer_is_set_aside_not_called_fabricated(katex):
    case = CASES["greek_spelled_by_the_layer"]
    number = case["page"] - 1
    data = case["result"]
    scale = case["width"] / data["width"]
    big = max(blocks(data), key=lambda b: len(b["block_content"]))
    x0, y0, x1, y1 = (v * scale for v in big["block_bbox"])
    # the layer of the same page without the words that sit in that one block
    words = [w for w in case["words"] if not (x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1)]
    assert len(words) < len(case["words"])
    ref = textlayer.reference_from_words(number, case["width"], case["height"], words)
    v = checks.check_page(mdtext.parse_page(data, ref.width / data["width"], number), ref, katex)
    assert v.metrics["unwitnessed_blocks"] == 1 and v.status == "pass", v.reasons


# ------------------------------------------- the second reader of display formulas (PP-FormulaNet)
# Two pages of one paper whose maths font has no usable ToUnicode map: the layer spells a Greek
# letter as "u", and a line of plain text in a figure (a "Hzt" the layout model took for a formula).
WITNESSED = ["text_in_a_figure_read_as_a_formula", "maths_font_without_unicode_map"]


@pytest.mark.parametrize("name", WITNESSED)
def test_a_formula_the_layer_cannot_spell_is_kept_when_a_second_reader_agrees(katex, name):
    refused = case_verdict(katex, name)
    assert refused.status == "fail" and codes(refused) == {"formula_symbol"}, refused.reasons
    v = case_verdict(katex, name, witness=recorded_reader(name))
    assert v.status == "pass" and v.metrics["formulas_witnessed"] >= 1, v.reasons


def test_a_second_reader_that_disagrees_does_not_rescue_a_page(katex):
    # the second reader sees something else in the box: the layer's refusal stands
    v = case_verdict(katex, "text_in_a_figure_read_as_a_formula", witness=lambda box: r"z^{2}")
    assert v.status == "fail" and codes(v) == {"formula_symbol"} and v.metrics["formulas_witnessed"] == 0, v.reasons


def test_a_formula_with_one_symbol_off_is_not_sent_to_the_second_reader(katex):
    # the model wrote "u" where the page has a "v": on record, and not worth a read of the box
    def never(box):
        raise AssertionError("the second reader was asked about a soft difference")

    v = case_verdict(katex, "u_for_v", witness=never)
    assert v.status == "pass" and v.metrics["formulas_witnessed"] == 0 and v.soft


# ------------------------------------------------- MinerU2.5-Pro, the second model (real answers of its server)
# tests/fixtures/pvl/<name>.page<N>.mineru.json is what a MinerU2.5-Pro-2605-1.2B vLLM server (mineru-vl-utils
# two_step_extract on the page rendered at 200 dpi) answered for a page: its content blocks, boxes as fractions
# of the page. ``rescued.pdf`` is a page PaddleOCR-VL got wrong (z_0, e_0 for what the page prints as other
# letters) that MinerU reads right; ``unrescued.pdf`` is one both get wrong (a table cell each).
from pipeline.paper_crawler.converters.extractors import mineru_pages  # noqa: E402
from pipeline.paper_crawler.converters.extractors.base import ExtractionStatus  # noqa: E402
from pipeline.paper_crawler.converters.extractors.paddleocr_vl import Batch, PaddleOCRVLExtractor  # noqa: E402


class RecordedServer:
    """The client of the MinerU server (``concurrent_two_step_extract``), answering with what the server
    answered when the fixture was recorded: everything downstream of the network call runs for real."""

    def __init__(self, stem: str, numbers: list[int], spoil: bool = False):
        self.recorded = [json.loads((FIXTURES / f"{stem}.page{n}.mineru.json").read_text(encoding="utf-8")) for n in numbers]
        self.spoil = spoil  # the second model wrote a sentence that is not on the page

    def concurrent_two_step_extract(self, images):
        assert [(i.width, i.height) for i in images] == [(r["width"], r["height"]) for r in self.recorded]
        answers = [r["blocks"] for r in self.recorded]
        if self.spoil:
            for blocks in answers:
                longest_text = max((b for b in blocks if b["type"] == "text"), key=lambda b: len(b.get("content") or ""))
                longest_text["content"] = longest_text["content"] + " " + FOREIGN
        return answers


def mineru_page(stem: str, number: int) -> tuple[dict, str]:
    raw = json.loads((FIXTURES / f"{stem}.page{number}.mineru.json").read_text(encoding="utf-8"))
    return mineru_pages.blocks_to_page(raw["blocks"], raw["width"], raw["height"], number)


@pytest.mark.parametrize("stem,number", [
    ("plain", 0), ("plain", 1), ("plain", 2), ("table", 0), ("table", 2),
    # the three formula pages PaddleOCR-VL writes wrongly (a tau for a T, a broken array, v_10)
    ("formula", 0), ("formula", 1), ("formula", 2), ("rescued", 0),
])
def test_mineru_blocks_land_on_the_text_layer_and_pass_the_same_checks(katex, stem, number):
    data, markdown = mineru_page(stem, number)
    ref = textlayer.page_reference(FIXTURES / f"{stem}.pdf", number)
    v = checks.check_page(mdtext.parse_page(data, ref.width / data["width"], number), ref, katex)
    assert v.status == "pass" and markdown.strip(), v.reasons
    assert v.metrics["unwitnessed_blocks"] == 0  # every block found its words: the boxes are in the right place


def test_mineru_boxes_are_pixels_of_the_rendered_page():
    data, _ = mineru_page("table", 0)
    ref = textlayer.page_reference(FIXTURES / "table.pdf", 0)
    scale = ref.width / data["width"]
    assert 0 < scale < 1 and data["width"] == round(ref.width * 200 / 72)
    table = next(b for b in data["parsing_res_list"] if b["block_label"] == "table")
    x0, y0, x1, y1 = (v * scale for v in table["block_bbox"])
    inside = [w for w in ref.words if x0 - 2 <= (w.bbox[0] + w.bbox[2]) / 2 <= x1 + 2 and y0 - 2 <= (w.bbox[1] + w.bbox[3]) / 2 <= y1 + 2]
    assert len(inside) >= 8  # the table's box encloses the table's words, in PDF points


def test_mineru_table_cells_glued_together_are_refused(katex):
    # "34.2" over "6.8x" in one cell came back as "34.26.8x": the page has no such number
    data, _ = mineru_page("table", 1)
    ref = textlayer.page_reference(FIXTURES / "table.pdf", 1)
    v = checks.check_page(mdtext.parse_page(data, ref.width / data["width"], 1), ref, katex)
    assert v.status == "fail" and {"number", "table"} <= codes(v), v.reasons


def test_escaped_dollar_signs_are_text_not_formula_delimiters(katex):
    # MinerU writes a price as \$0.09 and a SQL quote as \$\$: neither is a formula
    parts = mdtext.split_formulas(r"rates from \$0.09 to \$4.36 and as \$\$ body \$\$ end", (0, 0, 1, 1))
    assert all(isinstance(p, str) for p in parts)
    assert "".join(parts).count("$") == 0


def test_a_script_on_the_last_digit_of_a_name_is_the_page_number_the_layer_glued(katex):
    # P4_16 is "P416" in the text layer: the Markdown's "4" and "16" are on the page (a VLDB 2022 sentence)
    layer = ("It is not straightforward to implement the NetLR design because switch hardware has limited resources "
             "and strict timing constraints Therefore we use several techniques that translate the design to a working "
             "system Our switch data plane is written in P416 [10] and is compiled with Intel P4 Studio SDE 9.2.0").split()
    words = [(10.0 + (i % 10) * 55, 10.0 + (i // 10) * 12, 60.0 + (i % 10) * 55, 20.0 + (i // 10) * 12, w, 0, i // 10, i % 10)
             for i, w in enumerate(layer)]
    ref = textlayer.reference_from_words(0, 600.0, 200.0, words)
    markdown = " ".join(layer).replace("P416", "P4 $ _{16} $")
    data = {"page_index": 0, "width": 600, "height": 200, "model_settings": {"markdown_ignore_labels": []},
            "parsing_res_list": [{"block_id": 0, "group_id": 0, "block_label": "text", "block_bbox": [5, 5, 590, 120],
                                  "block_content": markdown}]}
    v = checks.check_page(mdtext.parse_page(data, 1.0, 0), ref, katex)
    assert v.status == "pass", v.reasons


def second_model_batch(extractor, stem: str, number: int = 0):
    """The state ``convert`` leaves for a PDF page PaddleOCR-VL's output failed twice (``number`` of ``stem.pdf``;
    the earlier pages are not part of the batch)."""
    pdf = FIXTURES / f"{stem}.pdf"
    path = FIXTURES / f"{stem}.page{number}.pvl.json"
    data = json.loads((path if path.exists() else FIXTURES / f"{stem}.page{number}.json").read_text(encoding="utf-8"))
    batch = Batch(pdfs=[pdf], t0=0.0)
    batch.runnable, batch.todo, batch.expected = [pdf], [pdf], {pdf: number + 1}
    batch.reports[pdf] = PaperReport(pdf.stem, str(pdf))
    batch.pages[pdf], batch.markdown_pages[pdf] = {number: data}, {number: ""}
    verdict = extractor._verdict(pdf, number, data)
    batch.reports[pdf].record(verdict)
    batch.reports[pdf].record(verdict, {"repetition_penalty": 1.05})  # the retry
    return pdf, batch


# "formula" page 2 is v_10 rendered as v_1 then 0 (a wrong number: hard); MinerU reads it right. In the second case the
# second model writes a sentence that is not on the page as well: it fails the same checks and the paper is refused.
@pytest.mark.parametrize("stem,number,spoil,paper_passes", [("formula", 2, False, True), ("formula", 2, True, False)])
def test_pages_paddleocr_vl_fails_go_to_mineru_and_a_paper_stands_only_if_every_page_does(
        katex, stem, number, spoil, paper_passes):
    extractor = PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex)
    pdf, batch = second_model_batch(extractor, stem, number)
    report = batch.reports[pdf]
    assert report.status == "failed" and batch.unresolved == [(pdf, number)]

    reader = mineru_pages.MineruReader("http://127.0.0.1:1/v1")
    reader._client = RecordedServer(stem, [number], spoil=spoil)
    extractor.redo_failed_pages([batch], reader)

    assert [a["engine"] for a in report.attempts[number]] == ["paddleocr_vl", "paddleocr_vl", "mineru"]
    assert (report.status != "failed") is paper_passes, report.failure_reasons()
    if paper_passes:
        assert batch.pages[pdf][number]["parsing_res_list"][0]["block_label"] and batch.markdown_pages[pdf][number].strip()
        assert report.to_json()["pages"][0]["engine"] == "mineru"
    else:
        result = extractor.assemble(batch)[pdf]
        assert result.status == ExtractionStatus.FAILED and not result.markdown
        assert f"page {number + 1}" in result.error
        assert [a["engine"] for a in result.extra["report"]["pages"][0]["attempts"]][-1] == "mineru"


# A table of notations whose symbols (an L, a beta) are in a maths font the layer cannot spell, and a word with a
# ligature ("fit"): once a display formula elsewhere in the PDF proved the font unreadable (math_layer_broken), the
# page's cells are not held to those symbols. PaddleOCR-VL's page and MinerU's (the same page, a different cell wording).
@pytest.mark.parametrize("name", ["table_symbols_in_a_maths_font", "table_symbols_in_a_maths_font_mineru"])
def test_table_symbols_in_a_font_the_layer_cannot_spell_pass_once_the_pdf_is_known_to_have_one(katex, name):
    case = CASES[name]
    number = case["page"] - 1
    ref = textlayer.reference_from_words(number, case["width"], case["height"], case["words"])
    data = case["result"]
    page = mdtext.parse_page(data, ref.width / data["width"], number)
    alone = checks.check_page(page, ref, katex)
    # a lone cell the layer cannot spell (a beta) is a recorded difference now, not a refusal; once the PDF is known
    # to have the font the cell is not held to the layer at all
    assert alone.status == "pass", alone.reasons
    known = checks.check_page(page, ref, katex, math_layer_broken=True)
    assert known.status == "pass" and len(known.soft) <= len(alone.soft), known.reasons


def test_a_merge_that_raises_fails_its_paper_and_not_the_batch(katex):
    extractor = PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex, pipeline=object())  # no restructure_pages
    pdf, batch = second_model_batch(extractor, "rescued")
    assert batch.reports[pdf].status != "failed"
    result = extractor.assemble(batch)[pdf]
    assert result.status == ExtractionStatus.FAILED and "merging the pages failed: AttributeError" in result.error


def test_a_first_page_the_second_model_wrote_still_merges_into_one_document(katex):
    # task 1302 (vldb-2022-p1350, p1559, p1911): page 1 was MinerU's, and the merge died with KeyError: 'input_path'.
    # restructure_pages concatenates into the first page's dict and the result reads its input_path (paddlex
    # result.py: ``result["input_path"] = self["input_path"]``); this stand-in does what that code does with it.
    pdf = FIXTURES / "rescued.pdf"
    data, markdown = mineru_page("rescued", 0)
    assert "input_path" not in data  # what blocks_to_page returns

    class Merged(dict):  # the merged result is dict-like: its blocks and the settings of the pages
        def _to_markdown(self, pretty=False):
            return {"markdown_texts": markdown}

    class Pipeline:
        def restructure_pages(self, res_list, **options):
            res_list[0]["res"]["input_path"]  # KeyError on the second model's page
            blocks = [SimpleNamespace(label=b["block_label"], content=b["block_content"]) for b in data["parsing_res_list"]]
            return [Merged(parsing_res_list=blocks, model_settings=data["model_settings"])]

    extractor = PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex, pipeline=Pipeline())
    batch = Batch(pdfs=[pdf], t0=0.0)
    batch.runnable, batch.todo, batch.expected = [pdf], [pdf], {pdf: 1}
    batch.reports[pdf] = PaperReport(pdf.stem, str(pdf))
    batch.pages[pdf], batch.markdown_pages[pdf] = {0: data}, {0: markdown}
    batch.reports[pdf].record(extractor._verdict(pdf, 0, data), {}, None, engine="mineru")
    result = extractor.assemble(batch)[pdf]
    assert result.status == ExtractionStatus.SUCCESS, result.error


def test_mineru_pages_carry_what_the_merge_of_both_models_pages_reads():
    data, _ = mineru_page("plain", 0)
    # restructure_pages takes the flags of the last page for all pages and reads layout_det_res of each
    assert data["model_settings"]["use_layout_detection"] is True and data["model_settings"]["use_doc_preprocessor"] is False
    assert len(data["layout_det_res"]["boxes"]) == len(data["parsing_res_list"])
    pvl = json.loads((FIXTURES / "rescued.page0.pvl.json").read_text(encoding="utf-8"))
    assert {"layout_det_res"} <= set(pvl) and set(pvl["model_settings"]) <= set(data["model_settings"])


# round 6 (task 1302's retry: 38 papers refused, 259 written): the pages of the refused papers, read against their PDFs.
# Same kinds as round 5: the checker's mistake says nothing, a small difference stays on record, a real error refuses.
LAYER_QUIRKS = [
    "code_set_one_glyph_at_a_time",                 # "u i n t 3 2 _ t": the layer has one glyph per word
    "product_operators_dropped_by_the_layer",       # 24*60*60*1000 is "246060*1000"
    "subscript_glued_to_the_figure_after_it",       # log_2 64 is "log264"
    "footnote_mark_after_the_full_stop",            # "December 2020." + footnote 17 is "2020.17"
    "table_cells_run_together_by_the_layer",        # "7391±1267" "10.8±4.5" is "7391±126710.8±4.5"
    "short_doi_wrapped_inside_its_digits",          # 10.1137/02 04032
    "power_in_plain_text_glued_to_its_base",        # 2^{30*(1-c)} in plain text is "230*(1-c)"
    "ligature_dropped_in_repeated_sentences",       # "\uffffnd persons": the page repeats the sentence as well
    "dash_spelled_as_s_acute",                      # "2962ś2970" in ACM Reference Format
    "citation_list_as_display_formula",             # \[12, 31, 34\]: figures only, nothing to check against symbols
    "times_spelled_as_arrow_to_bar_in_a_table",     # "38⇥" for 38x
]
SMALL_DIFFERENCES = {
    "thousands_separated_by_a_space": "number: written differently from the page: ['1192', '3462']",
    "thousands_separated_by_a_space_in_a_table": "number: written differently from the page",
    "thousands_separated_after_a_tilde": "number: written differently from the page: ['2219520']",
    "identifier_underscores_dropped_by_the_layer": "table: cell 'ocsmmkme'",   # OCSMM over KME in a header
    "digit_for_letter_in_table_words": "table: cell 'spectac1'",               # spectac1 for spectacl
    "letter_dropped_from_a_repeated_header": "table: cell 'l2htrate'",         # l2 htrate for l2 hitrate
}


@pytest.mark.parametrize("name", LAYER_QUIRKS)
def test_a_layer_quirk_of_a_refused_vldb_page_passes_without_a_trace(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons and not v.soft, (v.reasons, v.soft)


@pytest.mark.parametrize("name", sorted(SMALL_DIFFERENCES))
def test_a_small_difference_on_a_refused_vldb_page_is_recorded_not_refused(katex, name):
    v = case_verdict(katex, name)
    assert v.status == "pass" and not v.reasons, v.reasons
    assert any(s.startswith(SMALL_DIFFERENCES[name]) for s in v.soft), v.soft


def test_a_name_with_digits_is_a_whole_word_of_the_page_not_a_piece_of_one(katex):
    # "110ms" is not on a page that has "1110ms": the identifier rule must not turn a dropped digit into a name
    v = case_verdict(katex, "digit_dropped_from_1110")
    assert v.status == "fail" and "'110'" in " ".join(v.reasons)


def test_spaced_thousands_are_told_apart_in_a_row_of_figures():
    assert checks._spaced_figures("join 100 194 400 782 28 253.7") == [
        "100194", "100194400", "100194400782", "194400", "194400782", "400782", "28253.7"]
    assert checks._spaced_figures("about ∼2 219 520, which") == ["2219", "2219520", "219520"]
    assert checks._spaced_figures("p 1 134.83 x") == ["1134.83"]


def peer_of(name: str):
    case = CASES[name]
    return peer.peer_witness(case["peer"], case["width"])


# A maths font the layer cannot spell, on a page MinerU rewrote after PaddleOCR-VL failed it twice: the first model's
# page is the second reading of the display formulas. Two engines that agree on a formula's symbols are evidence
# against the layer, as one PP-FormulaNet reading is.
ENGINES_AGREE = ["maths_font_formula_the_other_engine_reads_alike", "maths_font_display_formulas_the_other_engine_reads_alike"]


# VLDB 2022 p2321 p4 (the display-formulas case) also has a table without its last column: the page's "NormA" values
# 4, 2.9, 4.7, 4.6, 4.7, 4.9 are not in the Markdown's cells. Nothing counted figures then; the table check does now, and the
# page stays refused for that, whoever reads its formulas.
TABLE_FIGURES_LOST = {"maths_font_display_formulas_the_other_engine_reads_alike"}


@pytest.mark.parametrize("name", ENGINES_AGREE)
def test_a_formula_two_engines_read_alike_is_kept_where_the_layer_cannot_spell_it(katex, name):
    lost = {"table"} if name in TABLE_FIGURES_LOST else set()
    refused = case_verdict(katex, name)
    assert refused.status == "fail" and codes(refused) == {"formula_symbol"} | lost, refused.reasons
    v = case_verdict(katex, name, witness=peer_of(name))
    assert v.status == ("fail" if lost else "pass") and codes(v) == lost and v.metrics["formulas_witnessed"] >= 1, v.reasons


def test_one_formula_the_engines_read_differently_still_refuses_the_page(katex):
    # tau_gamma = c_gamma / alpha: the first model wrote Y for the gamma
    name = "maths_font_formula_the_other_engine_reads_differently"
    v = case_verdict(katex, name, witness=peer_of(name))
    assert v.status == "fail" and codes(v) == {"formula_symbol"}, v.reasons


def test_a_peer_with_no_formula_on_the_box_reads_nothing():
    name = "maths_font_formula_the_other_engine_reads_alike"
    case = CASES[name]
    assert peer_of(name)((0.0, 0.0, 5.0, 5.0)) is None
    assert peer.peer_witness({**case["peer"], "parsing_res_list": []}, case["width"])((0, 0, 500, 700)) is None


class RedoneReader:
    """The second model's reader as the extractor sees it: what it redid is a real page, answered as the pipeline does."""
    name = "mineru"

    def __init__(self, number: int, data: dict):
        self._redone = {number: (data, "markdown")}

    def redo(self, pdf, numbers):
        return self._redone


def test_the_second_models_page_is_checked_against_the_first_models_formulas(katex, monkeypatch):
    name = ENGINES_AGREE[0]
    case = CASES[name]
    number = case["page"] - 1
    ref = textlayer.reference_from_words(number, case["width"], case["height"], case["words"])
    monkeypatch.setattr(paddleocr_vl, "page_reference", lambda pdf, n: ref)
    extractor = paddleocr_vl.PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex)
    pdf = Path(f"{case['paper']}.pdf")
    batch = paddleocr_vl.Batch(pdfs=[pdf], t0=0.0)
    batch.runnable, batch.todo, batch.expected = [pdf], [pdf], {pdf: case["page"]}
    batch.reports[pdf] = PaperReport(pdf.stem, str(pdf))
    batch.pages[pdf], batch.markdown_pages[pdf] = {number: case["peer"]}, {number: ""}  # the first model's page
    failing = extractor._verdict(pdf, number, case["result"])
    assert failing.status == "fail"  # alone, the layer refuses it
    batch.reports[pdf].record(failing)
    batch.reports[pdf].record(failing, {"repetition_penalty": 1.05})
    extractor.redo_failed_pages([batch], RedoneReader(number, case["result"]))
    assert batch.reports[pdf].status != "failed", batch.reports[pdf].failure_reasons()
    assert batch.pages[pdf][number] is case["result"]
