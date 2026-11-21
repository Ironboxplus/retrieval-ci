"""The verifier must not refuse a correct page of a mathematical paper for what the PDF's text layer does to mathematics.

Task 1308 (JMLR 2022) refused 96 of 353 papers; reading the failing pages against their PDFs showed that most of
the refusals were the reference's, not the model's (PLAN-2026-11-16-verify-textlayer-false-refusals.md). Every page
here is a real page of a JMLR 2022 PDF (a single page cut out with its fonts, tests/fixtures/pvl/jmlr2022/*.pdf)
with the blocks PaddleOCR-VL-1.6 really returned for it (the ``*.json`` next to it: the kept blocks of one attempt,
in the pixel width the report's metrics were reproduced at). One group per mechanism; the pages the model really got
wrong are at the end and must stay refused.

Needs node with KaTeX (``npm ci`` in tools/katex_check), like tests/test_pdf_markdown_verify.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.paper_crawler.converters.extractors import paddleocr_vl
from pipeline.paper_crawler.converters.verify import checks, mdtext, normalize, textlayer
from pipeline.paper_crawler.converters.verify.katex import KatexRenderer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pvl"
JMLR = FIXTURES / "jmlr2022"


@pytest.fixture(scope="module")
def katex():
    renderer = KatexRenderer()
    yield renderer
    renderer.close()


def jmlr(name: str):
    """(reference, Markdown blocks) of fixture page ``name``, e.g. ``18-131_p22`` (a VLDB page is named in full)."""
    stem = name if name.startswith("vldb-") else f"jmlr-2022-{name}"
    record = json.loads((JMLR / f"{stem}.json").read_text(encoding="utf-8"))
    return textlayer.page_reference(JMLR / f"{stem}.pdf", 0), record["result"]


def verdict(katex, name: str, ref=None) -> checks.PageVerdict:
    layer, data = jmlr(name)
    ref = ref or layer
    return checks.check_page(mdtext.parse_page(data, ref.width / data["width"], 0), ref, katex)


def clean(v: checks.PageVerdict) -> bool:
    return v.status == "pass" and not v.reasons


def codes(v: checks.PageVerdict) -> set[str]:
    return {reason.split(":")[0] for reason in v.reasons}


# ------------------------------------------- 1. glyphs of TeX's extension fonts are not text of the page
def test_cmex_wide_hats_and_radicals_are_not_words_of_the_page(katex):
    # p22 sets \widehat{B} and square roots with CMEX10 pieces: the layer spells them "b" and "r"
    v = verdict(katex, "18-131_p22")
    assert clean(v), v.reasons


def test_mathabx_delimiters_are_not_words_of_the_page(katex):
    # p11 sets its big parentheses with TeX-matha10 pieces: the layer spells them "p" "q" "t" "u"
    v = verdict(katex, "19-267_p11")
    assert clean(v), v.reasons


@pytest.mark.parametrize("font,skipped", [
    ("CMEX10", True), ("ABCDEF+CMEX7", True), ("Yhcmex", True), ("TeX-matha10", True), ("XYZABC+TeX-mathx8", True),
    ("mathb10", True), ("stmary10", True), ("txexs", True), ("esint10", True),
    # letters of mathematics are text: calligraphic, italic, blackboard bold, fraktur, script
    ("CMSY10", False), ("ABCDEF+CMSY8", False), ("CMMI10", False), ("MSBM10", False), ("MSAM10", False), ("EUFM10", False),
    ("rsfs10", False), ("dsrom10", False), ("BBOLD10", False), ("LMMathSymbols10-Regular", False), ("Times-Roman", False),
])
def test_only_fonts_that_draw_parts_of_formulas_are_left_out(font, skipped):
    assert textlayer.is_non_text_font(font) is skipped


def test_calligraphic_letters_of_cmsy_stay_in_the_reference():
    # p22 of 18-131 has CMSY10 spans "B" (the calligraphic B of the formulas) beside CMEX10 "b" (a wide hat): the
    # first is a letter of the page, the second is not
    import pymupdf

    with pymupdf.open(JMLR / "jmlr-2022-18-131_p22.pdf") as document:
        spans = [s for b in document[0].get_text("dict")["blocks"] for l in b.get("lines", ()) for s in l["spans"]]
        kept = "".join(w[4] for w in textlayer.page_words(document[0]))
    calligraphic = sum(s["text"].count("B") for s in spans if "CMSY" in s["font"])
    hats = sum(s["text"].count("b") for s in spans if "CMEX" in s["font"])
    assert calligraphic and hats
    assert kept.count("B") >= calligraphic
    assert kept.count("b") <= sum(s["text"].count("b") for s in spans) - hats


@pytest.mark.parametrize("pdf,pages", [("plain", 3), ("table", 3), ("formula", 3), ("scan", 1)])
def test_words_built_from_glyphs_are_pymupdfs_words_when_no_font_is_left_out(pdf, pages):
    import pymupdf

    with pymupdf.open(FIXTURES / f"{pdf}.pdf") as document:
        for number in range(pages):
            page = document[number]
            raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_WORDS)
            assert textlayer._rebuild_words(raw, skip=lambda font: False) == [tuple(w) for w in page.get_text("words")], (pdf, number)
            assert textlayer.page_words(page, skip=lambda font: False) == [tuple(w) for w in page.get_text("words")], (pdf, number)


def test_a_page_without_those_fonts_has_the_same_reference_words_as_before():
    import pymupdf

    for pdf in ("plain", "table", "formula"):
        with pymupdf.open(FIXTURES / f"{pdf}.pdf") as document:
            for number in range(3):
                page = document[number]
                fonts = {s["font"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", ()) for s in l["spans"]}
                if any(textlayer.is_non_text_font(f) for f in fonts):
                    continue
                before = textlayer.reference_from_words(number, page.rect.width, page.rect.height, page.get_text("words"))
                after = textlayer.page_reference(FIXTURES / f"{pdf}.pdf", number)
                assert [(w.text, w.bbox, w.block, w.mathy) for w in after.words] == [(w.text, w.bbox, w.block, w.mathy) for w in before.words]


# ----------------------- 2. accents, dotless letters and the increment sign are spelled alike on both sides
def test_a_letter_with_an_accent_is_the_letter_on_both_sides():
    assert checks.letters("Schl\u00f6r") == checks.letters("Schlo\u0308r") == checks.letters("Schlor") == "schlor"
    assert checks.alpha_tokens("na\u00efve na\u0131ve") == ["naive", "naive"]
    assert normalize.comparable("na\u00efve") == "naive"


def test_a_spacing_accent_glyph_of_the_layer_is_dropped_and_a_hat_is_not_a_letter():
    # CMMI sets a hat as its own character: "\u02c6" "\u02c7" "\u02dc" in the layer
    assert checks.letters("x\u02c6y\u02c7\u02dc") == "xy"
    assert normalize.comparable("x \u02c6") == "x"


def test_the_increment_sign_of_the_layer_is_the_delta_the_markdown_spells():
    # Computer Modern's capital Delta reaches the layer as U+2206; KaTeX (and the model's \Delta) as U+0394
    assert checks.letters("\u2206t") == checks.letters("\u0394t") == "deltat"
    assert checks._symbols("\u0394t") == checks._symbols("\u2206t") == {"delta", "t"}


def test_a_mark_on_a_figure_is_not_folded_away():
    # 0.3 + combining dot + 3: the dot over a digit makes the figure a repeating decimal; "0.33" is another figure
    assert "\u0307" in normalize.comparable("0.3\u03073")
    assert checks._numbers(normalize.comparable("0.3\u03073")) != checks._numbers("0.33")


@pytest.mark.parametrize("name", ["20-064_p37", "21-1148_p23", "20-1469_p22"])
def test_delta_and_barred_letters_do_not_change_the_length_or_the_table_words_of_a_page(katex, name):
    # 60 and 66 U+2206 Deltas the Markdown spells "delta", 29 combining bars on the layer's letters
    v = verdict(katex, name)
    assert clean(v), v.reasons


# ------------------------------------------------ 3. a multiplication sign counts the same on both sides of the length
def test_a_multiplication_sign_in_a_table_is_an_x_on_both_sides_of_the_length_ratio(katex):
    # 124 U+00D7 in the layer's table words; the Markdown's cells count each as the letter x (comparable_cell)
    assert checks.comparable_cell("2\u00d73") == checks.comparable_cell("2x3") == "2x3"
    v = verdict(katex, "20-667_p23")
    assert clean(v), v.reasons
    assert 0.85 <= v.metrics["length_ratio"] <= 1.15


# ---------------------------------- 4. words of a table cell: brackets and commas end a word, a literal \n is a break
def test_a_cell_that_sets_a_figure_and_its_deviation_together_is_two_words_of_the_layer(katex):
    # "0.015(0.00477)" where the layer has the figure and "(0.00477)" on the next line of the cell
    v = verdict(katex, "20-964_p19")
    assert not [r for r in v.reasons if r.startswith("table")], v.reasons


def test_a_literal_backslash_n_in_a_cell_is_a_line_break_not_the_letter_n(katex):
    v = verdict(katex, "21-0840_p26")
    assert clean(v), v.reasons


@pytest.mark.parametrize("html,expected", [
    ("<td>0.058\\n(0.001)</td>", "<td>0.058<br>(0.001)</td>"),
    ("<td>84.7\\pm1.0</td>", "<td>84.7\u00b11.0</td>"),
    ("<td>$\\nabla f$ and \\nabla g \\neq 0 \\nu</td>", "<td>$\\nabla f$ and \\nabla g \\neq 0 \\nu</td>"),
    ("<td>$a \\pm b$ \\n $\\nonumber$</td>", "<td>$a \\pm b$ <br> $\\nonumber$</td>"),
    ("<td>\\n</td>", "<td><br></td>"),
])
def test_table_html_keeps_latex_commands_and_breaks_lines(html, expected):
    assert mdtext.clean_table_html(html) == expected


def test_the_extractor_writes_the_line_break_and_the_sign_it_verified():
    record = json.loads((JMLR / "jmlr-2022-21-0840_p26.json").read_text(encoding="utf-8"))
    data = record["result"]
    table = next(b for b in data["parsing_res_list"] if b["block_label"] == "table")
    assert "\\n(" in table["block_content"]
    markdown = paddleocr_vl._tidy_tables(data, "text before\n\n" + table["block_content"] + "\n\ntext \\n after")
    assert "\\n(" not in table["block_content"] and "<br>(" in table["block_content"]
    assert "<br>(" in markdown and "\\n(" not in markdown
    assert markdown.endswith("text \\n after")  # only table cells are touched


def test_a_bare_plus_minus_in_a_cell_is_the_sign_of_the_page_and_a_real_error_on_the_page_stays(katex):
    # p16 of 21-0946: the cells read "0.759\\pm0.022"; the page also has a repeated sentence (a real error)
    v = verdict(katex, "21-0946_p16")
    assert codes(v) == {"repetition"}, v.reasons


# -------------------------------------------------------------------------------- 5. the layer glues figures
def changed(katex, name: str, old: str, new: str) -> checks.PageVerdict:
    """The page's verdict with ``old`` replaced by ``new`` in every block: one figure the model got wrong."""
    layer, data = jmlr(name)
    seen = 0
    for block in data["parsing_res_list"]:
        seen += block["block_content"].count(old)
        block["block_content"] = block["block_content"].replace(old, new)
    assert seen, f"{old!r} is not on {name}"
    return checks.check_page(mdtext.parse_page(data, layer.width / data["width"], 0), layer, katex)


@pytest.mark.parametrize("name", ["19-297_p12", "19-529_p27"])
def test_a_footnote_mark_after_the_final_period_of_a_figure_is_glued_to_it(katex, name):
    # the layer: "cutoff of 1.414.8" and "section 5.1.14" for the Markdown's "1.414." and "5.1." + a footnote 14
    v = verdict(katex, name)
    assert clean(v), v.reasons


def test_the_figure_before_a_footnote_mark_still_has_to_be_the_pages(katex):
    v = changed(katex, "19-297_p12", "1.414", "1.415")
    assert v.status == "fail" and codes(v) == {"number"} and "1.415" in v.reasons[0], v.reasons


def test_thousands_the_layer_sets_a_space_after_the_comma_of_are_the_same_figure(katex):
    # "p = 54, 675" in a table row of the layer, "54,675" in the Markdown's cell and in its prose
    v = verdict(katex, "20-108_p25")
    assert clean(v), v.reasons
    v = changed(katex, "20-108_p25", "54,675", "54,676")
    assert v.status == "fail" and codes(v) == {"number"} and "54,676" in v.reasons[0], v.reasons


def test_a_reference_number_the_layer_breaks_over_a_line_is_on_the_page(katex):
    # "JiangsuSF (BK" / "20200064)": eight digits
    v = verdict(katex, "20-1239_p42")
    assert clean(v), v.reasons
    v = changed(katex, "20-1239_p42", "20200064", "20200065")
    assert v.status == "fail" and codes(v) == {"number"} and "20200065" in v.reasons[0], v.reasons


@pytest.mark.parametrize("name,old,new,figure", [
    ("21-0052_p13", "_{17}", "_{18}", "'18'"),                  # \mathbf{1}_{17}: the layer has 117
    ("20-748_p41", "\\frac{144\\rho", "\\frac{145\\rho", "'145'"),  # 2\frac{144\rho..}{..}: the layer has 2144
    ("20-964_p17", "1.2^{40}", "1.2^{43}", "'43'"),            # a decimal base with a script: the layer has 1.240
    ("21-0078_p31", "^{2}4nM", "^{2}6nM", "'6'"),              # (..)^{2}4nM: the layer has 24nm
    ("21-1519_p20", "\\frac{2}{4}3c", "\\frac{2}{4}9c", "'9'"),  # a denominator and the figure after it
    ("21-0061_p44", "\\boxed{72958}", "\\boxed{72959}", "'72959'"),  # the layer sets the digits of a box one by one
    ("21-0345_p22", "1.01", "1.02", "'1.02'"),                 # 61.90 with the subscript 1.01: the layer has 61.901.01
])
def test_figures_of_one_formula_the_layer_runs_together_are_on_the_page_and_a_changed_one_is_not(katex, name, old, new, figure):
    v = verdict(katex, name)
    assert clean(v), v.reasons
    v = changed(katex, name, old, new)
    assert v.status == "fail" and codes(v) == {"number"} and figure in " ".join(v.reasons), v.reasons


def test_a_comma_joined_pair_of_decimals_is_two_figures():
    assert checks._numbers("(0.2,0.4)") == ["2,4"]
    assert checks._numbers("(0.2, 0.4) .045 0.045 1,000") == ["2", "4", "045", "045", "1,000"]


# ------------------------------------------- 6. a TeX layer writes the superscript before the subscript, KaTeX after
@pytest.mark.parametrize("tex,expected", [
    ("k_{ij}^{p}", "k^{p}{}_{ij}"),
    ("k^{p}_{ij}", "k^{p}{}_{ij}"),
    ("x_a^b + y", "x^{b}{}_{a} + y"),
    ("\\sum_{i=1}^{n} x_i", "\\sum^{n}{}_{i=1} x_i"),
    ("x_{a_{1}^{2}}^{b}", "x^{b}{}_{a^{2}{}_{1}}"),
    ("\\left(\\frac{a_i^2}{b}\\right)_{k}^{2}", "\\left(\\frac{a^{2}{}_{i}}{b}\\right)^{2}{}_{k}"),
    ("x_{a}", "x_{a}"), ("x^{b}", "x^{b}"), ("x_a y^b", "x_a y^b"), ("a \\{ x_1^2 \\}", "a \\{ x^{2}{}_{1} \\}"),
])
def test_scripts_are_written_superscript_first(tex, expected):
    from pipeline.paper_crawler.converters.verify.katex import layer_script_order

    assert layer_script_order(tex) == expected


def test_katex_gives_the_leaves_in_the_order_the_layer_writes_them(katex):
    render = katex.render(r"k_{ij}^{p}")
    assert render.ok and render.chars.split() == ["k", "i", "j", "p"] and render.layer_chars.split() == ["k", "p", "i", "j"]
    assert katex.render("x_i").layer_chars == "" and katex.render("x^2").layer_chars == ""
    # a formula KaTeX rejects stays rejected: the swapped copy is not what is judged
    assert not katex.render(r"\frac{a_i^2}{b").ok


@pytest.mark.parametrize("name", ["20-021_p48", "20-1304_p26"])
def test_formulas_the_layer_orders_differently_are_not_omissions(katex, name):
    # page text "dcovxkzikdcovzikzikcov..." for Markdown cov(x_k, z_{ik}) pieces with sub and sup in the other order
    v = verdict(katex, name)
    assert clean(v), v.reasons


def test_a_dropped_paragraph_is_still_an_omission_on_such_a_page(katex):
    layer, data = jmlr("20-1304_p26")
    text = [b for b in data["parsing_res_list"] if b["block_label"] == "text"]
    biggest = max(text, key=lambda b: len(b["block_content"]))
    biggest["block_content"] = ""
    v = checks.check_page(mdtext.parse_page(data, layer.width / data["width"], 0), layer, katex)
    assert v.status == "fail" and "omission" in codes(v), v.reasons


# ------------------------------------------------ 6b. the same order in a table cell's formula (the words of the table)
def test_a_table_cell_formula_the_layer_orders_differently_has_its_words_in_the_cells(katex):
    # \mu (\tau_{{C}^{\theta}_{\mu}}): the layer has "taucthetamu", KaTeX's leaves are tau, C, mu, theta
    v = verdict(katex, "21-0663_p16")
    assert clean(v), v.reasons


# ------------------------------------------- 7. a narrow header the layer reads line by line over its neighbours
def test_header_words_wrapped_over_two_lines_of_neighbouring_columns_are_in_the_table(katex):
    # "Discretization Invariance | Irregular Grids | Prediction | Solve PDE": the layer has "discretizairrepredicsolve" on
    # the first line, "tion invagular tion pde" on the second, "riance grids" on the third
    v = verdict(katex, "22-0297_p5")
    assert clean(v), v.reasons
    # a wrapped word stays on record like any table word that is only near the page's: nothing refuses the page
    assert [s.split("'")[1] for s in v.soft if s.startswith("table")] == [
        "discretizationinvariance", "irregulargrids", "prediction", "inverseproblem"], v.soft


@pytest.mark.parametrize("old,new,cell", [
    ("Discretization", "Quantization", "quantizationinvariance"),   # a word of the header that is not on the page at all
    ("Prediction", "Extrapolation", "extrapolation"),
    ("Inverse Problem", "Forward Mapping", "forwardmapping"),
])
def test_a_header_word_that_is_not_on_the_page_is_not_found_in_it(katex, old, new, cell):
    # one wrong header cell is a recorded small difference, as it always was; what matters is that it is not found
    v = changed(katex, "22-0297_p5", old, new)
    assert any(s.startswith(f"table: cell {cell!r}") and "% of its words are in the table region" in s for s in v.soft + v.reasons), (
        v.reasons, v.soft)


def test_two_header_words_that_are_not_on_the_page_refuse_the_page(katex):
    layer, data = jmlr("22-0297_p5")
    for block in data["parsing_res_list"]:
        block["block_content"] = block["block_content"].replace("Discretization", "Quantization").replace("Prediction", "Extrapolation")
    v = checks.check_page(mdtext.parse_page(data, layer.width / data["width"], 0), layer, katex)
    assert v.status == "fail" and codes(v) == {"table"}, v.reasons


# ------------------------------------------------ 8. a text layer whose letters are shifted is no reference
def test_a_layer_set_in_a_font_with_shifted_letters_is_no_reference():
    # p26 of 20-843: the caption and the chart text are "WKH" for "the", "SHUIRUPDQFH" for "performance" (23% of the page's
    # letters); the layer was the reference, so the correct caption was a fabrication and the garbled words were omitted
    ref = textlayer.page_reference(JMLR / "jmlr-2022-20-843_p26.pdf", 0)
    assert ref.source == "none" and not ref.usable
    assert "shifted by -3" in ref.reason, ref.reason


def test_a_page_whose_layer_is_garbled_is_left_to_the_ocr_reference(katex):
    layer, data = jmlr("20-843_p26")
    verdict_ = checks.check_page(mdtext.parse_page(data, layer.width / data["width"], 0), layer, katex)
    assert verdict_.status == "unverifiable" and codes(verdict_) == {"no_reference"}, verdict_.reasons


def test_the_shift_is_found_in_the_fonts_text_not_in_the_pages_words():
    from pipeline.paper_crawler.converters.verify import garble
    import pymupdf

    with pymupdf.open(JMLR / "jmlr-2022-20-843_p26.pdf") as document:
        share, shift = garble.garbled_share(document[0])
    assert shift == -3 and garble.DOMINANT_SHARE <= share < 0.5


@pytest.mark.parametrize("name", [
    "20-064_p34",    # a table of CMTT10 names: "MSR-q95" shifted by 2 is "OUT/s;7", and "out" is a word, a hundred times
    "20-830_p78",    # a table of CMR6: "Param/Opt/Bias-init" shifted by 1 is 35% common words
    "21-0819_p51",   # CMR10 formulas of "log" and "inf": under a shift of -24 they are the word "two"
    "20-064_p37", "18-131_p22", "22-0297_p5", "20-021_p48",
])
def test_the_layers_that_only_look_like_a_shift_stay_references(name):
    ref = textlayer.page_reference(JMLR / f"jmlr-2022-{name}.pdf", 0)
    assert ref.source == "textlayer", ref.reason


def test_a_font_needs_many_different_words_to_be_a_shifted_english():
    from pipeline.paper_crawler.converters.verify import garble

    english = ("the performance and the distribution of the base models show that the color bar and the size of the circle "
               "are the average error on the training and test data with the same parameters ") * 2
    moved = "".join(chr(ord(c) + 3) if c.isalpha() else c for c in english)
    assert garble.font_shift(english) is None            # as written it is English
    assert garble.font_shift(moved) == -3
    assert garble.font_shift("MSR-q95 PLS MRCE-Or " * 20) is None   # one word repeated, however often, is chance
    assert garble.font_shift(moved[:120]) is None        # the tags of a figure are too few to judge


# ------------------------------------------------ 9. the merged document's letters against its pages' (paper level)
def test_the_images_a_merge_leaves_out_are_left_out_of_the_pages_letters_too():
    # real PaddleOCR-VL pages with their image lines: "![](imgs/img_in_image_box_635_522_1116_729.jpg)" is 20 letters that
    # the merged document (written without images) does not have; counted on one side only they read as lost text
    pages = [(FIXTURES / name).read_text(encoding="utf-8") for name in
             ("formula.page1.md", "formula.page2.md", "table.page2.md", "plain.page1.md")]
    images = sum(len(paddleocr_vl._IMAGE_BLOCK.findall(page)) for page in pages)
    merged = "\n\n".join(paddleocr_vl._IMAGE_BLOCK.sub("", page) for page in pages)
    assert images == 3
    assert sum(len(checks.letters(page)) for page in pages) - len(checks.letters(merged)) == 20 * images
    assert paddleocr_vl._assembly_ratio(pages, merged) == pytest.approx(1.0)


def test_a_merge_that_drops_text_still_lowers_the_ratio():
    pages = [(FIXTURES / name).read_text(encoding="utf-8") for name in ("formula.page1.md", "plain.page1.md")]
    merged = "\n\n".join(paddleocr_vl._IMAGE_BLOCK.sub("", page) for page in pages)
    lower, upper = paddleocr_vl.ASSEMBLY_LETTER_RATIO
    assert lower <= paddleocr_vl._assembly_ratio(pages, merged) <= upper
    assert paddleocr_vl._assembly_ratio(pages, merged[: len(merged) * 9 // 10]) < lower


def test_a_figure_inside_a_longer_figure_of_the_page_is_not_the_page_figure_wrapped(katex):
    # VLDB 2022 p183 p12 (PaddleOCR-VL): the acknowledgement "No. 2019A1515110473" is written "2019A15110473", the "15" dropped.
    # "15110473" is in the page's text with its spaces taken out, but as the middle of a longer figure, not a figure
    v = verdict(katex, "vldb-2022-p183-chen_p12")
    assert v.status == "fail" and "number: not on the page: ['15110473']" in v.reasons, v.reasons


# ------------------------------------------------ 4b. a table that loses figures the page has
def test_a_table_that_loses_figures_the_page_has_is_refused(katex):
    # VLDB 2022 p375 p4 (MinerU): Table 2 has the relation D of the page ("d1: 1 5, d2: 2 7, d3: 2 8") with no figures in it.
    # Its words are letters and its figures are digits: only the first were counted, and the page passed
    v = verdict(katex, "vldb-2022-p375-konstantinidis_p4")
    assert v.status == "fail" and any(r.startswith("table: ") and "digits" in r for r in v.reasons), (v.status, v.reasons)


# ------------------------------------------------ the pages the model really got wrong stay refused
@pytest.mark.parametrize("name,kind,evidence", [
    ("21-0992_p47", "number", "'11'"),           # a table's "l" read as the figure 1: "11" is not on the page
    ("21-0211_p23", "number", "'33'"),           # a lost decimal point: ".33" read as "33"
    ("21-0946_p16", "repetition", "datasets with the best except svc and"),   # one phrase repeated three times
    ("19-882_p25", "formula_symbol", "lacks characters the page has there"),  # symbols of two display formulas read wrongly
    ("21-023_p26", "number", "['0']"),           # a figure the page does not have
    ("21-1146_p37", "fabrication", "is 42% on the page"),                     # text the page does not have
])
def test_a_page_the_model_got_wrong_is_still_refused(katex, name, kind, evidence):
    v = verdict(katex, name)
    assert v.status == "fail" and kind in codes(v), v.reasons
    assert any(evidence in reason for reason in v.reasons), v.reasons
