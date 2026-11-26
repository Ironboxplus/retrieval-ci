"""Per-page checks of generated Markdown against the page's own text layer.

Every check answers one question the model can get wrong without telling anyone:

* fabrication - prose that is not on the page (a hallucinated or duplicated passage);
* omission    - page text the Markdown does not have (a dropped paragraph, or a block the
                pipeline returned empty or cut short, which it does silently);
* numbers     - a digit string that is not on the page, or that appears more often;
* repetition  - an n-gram repeated more often than on the page (decoding loops);
* length      - Markdown and page text of grossly different size;
* table       - a cell whose text is not inside the table's region, or table text missing;
* formula_render / formula_symbol - a formula KaTeX (the frontend's renderer) rejects, or
                one whose letters are not on the page where it sits.

Prose is compared as letters only, formulas left out. A text layer glues inline
mathematics to its neighbours ("containinguand") and orders sub/superscripts differently
from LaTeX, so word- or window-level fuzzy matching drowns in noise on mathematical
pages. Instead a text is cut into overlapping SHINGLE-letter shingles, and a stretch of
text is "covered" when its shingles occur in the other side; a fabricated or missing
passage is a long stretch with almost no covered letters, while glue only nicks a few
letters around each variable. A table cell is judged by its words being in the table's
region (a header that wraps is read row by row, so a cell's text is not in one piece in the
layer); olmOCR's ``TextPresenceTest`` fuzzy matching was tried first and dropped for it.

The thresholds are calibrated on real pages; CALIBRATION says on which.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from .homoglyphs import present
from .katex import FormulaRender, KatexRenderer
from .mdtext import GREEK_NAMES, FormulaSpan, PageMarkdown, Part
from .normalize import comparable, fold_letters, greek_name, repair_math, spell_greek
from .textlayer import BBox, PageRef, RefWord

# ---- thresholds
SHINGLE = 8                    # letters per shingle
FABRICATION_WINDOW = 80        # letters of Markdown prose judged together
FABRICATION_MIN_COVERAGE = 0.50
OMISSION_WINDOW = 160          # letters of page text judged together
OMISSION_MIN_COVERAGE = 0.50
MIN_STRETCH = 40               # a shorter run of letters proves nothing either way
NUMBER_SLACK = 1
NUMBER_COUNTED_FROM_DIGITS = 3
LONG_FIGURE = 8                # characters: a figure this long is not a coincidence of two neighbouring words (BK20200064)
REPEAT_NGRAM = 8               # words
REPEAT_MIN_LETTERS = 24        # runs of variables and abbreviations ("ic ic ic") repeat legitimately
REPEAT_SLACK = 1
REPEAT_FORMULA_SLACK = 2
REPEAT_MIN_PIECE = 12          # letters of a gram between formulas that can be looked up on the page
LENGTH_RANGE = (0.85, 1.15)
TABLE_CELL_MIN_FOUND = 0.90    # share of a cell's words that are in its table's region
TABLE_MIN_RECALL = 0.90
TABLE_SOFT_FOUND = 0.80        # between this and the two above, a table issue is recorded and does not refuse the page
TABLE_SOFT_CELLS = 1           # a table with at most this many cells below it is recorded too
TABLE_MIN_WORDS = 3            # words in a table's region needed to judge how many of them its cells have
TABLE_DIGITS_MAX_LOST = 0.02   # share of the digits of a table's region that its cells may lack (formula glyphs, footnote marks)
TABLE_DIGITS_MIN_LOST = 4      # and how many digits at least: one footnote mark in a small table is no lost figure
LIGATURE_MIN_LETTERS = 8       # a word this long may lack one or two letters in the layer
SPACED_RUN = 4                 # one-character words in a row that are one word set glyph by glyph (code)
CHAIN_MAX = 4                  # neighbouring figures the layer may run together
NEAR_WORD_MIN_LETTERS = 6      # a table word this long may differ from the page's by one letter
WRAPPED_PART_MIN_LETTERS = 3   # letters of each of the two parts of a header word the layer sets on two lines
TABLE_REGION_MARGIN = 2.0  # slack around a table box when its cells are looked up
FORMULA_SYMBOL_MIN_COVERAGE = 0.95
FORMULA_REVERSE_MIN_COVERAGE = 0.90
FORMULA_REVERSE_SLACK = 1
FORMULA_SOFT_SYMBOLS = 1       # a formula with at most this many symbols the page lacks is recorded, not refused
FORMULA_WITNESS_AGREEMENT = 0.90  # share of each side's symbols the other one has
WITNESS_MIN_SHARE = 0.5        # the layer must hold this share of a block's letters to vouch for the block
WITNESS_MIN_CHARS = 20         # a box with less text than this in the layer is empty there
UNWITNESSED_MAX_SHARE = 0.5    # a page mostly of blocks the layer cannot vouch for is unverifiable
# Block labels of a reference list. User rule, 2026-11-15 ("参考文献无所谓了，正文没错就行"): what the model gets wrong
# inside a reference list (a page range it invents in a dense list, a repeated entry) is recorded and does not refuse the
# page. PaddleOCR-VL labels such a block ``reference_content``; MinerU's ``ref_text`` is mapped to the same label
# (extractors/mineru_pages.py). The blocks stay in the Markdown and in the omission comparison.
REFERENCE_LABELS = frozenset({"reference_content"})
REFERENCE_PREFIX = "reference list: "
CALIBRATION = (
    "2026-11-13, 388 pages of 30 VLDB 2022 papers converted by PaddleOCR-VL-1.6 (vLLM, kv auto) on the 4070 SUPER: "
    "of the 328 pages that pass, fabrication coverage has min 0.51 / p1 0.67 / p5 0.81 / median 0.98, omission "
    "min 0.55 / p1 0.70 / p5 0.83 / median 0.99, length ratio 0.95-1.05, table cell words found >= 0.95; each "
    "threshold sits in the gap between those and the pages that are wrong (a hallucinated paragraph or loop "
    "scores 0.00-0.40). Every failure was read against the PDF: tests/fixtures/pvl/calibration_cases.json keeps the "
    "pages behind the rules added by it, and the real model errors that must stay refused. "
    "2026-11-16, the table digit check (TABLE_DIGITS_MAX_LOST, TABLE_DIGITS_MIN_LOST): digits of a table's region that its cells "
    "lack, over the 1317 attempts that the quality reports kept blocks of (205 failing JMLR 2022 pages, 702 VLDB 2022 attempts) "
    "and the 72 cases: the passing tables with a digit lost lose at most 0.010 of them (8 of 788, formula glyphs the layer spells "
    "as digits) or fewer than 4 digits; the 8 pages this check refuses lose 0.032-0.559 of them in their best attempt (p375 p4 6 of 185 digits, p4002 p12 "
    "4.9%, p633 p10 5.5%, p2788 p7 7.2%) and each was read against the PDF: a column, a row or the figures of one are not in "
    "the Markdown's table."
)

_LEADING_ZERO = re.compile(r"(?<![\d.,])0(?=\.\d)")  # the 0 of 0.045
_FORMULA_MARK = "<formula>"
_LETTERS = re.compile(r"[^\W\d_]+")
# Private-use and Hangul code points: what a font without a usable ToUnicode map reaches the layer as
# (the Hangul block above U+D400 is repaired to mathematics by normalize.repair_math)
_UNMAPPED_GLYPH = re.compile(r"[\ue000-\uf8ff\uac00-\ud3ff]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# digits with a script made of digits: 2^{19}, 10_2, 1.2^{40}
_DIGIT_SCRIPT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[\^_]\s*\{*\s*(\d+(?:\.\d+)?)")
# an underbrace/overbrace drawn with the maths extension font reaches the text layer as "{z" "|{z}" "}" (bars, braces and a z)
_BRACE_GLYPHS = re.compile(r"[|{}z]*[|{}][|{}z]*")
_LIGATURES = ("ffi", "ffl", "ff", "fi", "fl")  # glyphs a text layer may leave out
_BARE_DIGIT_SCRIPT = re.compile(r"[\^_]\s*\{*\s*(\d+(?:\.\d+)?)\s*\}*")  # a formula that is only a script: _{16}, _{{1.01}}
_TRAILING_FIGURE = re.compile(r"(\d+(?:\.\d+)?\.?)\s*$")  # the figure a script may stand on: "2020." + "^{17}" is "2020.17"
_TRAILING_SCRIPT = re.compile(r"[\^_]\s*\{?\s*(\d+)\s*\}?$")  # a formula that ends in a digit script: log_2
_LEADING_DIGITS = re.compile(r"\s*(\d+)")
_OPERATOR_CHAIN = re.compile(r"\d+(?:\s*[*×·⋅∗+\-/]\s*\d+)+")  # 24*60*60: a layer may leave the operators out
# Figures that are neighbouring leaves of one rendered formula ("2 144" of 2\frac{144}{..}, "1 17" of \mathbf{1}_{17}, "2 4" of
# (..)^{2}4nM), with at most brackets, braces, bars and slashes between them ("1 ) 3" of (0,1)^3: the layer has no glyph for
# a big bracket that is drawn from pieces): a layer that sets numerator, script and base one after the other has them as one
# figure. A comma between them keeps them two: the layer has the comma.
_ADJACENT_FIGURES = re.compile(r"\d+(?:\.\d+)?(?:[\s)\]}|/{(\[]+\d+(?:\.\d+)?){1,3}")
# A script of one digit with a digit straight after it (v_10, x^23): in TeX that is the script and then a figure of its
# own, the classic way to write v_{10} wrong, so the figures of such a formula are never taken as glued.
_UNBRACED_SCRIPT_THEN_DIGIT = re.compile(r"[\^_]\s*\d\s*\d")
# a figure followed by a footnote mark after its final period: "section 5.1." + "14", "version 2.0." + "2"
_FOOTNOTED_FIGURE = re.compile(r"(?<![\d.,])(\d+\.\d+)\.\d{1,2}(?!\d)")
# the figures of a comma-joined pair the layer sets a space after the comma of: "54, 675", "(1, 3)"
_SPACED_COMMA = re.compile(r"(?<=\d),\s+(?=\d)")
# a word with letters and digits in it: mitdb_100_180, uint32_t
_ALNUM_WORD = re.compile(r"(?=\w*[^\W\d_])(?=\w*\d)\w+")
_GROUP_HEAD = re.compile(r"\D{0,2}(\d{1,3})")  # "∼2" of "∼2 219 520"
_GROUP_TAIL = re.compile(r"(\d{3}(?:\.\d+)?)[,;:.)\]]*")
_URL_FIGURE = re.compile(r"/(\d{5,})(?!\d)")  # the end of a DOI
_CITATION_LIST = re.compile(r"[\d\s,;.\u2013-]+")  # "12, 31, 34": figures and separators, no symbol to check
_CELL_WORD_BREAK = re.compile(r"[\s/\-()\[\],]+")  # where a table cell's words end


@dataclass
class PageVerdict:
    """``reasons`` are hard failures: real fabrication or omission, a figure that is not the page's, a
    table or formula that is not what the page holds. ``soft`` are issues that are recorded and leave the
    page standing: a formula KaTeX cannot parse, one symbol that differs, a table cell a little short of
    its words, a figure written another way (``02`` for ``O2``)."""
    page_no: int
    status: str  # "pass" | "fail" | "unverifiable"
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    soft: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


# ------------------------------------------------------------------ text helpers
def letters(text: str) -> str:
    """The letters of ``text`` (NFKC, case-folded, accents folded, Greek spelled out): the form two texts are
    compared in."""
    return "".join(_LETTERS.findall(_spelled(text)))


def _spelled(text: str) -> str:
    """``text`` as ``letters`` and ``alpha_tokens`` read it: the one folding both sides of every letter comparison go through."""
    return spell_greek(fold_letters(unicodedata.normalize("NFKC", repair_math(text)).casefold()))


def _without_accents(text: str) -> str:
    """``text`` decomposed, with its accents (combining marks, and the spacing ones a text
    layer writes for a hat or a bar) dropped."""
    return "".join(c for c in unicodedata.normalize("NFKD", fold_letters(text)) if unicodedata.category(c) not in ("Mn", "Lm", "Sk"))


def alpha_tokens(text: str, minimum: int = 3) -> list[str]:
    """Tokens of ``minimum`` or more letters, in the form ``letters`` compares."""
    return re.findall(rf"[^\W\d_]{{{minimum},}}", _spelled(text))


def _shingles(text: str) -> set[str]:
    return {text[i:i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}


def weakest_stretch(text: str, known: set[str], window: int) -> tuple[float, str]:
    """The lowest fraction of letters covered by ``known`` shingles over any ``window``-letter
    stretch of ``text`` (the whole text when it is shorter), and where that stretch is."""
    return weakest_stretch_pieces([text], known, window)


def weakest_stretch_pieces(pieces: list[str], known: set[str], window: int) -> tuple[float, str]:
    """``weakest_stretch`` of the pieces of a text laid end to end. A piece is a run of letters
    that is one piece on the other side too (the words between two formulas): no shingle spans
    two pieces, and a piece shorter than a shingle cannot be judged and is left out, so a
    paragraph that is mostly variables is neither matched across the gaps nor held against them."""
    judged = [p for p in pieces if len(p) >= SHINGLE]
    text = "".join(judged)
    if len(text) < MIN_STRETCH:
        return 1.0, ""
    covered: list[int] = []
    for piece in judged:
        flags = [0] * len(piece)
        for i in range(len(piece) - SHINGLE + 1):
            if piece[i:i + SHINGLE] in known:
                for j in range(i, i + SHINGLE):
                    flags[j] = 1
        covered += flags
    if len(text) <= window:
        return sum(covered) / len(text), text[:60]
    total = sum(covered[:window])
    best, at = total / window, 0
    for i in range(1, len(text) - window + 1):
        total += covered[i + window - 1] - covered[i - 1]
        if total / window < best:
            best, at = total / window, i
    return best, text[at:at + 60]


_LATEX_MARKUP = re.compile(r"\\(?!text)[A-Za-z]+|[{}^_$&\\]")


def _chars_of(formula: FormulaSpan, renders: dict[FormulaSpan, FormulaRender], layer_order: bool = False) -> str:
    """The characters KaTeX displays for ``formula``; for one it cannot parse, the words and letters of its
    source, which is what stays visible in the Markdown (so a block of pseudo-code in a broken array is
    still the page's text, not a hole in it). ``layer_order``: with the scripts in the order a TeX layer writes them."""
    render = renders[formula]
    return (render.layer_chars if layer_order and render.layer_chars else render.chars) if render.ok \
        else _LATEX_MARKUP.sub(" ", formula.tex)


def _flatten(parts: list[Part], renders: dict[FormulaSpan, FormulaRender], layer_order: bool = False) -> str:
    """Parts as text, each formula as the characters KaTeX displays for it."""
    return "".join(p if isinstance(p, str) else f" {_chars_of(p, renders, layer_order)} " for p in parts)


def _contains(region: BBox, box: BBox) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


@dataclass
class Attribution:
    """Which Markdown piece each page word belongs to: the smallest layout box holding its
    centre (a table region may overlap its own sub-captions, a merged text group spans two
    columns)."""
    prose: list[RefWord] = field(default_factory=list)  # in a prose box, or in none
    tables: list[list[RefWord]] = field(default_factory=list)
    skipped: list[RefWord] = field(default_factory=list)


def attribute(ref: PageRef, page: PageMarkdown) -> Attribution:
    boxes = [("prose", None, s.bbox) for s in page.prose]
    boxes += [("table", i, t.bbox) for i, t in enumerate(page.tables)]
    boxes += [("skip", None, b) for b in page.skip_boxes]
    out = Attribution(tables=[[] for _ in page.tables])
    for word in ref.words:
        holders = [b for b in boxes if _contains(b[2], word.bbox)]
        if holders:
            kind, index, _ = min(holders, key=lambda b: (b[2][2] - b[2][0]) * (b[2][3] - b[2][1]))
        else:
            kind, index = "prose", None  # text the layout found nowhere still has to be in the Markdown
        if kind == "table":
            out.tables[index].append(word)
        elif kind == "skip":
            out.skipped.append(word)
        else:
            out.prose.append(word)
    return out


# ------------------------------------------------------------------------ checks
# A second reader of a display formula's box: ``witness(box)`` is the LaTeX another model reads
# there (PP-FormulaNet), or None when it read nothing.
FormulaWitness = Callable[[BBox], Optional[str]]


def check_page(page: PageMarkdown, ref: PageRef, katex: KatexRenderer,
               formula_witness: Optional[FormulaWitness] = None, math_layer_broken: bool = False) -> PageVerdict:
    """``math_layer_broken``: a display formula elsewhere in the same PDF proved its maths font
    unreadable (``metrics["formulas_witnessed"]``), so inline formulas are not held to the layer."""
    reasons: list[str] = []
    soft: list[str] = []
    metrics: dict = {"reference": ref.source}

    unique = list(dict.fromkeys(page.formulas))
    renders = dict(zip(unique, katex.render_many([(f.tex, f.display) for f in unique])))
    broken = [f for f in unique if not renders[f].ok]
    metrics["formulas"] = len(unique)
    metrics["formulas_unrenderable"] = len(broken)
    # a formula KaTeX cannot parse stays in the Markdown as the raw LaTeX the model wrote: legible, not wrong
    for formula in broken[:5]:
        soft.append(f"formula_render: {formula.tex[:60]!r}: {renders[formula].error[:100]}")
    if page.stray_dollars:
        soft.append(f"formula_render: {page.stray_dollars} unpaired '$' delimiter(s)")

    if not ref.usable:
        reasons.append(f"no_reference: {ref.reason or 'too few words'}")
        return PageVerdict(page.page_no, "unverifiable", reasons, metrics, soft)

    page, unwitnessed, share = _witnessed(page, renders, ref)
    metrics["unwitnessed_blocks"] = unwitnessed
    metrics["unwitnessed_share"] = round(share, 3)
    where = attribute(ref, page)
    table_text =[[_flatten(cell, renders) for cell in table.cells] for table in page.tables]
    table_cells = [[comparable_cell(text) for text in texts] for texts in table_text]
    layer_cells = [[comparable_cell(_flatten(cell, renders, layer_order=True)) for cell in table.cells] for table in page.tables]
    prose = [[letters(p) for p in s.parts if isinstance(p, str)] for s in page.prose]
    md_flat = [_flatten(s.parts, renders) for s in page.prose]
    ref_letters = letters("".join(w.text for w in ref.words))
    ref_known = _shingles(ref_letters)
    # Formula text counts as Markdown text when asking what the page has that the Markdown
    # lacks: pseudo-code often arrives as a typeset array of \text{} lines.
    # A TeX layer writes a base's superscript before its subscript, KaTeX the other way round: both orders are the page's.
    md_known: set[str] = set()
    for segment, text in zip(page.prose, md_flat):
        md_known |= _shingles(letters(text)) | _shingles(letters(_flatten(segment.parts, renders, layer_order=True)))
        for part in segment.parts:
            md_known |= _shingles(letters(part if isinstance(part, str) else _chars_of(part, renders)))
            if not isinstance(part, str):
                md_known |= _shingles(letters(_chars_of(part, renders, layer_order=True)))

    is_reference = [s.label in REFERENCE_LABELS for s in page.prose]
    _check_fabrication([p for p, r in zip(prose, is_reference) if not r], ref_known, reasons, metrics)
    if any(is_reference):
        _check_fabrication([p for p, r in zip(prose, is_reference) if r], ref_known, soft, metrics,
                           "reference_fabrication_min_coverage", REFERENCE_PREFIX)
    _check_omission(where, md_known, reasons, metrics)
    _check_numbers(page, renders, table_text, ref, reasons, soft, metrics)
    _check_repetition(page, ref_letters, reasons, soft, metrics)
    _check_length(md_flat, table_cells, where, reasons, metrics)
    _check_tables(page, table_cells, layer_cells, table_text, where, ref, reasons, soft, metrics, math_layer_broken)
    witnessed = set(page.formulas)
    _check_formula_symbols([f for f in unique if f in witnessed], renders, ref, reasons, soft, metrics, katex,
                           formula_witness, math_layer_broken)
    status = "fail" if reasons else "unverifiable" if share > UNWITNESSED_MAX_SHARE else "pass"
    metrics["soft_issues"] = len(soft)
    return PageVerdict(page.page_no, status, reasons, metrics, soft)


def _witnessed(page: PageMarkdown, renders: dict[FormulaSpan, FormulaRender], ref: PageRef):
    """``page`` without the blocks the text layer cannot vouch for: little text of it where
    the block sits, and what the block says is not on the page either (a listing or a table
    drawn as outlines). Their words move to the skipped ones. Also how many blocks that is and
    what share of the page's Markdown characters they hold. A block that has the page's
    text in it, however much more (a paragraph copied five times), is still judged."""
    blocks = [(s.bbox, _flatten(s.parts, renders)) for s in page.prose]
    blocks += [(t.bbox, " ".join(_flatten(cell, renders) for cell in t.cells)) for t in page.tables]
    page_known = _shingles(comparable_cell("".join(w.text for w in ref.words)))
    bare, written, total = set(), 0, 0
    for box, text in blocks:
        mine = comparable_cell(text)
        total += len(mine)
        there = len(comparable_cell("".join(w.text for w in ref.words_in(box))))
        if len(mine) >= MIN_STRETCH and (there < WITNESS_MIN_CHARS or there < WITNESS_MIN_SHARE * len(mine)
                                         and weakest_stretch(mine, page_known, len(mine))[0] < WITNESS_MIN_SHARE):
            bare.add(box)
            written += len(mine)
    kept = replace(page, prose=[s for s in page.prose if s.bbox not in bare],
                   tables=[t for t in page.tables if t.bbox not in bare],
                   formulas=[f for f in page.formulas if f.bbox not in bare],
                   skip_boxes=[*page.skip_boxes, *bare])
    return kept, len(bare), written / total if total else 0.0


def comparable_cell(text: str) -> str:
    """A table cell as letters and digits only: a text layer spells quotes and dashes too
    variably, and a paper that prints ".045" is read back as "0.045"."""
    return "".join(c for c in _LEADING_ZERO.sub("", _times_as_x(comparable(text))) if c.isalnum())


def _times_as_x(text: str) -> str:
    """``text`` with its multiplication signs written as x: one is an x on the page as far as a reader can tell
    ("2x" for "2×"); some fonts reach the layer as the arrow to a bar U+21E5. A table cell is compared this way, so
    the page's cell words are counted this way too."""
    return text.replace("×", "x").replace("⇥", "x")


def _numbers(text: str) -> list[str]:
    """The figures of ``text``; "0.045" and ".045" are the same figure (a paper may print either), also inside a
    comma-joined one: "(0.2,0.4)" is the figures 2 and 4."""
    return [",".join(part[2:] if part.startswith("0.") else part for part in n.split(",")) for n in _NUMBER.findall(text)]


def _check_fabrication(prose: list[list[str]], ref_known: set[str], reasons: list[str], metrics: dict,
                       metric: str = "fabrication_min_coverage", prefix: str = "") -> None:
    worst, where = 1.0, ""
    for pieces in prose:
        coverage, excerpt = weakest_stretch_pieces(pieces, ref_known, FABRICATION_WINDOW)
        if coverage < worst:
            worst, where = coverage, excerpt
    metrics[metric] = round(worst, 3)
    if worst < FABRICATION_MIN_COVERAGE:
        reasons.append(f"{prefix}fabrication: Markdown text {where!r} is {worst:.0%} on the page (< {FABRICATION_MIN_COVERAGE:.0%})")


def _check_omission(where: Attribution, md_known: set[str], reasons: list[str], metrics: dict) -> None:
    # A word with typeset mathematics glued to it ("𝑄with") is not in reading order with its
    # neighbours: the page text is judged in the runs between such words.
    runs: list[list[str]] = [[]]
    for word in where.prose:
        if word.mathy:
            runs.append([])
        else:
            runs[-1].append(word.text)
    coverage, excerpt = weakest_stretch_pieces([letters("".join(run)) for run in runs], md_known, OMISSION_WINDOW)
    metrics["omission_min_coverage"] = round(coverage, 3)
    if coverage < OMISSION_MIN_COVERAGE:
        reasons.append(f"omission: page text {excerpt!r} is {coverage:.0%} in the Markdown (< {OMISSION_MIN_COVERAGE:.0%})")


def _split_commas(numbers: Counter) -> Counter:
    """``numbers`` and the parts of each comma-joined one: "901,1400" is also 901 and 1400 (a formula's
    rendering sets the comma apart, so the Markdown's figures are the parts), "1,000" also 1 and 000."""
    out = Counter(numbers)
    for number, count in numbers.items():
        if "," in number:
            for part in number.split(","):
                out[part] += count
    return out


def _variant_key(number: str) -> str:
    """A figure without what a paper may print either way: thousands separators and leading zeros."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", number):
        number = number.replace(",", "")
    return re.sub(r"^0+(?=\d)", "", number)


# what a glyph that reads as a digit may be on the page: the letter O or the empty set for 0, an l or i for 1; or
# nothing the text layer keeps
_DIGIT_LOOKALIKES = {"0": ("o", "ø", ""), "1": ("l", "i", "")}
_CONTEXT = 2  # letters and digits either side of a figure that must agree with the page for a look-alike to count


def _lookalike_figure(number: str, sources: list[str], page_alnum: str) -> bool:
    """The Markdown's ``number`` is a digit string the page has written with letters that look like its digits
    ("-02" for "-O2", "E0" for "EO"): the figure with those letters, and the two letters either side of it, is on
    the page. The context is what keeps a made-up figure from passing."""
    variants = {number.replace("0", a).replace("1", b) for a in ("0", *_DIGIT_LOOKALIKES["0"])
                for b in ("1", *_DIGIT_LOOKALIKES["1"])} - {number}
    for source in sources:
        for match in re.finditer(r"(?<![\d.,])" + re.escape(number) + r"(?![\d])", source):
            before = comparable_cell(source[:match.start()])[-_CONTEXT:]
            after = comparable_cell(source[match.end():])[:_CONTEXT]
            if any(before + variant + after in page_alnum for variant in variants):
                return True
    return False


def _collapse_spaced(text: str) -> str:
    """``text`` with every run of SPACED_RUN or more one-character words joined: a layer that sets code one glyph at
    a time ("u i n t 3 2 _ t") has no figure in it otherwise."""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        out.extend(["".join(run)] if len(run) >= SPACED_RUN else run)
        run.clear()

    for token in text.split(" "):
        if len(token) == 1:
            run.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return " ".join(out)


def _spaced_figures(text: str) -> list[str]:
    """The figures the page writes with a space between the thousands ("1 192", "100 000", "1 134.83"): each join of a
    group of up to three digits with the groups of three after it. Neighbouring figures in a table row ("100 194
    400 782") are told apart by trying every start."""
    tokens = text.split(" ")
    figures: list[str] = []
    for start, token in enumerate(tokens):
        head = _GROUP_HEAD.fullmatch(token)
        if not head:
            continue
        figure = head.group(1)
        for next_token in tokens[start + 1:]:
            tail = _GROUP_TAIL.fullmatch(next_token)
            if not tail:
                break
            figure += tail.group(1)
            figures.append(figure)
            if "." in tail.group(1) or tail.group(0) != tail.group(1):  # a decimal or a punctuation mark ends it
                break
    return figures


def _adjacent_figures(formula: FormulaSpan, render: FormulaRender) -> list[list[str]]:
    """The runs of figures that are neighbouring leaves of the formula's rendering, as chains to look for glued in the
    layer. Not for a formula with an unbraced one-digit script followed by a digit (``v_10``): that is the model's
    way of writing v_{10} wrong, and its figures 1 and 0 stay two figures."""
    if _UNBRACED_SCRIPT_THEN_DIGIT.search(formula.tex):
        return []
    return [_NUMBER.findall(run) for run in _ADJACENT_FIGURES.findall(render.chars)]


def _runs_together(chain: list[str], keys: list[str]) -> list[str]:
    """The figures of ``chain`` (neighbours in the Markdown) that the page has glued into one figure with a
    neighbour: "24", "60", "60" are "246060" there when the layer leaves the operators out."""
    glued: set[int] = set()
    for start in range(len(chain)):
        for end in range(start + 2, min(len(chain), start + CHAIN_MAX) + 1):
            joined = "".join(chain[start:end])
            joined = joined[2:] if joined.startswith("0.") else joined  # the keys are figures without the 0 of 0.5
            if any(joined in key for key in keys):
                glued.update(range(start, end))
    return [chain[i] for i in sorted(glued)]


def _check_numbers(page: PageMarkdown, renders: dict[FormulaSpan, FormulaRender], table_text: list[list[str]],
                   ref: PageRef, reasons: list[str], soft: list[str], metrics: dict) -> None:
    md_numbers: Counter = Counter()
    chains: list[list[str]] = []  # figures that are neighbours in the Markdown and may be one figure in the layer
    texts: list[str] = []  # the Markdown's own words, formulas left out
    # the Markdown's text, formulas as KaTeX displays them: what a figure's neighbours are, for a look-alike check
    sources: list[str] = [_flatten(segment.parts, renders) for segment in page.prose]
    body_figures: set[str] = set()  # the figures the Markdown has outside its reference list (body text, tables)
    for segment in page.prose:
        before = ""  # the text just before the part: a script of its own ("P4" + "_{16}") stands on its last digits
        in_body = segment.label not in REFERENCE_LABELS
        for index, part in enumerate(segment.parts):
            if isinstance(part, str):
                md_numbers.update(_numbers(part))
                if in_body:
                    body_figures.update(_numbers(part))
                texts.append(part)
                chains += [list(pair) for pair in _DIGIT_SCRIPT.findall(part)]  # 2^{30} written out in plain text
                chains += [_NUMBER.findall(chain) for chain in _OPERATOR_CHAIN.findall(part)]
                before = part
            else:
                md_numbers.update(_numbers(renders[part].chars))
                if in_body:
                    body_figures.update(_numbers(renders[part].chars))
                chains += [list(pair) for pair in _DIGIT_SCRIPT.findall(part.tex)]
                chains += [_NUMBER.findall(chain) for chain in _OPERATOR_CHAIN.findall(renders[part].chars)]
                chains += _adjacent_figures(part, renders[part])
                script, base = _BARE_DIGIT_SCRIPT.fullmatch(part.tex.strip()), _TRAILING_FIGURE.search(before)
                if script and base:
                    chains.append([base.group(1), script.group(1)])
                # a script that ends a formula stands on the digits that follow it ("log_2" + "64" is "264")
                ending = _TRAILING_SCRIPT.search(part.tex.strip())
                following = segment.parts[index + 1] if index + 1 < len(segment.parts) else None
                lead = _LEADING_DIGITS.match(following) if isinstance(following, str) else None
                if ending and lead:
                    chains.append([ending.group(1), lead.group(1)])
                before = ""
    for table in page.tables:  # a power of ten in a cell ("10^{13}") is glued to its base in the layer too
        for cell in table.cells:
            before = ""
            for part in cell:
                if isinstance(part, str):
                    before = part
                    continue
                chains += [list(pair) for pair in _DIGIT_SCRIPT.findall(part.tex)]
                chains += _adjacent_figures(part, renders[part])
                # a deviation set as a script of the figure before it ("6.07" + "_{0.03}") is "6.070.03" in the layer
                script, base = _BARE_DIGIT_SCRIPT.fullmatch(part.tex.strip()), _TRAILING_FIGURE.search(before)
                if script and base:
                    chains.append([base.group(1), script.group(1)])
                before = ""
    for cells in table_text:
        md_numbers.update(_numbers(" ".join(cells)))
        body_figures.update(_numbers(" ".join(cells)))
        sources.append(" ".join(cells))
        texts.extend(cells)
        # the layer sets a table's cells one after the other with no gap: "1267" + "10.8" is "126710.8"
        for left, right in zip(cells, cells[1:]):
            tail, head = _NUMBER.findall(left), _NUMBER.findall(right)
            if tail and head:
                chains.append([tail[-1], head[0]])
    page_text = " ".join(w.text for w in ref.words)
    page_alnum = comparable_cell(page_text)
    # A name with letters and digits in it ("mitdb_100_180", "uint32_t") is on the page when a word of the page has
    # its letters and digits, whatever the layer does with the underscores: its figures are the name's, not
    # figures of their own.
    page_words = {comparable_cell(token) for token in _collapse_spaced(page_text).split(" ")}
    named: Counter = Counter()
    for text in texts:
        for word in _ALNUM_WORD.findall(text):
            if comparable_cell(word) in page_words:  # a whole word of the page: "110ms" is not part of "1110ms"
                named.update(_numbers(word))
    md_numbers -= named
    ref_numbers = Counter(_numbers(page_text))
    # A figure followed by a footnote mark after its final period is glued to it: "5.1." + "14" is "5.1.14"
    ref_numbers |= Counter(n for base in _FOOTNOTED_FIGURE.findall(page_text) for n in _numbers(base))
    # "54,675" and "(1,3)" with a space after the comma ("54, 675", "(1, 3)")
    ref_numbers |= Counter(_numbers(_SPACED_COMMA.sub(",", page_text)))
    # A figure broken after its dot or slash at the end of a line (a DOI) is one figure
    ref_numbers |= Counter(_numbers(re.sub(r"(?<=[./])\s+(?=\d)", "", page_text)))
    # Code set one glyph at a time has its figures one digit apart
    ref_numbers |= Counter(_numbers(_collapse_spaced(page_text)))
    ref_numbers = _split_commas(ref_numbers)
    # A text layer glues a script to the digits it stands on ("2" and "19" of 2^{19} come as
    # "219"), and neighbours it sets with no gap: both are known when the page has them run together.
    keys = list(ref_numbers)
    for chain in chains:
        for glued in _runs_together(chain, keys):
            ref_numbers.update(_numbers(glued))
    # "[5,17]" is one number token to a regex and two to a reader: a comma-joined figure is
    # known when every part of it is.
    # A footnote mark is glued to the figure before it in a text layer ("10.1.47" + "2" is
    # "10.1.472"): a figure that is the start of a page figure with one digit more is known.
    # A long figure (a DOI) wrapped in the middle of its digits is two words in the layer: it is on the page when
    # the page's text with its spaces taken out has it as a figure of its own, not as the middle of a longer one
    # ("15110473" is not on a page that has "1515110473").
    run_together = re.sub(r"\s+", "", page_text)
    last_in_url = {n for text in texts for n in _URL_FIGURE.findall(text)}  # the end of a DOI wraps too
    unknown = sorted(n for n in md_numbers
                     if n not in ref_numbers and not all(part in ref_numbers for part in n.split(","))
                     and not any(r.startswith(n) and len(r) == len(n) + 1 for r in ref_numbers)
                     and not ((len(n) >= LONG_FIGURE or n in last_in_url) and re.search(rf"(?<!\d){re.escape(n)}(?!\d)", run_together)))
    overcounted = sorted(n for n, c in md_numbers.items()
                         if len(n) >= NUMBER_COUNTED_FROM_DIGITS and n in ref_numbers
                         and c > ref_numbers[n] + NUMBER_SLACK)
    # A figure that is another way of writing one the page has (a separator, a leading zero, a letter for a
    # digit) is recorded, not refused. One that differs ("110" for "1110") is a wrong number and refuses the page.
    page_keys = {_variant_key(n) for n in ref_numbers}
    page_keys |= {_variant_key(n) for figure in _spaced_figures(page_text) for n in _numbers(figure)}  # "1 192" for 1192
    variants = [n for n in unknown if _variant_key(n) in page_keys or _lookalike_figure(n, sources, page_alnum)]
    unknown = [n for n in unknown if n not in variants]
    if variants:
        soft.append(f"number: written differently from the page: {variants[:8]}")
    # A figure that only the reference list has is recorded, not refused; one the body or a table has too is the body's.
    reference_unknown = [n for n in unknown if n not in body_figures]
    unknown = [n for n in unknown if n in body_figures]
    reference_overcounted = [n for n in overcounted if n not in body_figures]
    overcounted = [n for n in overcounted if n in body_figures]
    metrics["numbers_unknown"] = len(unknown)
    metrics["reference_numbers_unknown"] = len(reference_unknown)
    if unknown:
        reasons.append(f"number: not on the page: {unknown[:8]}")
    if overcounted:
        reasons.append(f"number: more often than on the page: {overcounted[:8]}")
    if reference_unknown:
        soft.append(f"{REFERENCE_PREFIX}number: not on the page: {reference_unknown[:8]}")
    if reference_overcounted:
        soft.append(f"{REFERENCE_PREFIX}number: more often than on the page: {reference_overcounted[:8]}")


def _times_on_page(piece: str, ref_letters: str) -> int:
    """How often the page has ``piece``'s letters in a row; a layer that leaves the glyph of a ligature out
    ("\uffffnd" for "find") has the piece without those letters, and the page's own copies are counted both ways."""
    dropped = piece
    for ligature in _LIGATURES:
        dropped = dropped.replace(ligature, "")
    return ref_letters.count(piece) + (ref_letters.count(dropped) if dropped != piece else 0)


def _check_repetition(page: PageMarkdown, ref_letters: str, reasons: list[str], soft: list[str], metrics: dict) -> None:
    """Grams are counted apart for the body and for the reference list: a body repeat refuses the page, a
    reference-list repeat is recorded."""
    body, reference = _repeated_grams(page, ref_letters, reference=False), _repeated_grams(page, ref_letters, reference=True)
    metrics["repeated_ngrams"] = len(body)
    if any(s.label in REFERENCE_LABELS for s in page.prose):
        metrics["reference_repeated_ngrams"] = len(reference)
    for repeated, out, prefix in ((body, reasons, ""), (reference, soft, REFERENCE_PREFIX)):
        if repeated:
            gram, count, known = max(repeated, key=lambda item: item[1])
            out.append(f"{prefix}repetition: {' '.join(w for w in gram if w != _FORMULA_MARK)[:60]!r} appears {count} times "
                       f"(page has it {'?' if known is None else known})")


def _repeated_grams(page: PageMarkdown, ref_letters: str, reference: bool) -> list[tuple[tuple[str, ...], int, Optional[int]]]:
    """The grams repeated more often than the page has them, among the segments of the reference list
    (``reference``) or of everything else."""
    counts: Counter = Counter()
    for segment in page.prose:
        if (segment.label in REFERENCE_LABELS) != reference:
            continue
        # Every word, short ones too: the page is searched for the gram's letters run
        # together. A formula is a mark in the stream: the page has its symbols there.
        tokens: list[str] = []
        for part in segment.parts:
            tokens += alpha_tokens(part, 1) if isinstance(part, str) else [_FORMULA_MARK]
        for i in range(max(len(tokens) - REPEAT_NGRAM + 1, 0)):
            gram = tuple(tokens[i:i + REPEAT_NGRAM])
            if sum(map(len, gram)) >= REPEAT_MIN_LETTERS + gram.count(_FORMULA_MARK):
                counts[gram] += 1
    # The text layer glues words, so a phrase the page has twice may not be found in it at
    # all: a gram counts as present once, and flags only beyond that plus the slack. A gram
    # with a formula in it is looked up piece by piece; when no piece is long enough to look
    # up, it takes more copies to flag.
    repeated = []
    for gram, count in counts.items():
        pieces = "".join(t if t != _FORMULA_MARK else " " for t in gram).split()
        on_page = [_times_on_page(piece, ref_letters) for piece in pieces if len(piece) >= REPEAT_MIN_PIECE]
        known = min(on_page) if on_page else None
        marks = gram.count(_FORMULA_MARK)
        # a line of pseudo-code with its formulas ("h <- Propg(h, E) in Eq. (4)") legitimately comes back
        # once per line: each formula in the gram that cannot be looked up allows more copies
        if count > max(known or 0, 1) + REPEAT_SLACK + (REPEAT_FORMULA_SLACK * max(marks, 1) if known is None else 0):
            repeated.append((gram, count, known))
    return repeated


def _check_length(md_flat: list[str], table_cells: list[list[str]], where: Attribution, reasons: list[str],
                  metrics: dict) -> None:
    md_size = sum(len(letters(t)) for t in md_flat) + sum(len(letters(c)) for cells in table_cells for c in cells)
    # the Markdown's table cells count a multiplication sign as the letter x (comparable_cell), so do the page's
    ref_size = sum(len(letters(w.text)) for w in where.prose) + sum(len(letters(_times_as_x(w.text))) for ws in where.tables for w in ws)
    ratio = md_size / ref_size if ref_size else 1.0
    metrics["length_ratio"] = round(ratio, 3)
    if not LENGTH_RANGE[0] <= ratio <= LENGTH_RANGE[1]:
        reasons.append(f"length: Markdown/page letters = {ratio:.2f}, outside {LENGTH_RANGE}")


def _in_region(token: str, region: str) -> bool:
    """``token`` is in ``region``; a long one may lack one or two letters there (a layer that
    drops the glyph of a ligature writes "conguration")."""
    if token in region:
        return True
    # a word with a ligature in it ("fit") may be there without the ligature's letters, short as it is
    if any(ligature in token and token.replace(ligature, "", 1) in region for ligature in _LIGATURES):
        return True
    if len(token) < LIGATURE_MIN_LETTERS:
        return False
    return any(token[:i] + token[i + gap:] in region for gap in (1, 2) for i in range(1, len(token) - gap))


def _in_text(token: str, text: str) -> bool:
    """``token`` of the layer is in ``text``; a long one may be missing one or two letters
    the text has (the ligature of "configuration" again)."""
    if token in text:
        return True
    return len(token) >= LIGATURE_MIN_LETTERS and any(
        re.search(re.escape(token[:i]) + ".{1,2}" + re.escape(token[i:]), text) for i in range(1, len(token)))


def _near_word(token: str, region: str, words: set[str]) -> bool:
    """A table word that is not in the region, but is the page's with a small difference: a digit for the letter it
    looks like ("spectac1", "03" for "O3"), one letter more, less or other in a long word, or two words the layer
    sets apart (a header with a subscript under it: "OCSMM" over "KME"). A figure is none of these: two cells glued
    into "34.26.8x" is the model's mistake, however much of it the page has."""
    letters_in = sum(c.isalpha() for c in token)
    if any(digit in token for digit in "01") and (letters_in or len(token) <= 3):
        variants = {token.replace("0", a).replace("1", b) for a in ("0", "o", "ø") for b in ("1", "l", "i")} - {token}
        if any(variant in region for variant in variants):
            return True
    if len(token) >= NEAR_WORD_MIN_LETTERS and 2 * letters_in >= len(token):
        for i in range(1, len(token) - 1):
            head, tail = re.escape(token[:i]), re.escape(token[i + 1:])
            if token[:i] + token[i + 1:] in region or re.search(head + "." + tail, region) \
                    or re.search(head + "." + re.escape(token[i:]), region):
                return True
    return token.isalpha() and any(token[:i] in words and token[i:] in words for i in range(2, len(token) - 1))


def _wrapped_word(token: str, words: list[tuple[str, BBox]]) -> bool:
    """A table word the layer sets on two lines: a narrow header column wraps "Discretization" into "discretiza" and "tion",
    and the layer reads the headers of neighbouring columns line by line, so the pieces of one word are far apart in the
    region's text ("discretizairrepredicsolve" ... "tion"). ``token`` is there when its first part is in one layer word and
    its second part in another, on the line right below the first and overlapping it horizontally.
    ``words`` are (comparable text, box) of the region's words."""
    if len(token) < NEAR_WORD_MIN_LETTERS:
        return False
    for cut in range(WRAPPED_PART_MIN_LETTERS, len(token) - WRAPPED_PART_MIN_LETTERS + 1):
        head, tail = token[:cut], token[cut:]
        for upper_text, upper in words:
            if head not in upper_text:
                continue
            height = upper[3] - upper[1]
            for lower_text, lower in words:
                if tail in lower_text and upper[3] - height * 0.5 <= lower[1] <= upper[3] + height \
                        and lower[0] < upper[2] and lower[2] > upper[0]:
                    return True
    return False


def _lost_digits(layer_text: str, cell_text: str) -> tuple[int, int]:
    """(digits of the table region that the table's cells have not, digits of the region). Digits are counted as characters:
    a figure the layer glues to its neighbour, breaks with a space, or sets a script of ("10" "3" for 10^3) keeps its digits."""
    layer, cells = Counter(re.findall(r"\d", layer_text)), Counter(re.findall(r"\d", cell_text))
    return sum(max(0, count - cells[digit]) for digit, count in layer.items()), sum(layer.values())


def _check_tables(page: PageMarkdown, table_cells: list[list[str]], layer_cells: list[list[str]], table_text: list[list[str]],
                  where: Attribution, ref: PageRef, reasons: list[str], soft: list[str], metrics: dict,
                  layer_broken: bool = False) -> None:
    worst, missing = 1.0, 0.0
    lost_share = 0.0
    wrong: list[str] = []  # the cells of one table that are mostly not in it
    for table, cells, in_layer_order, inside, texts in zip(page.tables, table_cells, layer_cells, where.tables, table_text):
        inside_words = ref.words_in(table.bbox, TABLE_REGION_MARGIN)
        region = comparable_cell("".join(w.text for w in inside_words))
        region_words = {comparable_cell(w.text) for w in inside_words}
        placed_words = [(comparable_cell(w.text), w.bbox) for w in inside_words]
        near: dict[str, None] = {}  # cells with a word that is near the page's, in order
        for parts, cell in zip(table.cells, cells):
            # A cell with a formula in it is judged by the formula checks: the layer orders
            # its subscripts and symbols differently
            if len(cell) < 2 or any(not isinstance(p, str) for p in parts):
                continue
            # By words, not as one string: the layer reads a header that wraps over two lines
            # row by row, so the cell's text is not in one piece there. A bracket or a comma ends a word too: "0.015
            # (0.00477)" is two words of the layer's whichever way the cell sets them
            tokens = [t for t in (comparable_cell(w) for w in re.split(_CELL_WORD_BREAK, "".join(parts))) if t]
            if layer_broken:  # a symbol of a maths font the layer cannot spell ("L", a beta) is no word it can be held to
                tokens = [t for t in tokens if len(t) > 1 and t not in GREEK_NAMES]
            exact = [_in_region(t, region) for t in tokens]
            close = [e or _near_word(t, region, region_words) or _wrapped_word(t, placed_words) for e, t in zip(exact, tokens)]
            found = sum(close) / len(tokens) if tokens else 1.0
            if sum(close) > sum(exact):
                near[cell[:40]] = None
            worst = min(worst, found)
            if found < TABLE_CELL_MIN_FOUND:
                message = (f"table: cell {cell[:40]!r}: {found:.0%} of its words are in the table region "
                           f"(< {TABLE_CELL_MIN_FOUND:.0%})")
                # a cell a little short of its words is recorded; one mostly not in the table is wrong, but a lone
                # one (a misread letter in a header) is the same small difference as one wrong symbol
                (soft if found >= TABLE_SOFT_FOUND else wrong).append(message)
        # the same misread header over every column group is one misreading
        (soft if len(set(wrong)) <= TABLE_SOFT_CELLS else reasons).extend(dict.fromkeys(wrong))
        soft.extend(f"table: cell {cell!r}: a word is one small difference from the page's" for cell in near)
        wrong = []
        lost, digits = _lost_digits(" ".join(w.text for w in inside), " ".join(texts))
        if digits:
            lost_share = max(lost_share, lost / digits)
        if lost >= TABLE_DIGITS_MIN_LOST and lost / digits > TABLE_DIGITS_MAX_LOST:
            # the figures of a table are what it is for: cells that lack some are a table that lost rows or columns
            reasons.append(f"table: {lost} of the {digits} digits in the table region are not in its cells "
                           f"({lost / digits:.1%} > {TABLE_DIGITS_MAX_LOST:.0%})")
        words = [t for w in inside for t in alpha_tokens(w.text)]
        joined, joined_layer_order = "".join(cells), "".join(in_layer_order)
        if len(words) >= TABLE_MIN_WORDS:  # a recall over one or two words (a stray glyph "hhhh") proves nothing
            recall = sum(_in_text(t, joined) or _in_text(t, joined_layer_order) for t in words) / len(words)
            missing = max(missing, 1.0 - recall)
            if recall < TABLE_MIN_RECALL:
                (soft if recall >= TABLE_SOFT_FOUND else reasons).append(
                    f"table: only {recall:.0%} of the words in the table region are in its cells "
                    f"(< {TABLE_MIN_RECALL:.0%})")
    metrics["table_cell_min_found"] = round(worst, 3)
    metrics["table_max_missing"] = round(missing, 3)
    metrics["table_digits_missing"] = round(lost_share, 3)


def _symbols(chars: str) -> set[str]:
    """The letters of a rendered formula, a Greek one as its name: the symbols a reader tells apart.
    (accents KaTeX draws as modifier letters are no symbol of the page's)"""
    return {greek_name(c) or c for c in fold_letters(unicodedata.normalize("NFKC", chars).casefold())
            if unicodedata.category(c) in ("Ll", "Lu", "Lt", "Lo")}


def _witness_agrees(formula: FormulaSpan, render: FormulaRender, witness: FormulaWitness, katex: KatexRenderer) -> bool:
    """Another model, reading the formula's box on its own, finds the same symbols in it. Two
    readers that share nothing but the picture agreeing is evidence against a text layer whose
    maths font cannot be spelled (a Greek letter reaches it as "u", a sum as "õ")."""
    tex = witness(formula.bbox)
    if not tex or not tex.strip():
        return False
    other = katex.render(tex.strip(), True)
    mine, theirs = _symbols(render.chars), _symbols(other.chars) if other.ok else set()
    if not mine or not theirs:
        return False
    shared = len(mine & theirs)
    return shared / len(mine) >= FORMULA_WITNESS_AGREEMENT and shared / len(theirs) >= FORMULA_WITNESS_AGREEMENT


def _check_formula_symbols(unique: list[FormulaSpan], renders: dict[FormulaSpan, FormulaRender], ref: PageRef,
                           reasons: list[str], soft: list[str], metrics: dict, katex: KatexRenderer,
                           witness: Optional[FormulaWitness], layer_broken: bool = False) -> None:
    unverifiable, worst, witnessed = 0, 1.0, 0
    # formulas the layer disagrees with, and how: (formula, hard messages, soft messages)
    contradicted: list[tuple[FormulaSpan, list[str], list[str]]] = []
    for formula in unique:
        render = renders[formula]
        if not render.ok or _CITATION_LIST.fullmatch(formula.tex.strip()):  # "[12, 31, 34]" set as a formula: no symbol
            continue
        if ref.source == "ocr":  # recognised mathematics is no witness for a formula's symbols
            unverifiable += 1
            continue
        nearby = ref.words_in(formula.bbox)
        if not nearby:
            unverifiable += 1  # outlined or image-only glyphs: nothing to compare with
            continue
        there = "".join(w.text for w in nearby)  # comparable form: Greek letters spelled out
        if _UNMAPPED_GLYPH.search(there):
            unverifiable += 1  # a font the layer cannot spell
            continue
        have = set(there)
        # A symbol is a Latin letter or a Greek one (judged by its name, one unit)
        wanted = _symbols(render.chars)
        own: list[str] = []
        own_soft: list[str] = []
        if wanted:
            # a symbol the layer holds as a look-alike (nu as "v", Delta as a triangle) is the same symbol
            missing = {s for s in wanted if not (s in have if len(s) == 1 else s in there) and not present(s, have, there)}
            cover = 1 - len(missing) / len(wanted)
            worst = min(worst, cover)
            if cover < FORMULA_SYMBOL_MIN_COVERAGE:
                # one symbol that differs (an i for a j) is recorded, more than one refuses the page
                (own_soft if len(missing) <= FORMULA_SOFT_SYMBOLS else own).append(
                    f"formula_symbol: {formula.tex[:50]!r} has letters {sorted(missing)} "
                    f"that are not on the page there (coverage {cover:.2f} < {FORMULA_SYMBOL_MIN_COVERAGE})")
        if formula.display:
            # a box on a line of text also holds the words next to the formula: a plain word
            # of four letters or more is prose, a formula's words have symbols or digits in them
            own_text = "".join(w.text for w in nearby
                               if not (len(w.text) >= 4 and w.text.isalpha()) and not _BRACE_GLYPHS.fullmatch(w.text))
            page_chars = {c for c in _without_accents(own_text) if c.isalnum()}
            shown_text = _without_accents(spell_greek(unicodedata.normalize("NFKC", render.chars).casefold()))
            shown = set(shown_text)
            if page_chars:
                absent = {c for c in page_chars - shown if not present(c, shown, shown_text)}
                back = 1 - len(absent) / len(page_chars)
                # Symbol fonts reach the text layer as stray letters (a sum as "í", a join as
                # "z"), so one character the rendering lacks proves nothing.
                if back < FORMULA_REVERSE_MIN_COVERAGE and len(absent) > FORMULA_REVERSE_SLACK:
                    own.append(f"formula_symbol: display formula {formula.tex[:50]!r} lacks characters the page "
                               f"has there (coverage {back:.2f} < {FORMULA_REVERSE_MIN_COVERAGE})")
        if own or own_soft:
            contradicted.append((formula, own, own_soft))
    # A display formula has a box of its own, so a second model can read it. When it agrees with
    # the Markdown, it is the layer that is wrong, and the inline formulas of that page, which
    # have no box to read, cannot be held to the layer either. A formula with only a soft issue
    # is not worth the second reader: it is recorded.
    for formula, own, own_soft in sorted(contradicted, key=lambda item: not item[0].display):
        if own and witness is not None and formula.display and _witness_agrees(formula, renders[formula], witness, katex):
            witnessed += 1
            layer_broken = True
            continue
        if layer_broken and not formula.display:
            unverifiable += 1
            continue
        reasons.extend(own)
        soft.extend(own_soft)
    metrics["formulas_unverifiable"] = unverifiable
    metrics["formulas_witnessed"] = witnessed
    metrics["formula_symbol_min_coverage"] = round(worst, 3)
