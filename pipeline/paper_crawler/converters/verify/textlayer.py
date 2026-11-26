"""The PDF's own text layer as a per-page reference for a generated Markdown page.

The text layer is never used as body text; it only answers "is this word really on the
page" and "is this word missing from the Markdown". Words keep their bounding boxes so
the checks can leave out regions the Markdown is not expected to contain (headers,
figures) and look inside a table or formula region.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .garble import DOMINANT_SHARE, garbled_share
from .normalize import comparable

BBox = tuple[float, float, float, float]

# A page with fewer words than this has no usable text layer (a scan, or a cover image).
MIN_USABLE_WORDS = 30
# Extractors write unmappable glyphs as U+FFFD, private-use code points or "(cid:N)".
MAX_BROKEN_CHAR_FRACTION = 0.05

_PRIVATE_USE = re.compile("[\ue000-\uf8ff\ufffd]")
_CID = re.compile(r"\(cid:\d+\)")
_LINE_END_HYPHENS = ("-", "\u00ad")


@dataclass(frozen=True)
class RefWord:
    text: str  # comparable form (NFKC, case-folded, no whitespace)
    bbox: BBox
    block: int  # words of different blocks never form one window
    mathy: bool = False  # typeset mathematics glued into the word (``𝑄with``): its neighbours are not in reading order


@dataclass
class PageRef:
    page_no: int
    width: float
    height: float
    words: list[RefWord] = field(default_factory=list)
    source: str = "textlayer"  # "textlayer" | "ocr" | "none"
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.source != "none" and len(self.words) >= MIN_USABLE_WORDS

    @property
    def compact(self) -> str:
        return "".join(w.text for w in self.words)

    def words_in(self, box: BBox, margin: float = 0.0) -> list[RefWord]:
        """Words whose centre lies inside ``box`` grown by ``margin`` points."""
        x0, y0, x1, y1 = box
        return [w for w in self.words if _inside(w.bbox, (x0 - margin, y0 - margin, x1 + margin, y1 + margin))]

    def windows(self, boxes_to_skip: Sequence[BBox] = (), size: int = 8, stride: int = 4) -> list[str]:
        """Compact text of consecutive ``size``-word windows, per block, outside ``boxes_to_skip``."""
        kept = [w for w in self.words if not any(_inside(w.bbox, box) for box in boxes_to_skip)]
        return windows_of([(w.block, w.text) for w in kept], size, stride)

    @classmethod
    def from_lines(cls, page_no: int, width: float, height: float,
                   lines: Iterable[tuple[str, BBox]], source: str) -> "PageRef":
        """A reference from recognised text lines (OCR): every word shares its line's box."""
        words = []
        for number, (text, box) in enumerate(lines):
            words += [RefWord(w, box, number) for w in (comparable(t) for t in text.split()) if w]
        return cls(page_no, width, height, words, source=source)


def windows_of(tokens: Sequence[tuple[int, str]], size: int, stride: int) -> list[str]:
    """Compact text of consecutive ``size``-token windows; ``tokens`` are (block, text).

    A block shorter than ``size`` yields one window of everything it has, so a short
    caption is still checked."""
    out: list[str] = []
    start = 0
    while start < len(tokens):
        end = start
        while end < len(tokens) and tokens[end][0] == tokens[start][0]:
            end += 1
        block = [t for _, t in tokens[start:end]]
        if len(block) <= size:
            out.append("".join(block))
        else:
            out += ["".join(block[i:i + size]) for i in range(0, len(block) - size + stride, stride)]
        start = end
    return out


def _inside(box: BBox, region: BBox) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


# Fonts of TeX's extension and delimiter sets. Their glyphs are pieces of a drawing (a radical or a wide hat, a big
# bracket or brace, an integral sign, a big operator), but the text layer spells each one as the character at its code
# point, so a Latin letter ("b" for a wide hat, "p" "q" for the halves of a parenthesis) stands where the page has none
# (JMLR 2022: census of the 353 PDFs, tests/test_verify_textlayer_math_fonts.py). A font is matched by a part of its name
# so subset prefixes ("ABCDEF+CMEX10") and size suffixes need no listing. The fonts that carry letters of mathematics
# are kept: CMSY (calligraphic letters), CMMI, MSBM and the like (blackboard bold), EUFM/EUSM, rsfs, dsrom, bbold.
NON_TEXT_FONTS = re.compile(
    r"cmex"               # Computer Modern extension: big operators, radical and wide-accent pieces (also Yhcmex)
    r"|tex-math[abx]"     # mathabx: big parentheses/braces pieces, wide accents (TeX-matha10, TeX-mathx10, ...)
    r"|math[abx]\d"       # the same families without the "TeX-" prefix
    r"|stmary"            # St Mary Road: brackets and big operators
    r"|txex"              # txfonts extension (txex, txexs, txexas)
    r"|esint",            # integral signs
    re.IGNORECASE)


def is_non_text_font(name: str) -> bool:
    """``name`` is the font of glyphs that draw part of a formula and carry no text."""
    return NON_TEXT_FONTS.search(name) is not None


def glyph_flags(base: int) -> int:
    """``base`` extraction flags that report the glyphs printed on the page rather than a /ActualText replacement.
    The ECCV 2024 template attaches each formula's LaTeX source to its glyphs as /ActualText; MuPDF's default
    extraction spells that source one character per glyph (``\\m a thca l {L} _ {r`` for a printed ``Lrec``), which
    no Markdown of the page can contain (tests/test_verify_textlayer_actualtext.py)."""
    import pymupdf

    return base | pymupdf.TEXT_IGNORE_ACTUALTEXT


def page_words(page, skip=is_non_text_font) -> list[tuple]:
    """``page.get_text("words")`` of ``page`` without the glyphs of fonts for which ``skip`` is true: PyMuPDF's own tuples
    (x0, y0, x1, y1, word, block_no, line_no, word_no) when the page has none of those fonts, so a page without them
    is the reference it always was; otherwise the words rebuilt from the glyphs (``_rebuild_words``). Both read the
    printed glyphs, not a /ActualText replacement (``glyph_flags``)."""
    import pymupdf

    # the words' own flags: the block numbers of an extraction that keeps images count differently
    flags = glyph_flags(pymupdf.TEXTFLAGS_WORDS)
    raw = page.get_text("rawdict", flags=flags)
    if not any(skip(span["font"]) for block in raw["blocks"] if block.get("type") == 0
               for line in block["lines"] for span in line["spans"]):
        return [tuple(word) for word in page.get_text("words", flags=flags)]
    return _rebuild_words(raw, skip)


def _rebuild_words(raw: dict, skip) -> list[tuple]:
    """The words of a ``get_text("rawdict")`` result as PyMuPDF's ``get_text("words")`` cuts them, leaving out the glyphs of
    the fonts for which ``skip`` is true: a word breaks at whitespace, no-break space and the bidi marks, and (like PyMuPDF)
    at a change of writing direction. Over the 353 JMLR 2022 PDFs (16037 pages) it is PyMuPDF's list on all but 110 pages,
    where PyMuPDF has a zero-width combining mark (the slash of a crossed-out letter) that rawdict has not, or one line
    number more: no difference to a comparison, which folds combining marks and looks at no line numbers but a hyphen's."""
    words: list[tuple] = []
    rtl_before = False
    for block_no, block in enumerate(raw["blocks"]):
        if block.get("type") != 0:
            continue
        for line_no, line in enumerate(block["lines"]):
            text: list[str] = []
            box: Optional[BBox] = None
            word_no = 0
            for span in line["spans"]:
                if skip(span["font"]):
                    continue
                for glyph in span["chars"]:
                    char = glyph["c"]
                    code = ord(char)
                    if not text and code == 0x200D:  # a zero width joiner cannot start a word
                        continue
                    breaks = code <= 32 or code == 160 or 0x202A <= code <= 0x202E
                    rtl = 0x590 <= code <= 0x900
                    if breaks or rtl != rtl_before:
                        if not text and breaks:
                            continue
                        if text:
                            words.append((*box, "".join(text), block_no, line_no, word_no))
                            word_no += 1
                        text, box = [], None
                        if breaks:
                            continue
                    text.append(char)
                    rtl_before = rtl
                    x0, y0, x1, y1 = glyph["bbox"]
                    box = (x0, y0, x1, y1) if box is None else (min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1))
            if text:
                words.append((*box, "".join(text), block_no, line_no, word_no))
    return words


def page_reference(pdf: Path, page_no: int) -> PageRef:
    """The text layer of page ``page_no`` (0-based); ``source`` is "none" when it has none."""
    import pymupdf

    with pymupdf.open(pdf) as document:
        page = document[page_no]
        raw = page_words(page)
        width, height = page.rect.width, page.rect.height
        garbled, shift = garbled_share(page)
    ref = reference_from_words(page_no, width, height, raw)
    if ref.source == "textlayer" and shift is not None and garbled >= DOMINANT_SHARE:
        # the layer's letters are other letters: it is no witness (the caller reads such a page by OCR, like a scan)
        ref.source, ref.reason = "none", f"{garbled:.0%} of the text layer is set in a font whose letters are shifted by {shift}"
    return ref


def reference_from_words(page_no: int, width: float, height: float, raw: Sequence[tuple]) -> PageRef:
    """Build a PageRef from PyMuPDF ``get_text("words")`` tuples
    (x0, y0, x1, y1, word, block_no, line_no, word_no)."""
    chars = sum(len(item[4]) for item in raw)
    broken = sum(len(m.group()) for item in raw for m in _PRIVATE_USE.finditer(item[4])) \
        + sum(len(m.group()) for item in raw for m in _CID.finditer(item[4]))
    ordered = sorted(raw, key=lambda item: (item[5], item[6], item[7]))

    words: list[RefWord] = []
    pending: Optional[tuple[BBox, str, int, int]] = None  # a word ending in a line-break hyphen
    for index, (x0, y0, x1, y1, text, block, line, _) in enumerate(ordered):
        box: BBox = (x0, y0, x1, y1)
        if pending is not None:
            pbox, ptext, pblock, pline = pending
            pending = None
            if pblock == block and line == pline + 1:
                # "compu-" "ting" is one word; "E5-" "2670" is a name with a hyphen in it
                joint = "" if ptext[-1].isalpha() and text[:1].isalpha() else "-"
                text, box = ptext + joint + text, (min(pbox[0], x0), min(pbox[1], y0), max(pbox[2], x1), max(pbox[3], y1))
            else:
                _append(words, ptext + "-", pbox, pblock)
        last_of_line = index + 1 >= len(ordered) or (ordered[index + 1][5], ordered[index + 1][6]) != (block, line)
        if last_of_line and len(text) > 1 and text.endswith(_LINE_END_HYPHENS):
            pending = (box, text[:-1], block, line)
            continue
        _append(words, text, box, block)
    if pending is not None:
        _append(words, pending[1] + "-", pending[0], pending[2])

    ref = PageRef(page_no, width, height, words)
    if len(words) < MIN_USABLE_WORDS:
        ref.source, ref.reason = "none", f"text layer has {len(words)} words (< {MIN_USABLE_WORDS})"
    elif chars and broken / chars >= MAX_BROKEN_CHAR_FRACTION:
        ref.source, ref.reason = "none", f"text layer is {broken / chars:.0%} unmappable glyphs"
    return ref


_S_ACUTE_BETWEEN_DIGITS = re.compile(r"(?<=\d)ś(?=\d)")  # the en dash of "2962–2970" in ACM Reference Format fonts


def _append(words: list[RefWord], text: str, box: BBox, block: int) -> None:
    folded = comparable(_S_ACUTE_BETWEEN_DIGITS.sub("–", text))
    if folded:
        words.append(RefWord(folded, box, block, is_mathy(text)))


def is_mathy(raw: str) -> bool:
    """``raw`` holds a glyph of typeset mathematics: a mathematical alphanumeric (also as the
    Hangul syllable a layer without its top bit writes), a Greek letter, an operator or an arrow."""
    return any(
        "\U0001d400" <= c <= "\U0001d7ff" or "\ud400" <= c <= "\ud7ff" or "\u0370" <= c <= "\u03ff"
        or unicodedata.category(c) in ("Sm", "Sk") for c in raw)
