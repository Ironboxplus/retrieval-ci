"""Text layers whose letters are the wrong ones: a font without a ToUnicode table is read through its glyph numbers, so
every letter is another letter by one constant (JMLR 2022, 20-843 p26: "WKH" for "the", "SHUIRUPDQFH" for "performance").

Such a layer says nothing of what is on the page, and a correct Markdown page then fails for "fabrication" (its words
are not in the layer) and "omission" (the layer's words are not in it). A font is judged by its own text on the page:
as written it holds no common English word, and under one constant shift of its character codes it holds many different
ones. A bundled list of common words decides; nothing is downloaded.
"""
from __future__ import annotations

import re
from typing import Optional

# letters a font must set on the page before it is judged: the tags of a figure are too few to tell a shift from chance
MIN_FONT_LETTERS = 150
# share of a font's letters (in words of 3+ letters) that common words cover as written, below which it is no English
MAX_PLAIN_COVERAGE = 0.05
# the same share under the best shift, and the number of different common words it finds: a table that repeats "OUT"
# a hundred times has the first and not the second
MIN_SHIFTED_COVERAGE = 0.3
MIN_SHIFTED_WORDS = 12
# the code distance between the glyph numbers and the letters of the fonts met (+-3 in the one page found; the rest is
# room for other producers, searched in the printable ASCII range)
MAX_SHIFT = 64
# share of the page's letters that garbled fonts set, from which the page's layer is no reference at all: the garbled
# caption and chart text of the one page found in the 16037 pages of JMLR 2022 is 23% of its letters, and every sentence
# of the Markdown that the garbled words stand for is then "fabricated" and every garbled word "omitted"
DOMINANT_SHARE = 0.10

COMMON_WORDS = frozenset("""
the and for are but not you all any can had her was one our out day get has him his how man new now old see two way who boy
did its let put say she too use that with have this will your from they know want been good much some time very when come
here just like long make many more only over such take than them well were what which their there these those would could
should about after again also because before between both each first found given however into large most must other same
show shown since small still then through under using where while whether within without results result method methods model
models data performance distribution based figure table proposed number function algorithm learning set sets case cases
problem problems theory theorem lemma proof following paper work approach different several used provide provides shows
consider assume suppose thus hence therefore holds hold true false exists exist every another often although though during
among above below left right high low mean standard average error rate rates size color bar circle training test testing
validation value values input output sample samples class classes feature features dimension dimensions space dimensional
matrix vector probability random variable variables bound bounds condition conditions general special simple natural
example examples note notation denote denotes define defined definition estimate estimates estimator estimation parameter
parameters optimal optimization convergence convergent section sections appendix figures tables equation equations page
pages
""".split())

_RUN = re.compile(r"[a-z]{3,}")


def _coverage(text: str) -> tuple[float, int]:
    """(share of the letters of the words of 3+ letters that a common word covers, different common words found)"""
    judged = covered = 0
    found: set[str] = set()
    for word in _RUN.findall(text.casefold()):
        judged += len(word)
        if word in COMMON_WORDS:
            covered += len(word)
            found.add(word)
    return (covered / judged if judged else 0.0), len(found)


def _shifted(text: str, shift: int) -> str:
    """``text`` with every printable ASCII character moved by ``shift`` codes; the rest (the control characters a garbled
    font uses for spaces and punctuation) is a space."""
    return "".join(chr(ord(c) + shift) if 33 <= ord(c) <= 126 and 33 <= ord(c) + shift <= 126 else " " for c in text)


def font_shift(text: str) -> Optional[int]:
    """The shift that turns ``text`` of one font into English, or None when it is no garbled text."""
    if sum(c.isalpha() for c in text) < MIN_FONT_LETTERS or _coverage(text)[0] >= MAX_PLAIN_COVERAGE:
        return None
    best: Optional[tuple[int, float, int]] = None
    for shift in sorted(range(-MAX_SHIFT, MAX_SHIFT + 1), key=abs):  # -3 and -35 read alike (case): the nearer one
        if shift:
            coverage, words = _coverage(_shifted(text, shift))
            if best is None or (words, coverage) > best[:2]:
                best = (words, coverage, shift)
    return best[2] if best and best[1] >= MIN_SHIFTED_COVERAGE and best[0] >= MIN_SHIFTED_WORDS else None


def garbled_share(page) -> tuple[float, Optional[int]]:
    """(share of the page's letters that the text of garbled fonts sets, the shift of the biggest such font)."""
    import pymupdf

    fonts: dict[str, list[str]] = {}
    # the printed glyphs, as the reference reads them (textlayer.glyph_flags): a font's letters, not /ActualText
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_IGNORE_ACTUALTEXT)["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                fonts.setdefault(span["font"], []).append(span["text"])
    letters = {font: sum(c.isalpha() for c in " ".join(texts)) for font, texts in fonts.items()}
    total, garbled, shift, biggest = sum(letters.values()), 0, None, 0
    for font, texts in fonts.items():
        found = font_shift(" ".join(texts))
        if found is not None:
            garbled += letters[font]
            if letters[font] > biggest:
                shift, biggest = found, letters[font]
    return (garbled / total if total else 0.0), shift
